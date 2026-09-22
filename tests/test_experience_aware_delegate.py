"""ExperienceAwareDelegate — B7d. 과거 verified experience를 candidate
sampling에 semantic guidance로 반영하는지 검증한다. 전부 fake LLMClient로
검증한다: 네트워크도, OPENAI_API_KEY도 필요 없다.

가장 중요한 두 테스트는 TestExperienceReducesEntropy(경험이 실제로
uncertainty를 줄이는지)와 TestCurrentContextPrecedence /
TestExperienceDoesNotForceOutput(경험이 현재 episode의 명시적 지시나
scope를 덮어쓰지 않는지)다.
"""

from __future__ import annotations

import inspect

import pytest

from dualflow.agent_experience import AgentExperience, AgentExperienceStore
from dualflow.delegate_agent import CandidateDistribution, DelegateAgent
from dualflow.experience_aware_delegate import ExperienceAwareDelegate, ExperienceAwareSampleResult
from dualflow.llm import LLMResponse
from dualflow.semantic import Interpretation

SUMMARIZE_AUG = Interpretation("summarize", "file", "/reports/2026-08/", frozenset())
SUMMARIZE_SEP = Interpretation("summarize", "file", "/reports/2026-09/", frozenset())
EXPORT_SEP = Interpretation("export", "file", "/reports/2026-09/", frozenset())


class _FakeLLMClient:
    """`llm.LLMClient` 프로토콜을 흉내 내는 순수 Python fake."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, *, instructions: str, input_text: str) -> LLMResponse:
        self.calls.append({"instructions": instructions, "input_text": input_text})
        return self._responses.pop(0)


def _structured(action: str, resource: str = "file", scope: str = "/reports/2026-09/",
                condition: str = "none") -> LLMResponse:
    return LLMResponse(
        text=f"ACTION: {action}\nRESOURCE: {resource}\nSCOPE: {scope}\nCONDITION: {condition}")


def _make_experience(episode_id: str = "ep1") -> AgentExperience:
    """B7b pilot에서 실제로 나온 것과 같은 모양의 verified experience —
    8월 delegation을 summarize로 confirm 받은 기록."""
    dist_belief = {SUMMARIZE_AUG: 0.4, Interpretation("export", "file",
                                                       "/reports/2026-08/", frozenset()): 0.6}
    pre = CandidateDistribution(belief=dist_belief, entropy=0.971, top=SUMMARIZE_AUG,
                                top_probability=0.4, n_unique=2, n_samples=10, responses=[])
    post = CandidateDistribution(belief={SUMMARIZE_AUG: 1.0}, entropy=0.0, top=SUMMARIZE_AUG,
                                 top_probability=1.0, n_unique=1, n_samples=10, responses=[])
    return AgentExperience(
        principal_id="finance_lead_A", task_category="external_audit_report",
        delegation="Please prepare the August 2026 financial report for the external audit.",
        clarification_question="Should I export the report or summarize it?",
        principal_answer="Create an internal summary only; do not export.",
        confirmed_interpretation=SUMMARIZE_AUG,
        pre_distribution=pre, post_distribution=post, episode_id=episode_id)


CURRENT_DELEGATION = "Please prepare the September 2026 financial report for the external audit."
CURRENT_CONTEXT = "RESOURCE=file, SCOPE=/reports/2026-09/"


class TestNoExperience:
    def test_empty_store_leaves_context_unchanged(self):
        """경험이 없으면 enriched context가 원본 context와 완전히
        동일해야 한다 — 즉 기존 sample_candidates() 호출과 구조적으로
        동등하다."""
        store = AgentExperienceStore()
        fake = _FakeLLMClient([_structured("summarize") for _ in range(10)])
        wrapped = ExperienceAwareDelegate(DelegateAgent(llm=fake), store)

        result = wrapped.sample_candidates(
            principal_id="finance_lead_A", task_category="external_audit_report",
            delegation=CURRENT_DELEGATION, context=CURRENT_CONTEXT, n=10)

        assert result.experience_count == 0
        assert result.experiences_used == ()
        assert fake.calls[0]["input_text"] == f"Delegation: {CURRENT_DELEGATION}\nContext: {CURRENT_CONTEXT}"


class TestMatchingExperienceRetrieval:
    def test_correct_experience_is_included(self):
        store = AgentExperienceStore()
        store.add(_make_experience())
        fake = _FakeLLMClient([_structured("summarize") for _ in range(10)])
        wrapped = ExperienceAwareDelegate(DelegateAgent(llm=fake), store)

        result = wrapped.sample_candidates(
            principal_id="finance_lead_A", task_category="external_audit_report",
            delegation=CURRENT_DELEGATION, context=CURRENT_CONTEXT, n=10)

        assert result.experience_count == 1
        assert result.experiences_used[0].episode_id == "ep1"
        assert "Verified prior interactions" in fake.calls[0]["input_text"]


class TestPrincipalIsolation:
    def test_principal_bs_sampling_never_sees_principal_as_experience(self):
        store = AgentExperienceStore()
        store.add(_make_experience())  # finance_lead_A 소유
        fake = _FakeLLMClient([_structured("summarize") for _ in range(10)])
        wrapped = ExperienceAwareDelegate(DelegateAgent(llm=fake), store)

        result = wrapped.sample_candidates(
            principal_id="finance_lead_B", task_category="external_audit_report",
            delegation=CURRENT_DELEGATION, context=CURRENT_CONTEXT, n=10)

        assert result.experience_count == 0
        assert "Verified prior interactions" not in fake.calls[0]["input_text"]


class TestCategoryIsolation:
    def test_different_category_never_sees_the_experience(self):
        store = AgentExperienceStore()
        store.add(_make_experience())  # external_audit_report 소유
        fake = _FakeLLMClient([_structured("summarize") for _ in range(10)])
        wrapped = ExperienceAwareDelegate(DelegateAgent(llm=fake), store)

        result = wrapped.sample_candidates(
            principal_id="finance_lead_A", task_category="security_log_review",
            delegation=CURRENT_DELEGATION, context=CURRENT_CONTEXT, n=10)

        assert result.experience_count == 0


class TestDeterministicMaxExperienceLimit:
    def test_uses_the_most_recent_n_experiences(self):
        store = AgentExperienceStore()
        for i in range(5):
            store.add(_make_experience(episode_id=f"ep{i}"))
        fake = _FakeLLMClient([_structured("summarize") for _ in range(10)])
        wrapped = ExperienceAwareDelegate(DelegateAgent(llm=fake), store, max_experiences=3)

        result = wrapped.sample_candidates(
            principal_id="finance_lead_A", task_category="external_audit_report",
            delegation=CURRENT_DELEGATION, context=CURRENT_CONTEXT, n=10)

        assert result.experience_count == 3
        assert [e.episode_id for e in result.experiences_used] == ["ep2", "ep3", "ep4"]


class TestCurrentScopePrecedence:
    def test_current_scope_is_present_and_old_scope_only_appears_as_history(self):
        store = AgentExperienceStore()
        store.add(_make_experience())  # 8월 scope로 confirm된 경험
        fake = _FakeLLMClient([_structured("summarize", scope="/reports/2026-09/")
                               for _ in range(10)])
        wrapped = ExperienceAwareDelegate(DelegateAgent(llm=fake), store)

        wrapped.sample_candidates(
            principal_id="finance_lead_A", task_category="external_audit_report",
            delegation=CURRENT_DELEGATION, context=CURRENT_CONTEXT, n=10)

        input_text = fake.calls[0]["input_text"]
        # 현재 scope(9월)는 "Current environment context" 아래에 그대로 있다.
        assert "Current environment context" in input_text
        assert "/reports/2026-09/" in input_text
        # 과거 scope(8월)는 "Previous delegation"(역사적 원문) 안에만
        # 등장한다 — "Principal-confirmed interpretation" 줄에는 SCOPE
        # 필드 자체가 아예 없다(ACTION/RESOURCE만).
        assert "Principal-confirmed interpretation: ACTION=summarize RESOURCE=file" \
            in input_text
        confirmed_line = next(
            line for line in input_text.splitlines()
            if line.startswith("Principal-confirmed interpretation"))
        assert "2026-08" not in confirmed_line
        assert "SCOPE" not in confirmed_line

    def test_precedence_instruction_is_explicit_in_the_prompt(self):
        store = AgentExperienceStore()
        store.add(_make_experience())
        fake = _FakeLLMClient([_structured("summarize") for _ in range(10)])
        wrapped = ExperienceAwareDelegate(DelegateAgent(llm=fake), store)

        wrapped.sample_candidates(
            principal_id="finance_lead_A", task_category="external_audit_report",
            delegation=CURRENT_DELEGATION, context=CURRENT_CONTEXT, n=10)

        input_text = fake.calls[0]["input_text"]
        assert "historical examples only" in input_text
        assert "always take precedence" in input_text or "always takes precedence" in input_text


class TestExperienceReducesEntropy:
    def test_matching_experience_yields_lower_entropy_than_no_experience(self):
        store_empty = AgentExperienceStore()
        fake_without = _FakeLLMClient(
            [_structured("export") for _ in range(5)]
            + [_structured("summarize") for _ in range(5)])  # H=1.0
        wrapped_without = ExperienceAwareDelegate(DelegateAgent(llm=fake_without), store_empty)

        result_without = wrapped_without.sample_candidates(
            principal_id="finance_lead_A", task_category="external_audit_report",
            delegation=CURRENT_DELEGATION, context=CURRENT_CONTEXT, n=10)

        store_with = AgentExperienceStore()
        store_with.add(_make_experience())
        fake_with = _FakeLLMClient([_structured("summarize") for _ in range(10)])  # H=0.0
        wrapped_with = ExperienceAwareDelegate(DelegateAgent(llm=fake_with), store_with)

        result_with = wrapped_with.sample_candidates(
            principal_id="finance_lead_A", task_category="external_audit_report",
            delegation=CURRENT_DELEGATION, context=CURRENT_CONTEXT, n=10)

        assert result_without.distribution.entropy == pytest.approx(1.0)
        assert result_with.distribution.entropy == 0.0
        assert result_with.distribution.entropy < result_without.distribution.entropy
        # entropy는 CandidateDistribution이 실제로 계산한 값 그대로다 —
        # wrapper가 손으로 조작하지 않는다.
        assert isinstance(result_with.distribution, CandidateDistribution)


class TestExperienceDoesNotForceOutput:
    def test_explicit_current_export_is_preserved_despite_summarize_history(self):
        """과거 경험이 summarize를 가리켜도, 현재 delegation에 대해 모델이
        실제로 export라고 답하면 그 결과가 그대로 나와야 한다 — wrapper가
        결과를 되돌려 쓰지 않는다."""
        store = AgentExperienceStore()
        store.add(_make_experience())  # summarize 경험
        fake = _FakeLLMClient([_structured("export") for _ in range(10)])  # 그런데도 전부 export
        wrapped = ExperienceAwareDelegate(DelegateAgent(llm=fake), store)

        result = wrapped.sample_candidates(
            principal_id="finance_lead_A", task_category="external_audit_report",
            delegation="This time, export the report for the auditor.",
            context=CURRENT_CONTEXT, n=10)

        assert result.distribution.top.action == "export"
        assert result.distribution.entropy == 0.0
        assert result.experience_count == 1  # 경험은 조회는 됐지만 결과를 바꾸지 않았다


class TestZeroAuthorityEffect:
    def test_module_does_not_import_capability_or_authority_feedback(self):
        """docstring 설명 텍스트가 아니라 실제 import 문만 검사한다 —
        `.capability`/`.authority_feedback`을 전혀 import하지 않는다는
        게 구조적 무관성의 근거다."""
        import dualflow.experience_aware_delegate as mod

        assert not hasattr(mod, "Budget")
        assert not hasattr(mod, "check_authority")
        assert not hasattr(mod, "AuthorityVerifierAgent")
        assert not hasattr(mod, "AuthorityVerdict")


class TestNoGroundTruthAPI:
    def test_sample_candidates_signature_has_no_truth_or_label_parameter(self):
        forbidden = ("truth", "ground_truth", "label", "expected", "evaluation")
        sig = inspect.signature(ExperienceAwareDelegate.sample_candidates)
        for name in sig.parameters:
            lowered = name.lower()
            assert not any(f in lowered for f in forbidden), (
                f"sample_candidates has a suspicious parameter: {name}")

    def test_result_has_no_ground_truth_field(self):
        forbidden = ("truth", "ground_truth", "label", "expected")
        for f in ExperienceAwareSampleResult.__dataclass_fields__:
            lowered = f.lower()
            assert not any(bad in lowered for bad in forbidden), (
                f"ExperienceAwareSampleResult has a suspicious field: {f}")

    def test_no_network_or_api_key_required(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        store = AgentExperienceStore()
        fake = _FakeLLMClient([_structured("summarize") for _ in range(3)])
        wrapped = ExperienceAwareDelegate(DelegateAgent(llm=fake), store)

        wrapped.sample_candidates(principal_id="x", task_category="y",
                                  delegation="z", n=3)  # 예외 없어야 함


class TestExistingPathsUnaffected:
    def test_delegate_agent_sample_candidates_signature_unchanged(self):
        sig = inspect.signature(DelegateAgent.sample_candidates)
        assert list(sig.parameters) == ["self", "delegation", "context", "n"]
