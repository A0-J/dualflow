"""AgentExperienceStore / build_verified_experience() — B7c. 순수 데이터
저장/조회만 검증한다: LLM 호출도, 네트워크도, fake LLMClient조차 필요
없다 — 이 모듈 자체에는 model 호출이 전혀 없다.
"""

from __future__ import annotations

import inspect

from dualflow.agent_experience import (
    AgentExperience, AgentExperienceStore, build_verified_experience,
)
from dualflow.clarification import ClarificationResult
from dualflow.delegate_agent import CandidateDistribution, ClarificationQuestion
from dualflow.llm import LLMResponse
from dualflow.principal_agent import PrincipalClarification
from dualflow.semantic import Interpretation

SUMMARIZE = Interpretation("summarize", "file", "/reports/2026-08/", frozenset())
EXPORT = Interpretation("export", "file", "/reports/2026-08/", frozenset())


def _distribution(belief: dict, entropy_value: float) -> CandidateDistribution:
    top = max(belief, key=lambda i: belief[i])
    return CandidateDistribution(
        belief=belief, entropy=entropy_value, top=top, top_probability=belief[top],
        n_unique=len(belief), n_samples=10, responses=[])


def _clarified_result(post_entropy: float, post_belief: dict | None = None
                      ) -> ClarificationResult:
    """실제 clarification이 일어난 ClarificationResult를 만든다 —
    post_entropy만 바꿔가며 저장 자격 기준을 테스트하기 위한 fixture."""
    pre = _distribution({SUMMARIZE: 0.6, EXPORT: 0.4}, 0.971)
    post_belief = post_belief or {SUMMARIZE: 1.0}
    post = _distribution(post_belief, post_entropy)
    question = ClarificationQuestion(
        question="Export or summarize?", raw_text="Export or summarize?",
        response=LLMResponse(text="Export or summarize?"), belief=pre.belief, entropy=pre.entropy)
    answer = PrincipalClarification(
        answer="Summarize only.", response=LLMResponse(text="Summarize only."))
    return ClarificationResult(
        clarified=True, pre_distribution=pre, post_distribution=post,
        question=question, answer=answer, final_interpretation=post.top)


def _unclarified_result() -> ClarificationResult:
    pre = _distribution({EXPORT: 0.8, SUMMARIZE: 0.2}, 0.722)
    return ClarificationResult(
        clarified=False, pre_distribution=pre, post_distribution=None,
        question=None, answer=None, final_interpretation=pre.top)


class TestBuildVerifiedExperience:
    def test_converged_clarified_interaction_is_accepted(self):
        result = _clarified_result(post_entropy=0.0)

        exp = build_verified_experience(
            result, principal_id="finance_lead_A", task_category="external_audit_report",
            delegation="Please prepare the August 2026 financial report.")

        assert exp is not None
        assert isinstance(exp, AgentExperience)
        assert exp.principal_id == "finance_lead_A"
        assert exp.task_category == "external_audit_report"
        assert exp.confirmed_interpretation == SUMMARIZE
        assert exp.pre_entropy == 0.971
        assert exp.post_entropy == 0.0
        assert exp.clarification_question == "Export or summarize?"
        assert exp.principal_answer == "Summarize only."

    def test_no_clarification_interaction_is_not_stored(self):
        result = _unclarified_result()

        exp = build_verified_experience(
            result, principal_id="finance_lead_A", task_category="external_audit_report",
            delegation="x")

        assert exp is None

    def test_high_post_entropy_is_not_stored_by_default(self):
        """post-entropy가 기본 max_verified_entropy(0.0)를 넘으면
        저장하지 않는다 — 여전히 애매한 채로 확정된 척하지 않는다."""
        result = _clarified_result(post_entropy=0.971,
                                   post_belief={SUMMARIZE: 0.6, EXPORT: 0.4})

        exp = build_verified_experience(
            result, principal_id="finance_lead_A", task_category="external_audit_report",
            delegation="x")

        assert exp is None

    def test_high_post_entropy_accepted_with_looser_configured_threshold(self):
        """threshold는 configurable하다 — 더 느슨하게 주면 통과할 수 있다."""
        result = _clarified_result(post_entropy=0.5,
                                   post_belief={SUMMARIZE: 0.85, EXPORT: 0.15})

        exp = build_verified_experience(
            result, principal_id="finance_lead_A", task_category="external_audit_report",
            delegation="x", max_verified_entropy=0.6)

        assert exp is not None
        assert exp.post_entropy == 0.5

    def test_no_forbidden_parameter_in_signature(self):
        forbidden = ("truth", "ground_truth", "label", "expected")
        sig = inspect.signature(build_verified_experience)
        for name in sig.parameters:
            lowered = name.lower()
            assert not any(f in lowered for f in forbidden), (
                f"build_verified_experience has a suspicious parameter: {name}")


class TestStoreLookupIsolation:
    def test_isolated_by_principal_id(self):
        store = AgentExperienceStore()
        exp_a = build_verified_experience(
            _clarified_result(0.0), principal_id="principal_A",
            task_category="external_audit_report", delegation="x")
        exp_b = build_verified_experience(
            _clarified_result(0.0), principal_id="principal_B",
            task_category="external_audit_report", delegation="y")
        store.add(exp_a)
        store.add(exp_b)

        retrieved_a = store.get("principal_A", "external_audit_report")
        retrieved_b = store.get("principal_B", "external_audit_report")

        assert retrieved_a == [exp_a]
        assert retrieved_b == [exp_b]
        assert exp_a not in retrieved_b
        assert exp_b not in retrieved_a

    def test_isolated_by_task_category(self):
        store = AgentExperienceStore()
        exp_audit = build_verified_experience(
            _clarified_result(0.0), principal_id="principal_A",
            task_category="external_audit_report", delegation="x")
        exp_logs = build_verified_experience(
            _clarified_result(0.0), principal_id="principal_A",
            task_category="security_log_review", delegation="y")
        store.add(exp_audit)
        store.add(exp_logs)

        retrieved_audit = store.get("principal_A", "external_audit_report")
        retrieved_logs = store.get("principal_A", "security_log_review")

        assert retrieved_audit == [exp_audit]
        assert retrieved_logs == [exp_logs]

    def test_unknown_key_returns_empty_list(self):
        store = AgentExperienceStore()
        assert store.get("nobody", "nothing") == []


class TestStoreOrderingAndMutations:
    def test_multiple_experiences_retained_in_insertion_order(self):
        store = AgentExperienceStore()
        experiences = [
            build_verified_experience(
                _clarified_result(0.0), principal_id="principal_A",
                task_category="external_audit_report", delegation="x",
                episode_id=f"ep{i}")
            for i in range(3)
        ]
        for exp in experiences:
            store.add(exp)

        retrieved = store.get("principal_A", "external_audit_report")

        assert [e.episode_id for e in retrieved] == ["ep0", "ep1", "ep2"]

    def test_get_returns_a_copy_not_internal_state(self):
        store = AgentExperienceStore()
        exp = build_verified_experience(
            _clarified_result(0.0), principal_id="principal_A",
            task_category="external_audit_report", delegation="x")
        store.add(exp)

        retrieved = store.get("principal_A", "external_audit_report")
        retrieved.append("garbage")  # 반환된 리스트를 훼손해본다

        assert store.get("principal_A", "external_audit_report") == [exp]  # 영향 없음

    def test_count(self):
        store = AgentExperienceStore()
        assert store.count("principal_A", "external_audit_report") == 0
        store.add(build_verified_experience(
            _clarified_result(0.0), principal_id="principal_A",
            task_category="external_audit_report", delegation="x"))
        assert store.count("principal_A", "external_audit_report") == 1

    def test_clear_all(self):
        store = AgentExperienceStore()
        store.add(build_verified_experience(
            _clarified_result(0.0), principal_id="principal_A",
            task_category="external_audit_report", delegation="x"))
        store.clear()
        assert store.get("principal_A", "external_audit_report") == []

    def test_clear_scoped_to_principal_and_category(self):
        store = AgentExperienceStore()
        store.add(build_verified_experience(
            _clarified_result(0.0), principal_id="principal_A",
            task_category="external_audit_report", delegation="x"))
        store.add(build_verified_experience(
            _clarified_result(0.0), principal_id="principal_A",
            task_category="security_log_review", delegation="y"))
        store.add(build_verified_experience(
            _clarified_result(0.0), principal_id="principal_B",
            task_category="external_audit_report", delegation="z"))

        store.clear(principal_id="principal_A", task_category="external_audit_report")

        assert store.get("principal_A", "external_audit_report") == []
        assert store.count("principal_A", "security_log_review") == 1  # 안 지워짐
        assert store.count("principal_B", "external_audit_report") == 1  # 안 지워짐


class TestNoLLMCallsAndNoAuthorityCoupling:
    def test_store_module_never_imports_llm_client_construction(self):
        """이 모듈에는 LLM 호출이 전혀 없다 — store 생성자가 어떤
        LLMClient/API 관련 인자도 받지 않는다는 것으로 구조적으로
        확인한다."""
        sig = inspect.signature(AgentExperienceStore.__init__)
        assert list(sig.parameters) == ["self"]

    def test_agent_experience_has_no_authority_or_budget_field(self):
        forbidden = ("authority", "budget", "capability", "truth",
                     "ground_truth", "label")
        for f in AgentExperience.__dataclass_fields__:
            lowered = f.lower()
            assert not any(bad in lowered for bad in forbidden), (
                f"AgentExperience has a suspicious field: {f}")
