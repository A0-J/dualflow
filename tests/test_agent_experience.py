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


def _clarified_result(post_entropy: float, post_belief: dict | None = None,
                      target_facets: frozenset[str] = frozenset()) -> ClarificationResult:
    """실제 clarification이 일어난 ClarificationResult를 만든다 —
    post_entropy만 바꿔가며 저장 자격 기준을 테스트하기 위한 fixture.
    `target_facets`는 `question.target_facets`에 그대로 들어간다(기본값은
    빈 frozenset — 기존 동작)."""
    pre = _distribution({SUMMARIZE: 0.6, EXPORT: 0.4}, 0.971)
    post_belief = post_belief or {SUMMARIZE: 1.0}
    post = _distribution(post_belief, post_entropy)
    question = ClarificationQuestion(
        question="Export or summarize?", raw_text="Export or summarize?",
        response=LLMResponse(text="Export or summarize?"), belief=pre.belief, entropy=pre.entropy,
        target_facets=target_facets)
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


class TestConfirmedFacetsProvenance:
    """v3 provenance 보완, revision 2(§23 follow-up 3) — build_verified_
    experience()는 더 이상 confirmed_facets를 스스로 계산하지 않는다.
    `result.question.target_facets`(clarification.py의 `_facets_that_
    varied()`가 질문을 만들기 *전에* 계산해서 넘긴 값)를 그대로
    물려받을 뿐이다 — 이 클래스는 그 pass-through만 확인한다. "어떤
    facet이 실제로 질문 대상이었는가"를 계산하는 로직 자체의 테스트는
    tests/test_clarification.py의 TestTargetFacets에 있다."""

    def test_confirmed_facets_is_exactly_the_question_target_facets(self):
        result = _clarified_result(post_entropy=0.0, target_facets=frozenset({"action"}))

        exp = build_verified_experience(
            result, principal_id="finance_lead_A", task_category="external_audit_report",
            delegation="Please prepare the August 2026 financial report.")

        assert exp is not None
        assert exp.confirmed_facets == frozenset({"action"})
        assert exp.confirmed_facets is result.question.target_facets

    def test_multiple_target_facets_are_all_passed_through(self):
        result = _clarified_result(post_entropy=0.0,
                                   target_facets=frozenset({"action", "scope"}))

        exp = build_verified_experience(
            result, principal_id="finance_lead_A", task_category="external_audit_report",
            delegation="x")

        assert exp is not None
        assert exp.confirmed_facets == frozenset({"action", "scope"})

    def test_no_target_facets_means_empty_confirmed_facets(self):
        """question.target_facets가 기본값(빈 frozenset)이면 confirmed_
        facets도 비어 있다 -- build_verified_experience()가 스스로 뭔가를
        추론해서 채워 넣지 않는다."""
        result = _clarified_result(post_entropy=0.0)  # target_facets 생략 -> 기본값

        exp = build_verified_experience(
            result, principal_id="finance_lead_A", task_category="external_audit_report",
            delegation="x")

        assert exp is not None
        assert exp.confirmed_facets == frozenset()

    def test_does_not_recompute_from_pre_distribution_variance(self):
        """핵심 회귀 방지: pre_distribution 자체는 action과 scope 둘 다
        갈리게 만들어 두되, question.target_facets는 "action"만으로
        명시한다 -- 결과는 반드시 target_facets를 따라야 하고,
        pre_distribution의 실제 variance(action+scope)를 다시 계산해서
        나오면 안 된다. 이게 §23 follow-up 2에서 발견된 바로 그 문제
        ("ambiguous_facets_before_clarification"으로 오염되는 것)를
        재현해서 고쳐졌는지 확인하는 테스트다."""
        summarize_sept = Interpretation("summarize", "file", "/reports/2026-09/", frozenset())
        export_aug = Interpretation("export", "file", "/reports/2026-08/", frozenset())
        pre = _distribution({summarize_sept: 0.6, export_aug: 0.4}, 0.971)  # action AND scope vary
        post = _distribution({summarize_sept: 1.0}, 0.0)
        question = ClarificationQuestion(
            question="Summarize or export?", raw_text="Summarize or export?",
            response=LLMResponse(text="Summarize or export?"),
            belief=pre.belief, entropy=pre.entropy,
            target_facets=frozenset({"action"}))  # question only actually asked about action
        answer = PrincipalClarification(answer="Summarize.", response=LLMResponse(text="Summarize."))
        result = ClarificationResult(
            clarified=True, pre_distribution=pre, post_distribution=post,
            question=question, answer=answer, final_interpretation=post.top)

        exp = build_verified_experience(
            result, principal_id="finance_lead_A", task_category="external_audit_report",
            delegation="x")

        assert exp is not None
        assert exp.confirmed_facets == frozenset({"action"})  # NOT {"action", "scope"}


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
