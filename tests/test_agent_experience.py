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
    """v3 provenance 최종 계약(§23 follow-up 5) — `confirmed_facets`는
    `result.question.target_facets`를 무조건 승격한 값이 아니다.
    `target_facets`에 있으면서 *동시에* `post_distribution`에서 실제로
    하나의 값으로 수렴한 facet만 confirmed로 승격한다
    (`build_verified_experience._facet_resolved()`). "이 facet이 질문
    대상으로 지정됐다"(target)와 "Principal이 실제로 이 facet을 확인했다"
    (confirmed)는 여전히 다른 사실이고, singleton 구조(§23 follow-up 4)
    만으로는 "모호하거나 회피하는 답변"까지 걸러지지 않는다는 게 이
    follow-up의 출발점이다 — 답변 텍스트를 재해석하지 않고 이미 있는
    `post_distribution` 구조로 그 잔여 위험을 닫는다."""

    def test_confirmed_facets_requires_both_target_and_post_resolution(self):
        """기본(엄격한) max_verified_entropy=0.0에서는 post_distribution
        자체가 이미 완전히 수렴해 있어야 하므로, target facet도 당연히
        수렴돼 있다 -- confirmed_facets == target_facets."""
        result = _clarified_result(post_entropy=0.0, target_facets=frozenset({"action"}))

        exp = build_verified_experience(
            result, principal_id="finance_lead_A", task_category="external_audit_report",
            delegation="Please prepare the August 2026 financial report.")

        assert exp is not None
        assert exp.confirmed_facets == frozenset({"action"})

    def test_target_facet_unresolved_post_clarification_is_not_confirmed(self):
        """핵심 케이스: max_verified_entropy를 느슨하게 줘서 joint-entropy
        gate는 통과하지만, target facet(action) 자체는 post-clarification
        에서도 여전히 두 값으로 갈려 있다 -- 이 경우 target=action이어도
        confirmed_facets는 비어 있어야 한다(모호한/미해결 답변을 confirmed로
        승격하지 않는다)."""
        summarize_aug = Interpretation("summarize", "file", "/reports/2026-08/", frozenset())
        export_aug = Interpretation("export", "file", "/reports/2026-08/", frozenset())
        pre = _distribution({summarize_aug: 0.6, export_aug: 0.4}, 0.971)
        post = _distribution({summarize_aug: 0.7, export_aug: 0.3}, 0.881)  # action STILL varies
        question = ClarificationQuestion(
            question="Summarize or export?", raw_text="Summarize or export?",
            response=LLMResponse(text="Summarize or export?"),
            belief=pre.belief, entropy=pre.entropy, target_facets=frozenset({"action"}))
        answer = PrincipalClarification(answer="Not totally sure, maybe summarize.",
                                       response=LLMResponse(text="Not totally sure, maybe summarize."))
        result = ClarificationResult(
            clarified=True, pre_distribution=pre, post_distribution=post,
            question=question, answer=answer, final_interpretation=post.top)

        exp = build_verified_experience(
            result, principal_id="finance_lead_A", task_category="external_audit_report",
            delegation="x", max_verified_entropy=0.9)  # loose enough for the joint gate to pass

        assert exp is not None  # joint gate passed
        assert exp.confirmed_facets == frozenset()  # but action itself never resolved

    def test_non_target_facet_never_confirmed_even_if_it_happens_to_resolve(self):
        """target_facets={"action"}뿐인데 post_distribution이 우연히
        단일 candidate로 수렴해서 scope도 "resolve된 것처럼" 보이는
        경우에도, scope는 애초에 target이 아니었으므로 confirmed_facets에
        들어가면 안 된다 -- follow-up 4의 불변식(confirmed_facets는
        항상 target_facets의 부분집합)이 이 gate 추가 후에도 유지되는지
        확인."""
        summarize_sept = Interpretation("summarize", "file", "/reports/2026-09/", frozenset())
        pre = _distribution({summarize_sept: 0.6,
                            Interpretation("export", "file", "/reports/2026-08/", frozenset()): 0.4},
                           0.971)
        post = _distribution({summarize_sept: 1.0}, 0.0)  # single candidate -> every facet "resolves"
        question = ClarificationQuestion(
            question="Summarize or export?", raw_text="Summarize or export?",
            response=LLMResponse(text="Summarize or export?"),
            belief=pre.belief, entropy=pre.entropy,
            target_facets=frozenset({"action"}))  # only action was ever targeted
        answer = PrincipalClarification(answer="Summarize.", response=LLMResponse(text="Summarize."))
        result = ClarificationResult(
            clarified=True, pre_distribution=pre, post_distribution=post,
            question=question, answer=answer, final_interpretation=post.top)

        exp = build_verified_experience(
            result, principal_id="finance_lead_A", task_category="external_audit_report",
            delegation="x")

        assert exp is not None
        assert exp.confirmed_facets == frozenset({"action"})
        assert "scope" not in exp.confirmed_facets

    def test_defensive_multi_target_facets_are_intersected_with_resolution(self):
        """build_verified_experience() 자체는 target_facets가 여러 개
        들어와도(실제 resolve()는 항상 singleton-or-empty만 만들지만,
        이 함수 자체의 계약은 그것에 의존하지 않는다) 방어적으로 동작한다
        -- 실제로 post에서 수렴한 것만 승격된다."""
        result = _clarified_result(post_entropy=0.0,
                                   target_facets=frozenset({"action", "scope"}))

        exp = build_verified_experience(
            result, principal_id="finance_lead_A", task_category="external_audit_report",
            delegation="x")

        assert exp is not None
        # post_entropy=0.0 -> _clarified_result()의 post는 단일 candidate이므로
        # action/scope 둘 다 (우연히) 수렴 상태 -- 방어적 필터를 통과해 둘 다 남는다.
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
