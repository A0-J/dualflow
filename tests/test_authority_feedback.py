"""AUTHORITY FEEDBACK LOOP — scope negotiation.

capability.check_authority 의 failure_kind/suggested 분류와
authority_feedback.run_feedback, semantic.Principal.review_authority 를
검증한다. DelegationBench-mini 9개 과제는 건드리지 않는다(§7-2, README 참고
— 벤치마크를 늘리면 기존 퍼센트 수치가 전부 흔들리므로, 메커니즘 자체는
독립된 시나리오로 검증한다).
"""

from __future__ import annotations

import random

from dualflow.authority_feedback import AuthorityFeedback, FeedbackDecision, run_feedback
from dualflow.capability import Budget, Privilege, check_authority
from dualflow.framework import Config, DelegationTask, DelegationVerifier, EXECUTE, REJECT
from dualflow.llm import ScriptedJudge
from dualflow.semantic import ExperienceStore, Interpretation as I, Principal


def mk_principal(truth, **kw):
    kw.setdefault("rng", random.Random(0))
    return Principal(truth, **kw)


# --------------------------------------------------------------------------
class TestCheckAuthorityClassification:
    """유무(no_grant) / 조건(condition_missing) / 범위(scope_exceeded) 를
    정확히 구분하고, scope_exceeded 일 때만 협상 가능한 suggested 를 준다."""

    def test_scope_exceeded_suggests_the_granted_boundary(self):
        eff = Budget.of(Privilege("read", "file", "/reports/2026-08/"))
        r = check_authority(eff, Privilege("read", "file", "/reports/"))
        assert r.failure_kind == "scope_exceeded"
        assert r.suggested == Privilege("read", "file", "/reports/2026-08/")

    def test_no_grant_offers_no_suggestion(self):
        eff = Budget.of(Privilege("read", "file", "/"))
        r = check_authority(eff, Privilege("delete", "file", "/tmp/"))
        assert r.failure_kind == "no_grant" and r.suggested is None

    def test_condition_missing_is_classified_separately_from_scope(self):
        """scope 는 이미 맞는 generator(*, {anonymized}) 가 있으니 scope_exceeded
        가 아니라 condition_missing 이어야 한다 — 두 실패를 섞으면 안 된다."""
        eff = Budget.of(Privilege("send", "email", "*.corp.com"),
                        Privilege("send", "email", "*", {"anonymized"}))
        r = check_authority(eff, Privilege("send", "email", "auditor.example.com"))
        assert r.failure_kind == "condition_missing"
        assert "anonymized" in r.reason


# --------------------------------------------------------------------------
class TestNegotiation:
    """T1-T7 (교수님 회의 재구성) — run_feedback() 단위 테스트."""

    def test_t1_already_within_budget_resolves_without_asking(self):
        eff = Budget.of(Privilege("read", "file", "/reports/"))
        proposal = I("read", "file", "/reports/2026-08/")
        principal = mk_principal(proposal)
        r = run_feedback(proposal, eff, principal)
        assert r.resolved and r.rounds == 0 and r.decision == FeedbackDecision.APPROVE
        assert principal.transcript == []          # 이미 권한 안 — A 를 부를 필요조차 없음

    def test_t2_overbroad_scope_gets_restricted_and_executes(self):
        eff = Budget.of(Privilege("read", "file", "/reports/2026-08/"))
        truth = I("read", "file", "/reports/2026-08/")
        proposal = I("read", "file", "/reports/")          # B 가 너무 넓게 제안
        principal = mk_principal(truth)
        r = run_feedback(proposal, eff, principal)
        assert r.resolved and r.decision == FeedbackDecision.RESTRICT
        assert r.confirmed.interpretation == truth
        assert r.rounds == 1                                # A 에게 실제로 물어본 건 1회뿐

    def test_t3_wrong_target_gets_corrected_to_exact_truth(self):
        eff = Budget.of(Privilege("read", "file", "/reports/"))     # 넓게 위임됨
        truth = I("read", "file", "/reports/2026-08/")              # 실제로 원하는 건 좁음
        proposal = I("read", "file", "*")                          # B 가 권한보다도 넓게 제안
        principal = mk_principal(truth)
        r = run_feedback(proposal, eff, principal)
        assert r.resolved and r.decision == FeedbackDecision.CORRECT
        assert r.confirmed.interpretation == truth

    def test_t4_unauthorized_action_is_not_negotiable(self):
        eff = Budget.of(Privilege("read", "file", "/"))
        proposal = I("delete", "file", "/tmp/")
        principal = mk_principal(proposal)
        r = run_feedback(proposal, eff, principal)
        assert not r.resolved and r.decision == FeedbackDecision.REJECT
        assert principal.transcript == []           # 협상 대상이 아님 — A 에게 묻지도 않음

    def test_t5_careless_principal_is_not_an_oracle(self):
        """A 가 확인 없이 범위 밖 제안을 그대로 승인해도(carelessness=1.0),
        non-amplification 검사(다음 라운드의 재검사)가 결국 막는다."""
        eff = Budget.of(Privilege("read", "file", "/reports/2026-08/"))
        truth = I("read", "file", "/finance/")             # 권한 밖의 것을 실제로 원함
        proposal = I("read", "file", "/")
        principal = mk_principal(truth, carelessness=1.0)
        r = run_feedback(proposal, eff, principal, max_rounds=3)
        assert principal.transcript[-1][1] == "(확인 안 하고) 네 그걸로 하세요"
        assert not r.resolved

    def test_t6_overcautious_principal_rejects_a_correct_suggestion(self):
        eff = Budget.of(Privilege("read", "file", "/reports/2026-08/"))
        truth = I("read", "file", "/reports/2026-08/")
        proposal = I("read", "file", "/reports/")
        principal = mk_principal(truth, overcaution=1.0)
        r = run_feedback(proposal, eff, principal)
        assert not r.resolved                               # 맞는 제안인데도 괜히 반려

    def test_t7_negotiated_result_never_exceeds_delegated_budget(self):
        """non-amplification — 협상이 성공하면 확정 결과는 항상 위임 예산 안에 있다."""
        eff = Budget.of(Privilege("read", "file", "/reports/2026-08/"))
        truth = I("read", "file", "/reports/2026-08/")
        proposal = I("read", "file", "/")
        for careless in (0.0, 1.0):
            principal = mk_principal(truth, carelessness=careless, rng=random.Random(1))
            r = run_feedback(proposal, eff, principal, max_rounds=3)
            if r.resolved:
                assert r.confirmed.interpretation.privilege() in eff

    def test_bounded_negotiation_terminates_even_against_a_broken_principal(self):
        """무한 협상 방지 — 매번 원래(권한 밖) 제안을 그대로 돌려주는 가짜
        Principal 을 상대로도 정확히 max_rounds 만 묻고 멈춘다."""
        class Stubborn:
            def __init__(self):
                self.calls = 0

            def review_authority(self, proposed, suggested):
                self.calls += 1
                return AuthorityFeedback(FeedbackDecision.RESTRICT, proposed)  # 안 좁혀짐

        eff = Budget.of(Privilege("read", "file", "/reports/2026-08/"))
        proposal = I("read", "file", "/reports/")
        stub = Stubborn()
        r = run_feedback(proposal, eff, stub, max_rounds=2)
        assert not r.resolved
        assert stub.calls == 2

    def test_run_feedback_only_needs_the_review_authority_method(self):
        """runtime 경계 — run_feedback 은 principal.review_authority() 의
        응답만 보고 task.truth 를 직접 들여다보지 않는다. 그런 속성 자체가
        없는 최소 객체로도 똑같이 동작한다는 것으로 구조적으로 확인한다."""
        class Minimal:
            def __init__(self, response):
                self.response = response
                self.calls = 0

            def review_authority(self, proposed, suggested):
                self.calls += 1
                return self.response

        eff = Budget.of(Privilege("read", "file", "/reports/2026-08/"))
        proposal = I("read", "file", "/reports/")
        confirmed = I("read", "file", "/reports/2026-08/")
        stub = Minimal(AuthorityFeedback(FeedbackDecision.RESTRICT, confirmed))
        r = run_feedback(proposal, eff, stub)
        assert r.resolved and r.confirmed.interpretation == confirmed
        assert stub.calls == 1
        assert not hasattr(stub, "truth")


# --------------------------------------------------------------------------
class TestPipelineIntegration:
    """DelegationVerifier.run() 전체 경로 — Fast 가 권한 밖으로 확신해도
    Authority Feedback Loop 가 task.truth 를 직접 보지 않고 살려낸다."""

    def _task(self, candidates, truth, ceiling_scope="/reports/2026-08/"):
        from dualflow.bench import PRINCIPAL, _sys
        return DelegationTask(
            name="overbroad_pilot", category="scope-negotiable",
            spec="8월 매출 리포트만 읽어줘",
            principal_budget=PRINCIPAL,
            ceilings=[Budget.of(Privilege("read", "file", ceiling_scope))],
            candidates=candidates, truth=truth, sysvars=_sys(),
        )

    def test_overbroad_fast_confirmation_is_recovered_by_authority_feedback(self):
        truth = I("read", "file", "/reports/2026-08/", label="8월만")
        overbroad = I("read", "file", "/reports/", label="전체")
        task = self._task(candidates=[(overbroad, 0.90), (truth, 0.10)], truth=truth)
        v = DelegationVerifier(Config(mode="fast"), ExperienceStore(),
                               ScriptedJudge({task.spec: truth}))
        r = v.run(task)
        assert r.decision == EXECUTE
        assert r.interpretation == truth
        assert r.authority_negotiated and r.n_authority_feedback >= 1

    def test_without_feedback_the_same_case_is_rejected(self):
        truth = I("read", "file", "/reports/2026-08/", label="8월만")
        overbroad = I("read", "file", "/reports/", label="전체")
        task = self._task(candidates=[(overbroad, 0.90), (truth, 0.10)], truth=truth)
        v = DelegationVerifier(Config(mode="fast", use_authority_feedback=False),
                               ExperienceStore(), ScriptedJudge({task.spec: truth}))
        r = v.run(task)
        assert r.decision == REJECT

    def test_belief_manipulation_does_not_defeat_authority_feedback(self):
        """공격이 후보를 과도한 범위 하나로 좁혀 H=0 을 만들어도, Authority
        Feedback 은 B 의 자기신고 확신이 아니라 A 의 실제 응답을 보므로
        영향받지 않는다 — Slow 축의 면역성과 같은 이유다."""
        truth = I("read", "file", "/reports/2026-08/", label="8월만")
        attack = I("read", "file", "/reports/", label="전체")
        task = self._task(candidates=[(attack, 1.0)], truth=truth)
        v = DelegationVerifier(Config(mode="fast"), ExperienceStore(),
                               ScriptedJudge({task.spec: truth}))
        r = v.run(task)
        assert r.decision == EXECUTE and r.interpretation == truth

    def test_bounded_rounds_are_reflected_in_cost(self):
        """협상 라운드가 비용에 반영된다 — Slow 리뷰와 같은 급의 비용(§ Config
        .cost_authority_feedback)으로 취급한다."""
        truth = I("read", "file", "/reports/2026-08/", label="8월만")
        overbroad = I("read", "file", "/reports/", label="전체")
        task = self._task(candidates=[(overbroad, 0.90), (truth, 0.10)], truth=truth)
        cfg = Config(mode="fast")
        v = DelegationVerifier(cfg, ExperienceStore(), ScriptedJudge({task.spec: truth}))
        r = v.run(task)
        assert r.cost(cfg) == r.n_authority_feedback * cfg.cost_authority_feedback
