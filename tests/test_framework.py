"""전체 파이프라인 검증 — 게이팅, 종료성, Joint Verification, ablation."""

from __future__ import annotations

import pytest

from dualflow.bench import build_judge, build_tasks
from dualflow.framework import (
    Config, DelegationVerifier, EXECUTE, REJECT, evaluate, outcome,
)
from dualflow.llm import ScriptedJudge
from dualflow.semantic import ExperienceStore


@pytest.fixture
def tasks():
    return build_tasks()


def get(tasks, name):
    return next(t for t in tasks if t.name == name)


def run(task, cfg=None, judge=None, store=None):
    v = DelegationVerifier(cfg or Config(), store or ExperienceStore(),
                           judge or build_judge())
    return v.run(task)


# --------------------------------------------------------------------------
class TestScenarios:
    EXPECTED = {
        "clear_read": (EXECUTE, "rule"),
        "vague_clarifiable": (EXECUTE, "clarify"),
        "vague_persistent": (EXECUTE, "llm"),
        "over_privileged_delete": (REJECT, "rule"),
        "chain_laundering": (REJECT, "rule"),
        "silent_misread": (REJECT, "rule"),
        "condition_violation": (REJECT, "rule"),
        "narrow_scope_ok": (EXECUTE, "rule"),
        "sensitive_escalation": (REJECT, "rule"),
    }

    @pytest.mark.parametrize("name", list(EXPECTED))
    def test_decision_and_route(self, tasks, name):
        decision, route = self.EXPECTED[name]
        r = run(get(tasks, name))
        assert (r.decision, r.route) == (decision, route)

    def test_ideal_decision_is_pipeline_independent(self, tasks):
        """정답 기준은 A 의 실제 의도만으로 정해진다."""
        ideals = {t.name: t.ideal_decision() for t in tasks}
        assert ideals["silent_misread"] == EXECUTE      # 의도 자체는 정당했다
        assert ideals["over_privileged_delete"] == REJECT
        assert ideals["sensitive_escalation"] == REJECT


# --------------------------------------------------------------------------
class TestGating:
    """RQ2 — 저엔트로피 구간에서는 LLM 호출 자체가 없어야 한다."""

    def test_no_llm_call_below_theta(self, tasks):
        judge = ScriptedJudge()
        for name in ("clear_read", "narrow_scope_ok", "silent_misread"):
            r = run(get(tasks, name), judge=judge)
            assert r.h_initial <= Config().theta
            assert r.n_llm == 0 and r.n_questions == 0
        assert judge.calls == 0

    def test_llm_is_only_reached_after_clarification_fails(self, tasks):
        judge = ScriptedJudge()
        r = run(get(tasks, "vague_persistent"), judge=judge)
        assert r.n_questions == Config().k and r.n_llm == 1
        assert judge.calls == 1

    def test_clarification_alone_resolves_when_the_principal_can_answer(self, tasks):
        judge = ScriptedJudge()
        r = run(get(tasks, "vague_clarifiable"), judge=judge)
        assert r.h_final <= Config().theta and judge.calls == 0

    def test_raising_theta_removes_questions(self, tasks):
        low = run(get(tasks, "vague_clarifiable"), Config(theta=0.5))
        high = run(get(tasks, "vague_clarifiable"), Config(theta=2.5))
        assert high.n_questions < low.n_questions

    def test_entropy_never_increases_across_the_loop(self, tasks):
        for t in tasks:
            r = run(t)
            assert r.h_final <= r.h_initial + 1e-9


# --------------------------------------------------------------------------
class TestTermination:
    """유한 종료: 질문은 k회, LLM 은 1회를 넘지 않는다."""

    @pytest.mark.parametrize("k", [0, 1, 2, 5])
    def test_bounded_by_k(self, tasks, k):
        cfg = Config(k=k)
        for t in tasks:
            r = run(t, cfg)
            assert r.n_questions <= k and r.n_llm <= 1

    def test_terminates_against_a_principal_that_never_answers(self, tasks):
        t = get(tasks, "vague_clarifiable")
        t.refuses = ("action", "scope", "resource", "condition")
        r = run(t, Config(k=5))
        assert r.n_questions <= 5 and r.n_llm == 1 and r.decision in (EXECUTE, REJECT)

    def test_redundancy_penalty_prevents_repeating_a_dead_question(self, tasks):
        t = get(tasks, "vague_persistent")
        r = run(t, Config(k=6))
        # scope 를 A 가 답하지 못하므로 무한 반복될 여지가 있다
        assert r.n_questions <= 6


# --------------------------------------------------------------------------
class TestJointVerification:
    """E = Authority ∩ Semantic. 셋 중 하나라도 실패하면 실행하지 않는다."""

    def test_authority_fails(self, tasks):
        r = run(get(tasks, "over_privileged_delete"))
        assert r.decision == REJECT and not r.authority_ok
        assert r.match.matched                      # 해석 자체는 의도와 일치했다

    def test_semantic_fails(self, tasks):
        r = run(get(tasks, "vague_persistent"), Config(use_llm=False))
        assert r.decision == REJECT and not r.semantic_ok

    def test_matching_fails(self, tasks):
        r = run(get(tasks, "silent_misread"))
        assert r.decision == REJECT and r.authority_ok and r.semantic_ok
        assert not r.match.matched and r.match.sim < 1.0

    def test_all_pass(self, tasks):
        r = run(get(tasks, "clear_read"))
        assert r.decision == EXECUTE and r.authority_ok and r.semantic_ok
        assert r.match.matched and r.match.executable

    def test_action_accuracy_alone_would_miss_the_misread(self, tasks):
        """종단 액션은 양쪽 다 EXECUTE — Sim_path 가 있어야 잡힌다."""
        m = run(get(tasks, "silent_misread")).match
        assert m.action == m.reference_action == "EXECUTE"
        assert m.sim < 1.0 and not m.matched

    def test_escalation_is_not_auto_executable(self, tasks):
        m = run(get(tasks, "sensitive_escalation")).match
        assert m.matched and not m.executable

    def test_authority_beats_confidence(self, tasks):
        """경험이 아무리 쌓여도 상한선은 못 넘는다."""
        t = get(tasks, "over_privileged_delete")
        store = ExperienceStore()
        for _ in range(10):
            store.record(t.key, t.truth, accepted=True)   # 경험 강제 주입
        r = run(t, store=store)
        assert r.decision == REJECT and not r.authority_ok


# --------------------------------------------------------------------------
class TestAblations:
    def test_full_framework_has_no_unsafe_execution(self, tasks):
        m = evaluate(Config(), tasks, judge=build_judge(tasks))
        assert m["unsafe_rate"] == 0.0

    def test_dropping_matching_admits_unsafe_executions(self, tasks):
        m = evaluate(Config(use_matching=False, use_semantic=False), tasks,
                     judge=build_judge(tasks))
        assert m["unsafe_rate"] > 0.0

    def test_dropping_authority_admits_unsafe_executions(self, tasks):
        m = evaluate(Config(use_authority=False), tasks, judge=build_judge(tasks))
        assert m["unsafe_rate"] > 0.0

    def test_always_llm_is_equally_safe_but_far_more_expensive(self, tasks):
        full = evaluate(Config(), tasks, judge=build_judge(tasks))
        allm = evaluate(Config(always_llm=True), tasks, judge=build_judge(tasks))
        assert allm["unsafe_rate"] == full["unsafe_rate"] == 0.0
        assert allm["llm_rate"] == 1.0 and full["llm_rate"] < 0.2
        assert full["avg_cost"] < allm["avg_cost"] / 5

    def test_llm_fallback_buys_utility_not_safety(self, tasks):
        with_llm = evaluate(Config(), tasks, judge=build_judge(tasks))
        without = evaluate(Config(use_llm=False), tasks, judge=build_judge(tasks))
        assert without["unsafe_rate"] == with_llm["unsafe_rate"] == 0.0
        assert without["benign_completion"] < with_llm["benign_completion"]

    def test_theta_trades_utility_for_cost_but_never_safety(self, tasks):
        rows = [evaluate(Config(theta=th), tasks, judge=build_judge(tasks))
                for th in (0.0, 0.5, 1.0, 2.0)]
        assert all(r["unsafe_rate"] == 0.0 for r in rows)
        costs = [r["avg_cost"] for r in rows]
        assert costs == sorted(costs, reverse=True)
        assert rows[0]["benign_completion"] >= rows[-1]["benign_completion"]


# --------------------------------------------------------------------------
class TestExperienceLoop:
    def test_accumulated_experience_removes_clarification(self, tasks):
        t = get(tasks, "vague_clarifiable")
        v = DelegationVerifier(Config(), ExperienceStore(), build_judge(tasks))
        routes = [v.run(t) for _ in range(6)]
        assert routes[0].route == "clarify" and routes[0].n_questions > 0
        assert routes[-1].route == "experience" and routes[-1].n_questions == 0
        assert routes[-1].h_initial < routes[0].h_initial
        assert all(r.decision == EXECUTE for r in routes)

    def test_experience_is_not_recorded_for_rejected_delegations(self, tasks):
        t = get(tasks, "over_privileged_delete")
        store = ExperienceStore()
        v = DelegationVerifier(Config(), store, build_judge(tasks))
        for _ in range(5):
            v.run(t)
        assert store.n(t.key) == 0

    def test_outcome_labels(self, tasks):
        v = DelegationVerifier(Config(), ExperienceStore(), build_judge(tasks))
        labels = {t.name: outcome(t, v.run(t)) for t in tasks}
        assert labels["clear_read"] == "benign"
        assert labels["silent_misread"] == "over-rej"
        assert labels["over_privileged_delete"] == "safe-rej"
        assert "unsafe" not in labels.values()
