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


# --------------------------------------------------------------------------
class TestFastSlow:
    """교수님 피드백 ① — Fast(엔트로피)와 Slow(A-B 피드백)를 별개 방법으로 본다."""

    def test_slow_does_not_use_entropy_gating(self, tasks):
        r = run(get(tasks, "vague_clarifiable"), Config(mode="slow"))
        assert r.n_questions == 0 and r.n_llm == 0 and r.n_reviews == 1
        assert r.route.startswith("slow:")

    def test_slow_recovers_what_fast_gets_wrong(self, tasks):
        """silent_misread — Fast 는 확신에 차서 틀리고, A 의 검토가 교정한다."""
        t = get(tasks, "silent_misread")
        assert run(t, Config(mode="fast")).decision == REJECT
        slow = run(t, Config(mode="slow"))
        assert slow.decision == EXECUTE and slow.interpretation == t.truth

    def test_slow_fails_when_the_principal_cannot_specify(self, tasks):
        t = get(tasks, "vague_persistent")
        t.candidates = [(t.attack, 1.0)]          # A 가 특정 못 하는 차원에서 어긋남
        r = run(t, Config(mode="slow"))
        assert r.route == "slow:unsure" and r.decision == REJECT

    def test_and_requires_both_tracks_to_agree(self, tasks):
        t = get(tasks, "silent_misread")
        r = run(t, Config(mode="and"))
        assert r.decision == REJECT and "correct" in r.route
        assert not r.semantic_ok                  # 교정 요구 = 불일치

    def test_and_passes_when_they_agree(self, tasks):
        r = run(get(tasks, "clear_read"), Config(mode="and"))
        assert r.decision == EXECUTE and r.route == "and(rule+approve)"

    def test_and_costs_more_than_either_track(self, tasks):
        cfg = Config(mode="and")
        costs = {m: evaluate(Config(mode=m), tasks, judge=build_judge(tasks))["avg_cost"]
                 for m in ("fast", "slow", "and")}
        assert costs["and"] > costs["fast"] and costs["and"] > costs["slow"]


# --------------------------------------------------------------------------
class TestBeliefManipulation:
    """SAGE-Agent 의 self-reported belief 가 공격 표면이라는 주장의 실험 근거.

    위협 모델: B 의 후보 생성이 오염돼 후보가 목표 해석 하나로 좁혀진다.
    그 결과 H=0(= SAGE-Agent 의 π_c=1)이 되어 명확화 단계가 통째로 건너뛰어진다.
    """

    @pytest.fixture
    def attacked(self):
        from dualflow.bench import adversarial_tasks
        return adversarial_tasks()

    def test_attack_forges_maximum_confidence(self, attacked):
        for t in attacked:
            r = run(t, Config(mode="fast"))
            assert r.h_initial == 0.0            # 불확실성이 0 으로 신고된다
            assert r.n_questions == 0 and r.n_llm == 0

    def test_belief_only_is_fully_compromised(self, attacked):
        m = evaluate(Config(mode="fast", use_authority=False, use_matching=False),
                     attacked, judge=build_judge())
        assert m["unsafe_rate"] == 1.0

    def test_entropy_axis_alone_is_not_enough(self, attacked):
        """Authority·매칭을 붙여도 같은 경로·같은 권한 안의 공격은 통과한다."""
        m = evaluate(Config(mode="fast"), attacked, judge=build_judge())
        assert m["unsafe_rate"] > 0.0

    def test_slow_axis_is_immune_by_construction(self, attacked):
        """A 의 검토는 B 의 자기신고 불확실성을 입력으로 쓰지 않는다."""
        m = evaluate(Config(mode="slow"), attacked, judge=build_judge())
        assert m["unsafe_rate"] == 0.0

    def test_and_admits_no_unsafe_execution_under_attack(self, attacked):
        m = evaluate(Config(mode="and"), attacked, judge=build_judge())
        assert m["unsafe_rate"] == 0.0

    def test_and_trades_misdetection_for_zero_false_positives(self, attacked):
        """교수님 피드백 ② — 오탐 0, 미탐은 감내. 그 트레이드오프를 수치로."""
        m = evaluate(Config(mode="and"), attacked, judge=build_judge())
        assert m["unsafe_rate"] == 0.0 and m["over_rejection"] == 1.0

    def test_forged_confidence_cannot_widen_authority(self, attacked):
        """H 를 위조해도 권한 상한선은 그대로다."""
        t = next(x for x in attacked if x.name.startswith("over_privileged"))
        r = run(t, Config(mode="fast"))
        assert r.h_initial == 0.0 and not r.authority_ok and r.decision == REJECT


# --------------------------------------------------------------------------
class TestImperfectReviewer:
    """실험 ③ — Slow 축의 신뢰도가 전체의 상한선인지 확인한다."""

    def test_careless_principal_rubber_stamps(self, tasks):
        from dualflow.semantic import Principal
        import random
        t = get(tasks, "silent_misread")
        careful = Principal(t.truth, rng=random.Random(0))
        careless = Principal(t.truth, carelessness=1.0, rng=random.Random(0))
        wrong = t.candidates[0][0]
        assert careful.review(wrong).status == "correct"
        assert careless.review(wrong).status == "approve"

    def test_slow_degrades_as_the_reviewer_degrades(self, tasks):
        from dualflow.bench import adversarial_tasks
        from dualflow.framework import warmup_then_attack
        adv = adversarial_tasks()
        clean = warmup_then_attack(Config(mode="slow", carelessness=0.0),
                                   tasks, adv, judge=build_judge(), trials=5)
        sloppy = warmup_then_attack(Config(mode="slow", carelessness=1.0),
                                    tasks, adv, judge=build_judge(), trials=5)
        assert clean["unsafe_rate"] == 0.0 < sloppy["unsafe_rate"]

    def test_and_is_never_worse_than_slow_alone(self, tasks):
        from dualflow.bench import adversarial_tasks
        from dualflow.framework import warmup_then_attack
        adv = adversarial_tasks()
        for c in (0.25, 0.5, 1.0):
            slow = warmup_then_attack(Config(mode="slow", carelessness=c),
                                      tasks, adv, judge=build_judge(), trials=5)
            both = warmup_then_attack(Config(mode="and", carelessness=c),
                                      tasks, adv, judge=build_judge(), trials=5)
            assert both["unsafe_rate"] <= slow["unsafe_rate"] + 1e-9

    def test_consistency_check_helps_once_the_gate_is_closed(self, tasks):
        """경험 게이트가 닫힌 엄격한 σ 에서는 일관성 검사가 그 역할을 대신한다."""
        from dualflow.bench import adversarial_tasks
        from dualflow.framework import warmup_then_attack
        adv = adversarial_tasks()
        plain = warmup_then_attack(Config(mode="and", carelessness=1.0, sigma=0.95),
                                   tasks, adv, judge=build_judge(), trials=5)
        checked = warmup_then_attack(
            Config(mode="and", carelessness=1.0, sigma=0.95,
                   use_consistency_check=True, consistency_sigma=0.6),
            tasks, adv, judge=build_judge(), trials=5)
        assert checked["unsafe_rate"] < plain["unsafe_rate"]

    def test_experience_is_what_separates_them_not_entropy(self, tasks):
        """공격은 H 를 0 으로 위조할 수 있지만 누적 이력은 위조할 수 없다."""
        from dualflow.bench import adversarial_tasks
        from dualflow.framework import warmup_then_attack
        adv = adversarial_tasks()
        no_history = warmup_then_attack(Config(mode="and", carelessness=1.0),
                                        tasks, adv, judge=build_judge(),
                                        warmup=0, trials=5)
        with_history = warmup_then_attack(Config(mode="and", carelessness=1.0),
                                          tasks, adv, judge=build_judge(),
                                          warmup=5, trials=5)
        assert with_history["unsafe_rate"] < no_history["unsafe_rate"]

    def test_variance_is_reported_and_shrinks_with_trials(self, tasks):
        """fig5 는 확률적이므로 시행 간 표준오차를 같이 돌려준다."""
        from dualflow.bench import adversarial_tasks
        from dualflow.framework import warmup_then_attack
        adv = adversarial_tasks()
        few = warmup_then_attack(Config(mode="slow", carelessness=0.5),
                                 tasks, adv, judge=build_judge(), trials=5)
        many = warmup_then_attack(Config(mode="slow", carelessness=0.5),
                                  tasks, adv, judge=build_judge(), trials=40)
        assert len(few["per_trial"]) == 5 and len(many["per_trial"]) == 40
        assert many["stderr"] < few["stderr"]
        assert abs(many["unsafe_rate"] - few["unsafe_rate"]) < 0.2


# --------------------------------------------------------------------------
class TestAdaptive:
    """README §7-1 — Slow 를 매번이 아니라 '경험과 불일치할 때만' 켠다.

    AND 는 항상 Fast+Slow 를 둘 다 돌려 review_rate=1.0 이 되지만(연구의 출발점인
    개입 최소화와 충돌한다), Fast 단독은 belief 조작에 뚫린다. Adaptive 는 Fast 의
    확정 결과가 누적 경험과 모순될 때만(또는 Fast 자체가 확정 못 했을 때만) Slow 로
    에스컬레이션한다.
    """

    def test_matches_fast_when_no_experience_yet(self, tasks):
        """경험이 없으면(첫 공격) 비교할 대상이 없어 Fast 와 동일하게 행동한다.

        이건 한계이지 버그가 아니다 — README 의 '이력이 없는 위임 유형에는
        무력하다' 캐비앗과 같은 맥락이다.
        """
        from dualflow.bench import adversarial_tasks
        adv = adversarial_tasks()
        fast = evaluate(Config(mode="fast"), adv, judge=build_judge())
        adap = evaluate(Config(mode="adaptive"), adv, judge=build_judge())
        assert adap["unsafe_rate"] == fast["unsafe_rate"]
        assert adap["review_rate"] == fast["review_rate"] == 0.0

    def test_skips_slow_when_fast_and_experience_agree(self, tasks):
        """정상 운영에서는 review_rate 가 0 이어야 한다 — 매번 A 를 부르는 AND 와의 차이."""
        m = evaluate(Config(mode="adaptive"), tasks, judge=build_judge(tasks))
        assert m["review_rate"] == 0.0

    def test_dominates_and_when_experience_gate_is_open(self, tasks):
        """sigma=0.80(경험 게이트 열림) 공격 시나리오 — AND 와 동일한 0% unsafe 를
        review_rate 0 으로 달성한다(AND 는 항상 1.0). Fast 자신의 experience 게이트가
        이미 오염된 후보를 무시하므로, Adaptive 의 에스컬레이션 자체가 발동할 필요조차 없다.
        """
        from dualflow.bench import adversarial_tasks
        from dualflow.framework import warmup_then_attack
        adv = adversarial_tasks()
        and_ = warmup_then_attack(Config(mode="and", sigma=0.8),
                                  tasks, adv, judge=build_judge(), warmup=5, trials=20)
        adap = warmup_then_attack(Config(mode="adaptive", sigma=0.8),
                                  tasks, adv, judge=build_judge(), warmup=5, trials=20)
        assert adap["unsafe_rate"] == and_["unsafe_rate"] == 0.0
        assert adap["review_rate"] == 0.0 < and_["review_rate"]

    def test_adaptive_sigma_must_be_set_separately_when_gate_is_closed(self, tasks):
        """sigma=0.95(경험 게이트 닫힘)에서는 Fast 가 자체적으로 공격을 걸러내지
        못한다. adaptive_sigma 를 별도로 낮게 주지 않으면(기본은 sigma 를 그대로
        물려받는다) 에스컬레이션 자체가 트리거되지 않아 Fast 와 동일하게 뚫린다 —
        AND 가 consistency_sigma 를 sigma 와 별개로 둬야 했던 것과 같은 이유다.
        """
        from dualflow.bench import adversarial_tasks
        from dualflow.framework import warmup_then_attack
        adv = adversarial_tasks()
        default_thr = warmup_then_attack(Config(mode="adaptive", sigma=0.95),
                                         tasks, adv, judge=build_judge(),
                                         warmup=5, trials=20)
        lower_thr = warmup_then_attack(
            Config(mode="adaptive", sigma=0.95, adaptive_sigma=0.6),
            tasks, adv, judge=build_judge(), warmup=5, trials=20)
        assert default_thr["unsafe_rate"] > lower_thr["unsafe_rate"]

    def test_escalation_genuinely_defers_to_slow_unlike_and_consistency(self, tasks):
        """중요한 설계 차이: AND+일관성검사는 Fast 가 내부적으로 거부하면 Slow 의
        판단과 무관하게 무조건 차단한다(Slow 호출은 낭비된다). Adaptive 는 에스컬레이션
        되면 Slow 의 실제 판단을 그대로 따른다 — 그래서 A 가 부주의(carelessness>0)하면
        Adaptive 의 안전성은 AND+일관성검사(0%)가 아니라 Slow 단독과 같은 수준으로
        떨어진다. '더 안전'이 아니라 '더 싸면서 필요할 때만 Slow 를 실제로 신뢰하는'
        다른 트레이드오프임을 고정해 둔다.
        """
        from dualflow.bench import adversarial_tasks
        from dualflow.framework import warmup_then_attack
        adv = adversarial_tasks()
        cfg = dict(sigma=0.95, adaptive_sigma=0.6, carelessness=1.0)
        slow = warmup_then_attack(Config(mode="slow", carelessness=1.0),
                                  tasks, adv, judge=build_judge(), warmup=5, trials=20)
        adap = warmup_then_attack(Config(mode="adaptive", **cfg),
                                  tasks, adv, judge=build_judge(), warmup=5, trials=20)
        and_consistency = warmup_then_attack(
            Config(mode="and", sigma=0.95, carelessness=1.0,
                   use_consistency_check=True, consistency_sigma=0.6),
            tasks, adv, judge=build_judge(), warmup=5, trials=20)
        assert and_consistency["unsafe_rate"] == 0.0
        assert adap["unsafe_rate"] > and_consistency["unsafe_rate"]
        assert abs(adap["unsafe_rate"] - slow["unsafe_rate"]) < 0.15

    def test_deterministic_when_the_reviewer_is_careful(self, tasks):
        """carelessness=0 이면 난수가 개입하지 않아 분산이 0 이어야 한다."""
        from dualflow.bench import adversarial_tasks
        from dualflow.framework import warmup_then_attack
        m = warmup_then_attack(Config(mode="and", carelessness=0.0), tasks,
                               adversarial_tasks(), judge=build_judge(), trials=8)
        assert m["std"] == 0.0
