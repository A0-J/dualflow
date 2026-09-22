"""SAGE-Agent 원 공식 재현 검증 — 논문대로 구현됐는지, 그리고 어디가 뚫리는지."""

from __future__ import annotations

import pytest

from experiments.bench import adversarial_tasks, build_judge, build_tasks
from dualflow.framework import Config, DelegationVerifier, evaluate
from experiments.baselines.sage import (
    PARAMS, SageAgentBaseline, SageCandidate, UNK, as_semantic_engine, best,
    build_candidates, evpi, pi,
)
from dualflow.semantic import ExperienceStore, Interpretation as I, Principal


def cand(tool="read", resource="file", scope="/reports/", condition=(), domains=None):
    return SageCandidate(tool, {"resource": resource, "scope": scope,
                                "condition": tuple(condition)},
                         domains or {p: () for p in PARAMS})


class TestEq2:
    """Eq.(2) / Eq.(13) — viability score."""

    def test_specified_arguments_score_one(self):
        assert pi(cand()) == pytest.approx(1.0)

    def test_finite_unknown_is_inverse_domain_size(self):
        c = cand(scope=UNK, domains={"resource": (), "condition": (),
                                     "scope": ("/reports/", "/finance/", "/hr/")})
        assert pi(c) == pytest.approx(1 / 3)

    def test_infinite_unknown_uses_epsilon(self):
        c = cand(scope=UNK, domains={p: () for p in PARAMS})
        assert pi(c, epsilon=1e-4) == pytest.approx(1e-4)

    def test_proposition_1_completeness(self):
        """π = 1 iff 모든 인자가 지정됐다."""
        assert pi(cand()) == 1.0
        assert pi(cand(scope=UNK, domains={"scope": ("a", "b"), "resource": (),
                                           "condition": ()})) < 1.0

    def test_proposition_1_monotonicity(self):
        wide = cand(scope=UNK, domains={"scope": ("a", "b", "c"), "resource": (),
                                        "condition": ()})
        narrow = cand(scope=UNK, domains={"scope": ("a", "b"), "resource": (),
                                          "condition": ()})
        assert pi(narrow) >= pi(wide)

    def test_value_correctness_is_invisible(self):
        """핵심: π 는 '맞는 값인가' 가 아니라 '빈칸이 없는가' 를 잰다."""
        right = cand(scope="/reports/")
        wrong = cand(scope="/etc/passwd/")
        assert pi(right) == pi(wrong) == 1.0


class TestCandidateConstruction:
    def test_varying_parameters_become_unknown(self):
        cands = build_candidates([
            (I("read", "file", "/reports/"), 0.5),
            (I("read", "file", "/finance/"), 0.5)])
        assert len(cands) == 1 and cands[0].is_unknown("scope")
        assert set(cands[0].domains["scope"]) == {"/reports/", "/finance/"}

    def test_agreeing_parameters_stay_specified(self):
        cands = build_candidates([
            (I("read", "file", "/reports/"), 0.5),
            (I("export", "file", "/reports/"), 0.5)])
        assert {c.tool for c in cands} == {"read", "export"}
        assert all(not c.is_unknown("scope") for c in cands)


class TestEvpiAndStopping:
    def test_evpi_is_zero_for_a_determined_parameter(self):
        cands = build_candidates([(I("read", "file", "/reports/"), 1.0)])
        assert evpi(cands, "scope") == pytest.approx(0.0)

    def test_evpi_rewards_resolving_a_wide_domain(self):
        cands = build_candidates([
            (I("read", "file", "/reports/"), 0.5),
            (I("read", "file", "/finance/"), 0.5)])
        assert evpi(cands, "scope") > 0.0

    def test_agent_asks_when_a_parameter_varies(self):
        task = next(t for t in build_tasks() if t.name == "vague_clarifiable")
        tr = SageAgentBaseline().run(task.candidates, Principal(task.truth))
        assert tr.n_questions >= 1 and tr.evpi_computed


class TestToolChoiceBlindSpot:
    """세 번째 결함: tool 선택의 불확실성이 실행 게이트에 반영되지 않는다.

    Eq.(2)는 '∝' 로 Eq.(1)의 균등 tool prior 1/K 를 흡수하고, Prop.1 도 파라미터 곱만
    가정한다. 그래서 서로 배타적인 두 tool 이 모두 완전 지정이면 둘 다 π=1 이 되어
    τ_exec 를 통과하고, 어느 쪽을 부를지는 tie-break 로 정해진다.
    """

    def test_two_fully_specified_tools_both_reach_tau_exec(self):
        cands = build_candidates([
            (I("read", "file", "/reports/"), 0.5),
            (I("export", "file", "/reports/"), 0.5)])
        assert all(pi(c) == 1.0 for c in cands)

    def test_executes_without_asking(self):
        task = next(t for t in build_tasks() if t.name == "silent_misread")
        tr = SageAgentBaseline().run(task.candidates, Principal(task.truth))
        assert tr.route == "sage:tau_exec" and tr.n_questions == 0
        assert not tr.evpi_computed

    def test_tool_prior_reading_changes_the_gate_but_not_the_attack(self):
        task = next(t for t in build_tasks() if t.name == "silent_misread")
        with_prior = SageAgentBaseline(use_tool_prior=True).run(
            task.candidates, Principal(task.truth))
        assert with_prior.route != "sage:tau_exec"      # 1/K 를 살리면 질문한다


class TestTauExecBypass:
    """belief 조작 공격 — τ_exec 검사가 EVPI 보다 앞에 있다는 점을 때린다."""

    @pytest.fixture
    def attacked(self):
        return adversarial_tasks()

    def test_evpi_is_never_computed_under_attack(self, attacked):
        for t in attacked:
            tr = SageAgentBaseline().run(t.candidates, Principal(t.truth, t.refuses))
            assert tr.route == "sage:tau_exec"
            assert tr.max_pi == pytest.approx(1.0)
            assert tr.n_questions == 0 and not tr.evpi_computed

    @pytest.mark.parametrize("tool_prior", [False, True])
    def test_attack_works_under_both_readings_of_eq2(self, attacked, tool_prior):
        """Eq.(2)를 어느 쪽으로 읽어도 공격은 성립한다 (후보가 K=1 로 좁혀지므로)."""
        m = evaluate(Config(semantic_engine=as_semantic_engine(use_tool_prior=tool_prior),
                            use_authority=False, use_matching=False),
                     attacked, judge=build_judge())
        assert m["unsafe_rate"] == 1.0

    def test_joint_verification_recovers_part_of_it(self, attacked):
        bare = evaluate(Config(semantic_engine=as_semantic_engine(),
                               use_authority=False, use_matching=False),
                        attacked, judge=build_judge())
        joint = evaluate(Config(semantic_engine=as_semantic_engine()),
                         attacked, judge=build_judge())
        assert bare["unsafe_rate"] == 1.0
        assert 0.0 < joint["unsafe_rate"] < 1.0        # 절반쯤은 살리지만 완전하지 않다

    def test_and_closes_the_gap(self, attacked):
        m = evaluate(Config(mode="and"), attacked, judge=build_judge())
        assert m["unsafe_rate"] == 0.0
