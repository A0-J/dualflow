"""SEMANTIC FLOW 검증 — 엔트로피 성질, 정보이득 비음수성, 역질의, 경험 축적."""

from __future__ import annotations

import math
import random
from collections import Counter

import pytest

from dualflow.semantic import (
    Answer, ExperienceStore, Interpretation as I, Principal, Question,
    apply_answer, build_belief, candidate_questions, conditional_entropy,
    entropy, information_gain, normalize, normalized_entropy, select_question, top,
)

A = I("read", "file", "/reports/", label="a")
B = I("read", "file", "/finance/", label="b")
C = I("export", "file", "/reports/", label="c")
D = I("write", "file", "/reports/", label="d")

CANDIDATES = [(C, 0.35), (A, 0.30), (B, 0.20), (D, 0.15)]


class TestEntropy:
    def test_singleton_has_zero_entropy(self):
        assert entropy({A: 1.0}) == 0.0

    def test_uniform_is_maximal(self):
        uni = normalize({A: 1, B: 1, C: 1, D: 1})
        assert entropy(uni) == pytest.approx(2.0)
        assert normalized_entropy(uni) == pytest.approx(1.0)

    def test_bounded_by_log_support(self):
        p = build_belief(CANDIDATES)
        assert 0.0 <= entropy(p) <= math.log2(len(p)) + 1e-12

    def test_concentration_lowers_entropy(self):
        assert entropy(normalize({A: 9, B: 1})) < entropy(normalize({A: 6, B: 4}))

    def test_never_negative_on_degenerate_input(self):
        assert entropy({A: 1.0, B: 0.0}) == 0.0


class TestInformationGain:
    """IG = I(해석 ; 답변) 이므로 비음수. SAGE-Agent 의 EVPI 가 잃어버린 성질이다."""

    def test_equals_entropy_minus_conditional(self):
        p = build_belief(CANDIDATES)
        for q in candidate_questions(p):
            assert information_gain(p, q) == pytest.approx(
                entropy(p) - conditional_entropy(p, q.dimension))

    def test_non_negative_on_random_beliefs(self):
        rng = random.Random(0)
        for _ in range(200):
            weights = {c: rng.random() for c in (A, B, C, D)}
            p = normalize(weights)
            for q in candidate_questions(p):
                assert information_gain(p, q) >= -1e-12

    def test_zero_for_already_determined_dimension(self):
        p = build_belief(CANDIDATES)
        q = Question("resource?", "resource")     # 모든 후보가 file
        assert information_gain(p, q) == pytest.approx(0.0)

    def test_full_resolution_gives_all_the_entropy(self):
        p = normalize({A: 1, B: 1})               # scope 만 다름
        assert information_gain(p, Question("", "scope")) == pytest.approx(1.0)

    def test_caveat_gain_is_not_submodular(self):
        """SAGE-Agent Prop.2(2) 의 '질문 순서에 대한 수확 체감' 은 일반적으로 거짓이다.

        엔트로피의 submodularity 로부터 유도했다고 적혀 있지만, 조건부 상호정보량
        I(X;S|A) 는 I(X;S) 보다 커질 수 있다(시너지). 아래가 그 반례다: scope 질문은
        단독으로는 0.72 bits 를 주지만, action 을 먼저 알고 나면 0.97 bits 를 준다.

        실용적 함의: 질문 예산 k 를 정할 때 '앞 질문이 더 이득' 이라고 가정하면 안 되고,
        매 라운드 IG 를 다시 계산해야 한다. 본 구현이 루프 안에서 재계산하는 이유다.
        """
        p = build_belief(CANDIDATES)
        before = information_gain(p, Question("", "scope"))
        after = information_gain(apply_answer(p, Answer("action", "read")),
                                 Question("", "scope"))
        assert after > before

    def test_joint_conditioning_still_reduces_entropy(self):
        """실제로 성립하는 성질: H(X) ≥ H(X|A) ≥ H(X|A,S). 종료성은 여기서 나온다."""
        p = build_belief(CANDIDATES)
        h = entropy(p)
        h_a = conditional_entropy(p, "action")

        h_as = 0.0
        for value in {i.value_of("action") for i in p}:
            bucket = {i: w for i, w in p.items() if i.value_of("action") == value}
            mass = sum(bucket.values())
            h_as += mass * conditional_entropy(normalize(bucket), "scope")

        assert h >= h_a - 1e-12 >= h_as - 1e-12
        assert h_as == pytest.approx(0.0)      # 두 질문이면 완전히 확정된다


class TestClarification:
    def test_answer_eliminates_and_renormalises(self):
        p = apply_answer(build_belief(CANDIDATES), Answer("action", "read"))
        assert set(p) == {A, B}
        assert sum(p.values()) == pytest.approx(1.0)

    def test_no_answer_leaves_belief_untouched(self):
        p = build_belief(CANDIDATES)
        assert apply_answer(p, Answer("scope", None)) == p

    def test_answer_outside_the_candidate_set_is_ignored(self):
        p = build_belief(CANDIDATES)
        assert apply_answer(p, Answer("action", "teleport")) == p

    def test_highest_gain_question_is_selected(self):
        p = build_belief(CANDIDATES)
        q, _ = select_question(p, Counter(), lam=0.1)
        assert q.dimension == "action"

    def test_redundancy_penalty_moves_on(self):
        p = build_belief(CANDIDATES)
        q, _ = select_question(p, Counter({"action": 9}), lam=0.1)
        assert q.dimension == "scope"

    def test_stops_when_nothing_left_to_gain(self):
        q, _ = select_question({A: 1.0}, Counter(), lam=0.1)
        assert q is None

    def test_principal_answers_only_what_is_asked(self):
        pr = Principal(A, refuses=("scope",))
        assert pr.answer(Question("", "action")).value == "read"
        assert pr.answer(Question("", "scope")).value is None


class TestExperience:
    def test_only_accepted_outcomes_are_recorded(self):
        s = ExperienceStore()
        s.record("k", A, accepted=False)
        assert s.n("k") == 0 and s.score("k") == 0.0

    def test_score_grows_with_consistent_evidence(self):
        s = ExperienceStore()
        seen = []
        for _ in range(5):
            s.record("k", A, accepted=True)
            seen.append(s.score("k"))
        assert seen == sorted(seen) and seen[-1] > 0.8
        assert s.best("k") == A

    def test_conflicting_evidence_keeps_score_low(self):
        s = ExperienceStore()
        for interp in (A, B, A, B):
            s.record("k", interp, accepted=True)
        assert s.score("k") < 0.8

    def test_prior_sharpens_the_belief(self):
        s = ExperienceStore()
        for _ in range(3):
            s.record("k", A, accepted=True)
        plain = build_belief(CANDIDATES)
        primed = build_belief(CANDIDATES, s.prior("k"))
        assert entropy(primed) < entropy(plain)
        assert top(primed) == A          # 경험이 1위 후보를 뒤집는다
        assert top(plain) == C
