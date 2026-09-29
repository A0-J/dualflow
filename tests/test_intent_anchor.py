"""PrincipalIntentAnchor / GroundedIntentVerifier -- Semantic Flow's Intent
Grounding boundary (docs/experiments/agent_connected_eval.md §29, Phase 3B).
All deterministic (fake `LLMClient`, no network, no `OPENAI_API_KEY`) --
mirrors `tests/test_source_semantic_gate.py`'s own fake-client patterns.
"""

from __future__ import annotations

import inspect

import pytest

from dualflow.intent_anchor import (
    GroundedIntentVerifier,
    build_intent_anchor,
    check_compatibility,
    sample_principal_intents,
)
from dualflow.llm import LLMResponse
from dualflow.principal_agent import PrincipalAgent
from dualflow.semantic import Interpretation


class _FakeLLMClient:
    """동일한 패턴, tests/test_source_semantic_gate.py 참고."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, *, instructions: str, input_text: str) -> LLMResponse:
        self.calls.append({"instructions": instructions, "input_text": input_text})
        return self._responses.pop(0)


def _structured(action: str, resource: str = "file", scope: str = "/reports/2026-08/",
               condition: str = "none") -> LLMResponse:
    return LLMResponse(
        text=f"ACTION: {action}\nRESOURCE: {resource}\nSCOPE: {scope}\nCONDITION: {condition}")


def _interp(action: str, scope: str = "/reports/2026-08/") -> Interpretation:
    return Interpretation(action, "file", scope, frozenset())


class TestSampleAndBuildAnchor:
    def test_stable_restate_confirms_every_facet(self):
        n = 3
        principal = PrincipalAgent(llm=_FakeLLMClient([_structured("summarize") for _ in range(n)]))

        anchor = build_intent_anchor(principal=principal, goal="Summarize the report.",
                                     context="", n=n, entropy_threshold=0.8)

        assert anchor.top_interpretation == _interp("summarize")
        for facet in ("action", "resource", "scope", "condition"):
            assert anchor.facets[facet].confirmed is True
            assert anchor.facets[facet].source == "principal_plan"
        assert anchor.facets["action"].value == "summarize"
        assert anchor.distribution.entropy == 0.0

    def test_unstable_restate_leaves_that_facet_unconfirmed(self):
        n = 4
        # action 2/2 split (H=1.0 > 0.8), scope 항상 동일.
        principal = PrincipalAgent(llm=_FakeLLMClient(
            [_structured("read"), _structured("read"),
            _structured("summarize"), _structured("summarize")]))

        anchor = build_intent_anchor(principal=principal, goal="Handle the report.",
                                     context="", n=n, entropy_threshold=0.8)

        assert anchor.facets["action"].confirmed is False
        assert anchor.facets["scope"].confirmed is True


class TestCompatibility:
    def test_exact_match_is_compatible(self):
        n = 3
        principal = PrincipalAgent(llm=_FakeLLMClient([_structured("summarize") for _ in range(n)]))
        anchor = build_intent_anchor(principal=principal, goal="g", context="", n=n)

        result = check_compatibility(anchor, _interp("summarize"))

        assert result.compatible is True
        assert result.matched_facets == {"action", "resource", "scope", "condition"}
        assert result.mismatched_confirmed_facets == frozenset()

    def test_confirmed_mismatch_is_incompatible(self):
        n = 3
        principal = PrincipalAgent(llm=_FakeLLMClient([_structured("summarize") for _ in range(n)]))
        anchor = build_intent_anchor(principal=principal, goal="g", context="", n=n)

        # Delegate가 confidently 다른 action을 제안한다 -- anchor의 action은
        # confirmed(n=3 전부 summarize)이므로 이 mismatch는 반드시 막혀야 한다.
        result = check_compatibility(anchor, _interp("read"))

        assert result.compatible is False
        assert result.mismatched_confirmed_facets == frozenset({"action"})

    def test_unconfirmed_mismatch_does_not_block(self):
        """Phase 3A가 진단한 핵심 원칙: anchor 자신도 반복 샘플링으로
        합의되지 않은 facet은 무조건 신뢰해서 막으면 안 된다."""
        n = 4
        principal = PrincipalAgent(llm=_FakeLLMClient(
            [_structured("read"), _structured("read"),
            _structured("summarize"), _structured("summarize")]))
        anchor = build_intent_anchor(principal=principal, goal="g", context="", n=n,
                                     entropy_threshold=0.8)
        assert anchor.facets["action"].confirmed is False  # 전제 확인

        result = check_compatibility(anchor, _interp("export"))  # action 전부와 다름

        assert result.compatible is True  # action 불일치지만 confirmed가 아니므로 안 막는다
        assert result.mismatched_unconfirmed_facets == frozenset({"action"})
        assert result.mismatched_confirmed_facets == frozenset()


class TestGroundedIntentVerifierStable:
    def test_compatible_interpretation_skips_clarification(self):
        n = 3
        principal = PrincipalAgent(llm=_FakeLLMClient([_structured("summarize") for _ in range(n)]))
        verifier = GroundedIntentVerifier(principal=principal, n=n, entropy_threshold=0.8)

        result = verifier.verify(goal="Summarize the report.", context="",
                                 interpretation=_interp("summarize"))

        assert result.clarified is False
        assert result.final_compatible is True
        assert result.post_anchor is None


class TestGroundedIntentVerifierClarifies:
    def test_confirmed_mismatch_triggers_single_round_and_resolves(self):
        n = 3
        principal_llm = _FakeLLMClient(
            [_structured("read") for _ in range(n)]           # pre restate_intent() x3 (confirmed: read)
            + [LLMResponse(text="Yes, summarize is correct.")]  # answer_clarification()
            + [_structured("summarize") for _ in range(n)])     # post restate_intent() x3 (수렴: summarize)
        principal = PrincipalAgent(llm=principal_llm)
        verifier = GroundedIntentVerifier(principal=principal, n=n, entropy_threshold=0.8)

        # Delegate가 실제로는 summarize를 제안했지만, anchor(read)와 충돌한다.
        result = verifier.verify(goal="Prepare the report.", context="finance team",
                                 interpretation=_interp("summarize"))

        assert result.pre_compatibility.compatible is False
        assert result.clarified is True
        assert result.target_facet == "action"
        assert "action = read" in result.question   # anchor(구 값)
        assert "action = summarize" in result.question  # delegate(현재 값)
        assert result.post_anchor is not None
        assert result.post_anchor.facets["action"].value == "summarize"
        assert result.final_compatible is True
        # answer_clarification()은 정확히 1회만 호출된다(결정론적 질문).
        assert len(principal_llm.calls) == n + 1 + n

    def test_multiple_mismatched_confirmed_facets_picks_one_by_fixed_priority(self):
        n = 2
        # action, scope 둘 다 delegate와 다르고 둘 다 confirmed.
        principal_llm = _FakeLLMClient(
            [_structured("read", scope="/reports/2026-08/") for _ in range(n)]
            + [LLMResponse(text="summarize the September report")]
            + [_structured("summarize", scope="/reports/2026-09/") for _ in range(n)])
        principal = PrincipalAgent(llm=principal_llm)
        verifier = GroundedIntentVerifier(principal=principal, n=n, entropy_threshold=0.8)

        result = verifier.verify(goal="g", context="",
                                 interpretation=_interp("summarize", scope="/reports/2026-09/"))

        assert result.pre_compatibility.mismatched_confirmed_facets == frozenset({"action", "scope"})
        # 우선순위(action이 scope보다 먼저)에 따라 action이 이번 라운드의 target.
        assert result.target_facet == "action"


class TestGroundedIntentVerifierResidualLimitation:
    """정직한 한계 테스트 -- anchor 자신의 반복 샘플링이 confidently 틀린
    값으로 수렴하면(Phase 2C Final의 실제 4건과 동일한 모양), Intent
    Grounding도 이걸 잡지 못한다. 이 모듈이 "완전한 해결책"이 아니라
    "단일 호출 신뢰의 문제를 닫는 것"이라는 범위를 명확히 한다."""

    def test_confidently_wrong_anchor_and_confidently_wrong_delegate_still_match(self):
        n = 3
        # anchor 쪽도 delegate 쪽도 똑같이 (틀리게) "read"로 수렴한다 --
        # Phase 2C Final 4건의 정확한 재현.
        principal = PrincipalAgent(llm=_FakeLLMClient([_structured("read") for _ in range(n)]))
        verifier = GroundedIntentVerifier(principal=principal, n=n, entropy_threshold=0.8)

        result = verifier.verify(goal="g", context="", interpretation=_interp("read"))

        assert result.pre_compatibility.compatible is True   # confirmed, 그리고 "일치"
        assert result.clarified is False
        assert result.final_compatible is True
        # 즉 이 결과만 보면 안전해 보이지만, ideal이 "summarize"라면 이건
        # 여전히 unsafe다 -- Intent Grounding은 "anchor와 delegate가
        # 서로 합의했는가"만 보장하지, ideal과 실제로 맞는지는 보장하지
        # 않는다(ground truth를 아예 안 쓰므로 구조적으로 알 수 없다).


class TestNoGroundTruthAPI:
    def test_verify_signature_has_no_truth_or_label_parameter(self):
        forbidden = ("truth", "ground_truth", "label", "expected", "ideal")
        sig = inspect.signature(GroundedIntentVerifier.verify)
        for name in sig.parameters:
            assert not any(f in name.lower() for f in forbidden)

    def test_sample_principal_intents_signature_has_no_truth_or_label_parameter(self):
        forbidden = ("truth", "ground_truth", "label", "expected", "ideal")
        sig = inspect.signature(sample_principal_intents)
        for name in sig.parameters:
            assert not any(f in name.lower() for f in forbidden)
