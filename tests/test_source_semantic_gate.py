"""SourceSemanticGate / SourceClarifyingPrincipal -- Semantic Flow's new
source-side boundary (docs/experiments/agent_connected_eval.md §29,
Phase 2C-P4 Finding P2C-F1). All deterministic (fake `LLMClient`, no
network, no `OPENAI_API_KEY` needed) -- mirrors `tests/test_clarification.
py`'s own fake-client patterns exactly.
"""

from __future__ import annotations

import pytest

from dualflow.delegate_agent import CandidateDistribution, DelegateAgent
from dualflow.llm import LLMResponse
from dualflow.principal_agent import PrincipalAgent
from dualflow.semantic import Interpretation
from dualflow.source_semantic_gate import (
    SourceClarifyingPrincipal,
    SourceSemanticGate,
    facet_entropies,
    sample_principal_delegations,
)


class _FakeLLMClient:
    """동일한 패턴, tests/test_clarification.py 참고."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, *, instructions: str, input_text: str) -> LLMResponse:
        self.calls.append({"instructions": instructions, "input_text": input_text})
        return self._responses.pop(0)


def _delegation_text(text: str = "Please handle the report.") -> LLMResponse:
    return LLMResponse(text=text)


def _structured(action: str, resource: str = "file", scope: str = "/reports/2026-08/",
               condition: str = "none") -> LLMResponse:
    return LLMResponse(
        text=f"ACTION: {action}\nRESOURCE: {resource}\nSCOPE: {scope}\nCONDITION: {condition}")


def _interp(action: str, scope: str = "/reports/2026-08/") -> Interpretation:
    return Interpretation(action, "file", scope, frozenset())


def _distribution(belief: dict[Interpretation, float]) -> CandidateDistribution:
    from dualflow.semantic import entropy as compute_entropy
    top = max(belief, key=lambda i: belief[i])
    return CandidateDistribution(belief=belief, entropy=compute_entropy(belief), top=top,
                                 top_probability=belief[top], n_unique=len(belief),
                                 n_samples=20, responses=[])


class TestFacetEntropies:
    def test_stable_distribution_has_zero_entropy_on_every_facet(self):
        dist = _distribution({_interp("read"): 1.0})
        entropies = facet_entropies(dist)
        assert entropies == {"action": 0.0, "resource": 0.0, "scope": 0.0, "condition": 0.0}

    def test_reproduces_p4_calib_scope_and_action_ambiguous_real_data(self):
        """Phase 2C-P4(§29)의 실측값 재현: read 14, summarize 5, write 1
        (n=20) -> action entropy ≈ 1.076 bits."""
        belief = {_interp("read"): 14 / 20, _interp("summarize"): 5 / 20,
                 _interp("write"): 1 / 20}
        dist = _distribution(belief)
        entropies = facet_entropies(dist)
        assert entropies["action"] == pytest.approx(1.076, abs=0.01)
        assert entropies["scope"] == pytest.approx(0.0)  # 이 fixture는 scope 전부 동일

    def test_only_the_varying_facet_has_nonzero_entropy(self):
        belief = {_interp("read", scope="/reports/2026-08/"): 0.5,
                 _interp("summarize", scope="/reports/2026-08/"): 0.5}
        dist = _distribution(belief)
        entropies = facet_entropies(dist)
        assert entropies["action"] == pytest.approx(1.0)
        assert entropies["scope"] == 0.0


def _make_principal_and_delegate(principal_llm, delegate_llm):
    return PrincipalAgent(llm=principal_llm), DelegateAgent(llm=delegate_llm)


class TestSampleAndGateStable:
    """V1 precedent: 모든 독립 draw가 같은 (action, scope)로 수렴."""

    def test_stable_case_gate_and_no_clarification(self):
        n = 3
        principal_llm = _FakeLLMClient([_delegation_text() for _ in range(n)])
        delegate_llm = _FakeLLMClient([_structured("read") for _ in range(n)])
        principal, delegate = _make_principal_and_delegate(principal_llm, delegate_llm)

        clarifier = SourceClarifyingPrincipal(principal=principal, delegate=delegate, n=n)
        result = clarifier.resolve(goal="Read the report.", context="")

        assert result.clarified is False
        assert result.final_interpretation == _interp("read")
        assert result.final_delegation  # a real (non-empty) delegation text
        assert result.post_distribution is None
        assert result.pre_distribution.entropy == 0.0

    def test_gate_directly_reports_stable(self):
        gate = SourceSemanticGate(entropy_threshold=0.8)
        dist = _distribution({_interp("read"): 1.0})
        decision = gate.decide(dist)
        assert decision.stable is True
        assert decision.unstable_facets == frozenset()

    def test_negative_control_designed_ambiguous_but_actually_stable(self):
        """calib_scope_ambiguous_action_fixed 정밀 재현: scope-ambiguous하게
        "설계"된 입력이라도 실제 output이 안정적이면 gate는 stable로 판단
        해야 한다 -- surface wording이 아니라 실제 entropy에 반응함을
        확인."""
        gate = SourceSemanticGate(entropy_threshold=0.8)
        dist = _distribution({_interp("export", scope="/reports/2026-09/"): 1.0})
        decision = gate.decide(dist)
        assert decision.stable is True


class TestSingleFacetDisagreement:
    """V2 precedent: action만 갈리고 scope는 일치."""

    def test_unstable_action_triggers_single_facet_clarification(self):
        n = 4
        # pre: action 2/2 split(H=1.0>0.8), scope 항상 동일
        principal_llm = _FakeLLMClient(
            [_delegation_text() for _ in range(n)]        # pre delegate() x4
            + [LLMResponse(text="summarize")]              # answer_clarification()
            + [_delegation_text() for _ in range(n)])       # post delegate() x4
        delegate_llm = _FakeLLMClient(
            [_structured("read"), _structured("read"),
            _structured("summarize"), _structured("summarize")]   # pre propose() x4
            + [_structured("summarize") for _ in range(n)])        # post propose() x4 (수렴)
        principal, delegate = _make_principal_and_delegate(principal_llm, delegate_llm)

        clarifier = SourceClarifyingPrincipal(principal=principal, delegate=delegate,
                                              n=n, entropy_threshold=0.8)
        result = clarifier.resolve(goal="Prepare the report.", context="finance team")

        assert result.clarified is True
        assert result.target_facet == "action"
        assert result.question is not None and "action" in result.question
        assert result.answer is not None and result.answer.answer == "summarize"
        assert result.post_distribution is not None
        assert result.post_distribution.entropy == 0.0
        assert result.final_interpretation == _interp("summarize")
        # answer_clarification()은 정확히 1회만 호출된다(결정론적 질문 -- 새 LLM 호출 없음).
        assert len(principal_llm.calls) == n + 1 + n

    def test_gate_flags_exactly_the_disagreeing_facet(self):
        gate = SourceSemanticGate(entropy_threshold=0.8)
        belief = {_interp("read", scope="/reports/2026-08/"): 0.5,
                 _interp("summarize", scope="/reports/2026-08/"): 0.5}
        decision = gate.decide(_distribution(belief))
        assert decision.unstable_facets == frozenset({"action"})
        assert decision.stable is False


class TestMultiFacetDisagreementSelectsOne:
    """V3 precedent: action과 scope 둘 다 갈려도, 이번 round는 정확히
    하나의 facet만 target으로 삼는다(기존 legacy IG 로직 재사용)."""

    def test_both_action_and_scope_vary_but_one_facet_is_targeted(self):
        n = 4
        principal_llm = _FakeLLMClient(
            [_delegation_text() for _ in range(n)]
            + [LLMResponse(text="summarize the August report")]
            + [_delegation_text() for _ in range(n)])
        delegate_llm = _FakeLLMClient(
            [_structured("write", scope="/reports/2026-08/"),
            _structured("read", scope="/reports/2026-08/"),
            _structured("summarize", scope="/reports/2026-09/"),
            _structured("summarize", scope="/reports/2026-08/")]
            + [_structured("summarize", scope="/reports/2026-08/") for _ in range(n)])
        principal, delegate = _make_principal_and_delegate(principal_llm, delegate_llm)

        clarifier = SourceClarifyingPrincipal(principal=principal, delegate=delegate, n=n)
        result = clarifier.resolve(goal="Handle the report.", context="")

        # gate 수준에서는 action/scope 둘 다 unstable로 잡혀야 한다.
        pre_decision = clarifier.gate.decide(result.pre_distribution)
        assert {"action", "scope"} <= pre_decision.unstable_facets
        # 하지만 이번 round가 실제로 target한 facet은 정확히 하나다.
        assert result.target_facet in ("action", "scope")
        assert result.clarified is True


class TestFinalDelegationIsRealNotSynthesized:
    def test_final_delegation_matches_one_of_the_actual_generated_texts(self):
        n = 3
        texts = ["Please read report A.", "Please read report B.", "Please read report C."]
        principal_llm = _FakeLLMClient([LLMResponse(text=t) for t in texts])
        delegate_llm = _FakeLLMClient([_structured("read") for _ in range(n)])
        principal, delegate = _make_principal_and_delegate(principal_llm, delegate_llm)

        clarifier = SourceClarifyingPrincipal(principal=principal, delegate=delegate, n=n)
        result = clarifier.resolve(goal="Read the report.", context="")

        assert result.final_delegation in texts

    def test_final_principal_delegation_is_a_real_PrincipalDelegation(self):
        """`final_principal_delegation`은 합성된 값이 아니라 실제로
        `principal.delegate()`가 만든 (delegation, response) 쌍이어야
        한다 -- runtime 통합에서 그대로 PrincipalDelegation 자리에
        넣을 수 있어야 하므로."""
        from dualflow.principal_agent import PrincipalDelegation

        n = 3
        principal_llm = _FakeLLMClient([_delegation_text("Please read it.") for _ in range(n)])
        delegate_llm = _FakeLLMClient([_structured("read") for _ in range(n)])
        principal, delegate = _make_principal_and_delegate(principal_llm, delegate_llm)

        clarifier = SourceClarifyingPrincipal(principal=principal, delegate=delegate, n=n)
        result = clarifier.resolve(goal="Read the report.", context="")

        assert isinstance(result.final_principal_delegation, PrincipalDelegation)
        assert result.final_principal_delegation.delegation == result.final_delegation
        assert result.final_principal_delegation.delegation == "Please read it."


class TestNoGroundTruthAPI:
    def test_resolve_signature_has_no_truth_or_label_parameter(self):
        import inspect
        forbidden = ("truth", "ground_truth", "label", "expected")
        sig = inspect.signature(SourceClarifyingPrincipal.resolve)
        for name in sig.parameters:
            assert not any(f in name.lower() for f in forbidden)
