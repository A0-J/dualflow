"""FrozenCandidateEvidenceHarness — v3 evidence-consumption contract.
Fully deterministic: no LLM client, no network, no OPENAI_API_KEY needed.

Locks in the consumption contract fixed before any main-experiment API run
(docs/experiments/agent_connected_eval.md, v3 main-experiment protocol):
  - stability gate first (existing entropy threshold, no new classifier)
  - eligibility gate (existing confirmed_facets provenance)
  - support gate (existing deterministic comparator, current candidates only)
  - facet-level resolution only -- never a synthesized historical
    Interpretation, never cross-facet amplification
  - explicit/stable current instructions are never overridden by
    conflicting historical evidence (F4 structurally prevented)
"""

from __future__ import annotations

from dualflow.agent_experience import AgentExperience
from dualflow.delegate_agent import CandidateDistribution
from dualflow.experience_decision import (
    AMBIGUOUS_NO_ELIGIBLE_EVIDENCE,
    AMBIGUOUS_NO_SUPPORT,
    AMBIGUOUS_RESOLVED_BY_EVIDENCE,
    STABLE_BASELINE,
    FrozenCandidateEvidenceHarness,
)
from dualflow.experience_evidence import EvidenceRelation, HistoricalEvidenceComparator
from dualflow.semantic import Interpretation

_SEPT = "/reports/2026-09/"
_AUG = "/reports/2026-08/"


def _interp(action: str, scope: str = _SEPT, resource: str = "file") -> Interpretation:
    return Interpretation(action, resource, scope, frozenset())


def _distribution(belief: dict[Interpretation, float], entropy_value: float) -> CandidateDistribution:
    top = max(belief, key=lambda i: belief[i])
    return CandidateDistribution(belief=belief, entropy=entropy_value, top=top,
                                 top_probability=belief[top], n_unique=len(belief),
                                 n_samples=20, responses=[])


def _experience(action: str, scope: str = _AUG,
               confirmed_facets: frozenset[str] = frozenset({"action"})) -> AgentExperience:
    interp = _interp(action, scope=scope)
    dummy = _distribution({interp: 1.0}, 0.0)
    return AgentExperience(
        principal_id="finance_lead_A", task_category="external_audit_report",
        delegation="Please prepare the August 2026 financial report.",
        clarification_question="Should I export or summarize?",
        principal_answer=f"Please {action} it.",
        confirmed_interpretation=interp,
        pre_distribution=dummy, post_distribution=dummy,
        confirmed_facets=confirmed_facets)


class _CountingComparator:
    """실제 `HistoricalEvidenceComparator`를 감싸서 `judge()` 호출 횟수를
    센다 — "이 stage에서는 comparator가 아예 호출되지 않아야 한다"를
    직접 확인하기 위한 spy."""

    def __init__(self):
        self._real = HistoricalEvidenceComparator()
        self.calls = 0

    def judge(self, **kwargs):
        self.calls += 1
        return self._real.judge(**kwargs)


class TestStabilityGate:
    """explicit/stable current distribution -> evidence never even
    consulted, regardless of what history says (F4 방지 지점)."""

    def test_stable_distribution_never_consults_evidence(self):
        stable = _distribution({_interp("export"): 1.0}, entropy_value=0.0)
        experience = _experience("summarize")  # conflicting historical value
        comparator = _CountingComparator()
        harness = FrozenCandidateEvidenceHarness(comparator=comparator, entropy_threshold=0.8)

        decision = harness.decide(distribution=stable, experience=experience, facet="action")

        assert decision.stage == STABLE_BASELINE
        assert decision.used_historical_evidence is False
        assert decision.final_value == "export"  # NOT overridden by historical "summarize"
        assert decision.resolved_value is None
        assert comparator.calls == 0  # evidence literally never touched

    def test_conflicting_history_never_overrides_explicit_current_instruction(self):
        """F4 (stale-history override)의 정확한 실패 재현 + 수정 확인:
        historical action이 summarize인데 현재는 명시적으로/안정적으로
        export를 가리키는 경우, 최종 결정은 반드시 export여야 한다."""
        stable_export = _distribution({_interp("export"): 0.95, _interp("summarize"): 0.05}, 0.286)
        experience = _experience("summarize")
        harness = FrozenCandidateEvidenceHarness(entropy_threshold=0.8)

        decision = harness.decide(distribution=stable_export, experience=experience, facet="action")

        assert decision.final_value == "export"
        assert decision.stage == STABLE_BASELINE


class TestEligibilityGate:
    """ambiguous하지만 해당 facet이 confirmed_facets에 없으면(F1) evidence를
    조회하지 않는다."""

    def test_facet_not_in_confirmed_facets_is_not_consulted(self):
        ambiguous = _distribution({_interp("summarize"): 0.6, _interp("export"): 0.4}, 0.971)
        experience = _experience("summarize", confirmed_facets=frozenset())  # nothing confirmed
        comparator = _CountingComparator()
        harness = FrozenCandidateEvidenceHarness(comparator=comparator, entropy_threshold=0.8)

        decision = harness.decide(distribution=ambiguous, experience=experience, facet="action")

        assert decision.stage == AMBIGUOUS_NO_ELIGIBLE_EVIDENCE
        assert decision.used_historical_evidence is False
        assert decision.final_value == decision.baseline_value  # unchanged
        assert comparator.calls == 0


class TestSupportGate:
    """eligible이지만 현재 candidate 중 historical 값과 일치하는 게 없으면
    (F2) evidence를 봤지만 적용하지 못한다."""

    def test_no_matching_candidate_falls_back_to_baseline(self):
        ambiguous = _distribution({_interp("export"): 0.6, _interp("read"): 0.4}, 0.971)
        experience = _experience("summarize")  # "summarize" isn't among current candidates at all
        harness = FrozenCandidateEvidenceHarness(entropy_threshold=0.8)

        decision = harness.decide(distribution=ambiguous, experience=experience, facet="action")

        assert decision.stage == AMBIGUOUS_NO_SUPPORT
        assert decision.used_historical_evidence is True  # evidence WAS consulted
        assert decision.resolved_value is None
        assert decision.final_value == decision.baseline_value


class TestResolutionByEvidence:
    """ambiguous + eligible + support 있음 -> facet만 resolve, 전체
    Interpretation을 historical 값으로 확정하지 않는다(H1이 측정하는
    성공 경로)."""

    def test_resolves_to_historically_confirmed_action(self):
        ambiguous = _distribution({_interp("summarize"): 0.3, _interp("export"): 0.7}, 0.881)
        experience = _experience("summarize")
        harness = FrozenCandidateEvidenceHarness(entropy_threshold=0.8)

        decision = harness.decide(distribution=ambiguous, experience=experience, facet="action")

        assert decision.stage == AMBIGUOUS_RESOLVED_BY_EVIDENCE
        assert decision.used_historical_evidence is True
        assert decision.resolved_value == "summarize"
        assert decision.final_value == "summarize"
        # baseline (majority) was export -- evidence flipped the decision
        assert decision.baseline_value == "export"

    def test_decision_interpretation_scope_comes_from_current_candidate_not_history(self):
        """F5 (cross-facet amplification) 방지 확인: historical scope는
        August인데, decision_interpretation의 scope는 반드시 현재
        (September) candidate 자신의 scope여야 한다 -- historical scope로
        절대 바뀌지 않는다."""
        current_summarize = _interp("summarize", scope=_SEPT)
        ambiguous = _distribution({current_summarize: 0.3, _interp("export", scope=_SEPT): 0.7}, 0.881)
        experience = _experience("summarize", scope=_AUG)  # historical scope is August
        harness = FrozenCandidateEvidenceHarness(entropy_threshold=0.8)

        decision = harness.decide(distribution=ambiguous, experience=experience, facet="action")

        assert decision.decision_interpretation is not None
        assert decision.decision_interpretation.scope == _SEPT  # current, NOT _AUG
        assert decision.decision_interpretation is current_summarize

    def test_multiple_current_interpretations_sharing_resolved_value_yield_no_single_decision(self):
        """같은 resolved action을 가진 서로 다른(다른 facet에서 갈리는)
        현재 Interpretation이 여러 개면, 임의로 하나를 골라 전체
        Interpretation을 확정하지 않는다 -- resolved_value는 나오되
        decision_interpretation은 None."""
        variant_a = Interpretation("summarize", "file", "/reports/2026-09/", frozenset())
        variant_b = Interpretation("summarize", "file", "/reports/2026-09/", frozenset({"urgent"}))
        ambiguous = _distribution({variant_a: 0.5, variant_b: 0.5}, 1.0)
        experience = _experience("summarize")
        harness = FrozenCandidateEvidenceHarness(entropy_threshold=0.8)

        decision = harness.decide(distribution=ambiguous, experience=experience, facet="action")

        assert decision.stage == AMBIGUOUS_RESOLVED_BY_EVIDENCE
        assert decision.resolved_value == "summarize"
        assert decision.decision_interpretation is None  # no arbitrary full-Interpretation pick


class TestRelationsAuditTrail:
    """`relations`는 comparator가 실제로 호출된 stage에서만 채워진다 --
    이걸로 "언제 evidence를 실제로 봤는지"를 감사할 수 있다."""

    def test_relations_empty_when_stable(self):
        stable = _distribution({_interp("export"): 1.0}, entropy_value=0.0)
        experience = _experience("summarize")
        harness = FrozenCandidateEvidenceHarness(entropy_threshold=0.8)

        decision = harness.decide(distribution=stable, experience=experience, facet="action")

        assert decision.relations == {}

    def test_relations_empty_when_not_eligible(self):
        ambiguous = _distribution({_interp("summarize"): 0.6, _interp("export"): 0.4}, 0.971)
        experience = _experience("summarize", confirmed_facets=frozenset())
        harness = FrozenCandidateEvidenceHarness(entropy_threshold=0.8)

        decision = harness.decide(distribution=ambiguous, experience=experience, facet="action")

        assert decision.relations == {}

    def test_relations_populated_for_every_distinct_candidate_value(self):
        ambiguous = _distribution({_interp("summarize"): 0.3, _interp("export"): 0.7}, 0.881)
        experience = _experience("summarize")
        harness = FrozenCandidateEvidenceHarness(entropy_threshold=0.8)

        decision = harness.decide(distribution=ambiguous, experience=experience, facet="action")

        assert decision.relations == {
            "summarize": EvidenceRelation.SUPPORT,
            "export": EvidenceRelation.CONFLICT,
        }


class TestNoNewClassifierOrWeighting:
    def test_entropy_threshold_matches_clarification_default(self):
        """새 threshold를 만들지 않는다 -- clarification.ClarifyingDelegate
        의 기본값(0.8)과 동일한 기본값을 쓴다."""
        harness = FrozenCandidateEvidenceHarness()
        assert harness.entropy_threshold == 0.8

    def test_only_one_facet_considered_per_decide_call(self):
        import inspect
        sig = inspect.signature(FrozenCandidateEvidenceHarness.decide)
        assert "facet" in sig.parameters
        assert sig.parameters["facet"].default == "action"


class TestStructuralIndependence:
    def test_module_does_not_reference_delegate_agent_or_generation(self):
        import dualflow.experience_decision as mod

        assert not hasattr(mod, "DelegateAgent")
        assert not hasattr(mod, "parse_structured_action")

    def test_module_does_not_reference_authority_or_budget(self):
        import dualflow.experience_decision as mod

        assert not hasattr(mod, "Budget")
        assert not hasattr(mod, "check_authority")
        assert not hasattr(mod, "AuthorityVerifierAgent")

    def test_module_does_not_reference_agent_delegation_runtime(self):
        import dualflow.experience_decision as mod

        assert not hasattr(mod, "AgentDelegationRuntime")
