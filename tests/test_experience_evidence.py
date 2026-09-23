"""HistoricalEvidenceComparator — v3 structured-facet comparator (revision
3, see src/dualflow/experience_evidence.py's module docstring for why the
natural-language version was replaced). Fully deterministic: no LLM
client, no network, no OPENAI_API_KEY needed for any of these tests.

Contract this file pins down (per the §23 audit's design review):
  - historical action == candidate action -> action SUPPORT
  - historical action != candidate action -> action CONFLICT
  - a historical/candidate SCOPE mismatch never corrupts the ACTION
    judgment (cross-facet contamination is the exact failure the old
    natural-language version had)
  - there is no channel left for prose/wording to enter the comparison
    at all (judge()'s signature takes only Interpretation objects)
  - the reused v2 Delegate-facing header is nowhere in this module
  - the module stays structurally independent of DelegateAgent, Authority/
    Budget, and AgentDelegationRuntime
"""

from __future__ import annotations

import inspect

import pytest

from dualflow.experience_evidence import (
    FACETS,
    EvidenceJudgment,
    EvidenceRelation,
    HistoricalEvidenceComparator,
    compare_facets,
)
from dualflow.semantic import Interpretation

_SEPT_SCOPE = "/reports/2026-09/"
_AUG_SCOPE = "/reports/2026-08/"


def _interp(action: str, scope: str = _SEPT_SCOPE, resource: str = "file",
           condition: frozenset = frozenset()) -> Interpretation:
    return Interpretation(action, resource, scope, condition)


class TestActionFacetContract:
    def test_matching_action_supports(self):
        """historical action=summarize, current action=summarize -> action SUPPORT."""
        comparator = HistoricalEvidenceComparator()
        candidate = _interp("summarize")
        historical = _interp("summarize", scope=_SEPT_SCOPE)

        result = comparator.judge(candidate=candidate, historical=historical, facet="action")

        assert result.relation == EvidenceRelation.SUPPORT

    def test_mismatched_action_conflicts(self):
        """historical action=export, current action=summarize -> action CONFLICT."""
        comparator = HistoricalEvidenceComparator()
        candidate = _interp("summarize")
        historical = _interp("export")

        result = comparator.judge(candidate=candidate, historical=historical, facet="action")

        assert result.relation == EvidenceRelation.CONFLICT

    def test_scope_mismatch_does_not_corrupt_action_support(self):
        """The exact §23 failure mode: historical action=summarize with
        historical scope=2026-08, current action=summarize with current
        scope=2026-09 -- the scope difference must not turn the action
        judgment into CONFLICT."""
        comparator = HistoricalEvidenceComparator()
        candidate = _interp("summarize", scope=_SEPT_SCOPE)
        historical = _interp("summarize", scope=_AUG_SCOPE)

        result = comparator.judge(candidate=candidate, historical=historical, facet="action")

        assert result.relation == EvidenceRelation.SUPPORT

    def test_scope_mismatch_is_still_visible_on_the_scope_facet(self):
        """The independence cuts both ways: asking about the scope facet
        directly still correctly reports the mismatch -- facets are
        independent, not "action always wins"."""
        comparator = HistoricalEvidenceComparator()
        candidate = _interp("summarize", scope=_SEPT_SCOPE)
        historical = _interp("summarize", scope=_AUG_SCOPE)

        result = comparator.judge(candidate=candidate, historical=historical, facet="scope")

        assert result.relation == EvidenceRelation.CONFLICT

    def test_candidate_and_historical_are_returned_unchanged(self):
        comparator = HistoricalEvidenceComparator()
        candidate = _interp("summarize")
        historical = _interp("summarize")

        result = comparator.judge(candidate=candidate, historical=historical)

        assert result.candidate is candidate
        assert result.historical is historical


class TestCompareFacetsIndependence:
    def test_all_four_facets_compared_independently(self):
        candidate = _interp("summarize", resource="file", scope=_SEPT_SCOPE,
                            condition=frozenset({"approved"}))
        historical = _interp("export", resource="db", scope=_AUG_SCOPE,
                             condition=frozenset())

        relations = compare_facets(candidate, historical)

        assert relations == {
            "action": EvidenceRelation.CONFLICT,
            "resource": EvidenceRelation.CONFLICT,
            "scope": EvidenceRelation.CONFLICT,
            "condition": EvidenceRelation.CONFLICT,
        }

    def test_matching_facets_support_independently_of_mismatching_ones(self):
        """resource/condition matching while action/scope differ -- each
        facet's relation must reflect only its own match, nothing else."""
        candidate = _interp("summarize", resource="file", scope=_SEPT_SCOPE)
        historical = _interp("export", resource="file", scope=_AUG_SCOPE)

        relations = compare_facets(candidate, historical)

        assert relations["action"] == EvidenceRelation.CONFLICT
        assert relations["resource"] == EvidenceRelation.SUPPORT
        assert relations["scope"] == EvidenceRelation.CONFLICT

    def test_only_support_and_conflict_are_ever_produced(self):
        """IRRELEVANT/UNCERTAIN stay valid EvidenceRelation values for
        interface stability, but this deterministic comparator never
        produces them -- every Interpretation field is always populated."""
        candidate = _interp("summarize")
        historical = _interp("export")

        relations = compare_facets(candidate, historical)

        assert set(relations.values()) <= {EvidenceRelation.SUPPORT, EvidenceRelation.CONFLICT}


class TestNoWordingChannel:
    """The exact fix for §23's finding: there is no parameter left through
    which natural-language text (an episode's wording, a paraphrase, the
    reused v2 header) could enter this comparison at all."""

    def test_judge_signature_has_no_text_or_wording_parameter(self):
        forbidden = ("text", "evidence_text", "delegation", "wording", "prompt",
                    "instructions", "context", "principal_answer")
        sig = inspect.signature(HistoricalEvidenceComparator.judge)
        for name in sig.parameters:
            lowered = name.lower()
            assert not any(f in lowered for f in forbidden), (
                f"judge() has a suspicious free-text parameter: {name}")

    def test_judge_takes_only_interpretation_objects_and_a_facet_name(self):
        sig = inspect.signature(HistoricalEvidenceComparator.judge)
        params = set(sig.parameters) - {"self"}
        assert params == {"candidate", "historical", "facet"}

    def test_unknown_facet_is_rejected(self):
        comparator = HistoricalEvidenceComparator()
        with pytest.raises(ValueError):
            comparator.judge(candidate=_interp("summarize"), historical=_interp("summarize"),
                             facet="not_a_real_facet")

    def test_all_declared_facets_are_queryable(self):
        comparator = HistoricalEvidenceComparator()
        candidate = _interp("summarize")
        historical = _interp("summarize")
        for facet in FACETS:
            result = comparator.judge(candidate=candidate, historical=historical, facet=facet)
            assert isinstance(result, EvidenceJudgment)
            assert result.facet == facet


class TestStructuralIndependence:
    def test_module_does_not_reference_delegate_agent_or_generation(self):
        import dualflow.experience_evidence as mod

        assert not hasattr(mod, "DelegateAgent")
        assert not hasattr(mod, "parse_structured_action")

    def test_module_does_not_reference_authority_or_budget(self):
        import dualflow.experience_evidence as mod

        assert not hasattr(mod, "Budget")
        assert not hasattr(mod, "check_authority")
        assert not hasattr(mod, "AuthorityVerifierAgent")
        assert not hasattr(mod, "AuthorityVerdict")

    def test_module_does_not_reference_agent_delegation_runtime(self):
        import dualflow.experience_evidence as mod

        assert not hasattr(mod, "AgentDelegationRuntime")

    def test_module_no_longer_references_any_llm_client_or_response(self):
        """Revision 3 removed the LLM call entirely -- there should be no
        LLMClient/LLMResponse dependency left at all."""
        import dualflow.experience_evidence as mod

        assert not hasattr(mod, "LLMClient")
        assert not hasattr(mod, "LLMResponse")

    def test_v2_delegate_header_is_not_reachable_from_this_module(self):
        """The exact §23 contaminant -- render_experience_block_v2's
        Delegate-facing header -- must not be importable from here."""
        import dualflow.experience_evidence as mod

        assert not hasattr(mod, "render_experience_block_v2")
        assert not hasattr(mod, "render_experience_block_v2_neutral")
        assert not hasattr(mod, "_EXPERIENCE_HEADER_V2")

    def test_judge_signature_has_no_truth_or_ground_truth_parameter(self):
        forbidden = ("truth", "ground_truth", "label", "expected_action", "verdict")
        sig = inspect.signature(HistoricalEvidenceComparator.judge)
        for name in sig.parameters:
            lowered = name.lower()
            assert not any(f in lowered for f in forbidden), (
                f"judge() has a suspicious parameter: {name}")
