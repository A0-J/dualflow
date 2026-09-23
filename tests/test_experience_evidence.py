"""HistoricalEvidenceComparator — v3 structured-facet comparator, revision
4 (provenance-gated). See src/dualflow/experience_evidence.py's module
docstring for the full revision history. Fully deterministic: no LLM
client, no network, no OPENAI_API_KEY needed for any of these tests.

Contract this file pins down (per the provenance-gap review):
  - a structured value existing in confirmed_interpretation is NOT enough
    to make that facet usable evidence -- the facet must also be in
    experience.confirmed_facets (i.e. the Principal actually clarified it)
  - a facet not in confirmed_facets returns IRRELEVANT, even if its values
    happen to match (no accidental SUPPORT-by-coincidence)
  - a historical scope mismatch never corrupts the action judgment when
    only "action" was confirmed -- scope querying itself returns
    IRRELEVANT, not CONFLICT, when scope was never confirmed
  - once a facet IS in confirmed_facets, comparison for it activates
    normally (SUPPORT/CONFLICT)
  - provenance for one facet never leaks into another facet's judgment
  - there is still no channel for prose/wording to enter the comparison
  - the module stays structurally independent of DelegateAgent, Authority/
    Budget, and AgentDelegationRuntime
"""

from __future__ import annotations

import inspect

import pytest

from dualflow.agent_experience import AgentExperience
from dualflow.delegate_agent import CandidateDistribution
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


def _dummy_distribution(top: Interpretation) -> CandidateDistribution:
    return CandidateDistribution(belief={top: 1.0}, entropy=0.0, top=top,
                                 top_probability=1.0, n_unique=1, n_samples=1, responses=[])


def _experience(confirmed_interpretation: Interpretation,
               confirmed_facets: frozenset[str] = frozenset()) -> AgentExperience:
    return AgentExperience(
        principal_id="finance_lead_A", task_category="external_audit_report",
        delegation="Please prepare the August 2026 financial report for the external audit.",
        clarification_question="Should I export it, or summarize it?",
        principal_answer="Please summarize it.",
        confirmed_interpretation=confirmed_interpretation,
        pre_distribution=_dummy_distribution(confirmed_interpretation),
        post_distribution=_dummy_distribution(confirmed_interpretation),
        confirmed_facets=confirmed_facets,
    )


class TestProvenanceGate:
    def test_action_confirmed_matching_action_supports(self):
        """action만 clarified/confirmed된 historical experience -- matching
        action -> SUPPORT."""
        comparator = HistoricalEvidenceComparator()
        historical = _experience(_interp("summarize", scope=_AUG_SCOPE),
                                 confirmed_facets=frozenset({"action"}))
        candidate = _interp("summarize", scope=_SEPT_SCOPE)

        result = comparator.judge(candidate=candidate, historical=historical, facet="action")

        assert result.relation == EvidenceRelation.SUPPORT

    def test_action_confirmed_mismatched_action_conflicts(self):
        comparator = HistoricalEvidenceComparator()
        historical = _experience(_interp("export"), confirmed_facets=frozenset({"action"}))
        candidate = _interp("summarize")

        result = comparator.judge(candidate=candidate, historical=historical, facet="action")

        assert result.relation == EvidenceRelation.CONFLICT

    def test_unconfirmed_resource_scope_condition_are_irrelevant(self):
        """action만 confirmed -> resource/scope/condition을 물으면 값이
        무엇이든 IRRELEVANT."""
        comparator = HistoricalEvidenceComparator()
        historical = _experience(_interp("summarize"), confirmed_facets=frozenset({"action"}))
        candidate = _interp("summarize")

        for facet in ("resource", "scope", "condition"):
            result = comparator.judge(candidate=candidate, historical=historical, facet=facet)
            assert result.relation == EvidenceRelation.IRRELEVANT, facet

    def test_scope_mismatch_does_not_corrupt_action_when_only_action_confirmed(self):
        """정확히 이번 지시의 핵심 케이스: historical scope=2026-08,
        confirmed_facets={"action"}. current scope=2026-09여도 action은
        여전히 SUPPORT — scope CONFLICT가 새어 들어오지 않는다."""
        comparator = HistoricalEvidenceComparator()
        historical = _experience(_interp("summarize", scope=_AUG_SCOPE),
                                 confirmed_facets=frozenset({"action"}))
        candidate = _interp("summarize", scope=_SEPT_SCOPE)

        action_result = comparator.judge(candidate=candidate, historical=historical, facet="action")
        scope_result = comparator.judge(candidate=candidate, historical=historical, facet="scope")

        assert action_result.relation == EvidenceRelation.SUPPORT
        assert scope_result.relation == EvidenceRelation.IRRELEVANT  # NOT CONFLICT

    def test_scope_activates_only_when_actually_confirmed(self):
        """scope가 실제로 Principal-confirmed된 experience라면(confirmed_
        facets에 "scope" 포함), scope 비교가 그때는 활성화된다."""
        comparator = HistoricalEvidenceComparator()
        historical = _experience(_interp("summarize", scope=_AUG_SCOPE),
                                 confirmed_facets=frozenset({"action", "scope"}))
        matching_scope_candidate = _interp("summarize", scope=_AUG_SCOPE)
        mismatched_scope_candidate = _interp("summarize", scope=_SEPT_SCOPE)

        supports = comparator.judge(candidate=matching_scope_candidate, historical=historical,
                                    facet="scope")
        conflicts = comparator.judge(candidate=mismatched_scope_candidate, historical=historical,
                                     facet="scope")

        assert supports.relation == EvidenceRelation.SUPPORT
        assert conflicts.relation == EvidenceRelation.CONFLICT

    def test_empty_values_do_not_become_support_by_coincidence(self):
        """empty string / empty condition이 단순히 존재한다는 이유만으로
        SUPPORT evidence가 되면 안 된다 -- confirmed_facets에 없으면
        candidate와 historical의 값이 우연히 똑같이 비어 있어도
        IRRELEVANT여야 한다."""
        comparator = HistoricalEvidenceComparator()
        historical = _experience(_interp("summarize", condition=frozenset()),
                                 confirmed_facets=frozenset({"action"}))  # NOT "condition"
        candidate = _interp("summarize", condition=frozenset())  # coincidentally also empty

        result = comparator.judge(candidate=candidate, historical=historical, facet="condition")

        assert result.relation == EvidenceRelation.IRRELEVANT

    def test_no_confirmed_facets_at_all_makes_every_facet_irrelevant(self):
        """provenance가 아예 없으면(기본값 frozenset()) 모든 facet이
        IRRELEVANT -- 이게 기존 직접-생성 코드(agent_smoke.py/
        experience_transfer.py)가 confirmed_facets를 넘기지 않을 때의
        안전한 기본 동작이다."""
        comparator = HistoricalEvidenceComparator()
        historical = _experience(_interp("summarize"))  # confirmed_facets defaults to frozenset()
        candidate = _interp("summarize")

        for facet in FACETS:
            result = comparator.judge(candidate=candidate, historical=historical, facet=facet)
            assert result.relation == EvidenceRelation.IRRELEVANT, facet

    def test_provenance_does_not_leak_across_facets(self):
        """confirmed_facets={"action", "scope"}인데 resource/condition은
        여전히 IRRELEVANT여야 한다 -- 다른 facet이 confirmed라고 해서
        전파되지 않는다."""
        comparator = HistoricalEvidenceComparator()
        historical = _experience(_interp("summarize"),
                                 confirmed_facets=frozenset({"action", "scope"}))
        candidate = _interp("summarize")

        assert comparator.judge(candidate=candidate, historical=historical,
                                facet="resource").relation == EvidenceRelation.IRRELEVANT
        assert comparator.judge(candidate=candidate, historical=historical,
                                facet="condition").relation == EvidenceRelation.IRRELEVANT

    def test_candidate_and_historical_are_returned_unchanged(self):
        comparator = HistoricalEvidenceComparator()
        candidate = _interp("summarize")
        historical = _experience(_interp("summarize"), confirmed_facets=frozenset({"action"}))

        result = comparator.judge(candidate=candidate, historical=historical)

        assert result.candidate is candidate
        assert result.historical is historical


class TestCompareFacetsIsPureAndProvenanceAgnostic:
    """compare_facets()는 provenance를 전혀 보지 않는 순수 값 비교다--
    게이트는 judge()의 책임이다."""

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
        candidate = _interp("summarize", resource="file", scope=_SEPT_SCOPE)
        historical = _interp("export", resource="file", scope=_AUG_SCOPE)

        relations = compare_facets(candidate, historical)

        assert relations["action"] == EvidenceRelation.CONFLICT
        assert relations["resource"] == EvidenceRelation.SUPPORT
        assert relations["scope"] == EvidenceRelation.CONFLICT

    def test_only_support_and_conflict_are_ever_produced(self):
        relations = compare_facets(_interp("summarize"), _interp("export"))
        assert set(relations.values()) <= {EvidenceRelation.SUPPORT, EvidenceRelation.CONFLICT}


class TestNoWordingChannel:
    def test_judge_signature_has_no_text_or_wording_parameter(self):
        forbidden = ("text", "evidence_text", "delegation", "wording", "prompt",
                    "instructions", "context", "principal_answer")
        sig = inspect.signature(HistoricalEvidenceComparator.judge)
        for name in sig.parameters:
            lowered = name.lower()
            assert not any(f in lowered for f in forbidden), (
                f"judge() has a suspicious free-text parameter: {name}")

    def test_judge_takes_only_candidate_historical_experience_and_facet(self):
        sig = inspect.signature(HistoricalEvidenceComparator.judge)
        params = set(sig.parameters) - {"self"}
        assert params == {"candidate", "historical", "facet"}

    def test_unknown_facet_is_rejected(self):
        comparator = HistoricalEvidenceComparator()
        historical = _experience(_interp("summarize"), confirmed_facets=frozenset({"action"}))
        with pytest.raises(ValueError):
            comparator.judge(candidate=_interp("summarize"), historical=historical,
                             facet="not_a_real_facet")

    def test_all_declared_facets_are_queryable(self):
        comparator = HistoricalEvidenceComparator()
        candidate = _interp("summarize")
        historical = _experience(_interp("summarize"), confirmed_facets=frozenset(FACETS))
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
        import dualflow.experience_evidence as mod

        assert not hasattr(mod, "LLMClient")
        assert not hasattr(mod, "LLMResponse")

    def test_v2_delegate_header_is_not_reachable_from_this_module(self):
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


class TestAgentExperienceConfirmedFacetsDefault:
    """agent_experience.py 쪽 계약: 직접 생성(agent_smoke.py의
    EXAMPLE_PRIOR_EXPERIENCE, experience_transfer.py의 make_experience())
    은 confirmed_facets를 넘기지 않으므로 기본값 frozenset()이 되어야
    한다 -- 이 기본값이 안전한 이유는 위 test_no_confirmed_facets_at_
    all_makes_every_facet_irrelevant가 보여준다."""

    def test_default_confirmed_facets_is_empty_frozenset(self):
        experience = _experience(_interp("summarize"))  # no confirmed_facets kwarg
        assert experience.confirmed_facets == frozenset()
