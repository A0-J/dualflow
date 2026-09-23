"""FrozenCandidateEvidenceHarness -- v3 evidence-consumption contract.
Fully deterministic: no LLM client, no network, no OPENAI_API_KEY needed.

Locks in the consumption contract fixed before any main-experiment API run
(docs/experiments/agent_connected_eval.md, v3 main-experiment protocol),
AS REVISED by Option B (conservative abstention -- see §24 Follow-up,
"Option B: conservative abstention", and REVISION 2 in
src/dualflow/experience_decision.py's module docstring):
  - stability gate first (existing entropy threshold, no new classifier)
  - eligibility gate (existing confirmed_facets provenance)
  - support gate (existing deterministic comparator, current candidates only)
  - NO automatic resolution -- even when support exists, the harness never
    applies it; it only surfaces a `historical_evidence_requires_
    clarification` signal (relations recorded for audit, resolved_value
    kept for reference) and leaves final_value at baseline. Facet-level-
    only scoping (never a synthesized historical Interpretation, never
    cross-facet amplification) still holds, trivially, because nothing is
    ever applied automatically at all.
  - PRECISE scope of the stability guarantee, now stronger than before: a
    distribution that is already stable is never overridden (unchanged),
    AND an unstable/ambiguous distribution is *also* never automatically
    overridden any more -- see `TestH2StructuralCounterexample` below,
    which re-runs the same deterministic counterexample that originally
    exposed F4 (stale-history override) under the REVISION 1 contract, and
    confirms it no longer overrides anything under this (REVISION 2)
    contract. REVISION 1's automatic-resolution behavior is preserved only
    as historical record (git history / §24's already-run 200-call main
    experiment results) -- `decide()` no longer produces it.
"""

from __future__ import annotations

from dualflow.agent_experience import AgentExperience
from dualflow.delegate_agent import CandidateDistribution
from dualflow.experience_decision import (
    AMBIGUOUS_NO_ELIGIBLE_EVIDENCE,
    AMBIGUOUS_NO_SUPPORT,
    AMBIGUOUS_REQUIRES_CLARIFICATION,
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

    def test_conflicting_history_never_overrides_a_stable_current_distribution(self):
        """정확한 범위: 이 테스트가 보장하는 건 "current distribution이
        이미 stable하면(entropy<=threshold) conflicting history가 override
        못 한다"이지, "explicit current instruction은 항상 override 못
        한다"가 아니다 -- 후자는 entropy가 threshold를 넘으면 성립하지
        않는다(`TestH2StructuralCounterexample` 참고, 실제로 재현됨)."""
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


class TestAmbiguousSupportRequiresClarification:
    """ambiguous + eligible + support 있음 -> Option B(conservative
    abstention): relation은 감사용으로 기록되고 resolved_value는 참고로
    남지만, harness는 절대 자동으로 final_value를 historical 값으로 바꾸지
    않는다. 전체 Interpretation을 historical 값으로 확정하는 일도 당연히
    없다 -- 애초에 아무것도 자동 적용되지 않으므로."""

    def test_support_is_recorded_but_not_applied(self):
        ambiguous = _distribution({_interp("summarize"): 0.3, _interp("export"): 0.7}, 0.881)
        experience = _experience("summarize")
        harness = FrozenCandidateEvidenceHarness(entropy_threshold=0.8)

        decision = harness.decide(distribution=ambiguous, experience=experience, facet="action")

        assert decision.stage == AMBIGUOUS_REQUIRES_CLARIFICATION
        assert decision.used_historical_evidence is True  # evidence WAS consulted
        assert decision.resolved_value == "summarize"     # recorded for reference/audit
        assert decision.final_value == "export"            # NOT applied -- baseline preserved
        assert decision.baseline_value == "export"

    def test_decision_interpretation_is_never_synthesized_from_history(self):
        """F5 (cross-facet amplification) 방지 확인: historical scope는
        August인데, harness는 애초에 어떤 decision_interpretation도 자동
        생성하지 않는다 -- scope가 historical 값으로 바뀔 여지 자체가
        없다. final_value(및 실제로 쓰이는 현재 candidate)는 항상 현재
        (September) candidate 자신의 scope를 유지한다."""
        current_summarize = _interp("summarize", scope=_SEPT)
        ambiguous = _distribution({current_summarize: 0.3, _interp("export", scope=_SEPT): 0.7}, 0.881)
        experience = _experience("summarize", scope=_AUG)  # historical scope is August
        harness = FrozenCandidateEvidenceHarness(entropy_threshold=0.8)

        decision = harness.decide(distribution=ambiguous, experience=experience, facet="action")

        assert decision.decision_interpretation is None  # no automatic Interpretation at all
        assert decision.final_value == "export"            # baseline (current) preserved

    def test_multiple_current_interpretations_sharing_supported_value_still_no_auto_decision(self):
        """같은 supported action을 가진 서로 다른(다른 facet에서 갈리는)
        현재 Interpretation이 여러 개여도 결론은 동일하다 -- 이 harness는
        어느 경우든 자동으로 하나를 고르지 않는다."""
        variant_a = Interpretation("summarize", "file", "/reports/2026-09/", frozenset())
        variant_b = Interpretation("summarize", "file", "/reports/2026-09/", frozenset({"urgent"}))
        ambiguous = _distribution({variant_a: 0.5, variant_b: 0.5}, 1.0)
        experience = _experience("summarize")
        harness = FrozenCandidateEvidenceHarness(entropy_threshold=0.8)

        decision = harness.decide(distribution=ambiguous, experience=experience, facet="action")

        assert decision.stage == AMBIGUOUS_REQUIRES_CLARIFICATION
        assert decision.resolved_value == "summarize"
        assert decision.decision_interpretation is None
        assert decision.final_value == decision.baseline_value


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


class TestH2StructuralCounterexample:
    """B7d-v3 main experiment (docs/experiments/agent_connected_eval.md
    SS24) 결과 이후 발견된 구조적 문제를 API 호출 없이 deterministic하게
    고정한다. 5/5 explicit-export preservation은 그 배치에서 entropy가
    매번 0.000이었기 때문이었다 -- 진짜 질문은 "entropy가 threshold를
    넘는 explicit-export sampling이 real API에서 실제로 나오면 어떻게
    되는가"였다.

    REVISION 1(원래 automatic-resolution contract)에서는 이 counterexample
    (export=.55/summarize=.45, entropy≈.993 > threshold 0.8)이 실제로
    harness에 의해 "summarize"로 override됐다 -- F4(stale-history
    override)가 현재 contract상 도달 가능함을 증명한 재현이었다. 이게
    바로 Option B(conservative abstention, REVISION 2) 재설계의 직접적인
    동기다.

    이 클래스는 이제 같은 counterexample을 REVISION 2 contract에 대해
    다시 고정한다: automatic resolution이 제거됐으므로, 같은 입력에서도
    더 이상 override가 일어나지 않는다는 것, 즉 F4가 이 harness를 통해서는
    더 이상 도달 가능하지 않다는 것을 확인한다. relations는 여전히
    계산/기록되므로(감사 가능성은 유지) "evidence를 못 봤다"가 아니라
    "봤지만 적용하지 않았다"임을 함께 확인한다."""

    def test_unstable_explicit_export_sampling_is_no_longer_overridden(self):
        """REVISION 1에서 F4를 재현했던 것과 정확히 같은 입력. REVISION 2
        에서는 relations가 여전히 CONFLICT/SUPPORT로 계산되지만
        final_value는 baseline("export")에서 바뀌지 않는다."""
        export = _interp("export")
        summarize = _interp("summarize")
        belief = {export: 0.55, summarize: 0.45}
        from dualflow.semantic import entropy as compute_entropy
        h = compute_entropy(belief)
        assert h > 0.8  # 이게 바로 "unstable sampling"의 정의다

        unstable_explicit_export = _distribution(belief, entropy_value=h)
        experience = _experience("summarize")  # conflicting historical value
        harness = FrozenCandidateEvidenceHarness(entropy_threshold=0.8)

        decision = harness.decide(distribution=unstable_explicit_export,
                                  experience=experience, facet="action")

        assert decision.stage == AMBIGUOUS_REQUIRES_CLARIFICATION
        assert decision.baseline_value == "export"   # majority/plurality current decision
        assert decision.final_value == "export"       # NOT overridden -- F4 blocked
        assert decision.used_historical_evidence is True  # evidence WAS consulted
        assert decision.resolved_value == "summarize"  # recorded for reference only
        assert decision.relations == {
            "export": EvidenceRelation.CONFLICT,
            "summarize": EvidenceRelation.SUPPORT,
        }

    def test_the_guarantee_now_covers_ambiguous_cases_too_not_just_stable_ones(self):
        """REVISION 1에서는 "harness는 explicitness가 아니라 stability만
        본다"가 정확한 진술이었다 -- stable/unstable 여부에 따라 결과가
        갈렸다. REVISION 2에서는 stable이든 unstable이든 이 harness가
        자동으로 override하는 경우가 아예 없으므로, 두 경우 모두
        final_value가 보존된다."""
        experience = _experience("summarize")
        harness = FrozenCandidateEvidenceHarness(entropy_threshold=0.8)

        stable = _distribution({_interp("export"): 0.95, _interp("summarize"): 0.05}, 0.286)
        unstable = _distribution({_interp("export"): 0.55, _interp("summarize"): 0.45}, 0.993)

        stable_decision = harness.decide(distribution=stable, experience=experience, facet="action")
        unstable_decision = harness.decide(distribution=unstable, experience=experience, facet="action")

        assert stable_decision.final_value == "export"    # preserved (evidence not consulted)
        assert unstable_decision.final_value == "export"  # ALSO preserved (evidence consulted,
                                                            # relation recorded, not applied)
        assert stable_decision.stage == STABLE_BASELINE
        assert unstable_decision.stage == AMBIGUOUS_REQUIRES_CLARIFICATION
        # 두 stage는 다르지만(evidence를 봤는지 여부), 둘 다 automatic
        # override는 발생하지 않는다는 결론은 동일하다.
