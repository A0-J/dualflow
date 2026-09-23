"""
V3 FROZEN-CANDIDATE EVIDENCE-CONSUMPTION HARNESS (opt-in, offline
experimental harness — NOT wired into `AgentDelegationRuntime` yet).

The v3 contract smoke test (docs/experiments/agent_connected_eval.md §23
follow-up 5's closing validation) confirmed the provenance chain (IG target
selection → singleton clarifying question → post-clarification resolution
→ `confirmed_facets` → `HistoricalEvidenceComparator`) holds against real
API-generated data. `HistoricalEvidenceComparator` itself only classifies
a relation (SUPPORT/CONFLICT/IRRELEVANT/UNCERTAIN) — it does not decide
anything. This module is the next, deliberately narrow piece: **how a
relation gets consumed to produce an actual semantic decision**, without
re-injecting historical text into any generation prompt (that would undo
everything B7d.6/§22 established) and without inventing a new
ambiguity/explicitness classifier (the existing entropy-threshold gate
already used by `clarification.ClarifyingDelegate` is reused unchanged).

Consumption contract (fixed before any main-experiment API run):

  1. **Stability gate first, using the existing criterion.** If the
     current candidate distribution's entropy is already at or below
     `entropy_threshold` (the same threshold `ClarifyingDelegate` uses),
     the current facet is already stable/explicit — historical evidence
     is never even consulted, and the baseline decision (`distribution.
     top`) is returned unchanged. This is what structurally prevents
     `stale-history override` (a historical value overriding an explicit
     current instruction) — it's a gate on *consulting* evidence, not a
     rule applied after generating a decision.
  2. **Eligibility gate, using existing provenance.** If the facet under
     consideration is not in `experience.confirmed_facets`, there is
     nothing safe to consult — evidence is not looked at, decision stays
     baseline (`evidence unavailable`).
  3. **Support gate, using the existing deterministic comparator.** Among
     the *already-generated* current candidates (never re-sampled, never
     re-prompted), the distinct values of the facet are compared against
     the historical experience via `HistoricalEvidenceComparator`. If none
     is `SUPPORT`, evidence is consulted but not applicable — decision
     stays baseline (`support absent`).
  4. **Facet-level resolution only.** If exactly one facet value is
     `SUPPORT`ed, that is the resolved value for *that facet only*. The
     decision interpretation, if uniquely determinable, is one of the
     *current* candidates that already carries that value — never a
     synthesized `Interpretation` built from historical resource/scope/
     condition values. If more than one *distinct current Interpretation*
     shares the resolved facet value (only possible if other facets also
     vary, which the current pilot scenario does not exercise), no single
     point decision is picked — the facet-level resolution is still
     reported, but `decision_interpretation` is `None`.

이 모듈이 하지 않는 것:
  - Delegate candidate generation에 관여하는 것 — `distribution`은 항상
    호출자가 이미(history 없이) 만들어서 넘긴다.
  - Authority/Budget/AuthorityVerifierAgent에 관여하는 것.
  - `AgentDelegationRuntime.run()`에 연결되는 것 — 아직 연결되지 않는다.
  - 새 explicitness classifier나 새 confidence 가중치를 만드는 것 —
    entropy threshold 게이트는 `clarification.py`가 이미 쓰는 것과 정확히
    같은 기준이다.
  - 여러 facet을 동시에 resolve하는 것 — `decide()`는 facet 하나(기본값
    `"action"`)만 다룬다.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .agent_experience import AgentExperience
from .delegate_agent import CandidateDistribution
from .experience_evidence import EvidenceRelation, HistoricalEvidenceComparator
from .semantic import Interpretation

# 각 stage 이름은 그대로 failure-taxonomy 매핑에 쓰인다(문서 참고):
#   STABLE_BASELINE                -> 애초에 evidence를 볼 필요 없음 (F4 방지 지점)
#   AMBIGUOUS_NO_ELIGIBLE_EVIDENCE -> F1 (evidence unavailable)
#   AMBIGUOUS_NO_SUPPORT           -> F2 (support absent)
#   AMBIGUOUS_RESOLVED_BY_EVIDENCE -> 성공 경로 (H1이 측정하는 것)
STABLE_BASELINE = "stable_baseline"
AMBIGUOUS_NO_ELIGIBLE_EVIDENCE = "ambiguous_no_eligible_evidence"
AMBIGUOUS_NO_SUPPORT = "ambiguous_no_support"
AMBIGUOUS_RESOLVED_BY_EVIDENCE = "ambiguous_resolved_by_evidence"


@dataclass(frozen=True)
class FacetDecision:
    """`FrozenCandidateEvidenceHarness.decide()` 호출 1회의 결과.

    `final_value`가 실제로 보고해야 할 "이번 decision의 facet 값"이다 —
    baseline이든 evidence-resolved든 상관없이 항상 채워진다. `resolved_
    value`/`decision_interpretation`은 evidence가 실제로 적용됐을 때만
    `None`이 아니다."""

    facet: str
    stage: str
    used_historical_evidence: bool

    baseline_value: object          # distribution.top의 해당 facet 값 (개입 전)
    resolved_value: object | None   # evidence로 확인된 값 (stage가 RESOLVED일 때만)
    decision_interpretation: Interpretation | None  # 유일하게 결정 가능할 때만

    final_value: object             # 이번 decision에서 실제로 쓸 값

    relations: dict = None          # 감사/기록용 — {facet 값: EvidenceRelation}.
                                    # comparator가 실제로 호출된 stage(support
                                    # gate 이후)에서만 채워진다; STABLE_BASELINE/
                                    # AMBIGUOUS_NO_ELIGIBLE_EVIDENCE에서는 빈 dict
                                    # (comparator를 아예 부르지 않았다는 뜻).

    def __post_init__(self):
        if self.relations is None:
            object.__setattr__(self, "relations", {})


class FrozenCandidateEvidenceHarness:
    """이미 생성된(history 없이 sampling된) `CandidateDistribution`에
    historical evidence를 어떻게 적용할지 결정한다. candidate를 다시
    생성하지 않는다 — `DelegateAgent`를 참조하지도 않는다."""

    def __init__(self, comparator: HistoricalEvidenceComparator | None = None,
                entropy_threshold: float = 0.8):
        self.comparator = comparator or HistoricalEvidenceComparator()
        self.entropy_threshold = entropy_threshold

    def decide(self, *, distribution: CandidateDistribution, experience: AgentExperience,
              facet: str = "action") -> FacetDecision:
        baseline_value = getattr(distribution.top, facet)

        # 1. Stability gate — 기존 entropy threshold 그대로 재사용.
        if distribution.entropy <= self.entropy_threshold:
            return FacetDecision(facet, STABLE_BASELINE, False,
                                 baseline_value, None, distribution.top, baseline_value)

        # 2. Eligibility gate — 기존 provenance(confirmed_facets)만 본다.
        if facet not in experience.confirmed_facets:
            return FacetDecision(facet, AMBIGUOUS_NO_ELIGIBLE_EVIDENCE, False,
                                 baseline_value, None, None, baseline_value)

        # 3. Support gate — 이미 있는 candidate들만 비교한다(재생성 없음).
        by_value: dict[object, list[Interpretation]] = defaultdict(list)
        for interp in distribution.belief:
            by_value[getattr(interp, facet)].append(interp)

        supported_values = []
        relations: dict = {}
        for value, interps in by_value.items():
            judgment = self.comparator.judge(candidate=interps[0], historical=experience, facet=facet)
            relations[value] = judgment.relation
            if judgment.relation == EvidenceRelation.SUPPORT:
                supported_values.append(value)

        if not supported_values:
            return FacetDecision(facet, AMBIGUOUS_NO_SUPPORT, True,
                                 baseline_value, None, None, baseline_value,
                                 relations=relations)

        # equality 기반 비교이므로(compare_facets) 하나의 historical 값과
        # 동시에 같을 수 있는 서로 다른 현재 facet 값은 논리적으로 하나뿐이다.
        assert len(supported_values) == 1, (
            f"HistoricalEvidenceComparator supported more than one distinct "
            f"{facet!r} value at once: {supported_values} — this should be "
            f"structurally impossible for an equality-based comparator against "
            f"a single historical value.")
        resolved_value = supported_values[0]

        # 4. Facet-level resolution만 — 전체 Interpretation을 새로 만들지 않는다.
        matches = by_value[resolved_value]
        decision_interp = matches[0] if len(matches) == 1 else None

        return FacetDecision(facet, AMBIGUOUS_RESOLVED_BY_EVIDENCE, True,
                             baseline_value, resolved_value, decision_interp, resolved_value,
                             relations=relations)
