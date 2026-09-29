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

Consumption contract (fixed before any main-experiment API run; revised
under REVISION 2 below):

  1. **Stability gate first, using the existing criterion.** If the
     current candidate distribution's entropy is already at or below
     `entropy_threshold` (the same threshold `ClarifyingDelegate` uses),
     historical evidence is never even consulted, and the baseline
     decision (`distribution.top`) is returned unchanged — a gate on
     *consulting* evidence, not a rule applied after generating a
     decision. **Precise scope**: this gate does not recognize "this is an
     explicit instruction" as a natural-language property — it only
     checks whether the *already-sampled* distribution happens to satisfy
     the existing stability criterion. When it does, a historical value
     cannot override it, by construction. If real sampling on an
     explicit-change delegation turns out unstable despite the text being
     explicit, this harness *will* proceed to consult history — but, as of
     REVISION 2, consulting history no longer means automatically
     overriding the decision (see point 4).
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
  4. **No automatic resolution (REVISION 2 / Option B: conservative
     abstention).** If exactly one facet value is `SUPPORT`ed, this
     harness does **not** apply it as the decision. Historical evidence is
     advisory, not authoritative: it identifies that a prior confirmed
     interpretation exists and records the relation for audit, but the
     current decision stays at the baseline value until a Principal
     confirms it through the existing clarification path. The stage
     `AMBIGUOUS_REQUIRES_CLARIFICATION` is a signal meant to be consumed by
     that path (not yet wired here — see "이 모듈이 하지 않는 것" below),
     carrying the supported value as `resolved_value` for reference and the
     full relation map in `relations` for audit — but `final_value` always
     equals `baseline_value` at this stage. No synthesized `Interpretation`
     built from historical resource/scope/condition values is ever
     produced, and no cross-facet amplification is possible, because
     nothing is ever applied automatically in the first place.

REVISION HISTORY:
  - REVISION 1 (original): step 4 above *did* automatically resolve the
    facet to the `SUPPORT`ed historical value when exactly one candidate
    value matched, producing `AMBIGUOUS_RESOLVED_BY_EVIDENCE` with
    `final_value = resolved_value`. The B7d-v3 main experiment (real API,
    200 calls, docs/experiments/agent_connected_eval.md §24) validated this
    against real ambiguous/explicit-change delegations. A deterministic
    (zero-API) counterexample then showed this contract *does* override an
    explicit-export delegation's decision when real sampling on it happens
    to be unstable (entropy > threshold) — a genuinely reachable F4
    (stale-history override), not a hypothetical. A follow-up audit of the
    entire sender-side pipeline (`principal_agent.py`, `agent_runtime.py`)
    found no structural, non-textual signal — distinct from entropy —
    for "this facet was explicitly asserted by the current request" as
    opposed to "the model is merely confident about it": `PrincipalIntent`
    (`restate_intent()`) is itself a fresh per-invocation LLM inference,
    not pre-existing sender state, and its prompt forces all four fields
    regardless of what the original goal specified. Without trustworthy
    current-side provenance, an automatic-override contract cannot
    distinguish the two cases it most needs to distinguish.
  - REVISION 2 (this revision, Option B): automatic resolution is
    withdrawn. Historical evidence downgrades from "authoritative decision
    source" to "advisory evidence that flags where clarification is
    needed." F4 (stale-history override) is now structurally unreachable
    through this harness, because nothing is ever applied automatically —
    see `tests/test_experience_decision.py::TestH2StructuralCounterexample`
    for the same deterministic counterexample re-run against this
    revision, now confirming no override occurs. This is not a rollback of
    REVISION 1's finding (the ambiguous-transfer signal it measured, §24
    H1, still stands as a real observation) — it is a narrowing of what
    this harness is trusted to decide on its own, in direct response to
    what REVISION 1 empirically exposed. See §24 Follow-up ("Option B:
    conservative abstention") in agent_connected_eval.md for the full
    design rationale.

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
#   STABLE_BASELINE                  -> 이번 candidate distribution이 안정적이어서
#                                        evidence를 아예 안 봄. F4는 여기서
#                                        구조적으로 불가능하다(history를
#                                        보지도 않으므로).
#   AMBIGUOUS_NO_ELIGIBLE_EVIDENCE   -> F1 (evidence unavailable)
#   AMBIGUOUS_NO_SUPPORT             -> F2 (support absent)
#   AMBIGUOUS_REQUIRES_CLARIFICATION -> REVISION 2(Option B)의 새 경로 —
#                                        eligible history가 현재 candidate
#                                        중 하나를 SUPPORT하지만, 자동으로
#                                        적용하지 않는다. relations는
#                                        감사용으로 채워지고 resolved_value는
#                                        참고용으로 남지만 final_value는
#                                        항상 baseline_value다. 향후
#                                        ClarifyingDelegate 경로로 넘길 수
#                                        있는 signal.
#   AMBIGUOUS_RESOLVED_BY_EVIDENCE   -> REVISION 1(원래) contract에서만
#                                        쓰였던 자동-해결 성공 경로. 이
#                                        모듈의 `decide()`는 더 이상 이
#                                        stage를 반환하지 않는다 — 상수는
#                                        과거 결과(§24 main experiment
#                                        JSON, 옛 테스트/문서)를 참조할 때를
#                                        위해서만 남겨둔다.
STABLE_BASELINE = "stable_baseline"
AMBIGUOUS_NO_ELIGIBLE_EVIDENCE = "ambiguous_no_eligible_evidence"
AMBIGUOUS_NO_SUPPORT = "ambiguous_no_support"
AMBIGUOUS_REQUIRES_CLARIFICATION = "historical_evidence_requires_clarification"
AMBIGUOUS_RESOLVED_BY_EVIDENCE = "ambiguous_resolved_by_evidence"  # REVISION 1 only; no longer emitted.


@dataclass(frozen=True)
class FacetDecision:
    """`FrozenCandidateEvidenceHarness.decide()` 호출 1회의 결과.

    `final_value`가 실제로 보고해야 할 "이번 decision의 facet 값"이다 —
    항상 채워진다. REVISION 2(Option B) 이후로 `final_value`는 evidence가
    이 harness에 의해 자동으로 적용된 적이 없으므로(`AMBIGUOUS_REQUIRES_
    CLARIFICATION`을 포함해 모든 stage에서) 항상 `baseline_value`와 같다
    — evidence가 실제로 current decision을 바꾸는 것은 이 harness 밖의
    별도 clarification/Principal-confirmation 단계에서만 일어날 수 있다.
    `resolved_value`는 REVISION 2부터 "자동으로 채택된 값"이 아니라
    "historical evidence가 support했다고 감사용으로 기록해두는 값"이다 —
    `decision_interpretation`은 STABLE_BASELINE에서만 채워진다(그 외
    stage는 애초에 자동 decision을 만들지 않으므로 항상 `None`)."""

    facet: str
    stage: str
    used_historical_evidence: bool

    baseline_value: object          # distribution.top의 해당 facet 값 (개입 전)
    resolved_value: object | None   # evidence가 support한 값(참고/감사용) — REVISION 2부터
                                    # final_value에 자동 반영되지 않는다.
    decision_interpretation: Interpretation | None  # STABLE_BASELINE에서만 채워짐

    final_value: object             # 이번 decision에서 실제로 쓸 값 (REVISION 2:
                                    # 항상 baseline_value와 같다 — 이 harness는
                                    # 더 이상 자동으로 override하지 않는다)

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

        # 4. REVISION 2 (Option B: conservative abstention) — eligible
        #    history가 현재 candidate 중 하나를 SUPPORT하더라도, 이
        #    harness는 자동으로 채택하지 않는다. Historical evidence는
        #    advisory일 뿐 authoritative decision source가 아니다:
        #    "prior confirmed interpretation이 존재하고, 이 candidate를
        #    support한다"는 사실 자체는 relations에 감사 가능한 형태로
        #    기록하고, resolved_value에 참고용으로 남기지만, final_value는
        #    baseline_value 그대로 유지한다. 실제로 이 facet을 확정하는
        #    것은 (아직 이 harness에 연결되지 않은) 기존 clarification/
        #    Principal-confirmation 경로의 몫이다 — 전체 Interpretation을
        #    historical 값으로 새로 만드는 일은 어차피 일어나지 않는다.
        return FacetDecision(facet, AMBIGUOUS_REQUIRES_CLARIFICATION, True,
                             baseline_value, resolved_value, None, baseline_value,
                             relations=relations)
