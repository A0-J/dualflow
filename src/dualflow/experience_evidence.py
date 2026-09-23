"""
V3 STRUCTURED-FACET COMPARATOR (opt-in, not wired into any runtime yet).

REVISION HISTORY OF THIS MODULE (kept here so the reasoning survives —
see docs/experiments/agent_connected_eval.md §22/§23 for the full record):

  1. First version (commit 52de6824e6376867a998e0e36f0573a21f4b5b08):
     `HistoricalEvidenceComparator` re-rendered historical experience as
     natural-language text (reusing B7d.5/B7d.6's N/L/P/S renderers) and
     asked an LLM to classify SUPPORT/CONFLICT/IRRELEVANT/UNCERTAIN
     against a fixed candidate.
  2. A 180-call pilot against that version was **inconclusive** (§23):
     even the maximally-aligned case (structured evidence confirming
     `summarize`, candidate `summarize`) produced CONFLICT more often
     than SUPPORT. A no-API audit of the exact prompts found two
     concrete contaminants: (a) the historical evidence text exposed the
     *whole* past episode, including the non-transferable `August 2026`
     scope, right next to the current candidate's `September 2026` scope
     -- risking a whole-episode mismatch being read as an action
     conflict; (b) the reused `v2` header ("If the current delegation
     explicitly specifies an action, follow the current delegation...")
     was written as an instruction for whoever decides an action, not as
     evidence data, and was contaminating a classification prompt it was
     never designed for.
  3. Second version: the natural-language re-rendering step was removed
     entirely, replaced with a deterministic, per-facet comparison of
     `candidate`'s fields against a bare historical `Interpretation`'s
     fields (`rule_engine.field_match()`, reused unchanged).
  4. **This version**: revision 3 had its own gap, found on review before
     any API validation was run against it -- it treated every field of
     `AgentExperience.confirmed_interpretation` as usable evidence just
     because the field held a value. But a structured value existing is
     not the same as the Principal having actually confirmed that facet.
     In this pilot scenario, resource/scope are always fixed by the
     grounded environment (B6.1) -- clarification only ever targets
     `action` -- so `confirmed_interpretation.scope` (e.g. the historical
     episode's `2026-08`) is just whatever the environment happened to
     fix it to, NOT a reusable fact the Principal confirmed. Revision 3's
     `judge()` would have let a `scope` query silently become CONFLICT
     evidence, reintroducing exactly the whole-episode contamination §23
     found, just moved one layer down.

     The fix: `judge()` now takes the whole `AgentExperience` (not a bare
     `Interpretation`) and checks `experience.confirmed_facets`
     (`agent_experience.py`, computed from which facets actually varied in
     `pre_distribution` before clarification -- the only evidence that a
     facet was genuinely in question) before treating a facet as
     comparable evidence at all. A facet not in `confirmed_facets` returns
     `IRRELEVANT`, regardless of whether its values happen to match or not
     -- "no provenance" must not be able to produce SUPPORT by coincidence.

Because of (3)/(4), paraphrase invariance is no longer this module's
responsibility, and neither is deciding which facets were actually
clarified. Both move earlier in the pipeline: turning a Principal's
natural-language answer into a canonical `confirmed_interpretation`, and
recording which facets that answer actually addressed, both happen in
`build_verified_experience()` (`agent_experience.py`) -- unchanged by this
module and out of scope here.

이 모듈이 하지 않는 것 (전부 의도적, revision 4에서도 변하지 않음):
  - Delegate candidate generation에 관여하는 것. `DelegateAgent`를 import
    조차 하지 않는다 — candidate는 항상 호출자가 이미 만들어서(history
    없이 생성된) `Interpretation`으로 넘겨준다. 이 모듈은 그 candidate를
    "이미 주어진 것"으로만 다루고, 절대 새로 생성하지 않는다.
  - Authority/Budget/AuthorityVerifierAgent에 관여하는 것 — import도,
    참조도 하지 않는다. 권한과 완전히 무관하다.
  - `AgentDelegationRuntime.run()`에 연결되는 것 — 아직 연결되지 않는다.
    기존 runtime/Authority Flow/`SemanticVerifierAgent.
    verify_agent_proposal()`의 동작은 이 모듈로 인해 전혀 바뀌지 않는다.
  - 여러 verified experience를 하나로 합치거나 우선순위를 매기는 것 —
    `judge()`는 historical experience 하나(`AgentExperience` 하나)만
    받는다. multi-experience aggregation/순서는 여전히 나중 문제다.
  - facet 사이에 서로 영향을 주는 것 — `compare_facets()`는 각 facet을
    완전히 독립적으로 비교한다. historical scope가 다르다고 해서 action
    비교 결과가 오염되지 않는다.
  - provenance 없는 facet을 evidence로 취급하는 것 — `confirmed_facets`에
    없는 facet은 값이 우연히 같아도 무조건 `IRRELEVANT`다(아래 참고).

핵심 invariant: `judge()`는 주어진 `candidate: Interpretation`을 그대로
반환한다(생성/수정하지 않는다) — 오직 그 candidate와 historical experience
사이의, 지정한 facet 하나에 대한 관계만 판단하고, 그 facet이 실제로
Principal-confirmed된 경우에만 SUPPORT/CONFLICT를 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .agent_experience import AgentExperience
from .rule_engine import Fields, field_match
from .semantic import Interpretation

FACETS = ("action", "resource", "scope", "condition")


class EvidenceRelation(str, Enum):
    """이 comparator가 반환할 수 있는 값의 전체 집합.

    `compare_facets()`는 순수 값 비교이므로 `SUPPORT`/`CONFLICT`만
    만든다. `HistoricalEvidenceComparator.judge()`는 그 위에 provenance
    게이트를 하나 더 두므로 `IRRELEVANT`도 실제로 만든다 — 요청한 facet이
    `experience.confirmed_facets`에 없으면(=Principal이 그 facet을 실제로
    확인한 적이 없으면) 값이 우연히 같더라도 `IRRELEVANT`를 돌려준다.
    `UNCERTAIN`은 이 revision의 어떤 경로에서도 만들어지지 않는다 —
    인터페이스 안정성을 위해 열거형에만 남겨둔다(다른/미래의 comparator
    구현이 쓸 수 있다)."""

    SUPPORT = "SUPPORT"
    CONFLICT = "CONFLICT"
    IRRELEVANT = "IRRELEVANT"
    UNCERTAIN = "UNCERTAIN"


def _as_fields(interp: Interpretation) -> Fields:
    """`rule_engine.Fields`로 감싸는 최소 어댑터 — `field_match()`를
    그대로 재사용하기 위해서다(새 동일성 규칙을 만들지 않는다). SOP
    분류값(action_type/sensitivity/scope_breadth/condition_met)은
    `sysvars`가 있어야 `classify()`로 계산되는데, 이 비교는 그 분류가
    전혀 필요 없다 — `field_match()`가 실제로 읽는 건
    action/resource/scope/condition 원본 값뿐이다(rule_engine.py
    117-130행). 그래서 분류값은 자리만 채우는 placeholder다."""
    return Fields(
        action_type="", sensitivity="", scope_breadth="", condition_met=False,
        action=interp.action, resource=interp.resource,
        scope=interp.scope, condition=interp.condition,
    )


def compare_facets(candidate: Interpretation,
                   historical: Interpretation) -> dict[str, EvidenceRelation]:
    """`candidate`의 네 facet 각각을 `historical`의 같은 facet과 독립적으로
    비교한다 — 순수 값 비교이고 provenance는 전혀 보지 않는다(그건 이
    함수를 호출하는 `judge()`의 책임이다). `rule_engine.field_match()`를
    그대로 재사용한다: action이 같으면 그 facet만 SUPPORT, 다르면 그
    facet만 CONFLICT — 다른 facet의 일치/불일치는 전혀 영향을 주지
    않는다(`field_match()`가 애초에 네 독립된 boolean을 반환하도록
    설계돼 있다 — rule_engine.py 117행)."""
    fm = field_match(_as_fields(candidate), _as_fields(historical))
    return {facet: (EvidenceRelation.SUPPORT if matched else EvidenceRelation.CONFLICT)
            for facet, matched in fm.items()}


@dataclass(frozen=True)
class EvidenceJudgment:
    """`judge()` 호출 1회의 결과. `candidate`/`historical`은 호출자가 넘긴
    값 그대로다 — 이 클래스도, `HistoricalEvidenceComparator`도 이 필드를
    절대 다른 값으로 바꾸지 않는다."""

    candidate: Interpretation
    historical: AgentExperience
    facet: str
    relation: EvidenceRelation


class HistoricalEvidenceComparator:
    """Principal-confirmed historical experience를 자연어로 다시
    렌더링하지 않고, 이미 구조화된 값끼리 결정론적으로 비교한다 — 단,
    `experience.confirmed_facets`에 있는 facet만 비교 대상으로 삼는다.
    LLM 호출이 전혀 없다 — 그래서 생성자도 아무것도 받지 않는다.
    `judge()`는 지정한 facet 하나에 대한 관계만 돌려준다(기본값
    `"action"` — 이 시나리오에서 transfer 대상인 facet)."""

    def judge(self, *, candidate: Interpretation, historical: AgentExperience,
             facet: str = "action") -> EvidenceJudgment:
        if facet not in FACETS:
            raise ValueError(f"unknown facet: {facet!r} (expected one of {FACETS})")

        if facet not in historical.confirmed_facets:
            # provenance 없음 — 값이 우연히 같아도 evidence로 승격시키지
            # 않는다. 이게 이 revision의 핵심 계약이다.
            return EvidenceJudgment(candidate=candidate, historical=historical,
                                    facet=facet, relation=EvidenceRelation.IRRELEVANT)

        relations = compare_facets(candidate, historical.confirmed_interpretation)
        return EvidenceJudgment(candidate=candidate, historical=historical,
                                facet=facet, relation=relations[facet])
