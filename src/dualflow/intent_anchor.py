"""
PRINCIPAL INTENT ANCHOR — Semantic Flow's Intent Grounding boundary
(docs/experiments/agent_connected_eval.md §29, Phase 3).

Phase 2C Final (140 real episodes) found Safety FAILED, concentrated in
`confident_semantic_misread`: 4/20 (20%) unsafe executions, in every
case because `PrincipalAgent.restate_intent()` — the independent
reconstruction `principal_match` relies on as its sole safety backstop —
itself confidently agreed with the Delegate's wrong interpretation.
Phase 3A (post-hoc, 0 new API calls, computed entirely from the already-
frozen 140-episode data — `restate_intent()`'s inputs never vary across
that task's 20 episodes, so the 20 already-recorded
`principal_intent_action` values already are 20 independent samples of
the distribution in question) confirmed `restate_intent()` is not the
ground truth it was implicitly treated as: it has its own ~20% per-call
error rate (16/20 summarize, 4/20 read on the fixed goal/context), and
while its errors are not strongly correlated with the Delegate's own
errors (0.25 conditional vs. 0.20 marginal — close to independent), a
*single* `restate_intent()` call is not a reliable comparison anchor.

This module adds Intent Grounding as a third component of Semantic Flow
(alongside the existing source-side stability check, §29 Phase 2C-P4/P5,
and the unchanged receiver-side stability check) — comparing the
Delegate's interpretation not against one `restate_intent()` call, but
against a `PrincipalIntentAnchor`: a per-facet structured statement of
what the Principal actually intends, where each facet's `confirmed`
status is *earned* via the same repeated-sampling + entropy methodology
already validated for source-side stability, not granted by a single
trusted call. DualFlow stays exactly `Semantic Flow x Authority Flow` —
this is not a fourth top-level Flow.

Reused, unmodified: `CandidateDistribution` (`.delegate_agent`),
`dualflow.semantic.entropy()`, `source_semantic_gate.facet_entropies()`
(generic over any `CandidateDistribution` — not duplicated here),
`source_semantic_gate._ALL_FACETS`, the same 0.8 `entropy_threshold`
(no new threshold invented), and the single-facet-per-round,
deterministic-question, re-sample-with-enriched-context clarification
pattern already established by `SourceClarifyingPrincipal` — the
"asker" and "answerer" are the same entity (the Principal) here too, so
the clarifying question is a plain template, not a new LLM call.

This module does NOT parse a clarification answer directly into a facet
value — exactly like `SourceClarifyingPrincipal`, the answer is folded
into context and a fresh, structured, repeated-sampling round produces
the corrected (and freshly re-confirmed) anchor. No free-text-to-value
parser is introduced.

Standalone only — not yet wired into `AgentDelegationRuntime`. Mirrors
exactly how `source_semantic_gate.py` itself was validated: design ->
deterministic tests (this commit) -> real-API validation -> THEN opt-in
runtime integration, each its own explicit step.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .delegate_agent import CandidateDistribution
from .principal_agent import PrincipalAgent, PrincipalClarification
from .semantic import Interpretation, entropy
from .source_semantic_gate import _ALL_FACETS, facet_entropies


def sample_principal_intents(*, principal: PrincipalAgent, goal: str, context: str,
                             n: int) -> CandidateDistribution:
    """`restate_intent()`의 반복 샘플링. `sample_principal_delegations()`
    (source_semantic_gate.py)보다 단순하다 — `restate_intent()`가 이미
    구조화된 `Interpretation`을 직접 반환하므로 canonicalize용
    `delegate.propose()` 호출이 필요 없다. 기존 `CandidateDistribution`을
    그대로 재사용한다(새 타입을 만들지 않는다)."""
    samples: list[Interpretation] = []
    responses = []
    for _ in range(n):
        intent = principal.restate_intent(goal=goal, context=context)
        samples.append(intent.intended_action)
        responses.append(intent.response)

    counts = Counter(samples)
    total = len(samples)
    belief = {interp: c / total for interp, c in counts.items()}
    top = max(counts, key=lambda i: (counts[i], str(i)))
    return CandidateDistribution(
        belief=belief, entropy=entropy(belief), top=top, top_probability=counts[top] / total,
        n_unique=len(counts), n_samples=total, responses=responses)


@dataclass(frozen=True)
class AnchorFacet:
    """Anchor 안의 facet 하나. `confirmed`는 Principal이 "그렇다고
    말했다"가 아니라, 반복 샘플링에서 실제로 합의됐다(entropy <=
    threshold)는 뜻이다 — Phase 3A가 진단한 "단일 호출은 신뢰할 수 있는
    anchor가 아니다"라는 gap을 직접 닫는 필드."""

    value: object                # str(action/resource/scope) 또는 frozenset[str](condition)
    source: str                  # provenance tag -- 현재는 항상 "principal_plan"
    confirmed: bool              # == (해당 facet의 반복-샘플링 entropy <= threshold)


@dataclass(frozen=True)
class PrincipalIntentAnchor:
    """`build_intent_anchor()` 호출 1회의 결과."""

    facets: dict[str, AnchorFacet]           # key는 source_semantic_gate._ALL_FACETS
    distribution: CandidateDistribution      # 근거가 된 반복 샘플링 원본(감사용, 새로 계산 안 함)
    top_interpretation: Interpretation       # == distribution.top, 편의용


def build_intent_anchor(*, principal: PrincipalAgent, goal: str, context: str, n: int,
                        entropy_threshold: float = 0.8) -> PrincipalIntentAnchor:
    """`sample_principal_intents()`로 얻은 분포를 anchor로 바꾼다.
    facet별 entropy는 `facet_entropies()`(source_semantic_gate.py,
    무수정 재사용)로 계산한다 — 새 entropy 공식을 만들지 않는다."""
    distribution = sample_principal_intents(
        principal=principal, goal=goal, context=context, n=n)
    return _anchor_from_distribution(distribution, entropy_threshold=entropy_threshold)


def _anchor_from_distribution(distribution: CandidateDistribution, *,
                              entropy_threshold: float) -> PrincipalIntentAnchor:
    entropies = facet_entropies(distribution)
    facets = {
        facet: AnchorFacet(value=getattr(distribution.top, facet), source="principal_plan",
                          confirmed=entropies[facet] <= entropy_threshold)
        for facet in _ALL_FACETS
    }
    return PrincipalIntentAnchor(facets=facets, distribution=distribution,
                                 top_interpretation=distribution.top)


@dataclass(frozen=True)
class AnchorCompatibilityResult:
    """`check_compatibility()` 호출 1회의 결과. `principal_match`(단일
    boolean)의 facet-level, provenance-aware 확장 — 값이 다르다는 것과
    "confirmed facet이 다르다"는 것을 구분한다."""

    matched_facets: frozenset[str]
    mismatched_confirmed_facets: frozenset[str]     # 반드시 대응해야 하는 불일치
    mismatched_unconfirmed_facets: frozenset[str]    # 감사용 -- 자동으로 막지 않는다
    compatible: bool                                # == (not mismatched_confirmed_facets)


def check_compatibility(anchor: PrincipalIntentAnchor,
                        interpretation: Interpretation) -> AnchorCompatibilityResult:
    """Delegate의 (혹은 이미 fusion을 거친) `interpretation`을 anchor와
    facet별로 비교한다. `confirmed=False`인 facet의 불일치는 anchor
    자체가 아직 반복-샘플링으로 합의되지 않은 값이므로 — 자동으로
    막지 않는다(Phase 3A가 보여준 "단일 호출을 무조건 신뢰하면 안 된다"
    는 원칙을 anchor 자신에도 그대로 적용)."""
    matched: set[str] = set()
    mismatched_confirmed: set[str] = set()
    mismatched_unconfirmed: set[str] = set()
    for facet in _ALL_FACETS:
        anchor_value = anchor.facets[facet].value
        delegate_value = getattr(interpretation, facet)
        if anchor_value == delegate_value:
            matched.add(facet)
        elif anchor.facets[facet].confirmed:
            mismatched_confirmed.add(facet)
        else:
            mismatched_unconfirmed.add(facet)
    return AnchorCompatibilityResult(
        matched_facets=frozenset(matched),
        mismatched_confirmed_facets=frozenset(mismatched_confirmed),
        mismatched_unconfirmed_facets=frozenset(mismatched_unconfirmed),
        compatible=not mismatched_confirmed)


@dataclass(frozen=True)
class GroundedVerificationResult:
    """`GroundedIntentVerifier.verify()` 호출 1회의 결과. `clarification.
    ClarificationResult`/`source_semantic_gate.SourceClarificationResult`
    와 구조적으로 대응한다 — 같은 단일 라운드 설계."""

    anchor: PrincipalIntentAnchor
    interpretation: Interpretation
    pre_compatibility: AnchorCompatibilityResult
    clarified: bool
    target_facet: str | None
    question: str | None
    answer: PrincipalClarification | None
    post_anchor: PrincipalIntentAnchor | None
    post_compatibility: AnchorCompatibilityResult | None
    final_compatible: bool


class GroundedIntentVerifier:
    """`clarification.ClarifyingDelegate`/`source_semantic_gate.
    SourceClarifyingPrincipal`과 같은 구조(단일 라운드, 불일치 시 정확히
    하나의 facet만 확인, 자동 반복 없음)를 Intent Grounding 경계에
    적용한다."""

    def __init__(self, *, principal: PrincipalAgent, n: int = 20,
                entropy_threshold: float = 0.8):
        self.principal = principal
        self.n = n
        self.entropy_threshold = entropy_threshold

    def verify(self, *, goal: str, context: str,
              interpretation: Interpretation) -> GroundedVerificationResult:
        anchor = build_intent_anchor(
            principal=self.principal, goal=goal, context=context, n=self.n,
            entropy_threshold=self.entropy_threshold)
        pre = check_compatibility(anchor, interpretation)

        if pre.compatible:
            return GroundedVerificationResult(
                anchor=anchor, interpretation=interpretation, pre_compatibility=pre,
                clarified=False, target_facet=None, question=None, answer=None,
                post_anchor=None, post_compatibility=None, final_compatible=True)

        # 정확히 하나의 facet만 target으로 삼는다 -- 여러 개가 동시에
        # mismatched_confirmed면 _ALL_FACETS 순서(action/resource/scope/
        # condition)의 고정 우선순위로 하나만 고른다. 이 상황은 IG 기반
        # 분포 비교가 아니라 anchor-vs-delegate의 binary mismatch이므로,
        # 기존 legacy IG 로직(_select_target_facet)은 여기 적용 대상이
        # 아니다 -- 대신 같은 "한 라운드에 facet 하나" 원칙만 재사용한다.
        target_facet = next(f for f in _ALL_FACETS if f in pre.mismatched_confirmed_facets)
        anchor_value = anchor.facets[target_facet].value
        delegate_value = getattr(interpretation, target_facet)

        # 결정론적 질문 -- 새 LLM 호출 없음(source_semantic_gate.py와 같은
        # 원칙: 여기서도 "묻는 쪽"과 "답하는 쪽"이 같은 존재, Principal).
        question = (
            f"Current interpretation: {target_facet} = {delegate_value}. "
            f"Principal committed {target_facet}: {target_facet} = {anchor_value}. "
            f"Which {target_facet} should be delegated?")
        answer = self.principal.answer_clarification(goal=goal, context=context,
                                                      question=question)

        # 답변 텍스트를 직접 파싱하지 않는다 -- enriched context로 접어
        # 넣고 구조화된 재샘플링을 다시 돈다(SourceClarifyingPrincipal과
        # 정확히 같은 패턴).
        enriched_context = (f"{context}\nPrincipal clarification: {answer.answer}"
                            if context else f"Principal clarification: {answer.answer}")
        post_distribution = sample_principal_intents(
            principal=self.principal, goal=goal, context=enriched_context, n=self.n)
        post_anchor = _anchor_from_distribution(
            post_distribution, entropy_threshold=self.entropy_threshold)
        post = check_compatibility(post_anchor, interpretation)

        return GroundedVerificationResult(
            anchor=anchor, interpretation=interpretation, pre_compatibility=pre,
            clarified=True, target_facet=target_facet, question=question, answer=answer,
            post_anchor=post_anchor, post_compatibility=post, final_compatible=post.compatible)
