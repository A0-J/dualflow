"""
SOURCE SEMANTIC GATE — minimal Semantic Flow extension (docs/experiments/
agent_connected_eval.md §29, Phase 2C-P4 Finding P2C-F1).

Phase 2C's diagnostic sequence established, on real API data:
  - P3 (Measurement Audit): given a FIXED delegation string, `DelegateAgent`
    is perfectly deterministic (60/60 independent requests identical).
  - P4 (Principal Delegation Stability Audit): `PrincipalAgent.delegate()`
    itself shows substantial, quantified facet-level instability across
    independent calls on genuinely underspecified intents (action entropy
    up to 1.539, scope entropy up to 1.157), while a clear intent stays at
    H=0 and a task designed to look scope-ambiguous but that didn't
    actually produce unstable output ALSO stayed at H=0 (a negative
    control — the gate below responds to real output instability, not
    surface-level vague wording).

Conclusion: the existing (receiver-side only) Semantic Flow structurally
cannot see ambiguity that the Principal already collapsed before the
Delegate ever gets a chance to sample it. This module is the source-side
half of Semantic Flow — applied at the Principal→Delegate delegation-
generation boundary, using the exact same primitives (`CandidateDistribution`,
`dualflow.semantic.entropy()`, the legacy IG-based single-facet targeting
`clarification._select_target_facet()` already uses) the receiver side
already relies on, one layer upstream.

DualFlow stays exactly `Semantic Flow × Authority Flow` — this is NOT a
third top-level Flow; it is Semantic Flow's second boundary, sitting
before the existing receiver-side candidate-sampling/entropy check
(`DelegateAgent.sample_candidates()`/`clarification.ClarifyingDelegate`),
which remains completely unchanged.

이 모듈이 하지 않는 것:
  - `AgentDelegationRuntime`을 수정하거나 여기 연결하는 것 — 아직 전혀
    연결되지 않는다. Phase 2C-P5(독립 검증)까지만 이 단계의 범위다.
  - 새 entropy 공식을 만드는 것 — `dualflow.semantic.entropy()`를 그대로
    재사용한다(Delegate 쪽이 이미 쓰는 것과 정확히 같은 함수).
  - 새 threshold를 만드는 것 — 기존 0.8을 그대로 provisional로 쓴다
    (아키텍처 타당성 검증 목적이지, threshold가 최적이라는 주장이 아니다).
  - Delegate candidate 생성 로직을 바꾸는 것 — canonicalization 단계에서
    `DelegateAgent.propose()`를 무수정으로 재사용한다(P3가 이미 이 단계가
    고정 입력에 대해 deterministic함을 확인했다 — 새 noise원이 아니다).
  - 여러 라운드 clarification을 하는 것 — receiver-side `ClarifyingDelegate`
    와 마찬가지로 단일 라운드만 진행한다.
  - LLM으로 질문 문구를 새로 생성하는 것 — 아래 `SourceClarifyingPrincipal`
    의 질문은 결정론적 템플릿이다(이미 알려진 구조화된 사실을 그대로
    문장으로 옮길 뿐 — `clarification._render_clarify_input()`이 이미
    하는 것과 같은 종류의 결정론적 formatting이지, 새 classifier가 아니다).
    "묻는 쪽"과 "답하는 쪽"이 여기서는 같은 존재(Principal)이므로, B가
    A에게 자연스러운 질문을 만들어 보내는 receiver-side와 달리 LLM
    호출로 질문을 만들 이유가 없다.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from .clarification import _select_target_facet
from .delegate_agent import CandidateDistribution, DelegateAgent
from .principal_agent import PrincipalAgent, PrincipalClarification
from .semantic import Interpretation, entropy

_ALL_FACETS = ("action", "resource", "scope", "condition")


def sample_principal_delegations(*, principal: PrincipalAgent, delegate: DelegateAgent,
                                 goal: str, context: str, n: int
                                 ) -> tuple[CandidateDistribution, dict[Interpretation, str]]:
    """`DelegateAgent.sample_candidates()`의 source-side 대응. `principal.
    delegate()`를 `n`번 독립적으로 호출하고, 각 결과 delegation 문자열을
    `delegate.propose()`(수정 없음, 그대로 재사용) 한 번으로 canonicalize
    한다 — P3가 이미 이 canonicalization 단계 자체는 고정 입력에 대해
    deterministic함을 확인했으므로, 여기서 새 noise원이 되지 않는다.

    기존 `CandidateDistribution`을 그대로 재사용한다(새 타입을 안 만든다)
    — belief는 canonicalize된 `Interpretation`들의 empirical distribution
    이다. 두 번째 반환값은 각 `Interpretation`이 실제로 나온 delegation
    원문 하나를 보존한 매핑이다(합의된 값에 대해 나중에 "진짜 생성된"
    delegation 문자열을 복원하기 위해서 — 새로 합성하지 않는다,
    `agent_runtime._proposal_from_clarification()`의 원칙과 동일)."""
    samples: list[Interpretation] = []
    responses = []
    delegation_by_interp: dict[Interpretation, str] = {}
    for _ in range(n):
        delegation_result = principal.delegate(goal=goal, context=context)
        proposal = delegate.propose(delegation=delegation_result.delegation, context=context)
        interp = proposal.interpretation
        samples.append(interp)
        responses.append(delegation_result.response)
        delegation_by_interp.setdefault(interp, delegation_result.delegation)

    counts = Counter(samples)
    total = len(samples)
    belief = {interp: c / total for interp, c in counts.items()}
    top = max(counts, key=lambda i: (counts[i], str(i)))
    distribution = CandidateDistribution(
        belief=belief, entropy=entropy(belief), top=top, top_probability=counts[top] / total,
        n_unique=len(counts), n_samples=total, responses=responses)
    return distribution, delegation_by_interp


def facet_entropies(distribution: CandidateDistribution,
                   facets: tuple[str, ...] = _ALL_FACETS) -> dict[str, float]:
    """`distribution.belief`를 facet 값으로 묶어(이미
    `experience_decision.FrozenCandidateEvidenceHarness.decide()`가 쓰는
    `by_value` grouping과 같은 패턴) 각 facet의 Shannon entropy를
    `dualflow.semantic.entropy()`(무수정, Delegate 쪽과 정확히 같은 함수)
    로 계산한다."""
    result = {}
    for facet in facets:
        by_value: dict[object, float] = defaultdict(float)
        for interp, p in distribution.belief.items():
            value = getattr(interp, facet)
            key = tuple(sorted(value)) if isinstance(value, frozenset) else value
            by_value[key] += p
        result[facet] = entropy(by_value)
    return result


@dataclass(frozen=True)
class SourceGateDecision:
    """`SourceSemanticGate.decide()` 호출 1회의 결과."""

    distribution: CandidateDistribution
    facet_entropies: dict[str, float]
    unstable_facets: frozenset[str]   # entropy_threshold를 넘은 facet들
    stable: bool                     # == (not unstable_facets)


class SourceSemanticGate:
    """Phase 2C-P4에서 검증된 provisional gate. 기존 0.8 threshold를
    그대로 재사용한다 — 아키텍처 타당성 검증 목적이며 threshold 최적화
    주장이 아니다(§29 follow-up)."""

    def __init__(self, entropy_threshold: float = 0.8):
        self.entropy_threshold = entropy_threshold

    def decide(self, distribution: CandidateDistribution) -> SourceGateDecision:
        entropies = facet_entropies(distribution)
        unstable = frozenset(f for f, h in entropies.items() if h > self.entropy_threshold)
        return SourceGateDecision(distribution=distribution, facet_entropies=entropies,
                                  unstable_facets=unstable, stable=not unstable)


@dataclass(frozen=True)
class SourceClarificationResult:
    """`SourceClarifyingPrincipal.resolve()` 호출 1회의 결과. `clarification.
    ClarificationResult`와 구조적으로 대응한다 — 같은 단일 라운드 설계."""

    clarified: bool
    pre_distribution: CandidateDistribution
    post_distribution: CandidateDistribution | None
    target_facet: str | None
    question: str | None
    answer: PrincipalClarification | None
    final_interpretation: Interpretation
    final_delegation: str


class SourceClarifyingPrincipal:
    """`clarification.ClarifyingDelegate`의 source-side 대응 — 정확히 같은
    단일 라운드 구조(불안정하면 한 번만 묻고, 그 답을 반영해 다시
    sampling, 자동 반복 없음)를 한 layer 위(Principal 자신의 delegation
    생성)에 적용한다."""

    def __init__(self, *, principal: PrincipalAgent, delegate: DelegateAgent,
                n: int = 20, entropy_threshold: float = 0.8):
        self.principal = principal
        self.delegate = delegate
        self.n = n
        self.gate = SourceSemanticGate(entropy_threshold=entropy_threshold)

    def resolve(self, *, goal: str, context: str) -> SourceClarificationResult:
        pre, pre_delegations = sample_principal_delegations(
            principal=self.principal, delegate=self.delegate, goal=goal, context=context,
            n=self.n)
        decision = self.gate.decide(pre)

        if decision.stable:
            final_delegation = pre_delegations.get(pre.top, next(iter(pre_delegations.values())))
            return SourceClarificationResult(
                clarified=False, pre_distribution=pre, post_distribution=None,
                target_facet=None, question=None, answer=None,
                final_interpretation=pre.top, final_delegation=final_delegation)

        # 정확히 하나의 facet만 target으로 삼는다 -- 기존 legacy IG 로직
        # (clarification._select_target_facet(), select_question()/
        # information_gain()) 그대로 재사용, 새 heuristic 아님.
        target_facet = _select_target_facet(pre.belief)

        # 결정론적 질문 -- 새 LLM 호출 없음. 이미 알려진 구조화된 사실
        # (어느 facet이, 어떤 값들로 갈렸는지)을 그대로 문장으로 옮길 뿐
        # 이다 -- clarification._render_clarify_input()과 같은 종류의
        # 결정론적 formatting.
        candidate_values = sorted({getattr(i, target_facet) for i in pre.belief})
        question = (f"Do you intend the delegated {target_facet} to be one of: "
                   f"{candidate_values}?")
        answer = self.principal.answer_clarification(goal=goal, context=context,
                                                      question=question)

        enriched_context = (f"{context}\nPrincipal clarification: {answer.answer}"
                            if context else f"Principal clarification: {answer.answer}")
        post, post_delegations = sample_principal_delegations(
            principal=self.principal, delegate=self.delegate, goal=goal,
            context=enriched_context, n=self.n)

        final_delegation = post_delegations.get(post.top, next(iter(post_delegations.values())))
        return SourceClarificationResult(
            clarified=True, pre_distribution=pre, post_distribution=post,
            target_facet=target_facet, question=question, answer=answer,
            final_interpretation=post.top, final_delegation=final_delegation)
