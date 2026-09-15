"""
SEMANTIC FLOW — Experience → Entropy → Clarification → LLM.

SAGE-Agent (Structured Uncertainty guided Clarification, arXiv 2511.08798) 의
structured belief 를 가져오되, 불확실성 척도를 EVPI 가 아니라 Shannon entropy H 로
바꾸고 θ 게이팅을 앞단에 둔다.

원 논문과의 차이 (미팅자료 13페이지 표):
  SAGE-Agent   Belief + EVPI. LLM 을 파이프라인 전체에서 상시 사용하고 질문 횟수만 절감.
  본 구현      Entropy + θ. H ≤ θ 구간에서는 LLM 호출 자체가 발생하지 않는다.
               게다가 Experience Score 가 충분하면 엔트로피 계산 이전에 자율 판단.

엔트로피를 쓰면 부수적으로 얻는 것이 하나 더 있다. SAGE-Agent 의 EVPI 는 정규화되지
않은 viability 위에서 정의돼 있어 Proposition 2 의 비음수성이 실제로는 깨지지만
(선두 후보가 제거되는 질문의 EVPI 가 음수가 된다), 정보이득 IG = H(p) - E_r[H(p|r)]
은 상호정보량이므로 항상 0 이상이다. tests/test_semantic.py 참고.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .capability import Privilege, scope_leq

DIMENSIONS = ("action", "resource", "scope", "condition")


# --------------------------------------------------------------------------
# Action Space — Agent B 가 떠올릴 수 있는 해석 후보들
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Interpretation:
    """위임 명세에 대한 하나의 구체적 해석."""
    action: str
    resource: str
    scope: str = "*"
    condition: frozenset[str] = frozenset()
    label: str = field(default="", compare=False)   # 표시용 — 동일성 비교에서 제외

    def __post_init__(self):
        if not isinstance(self.condition, frozenset):
            object.__setattr__(self, "condition", frozenset(self.condition))

    def privilege(self) -> Privilege:
        return Privilege(self.action, self.resource, self.scope, self.condition)

    def value_of(self, dimension: str) -> object:
        if dimension == "condition":
            return tuple(sorted(self.condition))
        return getattr(self, dimension)

    def __str__(self) -> str:
        return self.label or str(self.privilege())


Belief = dict[Interpretation, float]


def normalize(weights: dict[Interpretation, float]) -> Belief:
    total = sum(weights.values())
    if total <= 0:
        n = max(len(weights), 1)
        return {k: 1.0 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


def entropy(p: Belief) -> float:
    """H(p) in bits. 후보가 하나뿐이면 0."""
    return max(0.0, -sum(v * math.log2(v) for v in p.values() if v > 0))


def normalized_entropy(p: Belief) -> float:
    """H / log2|Ω| ∈ [0,1]. |Ω| 가 다른 과제끼리 θ 를 비교할 때 사용."""
    if len(p) <= 1:
        return 0.0
    return entropy(p) / math.log2(len(p))


def top(p: Belief) -> Interpretation:
    return max(p, key=lambda k: (p[k], str(k)))


# --------------------------------------------------------------------------
# Experience Score — 과거 유사 위임의 판정 결과가 누적되어 다음 위임에 반영
# --------------------------------------------------------------------------
@dataclass
class ExperienceStore:
    smoothing: int = 1                     # n0: 표본이 적을 때 신뢰도를 눌러주는 상수
    _counts: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    _index: dict[str, dict[str, Interpretation]] = field(
        default_factory=lambda: defaultdict(dict))

    def record(self, key: str, interp: Interpretation, accepted: bool) -> None:
        """Joint Verification 이 Execute 로 끝난 해석만 경험으로 축적한다."""
        if not accepted:
            return
        ik = str(interp.privilege())
        self._counts[key][ik] += 1
        self._index[key][ik] = interp

    def prior(self, key: str) -> dict[Interpretation, float]:
        return {self._index[key][ik]: n for ik, n in self._counts[key].items()}

    def score(self, key: str) -> float:
        """0~1. 표본이 많고 한 해석에 몰려 있을수록 높다."""
        c = self._counts.get(key)
        if not c:
            return 0.0
        n = sum(c.values())
        share = max(c.values()) / n
        return share * n / (n + self.smoothing)

    def best(self, key: str) -> Interpretation | None:
        c = self._counts.get(key)
        if not c:
            return None
        ik = max(c, key=lambda k: (c[k], k))
        return self._index[key][ik]

    def n(self, key: str) -> int:
        return sum(self._counts.get(key, Counter()).values())


def build_belief(candidates: Sequence[tuple[Interpretation, float]],
                 experience: dict[Interpretation, float] | None = None,
                 weight: float = 1.0) -> Belief:
    """사전 그럴듯함(파서/추론 결과)과 경험 카운트를 결합한 초기 믿음."""
    experience = experience or {}
    raw = {}
    for interp, plausibility in candidates:
        raw[interp] = max(plausibility, 1e-9) * (1.0 + weight * experience.get(interp, 0.0))
    return normalize(raw)


# --------------------------------------------------------------------------
# Clarification — Agent B → Agent A 역질의
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Question:
    text: str
    dimension: str


@dataclass(frozen=True)
class Answer:
    dimension: str
    value: object | None            # None = "그건 나도 특정 못 한다"


def candidate_questions(p: Belief) -> list[Question]:
    """아직 값이 갈리는 차원마다 하나씩. 이미 확정된 차원도 넣어두고
    정보이득이 걸러내게 한다(생성기는 단순하게, 판단은 IG 가)."""
    qs = []
    for dim in DIMENSIONS:
        values = {i.value_of(dim) for i in p}
        label = {"action": "무엇을 하라는", "resource": "어떤 자원에 대한",
                 "scope": "어디까지의 범위인지", "condition": "어떤 조건이 붙는지"}[dim]
        qs.append(Question(f"{label} 것인지 구체화해 주세요. (후보 {len(values)}개)", dim))
    return qs


def conditional_entropy(p: Belief, dimension: str) -> float:
    """E_r[H(p | r)] — 해당 차원의 값을 알게 됐을 때의 기대 잔여 엔트로피."""
    buckets: dict[object, dict[Interpretation, float]] = defaultdict(dict)
    for interp, prob in p.items():
        buckets[interp.value_of(dimension)][interp] = prob
    total = 0.0
    for bucket in buckets.values():
        mass = sum(bucket.values())
        if mass <= 0:
            continue
        total += mass * entropy(normalize(bucket))
    return total


def information_gain(p: Belief, q: Question) -> float:
    """IG = H(p) - E_r[H(p|r)] = I(해석 ; 답변). 상호정보량이므로 항상 ≥ 0."""
    return entropy(p) - conditional_entropy(p, q.dimension)


def select_question(p: Belief, asked: Counter, lam: float = 0.1
                    ) -> tuple[Question | None, list[tuple[Question, float, float]]]:
    """IG - λ·(이미 물어본 횟수) 가 최대인 질문. 이득이 0 이면 묻지 않는다."""
    scored = []
    for q in candidate_questions(p):
        ig = information_gain(p, q)
        scored.append((q, ig, ig - lam * asked[q.dimension]))
    scored.sort(key=lambda r: -r[2])
    best = scored[0]
    if best[1] <= 1e-12 or best[2] <= 0.0:
        return None, scored
    return best[0], scored


def apply_answer(p: Belief, ans: Answer) -> Belief:
    """답변과 모순되는 후보를 제거하고 재정규화. 답이 없으면 그대로."""
    if ans.value is None:
        return p
    kept = {i: w for i, w in p.items() if i.value_of(ans.dimension) == ans.value}
    if not kept:                       # A 의 답이 후보 밖 → 후보 집합이 틀린 경우
        return p
    return normalize(kept)


@dataclass(frozen=True)
class Review:
    """Slow 경로에서 Agent A 가 돌려주는 검토 결과.

    approve  제시된 해석이 의도와 일치
    correct  일치하지 않지만 A 가 교정해 줄 수 있음 (교정본 동봉)
    unsure   A 자신도 특정하지 못하는 차원에서 어긋남 — 확정 불가
    """
    status: str
    interpretation: Interpretation

    @property
    def approved(self) -> bool:
        return self.status == "approve"


class Principal:
    """Agent A 를 역질의 상대로 세운 오라클.

    - 실제 의도(true interpretation)를 알고 있고, 물어본 차원만 답한다.
    - refuses 에 든 차원은 A 자신도 특정하지 못한다("적절히 알아서 해") →
      역질의를 반복해도 H 가 안 떨어지는 상황을 재현한다.
    """

    def __init__(self, truth: Interpretation, refuses: Iterable[str] = (),
                 carelessness: float = 0.0, rng=None, overcaution: float = 0.0):
        self.truth = truth
        self.refuses = set(refuses)
        # A 가 제안을 제대로 안 읽고 그냥 승인해 버릴 확률(안전성 손실 방향).
        # 현실의 검토자는 완벽하지 않으며, 이 값이 Slow 축의 신뢰도 상한선을 결정한다.
        self.carelessness = carelessness
        # 맞는 제안인데도 괜히 의심하고 다시 확인을 요구할 확률(유용성 손실 방향).
        # carelessness 와 독립적인 별도 축이다 — 둘을 같은 확률로 묶으면 carelessness=1
        # 에서 "틀린 건 항상 승인 + 맞는 것도 항상 반려"가 되어, 애초에 Slow 가 아무것도
        # 확정 못 하게 되므로 experience 축적 자체가 막힌다(§실험③의 전제가 깨짐).
        self.overcaution = overcaution
        self.rng = rng or random.Random(0)
        self.transcript: list[tuple[str, str]] = []

    # ---- Fast 경로: 차원 하나씩 되묻기 ---------------------------------
    def answer(self, q: Question) -> Answer:
        if q.dimension in self.refuses:
            self.transcript.append((q.text, "그건 상황에 맞게 판단해 주세요"))
            return Answer(q.dimension, None)
        value = self.truth.value_of(q.dimension)
        self.transcript.append((q.text, f"{q.dimension} = {value}"))
        return Answer(q.dimension, value)

    # ---- Slow 경로: 해석 전체를 제시하고 승인/교정받기 --------------------
    def review(self, proposed: Interpretation) -> Review:
        """B 가 정리한 해석 전체를 A 가 검토한다.

        B 의 자기신고 불확실성(H)에 전혀 의존하지 않는 유일한 경로다.
        후보 집합이 조작돼 H=0 이 되더라도 이 경로는 영향을 받지 않는다.
        """
        summary = f"제안: {proposed}"
        if proposed != self.truth and self.carelessness > 0 and self.rng.random() < self.carelessness:
            self.transcript.append((summary, "(대충 훑고) 네 그렇게 하세요"))
            return Review("approve", proposed)      # 부주의한 승인
        if proposed == self.truth:
            # overcaution==0(기본값)일 때 rng.random() 을 아예 호출하지 않는다 —
            # 호출은 하되 항상 거짓이 되게만 짜면 결과는 같아 보여도 그 한 번의
            # draw 가 이후 모든 확률적 실험의 RNG 시퀀스를 밀어버린다. 실제로
            # 이 때문에 fig5/fig6 수치가 도입 당시 "그대로 유지된다"던 주장과
            # 달리 조용히 바뀌어 있었다(§EXPERIMENTS.md 참고). and 의 단락평가로
            # 그 draw 자체를 건너뛰어 이전 RNG 시퀀스와 완전히 동일하게 만든다.
            if self.overcaution > 0 and self.rng.random() < self.overcaution:
                self.transcript.append((summary, "(건성으로 훑다 괜히) 이거 다시 확인해주세요"))
                return Review("unsure", proposed)   # 과잉반려 — 맞는 걸 괜히 붙잡음
            self.transcript.append((summary, "네, 그 해석이 맞습니다"))
            return Review("approve", proposed)

        diff = [d for d in DIMENSIONS
                if proposed.value_of(d) != self.truth.value_of(d)]
        fixable = [d for d in diff if d not in self.refuses]
        patched = Interpretation(
            **{**{d: getattr(self.truth if d in fixable else proposed, d)
                  for d in DIMENSIONS},
               "label": self.truth.label if set(diff) <= set(fixable) else proposed.label})

        if patched == self.truth:
            self.transcript.append((summary, f"아니요, 이렇게 해주세요: {self.truth}"))
            return Review("correct", self.truth)
        self.transcript.append((summary, "그 부분은 저도 확실하지 않습니다"))
        return Review("unsure", proposed)

    # ---- Authority Feedback Loop: 권한 범위 협상 --------------------------
    def review_authority(self, proposed: Interpretation, suggested: Interpretation):
        """B 가 권한 상한을 넘겨 제안했을 때 A 가 범위를 확인/축소해준다.

        `review()` 와 묻는 것이 다르다 — "의미가 맞는지" 가 아니라 "이 범위로
        좁혀도 내가 원하는 걸 할 수 있는지" 다. carelessness/overcaution 을
        review() 와 같은 의미로 재사용한다: 부주의하면 범위 밖 제안을 확인 없이
        그대로 승인해버리고(그 결과 non-amplification 검사가 이후 단계에서
        따로 걸러내야 한다), 과잉신중하면 이미 충분한 제안도 괜히 반려한다.
        """
        from .authority_feedback import AuthorityFeedback, FeedbackDecision  # 순환 참조 방지

        summary = f"제안: {proposed} 는 권한 밖 — 최대 {suggested} 까지 가능"

        if self.carelessness > 0 and self.rng.random() < self.carelessness:
            self.transcript.append((summary, "(확인 안 하고) 네 그걸로 하세요"))
            return AuthorityFeedback(FeedbackDecision.APPROVE, proposed)

        truth = self.truth
        same_target = truth.action == suggested.action and truth.resource == suggested.resource
        fits = same_target and scope_leq(truth.scope, suggested.scope)

        if not fits:
            self.transcript.append((summary, "그 범위로는 제가 원하는 걸 할 수 없습니다"))
            return AuthorityFeedback(FeedbackDecision.REJECT, None)

        if truth.scope == suggested.scope and truth.condition <= suggested.condition:
            if self.overcaution > 0 and self.rng.random() < self.overcaution:
                self.transcript.append((summary, "(괜히) 그 범위도 다시 확인해주세요"))
                return AuthorityFeedback(FeedbackDecision.REJECT, None)
            self.transcript.append((summary, f"네, {suggested} 까지만 하면 됩니다"))
            return AuthorityFeedback(FeedbackDecision.RESTRICT, suggested)

        if "scope" in self.refuses:
            self.transcript.append((summary, f"정확히는 모르겠지만 {suggested} 까지면 안전합니다"))
            return AuthorityFeedback(FeedbackDecision.RESTRICT, suggested)

        self.transcript.append((summary, f"아니요, 정확히는: {truth}"))
        return AuthorityFeedback(FeedbackDecision.CORRECT, truth)
