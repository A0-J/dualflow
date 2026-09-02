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
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .capability import Privilege

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
    label: str = ""

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


class Principal:
    """Agent A 를 역질의 상대로 세운 오라클.

    - 실제 의도(true interpretation)를 알고 있고, 물어본 차원만 답한다.
    - refuses 에 든 차원은 A 자신도 특정하지 못한다("적절히 알아서 해") →
      역질의를 반복해도 H 가 안 떨어지는 상황을 재현한다.
    """

    def __init__(self, truth: Interpretation, refuses: Iterable[str] = ()):
        self.truth = truth
        self.refuses = set(refuses)
        self.transcript: list[tuple[str, str]] = []

    def answer(self, q: Question) -> Answer:
        if q.dimension in self.refuses:
            self.transcript.append((q.text, "그건 상황에 맞게 판단해 주세요"))
            return Answer(q.dimension, None)
        value = self.truth.value_of(q.dimension)
        self.transcript.append((q.text, f"{q.dimension} = {value}"))
        return Answer(q.dimension, value)
