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
from typing import TYPE_CHECKING, Iterable, Protocol, Sequence

from .capability import Privilege, scope_leq

if TYPE_CHECKING:  # 순환 참조 방지 — 타입 힌트용으로만, 런타임에는 import 안 됨.
    from .framework import DelegationTask

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


# --------------------------------------------------------------------------
# SEMANTIC VERIFIER AGENT — Semantic Flow를 독립적으로 호출 가능한 단위로 묶는다.
#
# B의 제안이 실제로 A의 의도와 맞는지 판단하는 게 이 클래스의 유일한 책임이다 —
# 그 제안이 허용된 권한 범위 안인지는 Authority Flow 가 별도로 본다(framework.py
# 의 Joint Verification 에서 둘을 합친다). 아래 알고리즘(Fast/Slow/AND/Adaptive)
# 자체는 framework.DelegationVerifier 에서 그대로 옮겨온 것으로, 동작에 변화가
# 없다 — 옮긴 목적은 "이 판단이 어디서 어떻게 내려지는지" 를 하나의 클래스 밖에서
# 도 재사용/단독 테스트 가능하게 만드는 것뿐이다.
# --------------------------------------------------------------------------
class _JudgeProtocol(Protocol):
    """`llm.LLMJudge` 와 동일한 형태 — 여기서 `llm.py` 를 import하면 순환
    참조가 된다(llm.py 가 이미 semantic.py 를 import한다). `authority_feedback.
    AuthorityPrincipal` 이 `semantic.Principal` 을 구체 타입으로 참조하지 않고
    프로토콜만 보는 것과 같은 이유로, 여기서도 형태만 복제한다."""
    calls: int

    def judge(self, spec: str, belief: "Belief",
              transcript: Sequence[tuple[str, str]]) -> "Interpretation": ...


@dataclass
class SemanticVerdict:
    """Semantic Flow 한 번의 결과. Fast/Slow/AND 가 공통으로 돌려준다."""
    interpretation: Interpretation
    confirmed: bool
    route: str
    h_initial: float = 0.0
    h_final: float = 0.0
    n_questions: int = 0
    n_llm: int = 0
    n_reviews: int = 0

    @property
    def status(self) -> str:
        """AuthorityVerdict.status(allow/deny/unknown) 와의 서술적 대구 —
        저장된 새 상태가 아니라 confirmed 의 별칭이다."""
        return "resolved" if self.confirmed else "unresolved"


class SemanticVerifierAgent:
    """Semantic Flow, 독립 단위. "B 의 제안이 실제로 무엇을 의미하는가" 만
    판단한다 — 그게 허용되는지는 모른다(Authority Flow 의 책임).

    Fast(entropy+역질의) / Slow(A 검토) / AND(둘 다) / Adaptive(평소엔 Fast,
    경험과 모순되거나 미확정일 때만 Slow 에스컬레이션) 네 전략을 지원한다.
    알고리즘은 추출 이전과 동일하다 — DelegationVerifier 가 갖고 있던
    _fast/_slow/_adaptive/_consistent/_experience_conflict 를 그대로 옮겼다.
    """

    def __init__(self, *, mode: str = "fast", theta: float = 0.5, k: int = 2,
                 sigma: float = 0.8, lam: float = 0.1,
                 experience_weight: float = 1.0, always_llm: bool = False,
                 use_experience: bool = True, use_llm: bool = True,
                 use_consistency_check: bool = False,
                 consistency_sigma: float | None = None,
                 adaptive_sigma: float | None = None,
                 experience: ExperienceStore | None = None,
                 judge: _JudgeProtocol | None = None):
        self.mode = mode
        self.theta, self.k, self.sigma, self.lam = theta, k, sigma, lam
        self.experience_weight = experience_weight
        self.always_llm = always_llm
        self.use_experience, self.use_llm = use_experience, use_llm
        self.use_consistency_check = use_consistency_check
        self.consistency_sigma = consistency_sigma
        self.adaptive_sigma = adaptive_sigma
        self.experience = experience if experience is not None else ExperienceStore()
        self.judge = judge

    def verify(self, task: DelegationTask, principal: "Principal",
               log: list[str]) -> SemanticVerdict:
        """mode 에 따라 분기한다 — fast | slow | and | adaptive."""
        if self.mode == "fast":
            return self._fast(task, principal, log)
        if self.mode == "slow":
            return self._slow(task, principal, log)
        if self.mode == "adaptive":
            return self._adaptive(task, principal, log)

        # AND 결합 — Fast 로 해석을 좁힌 뒤 그 결과를 A 에게 확인받는다.
        fast = self._fast(task, principal, log)
        slow = self._slow(task, principal, log, proposed=fast.interpretation, strict=True)
        agreed = fast.confirmed and slow.confirmed
        log.append(f"[and] Fast={'확정' if fast.confirmed else '미확정'} · "
                   f"Slow={'승인' if slow.confirmed else '불일치'} → "
                   f"{'통과' if agreed else '차단'}")
        return SemanticVerdict(
            fast.interpretation, agreed,
            f"and({fast.route}+{slow.route.split(':')[-1]})",
            fast.h_initial, fast.h_final,
            fast.n_questions, fast.n_llm, slow.n_reviews)

    # ---- SEMANTIC FLOW (Fast) — Experience → Entropy → Clarification → LLM
    def _fast(self, task: DelegationTask, principal: "Principal", log: list[str]):
        prior = self.experience.prior(task.key) if self.use_experience else {}
        belief = build_belief(task.candidates, prior, self.experience_weight)
        h0 = entropy(belief)
        log.append(f"[fast] |Ω|={len(belief)}  H={h0:.3f} bits "
                   f"(정규화 {normalized_entropy(belief):.2f})")

        # 1) Experience Score 게이트 — 충분하면 자율 판단
        if self.use_experience and not self.always_llm:
            s = self.experience.score(task.key)
            if s >= self.sigma:
                best = self.experience.best(task.key)
                if best is not None:
                    log.append(f"[fast] Experience Score={s:.2f} ≥ σ={self.sigma} "
                               f"→ 자율 판단: {best}")
                    return SemanticVerdict(best, True, "experience", h0, 0.0)

            elif s > 0:
                log.append(f"[fast] Experience Score={s:.2f} < σ={self.sigma} → 엔트로피 경로")

        # SAGE-Agent 식 프로파일: 게이팅 없이 항상 LLM
        if self.always_llm:
            interp = self.judge.judge(task.spec, belief, principal.transcript)
            log.append(f"[fast] (always-LLM) → {interp}")
            return SemanticVerdict(interp, True, "llm", h0, entropy(belief), n_llm=1)

        # 2) Entropy 게이트 + 3) Clarification 루프
        asked: Counter[str] = Counter()
        n_q = 0
        h = h0
        for _ in range(self.k):
            if h <= self.theta:
                break
            q, scored = select_question(belief, asked, self.lam)
            for cand, ig, sc in scored[:3]:
                log.append(f"         IG={ig:+.3f} score={sc:+.3f} | {cand.dimension}")
            if q is None:
                log.append("[fast] 정보이득이 남은 질문이 없음 → 역질의 중단")
                break
            n_q += 1
            asked[q.dimension] += 1
            ans = principal.answer(q)
            belief = apply_answer(belief, ans)
            h = entropy(belief)
            log.append(f"[fast] 역질의 {n_q}회차 ({q.dimension}) → "
                       f"{'무응답' if ans.value is None else ans.value}, H={h:.3f}")

        if h <= self.theta:
            interp = top(belief)
            route = "clarify" if n_q else "rule"
            log.append(f"[fast] H={h:.3f} ≤ θ={self.theta} → 확정: {interp}  (LLM 미호출)")
            ok = self._consistent(task, interp, log)
            return SemanticVerdict(interp, ok, route if ok else route + ":inconsistent",
                                   h0, h, n_q)

        # 4) LLM fallback
        if not self.use_llm:
            log.append(f"[fast] H={h:.3f} > θ 이고 LLM 비활성 → 의미 판단 실패")
            return SemanticVerdict(top(belief), False, "unresolved", h0, h, n_q)
        interp = self.judge.judge(task.spec, belief, principal.transcript)
        log.append(f"[fast] k={self.k} 소진, H={h:.3f} > θ → LLM 호출 → {interp}")
        ok = self._consistent(task, interp, log)
        return SemanticVerdict(interp, ok, "llm" if ok else "llm:inconsistent",
                               h0, h, n_q, n_llm=1)

    def _consistent(self, task, interp, log) -> bool:
        """경험과 정면으로 모순되는 해석은 Fast 가 확정하지 않는다 (opt-in, Fast 전용).

        H 만으로는 후보 집합이 조작됐는지 알 수 없다. 하지만 같은 유형의 위임이
        반복해서 X 로 확정돼 왔는데 갑자기 Y 를 확신한다면, 그 불일치 자체가
        신호다. ExperienceStore 에 이미 필요한 통계가 들어 있다.
        """
        if not self.use_consistency_check:
            return True
        return self._experience_conflict(task, interp, log, self.consistency_sigma,
                                          tag="fast") is None

    def _experience_conflict(self, task, interp, log, thr: float | None = None,
                             tag: str = "adaptive") -> str | None:
        """경험과 모순되면 그 사유를, 아니면 None 을 돌려준다.

        `_consistent` (Fast 의 opt-in 게이트) 와 `_adaptive` (Slow 에스컬레이션
        트리거) 가 공유하는 판정 로직이다. 임계치를 지정하지 않으면 self.sigma.
        """
        thr = self.sigma if thr is None else thr
        s = self.experience.score(task.key)
        best_ = self.experience.best(task.key)
        if best_ is None or s < thr or best_ == interp:
            return None
        reason = (f"경험 불일치 — 누적 {self.experience.n(task.key)}회는 "
                  f"{best_} 였는데 {interp} 를 확신함 (Score={s:.2f})")
        log.append(f"[{tag}] {reason}")
        return reason

    # ---- SEMANTIC FLOW (Slow) — 해석 전체를 A 에게 제시하고 승인받기 ------
    def _slow(self, task: DelegationTask, principal: "Principal", log: list[str],
              proposed: Interpretation | None = None, strict: bool = False):
        """B 가 정리한 해석을 A 가 검토한다.

        strict=True (AND 결합용) 이면 A 의 교정은 '불일치' 로 처리해 확정하지 않는다.
        오탐을 0 으로 유지하는 대신 미탐을 감내하는 보수적 결합이다.
        """
        prior = self.experience.prior(task.key) if self.use_experience else {}
        belief = build_belief(task.candidates, prior, self.experience_weight)
        h0 = entropy(belief)
        if proposed is None:
            proposed = top(belief)
        log.append(f"[slow] A 에게 해석 제시: {proposed}")

        review = principal.review(proposed)
        if review.approved:
            log.append("[slow] A 승인")
            return SemanticVerdict(proposed, True, "slow:approve", h0, h0, n_reviews=1)
        if review.status == "correct":
            if strict:
                log.append(f"[slow] A 교정 요구({review.interpretation}) → AND 결합에서는 불일치 처리")
                return SemanticVerdict(proposed, False, "slow:correct", h0, h0, n_reviews=1)
            log.append(f"[slow] A 교정: {review.interpretation}")
            return SemanticVerdict(review.interpretation, True, "slow:correct",
                                   h0, h0, n_reviews=1)
        log.append("[slow] A 도 확정하지 못함 → 의미 판단 실패")
        return SemanticVerdict(proposed, False, "slow:unsure", h0, h0, n_reviews=1)

    # ---- SEMANTIC FLOW (Adaptive) — 평소엔 Fast 만, 필요할 때만 Slow ----------
    def _adaptive(self, task: DelegationTask, principal: "Principal", log: list[str]):
        """README §7-1 이 다음 단계로 지목한 안 — Slow 를 매번이 아니라 선별적으로.

        AND 는 안전하지만 모든 위임에서 A 를 호출한다(검토율 1.00). Fast 단독은
        저렴하지만 belief 조작에 44.4% 가 뚫린다. 둘 사이의 신호로 쓸 수 있는 것이
        '경험과의 불일치' 다 — H 는 후보 집합만 오염시키면 위조되지만, 누적 이력은
        공격자가 손댈 수 없다(§실험③).

        트리거 조건 (둘 중 하나면 Slow 에스컬레이션):
          1. Fast 가 아예 확정하지 못함 (역질의도 LLM 도 답을 못 냄)
          2. Fast 의 확정 결과가 누적 경험과 정면으로 모순됨 — use_consistency_check
             설정과 무관하게 여기서는 항상 검사한다(이게 adaptive 의 핵심 판단 기준이므로).

        그 외(경험 자체로 판단했거나, 역질의/규칙으로 확정됐고 경험과 일치)에는
        Slow 를 생략한다 — 정상 운영에서는 review_rate 가 0 에 가까워야 한다.
        """
        fast = self._fast(task, principal, log)

        conflict = None
        if fast.confirmed:
            conflict = self._experience_conflict(task, fast.interpretation, log,
                                                  self.adaptive_sigma)
        escalate = (not fast.confirmed) or (conflict is not None)

        if not escalate:
            log.append(f"[adaptive] Fast 확정({fast.route}), 경험과 일치 → Slow 생략")
            return fast

        why = conflict or f"Fast 미확정({fast.route})"
        log.append(f"[adaptive] {why} → Slow 에스컬레이션")
        slow = self._slow(task, principal, log, proposed=fast.interpretation, strict=True)
        agreed = slow.confirmed
        log.append(f"[adaptive] Slow={'승인' if agreed else '불일치'} → "
                   f"{'통과' if agreed else '차단'}")
        return SemanticVerdict(
            fast.interpretation, agreed,
            f"adaptive({fast.route}+{slow.route.split(':')[-1]})",
            fast.h_initial, fast.h_final,
            fast.n_questions, fast.n_llm, slow.n_reviews)
