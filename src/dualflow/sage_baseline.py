"""
SAGE-Agent 원 공식 재현 — 비교 베이스라인.

arXiv 2511.08798 (Findings of ACL 2026) 의 Algorithm 1 과 Eq.(2), Def.4, Def.5,
Eq.(13) 을 논문에 적힌 그대로 옮긴다. 우리 엔트로피 근사가 아니라 **논문의 π_c 와
EVPI 위에서** 공격이 성립하는지 확인하기 위한 것이다.

논문 대응
    Eq.(2)/(13)  π_c = Π_j p(θ_j),  지정=1 / 유한 미지정=1/|D| / 무한=ε   → pi()
    Def.4        EVPI(q,B) = E_r[max_c π_c(t|q,r)] − max_c π_c(t)         → evpi()
    Def.5        Cost(q,t) = λ Σ_{a∈A(q)} n_a(t)                          → cost()
    Step 1       max_c π_c(t) ≥ τ_exec 이면 즉시 실행하고 종료
    Step 3       Score = EVPI − Cost, max Score < α·max π 이면 실행
    Step 4       응답을 파라미터 도메인 제약으로 반영

구현상 두 가지는 논문의 서술을 따랐다.
  · π 는 **정규화하지 않은 곱**이다. Eq.(2)는 '∝' 로 적혀 있지만 Prop.1(완전성:
    π=1 iff 모든 인자 지정)과 Eq.(12-13)의 보상 정의는 비정규화 곱이라야 성립한다.
  · EVPI 는 Step 3 의 계산 단축(미지정 인자에 |D_a| 를 곱해 완전해소를 모사)을 쓴다.

핵심: **Step 1 의 τ_exec 검사가 EVPI 보다 앞에 있다.** π 가 τ_exec 를 넘기면
질문 생성도 EVPI 계산도 일어나지 않는다. 이것이 belief 조작의 공격 지점이다.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .framework import DelegationTask
from .semantic import DIMENSIONS, Interpretation, Principal, Question, SemanticVerdict

UNK = "<UNK>"
PARAMS = ("resource", "scope", "condition")     # tool = action, 나머지가 파라미터


# --------------------------------------------------------------------------
# Definition 2 — tool call candidate
# --------------------------------------------------------------------------
@dataclass
class SageCandidate:
    tool: str                                   # 논문의 T_i
    args: dict[str, object]                     # 값 또는 UNK
    domains: dict[str, tuple]                   # D_{i,j}

    def is_unknown(self, param: str) -> bool:
        return self.args.get(param, UNK) is UNK

    def to_interpretation(self) -> Interpretation:
        """미지정 인자는 기본값을 가정한다 — 논문 §B.3 의 지시와 같다."""
        vals = {}
        for p in PARAMS:
            v = self.args[p]
            if v is UNK:
                dom = self.domains.get(p, ())
                v = dom[0] if dom else ("" if p != "condition" else ())
            vals[p] = v
        return Interpretation(self.tool, vals["resource"], vals["scope"],
                              frozenset(vals["condition"]))


def build_candidates(candidates: Sequence[tuple[Interpretation, float]]
                     ) -> list[SageCandidate]:
    """해석 후보 집합 → 논문식 tool call 후보.

    reasoner 는 tool 마다 하나의 호출을 내놓고, 명세로 결정되지 않는 파라미터
    (= 후보 전체에서 값이 갈리는 파라미터)를 <UNK> 로 표시한다.
    """
    interps = [i for i, _ in candidates]
    varying = {p for p in PARAMS
               if len({i.value_of(p) for i in interps}) > 1}
    domains = {p: tuple(sorted({i.value_of(p) for i in interps}, key=str))
               for p in PARAMS}

    out: dict[str, SageCandidate] = {}
    for i in interps:
        if i.action in out:
            continue
        args = {p: (UNK if p in varying else i.value_of(p)) for p in PARAMS}
        out[i.action] = SageCandidate(i.action, args, dict(domains))
    return list(out.values())


# --------------------------------------------------------------------------
# Eq. (2) / Eq. (13) — viability score
# --------------------------------------------------------------------------
def pi(c: SageCandidate, epsilon: float = 1e-4, tool_prior: float = 1.0) -> float:
    """tool_prior=1/K 를 곱하면 Eq.(1)의 균등 tool prior 를 살린 해석이 된다.

    논문 본문은 Eq.(2)에서 '∝' 로 1/K 를 흡수하고, Prop.1(π=1 iff 모든 인자 지정)도
    파라미터 곱만을 가정한다. 그래서 기본값은 tool_prior=1.0 이다. 다만 이 해석에서는
    **서로 배타적인 두 tool 이 모두 완전 지정이면 π=1 이 되어 τ_exec 를 통과한다** —
    tool 선택의 불확실성이 실행 게이트에 전혀 반영되지 않는다. 두 해석을 모두
    돌려볼 수 있게 인자로 남겨 둔다.
    """
    w = tool_prior
    for p in PARAMS:
        if not c.is_unknown(p):
            continue                            # 지정된 인자는 p = 1
        dom = c.domains.get(p, ())
        if not dom:
            w *= epsilon                        # 무한/연속 도메인
        else:
            w *= 1.0 / len(dom)
    return w


def max_pi(cands: Iterable[SageCandidate], epsilon: float = 1e-4,
           tool_prior: float = 1.0) -> float:
    return max((pi(c, epsilon, tool_prior) for c in cands), default=0.0)


def best(cands: Sequence[SageCandidate], epsilon: float = 1e-4) -> SageCandidate:
    return max(cands, key=lambda c: (pi(c, epsilon), c.tool))


# --------------------------------------------------------------------------
# Definition 4 — EVPI (Step 3 의 계산 단축)
# --------------------------------------------------------------------------
def evpi(cands: Sequence[SageCandidate], param: str, epsilon: float = 1e-4,
         tool_prior: float = 1.0) -> float:
    prior = max_pi(cands, epsilon, tool_prior)
    resolved = []
    for c in cands:
        w = pi(c, epsilon, tool_prior)
        if c.is_unknown(param):
            dom = c.domains.get(param, ())
            w *= len(dom) if dom else 1.0 / epsilon
        resolved.append(w)
    return max(resolved, default=0.0) - prior


def cost(param: str, asked: Counter, lam: float = 0.5) -> float:
    """Definition 5 — 같은 aspect 를 다시 물으면 페널티."""
    return lam * asked[param]


# --------------------------------------------------------------------------
# Algorithm 1
# --------------------------------------------------------------------------
@dataclass
class SageTrace:
    interpretation: Interpretation
    route: str
    n_questions: int = 0
    n_llm: int = 0
    max_pi: float = 0.0
    evpi_computed: bool = False
    log: list[str] = field(default_factory=list)


class SageAgentBaseline:
    """논문 Algorithm 1. τ_exec, α, λ, ε 는 §7 의 값을 그대로 쓴다."""

    def __init__(self, tau_exec: float = 0.999, alpha: float = 0.1,
                 lam: float = 0.5, epsilon: float = 1e-4, max_steps: int = 5,
                 use_tool_prior: bool = False):
        self.tau_exec, self.alpha = tau_exec, alpha
        self.lam, self.epsilon, self.max_steps = lam, epsilon, max_steps
        self.use_tool_prior = use_tool_prior

    def run(self, candidates, principal: Principal) -> SageTrace:
        cands = build_candidates(candidates)
        prior = 1.0 / len(cands) if (self.use_tool_prior and cands) else 1.0
        asked: Counter[str] = Counter()
        log = [f"[sage] 후보 {len(cands)}개"
               + (f", tool prior 1/K={prior:.3f}" if self.use_tool_prior else "")
               + f", max π = {max_pi(cands, self.epsilon, prior):.4f}"]
        n_q, n_llm, evpi_done = 0, 1, False       # 후보 생성에 LLM 1회

        for _ in range(self.max_steps):
            m = max_pi(cands, self.epsilon, prior)

            # Step 1 — τ_exec 검사가 EVPI 보다 먼저다
            if m >= self.tau_exec:
                log.append(f"[sage] max π={m:.4f} ≥ τ_exec={self.tau_exec} "
                           f"→ 즉시 실행 (질문 생성·EVPI 계산 없음)")
                c = best(cands, self.epsilon)
                return SageTrace(c.to_interpretation(), "sage:tau_exec",
                                 n_q, n_llm, m, evpi_done, log)

            # Step 2 — 질문 생성 (LLM)
            n_llm += 1
            targets = [p for p in PARAMS if any(c.is_unknown(p) for c in cands)]
            targets += [p for p in PARAMS if p not in targets]   # 이미 확정된 것도 후보로
            if not targets:
                break

            # Step 3 — EVPI − Cost 로 점수화하고 정지 조건 확인
            evpi_done = True
            scored = sorted(((p, evpi(cands, p, self.epsilon, prior),
                              cost(p, asked, self.lam)) for p in targets),
                            key=lambda r: -(r[1] - r[2]))
            for p, v, cst in scored[:3]:
                log.append(f"         EVPI={v:+.4f} cost={cst:.2f} "
                           f"score={v - cst:+.4f} | {p}")
            param, v, cst = scored[0]
            if v - cst < self.alpha * m:
                log.append(f"[sage] max Score={v - cst:+.4f} < α·max π="
                           f"{self.alpha * m:.4f} → 실행")
                c = best(cands, self.epsilon)
                return SageTrace(c.to_interpretation(), "sage:stop",
                                 n_q, n_llm, m, evpi_done, log)

            # Step 4 — 응답을 도메인 제약으로 반영
            n_q += 1
            asked[param] += 1
            ans = principal.answer(Question(f"{param} 을(를) 알려주세요", param))
            log.append(f"[sage] 질문 {n_q}회차 ({param}) → "
                       f"{'무응답' if ans.value is None else ans.value}")
            if ans.value is not None:
                for c in cands:
                    c.domains[param] = (ans.value,)
                    if c.is_unknown(param):
                        c.args[param] = ans.value
                cands = [c for c in cands if c.args[param] == ans.value] or cands
                if self.use_tool_prior:
                    prior = 1.0 / len(cands)

        c = best(cands, self.epsilon)
        log.append(f"[sage] 최대 스텝 소진 → 실행")
        return SageTrace(c.to_interpretation(), "sage:maxsteps", n_q, n_llm,
                         max_pi(cands, self.epsilon, prior), evpi_done, log)


# --------------------------------------------------------------------------
# framework.Config.semantic_engine 어댑터 — Authority Flow/Joint Verification은
# DualFlow 그대로 두고 semantic 단계만 SAGE-Agent 원 알고리즘으로 대체하고 싶을
# 때 쓴다("SAGE + Joint" 비교). framework.py는 이 함수의 존재를 모른다 — 반대로
# 이 모듈이 semantic.SemanticVerdict 에 맞춰 결과를 포장한다.
# --------------------------------------------------------------------------
def as_semantic_engine(epsilon: float = 1e-4, use_tool_prior: bool = False
                       ) -> Callable[[DelegationTask, Principal, list[str]], SemanticVerdict]:
    baseline = SageAgentBaseline(epsilon=epsilon, use_tool_prior=use_tool_prior)

    def _engine(task: DelegationTask, principal: Principal, log: list[str]) -> SemanticVerdict:
        tr = baseline.run(task.candidates, principal)
        log.extend(tr.log)
        return SemanticVerdict(tr.interpretation, True, tr.route,
                               tr.max_pi, tr.max_pi, tr.n_questions, tr.n_llm)

    return _engine
