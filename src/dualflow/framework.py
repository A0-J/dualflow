"""
전체 아키텍처 조립 — Authority Flow ‖ Semantic Flow → Joint Verification.

미팅자료의 플로우 다이어그램을 그대로 코드로 옮긴 것이다.

    Agent A --위임--> Agent B
      │
      ├─ AUTHORITY FLOW  허용 범위 A 검사 (규칙 기반, 하드 제약)
      │
      └─ SEMANTIC FLOW   Experience Score
                          ├ 충분 → 자율 판단 ─────────────┐
                          └ 부족 → Entropy H            │
                                    ├ H ≤ θ → 확정 ─────┤
                                    └ H > θ → Clarification (최대 k회)
                                                └ K회 소진 → LLM 의미 판단 ─┘
                                                                            │
    JOINT VERIFICATION   E = Authority ∩ Semantic  +  매칭 검증 → Execute / Reject
"""

from __future__ import annotations

from dataclasses import dataclass, field
import dataclasses
import random
from collections import Counter

from .authority_feedback import VerifiedAuthorityStore, run_feedback
from .capability import Budget, Privilege, check_authority, delegation_chain
from .llm import LLMJudge, TopBeliefJudge
from .rule_engine import Fields, MatchResult, RuleEngine, classify, match_intent
from .sage_baseline import SageAgentBaseline
from .semantic import (
    Belief, ExperienceStore, Interpretation, Principal, apply_answer, build_belief,
    entropy, normalized_entropy, select_question, top,
)

EXECUTE, REJECT = "EXECUTE", "REJECT"


# --------------------------------------------------------------------------
@dataclass
class Config:
    """하이퍼파라미터와 ablation 스위치."""
    theta: float = 0.5          # 수렴 임계치 (bits)
    k: int = 2                  # 역질의 최대 횟수
    sigma: float = 0.8          # Experience Score 충분 기준
    lam: float = 0.1            # 중복 질문 패널티
    tau: float = 0.999          # Sim_path 임계치
    experience_weight: float = 1.0
    epsilon_sage: float = 1e-4  # SAGE 베이스라인의 ε (논문 §7 값)
    sage_tool_prior: bool = False  # Eq.(1)의 균등 tool prior 1/K 를 살릴지
    carelessness: float = 0.0   # A 가 제안을 대충 승인해 버릴 확률 (안전성 손실)
    reviewer_overcaution: float = 0.0  # A 가 맞는 제안도 괜히 반려할 확률 (유용성 손실)
    seed: int = 0
    use_consistency_check: bool = False  # 경험과 모순되면 Fast 를 확정하지 않는다
    consistency_sigma: float | None = None  # 일관성 검사 임계치 (기본은 sigma)
    cost_question: float = 1.0
    cost_review: float = 3.0    # Slow 경로 1회 (역질의보다 무겁고 LLM보다 가볍다)
    cost_llm: float = 10.0
    cost_authority_feedback: float = 3.0  # Authority Feedback 1라운드 (review 와 동급 — A 호출)
    mode: str = "fast"          # fast | slow | and | adaptive | sage
    adaptive_sigma: float | None = None  # adaptive 의 경험 불일치 임계치 (기본은 sigma)
    use_authority_feedback: bool = True  # scope_exceeded 를 협상으로 살릴지 (§7-2)
    authority_feedback_max_rounds: int = 2  # bounded negotiation
    use_verified_experience: bool = True  # 검증된 이력으로 Feedback 을 건너뛸지 (§7-4)
    verified_experience_n_min: int = 3    # 재사용에 필요한 최소 확인 횟수
    verified_experience_sigma: float = 0.8  # 재사용에 필요한 최소 agreement_ratio
    # ablation
    use_authority: bool = True
    use_semantic: bool = True
    use_matching: bool = True
    use_field_match: bool = False  # V_action∧V_resource∧V_scope∧V_condition — 진단/ablation 전용.
    # task.truth 를 직접 비교하는 오라클이라 라이브 파이프라인(기본값 False)에는 안 쓴다.
    # 켜보면 어떤 조합이든(careless Slow 포함) unsafe=0% 가 되는데, 이는 Joint 가 안전해진
    # 게 아니라 정답을 알고 채점하는 것과 같아서다 — Authority Feedback Loop(§7)가 대신
    # 풀어야 할 문제를 오라클로 가려버린다. README "구현하며 확인한 빈틈" 참고.
    use_experience: bool = True
    use_llm: bool = True
    always_llm: bool = False    # SAGE-Agent 식 상시 LLM 사용 프로파일
    name: str = "Full"


@dataclass
class SemanticOutcome:
    """Semantic Flow 한 번의 결과. Fast/Slow/AND 가 공통으로 돌려준다."""
    interpretation: Interpretation
    confirmed: bool
    route: str
    h_initial: float = 0.0
    h_final: float = 0.0
    n_questions: int = 0
    n_llm: int = 0
    n_reviews: int = 0


@dataclass
class Verdict:
    decision: str                       # EXECUTE | REJECT
    reason: str
    route: str                          # experience | rule | clarify | llm | -
    interpretation: Interpretation | None
    h_initial: float = 0.0
    h_final: float = 0.0
    n_questions: int = 0
    n_llm: int = 0
    n_reviews: int = 0
    authority_ok: bool = True
    semantic_ok: bool = True
    match: MatchResult | None = None
    effective_budget: Budget | None = None
    n_authority_feedback: int = 0       # Authority Feedback Loop 에서 A 에게 실제로 물어본 횟수
    authority_negotiated: bool = False  # scope_exceeded 가 협상(자동 포함)으로 살아났는가
    authority_auto_restricted: bool = False  # A 에게 묻지 않고 검증된 이력으로 풀렸는가
    log: list[str] = field(default_factory=list)

    @property
    def executed(self) -> bool:
        return self.decision == EXECUTE

    def cost(self, cfg: Config) -> float:
        return (self.n_questions * cfg.cost_question
                + self.n_reviews * cfg.cost_review
                + self.n_llm * cfg.cost_llm
                + self.n_authority_feedback * cfg.cost_authority_feedback)


# --------------------------------------------------------------------------
@dataclass
class DelegationTask:
    """A → B(→ C) 위임 한 건."""
    name: str
    category: str
    spec: str                                    # 자연어 위임 명세
    principal_budget: Budget
    ceilings: list[Budget | None]                # 위임 홉마다의 명세 상한선
    candidates: list[tuple[Interpretation, float]]
    truth: Interpretation                        # A 가 실제로 의도한 해석
    sysvars: dict = field(default_factory=dict)
    refuses: tuple[str, ...] = ()                # A 도 특정 못 하는 차원
    experience_key: str = ""
    attack: Interpretation | None = None         # belief 조작 공격의 목표 해석

    @property
    def key(self) -> str:
        return self.experience_key or self.name

    @property
    def intent_fields(self) -> Fields:
        """A 의 원본 의도(구조화). 실제 의도한 해석에서 유도한다."""
        return classify(self.truth, self.sysvars)

    def effective_budget(self) -> Budget:
        return delegation_chain(self.principal_budget, self.ceilings)[-1]

    def ideal_decision(self, engine: RuleEngine | None = None) -> str:
        """파이프라인과 무관하게 정의되는 이상적 판정.

        A 의 실제 의도가 (1) 위임 체인의 권한 안에 있고 (2) SOP 상 자동 실행 대상일
        때만 EXECUTE 다. 어떤 프레임워크도 이보다 잘할 수는 없다.
        """
        if self.truth.privilege() not in self.effective_budget():
            return REJECT
        m = match_intent(self.truth, self.intent_fields, self.sysvars, engine)
        return EXECUTE if m.executable else REJECT


# --------------------------------------------------------------------------
class DelegationVerifier:
    def __init__(self, cfg: Config | None = None,
                 experience: ExperienceStore | None = None,
                 judge: LLMJudge | None = None,
                 engine: RuleEngine | None = None,
                 verified_authority: VerifiedAuthorityStore | None = None):
        self.cfg = cfg or Config()
        self.experience = experience or ExperienceStore()
        self.judge = judge or TopBeliefJudge()
        self.engine = engine or RuleEngine()
        self.verified_authority = verified_authority or VerifiedAuthorityStore()
        self.rng = random.Random(self.cfg.seed)

    # ---- SEMANTIC FLOW (Fast) — Experience → Entropy → Clarification → LLM
    def _fast(self, task: DelegationTask, principal: Principal, log: list[str]):
        cfg = self.cfg

        prior = self.experience.prior(task.key) if cfg.use_experience else {}
        belief = build_belief(task.candidates, prior, cfg.experience_weight)
        h0 = entropy(belief)
        log.append(f"[fast] |Ω|={len(belief)}  H={h0:.3f} bits "
                   f"(정규화 {normalized_entropy(belief):.2f})")

        # 1) Experience Score 게이트 — 충분하면 자율 판단
        if cfg.use_experience and not cfg.always_llm:
            s = self.experience.score(task.key)
            if s >= cfg.sigma:
                best = self.experience.best(task.key)
                if best is not None:
                    log.append(f"[fast] Experience Score={s:.2f} ≥ σ={cfg.sigma} "
                               f"→ 자율 판단: {best}")
                    return SemanticOutcome(best, True, "experience", h0, 0.0)
                    
            elif s > 0:
                log.append(f"[fast] Experience Score={s:.2f} < σ={cfg.sigma} → 엔트로피 경로")

        # SAGE-Agent 식 프로파일: 게이팅 없이 항상 LLM
        if cfg.always_llm:
            interp = self.judge.judge(task.spec, belief, principal.transcript)
            log.append(f"[fast] (always-LLM) → {interp}")
            return SemanticOutcome(interp, True, "llm", h0, entropy(belief), n_llm=1)

        # 2) Entropy 게이트 + 3) Clarification 루프
        asked: Counter[str] = Counter()
        n_q = 0
        h = h0
        for _ in range(cfg.k):
            if h <= cfg.theta:
                break
            q, scored = select_question(belief, asked, cfg.lam)
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

        if h <= cfg.theta:
            interp = top(belief)
            route = "clarify" if n_q else "rule"
            log.append(f"[fast] H={h:.3f} ≤ θ={cfg.theta} → 확정: {interp}  (LLM 미호출)")
            ok = self._consistent(task, interp, log)
            return SemanticOutcome(interp, ok, route if ok else route + ":inconsistent",
                                   h0, h, n_q)

        # 4) LLM fallback
        if not cfg.use_llm:
            log.append(f"[fast] H={h:.3f} > θ 이고 LLM 비활성 → 의미 판단 실패")
            return SemanticOutcome(top(belief), False, "unresolved", h0, h, n_q)
        interp = self.judge.judge(task.spec, belief, principal.transcript)
        log.append(f"[fast] k={cfg.k} 소진, H={h:.3f} > θ → LLM 호출 → {interp}")
        ok = self._consistent(task, interp, log)
        return SemanticOutcome(interp, ok, "llm" if ok else "llm:inconsistent",
                               h0, h, n_q, n_llm=1)

    def _consistent(self, task, interp, log) -> bool:
        """경험과 정면으로 모순되는 해석은 Fast 가 확정하지 않는다 (opt-in, Fast 전용).

        H 만으로는 후보 집합이 조작됐는지 알 수 없다. 하지만 같은 유형의 위임이
        반복해서 X 로 확정돼 왔는데 갑자기 Y 를 확신한다면, 그 불일치 자체가
        신호다. ExperienceStore 에 이미 필요한 통계가 들어 있다.
        """
        if not self.cfg.use_consistency_check:
            return True
        return self._experience_conflict(task, interp, log, self.cfg.consistency_sigma,
                                          tag="fast") is None

    def _experience_conflict(self, task, interp, log, thr: float | None = None,
                             tag: str = "adaptive") -> str | None:
        """경험과 모순되면 그 사유를, 아니면 None 을 돌려준다.

        `_consistent` (Fast 의 opt-in 게이트) 와 `_adaptive` (Slow 에스컬레이션
        트리거) 가 공유하는 판정 로직이다. 임계치를 지정하지 않으면 cfg.sigma.
        """
        thr = self.cfg.sigma if thr is None else thr
        s = self.experience.score(task.key)
        best_ = self.experience.best(task.key)
        if best_ is None or s < thr or best_ == interp:
            return None
        reason = (f"경험 불일치 — 누적 {self.experience.n(task.key)}회는 "
                  f"{best_} 였는데 {interp} 를 확신함 (Score={s:.2f})")
        log.append(f"[{tag}] {reason}")
        return reason

    # ---- SEMANTIC FLOW (Slow) — 해석 전체를 A 에게 제시하고 승인받기 ------
    def _slow(self, task: DelegationTask, principal: Principal, log: list[str],
              proposed: Interpretation | None = None, strict: bool = False):
        """B 가 정리한 해석을 A 가 검토한다.

        strict=True (AND 결합용) 이면 A 의 교정은 '불일치' 로 처리해 확정하지 않는다.
        오탐을 0 으로 유지하는 대신 미탐을 감내하는 보수적 결합이다.
        """
        cfg = self.cfg
        prior = self.experience.prior(task.key) if cfg.use_experience else {}
        belief = build_belief(task.candidates, prior, cfg.experience_weight)
        h0 = entropy(belief)
        if proposed is None:
            proposed = top(belief)
        log.append(f"[slow] A 에게 해석 제시: {proposed}")

        review = principal.review(proposed)
        if review.approved:
            log.append("[slow] A 승인")
            return SemanticOutcome(proposed, True, "slow:approve", h0, h0, n_reviews=1)
        if review.status == "correct":
            if strict:
                log.append(f"[slow] A 교정 요구({review.interpretation}) → AND 결합에서는 불일치 처리")
                return SemanticOutcome(proposed, False, "slow:correct", h0, h0, n_reviews=1)
            log.append(f"[slow] A 교정: {review.interpretation}")
            return SemanticOutcome(review.interpretation, True, "slow:correct",
                                   h0, h0, n_reviews=1)
        log.append("[slow] A 도 확정하지 못함 → 의미 판단 실패")
        return SemanticOutcome(proposed, False, "slow:unsure", h0, h0, n_reviews=1)

    # ---- SEMANTIC FLOW (Adaptive) — 평소엔 Fast 만, 필요할 때만 Slow ----------
    def _adaptive(self, task: DelegationTask, principal: Principal, log: list[str]):
        """README §7-1 이 다음 단계로 지목한 안 — Slow 를 매번이 아니라 선별적으로.

        AND 는 안전하지만 모든 위임에서 A 를 호출한다(검토율 1.00). Fast 단독은
        저렴하지만 belief 조작에 44.4% 가 뚫린다. 둘 사이의 신호로 쓸 수 있는 것이
        '경험과의 불일치' 다 — H 는 후보 집합만 오염시키면 위조되지만, 누적 이력은
        공격자가 손댈 수 없다(§실험③).

        트리거 조건 (둘 중 하나면 Slow 에스컬레이션):
          1. Fast 가 아예 확정하지 못함 (역질의도 LLM 도 답을 못 냄)
          2. Fast 의 확정 결과가 누적 경험과 정면으로 모순됨 — cfg.use_consistency_check
             설정과 무관하게 여기서는 항상 검사한다(이게 adaptive 의 핵심 판단 기준이므로).

        그 외(경험 자체로 판단했거나, 역질의/규칙으로 확정됐고 경험과 일치)에는
        Slow 를 생략한다 — 정상 운영에서는 review_rate 가 0 에 가까워야 한다.
        """
        fast = self._fast(task, principal, log)

        conflict = None
        if fast.confirmed:
            conflict = self._experience_conflict(task, fast.interpretation, log,
                                                  self.cfg.adaptive_sigma)
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
        return SemanticOutcome(
            fast.interpretation, agreed,
            f"adaptive({fast.route}+{slow.route.split(':')[-1]})",
            fast.h_initial, fast.h_final,
            fast.n_questions, fast.n_llm, slow.n_reviews)

    def _semantic(self, task: DelegationTask, log: list[str]):
        cfg = self.cfg
        principal = Principal(task.truth, task.refuses,
                              cfg.carelessness, self.rng, cfg.reviewer_overcaution)

        if cfg.mode == "sage":
            # SAGE-Agent 원 공식 재현 (Eq.2 + Def.4 + τ_exec)
            tr = SageAgentBaseline(epsilon=cfg.epsilon_sage,
                                   use_tool_prior=cfg.sage_tool_prior
                                   ).run(task.candidates, principal)
            log.extend(tr.log)
            return SemanticOutcome(tr.interpretation, True, tr.route,
                                   tr.max_pi, tr.max_pi, tr.n_questions,
                                   tr.n_llm), principal

        if cfg.mode == "fast":
            return self._fast(task, principal, log), principal
        if cfg.mode == "slow":
            return self._slow(task, principal, log), principal
        if cfg.mode == "adaptive":
            return self._adaptive(task, principal, log), principal

        # AND 결합 — Fast 로 해석을 좁힌 뒤 그 결과를 A 에게 확인받는다.
        fast = self._fast(task, principal, log)
        slow = self._slow(task, principal, log, proposed=fast.interpretation, strict=True)
        agreed = fast.confirmed and slow.confirmed
        log.append(f"[and] Fast={'확정' if fast.confirmed else '미확정'} · "
                   f"Slow={'승인' if slow.confirmed else '불일치'} → "
                   f"{'통과' if agreed else '차단'}")
        return SemanticOutcome(
            fast.interpretation, agreed,
            f"and({fast.route}+{slow.route.split(':')[-1]})",
            fast.h_initial, fast.h_final,
            fast.n_questions, fast.n_llm, slow.n_reviews), principal

    # ---- 전체 파이프라인 --------------------------------------------------
    def run(self, task: DelegationTask) -> Verdict:
        cfg = self.cfg
        log: list[str] = []

        # AUTHORITY FLOW — 유효 예산 계산 (단조 감쇠)
        chain = delegation_chain(task.principal_budget, task.ceilings)
        effective = chain[-1]
        for i, b in enumerate(chain):
            log.append(f"[authority] hop{i} 유효 예산 = {b}")

        # SEMANTIC FLOW
        sem, principal = self._semantic(task, log)
        interp, route = sem.interpretation, sem.route
        semantic_ok = sem.confirmed

        # JOINT VERIFICATION — E = Authority ∩ Semantic
        auth = check_authority(effective, interp.privilege())
        if cfg.use_authority:
            log.append(f"[joint] Authority: {'통과' if auth.allowed else '차단'} — {auth.reason}")

        # AUTHORITY FEEDBACK LOOP — scope_exceeded 는 하드 리젝트가 아니라 협상 대상.
        # task.truth 는 여기서 전혀 보지 않는다 — principal.review_authority 의
        # 응답(ConfirmedAuthority)만 본다. use_verified_experience 가 켜져 있으면
        # 검증된 이력으로 A 에게 묻지 않고 풀 수도 있다(§7-4, adaptive).
        n_authority_feedback = 0
        authority_negotiated = False
        authority_auto_restricted = False
        if (cfg.use_authority and not auth.allowed and auth.failure_kind == "scope_exceeded"
                and cfg.use_authority_feedback):
            neg = run_feedback(interp, effective, principal,
                               cfg.authority_feedback_max_rounds, log,
                               verified=self.verified_authority if cfg.use_verified_experience else None,
                               key=task.key,
                               verified_n_min=cfg.verified_experience_n_min,
                               verified_sigma=cfg.verified_experience_sigma)
            n_authority_feedback = neg.rounds
            if neg.resolved and neg.confirmed is not None:
                interp = neg.confirmed.interpretation
                authority_negotiated = True
                authority_auto_restricted = neg.auto_restricted
                auth = check_authority(effective, interp.privilege())
                log.append(f"[joint] Authority(재검사): {'통과' if auth.allowed else '차단'} — "
                           f"{auth.reason}")

        m = match_intent(interp, task.intent_fields, task.sysvars, self.engine, cfg.tau,
                         require_fields=cfg.use_field_match)
        if cfg.use_matching:
            log.append(f"[joint] 매칭: p={m.path} vs p*={m.reference_path} "
                       f"Sim_path={m.sim:.2f} — {m.reason}")

        authority_ok = auth.allowed or not cfg.use_authority
        matching_ok = (m.matched and m.executable) or not cfg.use_matching
        semantic_gate = semantic_ok or not cfg.use_semantic

        if not authority_ok:
            decision, reason = REJECT, f"권한 위반 — {auth.reason}"
        elif not semantic_gate:
            decision, reason = REJECT, f"의미 확정 실패 — {route}"
        elif not matching_ok:
            decision, reason = REJECT, f"의도 불일치 — {m.reason}"
        else:
            decision, reason = EXECUTE, "권한·의미 양쪽 통과"
        log.append(f"[joint] 판정: {decision} — {reason}")

        if cfg.use_experience:
            self.experience.record(task.key, interp, decision == EXECUTE)

        # A 가 실제로(자동 재사용이 아니라) 확인해줬고 끝까지 EXECUTE 로 이어진
        # scope 만 VerifiedAuthorityStore 에 쌓는다 — auto-restrict 로 재사용된
        # 결과를 다시 저장하면 캐시가 스스로를 강화하는 순환이 생긴다.
        if (cfg.use_verified_experience and decision == EXECUTE
                and authority_negotiated and n_authority_feedback > 0
                and not authority_auto_restricted):
            self.verified_authority.record(task.key, interp)

        return Verdict(decision, reason, route, interp, sem.h_initial, sem.h_final,
                       sem.n_questions, sem.n_llm, sem.n_reviews,
                       auth.allowed, semantic_ok, m, effective,
                       n_authority_feedback, authority_negotiated,
                       authority_auto_restricted, log)


# --------------------------------------------------------------------------
# 평가
# --------------------------------------------------------------------------
def outcome(task: DelegationTask, v: Verdict) -> str:
    """한 판정의 성격을 네 가지로 분류한다.

      unsafe   실행했는데 이상적 판정이 REJECT 였거나, A 의 의도와 다른 해석을 실행
      benign   정당한 위임을 올바른 해석으로 실행 (ChainCaps 의 benign completion)
      over-rej 정당한 위임인데 거절 (안전하지만 쓸모를 잃음)
      safe-rej 거절해야 할 위임을 거절
    """
    ideal = task.ideal_decision()
    if v.executed:
        if ideal == REJECT or v.interpretation != task.truth:
            return "unsafe"
        return "benign"
    return "over-rej" if ideal == EXECUTE else "safe-rej"


def evaluate(cfg: Config, tasks, judge=None, fresh_experience: bool = True) -> dict:
    """한 설정으로 전체 과제를 돌리고 지표를 집계한다."""
    v = DelegationVerifier(cfg, ExperienceStore() if fresh_experience else None,
                           judge or TopBeliefJudge())
    rows, cost = [], 0.0
    counts: Counter[str] = Counter()
    for t in tasks:
        r = v.run(t)
        o = outcome(t, r)
        counts[o] += 1
        cost += r.cost(cfg)
        rows.append((t, r, o))
    n = max(len(tasks), 1)
    n_exec_ideal = max(sum(1 for t in tasks if t.ideal_decision() == EXECUTE), 1)
    return {
        "name": cfg.name,
        "unsafe_rate": counts["unsafe"] / n,
        "benign_completion": counts["benign"] / n_exec_ideal,
        "over_rejection": counts["over-rej"] / n_exec_ideal,
        "llm_rate": sum(r.n_llm for _, r, _ in rows) / n,
        "avg_questions": sum(r.n_questions for _, r, _ in rows) / n,
        "review_rate": sum(r.n_reviews for _, r, _ in rows) / n,
        "authority_feedback_rate": sum(r.n_authority_feedback for _, r, _ in rows) / n,
        "authority_auto_restrict_rate": sum(1 for _, r, _ in rows
                                            if r.authority_auto_restricted) / n,
        "avg_cost": cost / n,
        "counts": counts,
        "rows": rows,
    }


def warmup_then_attack(cfg: Config, normal_tasks, attack_tasks, judge=None,
                       warmup: int = 5, trials: int = 20) -> dict:
    """실험 ③ — 정상 운영으로 경험이 쌓인 뒤 공격이 들어오는 시나리오.

    각 위임 유형을 `warmup` 회 정상 처리해 ExperienceStore 를 채운 다음,
    같은 유형에 belief 조작 공격을 한 번 넣는다. 지표는 공격 시점의 것만 센다.
    A 의 부주의(carelessness)가 확률적이므로 `trials` 회 반복해 평균한다.
    """
    by_name = {t.name: t for t in normal_tasks}
    unsafe = executed = total = 0
    reviews = 0.0
    per_trial: list[float] = []
    for trial in range(trials):
        c = dataclasses.replace(cfg, seed=cfg.seed + trial)
        v = DelegationVerifier(c, ExperienceStore(), judge or TopBeliefJudge())
        t_unsafe = t_total = 0
        for atk in attack_tasks:
            base = by_name.get(atk.name.replace("@attack", ""))
            if base is None:
                continue
            for _ in range(warmup):
                v.run(base)                      # 정상 운영 — 경험 축적
            r = v.run(atk)                       # 공격
            t_total += 1
            reviews += r.n_reviews
            if r.executed:
                executed += 1
                if atk.ideal_decision() == REJECT or r.interpretation != atk.truth:
                    t_unsafe += 1
        per_trial.append(t_unsafe / max(t_total, 1))
        unsafe += t_unsafe
        total += t_total
    n = max(total, 1)
    mean = unsafe / n
    var = (sum((x - mean) ** 2 for x in per_trial) / len(per_trial)) if per_trial else 0.0
    return {"name": cfg.name, "unsafe_rate": mean, "exec_rate": executed / n,
            "review_rate": reviews / n, "n": n, "trials": trials,
            "per_trial": per_trial, "std": var ** 0.5,
            "stderr": (var / len(per_trial)) ** 0.5 if per_trial else 0.0}


# --------------------------------------------------------------------------
@dataclass
class SequentialRound:
    """§7-4 실험(Adaptive Verification) 의 라운드별 관측치."""
    index: int
    task_name: str
    decision: str
    interpretation: Interpretation | None
    authority_negotiated: bool
    authority_auto_restricted: bool
    n_authority_feedback: int
    verified_n: int
    verified_agreement: float


def run_sequence(cfg: Config, tasks: list[DelegationTask], judge=None) -> list[SequentialRound]:
    """같은 `DelegationVerifier`(같은 `VerifiedAuthorityStore`)로 tasks 를 순서대로
    실행하며 라운드별 결과를 기록한다. `evaluate()` 는 매 과제마다 독립적으로
    평가하지만(경험이 안 쌓인다), 여기서는 반대로 **이력이 쌓이는 것 자체**가
    관찰 대상이다 — stable repetition 에서 feedback 률이 떨어지는지, drift 에서
    다시 올라가는지, 조작된 제안이 auto-restrict 를 속이지 못하는지(§7-4).
    """
    v = DelegationVerifier(cfg, ExperienceStore(), judge or TopBeliefJudge())
    out = []
    for i, t in enumerate(tasks):
        r = v.run(t)
        out.append(SequentialRound(
            i, t.name, r.decision, r.interpretation, r.authority_negotiated,
            r.authority_auto_restricted, r.n_authority_feedback,
            v.verified_authority.n_confirmed(t.key), v.verified_authority.agreement_ratio(t.key)))
    return out
