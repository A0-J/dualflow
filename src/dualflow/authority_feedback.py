"""
AUTHORITY FEEDBACK LOOP — 권한 범위 협상.

Authority Flow 가 요청을 막았을 때, 그 이유가 *유무*(no_grant) 문제면 재협상
불가능한 하드 리젝트지만, *범위*(scope_exceeded) 문제면 B 가 제안한 scope 를
A 에게 확인/축소받아 재실행할 수 있다:

    B proposes execution (action, resource, scope, condition)
                    │
              Authority Check
        ┌───────────┴────────────┐
        │                        │
   유무 실패(no_grant) /     scope 초과(scope_exceeded)
   조건 미충족               (범위만 넘음 — 협상 가능)
        │                        │
      REJECT              Authority Feedback (A 에게 확인)
                       ┌─────┬─────────┬────────┐
                    APPROVE CORRECT  RESTRICT  REJECT
                       │     │         │         │
                       └──┬──┴────┬────┘         │
                    non-amplification 검사        │
                    (확정 결과가 위임 예산 밖이면  │
                     APPROVE 라도 거부)            │
                          │                        │
                    ConfirmedAuthority         REJECT
                          │
                    (재실행 → Joint Verification)

핵심 경계: `task.truth` 는 `Principal` 안에서만 쓰인다 — A 의 응답을 시뮬레이션
하는 오라클로서만 쓰이고, 이 파일의 협상 로직(`run_feedback`)이나 framework.py 는
`task.truth` 를 직접 들여다보지 않는다. `run_feedback` 이 보는 것은 Principal 의
*응답*(`AuthorityFeedback`)뿐이다 — §7-2 에서 확인한 "task.truth 를 직접 비교하는
채점기" 문제를 반복하지 않기 위한 설계다. 최종적으로 Joint Verification 이 비교하는
기준도 `task.truth` 가 아니라 이 협상으로 확정된 `ConfirmedAuthority` 다.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from .capability import Budget, Privilege, check_authority
from .semantic import Interpretation, parse_structured_action

if TYPE_CHECKING:  # 순환 참조 방지 — 타입 힌트용으로만, 런타임에는 import 안 됨.
    from .llm import LLMClient, LLMResponse


class FeedbackDecision(Enum):
    APPROVE = "approve"     # 제안이 이미(또는 그대로) 맞음
    CORRECT = "correct"     # A 가 정확한 값으로 고쳐줌 — suggested 보다 더 좁을 수 있음
    RESTRICT = "restrict"   # 제안된 상한(suggested)까지만 축소하면 충분
    REJECT = "reject"       # A 가 원하는 것 자체가 이 범위 밖이거나, A 가 확인을 거부


@dataclass(frozen=True)
class AuthorityFeedback:
    decision: FeedbackDecision
    confirmed: Interpretation | None = None   # APPROVE/CORRECT/RESTRICT 일 때만 값이 있음


class AuthorityPrincipal(Protocol):
    """`semantic.Principal` 이 구현하는 인터페이스. 여기서는 타입 힌트 용도일 뿐,
    이 모듈은 concrete Principal 의 내부(task.truth)에 접근하지 않는다."""
    def review_authority(self, proposed: Interpretation,
                         suggested: Interpretation) -> AuthorityFeedback: ...


@dataclass(frozen=True)
class ConfirmedAuthority:
    """협상으로 확정된 권한 — Joint Verification 이 비교할 기준.

    `task.truth` 가 아니라 이 값을 쓴다: Principal 의 실제 응답을 거쳐 나온
    값이므로, 오라클이 아니라 "A 가 실제로 확인해준 것"이라는 근거가 있다.
    """
    interpretation: Interpretation
    rounds: int
    decision: FeedbackDecision


@dataclass
class NegotiationResult:
    resolved: bool
    confirmed: ConfirmedAuthority | None
    rounds: int                          # A 에게 실제로 물어본 횟수 (§ run_feedback 참고)
    decision: FeedbackDecision
    log: list[str] = field(default_factory=list)
    auto_restricted: bool = False        # A 에게 묻지 않고 VerifiedAuthorityStore 로 해결됐는가


# --------------------------------------------------------------------------
# ADAPTIVE VERIFICATION — "언제 A 에게 다시 물어볼 것인가"
# --------------------------------------------------------------------------
@dataclass
class VerifiedAuthorityStore:
    """Principal.review_authority 로 A 가 실제로 확인해주고, 재검증까지 통과해
    최종 EXECUTE 로 이어진 scope 만 저장한다.

    무슨 실행이든 담는 `semantic.ExperienceStore` 와 달리, 여기 들어가는 값은
    전부 "A 가 이 범위를 직접 확인해줬다" 는 근거가 있는 것뿐이다 — auto-restrict
    로 재사용된 결과는 기록하지 않는다(그러면 캐시가 스스로를 강화하는 순환이
    생긴다). `framework.DelegationVerifier` 가 EXECUTE 확정 후에만 record 한다.
    """
    _counts: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    _index: dict[str, dict[str, Interpretation]] = field(default_factory=lambda: defaultdict(dict))

    def record(self, key: str, interp: Interpretation) -> None:
        """A 가 실제로 확인해준 값을 쌓는다.

        새로 확인된 값이 지금까지의 이력과 다르면(=A 가 이번엔 다른 scope 를
        확인해줬다) 낡은 이력을 버리고 새로 시작한다. 이게 없으면 위임 범위가
        영구히 바뀐 뒤에도(intent drift) 예전 값의 개수가 새 값을 계속 압도해서
        agreement_ratio 가 오래도록 회복되지 않는다 — reset 은 "복잡한 risk
        score" 대신 쓸 수 있는 가장 단순한 해석 가능한 규칙이다.
        """
        ik = str(interp.privilege())
        counts = self._counts[key]
        if counts and ik not in counts:
            counts.clear()
            self._index[key].clear()
        counts[ik] += 1
        self._index[key][ik] = interp

    def n_confirmed(self, key: str) -> int:
        return sum(self._counts.get(key, Counter()).values())

    def agreement_ratio(self, key: str) -> float:
        """가장 많이 확인된 scope 가 전체 확인 이력에서 차지하는 비율."""
        c = self._counts.get(key)
        if not c:
            return 0.0
        return max(c.values()) / sum(c.values())

    def best(self, key: str) -> Interpretation | None:
        c = self._counts.get(key)
        if not c:
            return None
        ik = max(c, key=lambda k: (c[k], k))
        return self._index[key][ik]


def run_feedback(proposal: Interpretation, effective: Budget, principal: AuthorityPrincipal,
                 max_rounds: int = 2, log: list[str] | None = None,
                 verified: VerifiedAuthorityStore | None = None, key: str | None = None,
                 verified_n_min: int = 3, verified_sigma: float = 0.8) -> NegotiationResult:
    """Bounded authority negotiation. scope 초과를 만날 때마다:

    1. `verified`/`key` 가 주어졌고(adaptive 모드) 아직 시도 안 했으면, 검증된
       이력이 충분한지(n_confirmed≥verified_n_min, agreement≥verified_sigma) 본다.
       충분하면 그 값과 **현재** 허용 상한(auth.suggested)의 교집합
       (`Privilege.meet` — 절대 그대로 신뢰하지 않고 항상 현재 예산과 다시
       교집합한다: C_adaptive = C_experience ∩ C_current_budget)을 시도하고,
       그 결과를 A 에게 묻지 않고 곧장 재검증한다. 겹치지 않으면(오래된 경험이
       지금 상황과 안 맞음 — 위임 범위가 바뀐 drift) 그냥 실제 Feedback 으로
       넘어간다. 이 자동 재사용은 협상당 최대 1회만 시도한다.
    2. 그래도 안 풀리면(또는 애초에 이력이 부족하면) `principal.review_authority`
       로 A 에게 실제로 물어본다 — 최대 `max_rounds` 회.

    매 확정 전에는 항상 top 의 `check_authority` 로 다시 검증한다 — A 의 응답도,
    재사용한 경험도 그 자체로는 최종 권한이 아니고 위임 예산이 최종 상한이라는
    뜻이다(non-amplification, §6). 이게 auto-restrict 를 켜도 안전한 이유다:
    경험은 "무엇을 시도해볼지" 를 줄여줄 뿐 "허용되는지" 를 대신 판단하지 않는다.

    `NegotiationResult.rounds` 는 A 에게 **실제로** 물어본 횟수다 — auto-restrict
    로 풀리면 0, 애초에 협상 불가로 즉시 거부돼도 0 이다.
    """
    log = log if log is not None else []
    current = proposal
    last_decision = FeedbackDecision.APPROVE
    asked = 0
    tried_auto = False
    auto_restricted = False

    for _ in range(max_rounds + 2):   # 실제 문의는 asked<max_rounds 로 따로 제한한다
        auth = check_authority(effective, current.privilege())
        if auth.allowed:
            tag = "auto-restrict" if auto_restricted else f"{asked}회 문의"
            log.append(f"[authority-feedback] 확정({tag}): {current}")
            return NegotiationResult(True, ConfirmedAuthority(current, asked, last_decision),
                                     asked, last_decision, log, auto_restricted)

        if auth.failure_kind != "scope_exceeded" or auth.suggested is None:
            log.append(f"[authority-feedback] 협상 불가({auth.failure_kind}) — A 에게 묻지 않고 거부: "
                       f"{auth.reason}")
            return NegotiationResult(False, None, asked, FeedbackDecision.REJECT, log)

        if verified is not None and key is not None and not tried_auto:
            tried_auto = True
            cand = verified.best(key)
            n, agree = verified.n_confirmed(key), verified.agreement_ratio(key)
            if (cand is not None and n >= verified_n_min and agree >= verified_sigma
                    and cand.action == auth.suggested.action
                    and cand.resource == auth.suggested.resource):
                intersected = cand.privilege().meet(auth.suggested)
                if intersected is not None:
                    current = Interpretation(intersected.action, intersected.resource,
                                             intersected.scope, intersected.condition)
                    auto_restricted = True
                    log.append(f"[authority-feedback] 검증된 이력 재사용(n={n}, "
                               f"agreement={agree:.2f}) ∩ 현재 상한 → {current} "
                               f"(A 에게 묻지 않음)")
                    continue
                log.append(f"[authority-feedback] 검증된 이력({cand})이 현재 허용 상한과 "
                           f"안 겹침(drift) → Feedback 으로 진행")

        if asked >= max_rounds:
            log.append(f"[authority-feedback] {max_rounds}회 문의 소진 → 미해결")
            return NegotiationResult(False, None, asked, FeedbackDecision.REJECT, log)

        # label 은 일부러 비워 둔다 — current(원래 넓은 제안)의 label 을 물려받으면
        # RESTRICT 로 좁아진 뒤에도 옛 설명이 남아 로그가 헷갈린다. 비우면
        # Interpretation.__str__ 이 privilege 문자열로 대체해 정확한 값을 보여준다.
        suggested = Interpretation(auth.suggested.action, auth.suggested.resource,
                                   auth.suggested.scope, auth.suggested.condition)
        fb = principal.review_authority(current, suggested)
        asked += 1
        last_decision = fb.decision
        detail = f" → {fb.confirmed}" if fb.confirmed is not None else ""
        log.append(f"[authority-feedback] {asked}회차 문의 — 제안 {current} "
                   f"(허용 상한 {suggested}) → A: {fb.decision.value}{detail}")

        if fb.decision == FeedbackDecision.REJECT or fb.confirmed is None:
            return NegotiationResult(False, None, asked, FeedbackDecision.REJECT, log)

        current = fb.confirmed
        auto_restricted = False   # 실제 Feedback 이 일어났으니 다음 확정은 auto 가 아니다
        # 다음 반복으로 — top 의 check_authority 가 current 를 다시 검증한다.

    log.append(f"[authority-feedback] 반복 한도 소진 → 미해결")
    return NegotiationResult(False, None, asked, FeedbackDecision.REJECT, log)


# --------------------------------------------------------------------------
# AUTHORITY VERIFIER — Authority Flow를 독립적으로 호출 가능한 단위로 묶는다.
#
# B의 제안이 위임된 권한 범위 안에 있는지만 판단한다 — 그게 A의 의도와 맞는지는
# 모른다(Semantic Flow의 책임, semantic.SemanticVerifierAgent). check_authority()
# (capability.py, 결정론적 규칙)와 run_feedback()(이 파일, bounded negotiation)를
# 그대로 호출한다 — 알고리즘은 추출 이전과 동일하고, 옮긴 것은 "이 판단이 어디서
# 내려지는지"뿐이다.
# --------------------------------------------------------------------------
@dataclass
class AuthorityVerdict:
    """Authority Flow 한 번의 결과.

    `interpretation`은 협상으로 수정됐다면 그 최종본이다 — 원래 제안이 아니다.
    Joint Verification의 매칭은 이 값을 기준으로 한다.
    """
    interpretation: Interpretation
    allowed: bool
    reason: str
    n_feedback: int = 0            # A 에게 실제로 물어본 횟수(controlled 협상)
    negotiated: bool = False       # scope_exceeded 가 협상(자동 포함)으로 살아났는가
    auto_restricted: bool = False  # A 에게 묻지 않고 검증된 이력으로 풀렸는가
    # Phase B 전용 — model-backed 경로(verify_agent_proposal)에서만 채워진다.
    # controlled verify() 경로는 이 필드들을 설정하지 않으므로 항상 기본값이다
    # — 하위 호환을 깨지 않는 순수 추가 필드다.
    n_llm: int = 0                          # 실제로 호출한 LLM 횟수
    response: "LLMResponse | None" = None   # 토큰/latency 회계용

    @property
    def status(self) -> str:
        """SemanticVerdict.status(resolved/unresolved) 와의 서술적 대구 —
        저장된 새 상태가 아니라 allowed 의 별칭이다."""
        return "allow" if self.allowed else "deny"


class AuthorityVerifierAgent:
    """Authority Flow, 독립 단위. "B 의 제안이 위임된 권한 범위 안인가" 만
    판단한다 — 그게 A 의 의도와 맞는지는 모른다(Authority Verification 이
    아니라 Joint Verification/Semantic Flow 의 책임).

    scope_exceeded(범위만 넘음)는 하드 리젝트가 아니라 bounded negotiation
    (`run_feedback`)으로 회복을 시도한다. no_grant/condition_missing 은
    협상 대상이 아니다(capability.check_authority 참고).

    Phase B(`research/agent-connected-eval`)에서 두 번째 진입점
    `verify_agent_proposal()`을 opt-in으로 추가했다 — 위 `verify()`
    (controlled, Principal 기반 bounded negotiation)와는 완전히 별개의
    model-backed 경로다. `llm_client`를 넘기지 않으면 그 메서드는 (scope_
    exceeded 협상이 실제로 필요한 경우에 한해) 쓸 수 없고, 기존 `verify()`
    의 동작에는 어떤 영향도 없다."""

    def __init__(self, *, use_authority: bool = True,
                 use_authority_feedback: bool = True,
                 authority_feedback_max_rounds: int = 2,
                 use_verified_experience: bool = True,
                 verified_experience_n_min: int = 3,
                 verified_experience_sigma: float = 0.8,
                 verified_authority: VerifiedAuthorityStore | None = None,
                 llm_client: "LLMClient | None" = None):
        self.use_authority = use_authority
        self.use_authority_feedback = use_authority_feedback
        self.authority_feedback_max_rounds = authority_feedback_max_rounds
        self.use_verified_experience = use_verified_experience
        self.verified_experience_n_min = verified_experience_n_min
        self.verified_experience_sigma = verified_experience_sigma
        self.verified_authority = (verified_authority if verified_authority is not None
                                   else VerifiedAuthorityStore())
        # Phase B opt-in — controlled verify() 는 이 필드를 전혀 참조하지
        # 않는다. verify_agent_proposal() 에서만, 그것도 scope_exceeded일
        # 때만 쓰인다.
        self.llm_client = llm_client

    def verify(self, interpretation: Interpretation, budget: Budget,
               principal: AuthorityPrincipal, log: list[str], key: str
               ) -> AuthorityVerdict:
        auth = check_authority(budget, interpretation.privilege())
        if self.use_authority:
            log.append(f"[joint] Authority: {'통과' if auth.allowed else '차단'} — {auth.reason}")

        interp = interpretation
        n_feedback = 0
        negotiated = False
        auto_restricted = False

        # scope_exceeded 는 하드 리젝트가 아니라 협상 대상. task.truth 는 여기서
        # 전혀 보지 않는다 — principal.review_authority 의 응답만 본다.
        if (self.use_authority and not auth.allowed
                and auth.failure_kind == "scope_exceeded" and self.use_authority_feedback):
            neg = run_feedback(interp, budget, principal,
                               self.authority_feedback_max_rounds, log,
                               verified=(self.verified_authority
                                        if self.use_verified_experience else None),
                               key=key,
                               verified_n_min=self.verified_experience_n_min,
                               verified_sigma=self.verified_experience_sigma)
            n_feedback = neg.rounds
            if neg.resolved and neg.confirmed is not None:
                interp = neg.confirmed.interpretation
                negotiated = True
                auto_restricted = neg.auto_restricted
                auth = check_authority(budget, interp.privilege())
                log.append(f"[joint] Authority(재검사): {'통과' if auth.allowed else '차단'} — "
                           f"{auth.reason}")

        return AuthorityVerdict(interp, auth.allowed, auth.reason,
                                n_feedback, negotiated, auto_restricted)

    # ---- AUTHORITY FLOW (Phase B, model-backed) — opt-in, controlled 경로와 별개 ----
    def verify_agent_proposal(self, *, proposal: Interpretation, budget: Budget,
                              context: str = "") -> AuthorityVerdict:
        """Phase B의 model-backed 진입점 — 위 `verify()`(controlled,
        Principal 기반 bounded negotiation)와 완전히 별개의 경로다.
        `framework.py` 없이 독립적으로 호출 가능하도록 만들었다 —
        `experiments/agent_smoke.py --role authority`가 이 메서드를 직접
        부른다.

        `check_authority()`가 유일하고 최종적인 authority 판정자다 — LLM은
        이미 허용된 것을 뒤집을 수도, 협상 불가능한 하드 리젝트(no_grant/
        condition_missing)를 되살릴 수도 없다. `scope_exceeded`(협상 가능한
        실패)일 때만 LLM을 부르고, 그 역할은 딱 하나 — `check_authority()`
        가 이미 계산해 둔 상한(`auth.suggested`)보다 넓은 것은 절대 제안할
        수 없는 채로, 자유 텍스트 `context`를 바탕으로 그 상한 이하의 최종
        authorized action을 "제안"하는 것뿐이다. LLM이 뭘 제안하든 다시
        `check_authority()`로 재검증한 뒤에만 채택한다 — LLM 판단 자체가
        최종 authority가 되는 일은 없다(non-amplification, §8).

        controlled `verify()`의 `run_feedback()`과 달리 다회 협상 루프가
        아니다 — LLM 호출은 최대 1회다(단일 제안 → 단일 재검증). 이건
        `SemanticVerifierAgent.verify_agent_proposal()`과 호출 구조를
        맞추기 위한 의도적 단순화다 — 다회 협상이 필요해지면 그건 이후
        단계의 일이다.

        이 메서드가 절대 받지 않는 것: `SemanticVerdict`(Semantic
        Verifier의 판단), `PrincipalIntent`(`PrincipalAgent.
        restate_intent()`의 결과), `task.truth`, 최종 fusion 결과 —
        시그니처 자체에 그런 정보가 들어갈 자리가 없다."""
        auth = check_authority(budget, proposal.privilege())

        if auth.allowed or auth.failure_kind != "scope_exceeded" or auth.suggested is None:
            # 이미 허용됐거나 협상 불가능한 하드 리젝트 — check_authority()
            # 자체가 이미 최종 결정이다. LLM 호출조차 필요 없다.
            return AuthorityVerdict(proposal, auth.allowed, auth.reason)

        if self.llm_client is None:
            raise ValueError(
                "verify_agent_proposal()에서 scope_exceeded 협상에는 "
                "llm_client가 필요하다 — AuthorityVerifierAgent(..., "
                "llm_client=...)로 생성하라.")

        ceiling = Interpretation(auth.suggested.action, auth.suggested.resource,
                                 auth.suggested.scope, auth.suggested.condition)
        input_text = _render_agent_authority_input(proposal, ceiling, context)
        response = self.llm_client.generate(
            instructions=_AGENT_AUTHORITY_INSTRUCTIONS, input_text=input_text)

        try:
            llm_proposed = parse_structured_action(response.text)
        except ValueError:
            # 파싱 실패 — fail closed. 원래(거부) 판정을 그대로 유지한다.
            return AuthorityVerdict(proposal, False, auth.reason, n_llm=1,
                                    negotiated=False, response=response)

        # LLM 판단을 신뢰하지 않고 항상 top의 check_authority()로 재검증한다
        # — non-amplification. 재검증을 통과하지 못하면 원래 proposal 로
        # 되돌린다(controlled verify() 가 협상 실패 시 원래 interpretation
        # 을 유지하는 것과 같은 관례).
        recheck = check_authority(budget, llm_proposed.privilege())
        final = llm_proposed if recheck.allowed else proposal
        return AuthorityVerdict(final, recheck.allowed, recheck.reason, n_llm=1,
                                negotiated=True, response=response)


# ----------------------------------------------------------------------------
# Phase B, model-backed — 프롬프트는 agent-specific 내용이므로 llm.py가 아니라
# 여기(호출하는 쪽) 산다.
# ----------------------------------------------------------------------------
_AGENT_AUTHORITY_INSTRUCTIONS = """\
You are an independent Authority Verifier in an agent-to-agent delegation \
system. Agent B (the Delegate) proposed an action that exceeds its \
delegated authority. A deterministic capability check has already \
computed the maximum possible scope that could be authorized — shown \
below as the ceiling. You cannot authorize anything broader than that \
ceiling under any circumstances; it is a hard limit, not a suggestion.

Given the proposed action, the ceiling, and the context below, decide the \
action that should actually be authorized: either the full ceiling, or \
something narrower than it if the context suggests a narrower scope is \
more appropriate. Do not propose anything wider than the ceiling.

Each field must be a single value, not a combination — do not invent \
aliases for the action/resource, and do not combine multiple values into \
one field (for example, never answer "read, review, summarize"; pick the \
single best one).

Respond in exactly this format, one field per line, no extra commentary:
ACTION: <must match the ceiling's action>
RESOURCE: <must match the ceiling's resource>
SCOPE: <the ceiling's scope, or a narrower one>
CONDITION: <the ceiling's conditions, or additional ones, or 'none'>"""


def _render_agent_authority_input(proposal: Interpretation, ceiling: Interpretation,
                                  context: str) -> str:
    def _fmt(i: Interpretation) -> str:
        condition = ",".join(sorted(i.condition)) or "none"
        return (f"ACTION={i.action} RESOURCE={i.resource} "
               f"SCOPE={i.scope} CONDITION={condition}")

    parts = [
        f"Proposed action: {_fmt(proposal)}",
        f"Maximum authorized ceiling: {_fmt(ceiling)}",
    ]
    if context:
        parts.append(f"Context: {context}")
    return "\n".join(parts)
