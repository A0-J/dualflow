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

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from .capability import Budget, Privilege, check_authority
from .semantic import Interpretation


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
    rounds: int
    decision: FeedbackDecision
    log: list[str] = field(default_factory=list)


def run_feedback(proposal: Interpretation, effective: Budget, principal: AuthorityPrincipal,
                 max_rounds: int = 2, log: list[str] | None = None) -> NegotiationResult:
    """Bounded authority negotiation — 최대 `max_rounds` 회. 매 라운드:

    1. 현재 제안이 이미 유효 예산 안이면 즉시 확정.
    2. 아니고 유무/조건 실패(협상 불가)면 즉시 거부.
    3. scope 초과면 `principal.review_authority` 로 A 에게 확인받고, 응답을
       반영한 다음 제안으로 다시 1 로 돌아간다.

    non-amplification 불변식(§6): 어떤 결정이든 확정 결과는 반드시
    `effective` 예산 안에 있어야 한다. A 가 부주의해서 범위 밖 제안을 그대로
    승인(APPROVE)해도 이 검사가 걸러낸다 — Principal 의 응답이 최종 권한이
    아니라 위임 예산이 최종 상한이라는 뜻이다.

    `NegotiationResult.rounds` 는 루프를 몇 바퀴 돌았는지가 아니라 **A 에게
    실제로 몇 번 물어봤는지**다(비용은 실제 상호작용에만 매긴다) — 애초에
    협상 불가(no_grant/condition_missing/suggested 없음)로 즉시 거부된
    경우는 A 를 부르지 않았으므로 0 이다.
    """
    log = log if log is not None else []
    current = proposal
    last_decision = FeedbackDecision.APPROVE
    asked = 0

    for round_no in range(1, max_rounds + 1):
        # 매 라운드 top 에서 다시 권한 검사한다 — A 의 응답을 그대로 승인하지
        # 않고 예산에 대고 재확인하는 것 자체가 non-amplification 검사다. A 가
        # 부주의해서 범위 밖 값을 승인해도, 다음 라운드에서 여전히 막힌다.
        auth = check_authority(effective, current.privilege())
        if auth.allowed:
            log.append(f"[authority-feedback] 확정({asked}회 문의): {current}")
            return NegotiationResult(True, ConfirmedAuthority(current, asked, last_decision),
                                     asked, last_decision, log)

        if auth.failure_kind != "scope_exceeded" or auth.suggested is None:
            log.append(f"[authority-feedback] 협상 불가({auth.failure_kind}) — A 에게 묻지 않고 거부: "
                       f"{auth.reason}")
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
        # 다음 라운드로 — top 의 check_authority 가 current 를 다시 검증한다.

    log.append(f"[authority-feedback] {max_rounds}회 문의 소진 → 미해결")
    return NegotiationResult(False, None, asked, FeedbackDecision.REJECT, log)
