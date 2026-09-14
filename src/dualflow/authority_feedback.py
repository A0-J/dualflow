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
