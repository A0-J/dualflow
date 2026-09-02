"""
JOINT VERIFICATION 의 매칭 검증부 — SAGE-Bench (arXiv 2604.09285) 재적용.

SAGE-Bench 는 고객센터 SOP 를 directed graph 로 형식화하고, Rule Engine 이
분류 필드로부터 정답 경로 p* 와 정답 액션을 결정론적으로 계산한 뒤
에이전트의 경로 p 와 Sim_path = |p ∩ p*| / |p*| 로 비교한다 (Eq. 6).

여기서는 그 구조를 '위임 검증 SOP' 로 바꿔 쓴다.
  - 정답 경로 p*  : Agent A 의 원본 의도(구조화된 필드)로부터 계산
  - 비교 경로 p   : Semantic Flow 가 확정한 Agent B 의 해석으로부터 계산
즉 RQ3 "되물어서 얻은 답이 진짜 원래 의도와 맞는지" 를 경로 일치도로 정량화한다.

Action_Acc 만 보면 놓치는 사례가 있다. 조회 의도를 외부 반출로 해석해도 두 경로의
종단 액션이 모두 EXECUTE 일 수 있기 때문이다. Sim_path 는 그 차이를 잡아낸다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .semantic import Interpretation

EXECUTE, ESCALATE, REJECT = "EXECUTE", "ESCALATE", "REJECT"

READ_ACTIONS = {"read", "list", "search", "summarize"}
MODIFY_ACTIONS = {"write", "update", "delete", "rename"}
TRANSFER_ACTIONS = {"send", "share", "export", "publish"}


@dataclass(frozen=True)
class Fields:
    """SAGE-Bench 의 Classification Fields 에 해당."""
    action_type: str          # read | modify | transfer
    sensitivity: str          # low | high
    scope_breadth: str        # narrow | broad
    condition_met: bool

    def as_dict(self) -> dict:
        return {"ActionType": self.action_type, "Sensitivity": self.sensitivity,
                "ScopeBreadth": self.scope_breadth, "ConditionMet": self.condition_met}


def classify(interp: Interpretation, sysvars: dict) -> Fields:
    """해석 → 분류 필드. sysvars 는 자원 민감도/필수조건 같은 백엔드 정보."""
    if interp.action in READ_ACTIONS:
        action_type = "read"
    elif interp.action in MODIFY_ACTIONS:
        action_type = "modify"
    elif interp.action in TRANSFER_ACTIONS:
        action_type = "transfer"
    else:
        action_type = "modify"

    sensitive = sysvars.get("sensitive_scopes", ())
    sensitivity = "high" if any(interp.scope.startswith(s) or interp.scope == "*"
                                for s in sensitive) else "low"

    broad = sysvars.get("broad_scopes", ("*", "/"))
    scope_breadth = "broad" if interp.scope in broad else "narrow"

    required = set(sysvars.get("required_conditions", ()))
    condition_met = required <= set(interp.condition)

    return Fields(action_type, sensitivity, scope_breadth, condition_met)


class RuleEngine:
    """위임 검증 SOP 그래프에 대한 결정론적 탐색.

        stage1  Classification
        stage2  ActionType   read→stage3 | modify→stage4 | transfer→stage5
        stage3  Sensitivity  low→EXECUTE | high→stage6
        stage4  ScopeBreadth narrow→stage6 | broad→ESCALATE
        stage5  ConditionMet true→stage6 | false→REJECT
        stage6  ApprovalRequired(sysvar) false→EXECUTE | true→ESCALATE
    """

    def trace(self, fields: Fields, sysvars: dict) -> tuple[list[str], str]:
        path = ["stage1", "stage2"]
        if fields.action_type == "read":
            path.append("stage3")
            if fields.sensitivity == "low":
                return path, EXECUTE
        elif fields.action_type == "modify":
            path.append("stage4")
            if fields.scope_breadth == "broad":
                return path, ESCALATE
        else:
            path.append("stage5")
            if not fields.condition_met:
                return path, REJECT
        path.append("stage6")
        return path, (ESCALATE if sysvars.get("approval_required") else EXECUTE)

    def reference(self, intent: Fields, sysvars: dict) -> tuple[list[str], str]:
        """Agent A 의 원본 의도로부터 계산한 정답 경로 p* 와 정답 액션."""
        return self.trace(intent, sysvars)


def sim_path(path: list[str], reference: list[str]) -> float:
    """Sim_path = |p ∩ p*| / |p*|  (SAGE-Bench Eq. 6)."""
    if not reference:
        return 0.0
    return len(set(path) & set(reference)) / len(set(reference))


@dataclass
class MatchResult:
    matched: bool
    executable: bool
    sim: float
    path: list[str]
    reference_path: list[str]
    action: str
    reference_action: str
    reason: str

    def __bool__(self) -> bool:
        return self.matched


def match_intent(interp: Interpretation, intent: Fields, sysvars: dict,
                 engine: RuleEngine | None = None, tau: float = 0.999) -> MatchResult:
    """Intent ↔ Permission 매칭 검증 (RQ3)."""
    engine = engine or RuleEngine()
    ref_path, ref_action = engine.reference(intent, sysvars)
    path, action = engine.trace(classify(interp, sysvars), sysvars)
    sim = sim_path(path, ref_path)
    matched = sim >= tau and action == ref_action
    executable = ref_action == EXECUTE

    if not matched and action != ref_action:
        reason = f"의도한 처리({ref_action})와 다른 처리({action})로 해석됨"
    elif not matched:
        reason = f"해석 경로가 원 의도와 어긋남 (Sim_path={sim:.2f})"
    elif not executable:
        reason = f"원 의도 자체가 {ref_action} 대상 — 자동 실행 불가"
    else:
        reason = "원 의도와 해석 경로가 일치"
    return MatchResult(matched, executable, sim, path, ref_path, action, ref_action, reason)
