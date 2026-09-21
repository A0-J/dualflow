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

from dataclasses import dataclass, field

from .semantic import Interpretation

EXECUTE, ESCALATE, REJECT = "EXECUTE", "ESCALATE", "REJECT"

READ_ACTIONS = {"read", "list", "search", "summarize"}
MODIFY_ACTIONS = {"write", "update", "delete", "rename"}
TRANSFER_ACTIONS = {"send", "share", "export", "publish"}


@dataclass(frozen=True)
class Fields:
    """SAGE-Bench 의 Classification Fields 에 해당.

    action_type/sensitivity/scope_breadth/condition_met 은 SOP 그래프 탐색용
    '거친(coarse)' 분류다 — 정책 엔진이 실제로 보는 수준을 흉내낸다. 같은
    버킷(예: 둘 다 low-sensitivity narrow read)으로 묶이는 서로 다른 자원은 SOP
    경로(Sim_path)만으로는 구분되지 않는다("/reports/2026-08/" 를 요청해놓고
    "/reports/2025-01/" 을 실행해도 같은 경로). action/resource/scope/condition
    은 그 구멍을 메우는 원본 값이고, Joint Verification 이 exact-field 매칭
    (V_action ∧ V_resource ∧ V_scope ∧ V_condition) 에 쓴다 — §3 실험 ⑤ 참고.
    """
    action_type: str          # read | modify | transfer
    sensitivity: str          # low | high
    scope_breadth: str        # narrow | broad
    condition_met: bool
    action: str = ""
    resource: str = ""
    scope: str = ""
    condition: frozenset[str] = frozenset()


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

    return Fields(action_type, sensitivity, scope_breadth, condition_met,
                  interp.action, interp.resource, interp.scope, interp.condition)


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


def field_match(fields: Fields, reference: Fields) -> dict[str, bool]:
    """V_action ∧ V_resource ∧ V_scope ∧ V_condition — 원본 값의 정확 일치.

    SOP 경로(Sim_path)는 분류 버킷(action_type/sensitivity/scope_breadth) 이
    같으면 자원이 달라도 같은 경로를 낸다 — "/reports/2026-08/" 를 요청해놓고
    "/reports/2025-01/" 을 실행해도 Sim_path=1.0 이 되는 것이 그 예다. 이 네
    필드는 그 틈을 메우려고 원본 값을 직접 비교한다.
    """
    return {
        "action": fields.action == reference.action,
        "resource": fields.resource == reference.resource,
        "scope": fields.scope == reference.scope,
        "condition": fields.condition == reference.condition,
    }


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
    field_match: dict[str, bool] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.matched


def match_intent(interp: Interpretation, intent: Fields, sysvars: dict,
                 engine: RuleEngine | None = None, tau: float = 0.999,
                 require_fields: bool = True) -> MatchResult:
    """Intent ↔ Permission 매칭 검증 (RQ3).

    두 겹으로 확인한다 — (1) SOP 경로 유사도(Sim_path, 정책 분류 수준의 일관성)
    와 (2) V = V_action ∧ V_resource ∧ V_scope ∧ V_condition (원본 값의 정확
    일치). 어느 한쪽만으로는 부족하다: Sim_path 는 자원 치환(같은 버킷의 다른
    자원)을 놓치고, 필드 매칭만으로는 "경로는 다른데 우연히 필드가 같은" 경우
    (예: 최종 액션은 같지만 중간에 다른 승인 단계를 거친 경우)를 놓친다.

    require_fields=False 는 ablation 용 — Sim_path 만으로 매칭하던 이전 동작을
    재현한다(§3 실험 ⑤, "Sim_path 만으로는 자원 치환을 놓친다"를 보이는 대조군).
    """
    engine = engine or RuleEngine()
    interp_fields = classify(interp, sysvars)
    ref_path, ref_action = engine.reference(intent, sysvars)
    path, action = engine.trace(interp_fields, sysvars)
    sim = sim_path(path, ref_path)
    fm = field_match(interp_fields, intent)
    fields_ok = all(fm.values()) if require_fields else True
    matched = sim >= tau and action == ref_action and fields_ok
    executable = ref_action == EXECUTE

    if not matched and action != ref_action:
        reason = f"의도한 처리({ref_action})와 다른 처리({action})로 해석됨"
    elif not matched and not fields_ok:
        bad = [k for k, ok in fm.items() if not ok]
        reason = f"경로는 일치하지만 원본 값이 다름 — V_{'∧V_'.join(bad)} 위반"
    elif not matched:
        reason = f"해석 경로가 원 의도와 어긋남 (Sim_path={sim:.2f})"
    elif not executable:
        reason = f"원 의도 자체가 {ref_action} 대상 — 자동 실행 불가"
    else:
        reason = "원 의도와 해석 경로·필드가 모두 일치"
    return MatchResult(matched, executable, sim, path, ref_path, action, ref_action,
                       reason, fm)
