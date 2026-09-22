"""
공유 파싱 유틸리티 — ACTION/RESOURCE/SCOPE/CONDITION 고정 포맷을
`semantic.Interpretation`으로 바꾼다.

`PrincipalAgent.restate_intent()`(Agent A)와 `DelegateAgent.propose()`
(Agent B)가 정확히 같은 응답 포맷을 요구하고 정확히 같은 규칙으로
파싱해야 한다 — 두 결과를 나중에 비교/대조해야 하는데 파싱 규칙이 조금이라도
다르면 그 비교 자체가 오염된다. 그래서 이 파서 하나만 공유 모듈로 뽑아둔다.

이 모듈은 model을 호출하지 않는다 — 순수 텍스트 파싱 함수 하나뿐이라
결정론적이고, API 없이 단위 테스트할 수 있다.

파싱에 실패하면 조용히 기본값/추정값으로 메꾸거나 ground truth로 되돌리지
않고 `ValueError`를 던진다 — Agent가 잘못 이해한 원문을 그대로 드러내는
것도 이 연구가 관찰하려는 대상이지, 파서가 그 실수를 대신 감추면 안 된다.
"""

from __future__ import annotations

from .semantic import Interpretation


def parse_structured_action(text: str) -> Interpretation:
    """다음 고정 스키마(한 줄에 한 필드, 그 외 설명 텍스트 없음)를 파싱한다.

        ACTION: <action verb>
        RESOURCE: <resource type>
        SCOPE: <scope/path, 생략 시 '*'(무제한)>
        CONDITION: <comma-separated conditions, 생략/'none' 시 빈 집합>

    ACTION/RESOURCE는 필수다 — 찾지 못하면 raw text를 포함한 `ValueError`를
    던진다."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().upper()
        value = value.strip()
        if key in ("ACTION", "RESOURCE", "SCOPE", "CONDITION"):
            fields[key] = value

    if "ACTION" not in fields or "RESOURCE" not in fields:
        raise ValueError(
            "parse_structured_action(): 응답에서 ACTION/RESOURCE를 찾을 수 "
            f"없다 — raw text: {text!r}"
        )

    scope = fields.get("SCOPE") or "*"
    condition_raw = fields.get("CONDITION", "")
    if condition_raw.strip().lower() in ("", "none"):
        condition: frozenset[str] = frozenset()
    else:
        condition = frozenset(c.strip() for c in condition_raw.split(",") if c.strip())

    return Interpretation(
        action=fields["ACTION"],
        resource=fields["RESOURCE"],
        scope=scope,
        condition=condition,
    )
