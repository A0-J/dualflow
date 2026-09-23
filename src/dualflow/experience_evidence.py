"""
V3 PROTOTYPE — HISTORICAL EXPERIENCE AS SEMANTIC EVIDENCE (opt-in,
diagnostic-only, not wired into any runtime yet).

B7d.6(docs/experiments/agent_connected_eval.md §22)이 보여준 것: v2가
Delegate의 candidate-generation prompt에 historical experience를 직접
주입하는 한, paraphrase를 아무리 바꿔도 "literal action token +
structured confirmed-interpretation slot"의 결합에서만 효과가 나타나는
slot-token interaction을 피하기 어려웠다(neither the literal token nor a
paraphrase alone reproduced the effect; only their conjunction did,
interaction contrast +0.225). v3의 핵심 원칙(사용자 승인):

    Historical experience should be semantic evidence for verification,
    not an instruction injected into Delegate candidate generation.

이 모듈은 그 원칙의 첫 구현 조각이다 — `HistoricalEvidenceComparator`.

이 모듈이 하지 않는 것 (전부 의도적):
  - Delegate candidate generation에 관여하는 것. `DelegateAgent`를 import
    조차 하지 않는다 — candidate는 항상 호출자가 이미 만들어서(history
    없이 생성된) `Interpretation`으로 넘겨준다. 이 모듈은 그 candidate를
    "이미 주어진 것"으로만 다루고, 절대 새로 생성하지 않는다.
  - Authority/Budget/AuthorityVerifierAgent에 관여하는 것 — import도,
    참조도 하지 않는다. 권한과 완전히 무관하다.
  - `AgentDelegationRuntime.run()`에 연결되는 것 — 아직 연결되지 않는다.
    기존 runtime/Authority Flow/`SemanticVerifierAgent.
    verify_agent_proposal()`의 동작은 이 모듈로 인해 전혀 바뀌지 않는다.
  - entropy 기준으로 "history를 볼지 말지"를 정하는 것 — 그 게이트는
    나중 설계 조각이고, 구현되더라도 "현재 지시가 explicit하다"는 판정이
    아니라 "history를 consult할 필요가 있는가"라는 훨씬 좁은 판단으로
    한정되어야 한다(v3 아키텍처 리뷰에서 명시적으로 합의됨).
    explicit-current-instruction 안전장치는 여전히 기존
    `PrincipalAgent.restate_intent()` + `AgentDelegationRuntime`의
    deterministic principal-match가 담당하며, 이 모듈은 그 검사를
    우회할 수 없다(우회할 방법 자체가 없다 — 이 모듈은 runtime에 연결돼
    있지 않다).
  - 여러 verified experience를 하나로 합치거나 우선순위를 매기는 것 —
    `judge()`는 `evidence_text` 문자열 하나만 받는다. 첫 v3 pilot은
    정확히 1개의 verified experience만 다룬다(multi-experience
    aggregation/순서는 나중 문제).

핵심 invariant: `judge()`는 주어진 `candidate: Interpretation`을 그대로
반환한다(생성/수정하지 않는다) — 오직 그 candidate와 historical evidence
사이의 관계 하나(`EvidenceRelation`)만 판단한다. 이건 `LLMJudge`(기존,
`llm.py`)가 "이미 구조화된 후보 중 하나를 고르는" 좁은 계약을 갖는 것과
같은 설계 원칙이다 — 자유 생성이 아니라 닫힌 분류(classification)이므로
환각/파싱 실패가 줄고, candidate 자체가 오염될 경로가 없다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .llm import LLMClient, LLMResponse
from .semantic import Interpretation


class EvidenceRelation(str, Enum):
    """`HistoricalEvidenceComparator.judge()`가 반환할 수 있는 유일한
    네 가지 값. 다섯 번째 값(예: 새 action 이름)은 존재하지 않는다 —
    닫힌 분류 문제로 설계했다는 것 자체가 구조적 안전장치다."""

    SUPPORT = "SUPPORT"
    CONFLICT = "CONFLICT"
    IRRELEVANT = "IRRELEVANT"
    UNCERTAIN = "UNCERTAIN"


_VALID_RELATIONS = {r.value: r for r in EvidenceRelation}

_COMPARE_INSTRUCTIONS = """\
You are checking whether a piece of historical evidence about a \
Principal's past confirmed intent supports, conflicts with, or is \
irrelevant to ONE SPECIFIC candidate interpretation of the CURRENT \
delegation.

You are NOT deciding what the current delegation means, and you must NOT \
propose a different interpretation. The candidate interpretation given to \
you is already fixed by an earlier, independent step.

Read the historical evidence and the candidate interpretation, then \
answer with EXACTLY ONE of these four words and nothing else:

SUPPORT     - the historical evidence indicates this Principal would
              likely confirm this same candidate action again for a task
              like the current one.
CONFLICT    - the historical evidence indicates this Principal would
              likely NOT confirm this candidate action for a task like
              the current one.
IRRELEVANT  - the historical evidence has no real bearing on this
              candidate (e.g. it concerns an unrelated kind of task).
UNCERTAIN   - the historical evidence does not clearly favor either
              SUPPORT or CONFLICT for this candidate.

Output only the single word — no punctuation, no explanation."""


def parse_evidence_relation(text: str) -> EvidenceRelation:
    """`text`에서 `EvidenceRelation` 값 정확히 하나만 엄격하게 파싱한다.
    `parse_structured_action()`(semantic.py)과 같은 fail-closed 원칙 —
    네 값 중 정확히 하나가 아니면(다른 말이 섞였거나, 값이 없거나, 다섯
    번째 값이면) `ValueError`. 호출자가 이 실패를 "파싱 실패"로 세고
    분포에서 제외해야 한다(`DelegateAgent.sample_candidates()`와 동일한
    관례)."""
    words = text.strip().split()
    if len(words) != 1:
        raise ValueError(
            f"parse_evidence_relation(): {text!r} must be exactly one word, "
            f"one of {sorted(_VALID_RELATIONS)}")
    token = words[0].strip(".,!:;\"'").upper()
    if token not in _VALID_RELATIONS:
        raise ValueError(
            f"parse_evidence_relation(): {text!r} is not exactly one of "
            f"{sorted(_VALID_RELATIONS)}")
    return _VALID_RELATIONS[token]


def _render_compare_input(*, delegation: str, candidate: Interpretation,
                          evidence_text: str) -> str:
    condition = ",".join(sorted(candidate.condition)) or "none"
    return (
        f"Current delegation: {delegation}\n\n"
        f"Candidate interpretation (already fixed — do not change it):\n"
        f"ACTION: {candidate.action}\n"
        f"RESOURCE: {candidate.resource}\n"
        f"SCOPE: {candidate.scope}\n"
        f"CONDITION: {condition}\n\n"
        f"Historical evidence:\n{evidence_text}"
    )


@dataclass(frozen=True)
class EvidenceJudgment:
    """`judge()` 호출 1회의 결과. `candidate`는 호출자가 넘긴 값 그대로다
    (이 클래스도, `HistoricalEvidenceComparator`도 이 필드를 절대 다른
    값으로 바꾸지 않는다 — 그 자체가 "새 action을 생성하지 않는다"는
    invariant의 실제 코드 표현이다)."""

    candidate: Interpretation
    relation: EvidenceRelation
    response: LLMResponse


class HistoricalEvidenceComparator:
    """이미 생성된(freeze된) `Interpretation` candidate 각각에 대해,
    주어진 historical evidence 텍스트와의 관계만 판단한다.

    `judge()`는 `DelegateAgent`를 참조하지 않고, 새 `Interpretation`을
    구성하지도 않는다 — 인자로 받은 `candidate`를 그대로 결과에 담아
    돌려줄 뿐이다. `evidence_text`가 무엇으로 렌더링되는지(neutral/
    lexical-only/semantic-paraphrase/structured 등)는 이 클래스의 관심사가
    아니다 — 그건 호출자(diagnostic 스크립트)가 정한다."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def judge(self, *, delegation: str, candidate: Interpretation,
             evidence_text: str) -> EvidenceJudgment:
        """`ValueError`를 던질 수 있다(파싱 실패) — 호출자가 그 경우를
        "이 샘플은 판단 불가"로 처리하고 제외해야 한다. 이 메서드는 그
        예외를 삼키지 않는다."""
        input_text = _render_compare_input(
            delegation=delegation, candidate=candidate, evidence_text=evidence_text)
        response = self.llm.generate(
            instructions=_COMPARE_INSTRUCTIONS, input_text=input_text)
        relation = parse_evidence_relation(response.text)
        return EvidenceJudgment(candidate=candidate, relation=relation, response=response)
