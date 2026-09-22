"""
Model-facing 인터페이스. 이 파일은 서로 다른 추상화 레벨의 두 계층을 담는다.

1. **`LLMJudge`** (기존, 변경 없음) — Semantic Flow의 LLM fallback 전용 좁은
   계약이다. `judge(spec, belief, transcript) -> Interpretation`처럼 이미
   구조화된 해석 후보 중 하나를 고르는, task-specific한 인터페이스다.
   프레임워크는 LLM 을 '호출 횟수를 세는 자원' 으로만 취급한다. 테스트와
   실험은 `ScriptedJudge` 로 결정론적으로 돌리고, 실제 모델은 `LLMJudge`
   인터페이스만 맞추면 그대로 갈아끼울 수 있다.

2. **`LLMClient`** (신규) — Phase B(`research/agent-connected-eval`)의 실제
   Agent들(`PrincipalAgent`/`DelegateAgent`/model-backed verifier)이 공통으로
   쓸, 훨씬 더 낮은 레벨의 범용 model-call 계약이다. `generate(instructions,
   input_text) -> LLMResponse`만 한다 — prompt를 보내고 text + usage/latency
   metadata를 돌려주는 것까지가 이 레이어의 책임 전부다. semantic verification
   로직, authority 판단, Budget, Principal confirmation, benchmark task,
   최종 EXECUTE/REJECT, agent별 prompt 내용은 전부 이 레이어 밖(각 Agent)의
   책임이다 — 이 계약을 만족하기만 하면 어떤 model provider든 갈아끼울 수
   있다. `LLMJudge`를 대체하지 않는다 — 서로 다른 추상화 레벨의 별개 계약이며,
   이번 커밋에서는 `SemanticVerifierAgent`/`AuthorityVerifierAgent` 어느 쪽도
   아직 `LLMClient`를 쓰지 않는다 (그건 이후 커밋의 일).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, Sequence

from .semantic import Belief, Interpretation, top


class LLMJudge(Protocol):
    calls: int

    def judge(self, spec: str, belief: Belief,
              transcript: Sequence[tuple[str, str]]) -> Interpretation:
        ...


class TopBeliefJudge:
    """호출 비용만 발생시키고 현재 믿음의 1위를 고르는 기본 구현."""

    def __init__(self) -> None:
        self.calls = 0

    def judge(self, spec, belief, transcript):
        self.calls += 1
        return top(belief)


class ScriptedJudge:
    """시나리오별 정답을 미리 심어둔 오라클. 실험을 결정론적으로 만든다."""

    def __init__(self, answers: dict[str, Interpretation] | None = None):
        self.answers = dict(answers or {})
        self.calls = 0
        self.seen: list[str] = []

    def register(self, spec: str, interp: Interpretation) -> None:
        self.answers[spec] = interp

    def judge(self, spec, belief, transcript):
        self.calls += 1
        self.seen.append(spec)
        return self.answers.get(spec) or top(belief)


class AnthropicJudge:
    """실제 모델을 붙일 때의 어댑터 골격.

    프레임워크는 후보 집합을 이미 구조화해 두었으므로, 모델에게는
    '후보 중 하나를 고르라' 는 좁은 질문만 던지면 된다. 자유 생성이 아니라
    분류이므로 파싱 실패와 환각이 줄고, 호출 1회로 끝난다.
    """

    PROMPT = """\
아래는 Agent A 가 Agent B 에게 내린 위임 명세와, B 가 떠올린 해석 후보들이다.
역질의를 {k}회 했는데도 어떤 해석인지 확정되지 않았다.

위임 명세: {spec}
역질의 기록:
{transcript}

해석 후보:
{options}

가장 타당한 해석의 번호만 출력하라. 다른 말은 쓰지 마라."""

    def __init__(self, client, model: str = "claude-sonnet-5"):
        self.client = client                 # anthropic.Anthropic() 등
        self.model = model
        self.calls = 0

    def judge(self, spec, belief, transcript):
        self.calls += 1
        options = list(belief)
        rendered = "\n".join(f"{i}. {o}  (p={belief[o]:.2f})" for i, o in enumerate(options))
        history = "\n".join(f"- Q: {q}\n  A: {a}" for q, a in transcript) or "- (없음)"
        msg = self.client.messages.create(
            model=self.model, max_tokens=16,
            messages=[{"role": "user", "content": self.PROMPT.format(
                spec=spec, k=len(transcript), transcript=history, options=rendered)}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        try:
            return options[int(text.strip().split()[0])]
        except (ValueError, IndexError):
            return top(belief)


# ----------------------------------------------------------------------------
# LLMClient — Phase B agent 공통 model adapter 레이어.
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class LLMResponse:
    """model 호출 1회의 결과 + 계측 metadata.

    `text` 외 필드는 실험 회계(LLM calls/task, token usage, latency)용이며
    모든 adapter가 이를 보고하는 건 아니므로 기본값은 `None`이다."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None


class LLMClient(Protocol):
    """실제 model 호출의 공통 계약. prompt/input을 받아 model을 호출하고
    text + 호출 metadata를 돌려주는 것까지만 한다 — 그 이상(무엇을 묻고,
    응답을 어떻게 판단으로 바꿀지)은 이 프로토콜을 쓰는 쪽(각 Agent)의
    책임이다. `LLMJudge`(위)와는 다른, 더 낮은 추상화 레벨의 계약이다."""

    def generate(self, *, instructions: str, input_text: str) -> LLMResponse:
        ...


class OpenAILLMClient:
    """OpenAI Responses API 어댑터.

    이 클래스는 `openai` 패키지를 직접 import하지 않는다 — 호출자가 이미
    만든 client 객체(예: `openai.OpenAI()`)를 주입받아 duck-typing으로만
    쓴다. API key도 여기서 읽지 않는다 — client 생성 시점에 호출자가
    `OPENAI_API_KEY` 환경변수 등으로 이미 구성해서 넘긴다. 모델명도 여기서
    하드코딩하지 않고 생성자 인자로 받는다.

    같은 client/adapter 인스턴스를 여러 Agent(PrincipalAgent, DelegateAgent,
    SemanticVerifierAgent, AuthorityVerifierAgent)가 공유해도 된다 — 독립성은
    "같은 adapter를 쓰느냐"가 아니라 "별도 API invocation, 별도 instructions,
    별도 input/context, 상대 verifier의 verdict 비공개"로 확보한다."""

    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def generate(self, *, instructions: str, input_text: str) -> LLMResponse:
        start = time.monotonic()
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input_text,
        )
        latency_ms = (time.monotonic() - start) * 1000.0

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage is not None else None
        output_tokens = getattr(usage, "output_tokens", None) if usage is not None else None

        return LLMResponse(
            text=response.output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )
