"""
LLM 의미 판단 — fallback. 소수 케이스에서만 호출되는 비용 최소화의 핵심 장치.

프레임워크는 LLM 을 '호출 횟수를 세는 자원' 으로만 취급한다. 테스트와 실험은
ScriptedJudge 로 결정론적으로 돌리고, 실제 모델은 LLMJudge 인터페이스만 맞추면
그대로 갈아끼울 수 있다.
"""

from __future__ import annotations

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
