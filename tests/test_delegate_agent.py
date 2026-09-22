"""DelegateAgent — Agent B의 propose(). 전부 fake LLMClient로 검증한다:
네트워크도, OPENAI_API_KEY도 필요 없다.
"""

from __future__ import annotations

import inspect

import pytest

from dualflow.delegate_agent import DelegateAgent, DelegateProposal
from dualflow.llm import LLMResponse
from dualflow.principal_agent import PrincipalAgent, PrincipalIntent
from dualflow.semantic import Interpretation


class _FakeLLMClient:
    """`llm.LLMClient` 프로토콜을 흉내 내는 순수 Python fake. 미리 준비한
    응답을 호출 순서대로 돌려준다 — 실제 model 호출도, 네트워크도 없다."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, *, instructions: str, input_text: str) -> LLMResponse:
        self.calls.append({"instructions": instructions, "input_text": input_text})
        return self._responses.pop(0)


def _structured(action="export", resource="file", scope="/reports/2026-08/",
                 condition="none") -> LLMResponse:
    return LLMResponse(
        text=f"ACTION: {action}\nRESOURCE: {resource}\nSCOPE: {scope}\nCONDITION: {condition}")


class TestPropose:
    def test_calls_llm_exactly_once_and_returns_interpretation(self):
        fake = _FakeLLMClient([_structured()])
        agent = DelegateAgent(llm=fake)

        result = agent.propose(delegation="Please export last month's report.")

        assert isinstance(result, DelegateProposal)
        assert result.interpretation == Interpretation(
            "export", "file", "/reports/2026-08/", frozenset())
        assert len(fake.calls) == 1

    def test_input_contains_delegation_and_context(self):
        fake = _FakeLLMClient([_structured()])
        agent = DelegateAgent(llm=fake)

        agent.propose(delegation="my delegation", context="my context")

        assert "my delegation" in fake.calls[0]["input_text"]
        assert "my context" in fake.calls[0]["input_text"]

    def test_raw_text_is_retained(self):
        raw = "ACTION: export\nRESOURCE: file\nSCOPE: *\nCONDITION: none"
        fake = _FakeLLMClient([LLMResponse(text=raw)])
        agent = DelegateAgent(llm=fake)

        result = agent.propose(delegation="x")

        assert result.raw_text == raw

    def test_metadata_preserved(self):
        resp = LLMResponse(text="ACTION: export\nRESOURCE: file",
                           input_tokens=30, output_tokens=6, latency_ms=77.0)
        fake = _FakeLLMClient([resp])
        agent = DelegateAgent(llm=fake)

        result = agent.propose(delegation="x")

        assert result.response.input_tokens == 30
        assert result.response.output_tokens == 6
        assert result.response.latency_ms == 77.0

    def test_raises_on_unparseable_response(self):
        fake = _FakeLLMClient([LLMResponse(text="not sure what to do")])
        agent = DelegateAgent(llm=fake)

        with pytest.raises(ValueError):
            agent.propose(delegation="x")

    def test_no_network_or_api_key_required(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        fake = _FakeLLMClient([_structured()])
        agent = DelegateAgent(llm=fake)

        agent.propose(delegation="x")  # 예외 없이 동작해야 한다


class TestNoGroundTruthOrPrincipalIntentAccess:
    def test_no_forbidden_parameter_in_public_api(self):
        """DelegateAgent의 어떤 public 메서드도 truth/ground_truth/label/
        verdict/intent/principal 같은 이름의 파라미터를 받지 않는다 — B가
        A의 원래 의도(PrincipalIntent)나 ground truth를 참조할 수 있는
        경로 자체가 시그니처에 없어야 한다."""
        forbidden = ("truth", "ground_truth", "label", "verdict",
                     "intent", "principal")
        for target in (DelegateAgent.__init__, DelegateAgent.propose):
            for name in inspect.signature(target).parameters:
                lowered = name.lower()
                assert not any(f in lowered for f in forbidden), (
                    f"{target.__qualname__} has a suspicious parameter: {name}")


class TestPreservesMisunderstanding:
    """B가 delegation을 잘못 이해해도 코드가 자동으로 고치지 않는다 — 이후
    SemanticVerifierAgent가 그 misread를 실제로 감지할 수 있어야 하므로,
    이 시점에서 진짜 misread가 그대로 살아남아야 한다."""

    def test_proposal_diverges_from_principals_intended_action_and_stays_diverged(self):
        # Principal의 독립적 restatement(B2, 별도 fake) — "read"가 진짜 의도.
        principal_fake = _FakeLLMClient([
            LLMResponse(text="ACTION: read\nRESOURCE: file\n"
                             "SCOPE: /reports/2026-08/\nCONDITION: none"),
        ])
        principal = PrincipalAgent(llm=principal_fake)
        intent: PrincipalIntent = principal.restate_intent(goal="Read last month's report")
        assert intent.intended_action.action == "read"

        # B는 같은 delegation을 잘못 이해해서 "export"를 제안한다 — 완전히
        # 별도의 fake/에이전트 인스턴스이므로 서로의 결과를 볼 수 없다.
        delegate_fake = _FakeLLMClient([_structured(action="export")])
        delegate = DelegateAgent(llm=delegate_fake)
        proposal = delegate.propose(delegation="ambiguous natural language delegation")

        # DelegateAgent는 애초에 PrincipalIntent를 받을 방법이 없었으므로(파라미터
        # 자체가 없다) 고칠 수도 없다 — 제안은 여전히 export로 남는다.
        assert proposal.interpretation.action == "export"
        assert proposal.interpretation != intent.intended_action


class TestIndependenceFromPrincipal:
    def test_shared_llm_client_still_makes_separate_calls_with_separate_instructions(self):
        """같은 LLMClient 인스턴스를 Principal/Delegate가 공유해도, 호출은
        완전히 분리돼야 한다 — 대화 상태 공유가 없다."""
        shared_fake = _FakeLLMClient([
            LLMResponse(text="Please read last month's report."),  # delegate() 몫
            _structured(action="export"),                          # propose() 몫
        ])
        principal = PrincipalAgent(llm=shared_fake)
        delegate = DelegateAgent(llm=shared_fake)

        deleg = principal.delegate(goal="Read last month's report")
        proposal = delegate.propose(delegation=deleg.delegation)

        assert len(shared_fake.calls) == 2
        assert shared_fake.calls[0]["instructions"] != shared_fake.calls[1]["instructions"]
        # Delegate의 입력에는 delegation 텍스트만 들어가고, Principal의 원래
        # goal이나 다른 어떤 hidden 정보도 섞이지 않는다.
        assert shared_fake.calls[1]["input_text"] == \
            f"Delegation: {deleg.delegation}"
