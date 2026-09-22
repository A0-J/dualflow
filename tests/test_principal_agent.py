"""PrincipalAgent — Agent A의 delegate()/restate_intent(). 전부 fake LLMClient로
검증한다: 네트워크도, OPENAI_API_KEY도 필요 없다.
"""

from __future__ import annotations

import inspect

import pytest

from dualflow.llm import LLMResponse
from dualflow.principal_agent import PrincipalAgent, PrincipalDelegation, PrincipalIntent
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


def _structured(action="read", resource="file", scope="/reports/2026-08/",
                 condition="none") -> LLMResponse:
    return LLMResponse(
        text=f"ACTION: {action}\nRESOURCE: {resource}\nSCOPE: {scope}\nCONDITION: {condition}")


class TestDelegate:
    def test_calls_llm_and_returns_delegation(self):
        fake = _FakeLLMClient([LLMResponse(text="Please read last month's report.")])
        agent = PrincipalAgent(llm=fake)

        result = agent.delegate(goal="Read last month's report", context="marketing team")

        assert isinstance(result, PrincipalDelegation)
        assert result.delegation == "Please read last month's report."
        assert result.response.text == "Please read last month's report."
        assert len(fake.calls) == 1

    def test_input_contains_goal_and_context(self):
        fake = _FakeLLMClient([LLMResponse(text="ok")])
        agent = PrincipalAgent(llm=fake)

        agent.delegate(goal="my goal", context="my context")

        assert "my goal" in fake.calls[0]["input_text"]
        assert "my context" in fake.calls[0]["input_text"]


class TestRestateIntent:
    def test_is_a_separate_call_and_parses_structured_action(self):
        fake = _FakeLLMClient([_structured()])
        agent = PrincipalAgent(llm=fake)

        result = agent.restate_intent(goal="Read last month's report",
                                      context="marketing team")

        assert isinstance(result, PrincipalIntent)
        assert result.intended_action == Interpretation(
            "read", "file", "/reports/2026-08/", frozenset())
        assert len(fake.calls) == 1

    def test_condition_parsing(self):
        fake = _FakeLLMClient([_structured(condition="approved, business-hours")])
        agent = PrincipalAgent(llm=fake)

        result = agent.restate_intent(goal="x")

        assert result.intended_action.condition == frozenset({"approved", "business-hours"})

    def test_missing_scope_defaults_to_wildcard(self):
        fake = _FakeLLMClient([LLMResponse(text="ACTION: read\nRESOURCE: file")])
        agent = PrincipalAgent(llm=fake)

        result = agent.restate_intent(goal="x")

        assert result.intended_action.scope == "*"
        assert result.intended_action.condition == frozenset()

    def test_raises_on_unparseable_response(self):
        fake = _FakeLLMClient([LLMResponse(text="I'm not sure what to do.")])
        agent = PrincipalAgent(llm=fake)

        with pytest.raises(ValueError):
            agent.restate_intent(goal="x")

    def test_accepts_optional_own_delegation_context(self):
        fake = _FakeLLMClient([_structured()])
        agent = PrincipalAgent(llm=fake)

        agent.restate_intent(goal="x", own_delegation="Please read the report.")

        assert "Please read the report." in fake.calls[0]["input_text"]

    def test_does_not_take_bs_proposal_as_input(self):
        """restate_intent()의 시그니처 자체에 B의 proposal을 받는 파라미터가
        없어야 한다 — anchoring을 피하기 위한 핵심 설계 제약."""
        sig = inspect.signature(PrincipalAgent.restate_intent)
        for name in sig.parameters:
            assert "proposal" not in name.lower()
            assert "verdict" not in name.lower()


class TestIndependence:
    def test_delegate_and_restate_are_independent_calls(self):
        fake = _FakeLLMClient([
            LLMResponse(text="Please export last month's report."),
            _structured(action="read"),
        ])
        agent = PrincipalAgent(llm=fake)

        deleg = agent.delegate(goal="Read last month's report")
        intent = agent.restate_intent(goal="Read last month's report")

        assert len(fake.calls) == 2
        assert fake.calls[0]["instructions"] != fake.calls[1]["instructions"]
        # restate_intent()의 입력에는 delegate()의 출력("export")이 전혀 섞이지 않는다.
        assert "export" not in fake.calls[1]["input_text"]
        assert deleg.delegation == "Please export last month's report."
        assert intent.intended_action.action == "read"


class TestLLMResponseMetadataPreserved:
    def test_delegate_preserves_metadata(self):
        resp = LLMResponse(text="ok", input_tokens=10, output_tokens=5, latency_ms=42.0)
        fake = _FakeLLMClient([resp])
        agent = PrincipalAgent(llm=fake)

        result = agent.delegate(goal="x")

        assert result.response.input_tokens == 10
        assert result.response.output_tokens == 5
        assert result.response.latency_ms == 42.0

    def test_restate_intent_preserves_metadata(self):
        resp = LLMResponse(text="ACTION: read\nRESOURCE: file",
                           input_tokens=20, output_tokens=8, latency_ms=99.0)
        fake = _FakeLLMClient([resp])
        agent = PrincipalAgent(llm=fake)

        result = agent.restate_intent(goal="x")

        assert result.response.input_tokens == 20
        assert result.response.output_tokens == 8
        assert result.response.latency_ms == 99.0


class TestNoGroundTruthAccess:
    def test_no_network_or_api_key_required(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        fake = _FakeLLMClient([LLMResponse(text="hi")])
        agent = PrincipalAgent(llm=fake)

        agent.delegate(goal="x")  # 예외 없이 동작해야 한다

    def test_no_truth_or_label_parameter_anywhere_in_the_public_api(self):
        forbidden = ("truth", "ground_truth", "label", "verdict")
        for target in (PrincipalAgent.__init__, PrincipalAgent.delegate,
                       PrincipalAgent.restate_intent):
            for name in inspect.signature(target).parameters:
                lowered = name.lower()
                assert not any(f in lowered for f in forbidden), (
                    f"{target.__qualname__} has a suspicious parameter: {name}")
