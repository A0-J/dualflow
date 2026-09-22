"""SemanticVerifierAgent.verify_agent_proposal() — Phase B의 opt-in,
model-backed semantic 검증 경로. controlled Fast/Slow/AND/Adaptive
(`verify()`)와는 완전히 별개다. 전부 fake LLMClient로 검증한다: 네트워크도,
OPENAI_API_KEY도 필요 없다.
"""

from __future__ import annotations

import inspect

import pytest

from dualflow.llm import LLMResponse
from dualflow.semantic import Interpretation, SemanticVerdict, SemanticVerifierAgent


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


_PROPOSAL = Interpretation("read", "file", "/reports/2026-08/", frozenset())


class TestVerifyAgentProposal:
    def test_requires_llm_client(self):
        agent = SemanticVerifierAgent()  # llm_client 미지정
        with pytest.raises(ValueError):
            agent.verify_agent_proposal(delegation="x", proposal=_PROPOSAL)

    def test_uses_injected_llm_client_exactly_once(self):
        fake = _FakeLLMClient([_structured()])
        agent = SemanticVerifierAgent(llm_client=fake)

        agent.verify_agent_proposal(delegation="Please read the report.",
                                    proposal=_PROPOSAL)

        assert len(fake.calls) == 1

    def test_matching_resolution_is_confirmed(self):
        fake = _FakeLLMClient([_structured(action="read")])
        agent = SemanticVerifierAgent(llm_client=fake)

        verdict = agent.verify_agent_proposal(
            delegation="Please read the report.", proposal=_PROPOSAL)

        assert isinstance(verdict, SemanticVerdict)
        assert verdict.confirmed is True
        assert verdict.interpretation == _PROPOSAL
        assert verdict.route == "agent_llm"
        assert verdict.status == "resolved"

    def test_mismatching_resolution_is_not_confirmed_and_keeps_own_reading(self):
        """B는 export를 제안했지만, 독립 판단은 read다 — B proposal로
        덮어쓰지 않고, verifier 자신의 판단을 interpretation에 보존한다."""
        mismatched_proposal = Interpretation("export", "file", "/reports/2026-08/",
                                             frozenset())
        fake = _FakeLLMClient([_structured(action="read")])
        agent = SemanticVerifierAgent(llm_client=fake)

        verdict = agent.verify_agent_proposal(
            delegation="Please read the report.", proposal=mismatched_proposal)

        assert verdict.confirmed is False
        assert verdict.interpretation.action == "read"  # verifier 자신의 판단
        assert verdict.interpretation != mismatched_proposal  # B의 제안이 아니다
        assert verdict.route == "agent_llm:mismatch"
        assert verdict.status == "unresolved"

    def test_unparseable_response_fails_closed_not_silently_confirmed(self):
        fake = _FakeLLMClient([LLMResponse(text="I cannot tell what this means.")])
        agent = SemanticVerifierAgent(llm_client=fake)

        verdict = agent.verify_agent_proposal(delegation="x", proposal=_PROPOSAL)

        assert verdict.confirmed is False
        assert verdict.route == "agent_llm:unparseable"
        assert verdict.status == "unresolved"

    def test_input_contains_delegation_and_proposal(self):
        fake = _FakeLLMClient([_structured()])
        agent = SemanticVerifierAgent(llm_client=fake)

        agent.verify_agent_proposal(delegation="my delegation text",
                                    proposal=_PROPOSAL, context="my context")

        input_text = fake.calls[0]["input_text"]
        assert "my delegation text" in input_text
        assert "my context" in input_text
        assert "read" in input_text  # B의 proposal 자체는 당연히 보인다
        assert "/reports/2026-08/" in input_text

    def test_metadata_preserved(self):
        resp = LLMResponse(text="ACTION: read\nRESOURCE: file",
                           input_tokens=40, output_tokens=9, latency_ms=123.0)
        fake = _FakeLLMClient([resp])
        agent = SemanticVerifierAgent(llm_client=fake)

        verdict = agent.verify_agent_proposal(delegation="x", proposal=_PROPOSAL)

        assert verdict.response is not None
        assert verdict.response.input_tokens == 40
        assert verdict.response.output_tokens == 9
        assert verdict.response.latency_ms == 123.0
        assert verdict.n_llm == 1

    def test_no_network_or_api_key_required(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        fake = _FakeLLMClient([_structured()])
        agent = SemanticVerifierAgent(llm_client=fake)

        agent.verify_agent_proposal(delegation="x", proposal=_PROPOSAL)  # 예외 없어야 함

    def test_no_forbidden_parameter_in_public_api(self):
        """truth/ground_truth/label/verdict(Authority)/intent(Principal의
        restate_intent 결과)를 받을 방법이 시그니처에 없어야 한다. `proposal`
        은 검증 대상 그 자체이므로 forbidden에서 제외한다."""
        forbidden = ("truth", "ground_truth", "label", "authority",
                     "principal", "verdict")
        sig = inspect.signature(SemanticVerifierAgent.verify_agent_proposal)
        for name in sig.parameters:
            lowered = name.lower()
            assert not any(f in lowered for f in forbidden), (
                f"verify_agent_proposal has a suspicious parameter: {name}")


class TestControlledPathUnaffected:
    """llm_client를 넘기지 않는 한 기존 Fast/Slow/AND/Adaptive 동작에는 아무
    영향이 없어야 한다 — 생성자에 필드 하나가 추가됐을 뿐이다."""

    def test_default_construction_has_no_llm_client(self):
        agent = SemanticVerifierAgent()
        assert agent.llm_client is None

    def test_verify_signature_unchanged(self):
        sig = inspect.signature(SemanticVerifierAgent.verify)
        assert list(sig.parameters) == ["self", "task", "principal", "log"]
