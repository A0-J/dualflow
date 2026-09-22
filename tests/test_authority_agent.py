"""AuthorityVerifierAgent.verify_agent_proposal() — Phase B의 opt-in,
model-backed authority 검증 경로. controlled `verify()`(Principal 기반
bounded negotiation)와는 완전히 별개다. 전부 fake LLMClient로 검증한다:
네트워크도, OPENAI_API_KEY도 필요 없다.
"""

from __future__ import annotations

import inspect

import pytest

from dualflow.authority_feedback import AuthorityVerdict, AuthorityVerifierAgent
from dualflow.capability import Budget, Privilege
from dualflow.llm import LLMResponse
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


# 예산: read /reports/2026-08/ 만 위임됨. 아래 대부분의 테스트가 이 예산 위에서
# "제안이 상한을 넘는" scope_exceeded 시나리오를 재현한다.
_NARROW_BUDGET = Budget.of(Privilege("read", "file", "/reports/2026-08/"))
_WITHIN_BUDGET_PROPOSAL = Interpretation("read", "file", "/reports/2026-08/", frozenset())
_OVER_BUDGET_PROPOSAL = Interpretation("read", "file", "/reports/", frozenset())


class TestAlreadyAllowedOrHardReject:
    """check_authority() 만으로 끝나는 경우 — LLM 호출조차 필요 없다."""

    def test_within_budget_needs_no_llm_call(self):
        agent = AuthorityVerifierAgent()  # llm_client 미지정이어도 문제없다
        verdict = agent.verify_agent_proposal(
            proposal=_WITHIN_BUDGET_PROPOSAL, budget=_NARROW_BUDGET)

        assert isinstance(verdict, AuthorityVerdict)
        assert verdict.allowed is True
        assert verdict.interpretation == _WITHIN_BUDGET_PROPOSAL
        assert verdict.n_llm == 0
        assert verdict.response is None

    def test_hard_reject_no_grant_needs_no_llm_call(self):
        """action/resource 자체가 위임 밖(no_grant) — 협상 대상이 아니다."""
        proposal = Interpretation("delete", "file", "/reports/2026-08/", frozenset())
        agent = AuthorityVerifierAgent()  # llm_client 없어도 여기선 안 쓰인다

        verdict = agent.verify_agent_proposal(proposal=proposal, budget=_NARROW_BUDGET)

        assert verdict.allowed is False
        assert verdict.n_llm == 0
        assert verdict.negotiated is False


class TestScopeExceededNegotiation:
    def test_requires_llm_client(self):
        agent = AuthorityVerifierAgent()  # llm_client 미지정
        with pytest.raises(ValueError):
            agent.verify_agent_proposal(
                proposal=_OVER_BUDGET_PROPOSAL, budget=_NARROW_BUDGET)

    def test_llm_proposal_within_ceiling_is_accepted_after_recheck(self):
        # LLM이 정확히 ceiling(=/reports/2026-08/)을 제안 → 재검증 통과 → allow
        fake = _FakeLLMClient([_structured(scope="/reports/2026-08/")])
        agent = AuthorityVerifierAgent(llm_client=fake)

        verdict = agent.verify_agent_proposal(
            proposal=_OVER_BUDGET_PROPOSAL, budget=_NARROW_BUDGET)

        assert verdict.allowed is True
        assert verdict.interpretation.scope == "/reports/2026-08/"
        assert verdict.negotiated is True
        assert verdict.n_llm == 1
        assert len(fake.calls) == 1

    def test_llm_cannot_exceed_the_deterministic_ceiling(self):
        """LLM이 ceiling보다 넓은 scope를 제안해도, check_authority() 재검증이
        그걸 다시 잡아낸다 — LLM 판단이 최종 authority가 될 수 없다."""
        fake = _FakeLLMClient([_structured(scope="/")])  # ceiling보다 훨씬 넓음
        agent = AuthorityVerifierAgent(llm_client=fake)

        verdict = agent.verify_agent_proposal(
            proposal=_OVER_BUDGET_PROPOSAL, budget=_NARROW_BUDGET)

        assert verdict.allowed is False
        # 재검증 실패 시 원래 proposal로 되돌아간다 — LLM의 과도한 제안을
        # 그대로 반영하지 않는다.
        assert verdict.interpretation == _OVER_BUDGET_PROPOSAL

    def test_unparseable_llm_response_fails_closed(self):
        fake = _FakeLLMClient([LLMResponse(text="I'm not sure about this.")])
        agent = AuthorityVerifierAgent(llm_client=fake)

        verdict = agent.verify_agent_proposal(
            proposal=_OVER_BUDGET_PROPOSAL, budget=_NARROW_BUDGET)

        assert verdict.allowed is False
        assert verdict.n_llm == 1
        assert verdict.negotiated is False

    def test_input_shows_proposal_and_ceiling_not_beyond(self):
        fake = _FakeLLMClient([_structured(scope="/reports/2026-08/")])
        agent = AuthorityVerifierAgent(llm_client=fake)

        agent.verify_agent_proposal(proposal=_OVER_BUDGET_PROPOSAL,
                                    budget=_NARROW_BUDGET, context="quarterly audit")

        input_text = fake.calls[0]["input_text"]
        assert "/reports/" in input_text          # 원래 제안
        assert "/reports/2026-08/" in input_text  # 결정론적 ceiling
        assert "quarterly audit" in input_text

    def test_metadata_preserved(self):
        resp = _structured(scope="/reports/2026-08/")
        resp = LLMResponse(text=resp.text, input_tokens=50, output_tokens=12,
                           latency_ms=200.0)
        fake = _FakeLLMClient([resp])
        agent = AuthorityVerifierAgent(llm_client=fake)

        verdict = agent.verify_agent_proposal(
            proposal=_OVER_BUDGET_PROPOSAL, budget=_NARROW_BUDGET)

        assert verdict.response is not None
        assert verdict.response.input_tokens == 50
        assert verdict.response.output_tokens == 12
        assert verdict.response.latency_ms == 200.0

    def test_no_network_or_api_key_required(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        fake = _FakeLLMClient([_structured(scope="/reports/2026-08/")])
        agent = AuthorityVerifierAgent(llm_client=fake)

        agent.verify_agent_proposal(proposal=_OVER_BUDGET_PROPOSAL,
                                    budget=_NARROW_BUDGET)  # 예외 없어야 함

    def test_llm_called_at_most_once_even_on_recheck_failure(self):
        """단일 제안 → 단일 재검증. controlled run_feedback() 처럼 여러 라운드
        묻지 않는다 — SemanticVerifierAgent.verify_agent_proposal() 과 같은
        1-call 구조."""
        fake = _FakeLLMClient([_structured(scope="/")])  # 재검증에서 거부됨
        agent = AuthorityVerifierAgent(llm_client=fake)

        agent.verify_agent_proposal(proposal=_OVER_BUDGET_PROPOSAL, budget=_NARROW_BUDGET)

        assert len(fake.calls) == 1


class TestNoForbiddenInputAccess:
    def test_no_forbidden_parameter_in_public_api(self):
        """truth/ground_truth/label/semantic(verdict)/principal(intent)을
        받을 방법이 시그니처에 없어야 한다. `budget`/`proposal`은 검증 대상
        그 자체이므로 forbidden에서 제외한다."""
        forbidden = ("truth", "ground_truth", "label", "semantic", "principal")
        sig = inspect.signature(AuthorityVerifierAgent.verify_agent_proposal)
        for name in sig.parameters:
            lowered = name.lower()
            assert not any(f in lowered for f in forbidden), (
                f"verify_agent_proposal has a suspicious parameter: {name}")


class TestControlledPathUnaffected:
    """llm_client를 넘기지 않는 한 기존 verify() 동작에는 아무 영향이
    없어야 한다 — 생성자에 필드 하나가 추가됐을 뿐이다."""

    def test_default_construction_has_no_llm_client(self):
        agent = AuthorityVerifierAgent()
        assert agent.llm_client is None

    def test_verify_signature_unchanged(self):
        sig = inspect.signature(AuthorityVerifierAgent.verify)
        assert list(sig.parameters) == [
            "self", "interpretation", "budget", "principal", "log", "key"]
