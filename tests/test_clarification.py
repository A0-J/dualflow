"""ClarifyingDelegate — B7b. Principal-Delegate 단일 라운드 clarification
루프. 전부 fake LLMClient로 검증한다: 네트워크도, OPENAI_API_KEY도
필요 없다. 아직 경험(verified experience) 저장/조회는 없다 — 이 파일은
"질문 한 번이 실제로 uncertainty를 줄이는가"만 검증한다.
"""

from __future__ import annotations

import inspect

import pytest

from dualflow.clarification import ClarificationResult, ClarifyingDelegate
from dualflow.delegate_agent import DelegateAgent
from dualflow.llm import LLMResponse
from dualflow.principal_agent import PrincipalAgent
from dualflow.semantic import Interpretation


class _FakeLLMClient:
    """`llm.LLMClient` 프로토콜을 흉내 내는 순수 Python fake. 미리 준비한
    응답을 호출 순서대로 돌려준다 — 실제 model 호출도, 네트워크도 없다.
    Principal/Delegate는 각자 별도 인스턴스를 쓴다."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, *, instructions: str, input_text: str) -> LLMResponse:
        self.calls.append({"instructions": instructions, "input_text": input_text})
        return self._responses.pop(0)


def _structured(action: str, resource: str = "file", scope: str = "/reports/2026-08/",
                condition: str = "none") -> LLMResponse:
    return LLMResponse(
        text=f"ACTION: {action}\nRESOURCE: {resource}\nSCOPE: {scope}\nCONDITION: {condition}")


SUMMARIZE = Interpretation("summarize", "file", "/reports/2026-08/", frozenset())
EXPORT = Interpretation("export", "file", "/reports/2026-08/", frozenset())


def _make(principal_llm, delegate_llm, n=10, entropy_threshold=0.8) -> ClarifyingDelegate:
    return ClarifyingDelegate(
        principal=PrincipalAgent(llm=principal_llm),
        delegate=DelegateAgent(llm=delegate_llm),
        n=n, entropy_threshold=entropy_threshold)


class TestLowEntropy:
    def test_no_clarification_when_entropy_is_at_or_below_threshold(self):
        delegate_llm = _FakeLLMClient([_structured("summarize") for _ in range(10)])
        principal_llm = _FakeLLMClient([])  # 절대 호출되면 안 된다

        clarifier = _make(principal_llm, delegate_llm, n=10, entropy_threshold=0.8)
        result = clarifier.resolve(goal="x", context="", delegation="Please prepare the report.")

        assert isinstance(result, ClarificationResult)
        assert result.clarified is False
        assert result.pre_distribution.entropy == 0.0
        assert result.post_distribution is None
        assert result.question is None
        assert result.answer is None
        assert result.final_interpretation == SUMMARIZE
        assert len(principal_llm.calls) == 0
        assert len(delegate_llm.calls) == 10


class TestHighEntropyTriggersClarification:
    def test_clarification_round_happens_and_returns_question_and_answer(self):
        delegate_llm = _FakeLLMClient(
            [_structured("summarize") for _ in range(5)]
            + [_structured("export") for _ in range(5)]          # pre: H=1.0
            + [LLMResponse(text="Internal summary or export?")]  # ask_clarification()
            + [_structured("summarize") for _ in range(10)])     # post: H=0.0
        principal_llm = _FakeLLMClient([LLMResponse(text="Create an internal summary only.")])

        clarifier = _make(principal_llm, delegate_llm, n=10, entropy_threshold=0.8)
        result = clarifier.resolve(goal="Prepare the report.", context="finance team",
                                   delegation="Please prepare the report.")

        assert result.clarified is True
        assert result.pre_distribution.entropy == pytest.approx(1.0)
        assert result.question is not None
        assert result.question.question == "Internal summary or export?"
        assert result.answer is not None
        assert result.answer.answer == "Create an internal summary only."
        assert result.post_distribution is not None
        assert result.post_distribution.entropy == 0.0
        assert result.final_interpretation == SUMMARIZE
        assert len(delegate_llm.calls) == 21  # 10 + 1 + 10
        assert len(principal_llm.calls) == 1


class TestIndependence:
    def test_clarification_question_generation_never_sees_goal(self):
        """ask_clarification()은 delegation/context/분포만 본다 — goal은
        answer_clarification()에만 전달된다. goal에만 있는 고유 문자열이
        delegate 쪽 호출 어디에도 새어 들어가면 안 된다."""
        delegate_llm = _FakeLLMClient(
            [_structured("summarize") for _ in range(5)]
            + [_structured("export") for _ in range(5)]
            + [LLMResponse(text="?")]
            + [_structured("summarize") for _ in range(10)])
        principal_llm = _FakeLLMClient([LLMResponse(text="Internal summary only.")])

        clarifier = _make(principal_llm, delegate_llm, n=10, entropy_threshold=0.8)
        clarifier.resolve(goal="UNIQUE_GOAL_MARKER_XYZ", context="",
                          delegation="Please prepare the report.")

        for call in delegate_llm.calls:
            assert "UNIQUE_GOAL_MARKER_XYZ" not in call["input_text"]

    def test_principal_answer_never_sees_candidate_probabilities(self):
        """answer_clarification()은 question 텍스트만 받는다 — B의 확률
        분포 수치("0.60" 같은)는 Principal에게 전달되지 않는다."""
        delegate_llm = _FakeLLMClient(
            [_structured("summarize") for _ in range(6)]
            + [_structured("export") for _ in range(4)]
            + [LLMResponse(text="Internal summary or export?")]
            + [_structured("summarize") for _ in range(10)])
        principal_llm = _FakeLLMClient([LLMResponse(text="Internal summary only.")])

        clarifier = _make(principal_llm, delegate_llm, n=10, entropy_threshold=0.8)
        clarifier.resolve(goal="x", context="", delegation="Please prepare the report.")

        assert len(principal_llm.calls) == 1
        input_text = principal_llm.calls[0]["input_text"]
        assert "0.60" not in input_text
        assert "0.40" not in input_text
        # question 텍스트 자체는 당연히 보인다.
        assert "Internal summary or export?" in input_text


class TestPostClarificationConvergence:
    def test_entropy_drops_after_clarification(self):
        delegate_llm = _FakeLLMClient(
            [_structured("summarize") for _ in range(5)]
            + [_structured("export") for _ in range(5)]           # H_before = 1.0
            + [LLMResponse(text="Internal summary or export?")]
            + [_structured("summarize") for _ in range(10)])      # H_after = 0.0
        principal_llm = _FakeLLMClient(
            [LLMResponse(text="Create an internal summary only; do not export.")])

        clarifier = _make(principal_llm, delegate_llm, n=10, entropy_threshold=0.8)
        result = clarifier.resolve(goal="x", context="", delegation="Please prepare the report.")

        assert result.pre_distribution.entropy == pytest.approx(1.0)
        assert result.post_distribution.entropy == 0.0
        assert result.final_interpretation == SUMMARIZE

    def test_enriched_context_carries_the_answer_into_post_sampling(self):
        """post-sampling 호출의 context에 Principal의 답변이 실제로
        포함되는지 직접 확인한다."""
        delegate_llm = _FakeLLMClient(
            [_structured("summarize") for _ in range(5)]
            + [_structured("export") for _ in range(5)]
            + [LLMResponse(text="Internal summary or export?")]
            + [_structured("summarize") for _ in range(10)])
        principal_llm = _FakeLLMClient(
            [LLMResponse(text="Create an internal summary only; do not export.")])

        clarifier = _make(principal_llm, delegate_llm, n=10, entropy_threshold=0.8)
        clarifier.resolve(goal="x", context="original env context",
                          delegation="Please prepare the report.")

        post_calls = delegate_llm.calls[11:]  # 10(pre) + 1(question) 다음부터
        assert len(post_calls) == 10
        for call in post_calls:
            assert "original env context" in call["input_text"]
            assert "Create an internal summary only; do not export." in call["input_text"]


class TestStillAmbiguousAfterClarification:
    def test_returns_honestly_without_a_second_round(self):
        delegate_llm = _FakeLLMClient(
            [_structured("summarize") for _ in range(5)]
            + [_structured("export") for _ in range(5)]           # pre: H=1.0
            + [LLMResponse(text="Internal summary or export?")]
            + [_structured("summarize") for _ in range(6)]
            + [_structured("export") for _ in range(4)])          # post: 여전히 섞임
        principal_llm = _FakeLLMClient([LLMResponse(text="Not sure, use your judgment.")])

        clarifier = _make(principal_llm, delegate_llm, n=10, entropy_threshold=0.8)
        result = clarifier.resolve(goal="x", context="", delegation="Please prepare the report.")

        assert result.clarified is True
        assert result.post_distribution.entropy > 0.0  # 여전히 애매함을 그대로 보고한다
        assert result.final_interpretation == SUMMARIZE  # 6/10으로 더 많은 쪽
        # 딱 1라운드만 — 두 번째 질문을 만들지 않는다.
        assert len(delegate_llm.calls) == 21
        assert len(principal_llm.calls) == 1


class TestConfigurableThreshold:
    def test_same_distribution_clarifies_or_not_depending_on_threshold(self):
        # 두 실행 모두 pre-distribution은 6/4(H≈0.97)로 동일하게 만든다.
        def _pre_responses():
            return ([_structured("summarize") for _ in range(6)]
                    + [_structured("export") for _ in range(4)])

        low_threshold_delegate = _FakeLLMClient(
            _pre_responses()
            + [LLMResponse(text="?")]
            + [_structured("summarize") for _ in range(10)])
        low_threshold_principal = _FakeLLMClient([LLMResponse(text="Internal summary only.")])
        clarifier_low = _make(low_threshold_principal, low_threshold_delegate,
                              n=10, entropy_threshold=0.5)
        result_low = clarifier_low.resolve(goal="x", context="",
                                           delegation="Please prepare the report.")
        assert result_low.pre_distribution.entropy > 0.5
        assert result_low.clarified is True

        high_threshold_delegate = _FakeLLMClient(_pre_responses())  # clarification 없음
        high_threshold_principal = _FakeLLMClient([])
        clarifier_high = _make(high_threshold_principal, high_threshold_delegate,
                               n=10, entropy_threshold=1.5)
        result_high = clarifier_high.resolve(goal="x", context="",
                                             delegation="Please prepare the report.")
        assert result_high.pre_distribution.entropy <= 1.5
        assert result_high.clarified is False
        assert len(high_threshold_principal.calls) == 0


class TestNoGroundTruthAPI:
    def test_resolve_signature_has_no_truth_or_label_parameter(self):
        forbidden = ("truth", "ground_truth", "label", "expected")
        sig = inspect.signature(ClarifyingDelegate.resolve)
        for name in sig.parameters:
            lowered = name.lower()
            assert not any(f in lowered for f in forbidden), (
                f"resolve() has a suspicious parameter: {name}")

    def test_result_has_no_ground_truth_field(self):
        forbidden = ("truth", "ground_truth", "label", "expected")
        for f in ClarificationResult.__dataclass_fields__:
            lowered = f.lower()
            assert not any(bad in lowered for bad in forbidden), (
                f"ClarificationResult has a suspicious field: {f}")

    def test_no_network_or_api_key_required(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        delegate_llm = _FakeLLMClient([_structured("summarize") for _ in range(3)])
        principal_llm = _FakeLLMClient([])
        clarifier = _make(principal_llm, delegate_llm, n=3, entropy_threshold=0.8)

        clarifier.resolve(goal="x", context="", delegation="y")  # 예외 없어야 함
