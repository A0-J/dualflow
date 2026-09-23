"""HistoricalEvidenceComparator — v3 prototype. 전부 fake LLMClient로
검증한다: 네트워크도, OPENAI_API_KEY도 필요 없다.

핵심으로 확인해야 하는 것: 이 모듈은 candidate를 절대 새로 생성하지 않고
(judge()가 받은 Interpretation을 그대로 결과에 담아 돌려준다), Delegate
candidate generation이나 Authority/Budget과 구조적으로 완전히 무관하다."""

from __future__ import annotations

import inspect

import pytest

from dualflow.experience_evidence import (
    EvidenceJudgment,
    EvidenceRelation,
    HistoricalEvidenceComparator,
    parse_evidence_relation,
)
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


_SUMMARIZE = Interpretation("summarize", "file", "/reports/2026-09/", frozenset())
_EXPORT = Interpretation("export", "file", "/reports/2026-09/", frozenset())


class TestParseEvidenceRelation:
    @pytest.mark.parametrize("text,expected", [
        ("SUPPORT", EvidenceRelation.SUPPORT),
        ("support", EvidenceRelation.SUPPORT),
        ("  Support.\n", EvidenceRelation.SUPPORT),
        ("CONFLICT", EvidenceRelation.CONFLICT),
        ("IRRELEVANT", EvidenceRelation.IRRELEVANT),
        ("UNCERTAIN", EvidenceRelation.UNCERTAIN),
        ('"SUPPORT"', EvidenceRelation.SUPPORT),
    ])
    def test_parses_exact_and_lightly_decorated_labels(self, text, expected):
        assert parse_evidence_relation(text) == expected

    @pytest.mark.parametrize("text", [
        "",
        "   ",
        "I think this supports it.",
        "SUPPORT CONFLICT",
        "ACTION: summarize",  # a generation-style answer must NOT parse
        "MAYBE",
        "5",
        "BANANA",  # a nonsense out-of-contract single word
        "SUPPORT because it aligns with the confirmed meaning",  # valid
        # word followed by free-form justification -- must not be treated
        # as a bare SUPPORT
    ])
    def test_rejects_anything_not_exactly_one_of_the_four_labels(self, text):
        with pytest.raises(ValueError):
            parse_evidence_relation(text)

    def test_malformed_response_never_falls_back_to_support_or_conflict(self):
        """가장 위험한 실패 모드를 직접 확인한다: OOV/malformed 응답이
        조용히 SUPPORT나 CONFLICT로 fallback되면 안 된다. ValueError로
        fail-closed하는 것이 유일하게 허용된 동작이다 -- UNCERTAIN으로도
        자동 치환하지 않는다(그건 모델이 실제로 UNCERTAIN이라고 답했을
        때만 나오는 값이어야 한다)."""
        for bad in ("BANANA", "SUPPORT because it aligns with the confirmed meaning"):
            with pytest.raises(ValueError):
                parse_evidence_relation(bad)

    def test_exactly_four_relations_exist(self):
        """다섯 번째 값(예: 새 action 이름)이 추가되면 안 된다 — 닫힌
        분류 문제라는 설계 자체가 구조적 안전장치다."""
        assert {r.value for r in EvidenceRelation} == {
            "SUPPORT", "CONFLICT", "IRRELEVANT", "UNCERTAIN"}


class TestHistoricalEvidenceComparatorJudge:
    def test_calls_llm_exactly_once_and_returns_judgment(self):
        fake = _FakeLLMClient([LLMResponse(text="SUPPORT")])
        comparator = HistoricalEvidenceComparator(llm=fake)

        result = comparator.judge(
            delegation="Please prepare the September 2026 financial report.",
            candidate=_SUMMARIZE, evidence_text="some evidence text")

        assert isinstance(result, EvidenceJudgment)
        assert result.relation == EvidenceRelation.SUPPORT
        assert len(fake.calls) == 1

    def test_returns_the_exact_candidate_passed_in_unchanged(self):
        """핵심 invariant: judge()는 candidate를 그대로 돌려줄 뿐, 새
        Interpretation을 만들지 않는다."""
        fake = _FakeLLMClient([LLMResponse(text="CONFLICT")])
        comparator = HistoricalEvidenceComparator(llm=fake)

        result = comparator.judge(
            delegation="d", candidate=_EXPORT, evidence_text="e")

        assert result.candidate is _EXPORT

    def test_input_contains_delegation_candidate_and_evidence_text(self):
        fake = _FakeLLMClient([LLMResponse(text="IRRELEVANT")])
        comparator = HistoricalEvidenceComparator(llm=fake)

        comparator.judge(
            delegation="my delegation text", candidate=_SUMMARIZE,
            evidence_text="my evidence block text")

        sent = fake.calls[0]["input_text"]
        assert "my delegation text" in sent
        assert "ACTION: summarize" in sent
        assert "my evidence block text" in sent

    def test_instructions_never_ask_the_model_to_propose_a_new_interpretation(self):
        """프롬프트 자체가 "새 해석을 제안하지 말라"고 명시하는지 확인 —
        생성이 아니라 분류 task라는 설계 의도가 실제 프롬프트에도 있어야
        한다."""
        fake = _FakeLLMClient([LLMResponse(text="SUPPORT")])
        comparator = HistoricalEvidenceComparator(llm=fake)

        comparator.judge(delegation="d", candidate=_SUMMARIZE, evidence_text="e")

        instructions = fake.calls[0]["instructions"]
        assert "NOT deciding what the current delegation means" in instructions
        assert "must NOT" in instructions

    def test_propagates_parse_failure_instead_of_swallowing_it(self):
        """DelegateAgent.sample_candidates()의 fail-closed 관례와 동일 —
        파싱 실패를 조용히 어떤 기본값으로 바꾸지 않는다. 호출자가 이
        예외를 잡아서 "판단 불가 샘플"로 세고 제외해야 한다."""
        fake = _FakeLLMClient([LLMResponse(text="this is not a valid label")])
        comparator = HistoricalEvidenceComparator(llm=fake)

        with pytest.raises(ValueError):
            comparator.judge(delegation="d", candidate=_SUMMARIZE, evidence_text="e")

    @pytest.mark.parametrize("bad_response", [
        "BANANA",
        "SUPPORT because it aligns with the confirmed meaning",
    ])
    def test_judge_never_falls_back_on_malformed_response_end_to_end(self, bad_response):
        """parse_evidence_relation()뿐 아니라 judge() 전체 경로에서도
        malformed 응답이 SUPPORT/CONFLICT로 조용히 fallback되지 않는지
        확인한다."""
        fake = _FakeLLMClient([LLMResponse(text=bad_response)])
        comparator = HistoricalEvidenceComparator(llm=fake)

        with pytest.raises(ValueError):
            comparator.judge(delegation="d", candidate=_SUMMARIZE, evidence_text="e")

    @pytest.mark.parametrize("relation_text,expected", [
        ("SUPPORT", EvidenceRelation.SUPPORT),
        ("CONFLICT", EvidenceRelation.CONFLICT),
        ("IRRELEVANT", EvidenceRelation.IRRELEVANT),
        ("UNCERTAIN", EvidenceRelation.UNCERTAIN),
    ])
    def test_all_four_relations_round_trip(self, relation_text, expected):
        fake = _FakeLLMClient([LLMResponse(text=relation_text)])
        comparator = HistoricalEvidenceComparator(llm=fake)

        result = comparator.judge(delegation="d", candidate=_EXPORT, evidence_text="e")

        assert result.relation == expected


class TestStructuralIndependence:
    def test_module_does_not_reference_delegate_agent_or_generation(self):
        """이 모듈은 candidate generation에 관여하지 않는다 — candidate는
        항상 호출자가 이미 만들어서 넘겨준다. DelegateAgent를 참조하지
        않는다는 걸 실제 import 문(모듈 namespace)으로 확인한다 — docstring
        문구가 아니라."""
        import dualflow.experience_evidence as mod

        assert not hasattr(mod, "DelegateAgent")
        assert not hasattr(mod, "parse_structured_action")

    def test_module_does_not_reference_authority_or_budget(self):
        import dualflow.experience_evidence as mod

        assert not hasattr(mod, "Budget")
        assert not hasattr(mod, "check_authority")
        assert not hasattr(mod, "AuthorityVerifierAgent")
        assert not hasattr(mod, "AuthorityVerdict")

    def test_module_does_not_reference_agent_delegation_runtime(self):
        import dualflow.experience_evidence as mod

        assert not hasattr(mod, "AgentDelegationRuntime")

    def test_judge_signature_has_no_truth_or_ground_truth_parameter(self):
        forbidden = ("truth", "ground_truth", "label", "expected_action", "verdict")
        sig = inspect.signature(HistoricalEvidenceComparator.judge)
        for name in sig.parameters:
            lowered = name.lower()
            assert not any(f in lowered for f in forbidden), (
                f"judge() has a suspicious parameter: {name}")
