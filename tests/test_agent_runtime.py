"""AgentDelegationRuntime — Phase B의 oracle-free end-to-end runtime. 전부
fake LLMClient로 검증한다: 네트워크도, OPENAI_API_KEY도 필요 없다.

가장 중요한 테스트는 TestSilentMisread다 — Delegate와 Semantic Verifier가
둘 다 같은 잘못(export)에 동의하고 Authority도 허용해도, Principal의 독립
재구성(read)이 최종 실행 후보와 다르면 REJECT돼야 한다. 이게 controlled
경로에서 `SemanticVerdict.interpretation`을 오라클 대신 쓰다가 실패했던
바로 그 실패 모드를, task.truth 없이 실제로 막는지 보여주는 테스트다.
"""

from __future__ import annotations

import inspect

from dualflow.agent_runtime import AgentDelegationRuntime, AgentRuntimeResult
from dualflow.authority_feedback import AuthorityVerifierAgent
from dualflow.capability import Budget, Privilege
from dualflow.delegate_agent import DelegateAgent
from dualflow.framework import EXECUTE, REJECT
from dualflow.llm import LLMResponse
from dualflow.principal_agent import PrincipalAgent
from dualflow.semantic import Interpretation, SemanticVerifierAgent


class _FakeLLMClient:
    """`llm.LLMClient` 프로토콜을 흉내 내는 순수 Python fake. 미리 준비한
    응답을 호출 순서대로 돌려준다 — 실제 model 호출도, 네트워크도 없다.
    Agent마다 별도 인스턴스를 만들어 쓴다 — 같은 fake를 공유하지 않는다."""

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


def _make_runtime(principal_llm, delegate_llm, semantic_llm, authority_llm
                  ) -> AgentDelegationRuntime:
    """네 개의 서로 다른 fake LLMClient로 완전히 독립된 네 Agent를 만든다."""
    return AgentDelegationRuntime(
        principal=PrincipalAgent(llm=principal_llm),
        delegate=DelegateAgent(llm=delegate_llm),
        semantic_verifier=SemanticVerifierAgent(llm_client=semantic_llm),
        authority_verifier=AuthorityVerifierAgent(llm_client=authority_llm),
    )


class TestBenignExecution:
    def test_everything_agrees_and_within_budget_executes(self):
        # Principal 독립 재구성, Delegate proposal, Semantic 독립 판단이
        # 전부 read/file/2026-08 로 일치하고, budget도 그걸 정확히
        # 허용한다 — Authority는 즉시 allow(협상 불필요, LLM 0회).
        principal_llm = _FakeLLMClient([
            LLMResponse(text="Please read last month's report."),   # delegate()
            _structured("read"),                                    # restate_intent()
        ])
        delegate_llm = _FakeLLMClient([_structured("read")])
        semantic_llm = _FakeLLMClient([_structured("read")])
        authority_llm = _FakeLLMClient([])  # 호출되지 않아야 한다

        runtime = _make_runtime(principal_llm, delegate_llm, semantic_llm, authority_llm)
        budget = Budget.of(Privilege("read", "file", "/reports/2026-08/"))

        result = runtime.run(goal="Read last month's report.", context="finance team",
                             budget=budget)

        assert isinstance(result, AgentRuntimeResult)
        assert result.decision == EXECUTE
        assert result.principal_match is True
        assert result.final_interpretation == Interpretation(
            "read", "file", "/reports/2026-08/", frozenset())
        assert len(authority_llm.calls) == 0  # 이미 허용됨 — LLM 호출 불필요


class TestSilentMisread:
    """가장 중요한 테스트. Delegate와 Semantic이 둘 다 잘못된 action(export)
    에 동의하고 Authority도 그걸 허용하지만, Principal의 독립 재구성만은
    delegation/proposal과 완전히 무관하게 만들어졌으므로 진짜 의도(read)를
    유지한다 — 그래서 REJECT돼야 한다."""

    def test_semantic_and_authority_both_wrong_but_principal_reference_catches_it(self):
        principal_llm = _FakeLLMClient([
            LLMResponse(text="Please handle last month's report as needed."),  # delegate()
            _structured("read"),                                               # restate_intent() — 진짜 의도
        ])
        # Delegate가 confidently 잘못 이해한다 — export.
        delegate_llm = _FakeLLMClient([_structured("export")])
        # Semantic Verifier도 독립적으로 판단했지만 *같은* 실수를 반복한다
        # — 자기 자신과 비교하면 tautology가 되는 바로 그 상황.
        semantic_llm = _FakeLLMClient([_structured("export")])
        # Authority 입장에서는 export가 실제로 허용된 권한이다(하드 리젝트
        # 아님) — 그래서 즉시 allow, LLM 호출도 없다.
        authority_llm = _FakeLLMClient([])

        runtime = _make_runtime(principal_llm, delegate_llm, semantic_llm, authority_llm)
        budget = Budget.of(Privilege("export", "file", "/reports/2026-08/"))

        result = runtime.run(goal="Handle last month's report.", context="finance team",
                             budget=budget)

        # Delegate와 Semantic 둘 다 export에 동의했고 Authority도 허용했다
        # — 그런데도 REJECT여야 한다.
        assert result.semantic_verdict.confirmed is True
        assert result.authority_verdict.allowed is True
        assert result.principal_match is False
        assert result.decision == REJECT
        assert "Principal" in result.reason

        # task.truth 없이, 오직 독립적으로 재구성된 Principal 의도만으로
        # 이 misread가 잡혔다는 걸 명시적으로 확인한다.
        assert result.principal_intent.intended_action.action == "read"
        assert result.final_interpretation.action == "export"


class TestSemanticReject:
    def test_semantic_unconfirmed_rejects_regardless_of_authority(self):
        principal_llm = _FakeLLMClient([
            LLMResponse(text="Please read the report."),
            _structured("read"),
        ])
        delegate_llm = _FakeLLMClient([_structured("export")])
        # Semantic의 독립 판단이 B의 proposal과 다르다 — confirmed=False.
        semantic_llm = _FakeLLMClient([_structured("read")])
        # budget이 export를 완전히 허용해도(Authority는 즉시 allow) 상관없다.
        authority_llm = _FakeLLMClient([])

        runtime = _make_runtime(principal_llm, delegate_llm, semantic_llm, authority_llm)
        budget = Budget.of(Privilege("export", "file", "/reports/2026-08/"))

        result = runtime.run(goal="Read the report.", context="", budget=budget)

        assert result.semantic_verdict.confirmed is False
        assert result.authority_verdict.allowed is True  # Authority는 통과했지만
        assert result.decision == REJECT                  # Semantic 게이트에서 이미 막힘
        assert "의미" in result.reason


class TestAuthorityReject:
    def test_authority_denied_rejects_regardless_of_semantic(self):
        principal_llm = _FakeLLMClient([
            LLMResponse(text="Please delete the old report."),
            _structured("delete"),
        ])
        delegate_llm = _FakeLLMClient([_structured("delete")])
        # Semantic은 완전히 동의한다(confirmed=True).
        semantic_llm = _FakeLLMClient([_structured("delete")])
        authority_llm = _FakeLLMClient([])  # no_grant — LLM 호출 없이 즉시 거부

        runtime = _make_runtime(principal_llm, delegate_llm, semantic_llm, authority_llm)
        # budget에는 delete 권한이 아예 없다 — read만 위임됨.
        budget = Budget.of(Privilege("read", "file", "/reports/2026-08/"))

        result = runtime.run(goal="Delete the old report.", context="", budget=budget)

        assert result.semantic_verdict.confirmed is True
        assert result.authority_verdict.allowed is False
        assert result.decision == REJECT
        assert "권한" in result.reason


class TestAuthorityNarrowing:
    """Authority가 협상으로 proposal을 좁혔을 때, fusion이 B의 원래(넓은)
    proposal이 아니라 좁혀진 authority_verdict.interpretation을 최종
    후보로 써야 한다."""

    def test_final_candidate_is_the_negotiated_interpretation_not_the_raw_proposal(self):
        principal_llm = _FakeLLMClient([
            LLMResponse(text="Please read last month's report."),
            _structured("read", scope="/reports/2026-08/"),  # 진짜 의도는 좁은 scope
        ])
        # Delegate는 더 넓게 제안한다 — scope_exceeded를 유발.
        delegate_llm = _FakeLLMClient([_structured("read", scope="/reports/")])
        # Semantic은 B의 (넓은) proposal에 동의한다 — Semantic 게이트는
        # 이 테스트의 관심사가 아니다.
        semantic_llm = _FakeLLMClient([_structured("read", scope="/reports/")])
        # Authority LLM이 ceiling(=/reports/2026-08/, budget이 허용하는
        # 만큼)까지 좁혀서 제안 → 재검증 통과.
        authority_llm = _FakeLLMClient([_structured("read", scope="/reports/2026-08/")])

        runtime = _make_runtime(principal_llm, delegate_llm, semantic_llm, authority_llm)
        budget = Budget.of(Privilege("read", "file", "/reports/2026-08/"))

        result = runtime.run(goal="Read last month's report.", context="", budget=budget)

        assert len(authority_llm.calls) == 1  # scope_exceeded 협상이 실제로 1회 일어났다
        assert result.authority_verdict.negotiated is True
        assert result.proposal.interpretation.scope == "/reports/"        # B의 원래 제안(넓음)
        assert result.final_interpretation.scope == "/reports/2026-08/"   # 협상 후 좁혀진 값
        # Principal의 독립 재구성(좁은 scope)과 "좁혀진 뒤의" 최종 action이
        # 일치하므로 EXECUTE — 원래 넓은 proposal과 비교했다면 불일치였을
        # 것이다.
        assert result.principal_match is True
        assert result.decision == EXECUTE


class TestPrincipalIndependence:
    def test_restate_intent_call_does_not_mention_delegates_wrong_action(self):
        """restate_intent()는 delegate()보다 먼저 실행되고, 그 시점엔
        Delegate의 proposal이 코드상으로도 존재하지 않는다 — 실제로
        해당 호출의 input_text에 Delegate가 나중에 제안하는 값("export")
        이 전혀 등장하지 않는다는 걸 직접 확인한다."""
        principal_llm = _FakeLLMClient([
            LLMResponse(text="Please handle the report."),
            _structured("read"),
        ])
        delegate_llm = _FakeLLMClient([_structured("export")])
        semantic_llm = _FakeLLMClient([_structured("export")])
        authority_llm = _FakeLLMClient([])

        runtime = _make_runtime(principal_llm, delegate_llm, semantic_llm, authority_llm)
        budget = Budget.of(Privilege("export", "file", "/reports/2026-08/"))
        runtime.run(goal="Handle the report.", context="finance team", budget=budget)

        assert len(principal_llm.calls) == 2
        delegate_call, restate_call = principal_llm.calls
        assert "export" not in delegate_call["input_text"]
        assert "export" not in restate_call["input_text"]
        # 두 호출은 서로 다른 instructions(별도 목적의 call)이었다.
        assert delegate_call["instructions"] != restate_call["instructions"]


class TestVerifierIndependence:
    def test_semantic_and_authority_inputs_never_mention_principals_reference(self):
        """Principal의 독립 재구성(read)이 Delegate/Semantic이 실제로
        다루는 값(export)과 다르게 설계된 시나리오를 그대로 재사용해서,
        Semantic/Authority에 전달된 input_text 어디에도 "read"가 새어
        들어가지 않았음을 직접 확인한다."""
        principal_llm = _FakeLLMClient([
            LLMResponse(text="Please handle the report."),
            _structured("read"),   # Principal의 진짜 의도 — Semantic/Authority는 이걸 모른다
        ])
        delegate_llm = _FakeLLMClient([_structured("export")])
        semantic_llm = _FakeLLMClient([_structured("export")])
        authority_llm = _FakeLLMClient([])

        runtime = _make_runtime(principal_llm, delegate_llm, semantic_llm, authority_llm)
        budget = Budget.of(Privilege("export", "file", "/reports/2026-08/"))
        runtime.run(goal="Handle the report.", context="finance team", budget=budget)

        assert len(semantic_llm.calls) == 1
        assert "read" not in semantic_llm.calls[0]["input_text"]
        # authority_llm은 이 시나리오에서 아예 호출되지 않는다(즉시 allow)
        # — 호출 자체가 없었다는 것도 독립성의 일부다.
        assert len(authority_llm.calls) == 0

    def test_semantic_never_receives_authority_verdict_and_vice_versa(self):
        """시그니처 자체를 점검한다 — AuthorityVerdict/SemanticVerdict를
        서로에게 넘길 방법이 아예 없어야 한다(B4/B5에서 이미 검증된
        내용이지만, runtime 조립 지점에서도 다시 확인한다)."""
        sem_sig = inspect.signature(SemanticVerifierAgent.verify_agent_proposal)
        auth_sig = inspect.signature(AuthorityVerifierAgent.verify_agent_proposal)
        for name in sem_sig.parameters:
            assert "authority" not in name.lower() and "verdict" not in name.lower()
        for name in auth_sig.parameters:
            assert "semantic" not in name.lower() and "verdict" not in name.lower()


class TestNoGroundTruthAPI:
    def test_run_signature_has_no_truth_or_label_parameter(self):
        forbidden = ("truth", "ground_truth", "label", "expected")
        sig = inspect.signature(AgentDelegationRuntime.run)
        for name in sig.parameters:
            lowered = name.lower()
            assert not any(f in lowered for f in forbidden), (
                f"run() has a suspicious parameter: {name}")

    def test_result_has_no_ground_truth_field(self):
        forbidden = ("truth", "ground_truth", "label", "expected")
        for f in AgentRuntimeResult.__dataclass_fields__:
            lowered = f.lower()
            assert not any(bad in lowered for bad in forbidden), (
                f"AgentRuntimeResult has a suspicious field: {f}")


class TestGroundTruthInvariance:
    """run()에 ground_truth를 넘길 방법 자체가 없으므로, 외부에서 같은
    입력(goal/context/budget)과 같은 model 응답 스크립트로 두 번 실행하면
    결과가 반드시 동일해야 한다 — 어디에도 "평가 라벨"이 끼어들 자리가
    없다는 걸 직접 보여준다."""

    def _run_once(self) -> AgentRuntimeResult:
        principal_llm = _FakeLLMClient([
            LLMResponse(text="Please read last month's report."),
            _structured("read"),
        ])
        delegate_llm = _FakeLLMClient([_structured("read")])
        semantic_llm = _FakeLLMClient([_structured("read")])
        authority_llm = _FakeLLMClient([])
        runtime = _make_runtime(principal_llm, delegate_llm, semantic_llm, authority_llm)
        budget = Budget.of(Privilege("read", "file", "/reports/2026-08/"))
        return runtime.run(goal="Read last month's report.", context="finance team",
                           budget=budget)

    def test_identical_inputs_produce_identical_decision(self):
        # 두 번 실행한다 — 마치 서로 다른 두 개의 평가 라벨을 "붙였다"고
        # 상상해도, run()은 애초에 그런 라벨을 받지 않으므로 결과에
        # 영향을 줄 방법이 없다.
        result_a = self._run_once()  # "라벨 A"라고 가정해도 run()은 모른다
        result_b = self._run_once()  # "라벨 B"라고 가정해도 run()은 모른다

        assert result_a.decision == result_b.decision == EXECUTE
        assert result_a.final_interpretation == result_b.final_interpretation
        assert result_a.principal_match == result_b.principal_match
