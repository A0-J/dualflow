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

from dualflow.agent_experience import AgentExperience, AgentExperienceStore
from dualflow.agent_runtime import AgentDelegationRuntime, AgentRuntimeResult
from dualflow.authority_feedback import AuthorityVerifierAgent
from dualflow.capability import Budget, Privilege
from dualflow.delegate_agent import CandidateDistribution, DelegateAgent
from dualflow.experience_decision import AMBIGUOUS_REQUIRES_CLARIFICATION
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


# --------------------------------------------------------------------------
# Runtime Integration Phase 1 (docs/experiments/agent_connected_eval.md §26)
# -- opt-in entropy-based clarification + Option B advisory-only evidence,
# wired into AgentDelegationRuntime for the first time. Everything above
# this line is unchanged from before this phase and continues to exercise
# use_clarification=False (the default) exclusively.
# --------------------------------------------------------------------------

def _make_experience(action: str, *, principal_id: str = "p1",
                     task_category: str = "cat1") -> AgentExperience:
    """`tests/test_experience_decision.py`의 `_experience()` 헬퍼와 같은
    패턴 -- pre/post_distribution은 필드를 채우기만 하면 되는 더미다(이
    experience의 confirmed_facets/confirmed_interpretation만 실제로
    쓰인다)."""
    interp = Interpretation(action, "file", "/reports/2026-08/", frozenset())
    dummy = CandidateDistribution(belief={interp: 1.0}, entropy=0.0, top=interp,
                                  top_probability=1.0, n_unique=1, n_samples=1, responses=[])
    return AgentExperience(
        principal_id=principal_id, task_category=task_category,
        delegation="Please prepare last month's report.",
        clarification_question="Should I export or summarize?",
        principal_answer=f"Please {action} it.",
        confirmed_interpretation=interp,
        pre_distribution=dummy, post_distribution=dummy,
        confirmed_facets=frozenset({"action"}))


class TestClarificationOptOut:
    """`use_clarification`을 아예 안 넘기는 것과 명시적으로 `False`로
    주는 것은 byte-identical해야 한다 -- 새 필드 3개도 항상 `None`이어야
    한다."""

    def _run_benign(self, use_clarification: bool | None):
        principal_llm = _FakeLLMClient([
            LLMResponse(text="Please read last month's report."),
            _structured("read"),
        ])
        delegate_llm = _FakeLLMClient([_structured("read")])
        semantic_llm = _FakeLLMClient([_structured("read")])
        authority_llm = _FakeLLMClient([])

        kwargs = {} if use_clarification is None else {"use_clarification": use_clarification}
        runtime = AgentDelegationRuntime(
            principal=PrincipalAgent(llm=principal_llm),
            delegate=DelegateAgent(llm=delegate_llm),
            semantic_verifier=SemanticVerifierAgent(llm_client=semantic_llm),
            authority_verifier=AuthorityVerifierAgent(llm_client=authority_llm),
            **kwargs)
        budget = Budget.of(Privilege("read", "file", "/reports/2026-08/"))
        result = runtime.run(goal="Read last month's report.", context="finance team",
                             budget=budget)
        return result, delegate_llm

    def test_omitted_and_explicit_false_are_identical(self):
        result_omitted, delegate_llm_omitted = self._run_benign(None)
        result_explicit, delegate_llm_explicit = self._run_benign(False)

        for result in (result_omitted, result_explicit):
            assert result.decision == EXECUTE
            assert result.principal_match is True
            assert result.clarification_result is None
            assert result.experience_decision is None
            assert result.candidate_experience is None

        # propose() 단일 호출만 있었다 -- sample_candidates()는 전혀 안 쓰였다.
        assert len(delegate_llm_omitted.calls) == 1
        assert len(delegate_llm_explicit.calls) == 1


class TestClarificationStableDistribution:
    def test_stable_distribution_skips_clarification_and_flows_through(self):
        principal_llm = _FakeLLMClient([
            LLMResponse(text="Please read last month's report."),
            _structured("read"),
        ])
        # n=3, 전부 같은 action -> H=0.0 <= threshold -> clarification 없음.
        delegate_llm = _FakeLLMClient([_structured("read") for _ in range(3)])
        semantic_llm = _FakeLLMClient([_structured("read")])
        authority_llm = _FakeLLMClient([])

        runtime = AgentDelegationRuntime(
            principal=PrincipalAgent(llm=principal_llm),
            delegate=DelegateAgent(llm=delegate_llm),
            semantic_verifier=SemanticVerifierAgent(llm_client=semantic_llm),
            authority_verifier=AuthorityVerifierAgent(llm_client=authority_llm),
            use_clarification=True, n=3, entropy_threshold=0.8)
        budget = Budget.of(Privilege("read", "file", "/reports/2026-08/"))

        result = runtime.run(goal="Read last month's report.", context="finance team",
                             budget=budget)

        assert result.clarification_result is not None
        assert result.clarification_result.clarified is False
        assert result.proposal.interpretation == Interpretation(
            "read", "file", "/reports/2026-08/", frozenset())
        assert result.final_interpretation == Interpretation(
            "read", "file", "/reports/2026-08/", frozenset())
        assert result.decision == EXECUTE
        # pre-sampling 3회뿐 -- ask_clarification()도 post-sampling도 없었다.
        assert len(delegate_llm.calls) == 3
        assert result.experience_decision is None    # experience_store를 안 줬다
        assert result.candidate_experience is None   # clarified=False -> 저장 후보 없음


class TestClarificationNoExperience:
    def test_ambiguous_distribution_clarifies_without_experience_fields(self):
        principal_llm = _FakeLLMClient([
            LLMResponse(text="Please handle last month's report."),      # delegate()
            _structured("read"),                                         # restate_intent()
            LLMResponse(text="Just read it, no export needed."),         # answer_clarification()
        ])
        delegate_llm = _FakeLLMClient(
            [_structured("read"), _structured("read"), _structured("export")]   # pre: H≈0.918>0.8
            + [LLMResponse(text="Read or export?")]                              # ask_clarification()
            + [_structured("read"), _structured("read"), _structured("read")])   # post: H=0.0
        semantic_llm = _FakeLLMClient([_structured("read")])
        authority_llm = _FakeLLMClient([])

        runtime = AgentDelegationRuntime(
            principal=PrincipalAgent(llm=principal_llm),
            delegate=DelegateAgent(llm=delegate_llm),
            semantic_verifier=SemanticVerifierAgent(llm_client=semantic_llm),
            authority_verifier=AuthorityVerifierAgent(llm_client=authority_llm),
            use_clarification=True, n=3, entropy_threshold=0.8)
        budget = Budget.of(Privilege("read", "file", "/reports/2026-08/"))

        result = runtime.run(goal="Handle last month's report, read only.",
                             context="finance team", budget=budget)

        assert result.clarification_result is not None
        assert result.clarification_result.clarified is True
        assert result.final_interpretation.action == "read"
        assert result.decision == EXECUTE
        # experience_store/principal_id/task_category를 전혀 안 줬다.
        assert result.experience_decision is None
        assert result.candidate_experience is None


class TestClarificationAdvisoryHistoryNeverOverrides:
    """Runtime Integration Phase 1의 핵심 회귀 -- Option B(v3, commit
    0f1cf7c, B7e Phase 1 real API로 검증 완료, commit 8b701d4)가 runtime
    레벨에서도 그대로 지켜지는지 직접 확인한다. History는 "summarize"를
    SUPPORT하지만, 현재 baseline도 실제 Principal의 clarification 답변도
    "read"다 -- 그러므로 최종 결정은 반드시 "read"여야 하고, advisory
    relation은 감사용으로만 기록돼야 한다."""

    def test_conflicting_history_is_recorded_but_never_changes_the_decision(self):
        store = AgentExperienceStore()
        store.add(_make_experience("summarize"))  # 과거엔 summarize가 confirmed였다

        principal_llm = _FakeLLMClient([
            LLMResponse(text="Please handle last month's report."),        # delegate()
            _structured("read"),                                          # restate_intent() -- 진짜 의도
            LLMResponse(text="Read it, no summary needed this time."),    # answer_clarification()
        ])
        delegate_llm = _FakeLLMClient(
            [_structured("read"), _structured("read"), _structured("summarize")]  # pre: H≈0.918>0.8
            + [LLMResponse(text="Read or summarize?")]                             # ask_clarification()
            + [_structured("read"), _structured("read"), _structured("read")])     # post: H=0.0, "read" 확인
        semantic_llm = _FakeLLMClient([_structured("read")])
        authority_llm = _FakeLLMClient([])

        runtime = AgentDelegationRuntime(
            principal=PrincipalAgent(llm=principal_llm),
            delegate=DelegateAgent(llm=delegate_llm),
            semantic_verifier=SemanticVerifierAgent(llm_client=semantic_llm),
            authority_verifier=AuthorityVerifierAgent(llm_client=authority_llm),
            use_clarification=True, n=3, entropy_threshold=0.8,
            experience_store=store)
        budget = Budget.of(Privilege("read", "file", "/reports/2026-08/"))

        result = runtime.run(goal="Handle last month's report, read only.",
                             context="finance team", budget=budget,
                             principal_id="p1", task_category="cat1")

        # relation은 계산·기록됐다 -- summarize가 SUPPORT다.
        assert result.experience_decision is not None
        assert result.experience_decision.stage == AMBIGUOUS_REQUIRES_CLARIFICATION
        assert result.experience_decision.resolved_value == "summarize"
        # 하지만 그 어떤 값도 실제 baseline/final을 바꾸지 않았다.
        assert result.experience_decision.baseline_value == "read"
        assert result.experience_decision.final_value == "read"
        # 그리고 실제 runtime 최종 결정도 "read"다 -- history가 이긴 게 아니라
        # 실제 clarification에서 Principal이 "read"를 확인해준 결과다.
        assert result.final_interpretation.action == "read"
        assert result.decision == EXECUTE
        assert result.principal_match is True


class TestClarificationExperienceNotAutoStored:
    def test_run_never_calls_store_add_itself(self):
        principal_llm = _FakeLLMClient([
            LLMResponse(text="Please handle last month's report."),
            _structured("read"),
            LLMResponse(text="Just read it."),
        ])
        delegate_llm = _FakeLLMClient(
            [_structured("read"), _structured("read"), _structured("summarize")]
            + [LLMResponse(text="Read or summarize?")]
            + [_structured("read"), _structured("read"), _structured("read")])
        semantic_llm = _FakeLLMClient([_structured("read")])
        authority_llm = _FakeLLMClient([])
        store = AgentExperienceStore()  # 처음엔 비어 있다 -- 이번 episode 이전 history 없음

        runtime = AgentDelegationRuntime(
            principal=PrincipalAgent(llm=principal_llm),
            delegate=DelegateAgent(llm=delegate_llm),
            semantic_verifier=SemanticVerifierAgent(llm_client=semantic_llm),
            authority_verifier=AuthorityVerifierAgent(llm_client=authority_llm),
            use_clarification=True, n=3, entropy_threshold=0.8,
            experience_store=store)
        budget = Budget.of(Privilege("read", "file", "/reports/2026-08/"))

        result = runtime.run(goal="Handle last month's report, read only.",
                             context="finance team", budget=budget,
                             principal_id="p1", task_category="cat1")

        assert result.candidate_experience is not None
        assert result.candidate_experience.confirmed_facets == frozenset({"action"})
        assert store.count("p1", "cat1") == 0   # run() 자신은 절대 add()하지 않는다

        store.add(result.candidate_experience)  # 호출자가 명시적으로 저장해야 한다
        assert store.count("p1", "cat1") == 1


# --------------------------------------------------------------------------
# Runtime integration -- source-side Semantic Flow gate (docs/experiments/
# agent_connected_eval.md §29, Phase 2C-P5, Finding P2C-F2). 6 integration-
# correctness tests, not new research experiments -- P5 already validated
# the gate's real-API behavior standalone (commit b0f8e36); this section
# only confirms the wiring into AgentDelegationRuntime.run() is correct.
# Everything above this line continues to exercise use_source_verification=
# False (the default) exclusively.
# --------------------------------------------------------------------------

class TestSourceVerificationOptOut:
    """(1) source verification OFF -> 지금까지의(pre-integration) runtime
    동작과 완전히 동일해야 한다. 안 넘기는 것과 명시적 False도
    byte-identical해야 한다."""

    def _run_benign(self, use_source_verification: bool | None):
        principal_llm = _FakeLLMClient([
            LLMResponse(text="Please read last month's report."),
            _structured("read"),
        ])
        delegate_llm = _FakeLLMClient([_structured("read")])
        semantic_llm = _FakeLLMClient([_structured("read")])
        authority_llm = _FakeLLMClient([])

        kwargs = ({} if use_source_verification is None
                 else {"use_source_verification": use_source_verification})
        runtime = AgentDelegationRuntime(
            principal=PrincipalAgent(llm=principal_llm),
            delegate=DelegateAgent(llm=delegate_llm),
            semantic_verifier=SemanticVerifierAgent(llm_client=semantic_llm),
            authority_verifier=AuthorityVerifierAgent(llm_client=authority_llm),
            **kwargs)
        budget = Budget.of(Privilege("read", "file", "/reports/2026-08/"))
        result = runtime.run(goal="Read last month's report.", context="finance team",
                             budget=budget)
        return result, principal_llm, delegate_llm

    def test_omitted_and_explicit_false_are_identical(self):
        result_omitted, principal_omitted, delegate_omitted = self._run_benign(None)
        result_explicit, principal_explicit, delegate_explicit = self._run_benign(False)

        for result in (result_omitted, result_explicit):
            assert result.decision == EXECUTE
            assert result.principal_match is True
            assert result.source_clarification_result is None
            assert result.final_interpretation == Interpretation(
                "read", "file", "/reports/2026-08/", frozenset())

        # delegate() 1회 + restate_intent() 1회뿐 -- source-side sampling이
        # 전혀 끼어들지 않는다(기존 Runtime Integration Phase 1의
        # TestClarificationOptOut과 정확히 같은 패턴).
        assert len(principal_omitted.calls) == 2
        assert len(principal_explicit.calls) == 2
        assert len(delegate_omitted.calls) == 1
        assert len(delegate_explicit.calls) == 1


class TestSourceVerificationStable:
    """(2) ON + clear input -> 여전히 같은 실행 경로, clarification 없음."""

    def test_stable_source_flows_through_without_clarification(self):
        source_n = 3
        principal_llm = _FakeLLMClient(
            [LLMResponse(text="Please read last month's report.") for _ in range(source_n)]
            + [_structured("read")])                                       # restate_intent()
        delegate_llm = _FakeLLMClient(
            [_structured("read") for _ in range(source_n)]                 # canonicalize x3(수렴)
            + [_structured("read")])                                       # step-3 propose()
        semantic_llm = _FakeLLMClient([_structured("read")])
        authority_llm = _FakeLLMClient([])

        runtime = AgentDelegationRuntime(
            principal=PrincipalAgent(llm=principal_llm),
            delegate=DelegateAgent(llm=delegate_llm),
            semantic_verifier=SemanticVerifierAgent(llm_client=semantic_llm),
            authority_verifier=AuthorityVerifierAgent(llm_client=authority_llm),
            use_source_verification=True, source_n=source_n, entropy_threshold=0.8)
        budget = Budget.of(Privilege("read", "file", "/reports/2026-08/"))

        result = runtime.run(goal="Read last month's report.", context="finance team",
                             budget=budget)

        assert result.source_clarification_result is not None
        assert result.source_clarification_result.clarified is False
        assert result.source_clarification_result.pre_distribution.entropy == 0.0
        assert result.final_interpretation == Interpretation(
            "read", "file", "/reports/2026-08/", frozenset())
        assert result.decision == EXECUTE
        assert result.principal_match is True


class TestSourceVerificationUnstableTriggersClarification:
    """(3) ON + unstable input -> source clarification이 실제로 발동한다."""

    def test_unstable_source_actually_triggers_clarification(self):
        source_n = 4
        principal_llm = _FakeLLMClient(
            [LLMResponse(text="Please prepare the report.") for _ in range(source_n)]
            + [LLMResponse(text="summarize")]                              # answer_clarification()
            + [LLMResponse(text="Please summarize the report.") for _ in range(source_n)]
            + [_structured("summarize")])                                  # restate_intent()
        delegate_llm = _FakeLLMClient(
            [_structured("read"), _structured("read"),
            _structured("summarize"), _structured("summarize")]           # pre: action H=1.0>0.8
            + [_structured("summarize") for _ in range(source_n)]          # post: 수렴, H=0.0
            + [_structured("summarize")])                                  # step-3 propose()
        semantic_llm = _FakeLLMClient([_structured("summarize")])
        authority_llm = _FakeLLMClient([])

        runtime = AgentDelegationRuntime(
            principal=PrincipalAgent(llm=principal_llm),
            delegate=DelegateAgent(llm=delegate_llm),
            semantic_verifier=SemanticVerifierAgent(llm_client=semantic_llm),
            authority_verifier=AuthorityVerifierAgent(llm_client=authority_llm),
            use_source_verification=True, source_n=source_n, entropy_threshold=0.8)
        budget = Budget.of(Privilege("summarize", "file", "/reports/2026-08/"))

        result = runtime.run(goal="Prepare the report.", context="finance team", budget=budget)

        assert result.source_clarification_result is not None
        assert result.source_clarification_result.clarified is True
        assert result.source_clarification_result.target_facet == "action"
        assert result.source_clarification_result.pre_distribution.entropy > 0.8
        assert result.source_clarification_result.post_distribution is not None
        assert result.source_clarification_result.post_distribution.entropy == 0.0
        assert result.final_interpretation.action == "summarize"
        assert result.decision == EXECUTE


class TestSourceVerificationConfirmedFacetIsStable:
    """(4) 확정된 facet이 이후 delegation 생성에서 다시 바뀌지 않는다 --
    run()의 2번(restate_intent)/3번(propose) 단계가 실제로 그 confirmed
    delegation 텍스트를 받았는지 직접 확인한다(합성/역행 없음)."""

    def test_confirmed_facet_flows_unchanged_into_later_delegation(self):
        source_n = 4
        principal_llm = _FakeLLMClient(
            [LLMResponse(text="Please prepare the report.") for _ in range(source_n)]
            + [LLMResponse(text="summarize")]
            + [LLMResponse(text="Please summarize the report.") for _ in range(source_n)]
            + [_structured("summarize")])
        delegate_llm = _FakeLLMClient(
            [_structured("read"), _structured("read"),
            _structured("summarize"), _structured("summarize")]
            + [_structured("summarize") for _ in range(source_n)]
            + [_structured("summarize")])
        semantic_llm = _FakeLLMClient([_structured("summarize")])
        authority_llm = _FakeLLMClient([])

        runtime = AgentDelegationRuntime(
            principal=PrincipalAgent(llm=principal_llm),
            delegate=DelegateAgent(llm=delegate_llm),
            semantic_verifier=SemanticVerifierAgent(llm_client=semantic_llm),
            authority_verifier=AuthorityVerifierAgent(llm_client=authority_llm),
            use_source_verification=True, source_n=source_n, entropy_threshold=0.8)
        budget = Budget.of(Privilege("summarize", "file", "/reports/2026-08/"))

        result = runtime.run(goal="Prepare the report.", context="finance team", budget=budget)

        confirmed_text = result.source_clarification_result.final_delegation
        assert confirmed_text == "Please summarize the report."
        # run() 자신이 담은 delegation은 confirmed된 것과 완전히 같다 --
        # 별도로 재생성되거나 pre-round 후보로 되돌아가지 않았다.
        assert result.delegation.delegation == confirmed_text
        assert result.delegation is result.source_clarification_result.final_principal_delegation
        # step-3 propose() 호출(2번째 delegate.propose(), 즉 실제 downstream
        # 호출)의 input_text에 confirmed delegation 텍스트가 그대로
        # 들어갔는지 직접 확인한다.
        step3_call = delegate_llm.calls[-1]
        assert confirmed_text in step3_call["input_text"]
        assert result.decision == EXECUTE


class TestSourceVerificationAuthorityInvariantUnaffected:
    """(5) Authority Flow / non-amplification invariant는 source-side
    verification이 켜져도 그대로 유지된다 -- B가 더 넓게 제안하면 여전히
    budget ceiling까지만 좁혀진다."""

    def test_authority_negotiation_still_narrows_with_source_verification_on(self):
        source_n = 2
        principal_llm = _FakeLLMClient(
            [LLMResponse(text="Please read last month's report.") for _ in range(source_n)]
            + [_structured("read", scope="/reports/2026-08/")])            # restate_intent() -- 진짜 의도는 좁은 scope
        delegate_llm = _FakeLLMClient(
            [_structured("read", scope="/reports/2026-08/") for _ in range(source_n)]  # canonicalize(안정)
            + [_structured("read", scope="/reports/")])                     # step-3 propose() -- B는 더 넓게 제안
        semantic_llm = _FakeLLMClient([_structured("read", scope="/reports/")])
        # Authority가 협상으로 ceiling까지 좁혀 제안 -> 재검증 통과.
        authority_llm = _FakeLLMClient([_structured("read", scope="/reports/2026-08/")])

        runtime = AgentDelegationRuntime(
            principal=PrincipalAgent(llm=principal_llm),
            delegate=DelegateAgent(llm=delegate_llm),
            semantic_verifier=SemanticVerifierAgent(llm_client=semantic_llm),
            authority_verifier=AuthorityVerifierAgent(llm_client=authority_llm),
            use_source_verification=True, source_n=source_n, entropy_threshold=0.8)
        budget = Budget.of(Privilege("read", "file", "/reports/2026-08/"))

        result = runtime.run(goal="Read last month's report.", context="", budget=budget)

        assert len(authority_llm.calls) == 1  # 협상이 여전히 1회 일어난다
        assert result.authority_verdict.negotiated is True
        assert result.proposal.interpretation.scope == "/reports/"         # B의 원래(넓은) 제안
        assert result.final_interpretation.scope == "/reports/2026-08/"    # 협상 후 좁혀진 값 --
                                                                             # source verification이 non-amplification을
                                                                             # 우회하게 하지 않는다
        assert result.principal_match is True
        assert result.decision == EXECUTE
