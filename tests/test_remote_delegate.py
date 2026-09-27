"""Runtime Integration Phase 2A -- deterministic (0 API calls) validation
of the HTTP transport between Process A (`AgentDelegationRuntime`/
`ClarifyingDelegate`) and Process B (`DelegateAgent`, wrapped by
`experiments/delegate_server.py`'s handler, reached via `dualflow.
remote_delegate.RemoteDelegateAgent`).

Every test here starts a real `http.server.ThreadingHTTPServer` on
127.0.0.1 (a real OS socket, real HTTP round trip over loopback -- not
mocked), backed by a `DelegateAgent` wired to a fake, in-process
`LLMClient` (so no network call ever reaches a real model). This proves
the serialize -> HTTP -> deserialize path is lossless, and that
`AgentDelegationRuntime` needs zero code changes to accept a
`RemoteDelegateAgent` in place of a local `DelegateAgent` -- confirmed by
running the SAME scenario through both and asserting identical results.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer

import pytest

from dualflow.agent_runtime import AgentDelegationRuntime
from dualflow.authority_feedback import AuthorityVerifierAgent
from dualflow.capability import Budget, Privilege
from dualflow.delegate_agent import CandidateDistribution, DelegateAgent
from dualflow.framework import EXECUTE
from dualflow.llm import LLMResponse
from dualflow.principal_agent import PrincipalAgent
from dualflow.remote_delegate import RemoteDelegateAgent, RemoteDelegateError
from dualflow.semantic import Interpretation, SemanticVerifierAgent

from experiments.delegate_server import make_handler


class _FakeLLMClient:
    """`tests/test_agent_runtime.py`의 것과 동일한 패턴."""

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


@contextmanager
def _running_server(delegate: DelegateAgent):
    """127.0.0.1의 임의의 빈 포트(port=0 -> OS가 골라줌)에서 서버를 백그라운드
    스레드로 띄운다 -- 테스트가 끝나면 항상 정리한다."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(delegate))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class TestProposeRoundTrip:
    def test_remote_propose_matches_local_propose(self):
        local_delegate = DelegateAgent(llm=_FakeLLMClient([_structured("read")]))
        local_result = local_delegate.propose(delegation="Please read the report.", context="")

        remote_delegate_llm = _FakeLLMClient([_structured("read")])
        with _running_server(DelegateAgent(llm=remote_delegate_llm)) as base_url:
            remote = RemoteDelegateAgent(base_url=base_url)
            remote_result = remote.propose(delegation="Please read the report.", context="")

        assert remote_result.interpretation == local_result.interpretation
        assert remote_result.raw_text == local_result.raw_text
        assert remote_result.response.text == local_result.response.text


class TestSampleCandidatesRoundTrip:
    def test_remote_sample_candidates_matches_local(self):
        responses = [_structured("read"), _structured("read"), _structured("export")]
        local_delegate = DelegateAgent(llm=_FakeLLMClient(list(responses)))
        local_dist = local_delegate.sample_candidates(delegation="Please handle the report.",
                                                       context="", n=3)

        with _running_server(DelegateAgent(llm=_FakeLLMClient(list(responses)))) as base_url:
            remote = RemoteDelegateAgent(base_url=base_url)
            remote_dist = remote.sample_candidates(delegation="Please handle the report.",
                                                    context="", n=3)

        assert remote_dist.entropy == pytest.approx(local_dist.entropy)
        assert remote_dist.top == local_dist.top
        assert remote_dist.n_samples == local_dist.n_samples
        assert remote_dist.n_unique == local_dist.n_unique
        # belief dict가 (Interpretation -> probability) 그대로 round-trip됐다.
        assert dict(remote_dist.belief) == pytest.approx(dict(local_dist.belief))


class TestAskClarificationRoundTrip:
    def test_remote_ask_clarification_matches_local(self):
        belief = {Interpretation("read", "file", "/reports/2026-08/", frozenset()): 2 / 3,
                 Interpretation("export", "file", "/reports/2026-08/", frozenset()): 1 / 3}
        distribution = CandidateDistribution(
            belief=belief, entropy=0.918, top=max(belief, key=lambda i: belief[i]),
            top_probability=2 / 3, n_unique=2, n_samples=3, responses=[])

        local_delegate = DelegateAgent(llm=_FakeLLMClient([LLMResponse(text="Read or export?")]))
        local_question = local_delegate.ask_clarification(
            delegation="Please handle the report.", distribution=distribution, context="")

        with _running_server(
                DelegateAgent(llm=_FakeLLMClient([LLMResponse(text="Read or export?")]))) as base_url:
            remote = RemoteDelegateAgent(base_url=base_url)
            remote_question = remote.ask_clarification(
                delegation="Please handle the report.", distribution=distribution, context="")

        assert remote_question.question == local_question.question == "Read or export?"
        # belief/entropy는 서버가 다시 계산하지 않는다 -- 호출자가 이미 가진
        # distribution에서 그대로 옮겨온 것이어야 한다(remote_delegate.py 참고).
        assert remote_question.belief == distribution.belief
        assert remote_question.entropy == distribution.entropy


class TestErrorHandling:
    def test_connection_failure_raises_remote_delegate_error(self):
        # 아무 서버도 안 띄운 포트 -- connection refused가 나야 한다.
        remote = RemoteDelegateAgent(base_url="http://127.0.0.1:1", timeout=2.0)
        with pytest.raises(RemoteDelegateError):
            remote.propose(delegation="x", context="")

    def test_unknown_path_returns_404_and_raises(self):
        delegate = DelegateAgent(llm=_FakeLLMClient([]))
        with _running_server(delegate) as base_url:
            remote = RemoteDelegateAgent(base_url=base_url)
            with pytest.raises(RemoteDelegateError):
                remote._post("/does-not-exist", {})


class TestFullRuntimeWithRemoteDelegate:
    """가장 강한 증명 -- AgentDelegationRuntime을 전혀 수정하지 않고,
    delegate 자리에 RemoteDelegateAgent를 넣기만 해서 로컬 DelegateAgent와
    똑같은 결과가 나오는지 확인한다."""

    def _make_runtime(self, *, delegate, use_clarification: bool = False, n: int = 3):
        principal_llm = _FakeLLMClient([
            LLMResponse(text="Please read last month's report."),
            _structured("read"),
        ])
        semantic_llm = _FakeLLMClient([_structured("read")])
        authority_llm = _FakeLLMClient([])
        return AgentDelegationRuntime(
            principal=PrincipalAgent(llm=principal_llm),
            delegate=delegate,
            semantic_verifier=SemanticVerifierAgent(llm_client=semantic_llm),
            authority_verifier=AuthorityVerifierAgent(llm_client=authority_llm),
            use_clarification=use_clarification, n=n)

    def test_stable_run_over_remote_delegate_matches_local(self):
        budget = Budget.of(Privilege("read", "file", "/reports/2026-08/"))

        local_runtime = self._make_runtime(delegate=DelegateAgent(llm=_FakeLLMClient([_structured("read")])))
        local_result = local_runtime.run(goal="Read last month's report.", context="finance team",
                                         budget=budget)

        with _running_server(DelegateAgent(llm=_FakeLLMClient([_structured("read")]))) as base_url:
            remote_runtime = self._make_runtime(delegate=RemoteDelegateAgent(base_url=base_url))
            remote_result = remote_runtime.run(goal="Read last month's report.",
                                               context="finance team", budget=budget)

        assert remote_result.decision == local_result.decision == EXECUTE
        assert remote_result.final_interpretation == local_result.final_interpretation
        assert remote_result.principal_match == local_result.principal_match

    def test_ambiguous_clarification_run_over_remote_delegate(self):
        """use_clarification=True 경로 -- /sample-candidates와 /ask-clarification
        둘 다 실제로 HTTP를 거쳐야 동작한다."""
        budget = Budget.of(Privilege("read", "file", "/reports/2026-08/"))

        def _script():
            return [_structured("read"), _structured("read"), _structured("export")] \
                + [LLMResponse(text="Read or export?")] \
                + [_structured("read"), _structured("read"), _structured("read")]

        principal_llm = _FakeLLMClient([
            LLMResponse(text="Please handle last month's report."),
            _structured("read"),
            LLMResponse(text="Just read it."),
        ])
        semantic_llm = _FakeLLMClient([_structured("read")])
        authority_llm = _FakeLLMClient([])

        with _running_server(DelegateAgent(llm=_FakeLLMClient(_script()))) as base_url:
            runtime = AgentDelegationRuntime(
                principal=PrincipalAgent(llm=principal_llm),
                delegate=RemoteDelegateAgent(base_url=base_url),
                semantic_verifier=SemanticVerifierAgent(llm_client=semantic_llm),
                authority_verifier=AuthorityVerifierAgent(llm_client=authority_llm),
                use_clarification=True, n=3)
            result = runtime.run(goal="Handle last month's report, read only.",
                                 context="finance team", budget=budget)

        assert result.clarification_result is not None
        assert result.clarification_result.clarified is True
        assert result.final_interpretation.action == "read"
        assert result.decision == EXECUTE
