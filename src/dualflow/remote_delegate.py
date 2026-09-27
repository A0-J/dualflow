"""
REMOTE DELEGATE — Runtime Integration Phase 2A (docs/experiments/
agent_connected_eval.md §27). HTTP transport for Agent B (the Delegate),
so `AgentDelegationRuntime`/`clarification.ClarifyingDelegate` (Process A)
can drive a Delegate running in a separate process — same machine first,
then a second physical machine — without Agent B ever verifying its own
delegation (that would collapse the independent-verification structure
this whole project has been built around).

`RemoteDelegateAgent` implements exactly the same three public methods as
`delegate_agent.DelegateAgent` (`propose`/`sample_candidates`/
`ask_clarification`), with the same keyword-only signatures. Neither
`AgentDelegationRuntime` nor `ClarifyingDelegate` type-checks its
`delegate` argument — both call it via plain duck typing — so this class
is a drop-in replacement wherever a `DelegateAgent` is expected, with zero
changes to either of those (already-frozen/validated) classes.

Standard-library only (`urllib.request`/`json`) — no new dependency, per
this repo's existing zero-core-dependency convention (`pyproject.toml`).

이 모듈이 하지 않는 것:
  - `DelegateAgent`/`ClarifyingDelegate`/`AgentDelegationRuntime`를
    수정하는 것 — 전부 그대로, import만 한다.
  - 실제 서버(Process B) — 그건 `experiments/delegate_server.py`가
    한다(이 모듈의 직렬화 helper를 그대로 재사용한다, 중복 정의 없음).
  - 인증/재시도/커넥션 풀링 — Phase 2A는 로컬 프로세스 간 smoke test,
    Phase 2B는 두 프로세스 간 real-API 단발 호출이 목표이므로, 이런
    production-hardening은 이번 phase의 범위 밖이다.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .delegate_agent import CandidateDistribution, ClarificationQuestion, DelegateProposal
from .llm import LLMResponse
from .semantic import Belief, Interpretation


class RemoteDelegateError(RuntimeError):
    """Remote Delegate 서버 호출이 실패했을 때(non-2xx 응답, 연결 실패,
    잘못된 JSON 응답 등) — 원인을 그대로 메시지에 담아 던진다."""


# ----------------------------------------------------------------------------
# Wire-format (de)serialization — 서버(experiments/delegate_server.py)와
# 클라이언트(RemoteDelegateAgent) 양쪽이 이 함수들을 그대로 재사용한다.
# 스키마를 두 곳에 따로 적지 않기 위해서다.
# ----------------------------------------------------------------------------

def interpretation_to_dict(interp: Interpretation) -> dict:
    return {"action": interp.action, "resource": interp.resource, "scope": interp.scope,
            "condition": sorted(interp.condition), "label": interp.label}


def interpretation_from_dict(d: dict) -> Interpretation:
    return Interpretation(action=d["action"], resource=d["resource"],
                          scope=d.get("scope", "*"),
                          condition=frozenset(d.get("condition", ())),
                          label=d.get("label", ""))


def llm_response_to_dict(r: LLMResponse) -> dict:
    return {"text": r.text, "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens, "latency_ms": r.latency_ms}


def llm_response_from_dict(d: dict) -> LLMResponse:
    return LLMResponse(text=d["text"], input_tokens=d.get("input_tokens"),
                       output_tokens=d.get("output_tokens"), latency_ms=d.get("latency_ms"))


def proposal_to_dict(p: DelegateProposal) -> dict:
    return {"interpretation": interpretation_to_dict(p.interpretation),
            "raw_text": p.raw_text, "response": llm_response_to_dict(p.response)}


def proposal_from_dict(d: dict) -> DelegateProposal:
    return DelegateProposal(interpretation=interpretation_from_dict(d["interpretation"]),
                            raw_text=d["raw_text"],
                            response=llm_response_from_dict(d["response"]))


def distribution_to_dict(dist: CandidateDistribution) -> dict:
    return {
        "belief": belief_to_list(dist.belief),
        "entropy": dist.entropy,
        "top": interpretation_to_dict(dist.top),
        "top_probability": dist.top_probability,
        "n_unique": dist.n_unique,
        "n_samples": dist.n_samples,
        "responses": [llm_response_to_dict(r) for r in dist.responses],
    }


def distribution_from_dict(d: dict) -> CandidateDistribution:
    belief = belief_from_list(d["belief"])
    return CandidateDistribution(
        belief=belief, entropy=d["entropy"], top=interpretation_from_dict(d["top"]),
        top_probability=d["top_probability"], n_unique=d["n_unique"], n_samples=d["n_samples"],
        responses=[llm_response_from_dict(r) for r in d.get("responses", [])])


def belief_to_list(belief: Belief) -> list[dict]:
    return [{"interpretation": interpretation_to_dict(i), "probability": p}
           for i, p in belief.items()]


def belief_from_list(items: list[dict]) -> Belief:
    return {interpretation_from_dict(item["interpretation"]): item["probability"]
           for item in items}


class RemoteDelegateAgent:
    """`DelegateAgent`와 정확히 같은 인터페이스를 HTTP로 구현한다.
    `base_url`은 `http://host:port`형태(트레일링 슬래시는 무시된다)."""

    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url, data=data, method="POST", headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RemoteDelegateError(f"POST {path} -> HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RemoteDelegateError(f"POST {path} -> connection failed: {e.reason}") from e
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise RemoteDelegateError(f"POST {path} -> invalid JSON response: {e}") from e

    def propose(self, *, delegation: str, context: str = "") -> DelegateProposal:
        result = self._post("/propose", {"delegation": delegation, "context": context})
        return proposal_from_dict(result)

    def sample_candidates(self, *, delegation: str, context: str = "",
                          n: int = 10) -> CandidateDistribution:
        result = self._post(
            "/sample-candidates", {"delegation": delegation, "context": context, "n": n})
        return distribution_from_dict(result)

    def ask_clarification(self, *, delegation: str, distribution: CandidateDistribution,
                          context: str = "",
                          target_facets: frozenset[str] = frozenset()) -> ClarificationQuestion:
        # `distribution.belief`만 실제로 서버가 필요로 한다(delegate_agent.
        # ask_clarification()의 prompt는 belief만 읽는다) -- entropy/top 등은
        # 이 호출의 반환값에서도 "이 질문의 근거가 된 분포를 그대로 옮겨온
        # 것"일 뿐(delegate_agent.py의 ClarificationQuestion docstring 참고)
        # 이미 호출자가 갖고 있는 `distribution`에서 그대로 채운다 -- 서버가
        # 새로 계산해서 돌려줄 필요가 없다.
        result = self._post("/ask-clarification", {
            "delegation": delegation, "context": context,
            "belief": belief_to_list(distribution.belief),
            "target_facets": sorted(target_facets),
        })
        return ClarificationQuestion(
            question=result["question"], raw_text=result["raw_text"],
            response=llm_response_from_dict(result["response"]),
            belief=distribution.belief, entropy=distribution.entropy,
            target_facets=target_facets)
