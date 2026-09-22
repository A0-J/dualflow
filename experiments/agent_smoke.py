"""Standalone smoke-test runner for each Phase B agent, hitting the REAL
OpenAI API — for debugging and eyeballing prompts/outputs by hand, not for
repeated/statistical experiments.

    python experiments/agent_smoke.py --role principal
    python experiments/agent_smoke.py --role delegate
    python experiments/agent_smoke.py --role semantic
    python experiments/agent_smoke.py --role authority
    python experiments/agent_smoke.py --role all
    python experiments/agent_smoke.py --role runtime

Role separation (do not blur these):
    experiments/agent_smoke.py     <- this file. One canned scenario,
                                       human-readable output, for checking
                                       each agent's real prompt/response by
                                       hand before running anything at scale.
    experiments/agent_benchmark.py <- (not built yet) repeated runs across
                                       many tasks and methods, saved as
                                       results/agent/runs.jsonl + summary.csv.
    experiments/plots.py           <- reads agent_benchmark.py's results and
                                       draws the paper-facing figures.

This script makes REAL, BILLED API calls. Requires:
    - `pip install -e ".[agent]"` (installs the `openai` package)
    - `OPENAI_API_KEY` set as an environment variable — never as a CLI
      argument or in code:
          bash:       export OPENAI_API_KEY=...
          PowerShell: $env:OPENAI_API_KEY="..."

--role all runs PrincipalAgent -> DelegateAgent -> (SemanticVerifierAgent +
AuthorityVerifierAgent) in sequence and prints each stage's raw output, but
deliberately does NOT compute a final EXECUTE/REJECT decision — it exists
to look at each stage's real output in isolation, independent of any fusion
rule.

--role runtime instead calls the real B6 oracle-free orchestrator,
dualflow.agent_runtime.AgentDelegationRuntime.run(), and prints its full
AgentRuntimeResult including the final EXECUTE/REJECT decision. This
script never reimplements fusion logic itself — --role runtime is a thin
wrapper around the actual runtime class, nothing more.

This script is not part of the DualFlow core package — it uses the
installed package like any other caller would (`pip install -e ".[agent]"`
from the repository root, then run as above).
"""

from __future__ import annotations

import os
import sys

from dualflow.agent_runtime import AgentDelegationRuntime, AgentRuntimeResult
from dualflow.authority_feedback import AuthorityVerdict, AuthorityVerifierAgent
from dualflow.capability import Budget, Privilege
from dualflow.delegate_agent import DelegateAgent, DelegateProposal
from dualflow.llm import LLMResponse, OpenAILLMClient
from dualflow.principal_agent import PrincipalAgent, PrincipalDelegation, PrincipalIntent
from dualflow.semantic import Interpretation, SemanticVerdict, SemanticVerifierAgent

# ----------------------------------------------------------------------------
# One canned scenario. Not configurable via CLI on purpose — this script is
# for eyeballing real output on a fixed case, not for running many scenarios
# (that's agent_benchmark.py's job).
# ----------------------------------------------------------------------------
EXAMPLE_GOAL = "Read last month's financial report."
EXAMPLE_CONTEXT = "You are Agent A, a finance team lead."
EXAMPLE_DELEGATION = "Please inspect last month's financial report and report its contents."
EXAMPLE_BUDGET = Budget.of(Privilege("read", "file", "/reports/2026-08/"))
# Deliberately wrong relative to EXAMPLE_DELEGATION (export != read) — this
# is what lets --role semantic (run standalone, without an actual
# DelegateAgent call) demonstrate a real action mismatch instead of a
# trivial match. --role all uses DelegateAgent's *actual* proposal instead.
EXAMPLE_MISREAD_PROPOSAL = Interpretation("export", "file", "/reports/2026-08/", frozenset())

# --role authority's standalone example needs a DIFFERENT proposal than the
# one above. AuthorityVerifierAgent.verify_agent_proposal() only calls the
# LLM when check_authority() returns scope_exceeded — an action/resource
# that was never delegated at all (like EXAMPLE_MISREAD_PROPOSAL's "export"
# against a read-only budget) is a no_grant hard reject, which returns
# deterministically *without* ever calling the LLM. That would make
# `--role authority` exercise zero real Authority LLM calls, defeating the
# whole point of this smoke runner. This proposal instead keeps the same
# action/resource as EXAMPLE_BUDGET (read/file) but requests a broader scope
# ("/reports/" instead of the granted "/reports/2026-08/"), which is exactly
# what check_authority() classifies as scope_exceeded — negotiable, so the
# model-backed path actually calls the LLM once. --role all is unaffected:
# it always uses DelegateAgent's real proposal, never this constant.
EXAMPLE_SCOPE_EXCEEDED_PROPOSAL = Interpretation("read", "file", "/reports/", frozenset())


def _fmt_interpretation(i: Interpretation) -> str:
    condition = ",".join(sorted(i.condition)) or "none"
    return f"ACTION: {i.action}\nRESOURCE: {i.resource}\nSCOPE: {i.scope}\nCONDITION: {condition}"


def _print_call_stats(responses: list[LLMResponse]) -> None:
    input_tokens = [r.input_tokens for r in responses if r.input_tokens is not None]
    output_tokens = [r.output_tokens for r in responses if r.output_tokens is not None]
    latencies = [r.latency_ms for r in responses if r.latency_ms is not None]

    print(f"LLM calls: {len(responses)}")
    print(f"Input tokens: {sum(input_tokens) if input_tokens else 'n/a'}")
    print(f"Output tokens: {sum(output_tokens) if output_tokens else 'n/a'}")
    if latencies:
        detail = ", ".join(f"{l:.1f}ms" for l in latencies)
        print(f"Latency: {sum(latencies):.1f}ms total ({detail})")
    else:
        print("Latency: n/a")


def build_llm_client(model: str) -> OpenAILLMClient:
    """실제 OpenAI API를 쓰는 client를 만든다. `openai` 패키지와
    `OPENAI_API_KEY` 환경변수가 필요하다 — 둘 중 하나라도 없으면 이 함수가
    바로 명확한 안내와 함께 실패한다. API key는 여기서도, 이 스크립트 어디
    에서도 코드/커맨드라인 인자로 받지 않는다 — `openai.OpenAI()`가
    환경변수에서 직접 읽는다."""
    try:
        import openai  # lazy — 이 스크립트를 실제로 실행할 때만 필요하다
    except ImportError as e:
        raise SystemExit(
            'openai 패키지가 설치돼 있지 않다. `pip install -e ".[agent]"`로 설치하라.'
        ) from e

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY 환경변수가 설정돼 있지 않다.\n"
            "  bash:       export OPENAI_API_KEY=...\n"
            '  PowerShell: $env:OPENAI_API_KEY="..."'
        )

    return OpenAILLMClient(client=openai.OpenAI(), model=model)


# ----------------------------------------------------------------------------
# Role runners — each prints its own "=== X ===" block and returns the raw
# result so --role all can thread real output from one stage into the next
# instead of re-printing or duplicating logic.
# ----------------------------------------------------------------------------
def run_principal(llm_client, goal: str = EXAMPLE_GOAL,
                  context: str = EXAMPLE_CONTEXT
                  ) -> tuple[PrincipalDelegation, PrincipalIntent]:
    print("=== PrincipalAgent ===\n")
    agent = PrincipalAgent(llm=llm_client)

    print("Goal:")
    print(goal)
    print()

    deleg = agent.delegate(goal=goal, context=context)
    print("Delegation:")
    print(deleg.delegation)
    print()

    # delegate()와 완전히 독립적인 두 번째 호출 — B의 proposal을 보지 않고
    # 원래 goal/context만으로 다시 의도를 구조화한다(B6의 oracle-free
    # reference가 될 값).
    intent = agent.restate_intent(goal=goal, context=context)
    print("Structured restatement (independent of any delegation/proposal):")
    print(_fmt_interpretation(intent.intended_action))
    print()

    _print_call_stats([deleg.response, intent.response])
    print()
    return deleg, intent


def run_delegate(llm_client, delegation: str = EXAMPLE_DELEGATION) -> DelegateProposal:
    print("=== DelegateAgent ===\n")
    agent = DelegateAgent(llm=llm_client)

    print("Delegation:")
    print(delegation)
    print()

    proposal = agent.propose(delegation=delegation)
    print("Proposed action:")
    print(_fmt_interpretation(proposal.interpretation))
    print()

    _print_call_stats([proposal.response])
    print()
    return proposal


def run_semantic(llm_client, delegation: str = EXAMPLE_DELEGATION,
                 proposal: Interpretation = EXAMPLE_MISREAD_PROPOSAL) -> SemanticVerdict:
    print("=== SemanticVerifierAgent ===\n")
    agent = SemanticVerifierAgent(llm_client=llm_client)

    print("Delegation:")
    print(delegation)
    print()
    print("Delegate proposal:")
    print(_fmt_interpretation(proposal))
    print()

    verdict = agent.verify_agent_proposal(delegation=delegation, proposal=proposal)
    print("Semantic verifier's own independent reading:")
    print(_fmt_interpretation(verdict.interpretation))
    print()
    print(f"Semantic verdict: {verdict.status} ({verdict.route})")
    print()

    responses = [verdict.response] if verdict.response is not None else []
    _print_call_stats(responses)
    print()
    return verdict


def run_authority(llm_client, proposal: Interpretation = EXAMPLE_SCOPE_EXCEEDED_PROPOSAL,
                  budget: Budget = EXAMPLE_BUDGET) -> AuthorityVerdict:
    """기본 proposal은 EXAMPLE_MISREAD_PROPOSAL(export)이 아니라
    EXAMPLE_SCOPE_EXCEEDED_PROPOSAL(read, 더 넓은 scope)이다 — action
    자체가 위임 밖인 export는 no_grant 하드 리젝트라 LLM을 전혀 부르지
    않는다. scope_exceeded여야 실제로 model-backed 협상 경로(LLM 호출
    1회 + 재검증)가 돌아간다."""
    print("=== AuthorityVerifierAgent ===\n")
    agent = AuthorityVerifierAgent(llm_client=llm_client)

    print("Proposal:")
    print(_fmt_interpretation(proposal))
    print()
    print(f"Delegated authority (budget): {budget}")
    print()

    verdict = agent.verify_agent_proposal(proposal=proposal, budget=budget)
    print(f"Authority verdict: {verdict.status.upper()} — {verdict.reason}")
    print()

    responses = [verdict.response] if verdict.response is not None else []
    _print_call_stats(responses)
    print()
    return verdict


def run_all(llm_client) -> None:
    print("############################################################")
    print("# Full chain: Principal -> Delegate -> Semantic + Authority")
    print("# Fusion (EXECUTE/REJECT) is deliberately NOT computed here —")
    print("# this role exists to look at each stage's raw, unfused output")
    print("# in isolation. For the actual fused decision, real B6 runtime,")
    print("# use --role runtime instead.")
    print("############################################################\n")

    deleg, intent = run_principal(llm_client)
    proposal = run_delegate(llm_client, delegation=deleg.delegation)
    sem_verdict = run_semantic(llm_client, delegation=deleg.delegation,
                               proposal=proposal.interpretation)
    auth_verdict = run_authority(llm_client, proposal=proposal.interpretation,
                                 budget=EXAMPLE_BUDGET)

    print("=== Comparison: Principal's independent restatement vs. B's proposal ===\n")
    print("Principal's independent restatement:")
    print(_fmt_interpretation(intent.intended_action))
    print()
    print("Delegate's actual proposal:")
    print(_fmt_interpretation(proposal.interpretation))
    print()
    if intent.intended_action != proposal.interpretation:
        print("NOTE: these diverge. This is exactly the independent runtime")
        print("signal B6's oracle-free fusion is designed to use — neither")
        print("value came from task.truth, and Semantic/Authority never saw")
        print("this restatement before producing their own verdicts.")
    else:
        print("NOTE: these agree in this run — not guaranteed on every call,")
        print("since both delegate() and restate_intent() are real, separate")
        print("model calls over natural language.")
    print()

    all_responses = [deleg.response, intent.response, proposal.response]
    if sem_verdict.response is not None:
        all_responses.append(sem_verdict.response)
    if auth_verdict.response is not None:
        all_responses.append(auth_verdict.response)

    print("=== Totals across all four agents ===\n")
    _print_call_stats(all_responses)


def run_runtime(llm_client, goal: str = EXAMPLE_GOAL, context: str = EXAMPLE_CONTEXT,
                budget: Budget = EXAMPLE_BUDGET) -> AgentRuntimeResult:
    """B6의 실제 oracle-free runtime을 그대로 호출한다 — fusion 로직을 여기서
    다시 구현하지 않는다. `AgentDelegationRuntime.run()`이 하는 것과 똑같이
    네 Agent를 실제 API로 순서대로 호출하고, 그 결과(`AgentRuntimeResult`)를
    그대로 출력만 한다."""
    print("############################################################")
    print("# AgentDelegationRuntime.run() — B6 oracle-free runtime, real API")
    print("# (calls the actual runtime class; fusion is not reimplemented")
    print("#  here)")
    print("############################################################\n")

    runtime = AgentDelegationRuntime(
        principal=PrincipalAgent(llm=llm_client),
        delegate=DelegateAgent(llm=llm_client),
        semantic_verifier=SemanticVerifierAgent(llm_client=llm_client),
        authority_verifier=AuthorityVerifierAgent(llm_client=llm_client),
    )
    result = runtime.run(goal=goal, context=context, budget=budget)

    print("Principal delegation:")
    print(result.delegation.delegation)
    print()

    print("Principal independent intent:")
    print(_fmt_interpretation(result.principal_intent.intended_action))
    print()

    print("Delegate proposal:")
    print(_fmt_interpretation(result.proposal.interpretation))
    print()

    print(f"Semantic: {result.semantic_verdict.status} ({result.semantic_verdict.route})")
    print("Semantic verifier's own independent reading:")
    print(_fmt_interpretation(result.semantic_verdict.interpretation))
    print()

    print(f"Authority: {result.authority_verdict.status.upper()} — "
         f"{result.authority_verdict.reason}")
    print()

    print("Final interpretation:")
    print(_fmt_interpretation(result.final_interpretation))
    print()

    print(f"Principal match: {result.principal_match}")
    print()

    print(f"FINAL DECISION: {result.decision}")
    print(f"Reason: {result.reason}")
    print()

    responses = [result.delegation.response, result.principal_intent.response,
                result.proposal.response]
    if result.semantic_verdict.response is not None:
        responses.append(result.semantic_verdict.response)
    if result.authority_verdict.response is not None:
        responses.append(result.authority_verdict.response)
    _print_call_stats(responses)
    return result


def main(argv: list[str] | None = None) -> int:
    try:  # Windows 기본 콘솔(cp949 등)의 UnicodeEncodeError 방지
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--role", required=True,
                   choices=["principal", "delegate", "semantic", "authority",
                            "all", "runtime"])
    p.add_argument("--model", default="gpt-4o-mini")
    args = p.parse_args(argv)

    llm_client = build_llm_client(args.model)

    if args.role == "principal":
        run_principal(llm_client)
    elif args.role == "delegate":
        run_delegate(llm_client)
    elif args.role == "semantic":
        run_semantic(llm_client)
    elif args.role == "authority":
        run_authority(llm_client)
    elif args.role == "runtime":
        run_runtime(llm_client)
    else:
        run_all(llm_client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
