"""Runtime Integration Phase 2B -- Real API E2E smoke (docs/experiments/
agent_connected_eval.md §28).

    python experiments/runtime_e2e_real.py \\
        --api-key-file "C:\\Users\\user\\Downloads\\files\\openai_api_key.txt"

Confirms that Process A (`PrincipalAgent` + `AgentDelegationRuntime` +
`SemanticVerifierAgent` + `AuthorityVerifierAgent`, all real API) and
Process B (`experiments/delegate_server.py`, launched as a genuinely
separate OS process via `subprocess.Popen`, real API) complete a full
end-to-end run over real HTTP (`dualflow.remote_delegate.RemoteDelegateAgent`),
for three smoke cases, exactly once each (ambiguous gets a small, capped
retry budget -- see below) -- NOT a statistics-gathering run (that was
B7e's job, already done; this asks a different question: does the
validated mechanism work over a real, separate-process Agent A/B
transport, end to end).

Verifiers stay on the Process A side throughout -- the Delegate process
never verifies its own delegation.

NOT MODIFIED for this experiment (confirmed via `git status` after this
commit): `AgentDelegationRuntime` (agent_runtime.py), `remote_delegate.py`,
and every B7 module (`clarification.py`, `agent_experience.py`,
`experience_decision.py`, `experience_evidence.py`, `delegate_agent.py`).
Sampling `n`, `entropy_threshold`, and the model default are the exact
values already validated by B7e/Phase 1 (n=20, entropy_threshold=0.8,
gpt-4o-mini-2024-07-18) -- not retuned here, so this experiment isolates
transport/integration validity from configuration effects.

Three cases (delegation/context/budget all reused from `agent_smoke.py`'s
canonical August scenario, except case 3's delegation/goal, which are new,
minimal, and mirror the existing TASK2_DELEGATION explicit-action phrasing
style from `experience_representation.py`):

  1. Clear      -- EXAMPLE_DELEGATION (already unambiguous: "create an
                   internal summary...") + EXAMPLE_BUDGET (grants
                   summarize/read). Expected (not forced): low entropy,
                   clarification skipped, EXECUTE.
  2. Ambiguous   -- EXAMPLE_AMBIGUOUS_DELEGATION ("prepare the report",
                   open action) + EXAMPLE_GOAL (Principal's true intent:
                   summarize only) + EXAMPLE_BUDGET. Expected: real
                   sampling lands ambiguous, a real `/ask-clarification`
                   round trip occurs, Principal answers for real,
                   post-sampling converges, EXECUTE. Retried up to
                   `--ambiguous-max-attempts` (default 3) ONLY if a given
                   attempt's real sampling happens to land stable -- never
                   retried to chase a particular outcome once clarified.
  3. Authority violation -- a new, deliberately explicit "export" request
                   against EXAMPLE_BUDGET, which grants no export privilege
                   at all. Expected: low entropy (explicit action),
                   Semantic confirms the (clear) proposal, Authority hard-
                   rejects (no_grant, zero Authority LLM calls) -> REJECT.

API key handling: read once from `--api-key-file` into this process's
`os.environ["OPENAI_API_KEY"]` (never printed, logged, or passed on the
command line) -- the delegate_server.py subprocess inherits it
automatically via normal environment inheritance. Removed again in a
`finally` block regardless of outcome.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXPERIMENTS_DIR))
from agent_smoke import (  # noqa: E402
    EXAMPLE_AMBIGUOUS_DELEGATION, EXAMPLE_BUDGET, EXAMPLE_CONTEXT, EXAMPLE_DELEGATION,
    EXAMPLE_GOAL, build_llm_client,
)

from dualflow.agent_runtime import AgentDelegationRuntime, AgentRuntimeResult  # noqa: E402
from dualflow.authority_feedback import AuthorityVerifierAgent  # noqa: E402
from dualflow.principal_agent import PrincipalAgent  # noqa: E402
from dualflow.remote_delegate import RemoteDelegateAgent  # noqa: E402
from dualflow.semantic import SemanticVerifierAgent  # noqa: E402

# B7e/Runtime Integration Phase 1's already-validated configuration --
# not retuned here, per instruction.
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_N = 20
DEFAULT_ENTROPY_THRESHOLD = 0.8

# Case 3 (authority violation) needs a delegation/goal pair that doesn't
# exist among agent_smoke.py's canonical constants -- new, minimal, and
# mirroring experience_representation.py's TASK2_DELEGATION explicit-
# action phrasing ("This time, export the ... report ...") for consistency
# with this project's established wording conventions, just applied to the
# same canonical August scope EXAMPLE_CONTEXT/EXAMPLE_BUDGET already use.
EXPORT_GOAL = ("Export the August 2026 financial report to the external auditor "
              "as a file.")
EXPORT_DELEGATION = ("Please export the August 2026 financial report to the "
                     "external auditor.")


def _read_api_key(path: str) -> str:
    key = Path(path).read_text(encoding="utf-8").strip()
    if not key:
        raise SystemExit(f"API key file is empty: {path!r}")
    return key


def _wait_for_port(host: str, port: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError as e:
            last_error = e
            time.sleep(0.2)
    raise RuntimeError(
        f"delegate_server did not start listening on {host}:{port} within {timeout}s "
        f"(last error: {last_error})")


def _start_delegate_server(*, model: str, host: str, port: int) -> subprocess.Popen:
    script = str(_EXPERIMENTS_DIR / "delegate_server.py")
    # 저장소 안이 아니라 OS temp dir에 남긴다 -- 이 로그는 이번 실행의 디버깅
    # 용도일 뿐, 커밋 대상이 아니다(요청 URL/상태 코드만 담고, API 키는
    # delegate_server.py 자신도 절대 출력하지 않는다 -- 확인됨).
    log_path = Path(tempfile.gettempdir()) / "dualflow_delegate_server_e2e.log"
    log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 -- closed by caller on cleanup
    proc = subprocess.Popen(
        [sys.executable, script, "--model", model, "--host", host, "--port", str(port)],
        stdout=log_file, stderr=subprocess.STDOUT)
    proc._e2e_log_file = log_file  # type: ignore[attr-defined] -- stash for cleanup
    return proc


def _stop_delegate_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    log_file = getattr(proc, "_e2e_log_file", None)
    if log_file is not None:
        log_file.close()


def _belief_summary(distribution) -> dict[str, float]:
    by_action: dict[str, float] = {}
    for interp, p in distribution.belief.items():
        by_action[interp.action] = by_action.get(interp.action, 0.0) + p
    return by_action


def _trace_row(case: str, result: AgentRuntimeResult) -> dict:
    """Phase 2B.1 follow-up: `pre_raw_responses`/`post_raw_responses` capture
    each individual sample's raw completion text (not just the aggregated
    belief) -- so a stable/low-entropy result can be told apart from "the
    raw text actually varied but parse_structured_action() collapsed it to
    the same action" vs. "the raw completions themselves were already
    near-identical". Phase 2B's original 3-case run (before this fix) did
    not capture this -- its saved trace only has the aggregated belief, and
    is not retroactively rewritten."""
    cr = result.clarification_result
    row = {
        "case": case,
        "principal_intent_action": result.principal_intent.intended_action.action,
        "proposal_action": result.proposal.interpretation.action,
        "pre_entropy": cr.pre_distribution.entropy if cr else None,
        "pre_belief": _belief_summary(cr.pre_distribution) if cr else None,
        "pre_raw_responses": ([r.text for r in cr.pre_distribution.responses] if cr else None),
        "clarified": cr.clarified if cr else None,
        "clarification_question": (cr.question.question if cr and cr.question else None),
        "principal_clarification_answer": (cr.answer.answer if cr and cr.answer else None),
        "post_entropy": (cr.post_distribution.entropy
                        if cr and cr.post_distribution else None),
        "post_belief": (_belief_summary(cr.post_distribution)
                        if cr and cr.post_distribution else None),
        "post_raw_responses": ([r.text for r in cr.post_distribution.responses]
                              if cr and cr.post_distribution else None),
        "final_interpretation": {
            "action": result.final_interpretation.action,
            "resource": result.final_interpretation.resource,
            "scope": result.final_interpretation.scope,
            "condition": sorted(result.final_interpretation.condition),
        },
        "semantic_confirmed": result.semantic_verdict.confirmed,
        "semantic_route": result.semantic_verdict.route,
        "authority_allowed": result.authority_verdict.allowed,
        "authority_reason": result.authority_verdict.reason,
        "principal_match": result.principal_match,
        "decision": result.decision,
        "reason": result.reason,
    }
    print(json.dumps(row, indent=2))
    return row


def run_case_clear(runtime: AgentDelegationRuntime) -> dict:
    print("\n=== Case 1: Clear ===")
    result = runtime.run(goal=EXAMPLE_GOAL, context=EXAMPLE_CONTEXT, budget=EXAMPLE_BUDGET)
    return _trace_row("clear", result)


def run_case_ambiguous(runtime: AgentDelegationRuntime, *, max_attempts: int) -> dict:
    print("\n=== Case 2: Ambiguous ===")
    last_row: dict | None = None
    for attempt in range(1, max_attempts + 1):
        print(f"-- attempt {attempt}/{max_attempts} --")
        result = runtime.run(goal=EXAMPLE_GOAL, context=EXAMPLE_CONTEXT, budget=EXAMPLE_BUDGET)
        last_row = _trace_row("ambiguous", result)
        last_row["attempt"] = attempt
        if last_row["clarified"]:
            return last_row
        print(f"(real sampling landed stable on attempt {attempt} -- not forcing, "
             f"{'retrying' if attempt < max_attempts else 'giving up, reporting as-is'})")
    return last_row  # last attempt's row, honestly reported even if never clarified


def run_case_authority_violation(runtime: AgentDelegationRuntime) -> dict:
    print("\n=== Case 3: Authority violation ===")
    budget = EXAMPLE_BUDGET  # no export privilege granted at all
    result = runtime.run(goal=EXPORT_GOAL, context=EXAMPLE_CONTEXT, budget=budget)
    return _trace_row("authority_violation", result)


def main(argv: list[str] | None = None) -> int:
    try:  # Windows 기본 콘솔(cp949 등)의 UnicodeEncodeError 방지
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-key-file", required=True,
                   help="path to a file containing only the OpenAI API key -- "
                        "never pass the key itself on the command line")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--entropy-threshold", type=float, default=DEFAULT_ENTROPY_THRESHOLD)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--ambiguous-max-attempts", type=int, default=3,
                   help="capped retry budget for case 2 ONLY, and only if real sampling "
                        "happens to land stable -- never retried to chase an outcome")
    p.add_argument("--output", default=None, help="path to save the raw trace as JSON")
    args = p.parse_args(argv)

    api_key = _read_api_key(args.api_key_file)
    os.environ["OPENAI_API_KEY"] = api_key
    del api_key  # 로컬 변수에도 더 오래 남겨두지 않는다

    server_proc: subprocess.Popen | None = None
    try:
        print("=== Runtime Integration Phase 2B -- Real API E2E smoke ===")
        print(f"model={args.model} n={args.n} entropy_threshold={args.entropy_threshold}")
        print(f"starting delegate_server.py as a separate process on 127.0.0.1:{args.port} ...")
        server_proc = _start_delegate_server(model=args.model, host="127.0.0.1", port=args.port)
        _wait_for_port("127.0.0.1", args.port)
        print(f"delegate_server.py is up (pid={server_proc.pid}, "
             f"log={Path(tempfile.gettempdir()) / 'dualflow_delegate_server_e2e.log'}).")

        remote_delegate = RemoteDelegateAgent(base_url=f"http://127.0.0.1:{args.port}")
        llm_client = build_llm_client(args.model)  # Process A's own real client (Principal/Semantic/Authority)

        def _make_runtime() -> AgentDelegationRuntime:
            return AgentDelegationRuntime(
                principal=PrincipalAgent(llm=llm_client),
                delegate=remote_delegate,
                semantic_verifier=SemanticVerifierAgent(llm_client=llm_client),
                authority_verifier=AuthorityVerifierAgent(llm_client=llm_client),
                use_clarification=True, n=args.n, entropy_threshold=args.entropy_threshold)

        rows = [
            run_case_clear(_make_runtime()),
            run_case_ambiguous(_make_runtime(), max_attempts=args.ambiguous_max_attempts),
            run_case_authority_violation(_make_runtime()),
        ]

        print("\n=== Phase 2B checklist ===")
        checklist = {
            "process_a_process_b_separate_processes": True,
            "http_real_delegate_call_confirmed": True,
            "clear_case_decision": rows[0]["decision"],
            "ambiguous_case_clarified": rows[1]["clarified"],
            "ambiguous_case_decision": rows[1]["decision"],
            "authority_violation_authority_allowed": rows[2]["authority_allowed"],
            "authority_violation_decision": rows[2]["decision"],
        }
        print(json.dumps(checklist, indent=2))

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump({"identity": {"model": args.model, "n": args.n,
                                        "entropy_threshold": args.entropy_threshold},
                          "checklist": checklist, "rows": rows}, f, indent=2)
            print(f"\nSaved raw trace to {args.output}")

        return 0
    finally:
        if server_proc is not None:
            _stop_delegate_server(server_proc)
        os.environ.pop("OPENAI_API_KEY", None)


if __name__ == "__main__":
    raise SystemExit(main())
