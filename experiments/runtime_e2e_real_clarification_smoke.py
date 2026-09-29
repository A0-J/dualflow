"""Runtime Integration Phase 2B.1 -- Targeted Real Clarification Smoke
(docs/experiments/agent_connected_eval.md §28 follow-up).

    python experiments/runtime_e2e_real_clarification_smoke.py \\
        --api-key-file "C:\\Users\\user\\Downloads\\files\\openai_api_key.txt"

Phase 2B's 3-case smoke fully validated Clear and Authority-violation over
real, separate-process Agent A/B HTTP -- but its Ambiguous case's real
sampling landed stable (entropy=0.000, summarize 20/20) on all 3 attempts.
That is a legitimate real-API observation, not a failure -- Phase 2A's
deterministic (fake-LLM) tests already proved the `/ask-clarification`
branch itself works correctly over real HTTP; what Phase 2B did NOT
observe is real model output actually reaching that branch end to end.

This script does exactly ONE additional targeted thing: re-run ONLY the
ambiguous case, up to `--max-attempts` (default 5) times, stopping at the
first real clarification. It deliberately does NOT change the delegation
wording, model, `n`, `entropy_threshold`, parser, runtime, or transport
code -- `EXAMPLE_AMBIGUOUS_DELEGATION`/`EXAMPLE_GOAL`/`EXAMPLE_CONTEXT` is
reused verbatim from `agent_smoke.py`, because it is not a new or
untested input -- B7e Phase 1's own three independent real chains already
observed real ambiguity with this EXACT wording (entropy 0.971 / 0.881 /
1.000, all in commit `8b701d4`'s real-API runs), so there is no reason to
engineer a new prompt; Phase 2B's 3/3-stable result was the statistical
outlier relative to that track record, not evidence the wording itself is
unreliable.

Success condition (the only thing this script tries to produce): one real
trace where `pre_entropy > entropy_threshold` -> `clarified == True` ->
a real `/ask-clarification` HTTP call actually happened -> a real Principal
answer -> a real post-clarification sampling round -> Semantic + Authority
verification -> a final runtime decision. If no attempt within the cap
produces this, the result is reported as NOT OBSERVED -- the threshold is
never lowered and the parser is never touched to manufacture a hit.

Every attempt's raw per-sample completions (not just the aggregated
belief) are recorded, so a stable result can be told apart from "the raw
text actually varied but parse_structured_action() collapsed it to the
same action" vs. "the raw completions were already near-identical" -- an
observation about what entropy actually measures (action-facet agreement
after parsing), not a bug either way.

AgentDelegationRuntime, remote_delegate.py, delegate_server.py, and every
B7 module are NOT modified for this script (confirmed via `git status`
after this commit) -- same as Phase 2B.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXPERIMENTS_DIR))
from agent_smoke import (  # noqa: E402
    EXAMPLE_AMBIGUOUS_DELEGATION, EXAMPLE_BUDGET, EXAMPLE_CONTEXT, EXAMPLE_GOAL, build_llm_client,
)
from runtime_e2e_real import (  # noqa: E402
    DEFAULT_ENTROPY_THRESHOLD, DEFAULT_MODEL, DEFAULT_N,
    _read_api_key, _start_delegate_server, _stop_delegate_server, _trace_row, _wait_for_port,
)

from dualflow.agent_runtime import AgentDelegationRuntime  # noqa: E402
from dualflow.authority_feedback import AuthorityVerifierAgent  # noqa: E402
from dualflow.principal_agent import PrincipalAgent  # noqa: E402
from dualflow.remote_delegate import RemoteDelegateAgent  # noqa: E402
from dualflow.semantic import SemanticVerifierAgent  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-key-file", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--entropy-threshold", type=float, default=DEFAULT_ENTROPY_THRESHOLD)
    p.add_argument("--port", type=int, default=8766)  # Phase 2B's own server may still be running
    p.add_argument("--max-attempts", type=int, default=5)
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)

    api_key = _read_api_key(args.api_key_file)
    os.environ["OPENAI_API_KEY"] = api_key
    del api_key

    server_proc = None
    try:
        print("=== Runtime Integration Phase 2B.1 -- Targeted Real Clarification Smoke ===")
        print(f"model={args.model} n={args.n} entropy_threshold={args.entropy_threshold} "
             f"max_attempts={args.max_attempts}")
        print("delegation/context/goal: EXAMPLE_AMBIGUOUS_DELEGATION/EXAMPLE_CONTEXT/"
             "EXAMPLE_GOAL, reused verbatim from agent_smoke.py -- unchanged from Phase 2B "
             "and from B7e Phase 1's own real chains.")

        server_proc = _start_delegate_server(model=args.model, host="127.0.0.1", port=args.port)
        _wait_for_port("127.0.0.1", args.port)
        print(f"delegate_server.py is up (pid={server_proc.pid}).")

        remote_delegate = RemoteDelegateAgent(base_url=f"http://127.0.0.1:{args.port}")
        llm_client = build_llm_client(args.model)

        rows: list[dict] = []
        observed = False
        for attempt in range(1, args.max_attempts + 1):
            print(f"\n-- attempt {attempt}/{args.max_attempts} --")
            runtime = AgentDelegationRuntime(
                principal=PrincipalAgent(llm=llm_client),
                delegate=remote_delegate,
                semantic_verifier=SemanticVerifierAgent(llm_client=llm_client),
                authority_verifier=AuthorityVerifierAgent(llm_client=llm_client),
                use_clarification=True, n=args.n, entropy_threshold=args.entropy_threshold)
            result = runtime.run(goal=EXAMPLE_GOAL, context=EXAMPLE_CONTEXT, budget=EXAMPLE_BUDGET)
            row = _trace_row("ambiguous_targeted", result)
            row["attempt"] = attempt
            rows.append(row)
            if row["clarified"]:
                observed = True
                print(f"\nOBSERVED on attempt {attempt}: pre_entropy={row['pre_entropy']:.3f} "
                     f"> threshold={args.entropy_threshold} -> real /ask-clarification -> "
                     f"post_entropy={row['post_entropy']:.3f} -> decision={row['decision']}")
                break
            print(f"(attempt {attempt}: pre_entropy={row['pre_entropy']:.3f} <= threshold "
                 f"-- real sampling landed stable, not forcing)")

        print("\n=== Phase 2B.1 result ===")
        if observed:
            print("Real clarification branch OBSERVED end to end -- Phase 2B can be closed PASS.")
        else:
            print("Real clarification branch NOT OBSERVED within the attempt cap. Reported "
                 "honestly, not forced -- threshold/parser were not touched. Phase 2B closes "
                 "with this branch recorded as an unobserved-in-this-run coverage gap "
                 "(the branch itself remains proven by Phase 2A's deterministic HTTP test).")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump({"identity": {"model": args.model, "n": args.n,
                                        "entropy_threshold": args.entropy_threshold,
                                        "max_attempts": args.max_attempts},
                          "observed": observed, "rows": rows}, f, indent=2)
            print(f"\nSaved raw trace to {args.output}")

        return 0
    finally:
        if server_proc is not None:
            _stop_delegate_server(server_proc)
        os.environ.pop("OPENAI_API_KEY", None)


if __name__ == "__main__":
    raise SystemExit(main())
