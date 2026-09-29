"""Phase 2C Measurement Audit (docs/experiments/agent_connected_eval.md
§29 follow-up) -- isolates WHERE the cross-run instability found in
calibration actually comes from, before concluding anything about entropy
estimators.

Calibration showed: within any single `sample_candidates(n=20)` call, all
20 completions converge to the same interpretation (near-byte-identical
raw text) -- but independent `AgentDelegationRuntime.run()` invocations on
the SAME goal/context land on DIFFERENT confident interpretations. That
alone does not tell us whether the instability is in:
  (a) the Delegate itself (same delegation text, in, still varies out), or
  (b) PrincipalAgent.delegate() producing a DIFFERENT delegation string
      each independent run (the Delegate would then be perfectly
      consistent given identical input -- the instability is upstream).

This script isolates exactly that, per instruction:
  1. Call `PrincipalAgent.delegate(goal, context)` ONCE per task (real API)
     -- freeze the resulting delegation string byte-for-byte.
  2. Call `DelegateAgent.propose(delegation=<frozen>, context=...)`
     `--requests` times (default 20), each a FULLY INDEPENDENT, separate
     API request/Python-level call -- deliberately NOT `sample_candidates
     (n=20)` (a single Python call making N `generate()` calls), which is
     exactly the distinction this audit exists to check does or doesn't
     matter. No history/state is shared between requests -- every call
     constructs nothing but a fresh `input_text`/`instructions` pair,
     identical to what `sample_candidates()` already does per-sample
     internally (same `_PROPOSE_INSTRUCTIONS`, hashed below for the
     record).

Bypasses `AgentDelegationRuntime`/`RemoteDelegateAgent`/`delegate_server.py`
entirely, on purpose -- this audits MODEL behavior (does an identical
prompt, sent as genuinely separate requests, produce a stable answer),
not transport. Calls `PrincipalAgent`/`DelegateAgent` directly, in-process.
No B7 module, `AgentDelegationRuntime`, or `remote_delegate.py` is
modified by this script.

Three tasks (per instruction): one clear control (`clear_read`) and the
two calibration candidates that showed cross-run instability
(`calib_scope_and_action_ambiguous`, `calib_stronger_misread_bait`).

    python experiments/agent_bench_measurement_audit.py \\
        --api-key-file "C:\\Users\\user\\Downloads\\files\\openai_api_key.txt" \\
        --requests 20 --output audit.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXPERIMENTS_DIR))
from agent_bench_calibration_tasks import CALIBRATION_TASKS  # noqa: E402
from agent_bench_tasks import TASKS  # noqa: E402
from agent_smoke import build_llm_client  # noqa: E402
from runtime_e2e_real import DEFAULT_MODEL, _read_api_key  # noqa: E402

from dualflow.delegate_agent import DelegateAgent, _PROPOSE_INSTRUCTIONS  # noqa: E402
from dualflow.principal_agent import PrincipalAgent  # noqa: E402

AUDIT_TASK_NAMES = ("clear_read", "calib_scope_and_action_ambiguous",
                   "calib_stronger_misread_bait")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _find_task(name: str):
    for task in list(TASKS) + list(CALIBRATION_TASKS):
        if task.name == name:
            return task
    raise KeyError(name)


def audit_one_task(task, *, principal: PrincipalAgent, model: str, n_requests: int) -> dict:
    delegation_result = principal.delegate(goal=task.goal, context=task.context)
    frozen_delegation = delegation_result.delegation
    print(f"  frozen delegation: {frozen_delegation!r}")

    rows = []
    for request_id in range(1, n_requests + 1):
        # A FRESH DelegateAgent + a fresh call each time -- no shared state,
        # no history, deliberately NOT DelegateAgent.sample_candidates(n=...).
        delegate = DelegateAgent(llm=build_llm_client(model))
        proposal = delegate.propose(delegation=frozen_delegation, context=task.context)
        rows.append({
            "task_id": task.name,
            "frozen_delegation": frozen_delegation,
            "frozen_delegation_sha256": _sha256(frozen_delegation),
            "system_prompt_sha256": _sha256(_PROPOSE_INSTRUCTIONS),
            "request_id": request_id,
            "raw_text": proposal.raw_text,
            "parsed_action": proposal.interpretation.action,
            "parsed_resource": proposal.interpretation.resource,
            "parsed_scope": proposal.interpretation.scope,
            "parsed_condition": sorted(proposal.interpretation.condition),
        })
        print(f"    request {request_id}/{n_requests}: "
             f"{proposal.interpretation.action}:{proposal.interpretation.scope}")

    distinct = {}
    for r in rows:
        key = (r["parsed_action"], r["parsed_scope"])
        distinct[key] = distinct.get(key, 0) + 1
    return {"task_id": task.name, "frozen_delegation": frozen_delegation, "rows": rows,
           "distinct_action_scope_counts": {f"{a}:{s}": c for (a, s), c in distinct.items()}}


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
    p.add_argument("--requests", type=int, default=20,
                   help="independent Delegate.propose() calls PER TASK against the frozen input")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    api_key = _read_api_key(args.api_key_file)
    os.environ["OPENAI_API_KEY"] = api_key
    del api_key
    try:
        tasks = [_find_task(name) for name in AUDIT_TASK_NAMES]
        print(f"=== Phase 2C Measurement Audit -- {len(tasks)} tasks x "
             f"(1 frozen delegate() + {args.requests} independent propose() calls) ===")
        print(f"model={args.model}")

        results = []
        with open(args.output, "w", encoding="utf-8") as out:
            for task in tasks:
                print(f"\n-- {task.name} --")
                principal = PrincipalAgent(llm=build_llm_client(args.model))
                result = audit_one_task(task, principal=principal, model=args.model,
                                        n_requests=args.requests)
                results.append(result)
                for row in result["rows"]:
                    out.write(json.dumps(row) + "\n")
                print(f"  distinct (action:scope) outcomes across {args.requests} independent "
                     f"requests to the SAME frozen delegation: "
                     f"{result['distinct_action_scope_counts']}")

        print("\n=== Interpretation guide (per instruction) ===")
        print("A. If a task's distinct_action_scope_counts has >1 entry: the Delegate itself "
             "varies given a byte-identical input -- genuine hidden semantic instability, "
             "independent of Principal delegation variance.")
        print("B. If every task's distinct_action_scope_counts has exactly 1 entry: the "
             "Delegate is stable given fixed input -- calibration's cross-run instability "
             "must have come from PrincipalAgent.delegate() itself varying, not the Delegate.")
        print(f"\nSaved {sum(len(r['rows']) for r in results)} rows to {args.output}")
        return 0
    finally:
        os.environ.pop("OPENAI_API_KEY", None)


if __name__ == "__main__":
    raise SystemExit(main())
