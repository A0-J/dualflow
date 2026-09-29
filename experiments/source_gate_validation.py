"""Phase 2C-P5 -- Source-Side Gate Validation (docs/experiments/
agent_connected_eval.md §29).

    python experiments/source_gate_validation.py \\
        --api-key-file "C:\\Users\\user\\Downloads\\files\\openai_api_key.txt"

Validates `dualflow.source_semantic_gate.SourceClarifyingPrincipal`
(commit `b1e2f86`, standalone, not wired into `AgentDelegationRuntime`)
against real API data, on 3 already-existing tasks (no new wording) --
the same 3 Phase 2C-P4 used, reused verbatim:

  V1 clear_read                        -- expect stable, no clarification
  V2 calib_scope_and_action_ambiguous  -- expect unstable (P4: action H=1.076),
                                          clarify, then real post-round convergence
  V3 calib_stronger_misread_bait       -- expect unstable (P4: action H=1.539,
                                          scope H=1.157), the strongest case

For V2/V3, after `SourceClarifyingPrincipal.resolve()` confirms a facet,
the resulting `final_delegation` is additionally passed through the
EXISTING, unmodified receiver-side pipeline
(`DelegateAgent.sample_candidates()`) to show the downstream effect: a
stable, confirmed delegation actually reaching the Delegate -- not merely
that source-side instability was detected.

No `delegate_server.py`/HTTP transport needed -- this audits Principal/
Delegate model behavior in-process, exactly like P3/P4 did, not
transport. `AgentDelegationRuntime`, `remote_delegate.py`,
`delegate_server.py`, `clarification.py`, and every B7 module are NOT
modified or touched by this script.
"""

from __future__ import annotations

import argparse
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

from dualflow.delegate_agent import DelegateAgent  # noqa: E402
from dualflow.principal_agent import PrincipalAgent  # noqa: E402
from dualflow.source_semantic_gate import SourceClarifyingPrincipal  # noqa: E402

V_TASK_NAMES = ("clear_read", "calib_scope_and_action_ambiguous", "calib_stronger_misread_bait")


def _find_task(name: str):
    for task in list(TASKS) + list(CALIBRATION_TASKS):
        if task.name == name:
            return task
    raise KeyError(name)


def validate_one_task(task, *, model: str, n: int) -> dict:
    principal = PrincipalAgent(llm=build_llm_client(model))
    delegate = DelegateAgent(llm=build_llm_client(model))
    clarifier = SourceClarifyingPrincipal(principal=principal, delegate=delegate, n=n)

    result = clarifier.resolve(goal=task.goal, context=task.context)

    row = {
        "task_id": task.name,
        "pre_entropy": result.pre_distribution.entropy,
        "pre_belief": {str(i): p for i, p in result.pre_distribution.belief.items()},
        "clarified": result.clarified,
        "target_facet": result.target_facet,
        "question": result.question,
        "answer": result.answer.answer if result.answer else None,
        "post_entropy": result.post_distribution.entropy if result.post_distribution else None,
        "final_interpretation": str(result.final_interpretation),
        "final_delegation": result.final_delegation,
    }

    # Downstream check: pass the confirmed final_delegation through the
    # EXISTING, unmodified receiver-side pipeline.
    receiver_delegate = DelegateAgent(llm=build_llm_client(model))
    receiver_distribution = receiver_delegate.sample_candidates(
        delegation=result.final_delegation, context=task.context, n=n)
    row["receiver_side_entropy_on_confirmed_delegation"] = receiver_distribution.entropy
    row["receiver_side_top_action"] = receiver_distribution.top.action

    return row


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
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)

    api_key = _read_api_key(args.api_key_file)
    os.environ["OPENAI_API_KEY"] = api_key
    del api_key
    try:
        print(f"=== Phase 2C-P5 -- Source-Side Gate Validation -- {len(V_TASK_NAMES)} tasks ===")
        print(f"model={args.model} n={args.n}")

        rows = []
        for name in V_TASK_NAMES:
            task = _find_task(name)
            print(f"\n-- {name} --")
            row = validate_one_task(task, model=args.model, n=args.n)
            rows.append(row)
            print(json.dumps(row, indent=2, default=str))

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, default=str)
            print(f"\nSaved to {args.output}")
        return 0
    finally:
        os.environ.pop("OPENAI_API_KEY", None)


if __name__ == "__main__":
    raise SystemExit(main())
