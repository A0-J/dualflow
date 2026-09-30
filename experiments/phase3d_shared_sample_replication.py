"""Phase 3D -- Shared-Sample Controlled Replication (docs/experiments/
agent_connected_eval.md §29; docs/experiments/phase3c_methods_for_paper.md
§9.1, §5.4).

    python experiments/phase3d_shared_sample_replication.py \\
        --api-key-file "C:\\Users\\user\\Downloads\\files\\openai_api_key.txt" \\
        --frozen-input "<path to the frozen phase2c_final.jsonl>" \\
        --replicate-id run1 \\
        --output "<path>"

**Purpose (fixed scope, per instruction)**: this is NOT a reproduction of
the full 5-task Phase 3C benchmark. It is a corrected, controlled
replication targeting only the 2 failure modes where Phase 3C actually
showed an Arm difference: `confident_semantic_misread` (Safety) and
`condition_violation` (Utility/false-reject).

**What this fixes relative to the original Phase 3C run** (found while
designing this follow-up, §5.4 of the methods doc, frozen Phase 3C
numbers left unchanged): the original run had Arm B and Arm C draw
*independent* sets of 20 `restate_intent()` samples per episode, so
their comparison could not cleanly separate "aggregation-rule
difference" from "sampling variance between two different draws." This
script draws **exactly one shared set of 20** `restate_intent()`
responses per episode; both Arm B's rule (full-Interpretation majority)
and Arm C's rule (facet-wise entropy + confirmed-facet blocking) are
computed from that same set in a separate analysis pass (`experiments/
phase3d_analysis.py`, not this collection script) -- this collection
script's only job is to gather and preserve the raw data, unmodified.

**Full raw logging** (this is the point of the whole exercise): all 20
individual `restate_intent()` responses are kept, in call order, per
episode -- action/resource/scope/condition for each, plus the raw
response text. This lets `phase3d_analysis.py` recompute, with ZERO
further API calls:
  - reproducibility at n=20/threshold=0.8 (the original Phase 3C
    setting) -- an independent replicate of the original pattern;
  - the fixed-n sample-count curve (n=1,3,5,10,15,20, via prefixes of
    the same 20 draws) -- n=1 doubles as an independent Arm-A
    reproducibility check for free;
  - a partial threshold ablation (0.4/0.6/0.8/1.0), by re-deriving
    facet-wise entropy from the same raw responses at whatever n.

Delegate(B)'s output, the Semantic Verifier's judgment, and Authority's
judgment are all reused unchanged from the frozen Phase 2C Final data
(0 new API cost for those) -- exactly the same "hold everything but the
Principal-side mechanism fixed" principle as the original Phase 3C.

Explicitly OUT of scope for this script and this phase (per instruction):
adaptive early stopping, runtime integration, model diversification, and
the other 3 Phase 2C Final tasks (no Arm difference was observed there).

API key handling identical to every other real-API step in this
project: read once from `--api-key-file` into `os.environ`, removed
again in a `finally` block. Never printed, logged, or passed on the
command line.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXPERIMENTS_DIR))
from agent_bench_tasks import EXPECTED_OUTCOMES, TASKS  # noqa: E402
from agent_smoke import build_llm_client  # noqa: E402
from runtime_e2e_real import DEFAULT_MODEL, DEFAULT_N, _read_api_key  # noqa: E402

from dualflow.principal_agent import PrincipalAgent  # noqa: E402

TARGET_TASK_NAMES = ("confident_semantic_misread", "condition_violation")


def _find_task(name: str):
    for task in TASKS:
        if task.name == name:
            return task
    raise KeyError(name)


def _load_frozen_episodes(jsonl_path: str, task_name: str) -> list[dict]:
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row["task_id"] == task_name:
                    rows.append(row)
    return rows


def collect_one_episode(principal: PrincipalAgent, task, frozen_row: dict, n: int) -> dict:
    """Draws ONE shared set of `n` independent `restate_intent()` calls --
    both Arm B and Arm C are computed from this same set in a later,
    separate analysis pass. Stores every raw response, in order."""
    ideal = EXPECTED_OUTCOMES[task.name].ideal  # copied forward for self-contained analysis
    responses = []
    for _ in range(n):
        intent = principal.restate_intent(goal=task.goal, context=task.context)
        interp = intent.intended_action
        resp = intent.response
        responses.append({
            "action": interp.action, "resource": interp.resource, "scope": interp.scope,
            "condition": sorted(interp.condition), "raw_text": intent.raw_text,
            # 2026-09-30 추가 -- 글자 수로 토큰/비용을 추정하지 않고, API가
            # 실제로 보고한 usage/model/sampling parameter를 매 호출
            # 그대로 기록한다(§29 cost/billing 불일치 논의 참고).
            "served_model": resp.model, "temperature": resp.temperature, "top_p": resp.top_p,
            "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
            "cached_input_tokens": resp.cached_input_tokens,
        })

    return {
        "task_id": task.name,
        "run_id": frozen_row["run_id"],
        # Copied forward, unchanged, from the frozen Phase 2C Final record --
        # NOT recomputed here (0 new API cost for these fields):
        "delegate_final_interpretation": frozen_row["final_interpretation"],
        "semantic_confirmed": frozen_row["semantic_confirmed"],
        "authority_allowed": frozen_row["authority_allowed"],
        "ideal_action": ideal.action, "ideal_resource": ideal.resource,
        "ideal_scope": ideal.scope, "ideal_condition": sorted(ideal.condition),
        "ideal_authorized": EXPECTED_OUTCOMES[task.name].ideal_authorized,
        # The one shared set of n raw responses, in call order -- the actual
        # new data this script collects:
        "restate_responses": responses,
    }


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-key-file", required=True)
    p.add_argument("--frozen-input", required=True)
    p.add_argument("--replicate-id", required=True,
                  help="label for this independent run, e.g. 'run1'/'run2' -- "
                       "stored in every row so replicates are never conflated")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--output", required=True)
    p.add_argument("--budget-only", action="store_true")
    args = p.parse_args(argv)

    episodes_by_task = {name: _load_frozen_episodes(args.frozen_input, name)
                        for name in TARGET_TASK_NAMES}
    n_episodes = sum(len(v) for v in episodes_by_task.values())
    total_calls = n_episodes * args.n
    print(f"=== Phase 3D shared-sample replication [{args.replicate_id}] ===")
    print(f"target tasks: {list(TARGET_TASK_NAMES)}  n_episodes={n_episodes}  n={args.n}")
    print(f"computed budget: {n_episodes} episodes x {args.n} calls = {total_calls} "
         f"total new API calls (shared by both Arm B and Arm C in later analysis)")
    if args.budget_only:
        return 0

    api_key = _read_api_key(args.api_key_file)
    os.environ["OPENAI_API_KEY"] = api_key
    del api_key
    try:
        principal = PrincipalAgent(llm=build_llm_client(args.model))
        rows = []
        completed = 0
        for task_name, frozen_rows in episodes_by_task.items():
            task = _find_task(task_name)
            for frozen_row in frozen_rows:
                completed += 1
                print(f"\n[{completed}/{n_episodes}] task={task_name} run={frozen_row['run_id']}",
                     flush=True)
                row = collect_one_episode(principal, task, frozen_row, args.n)
                row["replicate_id"] = args.replicate_id
                rows.append(row)
                actions = [r["action"] for r in row["restate_responses"]]
                from collections import Counter
                print(f"  action distribution: {dict(Counter(actions))}")

        with open(args.output, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"\nSaved {len(rows)} episodes ({args.replicate_id}) to {args.output}")
        return 0
    finally:
        os.environ.pop("OPENAI_API_KEY", None)


if __name__ == "__main__":
    raise SystemExit(main())
