"""Phase 2C-P4 -- Principal Delegation Stability Audit (docs/experiments/
agent_connected_eval.md §29 follow-up).

Phase 2C-P3 (Measurement Audit) showed the Delegate is perfectly stable
given a FIXED delegation string (60/60 independent requests identical).
That localizes the cross-run instability calibration (P2) found to
`PrincipalAgent.delegate()` itself -- but P3 only froze ONE delegation per
task; it never measured how much `delegate()`'s OWN output varies across
independent calls on the SAME (goal, context). This script measures
exactly that -- the one remaining open question:

    Given the same original intent, how stable are the SEMANTIC FACETS
    of the delegation PrincipalAgent.delegate() independently produces?

For 5 tasks (the clear control plus all 4 existing calibration
candidates -- no new wording authored, per instruction), calls
`PrincipalAgent.delegate(goal, context)` 20 independent times (fresh
call each time, no shared state). Each resulting NATURAL-LANGUAGE
delegation string is then canonicalized into structured (action,
resource, scope, condition) facets by calling `DelegateAgent.propose()`
on it EXACTLY ONCE -- reusing the existing Delegate mechanism that already
exists specifically to convert a delegation string into a structured
Interpretation (not a new parser/heuristic; P3 already showed this step
is deterministic given a fixed string, so canonicalizing this way adds no
meaningful additional variance of its own).

Facet-wise agreement (`max_v count(v) / R`) and Shannon entropy
(`dualflow.semantic.entropy()`, reused unmodified -- it only calls
`.values()` on whatever dict it's given, so a plain `{action_value:
probability}` dict works exactly like the `Belief` type it's normally
called with) are computed independently for the action and scope facets.

Does NOT touch `framework.py`'s Semantic Flow, `AgentDelegationRuntime`,
`remote_delegate.py`, `delegate_server.py`, or any B7 module -- this is an
independent, read-only measurement harness. No threshold/prompt/parser
change, no temperature increase -- per instruction, inducing artificial
randomness would obscure the real structural finding rather than confirm
or refute it.

    python experiments/principal_delegation_audit.py \\
        --api-key-file "C:\\Users\\user\\Downloads\\files\\openai_api_key.txt" \\
        --requests 20 --output audit_p4.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXPERIMENTS_DIR))
from agent_bench_calibration_tasks import CALIBRATION_TASKS  # noqa: E402
from agent_bench_tasks import EXPECTED_OUTCOMES, TASKS  # noqa: E402
from agent_smoke import build_llm_client  # noqa: E402
from runtime_e2e_real import DEFAULT_MODEL, _read_api_key  # noqa: E402

from dualflow.delegate_agent import DelegateAgent  # noqa: E402
from dualflow.principal_agent import PrincipalAgent  # noqa: E402
from dualflow.semantic import entropy as compute_entropy  # noqa: E402

AUDIT_TASK_NAMES = ("clear_read", "calib_scope_ambiguous_action_fixed",
                   "calib_no_qualifying_verb", "calib_scope_and_action_ambiguous",
                   "calib_stronger_misread_bait")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _find_task(name: str):
    for task in list(TASKS) + list(CALIBRATION_TASKS):
        if task.name == name:
            return task
    raise KeyError(name)


def _facet_stats(values: list[str]) -> dict:
    counts = Counter(values)
    n = len(values)
    probs = {v: c / n for v, c in counts.items()}
    top_value, top_count = counts.most_common(1)[0]
    return {
        "counts": dict(counts),
        "agreement": top_count / n,
        "entropy": compute_entropy(probs),
        "modal_value": top_value,
    }


def audit_one_task(task, *, model: str, n_requests: int) -> dict:
    ideal = EXPECTED_OUTCOMES.get(task.name)
    rows = []
    for request_id in range(1, n_requests + 1):
        # Fresh Principal call -- independent of every other request.
        principal = PrincipalAgent(llm=build_llm_client(model))
        delegation_result = principal.delegate(goal=task.goal, context=task.context)
        raw_delegation = delegation_result.delegation

        # Canonicalize via the EXISTING Delegate mechanism (not a new
        # parser) -- P3 already showed this is deterministic given a fixed
        # string, so one call here is sufficient, not an extra source of
        # noise.
        delegate = DelegateAgent(llm=build_llm_client(model))
        proposal = delegate.propose(delegation=raw_delegation, context=task.context)
        interp = proposal.interpretation

        principal_match = None
        if ideal is not None and ideal.ideal is not None:
            principal_match = (interp.action == ideal.ideal.action
                              and interp.resource == ideal.ideal.resource
                              and interp.scope == ideal.ideal.scope
                              and set(interp.condition) == set(ideal.ideal.condition))

        rows.append({
            "task_id": task.name,
            "original_goal": task.goal,
            "principal_request_id": request_id,
            "raw_delegation": raw_delegation,
            "delegation_sha256": _sha256(raw_delegation),
            "parsed_action": interp.action,
            "parsed_resource": interp.resource,
            "parsed_scope": interp.scope,
            "parsed_condition": sorted(interp.condition),
            "principal_match": principal_match,
        })
        print(f"    request {request_id}/{n_requests}: {interp.action}:{interp.scope} "
             f"(delegation={raw_delegation!r})")

    action_stats = _facet_stats([r["parsed_action"] for r in rows])
    scope_stats = _facet_stats([r["parsed_scope"] for r in rows])
    return {"task_id": task.name, "rows": rows,
           "action_agreement": action_stats["agreement"], "action_entropy": action_stats["entropy"],
           "action_counts": action_stats["counts"],
           "scope_agreement": scope_stats["agreement"], "scope_entropy": scope_stats["entropy"],
           "scope_counts": scope_stats["counts"]}


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
                   help="independent PrincipalAgent.delegate() calls PER TASK")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    api_key = _read_api_key(args.api_key_file)
    os.environ["OPENAI_API_KEY"] = api_key
    del api_key
    try:
        tasks = [_find_task(name) for name in AUDIT_TASK_NAMES]
        total = len(tasks) * args.requests
        print(f"=== Phase 2C-P4 -- Principal Delegation Stability Audit -- {len(tasks)} tasks "
             f"x ({args.requests} independent delegate() + 1 canonicalizing propose() each) "
             f"= {total} delegate() calls + {total} propose() calls ===")
        print(f"model={args.model}")

        results = []
        with open(args.output, "w", encoding="utf-8") as out:
            for task in tasks:
                print(f"\n-- {task.name} --")
                result = audit_one_task(task, model=args.model, n_requests=args.requests)
                results.append(result)
                for row in result["rows"]:
                    out.write(json.dumps(row) + "\n")
                print(f"  action: agreement={result['action_agreement']:.2f} "
                     f"entropy={result['action_entropy']:.3f} counts={result['action_counts']}")
                print(f"  scope:  agreement={result['scope_agreement']:.2f} "
                     f"entropy={result['scope_entropy']:.3f} counts={result['scope_counts']}")

        print(f"\nSaved {sum(len(r['rows']) for r in results)} rows to {args.output}")
        return 0
    finally:
        os.environ.pop("OPENAI_API_KEY", None)


if __name__ == "__main__":
    raise SystemExit(main())
