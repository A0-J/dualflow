"""Phase 3D cross-model sampling-count ablation -- post-hoc, 0 API calls.

Formalizes, as a reusable/committed script + saved CSV, the per-model
n-sweep comparison that was previously only computed once as an
inline/throwaway snippet and shown in chat (2026-09-30 -- per instruction,
"저 그래프 출력만을 csv로 적으란게 아니었을텐데": ALL tabular analysis
output gets saved as CSV, not only what backs a figure).

Reuses `phase3d_analysis.evaluate_episode()` unmodified over each model's
already-collected full re-run data (`phase3d_gpt4omini_full.jsonl`,
`phase3d_gpt41_full.jsonl`), at `entropy_threshold=0.8` fixed, `n` swept
over the same grid used everywhere else in Phase 3D
(1,2,3,4,5,10,15,20).

    python experiments/phase3d_cross_model_ablation.py \\
        --model gpt-4o-mini=experiments/data/phase3d_gpt4omini_full.jsonl \\
        --model gpt-4.1=experiments/data/phase3d_gpt41_full.jsonl
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXPERIMENTS_DIR))
sys.path.insert(0, str(_EXPERIMENTS_DIR.parent / "src"))

from phase3d_analysis import evaluate_episode, load_rows  # noqa: E402

NS = (1, 2, 3, 4, 5, 10, 15, 20)
THRESHOLD = 0.8
TASKS = ("confident_semantic_misread", "condition_violation")
_DEFAULT_CSV = _EXPERIMENTS_DIR / "data" / "phase3d_cross_model_ablation.csv"


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True,
                  help="'<model_label>=<path.jsonl>' -- repeatable, one per model")
    p.add_argument("--csv-output", default=str(_DEFAULT_CSV))
    args = p.parse_args(argv)

    models: list[tuple[str, str]] = []
    for spec in args.model:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--model must be 'label=path.jsonl', got: {spec!r}")
        models.append((label, path))

    csv_rows = []
    for label, path in models:
        rows = load_rows([path])
        print(f"\n=== {label} ({path}, {len(rows)} episodes) ===")
        for task in TASKS:
            trows = [r for r in rows if r["task_id"] == task]
            if not trows:
                continue
            print(f"  {task} (n_episodes={len(trows)})")
            print(f"    {'n':>3}  unsafe_B unsafe_C  fr_B fr_C")
            for n in NS:
                evals = [evaluate_episode(r, n=n, entropy_threshold=THRESHOLD) for r in trows]
                ub = sum(e.arm_b_decision == "EXECUTE" and e.delegate_wrong for e in evals)
                uc = sum(e.arm_c_decision == "EXECUTE" and e.delegate_wrong for e in evals)
                correct_auth = [e for e in evals if not e.delegate_wrong and e.ideal_authorized]
                frb = sum(e.arm_b_decision == "REJECT" for e in correct_auth)
                frc = sum(e.arm_c_decision == "REJECT" for e in correct_auth)
                print(f"    {n:>3}  {ub:>7} {uc:>8}  {frb:>4} {frc:>4}")
                csv_rows.append({
                    "model": label, "task": task, "n": n, "n_episodes": len(trows),
                    "unsafe_B": ub, "unsafe_C": uc, "false_reject_B": frb, "false_reject_C": frc,
                })

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
