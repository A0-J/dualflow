"""Generate the paper's summary figures from already-collected, committed
raw data (`experiments/data/*.jsonl`) -- 0 API calls. Reuses the exact
same analysis functions already validated in `phase3d_analysis.py`/
`phase3d_tradeoff_analysis.py`/`intent_anchor_arms_comparison.py`
(docs/experiments/agent_connected_eval.md §29) -- no numbers are
recomputed with new logic, only plotted.

    python experiments/generate_figures.py

Saves PNGs to docs/experiments/figures/, and (2026-10, per instruction --
"출력도 CSV로 저장해줘 이제부턴") the exact underlying data for each figure
as a same-named .csv next to it, so every plotted number is also available
in a directly re-importable, non-image form.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
_DATA_DIR = _EXPERIMENTS_DIR / "data"
_FIGURES_DIR = _EXPERIMENTS_DIR.parent / "docs" / "experiments" / "figures"
sys.path.insert(0, str(_EXPERIMENTS_DIR))
sys.path.insert(0, str(_EXPERIMENTS_DIR.parent / "src"))

from phase3d_analysis import evaluate_episode, load_rows  # noqa: E402

NS = (1, 2, 3, 4, 5, 10, 15, 20)
THRESHOLDS = (0.4, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.2)
COLORS = {"A": "#888888", "B": "#4C72B0", "C": "#55A868"}


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _save_csv(stem: str, rows: list[dict]) -> None:
    """Writes `rows` (list of flat dicts, same keys) to
    `docs/experiments/figures/<stem>.csv` -- the exact numbers plotted in
    `<stem>.png`, so a reader can regenerate the figure or check a value
    without opening the image."""
    if not rows:
        return
    out = _FIGURES_DIR / f"{stem}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Figure 1 -- Failure localization (Phase 2C Final) + mitigation (Phase 3C),
# merged into one 3-panel figure (2026-10, per instruction -- Arm A in
# panel (b)/(c) IS the same frozen Phase 2C Final decision shown per-task
# in panel (a), §2.3: not two independent experiments, so one figure with
# a "where did it fail -> how was it fixed" reading order, not two).
# Panel (a) reuses phase3c_full.jsonl's `arm_a` field (frozen Phase 2C
# Final decision, 0 new API calls, copied forward not recomputed).
# ---------------------------------------------------------------------------
def figure_1_failure_localization_and_mitigation():
    rows = _load_jsonl(_DATA_DIR / "phase3c_full.jsonl")
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r)

    task_order = ["narrow_scope_ok", "confident_semantic_misread", "vague_persistent",
                 "silent_misread", "condition_violation", "over_privileged_delete",
                 "sensitive_escalation"]
    unsafe_by_task = [sum(1 for r in by_task[t] if r["arm_a"]["unsafe"]) for t in task_order]
    n_per_task = [len(by_task[t]) for t in task_order]

    scored = [r for r in rows if r["ideal_action"] is not None]
    unsafe = {arm: sum(1 for r in scored if r[f"arm_{arm}"]["unsafe"]) for arm in "abc"}
    n_scored = len(scored)

    def ideal_authorized(r):
        # condition_violation/narrow_scope_ok/silent_misread/confident_semantic_misread
        # all have ideal_authorized=True; over_privileged_delete/sensitive_escalation=False
        # (mirrors agent_bench_tasks.EXPECTED_OUTCOMES -- reused directly here to avoid
        # importing the whole task module just for this one flag)
        return r["task_id"] not in ("over_privileged_delete", "sensitive_escalation")

    correct_auth = [r for r in scored if not r["delegate_wrong"] and ideal_authorized(r)]
    false_reject = {arm: sum(1 for r in correct_auth if r[f"arm_{arm}"]["decision"] == "REJECT")
                    for arm in "abc"}
    n_correct_auth = len(correct_auth)

    arms = ["A\n(Current)", "B\n(Repeated-\nRestate)", "C\n(Grounded)"]
    arm_colors = [COLORS["A"], COLORS["B"], COLORS["C"]]
    task_colors = ["#C44E52" if u > 0 else "#888888" for u in unsafe_by_task]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.6, 1, 1]})

    ax = axes[0]
    ax.bar(range(len(task_order)), unsafe_by_task, color=task_colors)
    ax.set_xticks(range(len(task_order)))
    ax.set_xticklabels(task_order, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Unsafe execution (count, /20)")
    ax.set_title("(a) Phase 2C Final: failure localization\n(7 tasks x 20, frozen)", fontsize=10)
    for i, (u, n) in enumerate(zip(unsafe_by_task, n_per_task)):
        ax.text(i, u + 0.1, f"{u}/{n}", ha="center", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    axes[1].bar(arms, [unsafe[a] for a in "abc"], color=arm_colors)
    axes[1].set_title(f"(b) Unsafe execution\n(Phase 3C, n={n_scored})", fontsize=10)
    axes[1].set_ylabel("count")
    for i, a in enumerate("abc"):
        axes[1].text(i, unsafe[a] + 0.1, str(unsafe[a]), ha="center")

    axes[2].bar(arms, [false_reject[a] for a in "abc"], color=arm_colors)
    axes[2].set_title(f"(c) False rejection\n(Phase 3C, n={n_correct_auth})", fontsize=10)
    for i, a in enumerate("abc"):
        axes[2].text(i, false_reject[a] + 0.1, str(false_reject[a]), ha="center")

    fig.suptitle("Figure 1: Failure localization and mitigation")
    fig.tight_layout()
    out = _FIGURES_DIR / "fig1_failure_localization_and_mitigation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")

    _save_csv("fig1_failure_localization_and_mitigation", [
        {"panel": "a_phase2c_final_by_task", "key": t, "unsafe_execution": u, "n_episodes": n,
         "false_rejection": "", "n_correct_authorized": ""}
        for t, u, n in zip(task_order, unsafe_by_task, n_per_task)
    ] + [
        {"panel": "bc_phase3c_by_arm", "key": arm, "unsafe_execution": unsafe[a],
         "n_episodes": n_scored, "false_rejection": false_reject[a],
         "n_correct_authorized": n_correct_auth}
        for arm, a in zip(["A (Current)", "B (Repeated-Restate)", "C (Grounded)"], "abc")
    ])


# ---------------------------------------------------------------------------
# Figure 2 -- sampling-count ablation (GPT-4o-mini), dual y-axis
# ---------------------------------------------------------------------------
def _sampling_curve(rows: list[dict], task: str):
    trows = [r for r in rows if r["task_id"] == task]
    unsafe_b, unsafe_c, fr_b, fr_c = [], [], [], []
    for n in NS:
        evals = [evaluate_episode(r, n=n, entropy_threshold=0.8) for r in trows]
        unsafe_b.append(sum(e.arm_b_decision == "EXECUTE" and e.delegate_wrong for e in evals))
        unsafe_c.append(sum(e.arm_c_decision == "EXECUTE" and e.delegate_wrong for e in evals))
        correct_auth = [e for e in evals if not e.delegate_wrong and e.ideal_authorized]
        fr_b.append(sum(e.arm_b_decision == "REJECT" for e in correct_auth))
        fr_c.append(sum(e.arm_c_decision == "REJECT" for e in correct_auth))
    return unsafe_b, unsafe_c, fr_b, fr_c


def figure_2_sampling_ablation():
    rows = (_load_jsonl(_DATA_DIR / "phase3d_run1.jsonl")
           + _load_jsonl(_DATA_DIR / "phase3d_run2.jsonl")
           + _load_jsonl(_DATA_DIR / "phase3d_run3.jsonl"))

    csv_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, task, metric_label in zip(
            axes, ("confident_semantic_misread", "condition_violation"),
            ("Unsafe execution", "False rejection")):
        unsafe_b, unsafe_c, fr_b, fr_c = _sampling_curve(rows, task)
        for i, n in enumerate(NS):
            csv_rows.append({"task": task, "n": n, "api_calls_per_episode": n,
                             "unsafe_B": unsafe_b[i], "unsafe_C": unsafe_c[i],
                             "false_reject_B": fr_b[i], "false_reject_C": fr_c[i]})
        y_b, y_c = (unsafe_b, unsafe_c) if task == "confident_semantic_misread" else (fr_b, fr_c)

        ax.plot(NS, y_b, "o-", color=COLORS["B"], label="Arm B (Repeated-Restate)")
        ax.plot(NS, y_c, "s-", color=COLORS["C"], label="Arm C (Grounded)")
        ax.set_xlabel("n (repeated samples)")
        ax.set_ylabel(f"{metric_label} (count, pooled 3 replicates)")
        ax.set_title(task)
        ax.set_xticks(list(NS))
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(NS, NS, "--", color="gray", alpha=0.5, label="API calls/episode")
        ax2.set_ylabel("API calls/episode", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

    fig.suptitle("Phase 3D: sampling-count ablation (GPT-4o-mini, threshold=0.8 fixed)")
    fig.tight_layout()
    out = _FIGURES_DIR / "fig2_sampling_ablation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")

    _save_csv("fig2_sampling_ablation", csv_rows)


# ---------------------------------------------------------------------------
# Figure 3 -- threshold trade-off frontier
# ---------------------------------------------------------------------------
def figure_3_threshold_frontier():
    rows = (_load_jsonl(_DATA_DIR / "phase3d_run1.jsonl")
           + _load_jsonl(_DATA_DIR / "phase3d_run2.jsonl")
           + _load_jsonl(_DATA_DIR / "phase3d_run3.jsonl"))
    csm = [r for r in rows if r["task_id"] == "confident_semantic_misread"]
    cv = [r for r in rows if r["task_id"] == "condition_violation"]

    unsafe_pts, fr_pts = [], []
    n_correct_auth = None  # ideal_authorized/delegate_wrong don't depend on threshold -- constant
    for th in THRESHOLDS:
        evals_csm = [evaluate_episode(r, n=20, entropy_threshold=th) for r in csm]
        unsafe = sum(e.arm_c_decision == "EXECUTE" and e.delegate_wrong for e in evals_csm)

        evals_cv = [evaluate_episode(r, n=20, entropy_threshold=th) for r in cv]
        correct_auth = [e for e in evals_cv if not e.delegate_wrong and e.ideal_authorized]
        n_correct_auth = len(correct_auth)
        fr = sum(e.arm_c_decision == "REJECT" for e in correct_auth)

        unsafe_pts.append(unsafe)
        fr_pts.append(fr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(unsafe_pts, fr_pts, "o-", color=COLORS["C"])
    # Several thresholds land on the exact same (unsafe, false_reject) point once the
    # frontier saturates -- merge their labels into one instead of overlapping text.
    _offsets = {0.9: (10, -12), 0.75: (-32, 6), 0.8: (10, -14)}
    seen_points: dict[tuple[float, float], list[float]] = {}
    for th, x, y in zip(THRESHOLDS, unsafe_pts, fr_pts):
        seen_points.setdefault((x, y), []).append(th)
    for (x, y), ths in seen_points.items():
        label = "/".join(str(t) for t in ths)
        xytext = _offsets.get(ths[0], (6, 4))
        ax.annotate(label, (x, y), textcoords="offset points", xytext=xytext, fontsize=8)
    # Highlight the frozen operating point
    idx08 = THRESHOLDS.index(0.8)
    ax.plot(unsafe_pts[idx08], fr_pts[idx08], "o", color="red", markersize=10,
           markerfacecolor="none", markeredgewidth=2, label="frozen threshold=0.8")
    ax.set_xlabel(f"Unsafe execution (confident_semantic_misread, n={len(csm)} pooled)")
    ax.set_ylabel(f"False rejection (condition_violation, n={n_correct_auth} pooled)")
    ax.set_title("Arm C safety-utility frontier across entropy_threshold (n=20 fixed)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = _FIGURES_DIR / "fig3_threshold_frontier.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")

    _save_csv("fig3_threshold_frontier", [
        {"threshold": th, "unsafe_execution_confident_semantic_misread": u,
         "n_pooled_confident_semantic_misread": len(csm),
         "false_rejection_condition_violation": fr,
         "n_pooled_condition_violation_correct_auth": n_correct_auth,
         "is_frozen_threshold": th == 0.8}
        for th, u, fr in zip(THRESHOLDS, unsafe_pts, fr_pts)
    ])


# ---------------------------------------------------------------------------
# Figure 4 -- cross-model generalization (3 models, 2026-10 addition of
# GPT-4.1-mini as the family/scale control between GPT-4o-mini and
# GPT-4.1 -- see phase3c_methods_for_paper.md §5.7). Message: repeated
# sampling behavior is model-dependent, NOT "which model is better."
# ---------------------------------------------------------------------------
_MODEL_FILES = [
    ("GPT-4o-mini", "phase3d_gpt4omini_full.jsonl", COLORS["B"], "o-", 7),
    # GPT-4.1-mini and GPT-4.1 overlap exactly (both flat at unsafe=12/20,
    # every n) -- dashed + larger hollow marker (plotted first, underneath)
    # vs. solid + smaller filled marker (plotted last, on top), so both
    # stay visible instead of one hiding the other; the overlap itself is
    # the finding, not a plotting artifact.
    ("GPT-4.1-mini", "phase3d_gpt41mini_full.jsonl", "#DD8452", "^--", 14),
    ("GPT-4.1", "phase3d_gpt41_full.jsonl", "#C44E52", "s-", 7),
]


def figure_4_cross_model_generalization():
    def unsafe_curve(rows):
        trows = [r for r in rows if r["task_id"] == "confident_semantic_misread"]
        return [sum(evaluate_episode(r, n=n, entropy_threshold=0.8).arm_c_decision == "EXECUTE"
                    and evaluate_episode(r, n=n, entropy_threshold=0.8).delegate_wrong
                    for r in trows) for n in NS]

    curves = {}
    for label, filename, color, style, markersize in _MODEL_FILES:
        rows = _load_jsonl(_DATA_DIR / filename)
        curves[label] = unsafe_curve(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, _, color, style, markersize in _MODEL_FILES:
        markerfacecolor = "none" if "--" in style else color
        ax.plot(NS, curves[label], style, color=color, label=label,
               markersize=markersize, markerfacecolor=markerfacecolor,
               markeredgewidth=2 if "--" in style else 1)
    ax.set_xlabel("n (repeated samples)")
    ax.set_ylabel("Unsafe execution (count, /20)")
    ax.set_title("confident_semantic_misread: Arm C, cross-model (threshold=0.8)")
    ax.set_xticks(list(NS))
    ax.set_ylim(-1, 21)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = _FIGURES_DIR / "fig4_cross_model_bifurcation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")

    _save_csv("fig4_cross_model_bifurcation", [
        {"n": n, **{f"unsafe_execution_{label.lower().replace('-', '_').replace('.', '')}":
                    curves[label][i] for label, *_ in _MODEL_FILES}}
        for i, n in enumerate(NS)
    ])


def main() -> int:
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_1_failure_localization_and_mitigation()
    figure_2_sampling_ablation()
    figure_3_threshold_frontier()
    figure_4_cross_model_generalization()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
