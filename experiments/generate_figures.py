"""Generate the paper's summary figures from already-collected, committed
raw data (`experiments/data/*.jsonl`) -- 0 API calls. Reuses the exact
same analysis functions already validated in `phase3d_analysis.py`/
`phase3d_tradeoff_analysis.py`/`intent_anchor_arms_comparison.py`
(docs/experiments/agent_connected_eval.md §29) -- no numbers are
recomputed with new logic, only plotted.

    python experiments/generate_figures.py

Saves PNGs to docs/experiments/figures/.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Figure 1 -- Phase 3C headline: Arm A/B/C safety + utility
# ---------------------------------------------------------------------------
def figure_1_phase3c_headline():
    rows = _load_jsonl(_DATA_DIR / "phase3c_full.jsonl")
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

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    arms = ["A\n(Current)", "B\n(Repeated-\nRestate)", "C\n(Grounded)"]
    colors = [COLORS["A"], COLORS["B"], COLORS["C"]]

    axes[0].bar(arms, [unsafe[a] for a in "abc"], color=colors)
    axes[0].set_title(f"Unsafe execution (n={n_scored})")
    axes[0].set_ylabel("count")
    for i, a in enumerate("abc"):
        axes[0].text(i, unsafe[a] + 0.1, str(unsafe[a]), ha="center")

    axes[1].bar(arms, [false_reject[a] for a in "abc"], color=colors)
    axes[1].set_title(f"False rejection (n={n_correct_auth})")
    for i, a in enumerate("abc"):
        axes[1].text(i, false_reject[a] + 0.1, str(false_reject[a]), ha="center")

    fig.suptitle("Phase 3C: Current vs. Repeated-Restate vs. Grounded (frozen result)")
    fig.tight_layout()
    out = _FIGURES_DIR / "fig1_phase3c_headline.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


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

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, task, metric_label in zip(
            axes, ("confident_semantic_misread", "condition_violation"),
            ("Unsafe execution", "False rejection")):
        unsafe_b, unsafe_c, fr_b, fr_c = _sampling_curve(rows, task)
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
    for th in THRESHOLDS:
        evals_csm = [evaluate_episode(r, n=20, entropy_threshold=th) for r in csm]
        unsafe = sum(e.arm_c_decision == "EXECUTE" and e.delegate_wrong for e in evals_csm)

        evals_cv = [evaluate_episode(r, n=20, entropy_threshold=th) for r in cv]
        correct_auth = [e for e in evals_cv if not e.delegate_wrong and e.ideal_authorized]
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
    ax.set_xlabel("Unsafe execution (confident_semantic_misread, n=40 pooled)")
    ax.set_ylabel("False rejection (condition_violation, n=10 pooled)")
    ax.set_title("Arm C safety-utility frontier across entropy_threshold (n=20 fixed)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = _FIGURES_DIR / "fig3_threshold_frontier.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Figure 4 -- cross-model bifurcation
# ---------------------------------------------------------------------------
def figure_4_cross_model_bifurcation():
    gpt4o = _load_jsonl(_DATA_DIR / "phase3d_gpt4omini_full.jsonl")
    gpt41 = _load_jsonl(_DATA_DIR / "phase3d_gpt41_full.jsonl")

    def unsafe_curve(rows):
        trows = [r for r in rows if r["task_id"] == "confident_semantic_misread"]
        return [sum(evaluate_episode(r, n=n, entropy_threshold=0.8).arm_c_decision == "EXECUTE"
                    and evaluate_episode(r, n=n, entropy_threshold=0.8).delegate_wrong
                    for r in trows) for n in NS]

    y_gpt4o = unsafe_curve(gpt4o)
    y_gpt41 = unsafe_curve(gpt41)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(NS, y_gpt4o, "o-", color=COLORS["B"], label="GPT-4o-mini")
    ax.plot(NS, y_gpt41, "s-", color="#C44E52", label="GPT-4.1")
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


def main() -> int:
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_1_phase3c_headline()
    figure_2_sampling_ablation()
    figure_3_threshold_frontier()
    figure_4_cross_model_bifurcation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
