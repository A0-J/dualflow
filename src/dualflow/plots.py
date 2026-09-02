"""
실험 결과를 그림으로 저장한다.

    dualflow-plots                        # figures/ 에 PNG 5장
    dualflow-plots careless --trials 50   # 논문용 — 밴드가 좁아진다
    python -m dualflow.plots --outdir 어디에 --dpi 300

라벨은 영문이다. matplotlib 기본 폰트에 한글 글리프가 없어 한글로 쓰면 네모로
깨지기 때문이고, 논문 그림도 어차피 영문이라 그대로 쓸 수 있다.
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .bench import adversarial_tasks, build_judge, build_tasks
from .framework import (
    Config, DelegationVerifier, ExperienceStore, evaluate, warmup_then_attack,
)

PALETTE = {"unsafe": "#c0392b", "benign": "#27ae60", "cost": "#2980b9",
           "slow": "#e67e22", "and": "#8e44ad", "cons": "#16a085"}


def _configs():
    return [Config(name="SAGE-Agent\n(Eq.2+Def.4)", mode="sage",
                   use_authority=False, use_matching=False),
            Config(name="SAGE\n+ Joint", mode="sage"),
            Config(name="Fast\n+ Joint", mode="fast"),
            Config(name="Slow\n+ Joint", mode="slow"),
            Config(name="AND\n(proposed)", mode="and")]


def _pilot_panel(ax, tasks, title):
    rows = [evaluate(c, tasks, judge=build_judge()) for c in _configs()]
    x = range(len(rows))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [r["unsafe_rate"] * 100 for r in rows], w,
           label="Unsafe execution", color=PALETTE["unsafe"])
    ax.bar([i + w / 2 for i in x], [r["benign_completion"] * 100 for r in rows], w,
           label="Benign completion", color=PALETTE["benign"])
    for i, r in enumerate(rows):
        ax.text(i - w / 2, r["unsafe_rate"] * 100 + 2,
                f"{r['unsafe_rate']*100:.0f}", ha="center", fontsize=8)
        ax.text(i + w / 2, r["benign_completion"] * 100 + 2,
                f"{r['benign_completion']*100:.0f}", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([c.name for c in _configs()], fontsize=8)
    ax.set_ylabel("%")
    ax.set_ylim(0, 124)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    return rows


def fig_pilot(outdir, dpi, **_):
    """Fig.1 — 정상 조건에서의 Fast/Slow/AND 파일럿."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    _pilot_panel(ax, build_tasks(), "Normal condition")
    fig.tight_layout()
    fig.savefig(outdir / "fig1_pilot_normal.png", dpi=dpi)
    plt.close(fig)


def fig_attack(outdir, dpi, **_):
    """Fig.2 — belief 조작 공격 하에서의 동일 비교."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    _pilot_panel(axes[0], build_tasks(), "Normal condition")
    _pilot_panel(axes[1], adversarial_tasks(), "Under belief manipulation")
    fig.tight_layout()
    fig.savefig(outdir / "fig2_attack.png", dpi=dpi)
    plt.close(fig)


def fig_theta(outdir, dpi, **_):
    """Fig.3 — θ 스윕. 안전성은 평평하고 유용성만 움직인다."""
    tasks = build_tasks()
    thetas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    rows = [evaluate(Config(theta=t), tasks, judge=build_judge()) for t in thetas]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(thetas, [r["benign_completion"] * 100 for r in rows], "o-",
            color=PALETTE["benign"], label="Benign completion")
    ax.plot(thetas, [r["unsafe_rate"] * 100 for r in rows], "s-",
            color=PALETTE["unsafe"], label="Unsafe execution")
    ax.set_xlabel("θ  (entropy threshold, bits)")
    ax.set_ylabel("%")
    ax.set_ylim(-5, 110)
    ax.grid(alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(thetas, [r["avg_cost"] for r in rows], "^--", color=PALETTE["cost"],
             label="Avg cost")
    ax2.set_ylabel("cost per delegation")

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="center right")
    ax.set_title("Safety is flat in θ; only utility and cost move", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "fig3_theta_sweep.png", dpi=dpi)
    plt.close(fig)


def fig_experience(outdir, dpi, **_):
    """Fig.4 — 경험 누적에 따른 개입 감소."""
    tasks = build_tasks()
    task = next(t for t in tasks if t.name == "vague_clarifiable")
    store = ExperienceStore()
    v = DelegationVerifier(Config(), store, build_judge(tasks))
    eps, q, h, sc = [], [], [], []
    for i in range(1, 9):
        r = v.run(task)
        eps.append(i); q.append(r.n_questions); h.append(r.h_initial)
        sc.append(store.score(task.key))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.step(eps, q, where="mid", color=PALETTE["cost"], label="Clarification rounds")
    ax.plot(eps, h, "o-", color=PALETTE["unsafe"], label="Initial entropy H (bits)")
    ax.set_xlabel("episode (same delegation type repeated)")
    ax.set_ylabel("rounds  /  bits")
    ax.grid(alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(eps, sc, "^--", color=PALETTE["benign"], label="Experience score")
    ax2.axhline(0.8, ls=":", color="grey", label="_nolegend_")
    ax2.text(1.1, 0.82, "σ = 0.8", fontsize=8, color="grey")
    ax2.set_ylabel("experience score")
    ax2.set_ylim(0, 1.05)

    lines = [l for l in ax.get_lines() + ax2.get_lines()
             if not l.get_label().startswith("_")]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="lower left")
    ax.set_title("Accumulated experience removes the clarification loop", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "fig4_experience.png", dpi=dpi)
    plt.close(fig)


def fig_careless(outdir, dpi, trials=10, warmup=5, **_):
    """Fig.5 — A 의 검토가 완벽하지 않을 때 (실험 ③).

    정상 운영으로 경험을 쌓은 뒤 같은 유형에 공격이 들어온다.
    왼쪽은 경험 게이트가 발동하는 기본 σ, 오른쪽은 게이트가 닫힌 엄격한 σ.

    확률적 실험이므로 시행 간 표준오차를 음영 밴드로 함께 그린다. 논문용으로는
    --trials 50 이상을 권한다 (밴드가 눈에 띄게 좁아진다).
    """
    normal, adv = build_tasks(), adversarial_tasks()
    cs = [0.0, 0.25, 0.5, 0.75, 1.0]
    judge = build_judge()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, sigma, title in ((axes[0], 0.8, "σ = 0.80  (experience gate active)"),
                             (axes[1], 0.95, "σ = 0.95  (gate closed)")):
        series = {"Slow only": ([], []), "AND": ([], []),
                  "AND + consistency": ([], [])}
        for c in cs:
            cfgs = [Config(mode="slow", carelessness=c, sigma=sigma),
                    Config(mode="and", carelessness=c, sigma=sigma),
                    Config(mode="and", carelessness=c, sigma=sigma,
                           use_consistency_check=True, consistency_sigma=0.6)]
            for key, cfg in zip(series, cfgs):
                m = warmup_then_attack(cfg, normal, adv, judge=judge,
                                       warmup=warmup, trials=trials)
                series[key][0].append(m["unsafe_rate"] * 100)
                series[key][1].append(m["stderr"] * 100)
        for (key, (ys, es)), color, marker in zip(
                series.items(),
                (PALETTE["slow"], PALETTE["and"], PALETTE["cons"]), ("o", "s", "^")):
            lo = [y - e for y, e in zip(ys, es)]
            hi = [y + e for y, e in zip(ys, es)]
            ax.fill_between(cs, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(cs, ys, marker=marker, color=color, label=key)
        ax.set_xlabel("carelessness of Agent A's review")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("Unsafe execution rate (%)")
    fig.suptitle(f"Experiment ③ — when the reviewer is not perfect "
                 f"(mean ± s.e., {trials} trials)", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "fig5_careless_reviewer.png", dpi=dpi)
    plt.close(fig)


FIGURES = {"pilot": fig_pilot, "attack": fig_attack, "theta": fig_theta,
           "experience": fig_experience, "careless": fig_careless}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="dualflow 실험 그림 생성")
    ap.add_argument("figures", nargs="*", choices=list(FIGURES) or None,
                    help="생략하면 전부 생성")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--trials", type=int, default=10,
                    help="fig5 의 시행 횟수. 논문용은 50 이상 권장")
    ap.add_argument("--warmup", type=int, default=5,
                    help="fig5 에서 공격 전에 정상 운영을 반복할 횟수")
    args = ap.parse_args(argv)

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for key in (args.figures or list(FIGURES)):
        FIGURES[key](outdir, args.dpi, trials=args.trials,
                     warmup=args.warmup)
        print(f"  생성: {key}")
    print(f"\n{outdir.resolve()} 에 저장했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
