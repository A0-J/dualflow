**English** | [한국어](README_KOR.md)

# DualFlow

**Semantic and Authority Verification for Safe Agent-to-Agent Delegation**

DualFlow is a verification framework for agent-to-agent delegation. It separates
two fundamentally different questions:

1. **Semantic verification** — did the delegate understand what the principal
   requested?
2. **Authority verification** — is the proposed execution within the delegated
   authority?

DualFlow combines both signals before execution and, when the delegated scope
cannot be safely resolved from policy alone, invokes a bounded **Authority
Feedback Loop** with the principal. The core safety mechanism does not depend
on historical experience — **verified experience is used only as an
optimization layer** to reduce repeated principal feedback.

The implementation is deterministic (no LLM calls required to run the
experiments below); a real model plugs in behind three interfaces — see
[Current Scope and Limitations](#current-scope-and-limitations).

## Why DualFlow?

A delegated action can fail in two different ways.

**Semantic mismatch.** The delegate may misread what was asked.

```
Principal:              "Read last month's report."
Delegate interpretation: /reports/2025-08/
```

The delegate can be highly confident in this interpretation, so low semantic
uncertainty alone does not imply correctness.

**Authority mismatch.** The delegate may understand the request correctly and
still propose something outside what was actually delegated.

```
Delegated scope:  read /reports/2026-08/*
Proposed scope:   read /reports/*
```

The action can be semantically clear but still exceed the delegated authority.

Therefore:

**Low uncertainty does not imply valid delegation, and a permitted action is
not necessarily the intended action.** DualFlow verifies both, explicitly and
separately, before execution.

## Architecture

DualFlow consists of a **Core Safety Mechanism**, which must be safe on its
own, and an optional **Optimization Layer**, which only reduces the cost of
using it.

```
Core Safety Mechanism ── safety holds even with zero prior experience
  1. Semantic Flow            resolves ambiguity: entropy, information gain, clarification
  2. Authority Flow           validates capability / resource / scope / condition
  3. Authority Feedback Loop  APPROVE / CORRECT / RESTRICT / REJECT, bounded rounds
  4. Joint Verification       final check against the resolved intent and confirmed authority

Optimization Layer ── reduces principal-interaction cost, changes nothing about safety
  Verified Authority Store → Adaptive Feedback Gate → principal called only when needed
```

Turning the Optimization Layer off entirely (`use_verified_experience=False`)
yields the exact same safety guarantees as the Core alone — it just calls the
principal every time.

| Component | Implementation | Related work |
|---|---|---|
| Authority budget, delegation chain | `capability.Budget`, `check_authority` | ChainCaps §3.2–3.3, Eq.(2) |
| Action space, interpretation distribution | `semantic.Interpretation`, `build_belief` | SAGE-Agent Def.2–3 |
| Entropy / clarification | `semantic.entropy`, `information_gain` | SAGE-Agent Def.4, reworked as information gain |
| LLM fallback | `llm.LLMJudge` | SAGE-Agent's always-on judge, demoted to a fallback |
| Joint matching | `rule_engine.match_intent`, `sim_path` | SAGE-Bench Eq.(6) |
| Authority Feedback Loop | `authority_feedback.run_feedback` | new |
| Verified Authority Store / Adaptive Gate | `authority_feedback.VerifiedAuthorityStore` | new |
| Orchestration | `framework.DelegationVerifier.run` | — |

## How It Works

**Semantic Verification.** The delegate produces a structured interpretation
`(action, resource, scope, condition)`. DualFlow estimates uncertainty over
candidate interpretations with Shannon entropy and, when uncertainty is high,
selects clarification questions by information gain.

**Authority Verification.** Every proposed execution is checked against the
current delegation budget. Failures are classified, not just flagged:

- `no_grant` — the action/resource itself was never delegated → hard reject
- `condition_missing` — scope is fine but a required condition (e.g.
  anonymization) is absent → hard reject, not something the delegate can fill in
- `scope_exceeded` — action/resource/condition are fine, only the range is too
  wide → **eligible for negotiation**
- otherwise → continue

**Authority Feedback.** For a `scope_exceeded` proposal, the principal is
asked to `APPROVE`, `CORRECT`, `RESTRICT`, or `REJECT` a system-computed
suggestion (the widest scope actually covered by the current budget).
Negotiation is bounded, and every revised proposal is revalidated against the
delegation budget before it can be accepted — a careless `APPROVE` of an
out-of-budget proposal is still rejected by that revalidation, not trusted.

**Joint Verification.** Execution is allowed only after the resolved semantic
interpretation and the confirmed authority agree with the original intent —
path similarity **and** terminal decision match (EXECUTE / ESCALATE / REJECT).

**Adaptive Feedback.** Repeated principal feedback is expensive. DualFlow
stores only *verified* authority states — confirmed by the principal **and**
revalidated by the system **and** followed by successful execution — and
reuses them under a simple, inspectable rule:

```
n_confirmed >= 3  and  agreement_ratio >= 0.8   =>  reuse without asking
```

Reuse is never a blind replay of history. Every reuse is recomputed as an
intersection with the *current* budget and re-validated:

```
C_adaptive = C_experience ∩ C_current_budget
C_adaptive ⊆ C_current_budget ⊆ C_A
```

**Verified experience can narrow the current authority; it can never create
or widen it.** When a newly confirmed state differs from the stored one
(the delegated scope genuinely changed), the stale history is invalidated and
rebuilt from that point — verified history is reusable, but not permanent.

## Key Results

### 1. Both verification axes are necessary

9-task pilot (`DelegationBench-mini`), normal conditions:

| Method | Unsafe ↓ | Benign ↑ |
|---|---:|---:|
| Authority only | 22.2% | 80.0% |
| Semantic only | 22.2% | 80.0% |
| **Full Core** | **0.0%** | 80.0% |

Each axis alone lets a different failure class through — dropping Authority
lets deletion/exfiltration/condition violations through; dropping semantic
matching lets silent misreads through. Both land at the same 22.2% unsafe by
different routes; only the combination reaches 0%. Full experimental detail,
including the always-on-LLM and θ-only baselines, is in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

### 2. Authority Feedback converts safe rejection into safe completion

Scope-negotiation pilot (`scope_negotiation_tasks()`), isolated from the
9-task benchmark above so its denominator never shifts the numbers there:

| Method | Unsafe ↓ | Benign ↑ | Feedback rate ↓ |
|---|---:|---:|---:|
| No feedback | 0.0% | 0.0% | 0.0% |
| **Feedback (proposed)** | 0.0% | **100.0%** | 66.7% |

![Authority Feedback Loop — normal and under-attack conditions](figures/fig7_authority_feedback.png)

Without feedback, every negotiable scope violation is safely rejected but
never completed. With feedback, the same cases complete safely. The result is
**identical under semantic-proposal manipulation** — feedback reads the
principal's real answer, not the delegate's self-reported semantic
uncertainty — and stays at 0% unsafe even as principal carelessness rises to
1.0, because non-amplification revalidates every round regardless of what the
principal said. This holds only as long as the authority budget and the
Principal feedback channel themselves are trusted; scope negotiation is not
triggered at all when the delegate picks a different-but-in-budget resource
(see [Current Scope and Limitations](#current-scope-and-limitations)). Detail
(including the carelessness sweep) is in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

### 3. Adaptive reuse survives repetition, drift, and manipulation

A single 9-round sequence run through one `VerifiedAuthorityStore`:

![Adaptive Verification — 9-round timeline](figures/fig8_adaptive_verification.png)

- **Stable repetition** (rounds 0–4): after 3 consistent principal
  confirmations, DualFlow reuses verified authority without another principal
  call — the feedback rate drops from 100% to 0% for the same delegation type.
- **Legitimate drift** (round 5): when the delegated scope genuinely changes,
  the stale history no longer intersects the new budget, feedback is
  immediately reactivated, and the old history is reset rather than left to
  outvote the new one.
- **Semantic-proposal manipulation** (round 8): even with a forged,
  fully-confident (H=0) proposal, adaptive reuse resolves to the *correct*
  scope without asking — because it never looks at the delegate's candidate,
  only at the principal-confirmed history intersected with the current
  budget. This holds as long as the current authority budget and
  `VerifiedAuthorityStore` themselves are trusted.

Full round-by-round data is in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Quick Start

```bash
pip install -e ".[dev]"

pytest -q                 # 179 tests
dualflow-demo              # all 11 experiments, as text
dualflow-plots             # 9 figures into figures/
```

### Selected experiments

```bash
python -m dualflow.demo fastslow attack careless authfeedback adaptiveauth
dualflow-plots careless --trials 50   # paper-quality error bars (default 10 is noisy)
```

## Reproducing the Experiments

Every number and figure in this README comes from the scripts above; nothing
is hand-transcribed. [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) walks through
each experiment (parameter sweeps, ablations, attack scenarios) in full, with
the reasoning behind each result. [docs/BASELINES.md](docs/BASELINES.md)
documents the SAGE-Agent baseline reproduction and six specific issues found
in its formulas along the way. [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md)
covers implementation decisions that didn't make the cut for the main results
— including one approach (matching against ground truth directly) that looked
like a shortcut and turned out to be an evaluation oracle instead.

## Repository Structure

| File | Purpose |
|---|---|
| `capability.py` | Authority model — privilege ordering, budget intersection, delegation-chain attenuation, failure classification |
| `semantic.py` | Semantic Flow — interpretation candidates, entropy, clarification, experience score |
| `authority_feedback.py` | Authority Feedback Loop and Verified Authority Store (adaptive reuse) |
| `rule_engine.py` | Joint Verification — delegation SOP graph, path similarity |
| `framework.py` | End-to-end orchestration, ablation switches, evaluation metrics |
| `sage_baseline.py` | SAGE-Agent baseline — direct reproduction of the published formulas |
| `bench.py` | Pilot scenarios: the 9-task benchmark, its attack variants, scope-negotiation and sequential mini-sets |
| `llm.py` | LLM interface — scripted for experiments, adapter skeleton for a real model |
| `demo.py` / `plots.py` | Text and figure output for every experiment |
| `tests/` | 179 tests — safety invariants, entropy properties, termination, ablations, attack scenarios |

## Current Scope and Limitations

This repository currently validates the mechanism using deterministic pilot
scenarios. It does not yet establish external validity with production LLM
agents. In particular:

- Semantic candidate generation is scripted, not model-generated.
- The principal is simulated (with configurable carelessness/overcaution),
  not a real human or agent endpoint.
- The pilot benchmark is a small number of controlled tasks (9 for the core
  benchmark, plus small dedicated sets for scope negotiation and adaptive
  verification), not a large or naturalistic distribution.
- Verified-authority reuse assumes the current authority budget and the
  `VerifiedAuthorityStore` itself are trusted; manipulating either is a
  different threat model than the one tested.
- An in-scope but unintended resource choice (the delegate picks a *different*
  resource that still happens to be within the granted budget) is not caught
  by Authority Feedback, since no `scope_exceeded` is ever raised — this
  remains a separate intent-confirmation problem.

Swapping in a real model touches exactly three interfaces (candidate
generation, principal responses, LLM fallback) and is documented in
[docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md).

## Roadmap

- **External validation** — real LLM-generated proposals, a larger benchmark,
  and baseline comparisons beyond the current pilot.
- **Per-agent principal reliability** — carelessness that varies by agent, and
  a review budget, turn "who gets Slow review" into an optimization problem.
  `warmup_then_attack` already provides the scaffolding.
- Splitting authority failures into "no permission at all" vs. "permission too
  narrow" is done; the next open question is *how much to trust* the principal
  once escalation happens, not *how often* to escalate.
- σ / k / λ sweeps, analogous to the existing θ sweep.

Detail on each item is in [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md).

## References

- ChainCaps: Composition-Safe Tool-Using Agents via Monotonic Capability Attenuation. arXiv 2605.26542
- Structured Uncertainty guided Clarification for LLM Agents. arXiv 2511.08798 (Findings of ACL 2026)
- SAGE: A Service Agent Graph-guided Evaluation Benchmark. arXiv 2604.09285
