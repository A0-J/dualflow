**English** | [한국어](README_KOR.md)

# DualFlow

**Semantic and Authority Verification for Safe Agent-to-Agent Delegation**

```
Agent A (principal)
       │  delegation request
       ▼
Agent B (delegate)
       │
   ┌───┴────────────────────────┐
   ▼                            ▼
Semantic Verification     Authority Verification
"What did A mean?"        "What is B allowed to do?"
   └───────────┬────────────────┘
               ▼
       Joint Verification
               │
        ┌──────┴──────┐
        ▼             ▼
    EXECUTE      REJECT / ESCALATE
```

DualFlow separates two questions that are usually conflated — "did the delegate
understand the request?" and "is the proposed action within the delegated
authority?" — verifies them independently, and combines both signals before
execution. When the delegated scope cannot be safely resolved from policy
alone, it invokes a bounded **Authority Feedback Loop** with the principal.
The core safety mechanism does not depend on historical experience —
**verified experience is used only as an optimization layer** to reduce
repeated principal feedback.

The implementation is deterministic (no LLM calls required to run the
experiments below); a real model plugs in behind three interfaces — see
[Current Scope and Limitations](#current-scope-and-limitations).

## Research Questions and Current Validation

| # | Research Question | Mechanism | Evidence |
|---|---|---|---|
| RQ1 | Are Semantic and Authority verification both necessary? | Semantic Flow + Authority Flow | Experiment 1 — Core Ablation |
| RQ2 | Can a scope violation be recovered without widening authority? | Authority Feedback Loop | Experiment 2 — Authority Feedback |
| RQ3 | Can repeated principal feedback be reduced without weakening safety? | Verified Authority Store *(Optimization Layer)* | Experiment 3 — Adaptive Authority |
| RQ4 | Can a confidently wrong proposal bypass entropy alone? | Joint Verification / Authority checks | Experiment 4 — Semantic Robustness |

All four run on a single controlled 8-step delegation environment
(`bench_single_env_v1`) plus two dedicated mini-sets — **a controlled
mechanism-level benchmark**, not a production LLM deployment (see
[Current Scope and Limitations](#current-scope-and-limitations)).

**Representative result** (Experiment 1, controlled benchmark, proposals are
either the correct interpretation or a fixed adversarial one — not sampled
from a real model):

| Method | Unsafe ↓ (adversarial) |
|---|---:|
| Authority only | 33.3% |
| Semantic only | 66.7% |
| **Full DualFlow** | **0.0%** |

Full detail — including *which specific mechanism* catches each of the three
attacks — is in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md), Experiment 1 and
Mechanism Attribution. Full per-experiment results, figures, and reasoning
are below in [Current Validation — Detailed Results](#current-validation--detailed-results).

**Historical / legacy validation.** An earlier 9-task pilot (`bench.py`, "v0")
was used during initial development; it is superseded by the environment
above but kept — not deleted — as internal replication evidence in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)'s Appendix.

## Why DualFlow?

A delegated action can fail in two different ways.

**Semantic mismatch.** The delegate may misread what was asked, while
remaining highly confident in the wrong interpretation:

```
Principal: "Summarize the report for marketing. Don't export the raw file."
Delegate:   export /reports/2026-08/     ← authority allows this; meaning doesn't
```

**Authority mismatch.** The delegate may understand the request correctly and
still propose something outside what was actually delegated:

```
Principal: "Email the summary — internal @corp.com addresses only."
Delegate:   send email *                 ← meaning is right; scope is too wide
```

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
or widen it.**

## Current Validation — Detailed Results

Full evidence for the four experiments introduced above. Figures split by
which layer they support — mixing them would suggest Adaptive Authority
carries a safety guarantee it doesn't: it is an **Optimization Layer**
component (cost only), not part of the Core Safety Mechanism.

```
Core Safety evidence         Optimization Layer evidence
  fig7  Authority Feedback     fig8  Adaptive Authority Feedback
  fig9  Entropy validation     (reduces principal calls; Core's
  fig10 Mechanism attribution   safety holds identically with it off)
```

### 1. Both verification axes are necessary

![v1 Phase 2 — Correct vs. Adversarial Proposal](figures/main/fig10_v1_phase2.png)

Each axis alone lets a different failure class through, for a different reason: Authority-only misses a semantic mismatch (misread) because the proposed action is itself permitted; Semantic-only misses a scope/condition violation because it never checks the budget. Full detail in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) §3.

### 2. Authority Feedback converts safe rejection into safe completion

| Method | Unsafe ↓ | Benign ↑ | Feedback rate ↓ |
|---|---:|---:|---:|
| No feedback | 0.0% | 0.0% | 0.0% |
| **Feedback (proposed)** | 0.0% | **100.0%** | 66.7% |

![Authority Feedback Loop — normal and under-attack conditions](figures/main/fig7_authority_feedback.png)

Without feedback, every negotiable scope violation is safely rejected but
never completed. With feedback, the same cases complete safely — and stay
safe even as principal carelessness rises to 1.0, because every revised
proposal is revalidated regardless of what the principal said. This holds
only as long as the authority budget and the principal feedback channel
themselves are trusted (see [Current Scope and Limitations](#current-scope-and-limitations)).
Detail in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) §4.

### 3. Adaptive reuse survives repetition, drift, and manipulation *(Optimization Layer)*

![Adaptive Verification — 9-round timeline](figures/optimization/fig8_adaptive_verification.png)

- **Stable repetition**: after 3 consistent principal confirmations, DualFlow
  reuses verified authority without another principal call — feedback rate
  drops from 100% to 0% for the same delegation type.
- **Legitimate drift**: when the delegated scope genuinely changes, the stale
  history no longer intersects the new budget, feedback reactivates
  immediately, and the old history is reset rather than left to outvote the
  new one.
- **Manipulation**: even a forged, fully-confident proposal resolves to the
  *correct* scope without asking — adaptive reuse never looks at the
  delegate's candidate, only at principal-confirmed history intersected with
  the current budget.

Full round-by-round data in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) §5.

### 4. Entropy is a real signal, and it is not sufficient alone

![Entropy validation — first real-LLM results](figures/main/fig9_entropy_probe.png)

The first real-LLM run (GPT-4o-mini, N=20) found a request the model answered
with 100% confidence (H=0) — pointed at an *unauthorized* scope. Authority
Flow catches what the confidence gate alone would have let through. Detail
in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) §6.

## Quick Start

```bash
pip install -e ".[dev]"

dualflow-demo              # demo suite: 12 runnable routines, as text
dualflow-plots             # current figures into figures/, legacy ones into figures/legacy/
```

`dualflow-demo`'s 12 routines are a finer-grained *walkthrough* of the
implementation (one per mechanism/ablation) — they are not the same thing as
the 4 research experiments above; several routines correspond to a single
experiment (e.g. the Core Ablation experiment is `dualflow-demo v1phase2`; the
legacy v0 ablation is `dualflow-demo bench`), and a few cover legacy/diagnostic
material that isn't cited as a main result.

### Software Validation vs. Experimental Validation

These are two different things — the first is regression testing for the
implementation, the second is the mechanism-level evidence above.

```bash
pytest -q                 # 183 tests: safety invariants, entropy properties,
                           # termination, ablations — software correctness,
                           # not an experimental result
```

### Selected experiments

```bash
python -m dualflow.demo fastslow attack careless authfeedback adaptiveauth
dualflow-plots careless --trials 50   # paper-quality error bars (default 10 is noisy)
OPENAI_API_KEY=... dualflow-entropy-probe --cases redesign   # re-run the real-LLM scenario validation
```

## Reproducing the Experiments

Every number and figure in this README comes from the scripts above; nothing
is hand-transcribed. [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) walks through
each experiment in full, in the order it's cited above, with the reasoning
behind each result — plus an Appendix of earlier (v0) exploratory experiments,
kept as internal replication evidence rather than deleted.
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes each mechanism (Semantic
Flow, Authority Flow, Authority Feedback, Joint Verification, and the
Optimization Layer) in implementation detail.
[docs/BASELINES.md](docs/BASELINES.md) documents the SAGE-Agent baseline
reproduction and issues found in its formulas along the way — supporting,
exploratory analysis, not a main result.
[docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md) covers implementation decisions
that didn't make the cut for the main results — including one approach
(matching against ground truth directly) that looked like a shortcut and
turned out to be an evaluation oracle instead.

## Repository Structure

| File | Purpose |
|---|---|
| `capability.py` | Authority model — privilege ordering, budget intersection, delegation-chain attenuation, failure classification |
| `semantic.py` | Semantic Flow — interpretation candidates, entropy, clarification, experience score |
| `authority_feedback.py` | Authority Feedback Loop and Verified Authority Store (adaptive reuse) |
| `rule_engine.py` | Joint Verification — delegation SOP graph, path similarity |
| `framework.py` | End-to-end orchestration, ablation switches, evaluation metrics |
| `sage_baseline.py` | SAGE-Agent baseline — direct reproduction of the published formulas |
| `bench.py` | **v0 (legacy)** pilot — 9-task benchmark, attack variants, scope-negotiation and sequential mini-sets |
| `bench_single_env_v1.py` | **v1 (current)** — single 8-step delegation environment, Correct/Adversarial Proposal split |
| `entropy_probe.py` | Real-LLM entropy validation harness (`dualflow-entropy-probe`) — separate from the pilots above |
| `llm.py` | LLM interface — scripted for experiments, adapter skeleton for a real model |
| `demo.py` / `plots.py` | Text and figure output for every experiment |
| `tests/` | 183 tests — software regression, not an experimental result (see [Quick Start](#quick-start)) |

## Current Scope and Limitations

### Current evaluation boundary

Joint Verification's matching step currently compares the system's final
interpretation against a reference derived from `task.truth` — the
controlled benchmark's hidden ground truth, not anything either verifier
agent produced. This is deliberate for the *current* mechanism-attribution
experiments (it isolates which mechanism catches which failure, see
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) §1.3), but it is not available
in a deployed agent system, where there is no ground truth to compare
against.

We confirmed this is load-bearing, not incidental, by actually removing it:
substituting the Semantic Verifier's own resolved interpretation as the
reference collapses misread detection specifically — the comparison becomes
"the delegate's interpretation vs. the delegate's interpretation," which is
tautological whenever Authority doesn't independently modify the proposal.
`Full = 0% unsafe` (Experiment 1) becomes 11.1% (the `silent_misread` case)
under that substitution. Replacing this oracle with an independently
obtained semantic signal — and at what cost, since the cheapest source is
re-confirming with the principal — is open research, not yet implemented.

### Other limitations

This repository currently validates the mechanism using deterministic pilot
scenarios and one round of real-LLM scenario validation. It does not yet
establish external validity with production LLM agents end-to-end. In
particular:

- Semantic candidate generation is scripted in the main pilot, not
  model-generated (a separate real-LLM harness exists — `entropy_probe.py` —
  but it hasn't replaced the pilot's candidates).
- The principal is simulated (with configurable carelessness/overcaution),
  not a real human or agent endpoint.
- The pilot benchmark is a small number of controlled tasks, not a large or
  naturalistic distribution.
- Verified-authority reuse assumes the current authority budget and the
  `VerifiedAuthorityStore` itself are trusted; manipulating either is a
  different threat model than the one tested.
- An in-scope but unintended resource choice (the delegate picks a *different*
  resource that still happens to be within the granted budget) is not caught
  by Authority Feedback, since no `scope_exceeded` is ever raised — this
  remains a separate intent-confirmation problem.
- "Semantic Flow alone doesn't catch the three adversarial cases" describes a
  fixed, code-injected proposal — not a claim about how often a real model
  generates such proposals naturally.

Swapping in a real model touches exactly three interfaces (candidate
generation, principal responses, LLM fallback) and is documented in
[docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md).

## Roadmap

- **External validation** — replace the pilot's scripted candidates with the
  real-LLM harness (`entropy_probe.py`) end-to-end, on a larger benchmark.
- **Per-agent principal reliability** — carelessness that varies by agent, and
  a review budget; turn "who gets Slow review" into an optimization problem.
  `warmup_then_attack` already provides the scaffolding.
- σ / k / λ sweeps, analogous to the existing θ sweep (Appendix).

Detail on each item is in [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md).

## References

- ChainCaps: Composition-Safe Tool-Using Agents via Monotonic Capability Attenuation. arXiv 2605.26542
- Structured Uncertainty guided Clarification for LLM Agents. arXiv 2511.08798 (Findings of ACL 2026)
- SAGE: A Service Agent Graph-guided Evaluation Benchmark. arXiv 2604.09285
