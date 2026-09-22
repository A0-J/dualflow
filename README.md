English | [한국어](README_KOR.md)

# DualFlow

**Semantic and Authority Verification for Safe Agent-to-Agent Delegation**

DualFlow is a research prototype for verifying agent-to-agent delegation before execution. It separates two questions that should not be collapsed into a single confidence score:

1. **Semantic verification** — did the delegate correctly understand what the principal requested?
2. **Authority verification** — is the proposed execution actually permitted by the delegated authority?

The current core exposes these checks as two independent verifier components:

- `SemanticVerifierAgent` → `SemanticVerdict`
- `AuthorityVerifierAgent` → `AuthorityVerdict`

Their results are combined by a deterministic joint-verification layer in `framework.py`.

> **Current status**
>
> The verifier-agent architecture is implemented, but the repository still evaluates it primarily with deterministic simulated scenarios. It is **not yet an end-to-end validation with independent production LLM agents A and B**. The next evaluation phase will connect real agent calls, remove the remaining simulation-only intent oracle from runtime matching, and regenerate the main experimental results.

---

## Motivation

Delegation can fail in two different ways.

### Semantic mismatch

The delegate can confidently misunderstand the request.

```text
Principal:              "Read last month's report."
Delegate interpretation: /reports/2025-08/
```

Low uncertainty does not guarantee that the interpretation is correct.

### Authority mismatch

The delegate can understand the request correctly while proposing an action outside the authority that was actually delegated.

```text
Delegated scope:  read /reports/2026-08/*
Proposed scope:   read /reports/*
```

A clear interpretation is not automatically an authorized action.

DualFlow therefore verifies semantics and authority independently before execution.

---

## Architecture

```text
                 Agent A (Principal)
                         |
                    delegation
                         v
                  Agent B (Delegate)
                         |
                  proposed action
                         |
             +-----------+-----------+
             |                       |
             v                       v
   SemanticVerifierAgent    AuthorityVerifierAgent
             |                       |
      SemanticVerdict          AuthorityVerdict
             |                       |
             +-----------+-----------+
                         |
                         v
               Deterministic Fusion
                  /             \
             EXECUTE           REJECT
```

### Semantic Verifier Agent

`SemanticVerifierAgent` is implemented in `semantic.py`.

It owns the existing semantic verification strategies:

- **Fast** — entropy / clarification based verification
- **Slow** — principal review
- **AND** — both paths must agree
- **Adaptive** — Fast first, with escalation when required

The verifier returns a `SemanticVerdict`. The current refactor preserves the behavior of the original Fast/Slow/AND/Adaptive implementation; the agent abstraction does not by itself add a new LLM call.

### Authority Verifier Agent

`AuthorityVerifierAgent` is implemented in `authority_feedback.py`.

It owns the complete authority-verification sequence:

1. deterministic authority check,
2. bounded authority feedback when negotiation is allowed,
3. re-validation of any revised interpretation,
4. production of an `AuthorityVerdict`.

The low-level authority primitives remain deterministic and live in `capability.py`.

### Deterministic capability layer

`capability.py` provides the authority primitives used by the Authority Verifier Agent:

- `Budget`
- budget intersection / attenuation
- delegation-chain composition
- non-amplification
- `check_authority()`

A delegated capability cannot become broader as it moves down a delegation chain.

### Joint verification

`framework.py` owns orchestration and final fusion. The rule engine supplies deterministic structured-intent matching; it is not a third verifier agent.

The intended deployment architecture is:

```text
Semantic evidence  -> SemanticVerifierAgent  -> SemanticVerdict
Authority evidence -> AuthorityVerifierAgent -> AuthorityVerdict
                                            \ /
                                  deterministic fusion
```

Future deployments may give each verifier its own model call and tool context while sharing the same API client.

---

## What Is Implemented Today

The current repository implements:

- independent Semantic and Authority verifier interfaces,
- deterministic capability checking and delegation attenuation,
- bounded authority feedback,
- semantic Fast / Slow / AND / Adaptive strategies,
- verified-authority reuse as an optimization layer,
- deterministic joint verification,
- a SAGE-Agent comparison implementation,
- deterministic benchmark and diagnostic experiments.

The core refactor is intentionally behavior-preserving. The verifier-agent abstractions were extracted from the existing implementation without changing the current benchmark outputs.

---

## Important Evaluation Boundary

The current experiments are **mechanism tests**, not yet production-agent validation.

The controlled benchmark uses simulated task state and a simulated principal. Ground truth is valid for evaluation and for simulating what the principal knows.

However, the current branch still contains a simulation-only dependency in final joint matching: the runtime intent reference is derived from `task.truth` through `DelegationTask.intent_fields`. A deployed verifier cannot assume access to that hidden truth.

We confirmed this is load-bearing, not incidental, by actually removing it: substituting the Semantic Verifier's own resolved interpretation as the reference collapses misread detection specifically — the comparison becomes "the delegate's interpretation vs. the delegate's interpretation," which is tautological whenever Authority doesn't independently modify the proposal. `Full = 0% unsafe` (Experiment 1) becomes 11.1% (the `silent_misread` case) under that substitution.

The next research-behavior change is therefore to make the runtime joint check use the resolved `SemanticVerdict` as its semantic reference, while keeping `task.truth` only for evaluation — and to design where an independent second semantic signal comes from in deployment (principal reconfirmation is the obvious candidate, which reopens the same cost/safety tradeoff Adaptive Feedback already explores).

Until that change and the real Agent A/B evaluation are complete, the existing pilot numbers should be treated as controlled sanity checks and ablations rather than the paper's final main-result evidence.

---

## Quick Start

```bash
pip install -e ".[dev]"
pytest -q
```

Current refactor baseline:

```text
183 passed
```

Run the minimal demo:

```bash
python experiments/demo.py
```

Run the canonical controlled benchmark:

```bash
python experiments/benchmark.py
```

Generate the retained experiment figures:

```bash
python experiments/plots.py --outdir figures
```

Inspect the entropy probe options:

```bash
python experiments/entropy_probe.py --help
```

---

## Experiments

The experiment scripts are intentionally kept outside the core package.

```text
experiments/
├── demo.py
├── benchmark.py
├── entropy_probe.py
├── plots.py
├── bench.py
└── baselines/
    └── sage.py
```

### `demo.py`

Minimal end-to-end examples of the current DualFlow pipeline.

### `benchmark.py`

Canonical controlled benchmark used during the core refactor.

### `entropy_probe.py`

Focused diagnostic experiment for semantic uncertainty behavior.

### `plots.py`

Generates the retained experiment figures. Plotting is separated from the `dualflow` package so the core implementation does not depend on visualization code. It imports `bench.py` and `benchmark.py` as sibling scripts, not from the `dualflow` package.

### `bench.py`

Not a single legacy file — it serves two different roles. `build_tasks()` and the 9-task pilot set it defines are the superseded v0 benchmark (`experiments/benchmark.py` is the canonical replacement); `scope_negotiation_tasks()` / `scope_negotiation_sequence()`, by contrast, are still the current canonical data source for the Authority Feedback and Adaptive Authority experiments. See `docs/EXPERIMENTS.md` §1.1 for the exact mapping.

### `baselines/sage.py`

Reproduces the SAGE-Agent paper's Algorithm 1 as a comparison baseline. It is not part of the `dualflow` core package — `framework.py` has no dependency on it and only knows it through the generic `Config.semantic_engine` injection hook, via `as_semantic_engine()`.

Historical pilot and diagnostic results that are no longer part of the main evaluation are documented in `docs/EXPERIMENTS.md`.

---

## Repository Structure

```text
src/dualflow/
├── capability.py
├── semantic.py
├── authority_feedback.py
├── rule_engine.py
├── framework.py
├── llm.py
└── ...

experiments/
├── demo.py
├── benchmark.py
├── entropy_probe.py
├── plots.py
├── bench.py
└── baselines/
    └── sage.py

docs/
tests/
figures/
```

### Core modules

| Module | Responsibility |
| --- | --- |
| `semantic.py` | Semantic representations and `SemanticVerifierAgent` |
| `authority_feedback.py` | `AuthorityVerifierAgent`, bounded authority feedback, verified-authority reuse |
| `capability.py` | Deterministic capability / budget / delegation primitives |
| `rule_engine.py` | Deterministic structured-intent compatibility checks |
| `framework.py` | End-to-end orchestration and deterministic fusion |
| `llm.py` | Model-facing interfaces / adapters |

The SAGE-Agent comparison baseline (`experiments/baselines/sage.py`) and the v0/mini-set benchmark scaffolding (`experiments/bench.py`) live under `experiments/`, not `src/dualflow/` — neither is part of the core package. See [Experiments](#experiments) above.

---

## Current Limitations

The current repository should not yet be read as evidence of production-ready multi-agent safety.

- Agent A and Agent B are not yet connected as independent real LLM agents in the canonical experiment.
- The verifier agents are structurally independent, but the current controlled experiments do not yet require two independent production model calls.
- Semantic candidate generation and principal behavior are still simulated in the controlled benchmark.
- Authority checking currently operates on explicit `Budget` / capability state; policy retrieval from natural-language organizational policies is future work.
- The current runtime joint match still contains simulation-derived ground-truth wiring, as described above.
- The benchmark is small and controlled rather than a naturalistic task distribution.
- Existing pilot figures are useful for regression testing and mechanism ablations, but they should not be treated as the final main experimental evidence for the agent-pair architecture.

---

## Next Steps

1. **Remove the runtime ground-truth oracle**
   - derive final semantic matching from `SemanticVerdict`, not `task.truth`;
   - retain ground truth only for scoring and simulation.

2. **Connect real Agent A and Agent B**
   - generate the delegation and proposal through independent model calls;
   - keep Semantic and Authority verifier contexts separate.

3. **Upgrade AuthorityVerifierAgent for deployment**
   - policy retrieval,
   - delegation / identity lookup,
   - resource-state inspection,
   - evidence-backed authority decisions,
   - explicit unknown / escalation handling.

4. **Run a new agent-connected benchmark**
   - no verification,
   - semantic-only,
   - authority-only,
   - single-verifier / joint-prompt baseline,
   - DualFlow independent verifier pair + deterministic fusion.

5. **Regenerate paper-facing figures**
   - correct execution rate,
   - unsafe execution / attack success rate,
   - false rejection rate,
   - clarification / escalation rate,
   - LLM calls,
   - latency,
   - token cost,
   - mechanism ablations.

---

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — architecture and safety invariants
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — controlled experiments and historical pilot results
- [`docs/BASELINES.md`](docs/BASELINES.md) — baseline notes
- [`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md) — implementation and research design notes
- [`docs/experiments/agent_connected_eval.md`](docs/experiments/agent_connected_eval.md) — real-LLM agent-connected evaluation log (B7a onward), including unexpected/negative results

Some documentation still reflects the pre-agent-pair implementation and is being updated as part of the refactor.

---

## References

The repository currently discusses or reproduces ideas from the following work:

- ChainCaps: *Composition-Safe Tool-Using Agents via Monotonic Capability Attenuation* (arXiv 2605.26542)
- SAGE-Agent: *Structured Uncertainty guided Clarification for LLM Agents* (arXiv 2511.08798, Findings of ACL 2026)
- SAGE-Bench: *A Service Agent Graph-guided Evaluation Benchmark* (arXiv 2604.09285)

See the documentation for the exact mapping between prior work and the DualFlow implementation.

---

## License

MIT
