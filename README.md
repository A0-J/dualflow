[한국어](README_KOR.md)

# DualFlow

**Independent Semantic and Authority Verification for LLM Agent Delegation**

DualFlow is a research framework for studying **safe delegation between LLM agents**.

When a Principal Agent delegates a task to a Delegate Agent, two different failures can occur:

1. the Delegate may **misinterpret what the Principal meant**, and
2. the Delegate may attempt an action **outside the authority it was given**.

DualFlow separates these problems into two independent verification paths:

- **Semantic Verifier** — checks whether the Delegate's proposed action matches the delegated intent.
- **Authority Verifier** — checks whether the proposed action stays within the delegated capability and scope.

The two verifiers operate over different evidence spaces and their results are combined through deterministic logic.

---

## Architecture

```text
Principal Agent
      |
      | natural-language delegation
      v
Delegate Agent
      |
      | proposed interpretation / action
      v
+-------------------------------+
|                               |
v                               v
Semantic Verifier          Authority Verifier
|                               |
| SemanticVerdict               | AuthorityVerdict
|                               |
+---------------+---------------+
                |
                v
       Deterministic Fusion
                |
         EXECUTE / REJECT
```

A key design goal is to keep semantic reasoning and authority enforcement separate. The Semantic Verifier cannot grant authority, and the Authority Verifier cannot redefine the Principal's intent.

---

## Design Principles

### Independent verification

Each agent reasons from a deliberately restricted evidence space. The Delegate does not receive benchmark truth or the Principal's hidden structured intent.

The Semantic Verifier independently evaluates: delegation + Delegate proposal + runtime-visible context.

The Authority Verifier evaluates: Delegate proposal + delegated `Budget` + runtime-visible context.

Verifier outputs are not shared with each other before fusion.

### Deterministic authority boundary

Authority is not decided by an LLM. DualFlow keeps authority enforcement behind deterministic mechanisms such as `Budget`, `Budget.meet()`, `delegation_chain()`, and `check_authority()`.

If a proposal exceeds the delegated scope, an LLM may suggest a narrower proposal, but the result must still pass `check_authority()` before it can be accepted. Semantic experience never grants or expands authority.

### Oracle-free agent runtime

The real agent runtime does not use `task.truth`, ground-truth labels, expected decisions, or evaluation labels to make execution decisions. Evaluation-only information remains outside the runtime path.

Legacy controlled benchmarks are retained separately for mechanism-level evaluation and regression testing.

---

## Agent-Connected Evaluation

The current research focuses on a repeated Principal–Delegate workflow. Instead of assuming that every delegation is immediately clear, the Delegate can estimate uncertainty over multiple candidate interpretations.

```text
Principal delegation
        |
        v
Delegate candidate sampling
        |
        v
Candidate distribution
        |
        v
Semantic entropy
        |
   +----+----+
   |         |
 low H      high H
   |         |
   |         v
   |    clarification request
   |         |
   |         v
   |      Principal
   |         |
   |         v
   |    clarified intent
   |         |
   +----+----+
        |
        v
verification / execution
```

The longer-term question is whether Principal-confirmed experience from previous interactions can reduce future clarification overhead without reducing semantic correctness or authority safety.

### Current experimental scenario

The current pilot uses a narrow, controlled workflow: **external-audit financial-report delegation**.

Example ambiguous delegation:

> Please prepare the September 2026 financial report for the external audit.

The experimental environment currently uses the canonical action ontology `read` / `summarize` / `export`, with a grounded resource and scope, e.g. `RESOURCE = file`, `SCOPE = /reports/2026-09/`.

The current entropy measurements therefore represent uncertainty over this canonical semantic action space, not over every possible free-form action an LLM could generate.

### Candidate sampling and semantic entropy

Instead of asking the model for a self-reported confidence score, DualFlow estimates uncertainty using independent LLM samples:

```python
distribution = delegate.sample_candidates(
    delegation=delegation,
    context=context,
    n=N,
)
```

Candidate probabilities are estimated empirically. Semantic entropy is computed as:

$$ H = -\sum_i p_i \log_2 p_i $$

with each candidate's contribution being $-p_i \log_2 p_i$. Entropy computation itself is local and requires no additional API calls.

### Observed ambiguity

For the delegation "Please prepare the August 2026 financial report for the external audit.", five independent runs with ten samples each produced:

| Run | export | summarize | read | Entropy |
| --- | --- | --- | --- | --- |
| 1 | 0.50 | 0.50 | 0.00 | 1.000 |
| 2 | 0.70 | 0.30 | 0.00 | 0.881 |
| 3 | 0.60 | 0.40 | 0.00 | 0.971 |
| 4 | 0.60 | 0.40 | 0.00 | 0.971 |
| 5 | 0.50 | 0.50 | 0.00 | 1.000 |

Across all 50 samples, the resource and scope remained correctly grounded. The ambiguity was concentrated on `summarize` vs. `export`. This provides a controlled real-API example of semantic ambiguity in natural-language delegation.

### Principal clarification

When candidate entropy exceeds a configurable threshold, the Delegate can ask the Principal for clarification:

> **Delegate:** Should I export the financial report for the external audit, or would you prefer a summary?
>
> **Principal:** Please summarize the August 2026 financial report.

The Delegate then samples its interpretation again using the clarification. In the initial real-API pilot, clarification was triggered in four of five runs. All four clarified runs changed from approximately `H = 0.88 ~ 0.97 bits` to `summarize = 1.00`, `H = 0.00`. The average entropy reduction among clarified runs was **0.904 bits**.

This pilot demonstrates that Principal clarification can substantially reduce semantic uncertainty.

### Important finding: entropy is not correctness

One non-clarified run produced `export = 0.80`, `summarize = 0.20`, `H = 0.722`. Because the pilot threshold was 0.8, clarification was skipped and `export` became the selected interpretation.

This exposed an important distinction: **low entropy does not imply correct interpretation.** DualFlow therefore evaluates two separate quantities:

- **Semantic uncertainty** — entropy `H`.
- **Semantic alignment** — the probability assigned to the Principal-confirmed intent.

A model can become more confident while becoming less aligned with the Principal.

### Verified experience

Clarification outcomes can be stored as Principal-confirmed semantic experiences, indexed deterministically by `(principal_id, task_category)`. No embedding retrieval, fuzzy matching, or LLM-based retrieval is required.

A verified experience records the previous delegation, the clarification question, the Principal's answer, the confirmed interpretation, and the pre-/post-clarification distributions. Low-entropy Delegate guesses that were never confirmed by the Principal are not treated as verified experience. Experience records contain no authority grants or capability information.

### Experience-transfer diagnostics

The initial experience representation exposed historical interactions primarily as a structured label, e.g. `Principal-confirmed interpretation: ACTION=summarize`. Real-API experiments showed that this representation did not reliably transfer the Principal-confirmed semantic pattern.

A 400-call diagnostic and a subsequent 500-call structure-only control showed that:

- `summarize` and `export` historical labels could produce essentially identical output distributions;
- a neutral historical block with no action semantics could still strongly change the candidate distribution;
- entropy could decrease while semantic alignment became worse.

This led to another central finding: **historical-context priming must be distinguished from genuine semantic transfer.** The detailed negative results and diagnostic conditions are preserved rather than removed from the experimental record — see [`docs/experiments/agent_connected_eval.md`](docs/experiments/agent_connected_eval.md).

### Semantic-memory representation v2

The current redesign represents experience as a relationship instead of a single historical label:

```text
previous ambiguous delegation
        |
        v
Delegate clarification question
        |
        v
Principal clarification
        |
        v
confirmed meaning
```

The model is instructed to use verified prior interactions specifically as evidence for resolving ambiguous parts of the current delegation, while explicit instructions in the current task must override historical experience.

The initial v2 pilot showed a partial semantic-transfer signal on the ambiguous task while preserving explicit current instructions. However, the experiment also revealed a context-wording drift confound, so the absolute baseline from that pilot should not be directly compared with earlier experiments. Exact prompt reproducibility is therefore the current priority before the next large evaluation.

---

## Repository Structure

```text
dualflow/
├── src/
│   └── dualflow/
│       ├── principal_agent.py
│       ├── delegate_agent.py
│       ├── semantic.py
│       ├── authority_feedback.py
│       ├── capability.py
│       ├── framework.py
│       ├── clarification.py
│       ├── agent_experience.py
│       ├── experience_aware_delegate.py
│       ├── agent_runtime.py
│       └── llm.py
│
├── experiments/
│   ├── agent_smoke.py
│   ├── demo.py
│   ├── benchmark.py
│   ├── bench.py
│   ├── plots.py
│   ├── entropy_probe.py
│   ├── baselines/
│   │   └── sage.py
│   │
│   ├── diagnostics/
│   │   ├── experience_transfer.py
│   │   └── experience_representation.py
│   │
│   └── scenarios/
│       └── external_audit_finance.json
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── EXPERIMENTS.md
│   ├── BASELINES.md
│   ├── DESIGN_NOTES.md
│   └── experiments/
│       └── agent_connected_eval.md
│
├── results/
│   └── agent/
│
├── figures/
│   └── agent/
│
└── tests/
```

- `src/dualflow/` contains the runtime mechanisms.
- `experiments/scenarios/` contains reproducible research scenarios — the literal, model-visible delegation/context strings, not just semantic fields code reconstructs prose from.
- `experiments/diagnostics/` contains experiments used to investigate specific failure modes (semantic-transfer priming, representation redesign).
- `experiments/baselines/sage.py` reproduces the SAGE-Agent paper's Algorithm 1 as a comparison baseline; it is not part of the `dualflow` core package.
- `docs/experiments/agent_connected_eval.md` preserves the chronological real-API experimental record, including unexpected and negative results.
- `results/agent/` and `figures/agent/` are reserved for the sequential agent evaluation and final research outputs.

---

## Installation

Install the core package:

```bash
pip install -e .
```

For running tests and regenerating figures:

```bash
pip install -e ".[dev]"
```

For real LLM-agent experiments:

```bash
pip install -e ".[agent]"
```

Set the API key using an environment variable:

```bash
export OPENAI_API_KEY="..."
```

PowerShell:

```powershell
$env:OPENAI_API_KEY="..."
```

Do not place API keys in source files or command-line arguments.

---

## Quick Checks

Run the test suite:

```bash
pytest -q
```

Run the controlled demonstration:

```bash
python experiments/demo.py
```

Run the controlled benchmark:

```bash
python experiments/benchmark.py
```

Generate the retained experiment figures:

```bash
python experiments/plots.py --outdir figures
```

Inspect available real-agent smoke modes:

```bash
python experiments/agent_smoke.py --help
```

Examples:

```bash
python experiments/agent_smoke.py --role principal
python experiments/agent_smoke.py --role delegate
python experiments/agent_smoke.py --role semantic
python experiments/agent_smoke.py --role authority
python experiments/agent_smoke.py --role runtime
python experiments/agent_smoke.py --role sample --n 10
python experiments/agent_smoke.py --role clarify --n 10 --entropy-threshold 0.8
python experiments/agent_smoke.py --role experience --n 10
```

### Reproducing experience-transfer diagnostics

The experience-transfer diagnostics are implemented separately from the runtime:

```bash
python experiments/diagnostics/experience_transfer.py --help
python experiments/diagnostics/experience_representation.py --help
```

The experiment scenario is stored under `experiments/scenarios/external_audit_finance.json`. Raw local API outputs are intentionally excluded from Git. The repository instead preserves the experimental conditions, reproducible diagnostic code, aggregate observations, known confounds, and the chronological research log.

---

## Controlled Benchmark vs. Agent-Connected Evaluation

The repository currently contains two different forms of evaluation.

**Controlled benchmark** — used for mechanism-level testing, regression testing, and controlled semantic/authority failure cases. Some legacy controlled evaluation paths use oracle information internally and should not be interpreted as real-agent runtime evaluation.

**Agent-connected evaluation** — uses real Principal, Delegate, Semantic Verifier, and Authority Verifier interactions. The actual runtime does not use benchmark truth to make decisions.

The two evaluation settings are intentionally documented separately.

---

## Current Research Status

| Item | Status |
| --- | --- |
| LLM abstraction | complete |
| Principal Agent | complete |
| Delegate Agent | complete |
| Semantic Verifier | complete |
| Authority Verifier | complete |
| Oracle-free deterministic runtime | complete |
| Candidate sampling | complete |
| Entropy-based clarification | complete |
| Verified experience storage | complete |
| Experience-aware sampling | implemented |
| v1 semantic-transfer diagnostic | complete |
| Structure-only control | complete |
| v2 semantic-memory redesign | implemented |
| v2 real-API pilot | partial evidence |
| Exact scenario/prompt reproducibility | current priority |
| Sequential multi-episode evaluation | not started |

The sequential experiment is intentionally postponed until the semantic-transfer behavior is reproducible under an exactly fixed model-visible scenario.

---

## Current Research Questions

1. Can verified Principal-specific experience improve semantic alignment on later ambiguous delegations?
2. Can it do so without overriding explicit changes in the current instruction?
3. Does accumulated experience reduce clarification rate, LLM calls, token usage, and latency?
4. Can these efficiency gains be achieved without degrading semantic correctness or authority safety?
5. How sensitive are the results to prompt wording and entropy thresholds?
6. How well does the method extend beyond a closed canonical action ontology?

---

## Documentation

- [`docs/DESIGN_INVARIANTS.md`](docs/DESIGN_INVARIANTS.md) — conditions that must never be broken by any future change, in one place
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — what must be fixed and recorded for a real-API result to be reproducible (scenario version, model, prompt fingerprint, ...)
- [`experiments/README.md`](experiments/README.md) — what each script under `experiments/` does and which one to use
- [`docs/experiments/agent_connected_eval.md`](docs/experiments/agent_connected_eval.md) — real-LLM agent-connected evaluation log, including unexpected/negative results
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — mechanism-level architecture explanation (source-code level)
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — controlled experiments and historical pilot results
- [`docs/BASELINES.md`](docs/BASELINES.md) — baseline notes
- [`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md) — abandoned approaches and implementation decisions
- [README_KOR.md](README_KOR.md) — Korean README

This repository is under active research development. Unexpected and negative experimental results are intentionally preserved in the documentation. In particular, lower semantic entropy is not treated as evidence of improved semantic correctness unless semantic alignment is also evaluated.

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
