# experiments/

This directory is intentionally kept outside the `dualflow` package (no
`__init__.py`, not registered in `pyproject.toml`) — nothing under
`src/dualflow/` depends on anything here.

**`agent_smoke.py` is a development smoke test, not how results are
reproduced.** It runs one hand-picked scenario against the real API and
prints raw, human-readable output for eyeballing a single agent's
prompt/response by hand. To reproduce a recorded research result, use
`scenarios/` + `diagnostics/` (or, once it exists, a structured benchmark
runner under `results/agent/`) instead.

```text
experiments/
├── agent_smoke.py
│   Development smoke test. One canned scenario per --role
│   (principal/delegate/semantic/authority/all/runtime/sample/clarify/
│   experience), real API calls, human-readable output. For checking an
│   agent's actual prompt/response by hand — not for statistical
│   experiments or reproducing a reported number.
│
├── diagnostics/
│   ├── experience_transfer.py
│   │   Fixed reproducer for the B7d.1/B7d.2 semantic-transfer priming
│   │   diagnostics (docs/experiments/agent_connected_eval.md §7/§10).
│   │   Conditions A-E (no experience / summarize / export / read history
│   │   / structure-only neutral control). Do not extend this file with
│   │   new conditions — it exists to keep reproducing exactly those
│   │   recorded results.
│   └── experience_representation.py
│       B7d.3: real-API v1 vs. v2 experience-representation comparison,
│       across an ambiguous task and an explicit-instruction-change task
│       (docs/experiments/agent_connected_eval.md §15). A separate script
│       from experience_transfer.py on purpose, for the same reason —
│       each diagnostic script answers one fixed question and stays
│       reproducible, rather than accumulating unrelated conditions.
│
├── scenarios/
│   └── external_audit_finance.json
│       Authoritative scenario definition for the reproducible diagnostic
│       experiments in this directory. Stores the literal model-visible
│       delegation/context strings consumed by those diagnostics rather
│       than semantic fields from which Python reconstructs prose.
│       See docs/REPRODUCIBILITY.md for why this matters (a paraphrase
│       here measurably changed real-API results once already).
│
├── demo.py
│   Legacy controlled benchmark — minimal end-to-end examples of the
│   deterministic simulated pipeline (framework.py/semantic.py). Mechanism
│   regression check, not agent-connected evaluation.
│
├── benchmark.py
│   Legacy controlled benchmark — canonical benchmark used during the core
│   refactor. Simulated task state and simulated principal; useful for
│   regression testing the deterministic fusion/authority layer, not a
│   real-agent result.
│
├── entropy_probe.py
│   Legacy controlled benchmark — focused diagnostic for semantic
│   uncertainty behavior in the deterministic simulated pipeline.
│
├── bench.py
│   Legacy controlled benchmark support. Two unrelated roles in one file:
│   build_tasks() (superseded v0 task set — benchmark.py replaced it) and
│   scope_negotiation_tasks()/scope_negotiation_sequence() (still the
│   canonical data source for the Authority Feedback / Adaptive Authority
│   experiments). See docs/EXPERIMENTS.md §1.1 for the exact mapping.
│
├── plots.py
│   Legacy controlled benchmark — generates the retained figures from
│   demo.py/benchmark.py/bench.py results. Not part of the dualflow
│   package; imports the other experiment scripts as siblings.
│
└── baselines/
    └── sage.py
        Reproduces the SAGE-Agent paper's Algorithm 1 as a comparison
        baseline for the legacy controlled benchmark. framework.py has no
        dependency on it — it's wired in only through the generic
        Config.semantic_engine injection hook, via as_semantic_engine().
```

## Which script should I run?

| I want to... | Use |
| --- | --- |
| Check one agent's real prompt/response by hand | `agent_smoke.py --role <name>` |
| Reproduce the B7d.1/B7d.2 priming diagnostic | `diagnostics/experience_transfer.py` |
| Reproduce the B7d.3 v1-vs-v2 representation pilot | `diagnostics/experience_representation.py` |
| Run the deterministic controlled benchmark / regression check | `benchmark.py`, `demo.py`, `entropy_probe.py` |
| Regenerate the retained figures | `plots.py --outdir figures` |

Every script's `--help` documents its exact flags and what it does — this
table is a map, not a substitute for reading it.

## Real-API scripts

The real-API modes of `agent_smoke.py` and the scripts under
`diagnostics/` can make billed OpenAI API calls. They require:

```bash
pip install -e ".[agent]"
```

and `OPENAI_API_KEY` must be provided through the environment, never as a
CLI argument.

Before running or reporting a real-API experiment, read
[`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md).

Any reported result should record enough metadata to identify the exact
experiment, including at least:

- Git commit SHA
- scenario ID and version
- model name
- samples per condition (N)
- number of repetitions
- delegation/context fingerprints when applicable

Ad-hoc stdout and scratch API outputs should remain local unless they are
intentionally promoted into a versioned research result under
`results/agent/`.
