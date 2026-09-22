# Reproducibility Rules — Agent-Connected Evaluation

> **Status.** This document exists because of a concrete failure: a
> "semantically equivalent" rewording of the current-episode context string
> measurably changed real-API results in the B7d.3 pilot (see
> [`docs/experiments/agent_connected_eval.md`](experiments/agent_connected_eval.md)
> §15). These rules are the direct response, not a hypothetical best
> practice — follow them for every real-API agent-connected experiment from
> here on.

## The core rule

> **A semantically equivalent prompt is not considered an experimentally
> equivalent prompt. All model-visible strings must be versioned and
> reproduced exactly.**

Two sentences that a person would read as saying "the same thing" are not
guaranteed to produce the same model behavior. This project has now
measured a case where they didn't: a `no_experience` baseline swung from
real run-to-run variance to a fully deterministic `P(export)=1.00` in all 5
runs, purely because of a paraphrase ("the report for the current period"
vs. "the September 2026 report") introduced while refactoring a diagnostic
script. Treat every model-visible string — delegation text, environment
context, system/instruction prompts — as an exact experimental variable,
not as prose that can be regenerated as long as the meaning is preserved.

## What every real-API experiment must fix and record

1. **Exact model-visible prompt/context, byte-for-byte.** Store the literal
   string that is sent to the model, not the fields it was assembled from.
   `experiments/scenarios/*.json`'s `current_episode.context` /
   `current_episode.delegation` are the canonical example — code
   (`build_current_context()`) must return them as a passthrough, never
   reconstruct them.
2. **No silent paraphrasing.** If a prompt genuinely needs to change,
   that's a new scenario version (see below), not an in-place edit assumed
   to be equivalent. Don't "clean up" wording in a script that's meant to
   reproduce an earlier result.
3. **Scenario version.** Every scenario file carries a `scenario_version`
   field. Bump it whenever any model-visible string in the file changes,
   even a single word.
4. **Git commit SHA.** Record the commit the run was executed against
   (`git rev-parse HEAD`). Code that builds prompts, parses responses, or
   computes entropy can change results just as much as the prompt itself.
5. **Model name.** Record the exact model string passed to the API
   (e.g. `gpt-4o-mini`) — never assume a default.
6. **N / repetitions / entropy threshold.** Record samples-per-condition
   (`N`), independent repetitions (`runs`), and any threshold used to gate
   a decision (e.g. the clarification entropy threshold). These are
   experimental parameters, not incidental script defaults.
7. **Delegation/context hash.** Record a SHA256 fingerprint of every
   model-visible delegation/context string actually used
   (`experience_transfer.scenario_fingerprint()` / `text_sha256()`). A
   fingerprint lets a raw results file be checked against the scenario file
   that claims to have produced it, without trusting a version number
   alone.
8. **Raw vs. aggregate output separation.**
   - Raw API stdout/JSON (individual completions, per-call responses) stays
     **local only** — the session scratchpad, never the repository, never
     committed. It is billed, non-deterministic, and too large to be a
     durable artifact.
   - Aggregate/summary results (the numbers a table or figure is built
     from) **are** preserved in the repository — today as recorded
     observations in
     [`docs/experiments/agent_connected_eval.md`](experiments/agent_connected_eval.md),
     and, once a structured experiment runner exists, as committed
     `results/agent/<experiment>/summary.csv` +
     `results/agent/<experiment>/manifest.json` (see below). Never commit
     a raw per-call transcript; always commit the aggregate that a reviewer
     would need to check a reported number.
9. **No evaluation truth in the runtime.** The real agent runtime
   (`AgentDelegationRuntime` and everything it calls) must never read
   `task.truth`, a ground-truth label, or any other evaluation-only
   signal to make an execution decision. This is a reproducibility rule as
   much as a safety one: if the runtime could see the answer, the
   real-API numbers would not be evaluating the thing this project claims
   to evaluate. See
   [`docs/DESIGN_INVARIANTS.md`](DESIGN_INVARIANTS.md) item 1 for the
   invariant this rule is downstream of.

## Experiment manifest (for the next structured run)

`experiments/diagnostics/*.py` currently print a scenario version and
delegation/context SHA256 fingerprints at the start of each run (added
after the B7d.3 wording-drift finding) — that is the minimum bar. Once a
result is committed as a durable artifact (starting with the canonical
B7d.3 re-run), it should carry a `manifest.json` alongside its aggregate
data, e.g.:

```json
{
  "experiment": "b7d3_canonical",
  "git_sha": "...",
  "scenario": "external_audit_finance",
  "scenario_version": "1.1",
  "model": "gpt-4o-mini",
  "samples_per_condition": 20,
  "repetitions": 5,
  "delegation_sha256": "...",
  "context_sha256": "...",
  "conditions": ["no_experience", "v1", "v2"]
}
```

A CSV/table of numbers on its own is not sufficient evidence going
forward — a reviewer (or a future session picking this project back up)
must be able to answer "exactly what code and exactly what prompt produced
this result?" without re-reading chat history.

## Checklist (copy this before running a real-API experiment)

- [ ] Model-visible strings come from a versioned scenario file, not
      reconstructed prose.
- [ ] Scenario version recorded/bumped if anything changed.
- [ ] Git commit SHA recorded.
- [ ] Model name recorded.
- [ ] N, repetitions, and any threshold recorded.
- [ ] Delegation/context SHA256 fingerprint printed and recorded.
- [ ] Raw API output saved to the local scratchpad only — not committed.
- [ ] Aggregate results written into
      `docs/experiments/agent_connected_eval.md` (or a committed
      `manifest.json` + `summary.csv` once the structured runner exists).
- [ ] Runtime code path confirmed not to read `task.truth` or any
      evaluation-only field.
