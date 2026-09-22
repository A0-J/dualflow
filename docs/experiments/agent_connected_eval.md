# Agent-Connected Evaluation Log

This document tracks the real-LLM, agent-connected experiments built on
`research/agent-connected-eval` (B7a onward). It intentionally does **not**
cover the earlier core refactor or controlled-benchmark history — see
`docs/EXPERIMENTS.md` for that. The purpose of this document is to:

1. record what experiments were actually run against the real API,
2. preserve unexpected/failed experimental results as-is,
3. record important observations and confounds,
4. make the experimental evolution understandable to a reviewer/advisor.

Negative and unexpected results are kept here deliberately, not cleaned up —
they are exactly what motivated later design decisions (most notably,
separating "uncertainty" from "semantic alignment" as two independent
metrics; see §11).

## 1. Experimental Goal

This experiment studies natural-language delegation between a Principal
Agent and a Delegate Agent.

When the Delegate receives an ambiguous delegation, it generates multiple
candidate interpretations and measures the uncertainty of those
interpretations using empirical entropy.

If uncertainty is sufficiently high, the Delegate asks the Principal for
clarification.

The current experiments investigate three questions:

1. Does natural-language delegation produce measurable semantic ambiguity?
2. Does Principal clarification reduce that ambiguity?
3. Can a Principal-confirmed experience from a previous episode reduce
   ambiguity in a later, similar episode?

The verifier architecture (B6) remains separate from this mechanism:

```text
Principal
    |
    | delegation
    v
Delegate
    |
    | proposed interpretation
    v
+-----------------------+
|                       |
v                       v
Semantic Verifier   Authority Verifier
|                       |
+-----------+-----------+
            |
            v
    deterministic fusion
```

Semantic experience is never treated as authority and cannot modify the
delegated budget or capability constraints.

## 2. Controlled Action Ontology

The current pilot uses a controlled canonical action ontology:

- `read`
- `summarize`
- `export`

Therefore, the entropy reported in the current experiments is **entropy
over the predefined canonical semantic action space**, not entropy over
every possible free-form action an LLM could generate.

A future open-ended experiment may allow free-form actions followed by
semantic clustering.

The controlled ontology is currently used to isolate semantic ambiguity
from surface-form variation such as "inspect", "review", "extract key
findings", "prepare a summary", "retrieve and summarize", etc.

## 3. B7a — Candidate Sampling

### Scenario

The initial ambiguity experiment used the following delegation:

> "Please prepare the August 2026 financial report for the external audit."

The environment was grounded so that:

```
RESOURCE = file
SCOPE    = /reports/2026-08/
```

Only the action interpretation was allowed to vary. Each run independently
sampled the Delegate's interpretation 10 times. Five independent runs were
performed.

### Results

| Run | export | summarize | read | Entropy |
| --- | --- | --- | --- | --- |
| 1 | 0.50 | 0.50 | 0.00 | 1.000 |
| 2 | 0.70 | 0.30 | 0.00 | 0.881 |
| 3 | 0.60 | 0.40 | 0.00 | 0.971 |
| 4 | 0.60 | 0.40 | 0.00 | 0.971 |
| 5 | 0.50 | 0.50 | 0.00 | 1.000 |

Across all 50 samples, `RESOURCE=file` / `SCOPE=/reports/2026-08/` was
preserved in every sample. The ambiguity was therefore concentrated on
**summarize vs. export**, not on resource or scope interpretation.

### Observation

The phrase "prepare the financial report for the external audit" was
consistently ambiguous between preparing an internal summary and
preparing/exporting the report for external delivery. This established
that the test scenario produces measurable semantic ambiguity in the real
API.

## 4. B7b — Entropy-Based Clarification

The Delegate was extended with a single-round clarification mechanism.

Pilot rule:

```
entropy <= 0.8   -> use top candidate
entropy >  0.8   -> ask Principal for clarification
```

The threshold of 0.8 is a pilot parameter, not a final research threshold.

### Real API Results

| Run | Pre distribution | H_before | Clarified | Post distribution | H_after | Final |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | export .80 / summarize .20 | 0.722 | No | — | — | export |
| 2 | export .70 / summarize .30 | 0.881 | Yes | summarize 1.00 | 0.000 | summarize |
| 3 | export .70 / summarize .30 | 0.881 | Yes | summarize 1.00 | 0.000 | summarize |
| 4 | export .60 / summarize .40 | 0.971 | Yes | summarize 1.00 | 0.000 | summarize |
| 5 | export .70 / summarize .30 | 0.881 | Yes | summarize 1.00 | 0.000 | summarize |

Aggregate:

- mean pre-entropy = 0.867 bits
- mean post-entropy (clarified runs) = 0.000 bits
- mean entropy reduction = 0.904 bits
- clarification trigger rate = 4/5

All four clarified runs converged to `summarize = 1.00`, `entropy = 0.00`.

### Observation

The real-API pilot supports the observation that **Principal clarification
can substantially reduce Delegate semantic uncertainty**.

However, Run 1 revealed an important boundary case: `export=0.80,
summarize=0.20, H=0.722`. Since the entropy was below the pilot threshold,
clarification was skipped and export was selected. Therefore: **low
entropy does not imply semantic correctness.** This motivated separating
uncertainty from semantic alignment in later experiments (§11).

## 5. B7c — Verified Experience Storage

Clarification outcomes can be stored as verified semantic experiences.
Experiences are indexed by `(principal_id, task_category)`. No embedding,
fuzzy matching, or LLM retrieval is used.

Only Principal-confirmed clarification outcomes are eligible for storage.
A low-entropy Delegate guess made without Principal clarification is never
treated as verified experience.

Current experience fields: `principal_id`, `task_category`, `delegation`,
`clarification_question`, `principal_answer`, `confirmed_interpretation`,
`pre_distribution`, `post_distribution`, `episode_id`.

Experience storage contains no `Budget`, authority grant, capability,
`task.truth`, or evaluation label. Semantic experience cannot expand
delegated authority.

## 6. B7d — Experience-Aware Sampling

The next experiment tested whether a verified experience from an earlier
episode reduces semantic uncertainty in a later, related episode.

- Previous episode: August 2026 external-audit report. Principal-confirmed
  action: `summarize`.
- Current episode: "Please prepare the September 2026 financial report for
  the external audit." Current scope: `/reports/2026-09/`.

The experiment compared:

- **A.** No previous experience
- **B.** One matching Principal-confirmed experience

### Initial 5-run result

| Run | H_without | H_with | ΔH | Top-1 without | Top-1 with |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.722 | 0.000 | 0.722 | export | export |
| 2 | 0.469 | 0.000 | 0.469 | export | export |
| 3 | 0.469 | 0.000 | 0.469 | export | export |
| 4 | 0.722 | 0.000 | 0.722 | export | export |
| 5 | 0.722 | 0.000 | 0.722 | export | export |

Aggregate: mean H_without = 0.621, mean H_with = 0.000, mean ΔH = 0.621.

At first glance the experience appeared highly effective because entropy
fell to zero in all five runs. However, candidate-level inspection
revealed the opposite semantic behavior:

- Without experience: mean P(export) ≈ 0.84, mean P(summarize) ≈ 0.16
- With a Principal-confirmed **summarize** experience: P(export) = 1.00,
  P(summarize) = 0.00

**The experience reduced uncertainty but did not transfer the
Principal-confirmed semantic pattern — instead, the Delegate became more
confident in export.** This is an important negative result.

## 7. B7d.1 — Semantic-Transfer Diagnostic

To determine whether the model was reacting to the semantic content of the
historical experience (rather than just its presence), a controlled
diagnostic was performed. The same current September episode was used in
every condition.

Four historical-context conditions were compared:

- **A.** No experience
- **B.** Historical confirmed action = `summarize`
- **C.** Historical confirmed action = `export`
- **D.** Historical confirmed action = `read`

Each condition: N=20 samples, 5 independent repetitions. Total: 4
conditions × 20 samples × 5 runs = **400 real API calls**.

### Aggregate results

| Condition | Mean H | Mean P(summarize) | Mean P(export) | Mean P(read) | Top-1 |
| --- | --- | --- | --- | --- | --- |
| No experience | 0.644 | 0.17 | 0.83 | 0.00 | export 5/5 |
| summarize experience | 0.000 | 0.00 | 1.00 | 0.00 | export 5/5 |
| export experience | 0.000 | 0.00 | 1.00 | 0.00 | export 5/5 |
| read experience | 0.878 | 0.70 | 0.30 | 0.00 | summarize 5/5 |

All 400 samples preserved `SCOPE=/reports/2026-09/` — no historical August
scope was ever copied into the current task.

## 8. Important Unexpected Result

The most important comparison is **summarize history vs. export history**.
Both produced exactly `P(export)=1.00, P(summarize)=0.00, H=0.00` in every
repetition.

Therefore, under the current experience representation, **changing the
historical confirmed ACTION from summarize to export did not change the
Delegate's output distribution.** This does not support the hypothesis
that the structured historical confirmed ACTION is currently being
transferred as intended.

For the summarize condition: P(summarize) without experience = 0.17,
P(summarize) with experience = 0.00, so ΔP_confirmed = **−0.17**. The
confirmed action became *less* likely after adding its own verified
experience.

## 9. Read-Experience Condition (Confounded — Do Not Over-Interpret)

The read diagnostic produced a different result: P(summarize)=0.70,
P(export)=0.30, P(read)=0.00, H=0.878.

This condition should not yet be strongly interpreted. Its Principal-answer
wording differed from the otherwise parallel summarize/export conditions:

> "Please just read the August 2026 financial report; no further action is
> needed."

This introduces a possible wording confound — the phrasing itself, not
just the labeled confirmed action, may have influenced the result. A
future diagnostic should make the three historical answers structurally
parallel, e.g.:

```
Please summarize the August 2026 financial report.
Please export the August 2026 financial report.
Please read the August 2026 financial report.
```

## 10. B7d.2 — Structure-Only Control

To distinguish semantic transfer from prompt/history-block priming, a fifth
condition was added, and the read condition's wording was made parallel to
summarize/export's (per §9's caveat: `"Please read the August 2026
financial report."`, no extra clause).

Conditions (same current September episode in every condition):

- **A.** No experience
- **B.** Confirmed action = `summarize`
- **C.** Confirmed action = `export`
- **D.** Confirmed action = `read` (now wording-symmetric with B/C)
- **E.** A neutral historical block with no action semantics at all — it
  states only that a prior interaction with this Principal/workflow
  existed, was clarified and confirmed, and that the current delegation
  and environment always take precedence. It names no action, no resource,
  and repeats none of the old delegation's content.

Each condition: N=20 samples, 5 independent repetitions. Total: 5
conditions × 20 samples × 5 runs = **500 real API calls**.

### Aggregate results

| Condition | Mean H | P(summarize) | P(export) | P(read) | Top-1 |
| --- | ---: | ---: | ---: | ---: | --- |
| No experience | 0.673 | 0.20 | 0.80 | 0.00 | export 5/5 |
| summarize experience | 0.000 | 0.00 | 1.00 | 0.00 | export 5/5 |
| export experience | 0.000 | 0.00 | 1.00 | 0.00 | export 5/5 |
| read experience (parallel wording) | 0.980 | 0.50 | 0.50 | 0.00 | summarize 3/5, export 2/5 |
| neutral history (structure-only) | 0.114 | 0.02 | 0.98 | 0.00 | export 5/5 |

All 500 samples again preserved `SCOPE=/reports/2026-09/`.

### Explicit comparisons

- **B vs. C** — identical for a second time (5/5 runs each at
  `H=0.000, P(export)=1.00`). Reconfirms that the summarize/export label
  itself is not what the model is reacting to.
- **B vs. D / C vs. D** — clearly different: B/C fully collapse to export;
  D sits near maximum two-way entropy (0.980), split roughly evenly
  between summarize and export, and its own top-1 is not even stable
  across the 5 repetitions (3 summarize, 2 export) — the only condition
  with that instability.
- **A vs. E** — clearly different: E's entropy (0.114) and P(export)
  (0.98) are far more concentrated than baseline A (0.673, 0.80), despite
  E containing zero action-level semantic content.
- **E vs. B/C** — E tracks B/C far more closely than it tracks A (P(export)
  0.98 vs. B/C's 1.00, vs. A's 0.80).

### Main observation

The neutral historical block (E) produced nearly the same behavior as the
summarize/export experience blocks (B/C). Therefore, the strong entropy
reduction observed in B7d cannot be interpreted as successful transfer of
the Principal-confirmed semantic action — the presence/format of a
historical-context block by itself strongly pushed the Delegate toward its
own pre-existing export-leaning interpretation.

The `read` condition shows the model is not entirely content-blind — D
changed the distribution substantially relative to B/C/E — but not in the
direction of adopting `read`; instead it increased uncertainty between
summarize and export. So content has *some* effect, just not the intended
"adopt the confirmed action" effect.

Combined with the D-condition re-run: making the wording parallel changed
the read condition substantially (previously P(summarize)=0.70 with a
non-parallel answer; now P(summarize)=0.50 with parallel wording),
confirming that part of the original B7d.1 read result was a wording
confound, not a pure content effect.

### Key lesson

Entropy reduction and semantic alignment must be evaluated separately. A
lower entropy can indicate *stronger confidence in an interpretation that
is less aligned with the Principal-confirmed intent* — exactly what B/C/E
show relative to the Principal's actual confirmed `summarize` outcome.

## 11. Key Experimental Lesson

B7d demonstrates that the following two quantities must be evaluated
**separately**:

- **Uncertainty** — measured using semantic entropy `H`.
- **Semantic alignment** — measured using the probability assigned to the
  Principal-confirmed action, `P(Principal-confirmed action)`.

The experiment demonstrated that **entropy decrease ≠ semantic correctness
improvement**. For example, `H: 0.644 -> 0.000` can occur while
`P(summarize): 0.17 -> 0.00`. Entropy alone cannot be used as evidence that
experience improves semantic understanding. §10's structure-only control
reconfirms this with a second, independent 500-call diagnostic and pins the
likely mechanism down further: concentration without correct transfer.

A third lesson was added later, in §15: **prompt/context wording is itself
an experimental variable**, not incidental scaffolding. A reconstructed
context sentence that was only semantically equivalent to (not byte-
identical to) the wording used in earlier real-API runs measurably changed
the model's behavior (a fully deterministic `no_experience` baseline where
earlier pilots had shown real variance). Scenario strings that reach the
model now have to be treated with the same rigor as the independent
variable being tested, not assembled fresh by each script.

## 12. Current Interpretation

The current implementation successfully demonstrates:

- deterministic Principal/task-category experience retrieval,
- preservation of the current task's scope,
- isolation of semantic experience from authority,
- candidate-distribution concentration after historical context is added.

However, across two independent diagnostics (§7 B7d.1, §10 B7d.2 — 900
real API calls total), the experiment does **not** demonstrate successful
transfer of the Principal-confirmed semantic action. The dominant effect
is best explained as **prompt/history-block priming toward the model's own
pre-existing interpretation** rather than Principal-specific semantic
learning: the neutral, content-free control (Condition E) tracked the
labeled summarize/export conditions far more closely than it tracked the
no-experience baseline. The read condition shows the mechanism is not
purely content-blind, but its content-sensitivity does not currently take
the form of "adopt the confirmed action."

**Conclusion: the current experience representation (a bare
`ACTION=<x> RESOURCE=<y>` label per historical example) is an inadequate
design for semantic-pattern transfer and needs to be redesigned (B7d.3)
before any sequential multi-episode evaluation (B7e) would be
meaningful.**

## 13. Known Experimental Caveats

- **Closed action ontology** — current entropy is measured over
  `read`/`summarize`/`export`, not arbitrary free-form Agent actions (§2).
- **Pilot threshold** — the clarification threshold 0.8 is experimental,
  not a final calibrated value.
- **Small early pilots** — B7a/B7b/B7d initial experiments used five
  repetitions and should be interpreted as pilot results, not statistically
  powered conclusions.
- **Experience representation is confirmed inadequate** — not merely
  "unproven" (§8, §10) — the structure-only control isolates prompt/format
  priming as the dominant effect, not semantic content.
- **Entropy is not correctness** — a concentrated candidate distribution
  can still be concentrated on an incorrect interpretation; this is the
  central finding of B7d/B7d.1/B7d.2, not just a caveat.
- ~~D-condition wording confound~~ — resolved in §10: re-run with parallel
  wording still shows D diverging from B/C (P(summarize) 0.50 vs. B/C's
  0.00), so content sensitivity is real, just not correctly directed.
- **B7d.3 pilot baseline not comparable to earlier baselines** — §15: a
  context-wording drift (reconstructed vs. literal canonical prose) made
  this pilot's `no_experience` baseline more deterministic than earlier
  B7d/B7d.1/B7d.2 baselines. Fixed for future runs (scenario file now holds
  the literal string, with a regression test and fingerprinting), but the
  600-call B7d.3 pilot numbers reported in §15 should be read as
  within-run (`v2` vs. `v1` vs. `no_experience`) evidence only, not as
  directly comparable to §7/§10's absolute numbers.
- **`v2` semantic transfer is not yet established** — §16's canonical
  re-run (wording confound removed) shows only a weak, run-inconsistent
  `P(summarize)` lift (`C1` beats `A1` in 3/5 runs, ties in 1/5, loses in
  1/5). Do not cite the §15 pilot's "5/5 consistent improvement" figure —
  it was measured against a baseline §16 shows was artificially
  deterministic. The current honest status is "`v2` is less harmful than
  `v1`, direction is positive on average, magnitude and consistency are
  not yet sufficient to call it transfer" — not "`v2` works."
- **`v2`'s positive `delta_sum`/`delta_export` in B7d.4 do not by
  themselves show semantic transfer** — §18: both deltas were `+0.23`, but
  entirely because `export`-history collapsed deterministically to
  `P(export)=1.00` (10/10 runs), not because `summarize`-history raised
  `P(summarize)` above the no-experience baseline (it did not — mean
  `0.230` vs. baseline `0.250`, 4 wins/2 ties/4 losses across 10 runs). A
  positive delta pair computed from an asymmetric, one-sided collapse must
  not be read as evidence of bidirectional content sensitivity — see §18's
  conclusion and B7d.5 (§19) for the follow-up needed to separate
  block-presence priming from genuine content effects.

## 14. Current Status

| Item | Status |
| --- | --- |
| Candidate ambiguity measurement (B7a) | complete |
| Real-API ambiguity validation | complete |
| Entropy-based clarification (B7b) | complete |
| Real-API clarification pilot | complete |
| Verified experience storage (B7c) | complete |
| Experience-aware sampling (B7d) | complete |
| Semantic-transfer diagnostic (B7d.1) | complete |
| Structure-only control (B7d.2) | complete |
| Experience representation | **confirmed inadequate — redesign needed** |
| Experience representation redesign (B7d.3) | **pilot complete — partial evidence + context-wording confound found (§15)** |
| Experiment reproducibility fix (canonical wording + fingerprinting) | complete (§15) |
| B7d.3 canonical-context re-validation (600 calls) | **complete — v1 harmful, v2 weak/inconsistent (§16)** |
| Semantic-content ablation (B7d.4) | **complete — asymmetric/prior-congruent priming, not established transfer (§18)** |
| Neutral-format control (B7d.5) | script prepared, not yet run (§19) |
| Sequential experience evaluation (B7e) | not started |

The sequential multi-episode experiment (B7e) stays intentionally
postponed. The canonical-context B7d.3 re-run (§16) removed the
context-wording confound and found that, against a correctly-behaving
(non-deterministic) baseline, `v1` is clearly harmful (it collapses the
model's natural ambiguity to a confident wrong answer) and `v2` shows only
a weak, run-inconsistent directional signal — not yet a demonstration that
verified experience reliably transfers semantic content. B7d.4 (§18) ran
that content-sensitivity check and found an asymmetric, prior-congruent
result: `export`-history collapsed deterministically toward the model's
pre-existing `export` lean (10/10 runs), while `summarize`-history showed
no consistent improvement over no-experience at all — so `v2` semantic
transfer is still not established. B7d.5 (§19) is designed to separate
whether that collapse comes from `v2`'s block presence/format alone or
specifically from prior-congruent content, before any `v3` redesign is
considered. The original plan for B7d.3 was to redesign the experience
representation so it presents ambiguity → clarification → confirmed-meaning
as a
relationship, not a bare action label — conceptually:

```
Previous ambiguous delegation:
"Prepare the August report for the external audit."

Principal clarification:
"Please summarize it. Do not export it."

Confirmed interpretation:
summarize
```

together with instructions that frame history as evidence for resolving
*current* ambiguity, not as a fallback the current request can be
defaulted to. B7d.3 must not be declared successful based on entropy
reduction alone. Its success criterion is two-part, tested against two
different current tasks in the same small comparison (A. no experience,
B. old representation, C. redesigned representation):

- **Ambiguous task** ("Please prepare the September report for the
  external audit.") — `P(Principal-confirmed recurring intent)` should
  increase, and `H` should decrease or stay flat.
- **Explicit-change task** ("This time, export the September report to
  the external auditor.") — `P(current explicit action)` must stay high;
  the prior summarize experience must not override an explicit current
  instruction.

B7d.3 only counts as progress if both conditions hold simultaneously.

## 15. B7d.3 — Semantic-Memory Representation Redesign
        (Pilot with Context-Wording Confound)

The two-task, three-condition, N=20 × 5-runs real-API pilot (600 calls;
`experiments/diagnostics/experience_representation.py`) against
`render_experience_block_v2` produced:

| Task | Condition | Mean H | P(summarize) | P(export) |
| --- | --- | --- | --- | --- |
| Ambiguous | no experience | 0.000 | 0.00 | 1.00 |
| Ambiguous | v1 | 0.057 | 0.01 | 0.99 |
| Ambiguous | v2 | 0.663 | 0.19 | 0.81 |
| Explicit change | no experience | 0.000 | 0.00 | 1.00 |
| Explicit change | v1 | 0.000 | 0.00 | 1.00 |
| Explicit change | v2 | 0.000 | 0.00 | 1.00 |

**Reading the result (entropy is secondary, per the §14 success rule):**
`v2` produced a higher `P(summarize)` than both `no experience` and `v1` in
**5/5 independent runs** on the ambiguous task (0.19 mean vs. 0.00/0.01) —
a real, directionally consistent signal that `v2` is not behaving like
`v1`'s pure-priming failure mode (§8, §10). On the explicit-change task,
`v2` left `P(export)` at 1.00 in every run, identical to `no experience`
and `v1` — the redesigned representation never overrode an explicit
current instruction, in 0/15 rows across the three conditions.

This is **partial evidence, not full success**: `P(summarize)=0.19` is far
below the illustrative target used when B7d.3 was scoped (`0.85`, always an
example, never a pass/fail threshold — §14), and `export` remained the
top-1 candidate in every ambiguous-task run even under `v2`. The result
does not match any single one of the four anticipated outcomes cleanly —
it sits between "v2 shows a real but modest improvement" and "v2 is too
weak to call the redesign solved" — and is reported as such rather than
forced into one label.

### Discovered confound: context-wording drift

While reading this pilot's raw output, the `no_experience` baseline was
noticeably *more* deterministic (P(export)=1.00, H=0.000, 5/5 runs) than
the baselines seen in the earlier B7d/B7d.1/B7d.2 pilots. Tracing this
down: `experiments/diagnostics/experience_transfer.py`'s
`build_current_context()` — reused by this pilot's script — reconstructed
the current-episode context prose from scenario fields via an f-string
template, and that reconstruction was only semantically equivalent, not
byte-identical, to the wording actually used in every earlier real-API
B7a/B7b/B7d experiment (`experiments/agent_smoke.py`'s
`EXAMPLE_CURRENT_CONTEXT`):

- Earlier, real-API-validated wording:
  `"The September 2026 report is located at /reports/2026-09/."`
- Reconstructed wording used (unnoticed) in this pilot:
  `"The report for the current period is located at /reports/2026-09/."`

**This is not cosmetic** — the real model responded differently to the two
versions of the sentence, exactly the kind of prompt-wording sensitivity
this whole diagnostic arc (§7 D-condition wording confound, §10 wording
symmetrization) already had reason to expect but had not yet controlled
for at the *environment-context* level. Because all six conditions in this
pilot shared the same (drifted) wording, the **relative** comparison
between `no_experience`/`v1`/`v2` above is still internally valid, but the
**absolute** baseline numbers are not directly comparable to B7d/B7d.1/
B7d.2's earlier baselines.

### Fix applied (reproducibility only — no representation/threshold/logic changes)

- `experiments/scenarios/external_audit_finance.json` (`scenario_version`
  bumped to `1.1`) now stores the literal, model-visible
  `current_episode.context` / `current_episode.delegation` strings
  directly, byte-identical to `agent_smoke.py`'s `EXAMPLE_CURRENT_CONTEXT`/
  `EXAMPLE_CURRENT_DELEGATION` — verified by a new regression test
  (`tests/test_scenario_reproducibility.py`) that imports both and asserts
  equality, so the two can never silently drift apart again.
- `build_current_context()` is now a plain passthrough (`return
  scenario["current_episode"]["context"]`) — it no longer assembles prose
  from vocabulary/temporal fields. Both diagnostic scripts consume the
  scenario's exact string as-is.
- Both diagnostic scripts now print the scenario version and a SHA256
  fingerprint of the delegation/context strings actually used at the start
  of every run (`experience_transfer.scenario_fingerprint()`/
  `text_sha256()`), so a raw results file can later be checked against the
  exact wording that produced it.
- Nothing in `src/dualflow/experience_aware_delegate.py`
  (`render_experience_block_v1`/`_v2`, `ExperienceAwareDelegate`),
  sampling, entropy, Authority, or clarification logic was touched. This
  was a scenario/harness reproducibility fix only.

**Third methodological lesson, added to §11's list:** alongside (1) entropy
≠ semantic alignment, and (2) block presence/format can prime independent
of content, this pilot adds (3) prompt/context wording is itself an
experimental variable that must be pinned down and version-controlled —
"semantically equivalent" rewording is not safe to treat as identical in
this kind of measurement.

### Next step

Re-run the identical 600-call B7d.3 comparison (2 tasks × 3 conditions ×
N=20 × 5 runs) against the now-canonical, fingerprinted scenario wording,
before drawing any final conclusion about `v2`. `render_experience_block_v1`/
`_v2` and `ExperienceAwareDelegate` remain unchanged going into that re-run.
A follow-up ablation (`v2`-with-summarize-history vs. `v2`-with-export-
history, mirroring §10's B/C separation) is planned for after the canonical
re-run, to separate "v2's format helps" from "v2 genuinely carries the
confirmed action's content" — not before.

## 16. B7d.3 Canonical-Context Re-Run

### Experiment identity

```
experiment_id:        b7d3_canonical_rerun
git_sha:               a714e360ac08c142a00af91e0185ed349737e341
scenario_id:            external_audit_finance
scenario_version:       1.1
delegation_sha256 (ambiguous):
  38b8fc2394a7bc9a4d043dd8078ecd6194f3855d710710d1fce23727a4a5fab1
delegation_sha256 (explicit_change):
  45984505604c517c35de0edf51292df211163fbaa2bf377080b03b576d2ab8ad
context_sha256:
  a1a5da5674cab40190610056581c30802e13277b70c058eb2317569085233892
model:                  gpt-4o-mini
samples_per_condition:  20
repetitions:            5
total calls:            600
script:                 experiments/diagnostics/experience_representation.py
```

Between `490f5a3` (the reproducibility fix commit, §15) and `a714e36` (the
commit this run executed against), only documentation files changed
(`README.md`, `README_KOR.md`, `docs/DESIGN_INVARIANTS.md`,
`docs/REPRODUCIBILITY.md`, `experiments/README.md`) — verified via `git
diff --stat 490f5a3..a714e36` before running. No source, experiment, or
scenario file differs between the two commits, so this run is code- and
scenario-identical to the fixed §15 wording.

### Canonical baseline restoration

The headline structural finding of this re-run: restoring the canonical
wording made the `no_experience` baseline **non-deterministic again**.
Under the drifted wording (§15), `A1 no_experience` was `H=0.000,
P(export)=1.00` in all 5 runs. Under the canonical wording, `A1` shows real
run-to-run variance (`H` 0.469–0.811, `P(summarize)` 0.10–0.25) — consistent
with the model genuinely being uncertain between `summarize`/`export` for
this delegation, matching the original B7a pilot's observed ambiguity (§3).
This directly confirms the §15 hypothesis: the drifted wording was not
cosmetic, it was artificially collapsing the model's natural uncertainty.
One practical consequence: **the §15 pilot's absolute baseline numbers were
not just "not directly comparable" as originally cautioned — they were
measuring a qualitatively different (artificially deterministic) baseline
condition**, not simply a slightly-shifted version of the same one.

### A1 / B1 / C1 aggregate — ambiguous task

| Condition | Mean H | P(summarize) | P(export) |
| --- | --- | --- | --- |
| A1 no experience | 0.707 | 0.200 | 0.800 |
| B1 v1 | 0.000 | 0.000 | 1.000 |
| C1 v2 | 0.784 | 0.250 | 0.750 |

### Run-level C1 − A1 comparison

| run | A1 | B1 | C1 | C1 − A1 |
| --- | --- | --- | --- | --- |
| 1 | 0.10 | 0.00 | 0.15 | +0.05 |
| 2 | 0.20 | 0.00 | 0.35 | +0.15 |
| 3 | 0.20 | 0.00 | 0.20 | 0.00 |
| 4 | 0.25 | 0.00 | 0.20 | **−0.05** |
| 5 | 0.25 | 0.00 | 0.35 | +0.10 |

`C1 > B1` in 5/5 runs (`B1` is always exactly 0.00). `C1 > A1` in only 3/5
runs, tied in 1/5, and **lower than A1** in 1/5 (run 4). Mean `C1 − A1` is
`+0.05` — positive, but not a consistent per-run improvement. `top1 =
export` in all 30 rows (A1/B1/C1 combined) — `v2` never flips the top
candidate.

### Explicit-change task (A2 / B2 / C2)

All 15 rows (5 runs × 3 conditions) show `H=0.000`, `P(summarize)=0.00`,
`P(export)=1.00`, with zero exceptions — identical to the §15 pilot. `v2`
never overrode the explicit current instruction, across either the
drifted-wording pilot or this canonical re-run.

### Structural checks

- **Scope**: all 30 rows show `scopes_seen = ['/reports/2026-09/']` only —
  600/600 samples, 0 August-scope leakage.
- **Clarification calls**: 0 — `experience_representation.py` only calls
  `ExperienceAwareDelegate.sample_candidates()`, never
  `ask_clarification()`/`ClarifyingDelegate` (confirmed by source
  inspection).
- **Authority calls**: 0 — `experience_aware_delegate.py`/
  `agent_experience.py` import neither `Budget`, `check_authority`, nor
  `AuthorityVerifierAgent` (matches `docs/DESIGN_INVARIANTS.md` item 4).
- **`task.truth`**: 0 — no such parameter exists anywhere in this call path.
- **Tokens**: 288,200 input / 15,364 output across 600 calls, in line with
  the §15 pilot's token profile.

### Conclusion

1. **`v1` is clearly harmful, not merely ineffective.** Against the
   canonical (non-deterministic) baseline, `v1` does not fail to transfer
   the confirmed action — it actively collapses the model's natural
   ambiguity (`H: 0.707 → 0.000`) into a fully confident WRONG answer
   (`export`, `P(summarize): 0.20 → 0.00`). This is a sharper restatement
   of the §7/§10 priming finding: the mere presence of a historical block
   in `v1`'s format does not just fail to help, it measurably suppresses
   the correct answer's probability below what doing nothing would have
   given.
2. **`v2` shows only weak and inconsistent directional improvement.** Mean
   `P(summarize)` rises from `0.200` (A1) to `0.250` (C1), but this is not
   a per-run-dominant effect — `C1` beats `A1` in 3/5 runs, ties in 1/5,
   and is lower than `A1` in 1/5. `top1` remains `export` throughout. This
   is a much weaker and less consistent result than the §15 pilot
   suggested, because the §15 pilot's apparent "5/5 consistent +0.19 lift"
   was measured against an artificially deterministic (`P(summarize)=0.00`
   always) baseline that this re-run shows was itself the wording-drift
   artifact, not the true no-experience condition.
3. **Therefore: semantic transfer is not yet established.** `v2`'s
   canonical-context signal is real in direction (mean lift is positive,
   and `v2` clearly outperforms `v1`) but too weak and too inconsistent to
   be read as evidence that verified experience reliably transfers
   Principal-confirmed semantic content. What canonical `v2` currently
   demonstrates is, at most, "somewhat better than `v1`'s harmful
   collapse" — not "successful semantic transfer." Whether the small
   positive signal `v2` does show is actually driven by the historical
   experience's *content* (summarize vs. export) rather than by `v2`'s
   format alone is exactly the question B7d.4 (§17) is designed to
   separate.

No code was modified to produce this re-run or this write-up —
`render_experience_block_v1`/`_v2` and `ExperienceAwareDelegate` are
unchanged from §15. The §15 pilot's numbers are preserved above, unedited,
as a separate historical record of the wording-confounded run; §16 does not
overwrite them.

## 17. B7d.4 — Semantic-Content Ablation (Prepared, Not Yet Run)

> **Status update**: this experiment has since been run — see §18 for the
> real-API results. This section is left unedited below as the original
> design record (what was prepared and why), per this document's standing
> policy of not overwriting earlier sections.

§16's canonical re-run leaves one question open: is `v2`'s weak positive
signal driven by the *content* of the historical experience (which action
was actually confirmed), or — like `v1` — mostly by the presence/format of
a historical block regardless of content? A content-only ablation is
needed to separate these, mirroring the logic of §10's structure-only
control but applied to `v2` instead of `v1`.

**New script, not an extension of a fixed reproducer**:
`experiments/diagnostics/experience_semantic_ablation.py`. Neither
`experience_transfer.py` (fixed B7d.1/B7d.2 reproducer) nor
`experience_representation.py` (fixed B7d.3 reproducer) is modified by
this script — it reuses `load_scenario()`/`build_current_context()`/
`make_experience()`/`text_sha256()`/`scenario_fingerprint()` from
`experience_transfer.py` via sibling import, and
`render_experience_block_v2` from `experience_aware_delegate.py`, exactly
as written.

**Design**: same canonical ambiguous September delegation as §16's Task 1,
three conditions:

- **A.** no experience
- **B.** `v2` rendering of a `summarize`-confirmed historical experience
- **C.** `v2` rendering of an `export`-confirmed historical experience

B and C are built from `make_experience()`'s shared answer template (the
§10 wording-symmetry fix) so they stay structurally identical — same
sentence shape, same header, same length — differing only in the one
confirmed-action word. `N=20`, `repetitions=10`, 3 conditions → 600 calls.

**Primary metrics** (entropy recorded but explicitly secondary, per §11):

```
P(summarize | summarize-history)   P(summarize | export-history)
P(export    | summarize-history)   P(export    | export-history)

delta_sum    = P(summarize | summarize-history) - P(summarize | export-history)
delta_export = P(export    | export-history)    - P(export    | summarize-history)
```

**Interpretation rule, fixed before running**: if `summarize`-history
selectively raises `P(summarize)` and `export`-history selectively raises
`P(export)` (both deltas clearly positive), that is evidence `v2` carries
genuine semantic-content sensitivity. If B and C behave similarly to each
other regardless of which action was confirmed, the current `v2`
representation still does not demonstrate semantic transfer — it would be
behaving like a `v1`-style presence/format effect, just weaker in
magnitude.

**Status**: script written and structurally verified (fake-client dry run,
`--help`, full `pytest -q` regression unaffected) — not yet run against the
real API. No prompt tuning is planned after seeing results, and B7e remains
not started regardless of this ablation's outcome, per the current gating
order (§14).

## 18. B7d.4 — Semantic-Content Ablation (Results)

### Experiment identity

```
experiment_id:          b7d4_semantic_content_ablation
git_sha:                 3c03dda8894da6a1f42394add26682d1d051b2bd
scenario_id:             external_audit_finance
scenario_version:        1.1
delegation_sha256:
  38b8fc2394a7bc9a4d043dd8078ecd6194f3855d710710d1fce23727a4a5fab1
context_sha256:
  a1a5da5674cab40190610056581c30802e13277b70c058eb2317569085233892
model:                   gpt-4o-mini
samples_per_condition:   20
repetitions:             10
total calls:             600
script:                  experiments/diagnostics/experience_semantic_ablation.py
```

Single ambiguous September task only (no explicit-change task this round).
`preflight`: before this run, a separate 1-call real-API check via the
unmodified `build_llm_client()`/`DelegateAgent.propose()` path confirmed a
real, non-zero-token response (`input_tokens=403`, `output_tokens=25`,
class `OpenAILLMClient`) — no source file was touched to perform it.

### Aggregate results

| Condition | Mean H | P(summarize) | P(export) |
| --- | --- | --- | --- |
| A no experience | 0.775 | 0.250 | 0.750 |
| B v2 summarize-history | 0.744 | 0.230 | 0.770 |
| C v2 export-history | **0.000** | **0.000** | **1.000** |

### Run-level A vs B vs C (10 runs)

| run | A no_exp | B sumhist | C exphist | B − A |
| --- | --- | --- | --- | --- |
| 1 | 0.25 | 0.15 | 0.00 | −0.10 |
| 2 | 0.30 | 0.25 | 0.00 | −0.05 |
| 3 | 0.35 | 0.10 | 0.00 | −0.25 |
| 4 | 0.15 | 0.25 | 0.00 | +0.10 |
| 5 | 0.35 | 0.25 | 0.00 | −0.10 |
| 6 | 0.25 | 0.20 | 0.00 | −0.05 |
| 7 | 0.30 | 0.45 | 0.00 | +0.15 |
| 8 | 0.30 | 0.30 | 0.00 | 0.00 |
| 9 | 0.05 | 0.15 | 0.00 | +0.10 |
| 10 | 0.20 | 0.20 | 0.00 | 0.00 |

`B` beats `A` in 4/10 runs, ties in 2/10, loses in 4/10 — no consistent
direction. `C` is `H=0.000`, `P(export)=1.00` in **10/10 runs**, no
exception.

### delta_sum / delta_export

```
delta_sum    = P(summarize|summarize-hist) - P(summarize|export-hist) = 0.23 - 0.00 = +0.23
delta_export = P(export|export-hist)    - P(export|summarize-hist)    = 1.00 - 0.77 = +0.23
```

Both positive — but this is **not sufficient evidence of semantic
transfer**. The composition of each delta matters: `delta_sum` is positive
almost entirely because `C`'s `P(summarize)` is forced to exactly `0.00`
in every run (total collapse), not because `B` meaningfully raises
`P(summarize)` above baseline — `B`'s mean (`0.230`) is in fact slightly
*below* `A`'s mean (`0.250`). Symmetric bidirectional evidence would
require `B` to clearly beat `A` on its own; it does not.

### Structural / safety checks

- **Scope**: all 30 rows show `scopes=['/reports/2026-09/']` only —
  600/600, 0 August-scope leakage.
- **Parse failures**: 0 — every row shows `calls=20` and
  `P(read)+P(summarize)+P(export)=1.00`.
- **Clarification calls**: 0 — confirmed by source inspection
  (`experience_semantic_ablation.py` only calls
  `ExperienceAwareDelegate.sample_candidates()`).
- **Authority calls**: 0 — `experience_aware_delegate.py`/
  `agent_experience.py` reference neither `Budget`, `check_authority`, nor
  `AuthorityVerifierAgent` (DESIGN_INVARIANTS.md item 4).
- **`task.truth`**: 0 — not present anywhere in this call path.
- **Tokens**: 283,800 input / 15,360 output across 600 calls.

### Conclusion

- `delta_sum = +0.23` and `delta_export = +0.23` — both positive, but this
  is **not** sufficient evidence of semantic transfer on its own.
- **`summarize`-history did not improve over baseline**: mean `P(summarize)`
  `0.230` vs. `A`'s `0.250` (slightly lower), and per-run it beats `A` in
  only 4/10 runs, ties in 2/10, loses in 4/10 — no consistent direction.
- **`export`-history collapsed deterministically toward the model's
  existing export prior**: `H=0.000`, `P(export)=1.00` in 10/10 runs, with
  zero exceptions — the same total-collapse pattern `v1` showed in §16,
  now appearing under `v2`'s format when the historical content happens to
  agree with the model's pre-existing lean toward `export`.
- **Therefore: `v2` semantic transfer is not established.** The result is
  best characterized as **asymmetric / prior-congruent priming**, not
  successful (bidirectional) semantic transfer — consistent with the
  presence/format-priming mechanism already identified in §7/§8/§10/§16,
  now shown to still dominate under `v2`'s reworded representation, at
  least in the direction that agrees with the model's own prior.
- **Open question this result cannot answer on its own**: whether `C`'s
  total collapse comes from (a) the mere presence of *any* `v2` historical
  block reinforcing the model's existing export lean regardless of its
  content, or (b) the export-specific content adding something beyond what
  block-presence alone would do. B7d.1/B7d.2/§16 already showed (a) is a
  real, dominant effect under `v1`'s bare-label format — B7d.5 (a neutral,
  content-free `v2`-shaped control) is designed to check whether it is
  *also* the dominant effect under `v2`'s reworded format, before any `v3`
  redesign is considered.

No code was modified to produce this result or this write-up.
`render_experience_block_v1`/`_v2` and `ExperienceAwareDelegate` remain
unchanged. All earlier B7d results (§6–§17) are preserved above, unedited.
B7e remains not started.

## 19. B7d.5 — Neutral-Format Control (Prepared, Not Yet Run)

§18's B7d.4 result cannot distinguish two explanations for
`export`-history's 10/10 deterministic collapse toward `export`: (a) `v2`'s
block presence/format alone reinforces whatever the model's existing prior
already favors, regardless of content — the same mechanism already found
under `v1`'s bare-label format (§7–§10) — or (b) `v2`'s presence is roughly
neutral and it is specifically the export-congruent *content* that adds an
extra push in the prior's direction. A neutral, content-free `v2`-shaped
control is needed to separate these, mirroring §10's structure-only control
(Condition E) but applied to `v2`'s format instead of `v1`'s.

**New script, not an extension of any fixed reproducer**:
`experiments/diagnostics/experience_neutral_control.py`. None of
`experience_transfer.py` (fixed B7d.1/B7d.2 reproducer),
`experience_representation.py` (fixed B7d.3 reproducer), or
`experience_semantic_ablation.py` (fixed B7d.4 reproducer) is modified.
`src/dualflow/experience_aware_delegate.py`'s `render_experience_block_v2`
is also **not modified** — the neutral control renderer
(`render_experience_block_v2_neutral`) lives only in the new script, so the
production/research implementation stays frozen regardless of this
experiment's outcome, per instruction.

**Design**: same canonical ambiguous September delegation as §16 Task 1 /
§18. Four conditions:

- **A.** no experience
- **B.** `v2` rendering of a `summarize`-confirmed historical experience
  (identical to §18's condition B)
- **C.** `v2` rendering of an `export`-confirmed historical experience
  (identical to §18's condition C)
- **D.** `v2`-shaped neutral-history control — same header
  (`_EXPERIENCE_HEADER_V2`, imported from `experience_aware_delegate.py`
  rather than retyped, to avoid exactly the kind of byte-for-byte wording
  drift `docs/REPRODUCIBILITY.md` exists to prevent), same "Example N"
  structure, same three content lines, but the two content-bearing lines
  (`Principal clarification`/`Confirmed interpretation`) are replaced with
  text naming no action at all:

  ```
  Principal clarification: The intended operation for this prior task
  was clarified with the Principal.
  Confirmed interpretation: a meaning was confirmed for this prior
  interaction; no specific action preference is indicated here.
  ```

  Verified structurally (dry run): the rendered block contains none of
  `summarize`/`export`/`read`, and its header is byte-identical to
  `render_experience_block_v2`'s.

`N=20`, `repetitions=10`, 4 conditions → 800 calls.

**Primary comparisons** (entropy recorded but secondary, per §11):

```
format_effect            = P(export | neutral-history) - P(export | no-experience)
summarize_content_effect = P(summarize | summarize-history) - P(summarize | neutral-history)
export_content_effect    = P(export | export-history) - P(export | neutral-history)
```

**Interpretation rule, fixed before running**:

- If neutral-history *also* collapses toward `export` (large
  `format_effect`), conclude `v2`'s block presence/format itself still
  strongly primes the existing export prior — the same mechanism found
  under `v1` (§7–§10), now shown to persist under `v2`'s reworded format.
- If neutral-history stays near the no-experience baseline (small
  `format_effect`) but `export`-history still collapses toward `export`
  (large `export_content_effect`), conclude `v2` shows asymmetric,
  prior-congruent content sensitivity — content matters, but only when it
  already agrees with the model's prior; counter-prior (`summarize`)
  transfer still fails.
- Only if `summarize`-history selectively raises `P(summarize)` *and*
  `export`-history selectively raises `P(export)`, both relative to the
  neutral condition, would that be evidence of bidirectional
  semantic-content sensitivity — the bar for treating `v2` as doing real
  semantic transfer.

**Status**: script written and structurally verified (fake-client dry run
including a direct assertion that the neutral block contains no action
words and shares `v2`'s exact header, `--help`, full `pytest -q` regression
unaffected — 318 passed, zero diff on every previously-fixed diagnostic
file and on `experience_aware_delegate.py`) — not yet run against the real
API. No prompt tuning is planned after seeing results, and B7e remains not
started regardless of this control's outcome, per the current gating order
(§14).
