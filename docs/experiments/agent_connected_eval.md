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

A fourth lesson was added in §22: **a positive effect relative to the
wrong control can look like content sensitivity when it is actually a
slot-token conjunction.** B7d.5/B7d.5R's counter-prior effect was real and
replicated against a properly-matched neutral control — but B7d.6 showed
it depends on the literal action token and the structured
`Confirmed interpretation` relation being present *together*; neither
alone reproduces it. Injecting historical text directly into the
Delegate's own candidate-generation prompt makes this kind of slot/token
priming difficult to rule out no matter how the injected text is worded —
this is what motivated moving history out of generation and into
verification for `v3` (§22).

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
- **`v2`'s counter-prior effect (§20/§21) is real and replicated, but is a
  slot-token conjunction, not paraphrase-invariant semantic transfer** —
  §22: a properly-matched neutral control did reveal a genuine,
  reproducible counter-prior `summarize` effect (two independent 800-call
  batches, `B>D` 10/10 both times). But B7d.6 found neither the literal
  token `"summarize"` alone (`lexical_effect=0.000`) nor a paraphrased
  version of the confirmed-meaning relation without that token
  (`semantic_without_token_effect=-0.005`) reproduces it — only their
  conjunction does (`interaction contrast +0.225`). Do not describe this as
  "semantic-content transfer is ruled out": B7d.6 tested exactly one
  paraphrase, not the space of all possible phrasings. The precise,
  supported claim is narrower: under the current `v2` representation,
  paraphrased semantic relation without the literal action token did not
  reproduce the observed effect. This is the finding that closed the `v2`
  investigation (§22) in favor of a `v3` architectural redesign.

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
| Neutral-format control (B7d.5) | **complete — real counter-prior content effect found, confounded by ceiling on export side (§20)** |
| Exact replication (B7d.5R) | **complete — counter-prior effect replicated, 10/10 both batches (§21)** |
| Lexical vs. semantic control (B7d.6) | **complete — effect depends on slot+token conjunction, not paraphrase-invariant (§22)** |
| `v2` representation investigation | **closed — moving to `v3` architectural redesign (§22)** |
| `v3` prototype — `HistoricalEvidenceComparator` (opt-in, not wired into runtime) | implemented (§22 review, code) |
| `v3` first pilot (180 calls) | **inconclusive — comparator calibration/semantic-target validity not established, not a verdict on the v3 hypothesis (§23)** |
| `v3` comparator redesign — structured-facet, deterministic (option B) | **implemented, regression-tested (§23 follow-up)** |
| `v3` provenance gate (`confirmed_facets`, structured value ≠ confirmed evidence) | **implemented, regression-tested (§23 follow-up 2)** |
| `v3` provenance correction (ambiguity ≠ confirmation; generation-time target_facets) | **implemented, regression-tested (§23 follow-up 3)** |
| `v3` single-facet clarification (target ≠ confirmed for multi-facet rounds, closed via IG-based single-facet targeting) | **implemented, regression-tested (§23 follow-up 4)** |
| `v3` post-clarification resolution gate (singleton target ≠ actual confirmation) | **implemented, regression-tested (§23 follow-up 5) — provenance chain design complete** |
| `v3` contract smoke test (42 real API calls) | **PASS — full chain held end to end (§24)** |
| B7d-v3 main experiment (H1 ambiguous transfer / H2 explicit preservation, paired frozen-candidate design) | **protocol + harness implemented, regression-tested — not yet run (§24)** |
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
no consistent improvement over no-experience at all. B7d.5/B7d.5R (§20/§21)
then found — twice, in independent 800-call batches — a real, replicated
counter-prior effect against a properly-matched neutral control
(`summarize_content_effect` +0.270 then +0.220, `B>D` 10/10 both times).
B7d.6 (§22) separated the literal token `"summarize"` from the structured
confirmed-meaning relation and found neither alone reproduces the effect
(`lexical_effect=0.000`, `semantic_without_token_effect=-0.005`) — only
their conjunction does (`interaction contrast +0.225`). This is enough
evidence to close the `v2` investigation: the effect depends on a specific
structural-slot + literal-token combination rather than demonstrating
paraphrase-invariant semantic transfer under the current representation.
The next step is a `v3` architectural redesign (reviewed as a proposal
before any implementation), not further `v2` prompt tuning and not B7e. The
original plan for B7d.3 was to redesign the experience
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

> **Status update**: this experiment has since been run, and replicated,
> and followed by a lexical-vs-semantic control (B7d.6) — see §20/§21/§22.
> This section is left unedited below as the original design record, per
> this document's standing policy of not overwriting earlier sections.

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

## 20. B7d.5 — Neutral-Format Control (Results)

### Experiment identity

```
experiment_id:          b7d5_v2_neutral_format_control
git_sha:                 8aa313e42ee1a88679c0caa89d8883ef2fa9e50c
scenario_id:             external_audit_finance
scenario_version:        1.1
delegation_sha256:
  38b8fc2394a7bc9a4d043dd8078ecd6194f3855d710710d1fce23727a4a5fab1
context_sha256:
  a1a5da5674cab40190610056581c30802e13277b70c058eb2317569085233892
model:                   gpt-4o-mini
samples_per_condition:   20
repetitions:             10
total calls:             800
script:                  experiments/diagnostics/experience_neutral_control.py
```

### Aggregate results

| Condition | Mean H | P(read) | P(summarize) | P(export) |
| --- | --- | --- | --- | --- |
| A no experience | 0.478 | 0.00 | 0.115 | 0.885 |
| B `v2` summarize-history | 0.768 | 0.00 | 0.275 | 0.725 |
| C `v2` export-history | 0.000 | 0.00 | 0.000 | 1.000 |
| D `v2` neutral-history | 0.029 | 0.00 | 0.005 | 0.995 |

`B` beats `A` in 9/10 runs (only run 7 lower). `C` and `D` are both
essentially at the `export` ceiling (`P(export)` 1.000 and 0.995), barely
distinguishable from each other — `D` alone (no content at all) already
collapses almost as completely as `C` (real export content) does.

### Primary effects

```
format_effect            = P(export|neutral) - P(export|no-exp)          = 0.995 - 0.885 = +0.110
summarize_content_effect = P(summarize|sumhist) - P(summarize|neutral)   = 0.275 - 0.005 = +0.270
export_content_effect    = P(export|exphist) - P(export|neutral)         = 1.000 - 0.995 = +0.005
```

### Structural checks

Scope preservation 800/800 (`/reports/2026-09/` only, 0 August leakage);
parse failures 0; clarification calls 0; Authority calls 0; `task.truth`
usage 0. Total tokens: 389,000 input / 20,483 output across 800 calls.

### Conclusion — does not fit cleanly into the three pre-registered cases (§19)

- The pre-registered Rule-A trigger ("neutral also strongly collapses
  toward export") is **literally true** — `D`'s `P(export)=0.995` is
  nearly identical to `C`'s `1.000`. Taken alone, this would support "`v2`
  block presence/format itself reinforces the pre-existing export prior."
- But `summarize_content_effect = +0.270` is large and real, and cannot be
  explained by format/presence alone — `D` (format only) sits at
  `P(summarize)=0.005`, actually *below* the raw no-experience baseline
  (`0.115`), while `B` (summarize content, same format) reaches `0.275`.
  This is evidence of genuine content sensitivity in the counter-prior
  direction that the "format explains everything" reading misses.
- `export_content_effect ≈ 0` **cannot be read as "export content doesn't
  matter"** — `D` already sits at `P(export)=0.995`, a near-ceiling value
  that leaves almost no headroom (`0.005`) for any additional content
  effect to show up, regardless of whether one exists. This is a genuine
  measurement confound (ceiling effect), not evidence against content
  sensitivity on the `export` side.
- **Honest summary**: format/presence dominates and likely saturates the
  prior-congruent (`export`) direction (masking any possible content
  effect there); the counter-prior (`summarize`) direction shows a real,
  substantial, format-independent content effect. This does not match any
  single one of the three pre-registered interpretation buckets cleanly —
  reported as such rather than forced into one.

## 21. B7d.5R — Exact Replication

Per instruction, an exact replication (same code, same scenario, same
config, zero changes) was run before drawing any conclusion from a single
800-call batch.

### Experiment identity

```
experiment_id:  b7d5r_v2_neutral_format_control_replication
git_sha:         9da9ab3939abe143d40ab54761452d5bd18f496c
```

Confirmed via `git diff --stat 8aa313e..9da9ab3` on every experiment-relevant
file (the diagnostic script, the scenario file, `experience_aware_delegate.py`)
before running: **zero differences** — the only commit between the two SHAs
added `docs/unexpected_findings/`, which does not touch any code or
scenario path this experiment depends on. Same scenario/delegation/context
hashes, same model, same `N=20`/`repetitions=10`/800 calls.

### B7d.5 vs B7d.5R — direct comparison

| metric | B7d.5 | B7d.5R |
| --- | --- | --- |
| P(summarize\|A) | 0.115 | 0.190 |
| P(summarize\|B) | 0.275 | 0.225 |
| P(summarize\|C) | 0.000 | 0.000 |
| P(summarize\|D) | 0.005 | 0.005 |
| P(export\|D) (ceiling) | 0.995 | 0.995 |
| format_effect | +0.110 | +0.185 |
| **summarize_content_effect (B−D)** | **+0.270** | **+0.220** |
| export_content_effect | +0.005 | +0.005 |
| run-level B > D | 10/10 | 10/10 |

`P(summarize|A)` (raw no-experience baseline) itself varies noticeably
between the two independent batches (0.115 vs 0.190) despite identical
scenario/model/hashes — a reminder that even canonical, fingerprinted
10-run batches carry real sampling variance in absolute terms, and that
**within-batch** comparisons (B vs D, computed from conditions run
contemporaneously) are far more trustworthy than **across-batch** absolute
comparisons. This is exactly why B7d.5/B7d.5R/B7d.6 all run every
condition together in one batch rather than reusing an earlier batch's
numbers for one condition.

### Conclusion: replicated

`B > D` in 10/10 runs in **both** independent 800-call batches, with a
consistent, large effect size (`+0.270` then `+0.220`). Per instruction,
this is **not** called "semantic transfer" — only that the counter-prior
`summarize`-history vs. structurally-matched `neutral`-history effect is a
real, reproducible phenomenon, not a one-batch fluke. What causes it
(structured semantic content vs. the literal token "summarize" vs. their
interaction) is exactly what B7d.6 was designed to separate.

## 22. B7d.6 — Lexical vs. Semantic Control (Results) — v2 Investigation Closed

### Experiment identity

```
experiment_id:          b7d6_lexical_vs_semantic_control
git_sha:                 9da9ab3939abe143d40ab54761452d5bd18f496c
scenario_id:             external_audit_finance
scenario_version:        1.1
delegation_sha256:
  38b8fc2394a7bc9a4d043dd8078ecd6194f3855d710710d1fce23727a4a5fab1
context_sha256:
  a1a5da5674cab40190610056581c30802e13277b70c058eb2317569085233892
model:                   gpt-4o-mini
samples_per_condition:   20
repetitions:             10
total calls:             800
script:                  experiments/diagnostics/experience_lexical_semantic_control.py
```

Zero diff confirmed on every previously-fixed diagnostic file and on
`experience_aware_delegate.py` before running. A pre-run structural check
(`verify_blocks()`) rendered and asserted all four experience blocks BEFORE
any billed API call: `N` contains no `summar`-root token; `L` contains the
literal token `"summarize"` but not on its `Confirmed interpretation` line
(i.e. not presented as the confirmed historical action); `P` contains no
`summar`-root token at all (the structured relation is preserved via
paraphrase — see design below); `S` is byte-identical to
`render_experience_block_v2`'s own real output. All four assertions passed.

### Design (2×2)

| Condition | Structured confirmed-meaning relation | Literal token "summarize" |
| --- | --- | --- |
| N neutral | ✗ | ✗ |
| L lexical-only | ✗ | ✓ (present, but disconnected from the confirmed action) |
| P semantic-paraphrase | ✓ (expressed as "produce a concise account of the report's contents") | ✗ |
| S structured summarize-history (existing `v2` condition, unchanged) | ✓ | ✓ |

### Aggregate results

| Condition | Mean H | P(summarize) | P(export) | top1 |
| --- | --- | --- | --- | --- |
| N neutral | 0.047 | 0.010 | 0.990 | export 10/10 |
| L lexical-only | 0.114 | 0.010 | 0.980 | export 10/10 |
| P semantic-paraphrase | 0.029 | 0.005 | 0.995 | export 10/10 |
| S structured summarize-history | 0.723 | **0.230** | 0.770 | export 10/10 |

### Primary effects and the 2×2 interaction contrast

```
lexical_effect                = P(sum|L) - P(sum|N) =  0.010 - 0.010 =  0.000
semantic_without_token_effect = P(sum|P) - P(sum|N) =  0.005 - 0.010 = -0.005
full_structured_effect        = P(sum|S) - P(sum|N) =  0.230 - 0.010 = +0.220

interaction contrast: S - L - P + N = 0.230 - 0.010 - 0.005 + 0.010 = +0.225
```

The interaction contrast (`+0.225`) is large — nearly the entire
`full_structured_effect` is *not* explained by the sum of the two
individual "arms" (`L` and `P` alone contribute almost nothing; their
combination in `S` produces almost all of it).

### Run-level (10 runs)

| run | N | L | P | S | S−N |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.00 | 0.00 | 0.00 | 0.20 | +0.20 |
| 2 | 0.00 | 0.05 | 0.00 | 0.15 | +0.15 |
| 3 | 0.00 | 0.00 | 0.00 | 0.40 | +0.40 |
| 4 | 0.10 | 0.05 | 0.00 | 0.35 | +0.25 |
| 5 | 0.00 | 0.00 | 0.00 | 0.30 | +0.30 |
| 6 | 0.00 | 0.00 | 0.00 | 0.35 | +0.35 |
| 7 | 0.00 | 0.00 | 0.00 | 0.10 | +0.10 |
| 8 | 0.00 | 0.00 | 0.05 | 0.15 | +0.15 |
| 9 | 0.00 | 0.00 | 0.00 | 0.05 | +0.05 |
| 10 | 0.00 | 0.00 | 0.00 | 0.25 | +0.25 |

`N`, `L`, and `P` are at or near zero in essentially every run (only one
isolated `0.05` sample each in a couple of runs — noise-floor level, out of
20 samples per row). `S` is above `N` in **10/10 runs**, ranging `+0.05` to
`+0.40`.

### Out-of-vocabulary action note

Two samples out of 800 total (2/200 within the `lexical_only` condition
specifically, 0/200 in every other condition) produced `ACTION=prepare` —
echoing the current delegation's own verb — instead of a vocabulary action
(`read`/`summarize`/`export`). This is not a parse failure (the response
parsed successfully as a valid `Interpretation`; `n_llm_calls` stayed at
20 for both affected rows) — it is a vocabulary-adherence lapse, distinct
from the fail-closed exclusion `DelegateAgent.sample_candidates()` already
applies to genuinely unparseable text. It does not change `P(summarize)`
for either affected row and does not affect any of the primary-effect
conclusions above. No rerun was performed or needed.
**Going forward, future diagnostic reports should include an
`out_of_vocabulary_action_rate` field alongside parse-failure counts** —
this run's rate was `2/800` overall (`2/200` within `lexical_only`). This
is a reporting-checklist addition only; the already-completed reproducer
scripts (`experience_transfer.py`/`experience_representation.py`/
`experience_semantic_ablation.py`/`experience_neutral_control.py`/
`experience_lexical_semantic_control.py`) are not modified retroactively.

### Conclusion — cautious wording, not over-generalized

**Under the current `v2` representation, paraphrased semantic relation
without the literal action token did not produce the observed counter-prior
effect. The effect appeared only when the structured `Confirmed
interpretation` relation and the literal `summarize` action token were
present together.** This is *not* the same claim as "semantic-content
transfer is universally ruled out" — `P` tested exactly one paraphrase
("produce a concise account of the report's contents"); it does not
sample the space of all possible semantically-equivalent phrasings, so the
correct scope of this finding is specifically about this representation and
this paraphrase, not a general proof that no wording could ever work.

What is established, precisely: the replicated B7d.5/B7d.5R effect is not
explained by either factor alone (`lexical_effect=0.000`,
`semantic_without_token_effect=-0.005`, both indistinguishable from zero
noise), and is almost entirely explained by their conjunction
(`interaction contrast = +0.225`). This is evidence that the current `v2`
effect is highly dependent on a specific structural slot + literal action
label combination, rather than evidence of paraphrase-invariant semantic
transfer.

### Decision: v2 investigation closed; move to v3 design

This provides sufficient evidence to close the `v2` representation
investigation here rather than continue probing the same injection-based
mechanism. `render_experience_block_v1`/`_v2` and `ExperienceAwareDelegate`
are not modified as a result of this finding — no further `v2` prompt
tuning is planned. The next step is a `v3` architectural redesign proposal
(reviewed before any implementation), motivated directly by this failure
mode: history injected into the Delegate's own candidate-generation prompt
is structurally prone to slot/token priming effects that are difficult to
distinguish from genuine semantic transfer no matter how the injected text
is reworded, because the model's own generation process is what's being
primed. B7e remains not started.

## 23. v3 Pilot — Comparator Contract Audit (Inconclusive)

### Experiment identity

```
experiment_id:   v3_pilot_evidence_comparator
git_sha:          52de6824e6376867a998e0e36f0573a21f4b5b08
scenario_id:      external_audit_finance
scenario_version: 1.1
model:            gpt-4o-mini-2024-07-18 (explicit snapshot, not the alias)
freeze_samples:   20
samples_per_cell: 20
total calls:      180 (freeze 20 + 2 frozen candidates x 4 conditions x 20)
script:           experiments/diagnostics/experience_evidence_comparator.py
```

### Result

| Condition | Candidate | SUPPORT | CONFLICT | IRRELEVANT | UNCERTAIN |
| --- | --- | --- | --- | --- | --- |
| N neutral | summarize | 0.10 | 0.70 | 0.20 | 0.00 |
| N neutral | export | 0.50 | 0.00 | 0.40 | 0.10 |
| L lexical-only | summarize | 0.00 | 0.70 | 0.30 | 0.00 |
| L lexical-only | export | 0.35 | 0.30 | 0.35 | 0.00 |
| P semantic-paraphrase | summarize | 0.00 | 1.00 | 0.00 | 0.00 |
| P semantic-paraphrase | export | 0.65 | 0.30 | 0.05 | 0.00 |
| S structured-summarize | summarize | 0.30 | 0.70 | 0.00 | 0.00 |
| S structured-summarize | export | 0.35 | 0.55 | 0.10 | 0.00 |

Parse/OOV failures: 0/160. Usage: comparator 160 calls / 62,520 input
tokens / 538 output tokens; freeze 20 calls / usage not aggregated by this
diagnostic (a real gap in the script, not backfilled with an estimate).

### Status: **inconclusive for the v3 hypothesis, not a failure of the v3 architecture**

This result does not confirm or refute "moving historical experience from
generation to evidence comparison removes slot-token dependence." It shows
`HistoricalEvidenceComparator`, as currently specified, is not reliably
measuring the intended semantic relation at all — evaluating the v3
hypothesis against this data would not be a valid test of it.

The two strongest signals:
- **`S` (structured evidence, `summarize` confirmed) + the `summarize`
  candidate — the maximally-aligned case — still produced `CONFLICT=0.70`
  vs. `SUPPORT=0.30`.** If the comparator worked as intended, this exact
  case should be the easiest possible `SUPPORT`.
- **`P` (semantic-paraphrase evidence, expressing "summarize" without the
  token) produced `SUPPORT=0.65` for the *`export`* candidate** — the
  paraphrase intended to point toward `summarize` instead favored the
  opposite action.

### Audit findings (no API calls, no prompt changes made)

1. **Exact model-visible prompts** for `S`+`summarize`, `S`+`export`, and
   `P`+`summarize` were rendered directly from the committed code (no
   network call) and inspected. All three share the same
   `_COMPARE_INSTRUCTIONS` and differ only in the candidate block and the
   evidence text, as designed.
2. **Candidate serialization**: `ACTION`/`RESOURCE`/`SCOPE`/`CONDITION`
   are all shown for every candidate. In this scenario `RESOURCE`/`SCOPE`
   are always `file`/`/reports/2026-09/` for both candidates (only
   `ACTION` varies) — but **the historical evidence text explicitly
   contains `"Please prepare the August 2026 financial report..."`**,
   putting `August 2026` and `/reports/2026-09/`/`September 2026` directly
   next to each other in the same prompt. A plausible reading: the model
   may be treating the differing month/report as a whole-episode mismatch
   ("this precedent was about a different report") rather than isolating
   the one facet (`action=summarize`) that was actually meant to transfer.
   This matches the concern raised before running: comparing full
   historical episodes (including non-transferable facets like the old
   month/scope) instead of the single confirmed facet risks exactly this
   kind of false conflict signal.
3. **v2 instructional header contamination confirmed present.** The exact
   header reused from `render_experience_block_v2`/`_neutral`/
   `_lexical_only`/`_semantic_paraphrase` — *"If the current delegation
   explicitly specifies an action, follow the current delegation even if
   it differs from prior behavior."* — appears verbatim inside the
   `Historical evidence:` section of every comparator prompt (all four
   conditions). This sentence was written as an instruction for whoever
   consumes the block to decide an action (originally the Delegate); it
   was never audited for what it means when handed to a *different* task
   (classifying a relation) that treats the whole block as evidence data,
   not as an instruction to itself. Its exact position (first sentence of
   the evidence text, before the `Example 1` block) was confirmed, not
   removed.
4. **`AgentExperience` already stores the confirmed action as structured
   data** — `confirmed_interpretation: Interpretation`
   (`src/dualflow/agent_experience.py`), with `.action`/`.resource`/
   `.scope`/`.condition` accessible directly, no re-parsing needed.
   `confirmed_interpretation.action` already gives the canonical
   transferable facet (`"summarize"`) deterministically. Re-asking an LLM
   to judge whether natural-language evidence "means" `summarize` again
   is not required by the current schema — it was an added step, not
   something the schema forced.

### Design recommendation (not implemented — proposal only)

Two paths were reviewed:

- **A. Keep and fix the natural-language comparator** — remove the v2
  header from evidence rendered for the comparator, and restrict the
  evidence text to the transferable facet only, then re-test.
- **B. Store and compare Principal-confirmed experience as a structured
  semantic facet, deterministically** — since `confirmed_interpretation`
  is already a structured `Interpretation`, `facet=action` comparison
  against a candidate's own `action` field can be a plain equality check,
  with no LLM call and no natural-language rendering of historical
  evidence at all.

**B is the recommended direction.** Re-asking an LLM to interpret a
natural-language rendering of an already-structured, already-verified
fact reintroduces exactly the representation/wording sensitivity this
whole B7d arc (§7-§22) spent effort eliminating. A only makes the
natural-language step more careful; B removes the step's failure surface
entirely for the one facet (`action`) this scenario's ambiguity is about.
Paraphrase invariance under this recommendation moves to a different,
earlier stage of the pipeline — whether a *new* clarification answer like
"produce a concise account of the report's contents" can be structured
into a canonical `confirmed_interpretation` at experience-creation time —
not to the comparison step. Neither path is implemented as of this
section; this is a proposal awaiting review.

Not recorded in `docs/unexpected_findings/`: per instruction, a design/
calibration gap discovered before the intended hypothesis could be tested
is not the kind of "materially changed our interpretation of a result"
finding that directory is for.

### Follow-up: structured-facet comparator (implemented, option B)

Following the audit above, `src/dualflow/experience_evidence.py` was
rewritten (not extended — the natural-language path is removed, this is
revision 3 of the same file, with its full history kept in the module
docstring) to implement recommendation **B**:

- `HistoricalEvidenceComparator.judge()` no longer takes `delegation`/
  `evidence_text` at all — its signature is now exactly `candidate:
  Interpretation, historical: Interpretation, facet: str = "action"`.
  There is no parameter left through which prose (an episode's wording, a
  paraphrase, the reused `v2` header) could enter the comparison.
- The comparison itself reuses `rule_engine.field_match()` unchanged (via
  a small `Fields` adapter that only fills the four raw value fields it
  reads) — no new equality/compatibility rule was invented, per
  instruction. Each facet (`action`/`resource`/`scope`/`condition`) is
  compared completely independently; a scope mismatch cannot affect the
  action relation, structurally (`field_match()` already returns four
  independent booleans — see rule_engine.py's own docstring at that
  function).
- No LLM call happens anywhere in this module now — it is fully
  deterministic, so `HistoricalEvidenceComparator`'s constructor no
  longer takes an `LLMClient`.

**Regression tests** (`tests/test_experience_evidence.py`, rewritten, 18
tests, no LLM/network needed) pin down exactly the contract requested:
`historical=summarize`/`candidate=summarize` → action `SUPPORT`;
`historical=export`/`candidate=summarize` → action `CONFLICT`;
`historical` scope `2026-08` vs. `candidate` scope `2026-09` with both
actions `summarize` → action relation stays `SUPPORT` (the exact §23
failure reproduced and shown fixed) while the *scope* facet, if queried
directly, correctly still reports the mismatch; `judge()`'s signature
audited to contain no free-text parameter at all; the reused `v2` header
and `render_experience_block_v2*` confirmed unreachable from this module
(`hasattr` on the actual namespace); existing independence checks
(`DelegateAgent`/`Budget`/`check_authority`/`AuthorityVerifierAgent`/
`AgentDelegationRuntime`) re-verified against the rewritten module.
All 18 pass; full suite 351 → 336 (the 33 old LLM-based tests replaced by
18 new ones, net −15, everything else unchanged).

**`experiments/diagnostics/experience_evidence_comparator.py` (the N/L/P/S
diagnostic from the first pilot) is now stale** — it still calls
`judge(delegation=..., candidate=..., evidence_text=...)`, which raises
`TypeError` against the new signature (confirmed, not fixed). It is not
rewritten in this pass, per instruction (no new diagnostic, no N/L/P/S
re-run yet). A minimal next diagnostic, when authorized, would freeze one
candidate set exactly as before, retrieve the one verified historical
experience, and call the new `judge()` directly for each candidate — no
API calls would be needed at all for the comparison step itself (it is
deterministic), so the only real-API cost left would be the one
`sample_candidates()` freeze call. This turns the "does structured
comparison remove slot-token dependence" question into something that no
longer needs a paid experiment to answer for the comparison step in
isolation — what still needs real-API validation is the *upstream* step
questioned in the design recommendation above: whether a new paraphrase
like "produce a concise account of the report's contents" can be reliably
structured into `confirmed_interpretation.action = "summarize"` at
experience-creation time.

Progression preserved, not overwritten: **old natural-language comparator
diagnostic was inconclusive (§23 top) → contract audit identified two
contaminants (§23 audit findings) → structured-facet comparator
implemented (this subsection), removing the failure surface the audit
found rather than tuning around it.**

### Follow-up 2: structured value ≠ Principal-confirmed evidence (provenance gate added)

Before any API validation of the structured-facet comparator, a second
gap was found on review: **a structured value existing in
`confirmed_interpretation` is not the same as the Principal having
actually confirmed that facet.** In this pilot scenario, `resource`/
`scope` are always fixed by the grounded environment (B6.1) — every real
clarification round only ever targets `action` — so
`confirmed_interpretation.scope` (e.g. the historical episode's
`2026-08`) was just whatever the environment happened to fix it to, not
a reusable fact the Principal confirmed. The revision-3 comparator would
still let a `scope` query silently become `CONFLICT` evidence, moving the
exact §23 whole-episode contamination one layer down rather than removing
it.

**Fix — `AgentExperience` gained a `confirmed_facets: frozenset[str]`
field** (`src/dualflow/agent_experience.py`, default `frozenset()` for
backward compatibility with the two existing direct-construction call
sites, `experiments/agent_smoke.py`'s `EXAMPLE_PRIOR_EXPERIENCE` and
`experiments/diagnostics/experience_transfer.py`'s `make_experience()` —
neither touched, neither reads this field, so both are unaffected).
`build_verified_experience()` now computes it automatically: a facet
counts as confirmed only if it actually varied across
`pre_distribution`'s candidates before clarification — the only evidence
the Delegate was genuinely uncertain about it, since
`post_distribution.entropy` is already required to have converged
(existing gate, unchanged). A facet that never varied pre-clarification
was never in question, regardless of what value ends up in
`confirmed_interpretation`.

`HistoricalEvidenceComparator.judge()` (revision 4) now takes the whole
`AgentExperience` (not a bare `Interpretation`) and checks
`experience.confirmed_facets` before comparing: a facet not in
`confirmed_facets` returns `IRRELEVANT` unconditionally — even if its
value happens to coincide with the candidate's. `compare_facets()` itself
stays a pure, provenance-agnostic value comparison; the gate lives one
layer up, in `judge()`.

**New tests** (`tests/test_experience_evidence.py`, rewritten again, 23
tests; `tests/test_agent_experience.py`, +3 tests) lock in exactly the
requested contract: action-only-confirmed experience supports/conflicts
correctly on `action` while resource/scope/condition all read
`IRRELEVANT`; the August-scope-vs-September-scope pair with only `action`
confirmed no longer produces a spurious scope `CONFLICT` (the exact
failure mode reproduced and shown fixed) while directly querying `scope`
on that same pair still correctly returns `IRRELEVANT`, not `SUPPORT` or
`CONFLICT`, absent provenance; scope activates normally
(`SUPPORT`/`CONFLICT`) once it actually is in `confirmed_facets`; two
coincidentally-equal empty values (`""`/`frozenset()`) do not produce
`SUPPORT` when their facet lacks provenance; provenance for one facet does
not leak into another's judgment; `build_verified_experience()`'s
automatic derivation is tested directly (single-facet, multi-facet, and
zero-facet-variance cases). All pass; full suite 336 → 344 (+8).

This does not reverse the option-B decision — it completes it. Full
progression: **natural-language comparator → contamination found (§23) →
deterministic structured-facet comparator → structured value alone found
insufficient → facet-level confirmation provenance added.** The next
minimal experiment described in Follow-up 1 above is accordingly revised:
it is still true that the comparison step itself needs no API calls, but
it now also depends on `confirmed_facets` being populated correctly by
`build_verified_experience()` for the historical experience used — which
this section's regression tests already exercise without any real
clarification round. No API calls, no candidate freeze, no N/L/P/S re-run,
no runtime integration, and no B7e were performed in this follow-up.

### Follow-up 3: ambiguity provenance ≠ confirmation provenance

Follow-up 2's `confirmed_facets` had its own gap, found on review before
any API validation: it was computed by
`_facets_with_pre_clarification_ambiguity()` from which facets *varied* in
`pre_distribution` — but **a facet being ambiguous before clarification is
not the same as the Principal having actually been asked about, or having
actually confirmed, that facet.** If a future scenario had `pre_
distribution` vary in both `action` and `scope` simultaneously, but the
free-text clarifying question (generated by `DelegateAgent.
ask_clarification()`, which lets the model choose freely what to ask
about) only actually addressed `action`, the old computation would still
mark `scope` as "confirmed" — reintroducing exactly the whole-episode
contamination Follow-up 1 removed, just moved to a different derivation
point. The precise, correct name for what Follow-up 2 computed was
`ambiguous_facets_before_clarification`, not `confirmed_facets`.

**Audit of the clarification-generation path** (`clarification.py`'s
`ClarifyingDelegate.resolve()` → `DelegateAgent.ask_clarification()` →
`PrincipalAgent.answer_clarification()`) found **no existing structural
signal for which facet a question actually targets**:
`ask_clarification()`'s prompt shows the model every candidate's full
`Interpretation` and asks it to "write a single, direct clarifying
question... that would resolve this specific uncertainty" — entirely the
model's free choice, with nothing recorded about which field(s) it chose
to address. `ClarificationQuestion`/`PrincipalClarification` are both
plain free-text (`question: str`/`answer: str`) with no facet metadata.
The only structural (non-text) signal available anywhere in the pipeline
is exactly the pre-distribution variance Follow-up 2 already used — which
is why Follow-up 2 conflated the two concepts: it was the only signal
that existed at the time.

**Fix — move the same computation earlier and use it as a generation
constraint, not a post-hoc label.** `clarification.py` gains
`_facets_that_varied(belief) -> frozenset[str]` (the same per-facet
variance check, now computed *before* the question is generated, in
`ClarifyingDelegate.resolve()`). This is passed to `DelegateAgent.
ask_clarification()` as a new `target_facets: frozenset[str] = frozenset()`
parameter (default empty — omitting it reproduces the exact prior
prompt/behavior byte-for-byte, confirmed by test). When non-empty, `_render_
clarify_input()` appends one line naming exactly which fields are "actually
uncertain (ask only about these)" — turning "which facet is this question
about" from an unconstrained model choice into an explicit input
constraint. `ClarificationQuestion` gains a `target_facets` field carrying
this value through. `build_verified_experience()` (`agent_experience.py`)
no longer recomputes anything from `pre_distribution` — it simply copies
`result.question.target_facets` into `AgentExperience.confirmed_facets`.

**This does change `ask_clarification()`'s real model-visible prompt** for
any future real clarification round where `ClarifyingDelegate.resolve()`
finds genuine pre-distribution variance (which is every real clarification
round recorded so far — B7b's real pilot always had exactly `action`
varying). The previously-recorded B7b real-API pilot numbers
(§4/agent_connected_eval.md) were produced under the prior, unconstrained
prompt and are unaffected (nothing here is retroactive), but any future
real run of `ClarifyingDelegate.resolve()` will use this new,
facet-constrained prompt — a deliberate, documented revision (B7b
clarification prompt, revision 2), not a silent drift.

**Known, accepted limitation, not solved here**: this constrains the
*question* to the target facets structurally, but does not verify that
the Principal's free-text *answer* actually engaged with them — doing so
would require re-interpreting the answer text, which was explicitly
out of scope (no answer-text heuristics, no keyword matching, no LLM
re-analysis of the answer). The correctness of `confirmed_facets` past
this point rests on the prompt constraint being followed by the model
generating the question, not on independently verifying the answer.

**Tests**: `tests/test_delegate_agent.py` (+3) confirm omitting
`target_facets` reproduces the byte-identical prior prompt, that supplying
it adds the constraint line and is carried onto the result, and the
default is empty. `tests/test_clarification.py` (+5, new `TestTargetFacets`)
confirm single- and multi-facet variance produce the correct
`target_facets`, that the value actually reaches `ask_clarification()`'s
prompt, that no clarification means no question to carry it, and —
directly addressing "no answer-text heuristics" — that changing the
Principal's answer wording does not change the computed `target_facets`
(it is computed purely from `pre_distribution`'s structured candidates,
before the answer even exists). `tests/test_agent_experience.py`'s
provenance tests are rewritten to test pure pass-through, plus one
regression test reproducing Follow-up 2's exact failure shape (`pre_
distribution` varies in both `action` and `scope`, but `question.
target_facets` is only `{"action"}`) and confirming `confirmed_facets`
follows the question's target, not the raw pre-distribution variance.
All pass; full suite 344 → 353 (+9).

Progression as of Follow-up 3: **natural-language comparator →
contamination found (§23) → deterministic structured-facet comparator →
structured value alone found insufficient → ambiguity-based provenance
added → ambiguity provenance found insufficient (confirmed ≠ ambiguous) →
explicit generation-time confirmation provenance added.** (This was called
"complete" at the time — Follow-up 4 below found one more gap in it,
before any API validation, so the label was premature; left unedited here
per this document's policy.)

### Follow-up 4: target facets ≠ confirmed facets (single-facet clarification)

Follow-up 3's `target_facets` closed the "ambiguity vs. confirmation" gap
for the single-facet case, but a subtler gap remained, found on review
before any API validation: `target_facets` could legitimately hold
**more than one facet** (whenever more than one facet varies together in
`pre_distribution`), while a clarification round is still exactly **one**
free-text question and **one** free-text answer. `build_verified_
experience()` copied the whole `target_facets` set into `confirmed_facets`
regardless — so if `action` and `scope` both varied and both became
`target_facets`, but the Principal's one-sentence answer only clearly
addressed `action`, `scope` would still be silently promoted to
"confirmed" evidence with no verification that it was actually addressed.
The precise distinction needed, restated: `target_facets` = "the question
was structurally constrained to ask about these facets"; `confirmed_
facets` = "these facets are safe to use as transferable historical
evidence" — the two are only trustworthy as identical when there is
exactly one facet in play, because then there is no "did the one answer
cover *all* of several targeted facets" ambiguity left to resolve.

**Fix: make every clarification round target exactly one facet, chosen by
information gain — reusing existing legacy logic, not inventing a new
one.** `src/dualflow/semantic.py` already has exactly this machinery from
the Phase A controlled framework: `Question`/`candidate_questions()`/
`conditional_entropy()`/`information_gain()`/`select_question()` — pick
the single dimension whose conditional entropy reduction (mutual
information with the answer) is largest, with an optional repeat penalty
(`asked: Counter`, unused here since B7b is single-round). `clarification.py`
gains `_select_target_facet(belief)`, calling `select_question(belief,
Counter())` and returning `question.dimension` (or `None` in the
mathematically-unreachable case where no facet has positive information
gain while `pre.entropy` already exceeds a positive `entropy_threshold` —
handled defensively as "confirm nothing," not asserted-and-crashed).
`ClarifyingDelegate.resolve()` now passes `target_facets = frozenset({facet})`
(or `frozenset()`) to `ask_clarification()` instead of the old
multi-facet `_facets_that_varied()` result — that function is kept, but
repurposed as **diagnostic-only**: `ClarificationResult` gains a new
`varied_facets: frozenset[str] = frozenset()` field carrying it, explicitly
documented as *not* usable for evidence eligibility (`varied_facets ⊇
target_facets` always holds; only `target_facets`, via `confirmed_facets`,
is ever evidence-eligible). `agent_experience.py`/`experience_evidence.py`
are unchanged — `confirmed_facets` still just copies `question.
target_facets`, which is now always a singleton-or-empty set, closing the
gap without any new gating logic needed downstream.

**`PrincipalAgent.answer_clarification()` audited, not modified.** It
takes `question: str` (the free-text question only, not the
`ClarificationQuestion` object) — there is no structural channel by which
`target_facets` could reach the Principal's answer-generation prompt today.
Threading it through (as an additional labeling/prompt-hint parameter, not
answer-text analysis) was considered and is a legitimate future
strengthening, but is **not implemented here**: once clarification targets
exactly one facet, the "did the answer cover every targeted facet"
ambiguity that motivated this whole follow-up no longer applies (there is
only one facet, and the one free-text answer is definitionally in response
to a question already constrained to it) — so this additional step is not
required to close the gap, only a possible future reinforcement. No
answer-text heuristic (keyword matching, LLM re-classification, wording
analysis) was added anywhere, per instruction.

**Tests**: `tests/test_clarification.py`'s `TestTargetFacets` — the
previous multi-facet test (which had asserted `target_facets ==
{"action", "scope"}`, now the *wrong* expectation) is replaced by
`test_multi_facet_variance_selects_exactly_one_target_facet` (same
perfectly-co-varying `action`+`scope` fixture; `varied_facets ==
{"action", "scope"}` while `target_facets` is a strict, length-1 subset of
it) and a new end-to-end `test_unselected_varied_facet_is_not_confirmed_
end_to_end` — the exact failure case requested: `varied_facets={action,
scope}`, only `action` selected as `target_facets`, carried through
`build_verified_experience()` into `AgentExperience.confirmed_facets ==
{"action"}` (`"scope" not in confirmed_facets`, asserted directly), and
then through `HistoricalEvidenceComparator.judge()`: querying `scope`
returns `IRRELEVANT`, querying `action` returns `SUPPORT` — the full chain
from candidate variance to comparator behavior, in one test. All 17 tests
in the file pass (16 → 17, one replaced + one added). Full suite
353 → 354.

Progression as of Follow-up 4: **natural-language comparator →
contamination found → deterministic structured-facet comparator →
structured value alone insufficient → ambiguity-based provenance →
ambiguity ≠ confirmation → generation-time target provenance → target ≠
confirmed when multi-facet → single-facet-per-round clarification (reusing
existing legacy IG machinery), closing the gap without new heuristics.**
(Again called "complete" at the time; Follow-up 5 below found one final,
narrower gap before any API validation — left unedited here, same policy
as Follow-up 3.)

### Follow-up 5: singleton target ≠ actual confirmation (post-clarification resolution gate)

Follow-up 4's singleton `target_facets` guarantees only **"at most one
facet can be a confirmation candidate per round"** — it does not by
itself guarantee the Principal's free-text answer was clear, on-topic, or
unambiguous. A vague, evasive, or off-topic answer is not structurally
ruled out by singleton targeting alone. The precise statement of what
remained missing: singleton targeting bounds *how many* facets could be
confirmed, not *whether* the one targeted facet actually *was*.

**Audit of the post-clarification path**
(target facet selection → question → answer → `post_distribution` →
`final_interpretation` → `build_verified_experience()`) found an existing,
already-computed structural signal that directly answers this, with no
new heuristic and no answer-text analysis: **`ClarificationResult.
post_distribution`** — the *already re-sampled* candidate distribution
taken *after* the Principal's answer was folded into context. If the
target facet's value is still split across more than one distinct value
among `post_distribution`'s candidates, the clarification round did not
actually resolve it, regardless of what `target_facets` said should have
been asked. This is exactly the same primitive already used for
`varied_facets` (`_facets_that_varied()`, "how many distinct values does
this facet have among these candidates") — reused here in single-facet
form on the *post* distribution instead of the *pre* distribution.

One subtlety found: `build_verified_experience()`'s **existing**
`max_verified_entropy` gate (default `0.0`) already requires the *joint*
`post_distribution` entropy to have converged before returning anything at
all — at the default, this means every facet (including the target one)
has necessarily collapsed to a single value, making the new per-facet
check redundant *at the default threshold*. It becomes load-bearing
specifically when a caller configures a **looser** `max_verified_entropy`
(already a supported, tested code path —
`test_high_post_entropy_accepted_with_looser_configured_threshold`): the
joint-entropy gate can pass while the target facet specifically still
shows real variance, and only the new per-facet check catches that case.

**Fix**: `agent_experience.py` gains `_facet_resolved(distribution, facet)`
(counts distinct values for one facet across a distribution's candidates —
the same low-level operation `_facets_that_varied()` already performs, not
a new one). `build_verified_experience()` now computes `confirmed_facets`
as the intersection of `result.question.target_facets` with the facets
that pass `_facet_resolved(result.post_distribution, facet)` — not a blind
copy of `target_facets` anymore. A facet that was targeted but never
resolved post-clarification is dropped, `confirmed_facets` becomes
`frozenset()` in that case rather than falsely reporting the target facet
as confirmed. `clarification.py`, `delegate_agent.py`, and
`experience_evidence.py` are all unchanged — the fix is entirely inside
`build_verified_experience()`, one filtering step added to an existing
function.

**`PrincipalAgent.answer_clarification()` audited again, still not
modified** — same conclusion as Follow-up 4: the post-clarification
resolution signal already available (`post_distribution`) is sufficient to
close this gap without needing any change to answer generation or any
answer-text analysis.

**Tests** (`tests/test_agent_experience.py`, `TestConfirmedFacetsProvenance`
rewritten, 6 tests): the default-threshold case (`confirmed_facets ==
target_facets` when the joint gate already forces full convergence); the
requested failure case with a deliberately loosened
`max_verified_entropy` where the target facet (`action`) still shows two
distinct post-clarification values — `confirmed_facets == frozenset()`,
not `{"action"}`; a non-target facet (`scope`) that happens to trivially
"resolve" (because `post_distribution` collapsed to one candidate) is
still never confirmed, because it was never in `target_facets` to begin
with (the Follow-up 4 subset invariant re-verified under the new gate);
and a defensive test confirming `build_verified_experience()` correctly
intersects even if (hypothetically) more than one target facet were ever
passed in. All pass; full suite 354 → 356 (+2).

Progression, now complete for this arc: **natural-language comparator →
contamination found → deterministic structured-facet comparator →
structured value alone insufficient → ambiguity-based provenance →
ambiguity ≠ confirmation → generation-time target provenance → target ≠
confirmed when multi-facet → single-facet-per-round clarification → a
singleton target still ≠ actual confirmation → post-clarification
resolution gate added, using only an already-computed distribution, no new
heuristic and no answer-text analysis.** No API calls, no candidate
freeze, no N/L/P/S re-run, no runtime integration, and no B7e were
performed in this follow-up. This closes the provenance chain design work
for now — the next step, when authorized, is real-API validation.

## 24. v3 Contract Smoke Test (PASS) and B7d-v3 Main Experiment Protocol (Prepared, Not Yet Run)

### Smoke test result: PASS

A minimal, single-run smoke test
(`experiments/diagnostics/experience_provenance_smoke.py`, git_sha
`b80fb15e5e2fa5017ba364fe94ef18d568e21b77`) confirmed the full provenance
chain (IG target selection → singleton clarifying question →
post-clarification resolution → `confirmed_facets` →
`HistoricalEvidenceComparator`) holds end to end against real
API-generated data, at 42 real API calls total:

```
varied_facets: ['action']
target_facets: ['action']
post_distribution: entropy=0.000, action values=['summarize'] (10/10 converged)
confirmed_facets: ['action']
candidate(summarize).action vs. historical(summarize).action -> SUPPORT   (expected)
candidate(export).action    vs. historical(summarize).action -> CONFLICT (expected)
candidate(summarize).scope  vs. historical (unconfirmed facet) -> IRRELEVANT (expected)
failure stage: none
```

Historical (August) and current (September) scopes genuinely differed in
this real run, and the `action`/`scope` isolation held on real data, not
just synthetic `Interpretation` objects. **Confirmed: PASS.** No code was
modified as a result of this run.

### Why the next experiment is not another N/L/P/S run

`HistoricalEvidenceComparator` is now fully deterministic (§23 follow-up
2). Re-running the same `summarize == summarize -> SUPPORT` check hundreds
of times across natural-language rendering conditions would produce no new
research information — that mechanism-level question (does history-block
presence/format prime the model independent of content) was already
answered by B7d.5/B7d.6 (§20-22). The open question now is one level
downstream: **once a relation exists, does actually consuming it produce a
useful semantic decision** — improving an ambiguous case while never
overriding an explicit one?

### Hypotheses

- **H1 (ambiguous transfer)**: when the current delegation's action is
  genuinely ambiguous, does Principal-confirmed historical action evidence
  move the semantic decision toward the historically confirmed action?
- **H2 (explicit-change preservation)**: when the current delegation
  explicitly specifies a *different* action, does historical evidence fail
  to override it?

### Evidence-consumption contract (fixed before any API run)

Implemented in `src/dualflow/experience_decision.py`
(`FrozenCandidateEvidenceHarness`), opt-in, not wired into
`AgentDelegationRuntime`:

1. **Stability gate first, using the existing entropy threshold** (the
   same one `clarification.ClarifyingDelegate` already uses, no new
   classifier) — if the current candidate distribution is already stable
   (`entropy <= entropy_threshold`), historical evidence is never even
   consulted. **Precise scope of this guarantee**: the gate does not
   recognize "this is an explicit instruction" as a natural-language
   property — it only checks whether the *already-sampled* current
   distribution happens to satisfy the existing stability criterion. When
   it does, history cannot override it, by construction (a gate on
   *whether to look*, not a rule applied after looking). But if real
   sampling on an explicit-change delegation turns out unstable (entropy >
   `entropy_threshold`) despite the instruction being explicit in the
   text, the harness *will* proceed to consult history — and if that
   consultation then changes the decision away from the explicit
   instruction, that is a real, observed F4 (stale-history override), not
   automatically a contract bug. This distinction is exactly what the main
   experiment's H2 is designed to measure empirically, not something
   assumed true by the code.
2. **Eligibility gate, using existing provenance** — if the facet is not
   in `experience.confirmed_facets`, evidence is not consulted (F1
   otherwise).
3. **Support gate, using the existing deterministic comparator** — among
   the *already-generated* current candidates (never re-sampled), the
   distinct values of the facet are compared against the historical
   experience. No `SUPPORT` among them means evidence was consulted but
   not applicable (F2).
4. **Facet-level resolution only** — if exactly one facet value is
   `SUPPORT`ed, that facet's value is resolved. The decision
   interpretation, if uniquely determinable, is one of the *current*
   candidates that already carries that value — never a synthesized
   `Interpretation` built from historical resource/scope/condition. If
   more than one distinct current `Interpretation` shares the resolved
   value, no single point decision is picked (structural prevention of F5,
   cross-facet amplification — regression-tested directly: a resolved
   decision's `scope` always comes from the current candidate, never from
   the historical experience's scope, even when they differ).

No new confidence weighting, no new threshold, no new LLM call anywhere in
this module.

### Design: paired, frozen-candidate, four conditions

Candidate generation and the v3 intervention are fully separated — for
each task, candidates are sampled **exactly once** per run (no history in
the generation prompt, the same clean B7a/B6 path), and that one frozen
distribution is reused for both the baseline and the v3 condition. This is
what removes B7d.5/B7d.6's history-block presence/format priming from the
comparison entirely: nothing can differ between a baseline and its v3
pair except what the harness does with an already-fixed distribution.

| Condition | Task | Historical evidence |
| --- | --- | --- |
| A | Ambiguous (canonical September delegation) | none consulted |
| B | Ambiguous (same frozen candidates as A) | confirmed_facets={"action"}, action="summarize" |
| C | Explicit export ("This time, export the September 2026 financial report to the external auditor.", reused from B7d.3/B7d.4/B7d.6's `TASK2_DELEGATION`) | none consulted |
| D | Explicit export (same frozen candidates as C) | same as B |

No export-history/read-history arms this round — those conditions already
characterized the old natural-language mechanism (§20-22); re-adding them
here would blur what this experiment isolates.

Historical evidence (identical, fixed, reused across every run — exactly
one verified experience, matching the established B7c/B7d convention):
built from `experience_transfer.make_experience()` (fixed reproducer,
unmodified) with only `confirmed_facets` overlaid via
`dataclasses.replace()` (that function predates the v3 provenance work and
defaults the field to `frozenset()`).

### Primary / secondary metrics

- **H1 primary**: `P(final decision == "summarize")`, baseline (A) vs. v3
  (B), paired per run. Also recorded: count of runs resolved by evidence /
  no support / no eligible evidence / stable-baseline.
- **H2 primary**: `P(final decision == "export")`, baseline (C) vs. v3
  (D), paired per run. Also recorded: **any** historical-override
  occurrence (D's decision ≠ "export") — flagged individually as an F4
  failure, never averaged away.
- Secondary: entropy, clarification-avoided/required (not applicable here
  since candidates are frozen pre-generated, no live clarification in this
  harness), API calls, token usage.
- Authority safety is explicitly out of scope (B7e territory) and not
  mixed into this result.

### Failure taxonomy (fixed before running)

| Code | Meaning |
| --- | --- |
| F1 | evidence unavailable — eligible historical facet absent |
| F2 | support absent — eligible history exists, no current candidate matches |
| F3 | wrong transfer — ambiguous case, evidence applied, decision semantically worse than baseline (interpretive; assessed when analyzing results, not a stage the harness emits) |
| F4 | stale-history override — explicit current instruction overridden by history |
| F5 | cross-facet amplification — action evidence changes an unconfirmed facet's decision (structurally prevented, regression-tested) |

### API budget (candidate generation only — comparator makes zero calls)

```
2 tasks × N=20 samples × 5 runs = 200 real API calls total
comparator / v3 decision: 0 additional calls (fully deterministic)
```

### Reproducibility

`experiments/diagnostics/experience_decision_experiment.py --output <path>`
saves one JSON payload: `identity` (scenario id/version, model, delegation/
context SHA256 for both tasks, N, repetitions, historical
`confirmed_facets`/confirmed action), `rows` (one entry per
task×condition×run: belief-by-action, entropy, baseline/final decision,
resolved value, stage, per-value comparator relations, token/call counts),
and `summary` (the H1/H2 aggregates above). Raw output stays in the
scratchpad only, per standing convention.

### Status

Implemented and structurally verified: `--help` runs cleanly, a
fake-client dry run confirmed correct end-to-end wiring (paired rows,
correct summary aggregation), `src/dualflow/experience_decision.py` has
its own 15 regression tests (stability/eligibility/support gates,
facet-only resolution, cross-facet non-amplification, explicit-instruction
preservation), full suite 356 → 371 passed. **Not yet run against the real
API.** No N/L/P/S re-run, no runtime integration, no B7e, no new
heuristic, no prompt tuning were performed to produce this protocol or
harness.
