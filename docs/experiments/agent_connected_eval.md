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
| Experience representation redesign (B7d.3) | not started |
| Sequential experience evaluation (B7e) | not started |

The sequential multi-episode experiment (B7e) stays intentionally
postponed. The next step is B7d.3: redesign the experience representation
so it presents ambiguity → clarification → confirmed-meaning as a
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
