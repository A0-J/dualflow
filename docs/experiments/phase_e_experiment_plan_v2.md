# Experiment Plan v2 — E1/E2/E3 Redesign

> Status: **design only, 0 API calls spent**. This document is the concrete,
> executable plan requested after the decision to redesign the paper's
> primary experiments around DualFlow's actual architectural claim
> (independent semantic/authority verification) rather than the Principal-
> restatement-sampling question Phase 2C–3D drifted toward. It covers task
> counts, prompt templates, arms, API budget, and what happens to the
> existing frozen work. Nothing here has been executed; the one thing
> that needs a real-API go-ahead is flagged explicitly in §3.

---

## 0. What changes, and why

Current title/contribution: *"Independent Semantic and Authority
Verification for Agent-to-Agent Delegation."* Phase 2C→3D's actual
empirical arc tested a narrower question (how many Principal restatements,
what entropy threshold) on one scenario (financial audit). That work is
real and correct, but it is evidence for a *sub-question* of the paper's
actual claim, not the claim itself. The redesign puts the architecture
claim under direct test (E1), keeps the semantic-verification-arm question
as a secondary, better-controlled experiment (E2, absorbing Phase 3C/3D's
shared-sample fix as E2's native design rather than a later correction),
and keeps sampling/threshold/model sensitivity as an ablation (E3) that
reuses E2's data at zero extra cost for the first two factors.

**Nothing existing is deleted.** Phase 2C Final through 3D stay exactly as
documented in `phase3c_methods_for_paper.md`/`agent_connected_eval.md` —
reframed, in the new paper, as the pilot evidence that motivated E1/E2's
design (§6 below), not as the headline results.

---

## 1. E1 — Does DualFlow actually separate the two failure modes?

### 1.1 Design

A 2×2 factorial crossing semantic correctness with authority validity:

| | Authority allowed | Authority exceeded (no_grant) |
|---|---|---|
| **Semantic correct** | C1: expect EXECUTE | C3: expect REJECT (Authority) |
| **Semantic wrong** | C2: expect REJECT (Semantic) | C4: expect REJECT (both) |

Evaluated under 4 verification configurations (arms), computed as four
different **fusion rules applied post-hoc to the same collected episode**
(see §1.3 — this is why E1 costs almost nothing extra):

- **No-verification**: always EXECUTE.
- **Semantic-only**: EXECUTE iff `semantic_confirmed`.
- **Authority-only**: EXECUTE iff `authority_allowed`.
- **DualFlow**: the existing fusion rule (`semantic_confirmed ∧
  authority_allowed ∧ principal_match`).

Expected pattern, if the architecture's claim is correct: Semantic-only
passes C3 (misses the authority violation); Authority-only passes C2
(misses the semantic error); DualFlow is the only arm that rejects both
C2 and C4 *and* still executes C1. That contrast — not any single arm's
raw accuracy — is the headline result:

> Semantic verification and authority verification protect against
> distinct failure modes; neither alone subsumes the other.

### 1.2 Task set

4 domains (Finance/HR/Security/Research, same read/summarize/export
mechanism, different vocabulary — `experiments/domain_benchmark.py`,
already written and verified) × 4 conditions × 5 wording variants = 80
base tasks.

**C1/C3 are guaranteed by construction** — built and verified today, 0 API
calls: `check_authority()` run directly against all 40 C1/C3 tasks
classifies every single one exactly as expected (`allowed=True` for C1,
`allowed=False, failure_kind=no_grant` for C3). This does not depend on
any model's behavior.

**C2/C4 are NOT guaranteed by construction** — "the Delegate is
semantically wrong" can only be elicited by a goal wording a real Delegate
LLM actually misreads, not assumed from wording alone. This is exactly
what Phase 2C-P1/P2 already discovered the hard way
(`confident_semantic_misread` was a differently-framed task, renamed only
after calibration showed it reliably misfires). `CANDIDATE_C2_C4` in
`domain_benchmark.py` holds one candidate wording per domain per
condition; **§3 below is the calibration pilot that must run before these
40 tasks can be trusted**, mirroring Phase 2C-P1's own process exactly.

### 1.3 Why E1 costs almost nothing extra

All four E1 arms are computable from the fields a single real episode
already produces (`semantic_confirmed`, `authority_allowed`,
`principal_match`, final `decision`) — this is literally how the existing
Arm A/B/C comparison already works (§5.2 of the frozen methods doc: "세
Arm 모두 같은 하나의 질문에 답한다"). E1 adds **0 new API calls** beyond
what collecting one real episode per task already costs (1 Delegate call
+ 1 Semantic Verifier call + ~0–1 Authority call, no sampling needed for
E1 alone, since E1 does not test the Principal-restatement mechanism).

---

## 2. E2 — How should the semantic verifier handle ambiguity?

### 2.1 Design (absorbs Phase 3C + the Phase 3D shared-sample fix natively)

Three arms, **one shared sample bank per episode** (not drawn separately
per arm — this was Phase 3D's correction to Phase 3C's original design;
here it is the design from the start, so there is no later confound to
discover and fix):

```
Principal sample bank (n=20 restate_intent() calls, drawn once)
  r1 r2 r3 ... r20
        │
        ├─ Arm A (Single)   = r1 only
        ├─ Arm B (Repeated) = majority over r1..r20, whole-interpretation
        └─ Arm C (Grounded) = facet-wise entropy over r1..r20, confirmed-
                              facet-only blocking
```

This is **not new harness code** — `experiments/phase3d_shared_sample_
replication.py` (collection) and `experiments/phase3d_analysis.py`
(Arm A/B/C recomputation from the raw bank, arbitrary `n`/threshold) already
implement exactly this design, reusing `dualflow.intent_anchor`'s
`sample_principal_intents()`/facet-entropy machinery unmodified. The only
change needed is parametrizing `TARGET_TASK_NAMES` (currently hardcoded to
the old `confident_semantic_misread`/`condition_violation`) to accept the
new domain task set instead.

### 2.2 Task set

Only the tasks where the three arms can actually disagree are informative
— per the existing methods doc's own finding (§5.2: 44/49 episodes gave
zero information because all arms agreed), E2 should run on the
**ambiguity-relevant subset**: the C2/C4 tasks (where semantic uncertainty
is the point) once calibrated in §3, not all 80 E1 tasks. C1/C3's clear
wordings would mostly produce the same ceiling-effect non-information E2
hit before.

### 2.3 Metrics

Unsafe execution rate, false rejection rate (both defined exactly as in
the frozen methods doc) — no new metrics.

---

## 3. Calibration pilot (the one thing that needs a real-API go-ahead now)

Before E1's C2/C4 or E2 can run, the 8 `CANDIDATE_C2_C4` wordings (2 per
domain × 4 domains) need to be checked the same way `confident_semantic_
misread` was originally calibrated: run each candidate wording through a
small number of real, independent Delegate proposals and check whether it
reliably produces the intended wrong interpretation (not by assumption).

**Design**: for each of the 8 candidates, 5 independent real Delegate
calls (`DelegateAgent.propose()`), classified against each domain's
`ideal` (already defined in `domain_benchmark.py`, never exposed to the
Delegate). A candidate "passes" calibration if it misreads consistently
(a high, stable misread rate across the 5 draws) in the specific wrong
direction `CANDIDATE_C2_C4` was designed to elicit (e.g., proposing `read`
instead of `export`) — mirroring Phase 2C-P1 Pilot-A's own pass criterion.
Candidates that don't misfire get revised and re-piloted, same process as
before, not treated as a one-shot failure of the whole plan.

**Budget**: 8 candidates × 5 draws = **40 real API calls** (Delegate only,
cheapest possible check before committing to anything bigger). This is
the recommended first real-API step — trivial cost, directly de-risks
everything below it.

---

## 4. E3 — Robustness (ablation, not a main claim)

Reuses E2's shared sample bank with **zero additional API calls** for 2 of
3 factors, exactly as Phase 3D already demonstrated:

1. **Sample count** `n ∈ {1, 3, 5, 10, 20}` — prefix-sliced from the same
   20-call bank.
2. **Entropy threshold** sweep — recomputed from the same raw per-facet
   entropy values, no new calls.
3. **Model** (2–3 models) — this factor DOES require new calls, since each
   model needs its own fresh n=20 bank per episode. Scope this to the
   calibrated C2/C4 subset only (not all 80 tasks), matching Phase 3D's
   own precedent of cross-model-testing only the arm-discriminating
   tasks, not the full benchmark.

Report as a single safety–utility trade-off characterization (§4.3/4.2 of
the current draft's structure already does this correctly) — explicitly
not a search for a new optimal threshold.

---

## 5. Statistics (fixed now, before any main run)

- A/B/C (E2) and the 4 arms (E1) are **paired** — same episodes, different
  fusion rules/aggregation rules. Use **McNemar's exact test** on
  discordant outcomes, exactly as already done for Phase 3C.
- Report **95% CI** on rates. Because 5 stochastic repeats share the same
  prompt template, treat repeats of the same template as one cluster, not
  5 independent datapoints — **cluster bootstrap** (resample templates,
  not individual episodes) for the CI, not a naive binomial CI.
- **Threshold**: freeze on a *development* subset, evaluate once on a
  disjoint *held-out* subset — not select-then-report on the same data.
  Concretely: split each domain's calibrated C2/C4 episodes roughly in
  half (e.g., by stochastic-repeat index, repeats 1–2 as dev, 3–5 as
  test, or similar — exact split decided once the calibration pilot shows
  how many usable episodes exist per domain). Threshold is picked on dev,
  reported once on test. This closes the exact gap this project already
  learned about the hard way once (`b4e17a5`) — a genuine pre-registration
  this time, fixed before the dev data is even collected.

---

## 6. What happens to Phase 2C–3D and the current draft

- **Nothing is deleted.** `docs/experiments/agent_connected_eval.md`,
  `phase3c_methods_for_paper.md`, all committed data/figures/scripts stay
  exactly as they are — append-only discipline unchanged.
- **Reframed, not discarded**: Phase 2C → "pilot evidence that a single
  independent restatement can share the Delegate's confident error."
  Phase 3C → "preliminary 3-arm comparison on one scenario, later found to
  have a sampling-independence confound." Phase 3D → "the shared-sample
  fix this plan's E2 design adopts natively, plus the sampling-count/
  threshold/cross-model characterization E3 reuses directly." All three
  become background/motivation sections, not the headline results table.
- **`docs/paper/draft_v1.md`** stays as a frozen snapshot of the
  Phase-2C–3D-centered paper (useful as a complete, fact-checked fallback
  if the redesign stalls) — a new draft gets written once E1/E2 produce
  real results, not by editing this one in place.

---

## 7. Budget summary (computed, nothing spent)

| Step | New API calls | Notes |
|---|---|---|
| E1 (80 tasks × 1 episode, C1/C3 now + C2/C4 after calibration) | ~80–120 | 1 Delegate + 1 Semantic + ~0–1 Authority call/episode; 4 arms computed post-hoc, free |
| Calibration pilot (§3) | **40** | 8 candidates × 5 Delegate-only draws — cheapest possible first step |
| E2 (calibrated C2/C4 subset only, say ~8 tasks × 5 repeats × 20-call bank) | ~800 | shared bank, half of what independent-sample Phase 3C cost per episode |
| E3 model sweep (×2 additional models, same ~8-task subset) | ~1,600 | reuses E2's n/threshold sweep at 0 extra cost; only the extra models cost calls |
| **Total, full plan** | **~2,500–2,600** | Smaller than Phase 3C's original 4,000-call run, because E1 is nearly free and E2/E3 reuse already-built shared-sample machinery |

This is a **much smaller real-API commitment than the rough order-of-
magnitude estimate given before this plan was written** (that estimate
assumed E1 needed its own separate sampling, which it does not — E1's 4
arms are a post-hoc relabeling of fields already collected for one
episode). The actual total is comparable to, not many times larger than,
prior phases of this project.

**Recommended next step**: run only §3's calibration pilot (40 calls) and
report back before anything else — exactly the same staged discipline
(design → cheap pilot → confirm → scale) this project has used at every
previous phase.
