# DualFlow: Independent Semantic and Authority Verification for Agent-to-Agent Delegation

> **Draft status**: first complete draft of the 5-page single-column paper body,
> written to lock in the exact experiment↔claim connections established across
> Phase 2C–3D before compressing to a professor's preferred style. Section
> lengths below track the agreed page budget (Introduction ~0.7–0.8p, Design
> ~1.0–1.1p, Setup ~0.5–0.6p, Evaluation ~1.7–1.9p, Discussion ~0.6–0.8p).
> Figures are referenced by their committed paths
> (`docs/experiments/figures/fig{1..4}_*.png`); all numbers are taken directly
> from `docs/experiments/phase3c_methods_for_paper.md` (frozen results) with no
> new computation. Title is a placeholder.

## Abstract

Agent-to-agent (A2A) delegation — one AI agent (the Principal) assigning a
task to another (the Delegate) — introduces two orthogonal risks: the
Delegate may *misunderstand* the Principal's intent, or it may propose an
action *outside* the authority it was actually granted, regardless of
whether it understood correctly. We present DualFlow, a delegation runtime
that verifies these two risks through independent, non-interacting checks —
a Semantic Verification Agent and a deterministic Authority Verification
Module — and evaluate it on a 7-task delegation benchmark using real LLM
agents. We first show empirically that naive one-shot semantic verification
fails specifically when the verifier shares the Delegate's confident
misinterpretation (20% unsafe execution on one task type). We then show that
repeating the verifier's independent reconstruction and grounding it at the
level of individual semantic facets — rather than the interpretation as a
whole — removes the observed unsafe executions while *reducing* unnecessary
rejections compared to a naive repeated-restatement baseline. We
characterize the resulting safety–utility trade-off across sampling count
and confidence threshold, and show across three LLMs that this benefit is
model-dependent: two models from the same generation exhibited no
exploitable output diversity for the defense to act on, while the deterministic
authority axis's guarantees held regardless of model.

---

## 1. Introduction

When one AI agent delegates a task to another, two questions must both be
answered before the action is allowed to run: *does the Delegate's proposed
action actually match what the Principal intended*, and *is the Delegate
authorized to take that action at all*? These are independent questions — a
correctly-understood request can still exceed authority, and an
authorized action can still be the wrong one — yet many delegation systems
either conflate them or verify only one.

A natural approach to the first question is semantic verification: have the
Principal independently restate its own intent, without seeing the
Delegate's proposal, and check whether the two agree. We show this is not
sufficient on its own. In a 140-episode evaluation of a production
delegation runtime backed by real LLM agents, unsafe execution was rare
overall (3.3%) but concentrated entirely in a single task designed to elicit
a *confident* semantic misread — 20% of episodes there resulted in unsafe
execution, and in every case, the Principal's own independent restatement
had confidently agreed with the Delegate's wrong interpretation. Two
nominally independent checks had shared the same error.

This paper makes four contributions. (1) We empirically demonstrate this
failure mode — confidently-shared misinterpretation defeating naive
independent-restatement verification — on a real, running delegation system,
not a synthetic worst case. (2) We evaluate two candidate fixes in a
controlled three-arm comparison: repeating the restatement and voting, and
repeating the restatement but grounding agreement at the level of individual
semantic facets with provenance (a facet only counts as evidence if the
Principal was itself consistent about it). The latter removes the observed
unsafe executions while reducing unnecessary rejections roughly five-fold
relative to the former. (3) We characterize the resulting safety–utility
trade-off across sampling count and confidence threshold, including a
previously unreported small-sample failure mode. (4) We show, across three
LLMs, that the semantic defense's benefit is model-dependent — a boundary
condition that a deterministic, LLM-independent authority check does not
share.

---

## 2. DualFlow Design

DualFlow separates delegation into two task agents and two independent
verification components, coordinated by a deterministic fusion rule.

**Principal Agent (A)** holds the original goal, context, and delegated
authority (a budget over allowed actions/resources/scopes/conditions). It
issues a delegation to the Delegate and, independently of that delegation,
can reconstruct what it actually intends by restating its own goal from
scratch — this reconstruction never sees the Delegate's output.

**Delegate Agent (B)** receives only the delegation text (and shared task
context) and proposes a concrete interpretation: an action over a resource,
at some scope, under some condition set.

**Semantic Verification Agent.** Given only the delegation text and B's
proposal — never A's original goal, never any ground truth — this agent
independently reconstructs what the delegation requires and compares that
reconstruction to B's proposal; it confirms iff they agree. Every
invocation is a real, independent LLM call with its own instructions; this
is a genuine second opinion, not a rule-based check.

**Authority Verification Module.** A deterministic capability check
compares B's proposed action/resource/scope/condition against the granted
budget. This check alone is the final arbiter for the large majority of
episodes: an already-permitted action, or a hard violation (no grant at
all, or a required condition missing), is decided without invoking any
model. Only one narrow, *recoverable* case — the proposed scope exceeds
what was granted, but a narrower scope within an already-computed ceiling
would be allowed — optionally invokes an LLM to propose that narrower
scope. That proposal is always re-validated by the same deterministic
check before being trusted: the LLM can narrow a request but can never
originate new authority or make the final call. We refer to this as
*non-amplification*.

**Fusion.** A deterministic function with no model calls combines the two
verdicts: reject if semantic verification did not confirm; reject (or
renegotiate, in the bounded case above) if authority does not allow it;
otherwise execute only if the Principal's own independent reconstruction of
intent also matches the final action. Neither verification axis needs to
be complete on its own, because the fusion rule lets either one catch what
the other misses — this is what we mean by *DualFlow = Semantic Flow ×
Authority Flow*, and Section 4.1 demonstrates it directly.

This paper's comparison is at the Principal's independent-reconstruction
step specifically, since that is the step shown (Section 4.1) to fail. We
compare three mechanisms for it, identical in every other respect:

- **Arm A (Current).** One restatement, compared to B's full proposal as a
  single unit.
- **Arm B (Repeated-Restate).** *n* independent restatements, majority-voted
  into one representative interpretation, still compared as a single unit.
- **Arm C (Grounded).** The same *n* restatements, but decomposed into four
  semantic facets (action, resource, scope, condition). A facet blocks
  execution only if (i) it disagrees with B's proposal *and* (ii) the
  Principal was itself consistent about it across the *n* restatements
  (entropy at or below a fixed confidence threshold). A facet the Principal
  could not itself resolve consistently is not used as grounds for
  rejection.

---

## 3. Experimental Setup

**Runtime.** All results use the production delegation runtime with real
Principal, Delegate, Semantic Verification, and Authority Verification
agents backed by the OpenAI API — no simulated agents or hand-specified
probability distributions.

**Benchmark.** Seven delegation tasks, 20 independently sampled episodes
each: an unambiguous clear delegation; a narrow-but-safe scope; a task
engineered to elicit a *confident* semantic misread; a persistently vague
request; a silent-misread task; a request missing a required authorization
condition; an over-privileged deletion request; and a sensitive-escalation
request — together spanning both semantic-ambiguity and authority-boundary
failure modes.

**Controls.** Across Arms A/B/C, the Delegate's output, the Semantic
Verifier's judgment, the Authority verdict, and the fusion order are held
identical (drawn from one frozen collection); only the Principal-side
mechanism (Section 2) varies. When repeated sampling is used, Arms B and C
consume the *same* drawn set of *n* restatements per episode, so their
comparison reflects only the aggregation rule, not sampling variance.

**Sampling and threshold.** *n*\=20 restatements per episode where repeated
sampling applies; entropy confidence threshold 0.8, a value already in use
from earlier stages of this project. Section 4.3 presents a sensitivity
analysis of this frozen operating point, not a search for an optimal value.

**Metrics.** *Unsafe execution rate* (executed when the Delegate's actual
interpretation was in fact wrong) and *false rejection rate* (rejected when
the Delegate's interpretation was in fact correct and authorized).

**Models.** GPT-4o-mini is the primary model for Sections 4.1–4.3.
Section 4.4 repeats the sampling-count sweep on GPT-4.1-mini and GPT-4.1
under the same protocol and identical episode inputs, with sampling
temperature and top-p explicitly fixed at 1.0 for every model.

**Validity note.** The `condition` facet is compared after a
canonicalization step that maps observed paraphrases of the same
requirement (e.g., "report has been reviewed" vs. "report is reviewed") to
one canonical token. This is a fix to a deterministic parser that failed to
recognize semantically identical phrasing — not a change to any model's
judgment rule or decision threshold — and was verified to leave every
previously-reported result, for every model, unchanged.

---

## 4. Evaluation

### 4.1 Where failures occur, and whether they are fixed (Figure 1)

Figure 1(a) shows unsafe execution by task in the frozen 140-episode
evaluation: 0/20 in six of seven tasks, and 4/20 (20%) concentrated
entirely in the task designed to elicit a confident semantic misread. In
all four cases, the Principal's single independent restatement had
confidently agreed with the Delegate's wrong interpretation — two
nominally independent checks sharing one error.

Figure 1(b)/(c) shows the controlled three-arm comparison this motivated:
unsafe execution 4/120 (Arm A) → 0/120 (Arm B) → 0/120 (Arm C); false
rejection, among authorized-and-correct episodes, 4/49 (A) → 5/49 (B) →
1/49 (C). Repeating the restatement alone (B) removes the observed unsafe
executions but is overly conservative — any single facet disagreement
blocks execution regardless of how confident the Principal actually was —
and false rejections rise slightly. Facet-level, provenance-aware grounding
(C) preserves the safety gain while reducing false rejections roughly
five-fold relative to naive repeated restatement.

Because all three arms are applied to the same episodes, we assess
significance with McNemar's exact test on the paired, discordant outcomes.
Neither comparison reaches conventional significance (unsafe execution: A
vs. B and A vs. C, both *p*\=0.125; false rejection: A vs. C *p*\=0.25, B
vs. C *p*\=0.125) — the number of episodes on which the arms actually
disagree is too small (3–4) for this test to have power. Every discordant
case, without exception, favored the repeated-verification arms; we report
this as a consistent directional finding, not a statistically significant
one.

### 4.2 How many samples are needed (Figure 2)

Sweeping *n* ∈ {1, 2, 3, 4, 5, 10, 15, 20} at the frozen threshold shows
Arm B's safety benefit saturates by *n*≈5. Arm C needs more samples to
reach its own stable region (*n*≈10–15) and is measurably *worse* than the
unverified single-call baseline at *n*\=4 (12 unsafe executions out of 60
pooled episodes, versus 3/60 at *n*\=1): with very few samples, common
splits already exceed the confidence threshold, so nearly every
disagreement is (incorrectly) treated as unconfirmed and ignored — a
concrete boundary condition on when facet-level grounding helps rather than
actively hurts. On the utility side, Arm B's false-rejection rate
*worsens* as *n* grows (more samples make its blunt, whole-interpretation
majority more confidently — and more often — wrong), while Arm C
stabilizes markedly below it once *n* is large enough.

### 4.3 Sensitivity of the frozen threshold (Figure 3)

At the deployed *n*\=20, sweeping the confidence threshold shows a
monotonic safety–utility trade-off: lower thresholds reduce unsafe
executions but increase false rejections, and vice versa. The frozen
operating threshold (0.8) sits just before false rejections begin climbing
steeply, while already capturing most of the achievable safety
improvement. We present this as a sensitivity analysis justifying
retention of an existing default, not as a post hoc search that produced
it.

### 4.4 Cross-model generalization (Figure 4)

Sections 4.1–4.3 use GPT-4o-mini. Repeating the sampling-count sweep on
GPT-4.1-mini and GPT-4.1, under an identical protocol and identical
episode inputs, shows a sharp contrast: GPT-4o-mini's unsafe-execution
curve is sensitive to *n* as above, while both GPT-4.1-mini and GPT-4.1
are completely flat — 12/20 unsafe executions at *every* value of *n* from
1 to 20 — because these models never exhibit the sample-to-sample
disagreement this defense depends on detecting. **Repeated sampling
behavior is model-dependent.** That two different scales of the same model
generation (mini and full) show identical flat behavior while a
prior-generation model does not is consistent with the pattern tracking
model family rather than model scale, but three models are not sufficient
to establish this causally; we report it as an observation (Section 5).

### 4.5 Supporting mechanism evidence

Two smaller diagnostics, prior to the full comparison, motivated its
design. A post-hoc reliability check of a single restatement found 80%
self-consistency, only weakly associated with whether the Delegate itself
had erred — evidence that one restatement alone is not a reliable
independent signal. A 20-episode validation of repeated intent anchoring
recovered all four previously-observed unsafe cases before the full
evaluation was run. Given their small sample size and pilot role, we
summarize these as supporting evidence rather than separate figures.

---

## 5. Discussion & Conclusion

Two implications follow from separating semantic and authority
verification into independent axes. First, the authority axis is a
deterministic, model-independent floor: because the final authority
decision is never delegated to a model (only a bounded, re-verified
negotiation for one recoverable violation type), its guarantees do not
depend on which model backs the system — unlike the semantic axis.
Second, the semantic axis's repeated-sampling defense is only as effective
as the underlying model's willingness to disagree with itself across
repeated queries; when a model converges confidently and consistently on
one interpretation, as both 4.1-generation models did here, no amount of
resampling or threshold adjustment recovers the signal this class of
defense depends on. This is an empirically observed boundary of the
approach, not a hypothetical one.

**Limitations.** (a) *Statistical power*: our controlled comparison's
conclusions are directionally consistent but not significant at
conventional levels, because too few episodes actually discriminate
between arms. (b) *Cost*: repeated sampling multiplies API calls by *n*
per episode, and Section 4.2's small-sample boundary condition means
simply reducing *n* is not a free fix. (c) *Model coverage*: three models
are enough to separate a family-related pattern from a pure scale effect
directionally, but not enough to establish either as the cause. (d) A
separate runtime case study (not detailed here) traces this system
end-to-end through real episodes and confirms both that redundant,
independent checks catch what any single layer misses, and the one case
where none of them can — a model and its Delegate sharing the exact same
confident error.

**Conclusion.** DualFlow's separation of semantic and authority
verification into two independently-checked axes provides a safety floor
that holds even when the semantic layer's own effectiveness is
model-dependent — a defense-in-depth property we verify empirically, not
merely assume, in every experiment in this paper.

**Future work.** Resolving the family-versus-scale question with
additional model families; adaptive early stopping to reduce the sampling
cost characterized in Section 4.2; integrating the facet-grounded verifier
into the production runtime's decision path itself (kept out of scope here
to hold this evaluation fixed); and a fuller account of the mechanism-design
history and detailed runtime traces that motivated this design — all
appropriate for an extended, thesis-length treatment.
