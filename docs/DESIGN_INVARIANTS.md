# DualFlow Design Invariants

> **Status.** This document collects the conditions that must never be
> violated by any future change to this codebase, in one place. They have
> accumulated across many design decisions and diagnostic experiments
> (recorded in detail in
> [`docs/experiments/agent_connected_eval.md`](experiments/agent_connected_eval.md)
> and [`docs/DESIGN_NOTES.md`](DESIGN_NOTES.md)) — this file exists so a
> reviewer, a future contributor, or a fresh session picking this project
> back up can find them without reading that history first.
>
> Each item notes **how it is currently enforced**. Some are structural —
> the code cannot violate them without an explicit, deliberate change (and
> a test would fail). Others are prompt-level instructions to a model,
> currently supported by real-API evidence but not mechanically
> guaranteed the way a deterministic check is — these are marked
> accordingly. Do not treat a prompt-level invariant as equivalent in
> strength to a code-level one.

1. **Runtime must never use `task.truth`.**
   Enforced structurally: `AgentDelegationRuntime.run()`
   (`src/dualflow/agent_runtime.py`) and everything it calls
   (`PrincipalAgent`, `DelegateAgent`, `SemanticVerifierAgent.
   verify_agent_proposal()`, `AuthorityVerifierAgent.verify_agent_proposal()`)
   take no `task.truth`/ground-truth argument at all — there is no
   parameter through which it could leak in. Every agent method is also
   swept by a test for forbidden parameter-name substrings (`truth`,
   `ground_truth`, `label`, ...). This is distinct from the legacy
   controlled benchmark (`framework.py`/`semantic.py`'s `Principal`), which
   *is* truth-aware by design for simulating what the principal knows —
   that path is not the real agent runtime and must not be confused with
   it.

2. **`SemanticVerifier` must not see `PrincipalIntent` before producing its
   verdict.**
   Enforced structurally: `AgentDelegationRuntime.run()`'s call order is
   `delegate() -> restate_intent() -> propose() -> semantic_verifier.
   verify_agent_proposal() -> authority_verifier.verify_agent_proposal() ->
   fuse()`. `PrincipalIntent` (from `restate_intent()`) is compared against
   the final interpretation only inside `_fuse()`, never passed into the
   Semantic Verifier's call. This is what makes `restate_intent()` a valid
   independent oracle-free reference instead of a self-match tautology
   (the `silent_misread` test in `tests/test_agent_runtime.py` is the
   canonical proof this matters).

3. **`AuthorityVerifier` must not see `SemanticVerdict` before producing
   its verdict.**
   Enforced structurally: `AuthorityVerifierAgent.verify_agent_proposal()`
   (`src/dualflow/authority_feedback.py`) takes the Delegate's proposal and
   the delegated `Budget`/context — it has no parameter for a
   `SemanticVerdict`. Integration tests use distinct fake-LLM-client
   instances per agent to structurally prove there is no cross-talk.

4. **Semantic experience must never grant authority.**
   Enforced structurally: `src/dualflow/agent_experience.py` and
   `src/dualflow/experience_aware_delegate.py` do not import or reference
   `Budget`, `check_authority`, or `AuthorityVerifierAgent` anywhere — a
   test asserts this against the actual imported namespace (not a source
   grep, which had a documented false-positive risk — see the B7d.3 commit
   history). `ExperienceAwareDelegate` only ever enriches the `context`
   string passed to `DelegateAgent.sample_candidates()`.

5. **`check_authority()` remains the final authority boundary.**
   Enforced structurally: `AuthorityVerifierAgent.verify_agent_proposal()`
   only calls the LLM when `check_authority()` returns `scope_exceeded`,
   and always re-verifies the LLM's suggested narrower proposal through
   `check_authority()` again before accepting it, falling back to the
   original proposal if the recheck fails. An LLM can suggest; it can
   never decide.

6. **Final fusion remains deterministic.**
   Enforced structurally: `AgentDelegationRuntime._fuse()` is a plain
   boolean/equality decision tree (`not semantic_verdict.confirmed ->
   REJECT`; `elif not authority_verdict.allowed -> REJECT`; `elif
   principal_intent.intended_action != authority_verdict.interpretation ->
   REJECT`; `else -> EXECUTE`) — no LLM call anywhere in it. Same principle
   in the legacy controlled framework's `DelegationVerifier` AND-gate.

7. **Historical experience must not override explicit current
   instructions.**
   Prompt-level, evidence-backed, not code-enforced. Both experience
   renderers (`render_experience_block_v1`/`_v2` in
   `src/dualflow/experience_aware_delegate.py`) instruct the model that the
   current delegation/context always takes precedence over historical
   examples, and `_enrich_context()` places the current context after the
   experience block with that precedence stated explicitly. The B7d.3
   real-API pilot observed 0 violations across 15 explicit-instruction
   rows (5 runs x 3 conditions, `P(export)` stayed 1.00 throughout) — real
   evidence, but from a small pilot, not a guarantee that no prompt or
   model can ever violate this. Treat as an actively-monitored property,
   not a closed proof.

8. **Old task-specific scope/date must not become current scope/date.**
   Enforced structurally: `render_experience_block_v1`/`_v2` deliberately
   omit any `SCOPE`/`CONDITION` field from the rendered "confirmed"
   summary — there is no code path by which a historical scope/date string
   could appear in the rendered block. Verified behaviorally across 900+
   real-API samples in the B7d/B7d.1/B7d.2/B7d.3 diagnostics: every single
   run's `scopes_seen` contained only the current episode's scope.

9. **Entropy is not treated as correctness.**
   Methodological principle, not a code check — there is no invariant a
   test can assert here beyond "these are reported as separate fields."
   `CandidateDistribution.entropy` (uncertainty) and
   `P(Principal-confirmed action)` (semantic alignment) are tracked and
   reported as two independent metrics everywhere real-API results are
   evaluated (`experiments/diagnostics/*.py`, `docs/experiments/
   agent_connected_eval.md`). This followed directly from measuring a case
   where entropy fell to 0 while the converged answer was wrong (B7d, §6 of
   the log) — low entropy must never be read as evidence of correct
   understanding on its own.

10. **Negative experimental results are not deleted from the research
    log.**
    Process invariant, not a code check.
    [`docs/experiments/agent_connected_eval.md`](experiments/agent_connected_eval.md)
    keeps unexpected/negative results (the B7d entropy-vs-correctness
    finding, the B7d.1/B7d.2 priming diagnostics, the B7d.3
    context-wording confound) as permanent record, not cleaned up after
    the fact — they are what motivated later design decisions and must
    stay legible to a reviewer, not be smoothed over into a narrative
    where everything worked the first time.

## Where these came from

Every invariant above was established through an explicit design decision
or a real-API diagnostic that found a violation risk and closed it — none
were assumed upfront. See
[`docs/experiments/agent_connected_eval.md`](experiments/agent_connected_eval.md)
for the chronological record, and
[`docs/DESIGN_NOTES.md`](DESIGN_NOTES.md) for abandoned approaches and why
they were rejected (e.g. why `rule_engine.match_intent()` was not reused
for oracle-free matching in `AgentDelegationRuntime`).
