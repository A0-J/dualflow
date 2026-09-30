"""Condition-facet canonicalization -- built from actually-observed
output variants, not a blind substring hack (docs/experiments/
agent_connected_eval.md §29, cross-model validation follow-up).

Every distinct `condition` value observed across ALL collected Phase 3D
data (GPT-4o-mini: run1/run2/run3, ~1,200 `condition_violation` calls;
GPT-4.1: gpt41_run1, 400 calls) was enumerated first (see the git commit
history for the exact enumeration command/output) BEFORE writing this
function -- the mapping below is a direct reading of that enumeration,
not a guess:

    866  ()                                                  -- no condition
    333  ('report has been reviewed',)
    331  ('reviewed',)                                        -- GPT-4o-mini's own canonical form
     36  ('report must be reviewed',)
     25  ('report is reviewed',)
      2  ('report reviewed',)
      2  ('September 2026 report has been reviewed',)
      2  ('report in /reports/2026-09/ has been reviewed',)
      1  ('review_completed',)
      1  ('report must have been reviewed',)
      1  ('September 2026 financial report has been reviewed',)

Every non-empty variant contains "review" and none express negation.
The rule: a non-empty condition tuple canonicalizes to `("reviewed",)`
iff every element contains "review" (case-insensitive) AND does not
contain a negation marker -- the negation guard is a safety net (no
negated variant was actually observed in this data; it exists so a
future collection that DOES produce one, e.g. "has not been reviewed",
is not silently miscanonicalized to the opposite of its meaning).
Anything that doesn't match either rule is left unrecognized rather than
guessed at, and should be inspected manually.

This canonicalizer is applied identically to BOTH models' raw text --
there is no special-casing per model. For GPT-4o-mini it is a near-total
no-op (its own output is already almost always the canonical form); for
GPT-4.1 it unifies the paraphrases. That symmetry is the point -- this
is not "fixing GPT-4.1's output," it's "comparing both models through
the same normalization step neither model was told to skip."
"""

from __future__ import annotations

_NEGATION_MARKERS = ("not ", "n't", "without", "unreviewed", "no longer", "never")


def canonicalize_condition(condition: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Returns a canonical condition tuple, or the original (unrecognized)
    tuple unchanged if it doesn't match a known pattern."""
    if not condition:
        return ()
    values = tuple(condition)
    all_review = all("review" in v.lower() for v in values)
    any_negated = any(marker in v.lower() for v in values for marker in _NEGATION_MARKERS)
    if all_review and not any_negated:
        return ("reviewed",)
    return values  # unrecognized -- left as-is, not guessed at


def canonicalize_interpretation_condition(interp_dict: dict) -> dict:
    """Convenience: returns a copy of a stored `restate_responses` entry
    (or `delegate_final_interpretation`-shaped dict) with `condition`
    canonicalized, all other fields untouched."""
    return {**interp_dict, "condition": list(canonicalize_condition(interp_dict["condition"]))}


if __name__ == "__main__":
    # Self-validation against the exact enumeration this table was built
    # from, plus explicit negation-guard cases (not observed, but the
    # safety net must work if they ever occur).
    observed = [
        ((), ()),
        (("report has been reviewed",), ("reviewed",)),
        (("reviewed",), ("reviewed",)),
        (("report must be reviewed",), ("reviewed",)),
        (("report is reviewed",), ("reviewed",)),
        (("report reviewed",), ("reviewed",)),
        (("September 2026 report has been reviewed",), ("reviewed",)),
        (("report in /reports/2026-09/ has been reviewed",), ("reviewed",)),
        (("review_completed",), ("reviewed",)),
        (("report must have been reviewed",), ("reviewed",)),
        (("September 2026 financial report has been reviewed",), ("reviewed",)),
    ]
    negation_guard = [
        (("report has not been reviewed",), ("report has not been reviewed",)),  # left unrecognized, not flipped to 'reviewed'
        (("unreviewed",), ("unreviewed",)),
    ]
    for raw, expected in observed + negation_guard:
        got = canonicalize_condition(raw)
        status = "OK" if got == expected else f"MISMATCH (got {got})"
        print(f"{raw!r:60} -> {got!r:20} {status}")
