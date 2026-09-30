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
The rule (Rule 1): a non-empty condition tuple canonicalizes to
`("reviewed",)` iff every element contains "review" (case-insensitive)
AND does not contain a negation marker -- the negation guard is a safety
net (no negated variant was actually observed in this data; it exists so
a future collection that DOES produce one, e.g. "has not been reviewed",
is not silently miscanonicalized to the opposite of its meaning).

**2026-10 addition (GPT-4.1-mini full re-run, 4/400 `condition_violation`
calls)**: a second, genuinely new variant appeared that Rule 1 does not
cover -- a *two-element* tuple where one element affirms review and the
other restates the task's reference date verbatim from context (not a
new requirement, just verbose restatement), e.g.:

      2  ('date is 2026-09-23', 'report is reviewed')
      1  ('date is 2026-09-23 or later', 'report has been reviewed')
      1  ('date is 2026-09-23', 'report has been reviewed')

**Rule 2** (added, narrow and enumeration-based, not a broadening of Rule
1's "every element" semantics to "any element" -- that would risk
silently collapsing a genuinely different second requirement in some
future task/model): after removing elements that are pure date
restatements (matched narrowly -- `v.lower()` starts with "date is",
exactly the observed phrasing, not a general date regex), if the
*remaining* elements are non-empty and all review-affirming/non-negated,
canonicalize to `("reviewed",)`. A tuple containing ONLY a date
restatement and no review-affirming element (not observed, but checked)
is correctly left unrecognized, not silently dropped to `()`.

This canonicalizer is applied identically to every model's raw text --
there is no special-casing per model. For GPT-4o-mini it is a near-total
no-op (its own output is already almost always the canonical form); for
GPT-4.1 it unifies the paraphrases; for GPT-4.1-mini both rules apply.
That symmetry is the point -- this is not "fixing model X's output,"
it's "comparing every model through the same normalization step none of
them was told to skip." Anything that doesn't match either rule is left
unrecognized rather than guessed at, and should be inspected manually.
"""

from __future__ import annotations

_NEGATION_MARKERS = ("not ", "n't", "without", "unreviewed", "no longer", "never")


def _is_date_restatement(value: str) -> bool:
    """Narrow, enumeration-based match for the one new non-review element
    actually observed (GPT-4.1-mini) -- not a general date parser."""
    return value.lower().startswith("date is")


def canonicalize_condition(condition: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Returns a canonical condition tuple, or the original (unrecognized)
    tuple unchanged if it doesn't match a known pattern."""
    if not condition:
        return ()
    values = tuple(condition)
    any_negated = any(marker in v.lower() for v in values for marker in _NEGATION_MARKERS)
    if any_negated:
        return values

    # Rule 1: every element review-affirming (unchanged since this module's
    # original version -- covers the vast majority of GPT-4o-mini/GPT-4.1 data).
    if all("review" in v.lower() for v in values):
        return ("reviewed",)

    # Rule 2: drop pure date-restatement elements, then re-check Rule 1 on
    # what's left (GPT-4.1-mini's two-element variant, above).
    core = tuple(v for v in values if not _is_date_restatement(v))
    if core and core != values and all("review" in v.lower() for v in core):
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
    gpt41mini_observed = [
        (("date is 2026-09-23", "report is reviewed"), ("reviewed",)),
        (("date is 2026-09-23 or later", "report has been reviewed"), ("reviewed",)),
        (("date is 2026-09-23", "report has been reviewed"), ("reviewed",)),
        # a pure date restatement with no review-affirming element must stay
        # unrecognized, not silently dropped to () or 'reviewed' -- not
        # actually observed, but the boundary must hold if it ever occurs.
        (("date is 2026-09-23",), ("date is 2026-09-23",)),
        # negation must still block Rule 2, same as Rule 1
        (("date is 2026-09-23", "report has not been reviewed"),
         ("date is 2026-09-23", "report has not been reviewed")),
    ]
    for raw, expected in observed + negation_guard + gpt41mini_observed:
        got = canonicalize_condition(raw)
        status = "OK" if got == expected else f"MISMATCH (got {got})"
        print(f"{raw!r:60} -> {got!r:20} {status}")
