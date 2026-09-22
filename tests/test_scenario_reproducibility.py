"""B7d.3 pilot found that experiments/diagnostics/experience_transfer.py's
`build_current_context()` was RECONSTRUCTING the current-episode context
prose from separate scenario fields via an f-string template, and that the
reconstructed wording was only semantically (not byte-) equivalent to the
wording actually used in the earlier real-API B7a/B7b/B7d experiments
(experiments/agent_smoke.py's EXAMPLE_CURRENT_CONTEXT /
EXAMPLE_CURRENT_DELEGATION). The real 600-call B7d.3 run showed this drift
changes measured behavior (no_experience baseline came out fully
deterministic, P(export)=1.00 5/5 runs, unlike earlier pilots' baselines) —
see docs/experiments/agent_connected_eval.md §15.

The fix: experiments/scenarios/external_audit_finance.json now stores the
literal, model-visible current_episode.context/delegation strings directly
(no template reconstruction), and build_current_context() is a plain
passthrough. These tests pin that down as a regression: the scenario file's
current-episode strings must stay byte-identical to agent_smoke.py's
EXAMPLE_CURRENT_CONTEXT/EXAMPLE_CURRENT_DELEGATION (still the canonical,
real-API-validated wording — agent_smoke.py itself is not yet migrated to
read from the scenario file, deliberately deferred to right before B7e),
and build_current_context() must never go back to reconstructing prose.
"""

from __future__ import annotations

import experiments.agent_smoke as agent_smoke
from experiments.diagnostics.experience_transfer import (
    build_current_context,
    load_scenario,
    scenario_fingerprint,
    text_sha256,
)

SCENARIO_NAME = "external_audit_finance"


class TestCanonicalWordingMatchesAgentSmoke:
    """The scenario file's current_episode strings must stay byte-identical
    to agent_smoke.py's canonical, real-API-validated constants. If either
    file changes independently, this test must fail — that is the whole
    point (it is what B7d.3's pilot lacked)."""

    def test_context_byte_identical(self):
        scenario = load_scenario(SCENARIO_NAME)
        assert scenario["current_episode"]["context"] == agent_smoke.EXAMPLE_CURRENT_CONTEXT

    def test_delegation_byte_identical(self):
        scenario = load_scenario(SCENARIO_NAME)
        assert scenario["current_episode"]["delegation"] == agent_smoke.EXAMPLE_CURRENT_DELEGATION

    def test_context_contains_canonical_wording_not_reconstructed_variant(self):
        """Pins the exact sentence B7d.3 found drifted -- guards against a
        future edit silently reintroducing the "current period" paraphrase
        instead of the validated "September 2026 report" wording."""
        scenario = load_scenario(SCENARIO_NAME)
        context = scenario["current_episode"]["context"]
        assert "The September 2026 report is located at /reports/2026-09/." in context
        assert "current period" not in context


class TestBuildCurrentContextIsPassthrough:
    """build_current_context() must not reconstruct prose from other scenario
    fields -- it must return current_episode.context verbatim, unchanged."""

    def test_returns_exact_stored_string(self):
        scenario = load_scenario(SCENARIO_NAME)
        assert build_current_context(scenario) is scenario["current_episode"]["context"] \
            or build_current_context(scenario) == scenario["current_episode"]["context"]

    def test_ignores_other_fields_entirely(self):
        """A fake scenario dict with a deliberately different vocabulary/
        resource/scope than what's embedded in its own literal context
        string still returns the literal string unchanged -- proving no
        template reconstruction happens."""
        fake = {
            "current_episode": {
                "context": "LITERAL CONTEXT STRING, UNRELATED TO OTHER FIELDS",
                "resource": "something_else",
                "scope": "/unrelated/",
                "scope_vocabulary": ["/unrelated/"],
            },
            "action_vocabulary": ["unrelated_action"],
            "resource_vocabulary": ["unrelated_resource"],
            "reference_date": "1999-01-01",
        }
        assert build_current_context(fake) == "LITERAL CONTEXT STRING, UNRELATED TO OTHER FIELDS"


class TestScenarioFingerprint:
    """SHA256 fingerprint helper used to stamp diagnostic run output, added
    after the B7d.3 wording-drift finding so a raw results file can later be
    checked against the exact scenario version/wording that produced it."""

    def test_deterministic(self):
        scenario = load_scenario(SCENARIO_NAME)
        assert scenario_fingerprint(scenario) == scenario_fingerprint(scenario)

    def test_changes_with_content(self):
        scenario = load_scenario(SCENARIO_NAME)
        fp_before = scenario_fingerprint(scenario)
        mutated = {
            "current_episode": {
                **scenario["current_episode"],
                "delegation": scenario["current_episode"]["delegation"] + " (edited)",
            }
        }
        fp_after = scenario_fingerprint(mutated)
        assert fp_after["delegation_sha256"] != fp_before["delegation_sha256"]
        assert fp_after["context_sha256"] == fp_before["context_sha256"]

    def test_text_sha256_is_plain_sha256_hex(self):
        import hashlib
        assert text_sha256("abc") == hashlib.sha256(b"abc").hexdigest()


class TestScenarioHasVersion:
    def test_scenario_version_present(self):
        scenario = load_scenario(SCENARIO_NAME)
        assert "scenario_version" in scenario
        assert isinstance(scenario["scenario_version"], str) and scenario["scenario_version"]
