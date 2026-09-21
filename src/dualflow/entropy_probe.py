"""Temporary compatibility entry point for the entropy validation harness.

The harness now lives in ``experiments/entropy_probe.py`` because it's a
real-LLM reproduction script (calls an external API), not part of the
DualFlow core package.

This compatibility shim is temporary and will be removed together with the
``dualflow-entropy-probe`` entry point during the final pyproject cleanup.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    """Run the repository-local entropy probe script."""
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "experiments" / "entropy_probe.py"

    if not script.is_file():
        raise SystemExit(
            "The entropy probe has moved to experiments/entropy_probe.py. "
            "Run `python experiments/entropy_probe.py` from the repository root."
        )

    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
