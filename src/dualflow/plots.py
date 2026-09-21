"""Compatibility entry point for the legacy ``dualflow-plots`` command.

Plot-generation code now lives in ``experiments/plots.py`` because plotting
and paper-reproduction utilities are not part of the DualFlow core package.

This compatibility shim is temporary and will be removed together with the
``dualflow-plots`` entry point during the final pyproject cleanup.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    """Run the repository-local experiment plotting script."""
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "experiments" / "plots.py"

    if not script.is_file():
        raise SystemExit(
            "Plot generation has moved to experiments/plots.py. "
            "Run `python experiments/plots.py` from the repository root."
        )

    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
