"""Temporary compatibility entry point for the DualFlow demo.

The demo now lives in ``experiments/demo.py`` because it's an executable
example of using the DualFlow package, not part of the package itself.

This compatibility shim is temporary and will be removed together with the
``dualflow-demo`` entry point during the final pyproject cleanup.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    """Run the repository-local demo script."""
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "experiments" / "demo.py"

    if not script.is_file():
        raise SystemExit(
            "The demo has moved to experiments/demo.py. "
            "Run `python experiments/demo.py` from the repository root."
        )

    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
