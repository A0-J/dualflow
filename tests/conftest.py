"""experiments/ is deliberately not part of the installed dualflow package
(no __init__.py, not registered in pyproject.toml) — but
tests/test_bench_single_env_v1.py needs to import the canonical benchmark
from experiments/benchmark.py. Put the repository root on sys.path so it's
importable as a plain namespace package (`from experiments.benchmark import
...`), without making experiments/ an installable package.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
