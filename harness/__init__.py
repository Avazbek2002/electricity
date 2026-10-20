"""Personal hidden-test harness: one command (evaluate.py) scores every model the same way.

    harness/contract.py   the Model interface every model implements, and the Panel it receives
    harness/data.py       reads one district (developer zone + vault) into a customer x day matrix
    harness/runner.py     per district: as-of panels per origin -> model.predict -> node sums (workers)
    harness/scoring.py    system WAPE per level (transformer, feeder, substation, district, national), headline
    harness/tracking.py   git guard, vault look budget, MLflow + local CSV logs
    harness/cli.py        the command line (evaluate.py)
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(REPO, "core")
for _p in (CORE, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
