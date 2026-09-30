"""Stage 03 - Random Forest baseline.

Grid search is restricted to the training split; the representation screening
uses fixed hyper-parameters so that only the representation changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_runner as R

GRID = {
    "screen_params": {"n_estimators": 300},
    "clf__n_estimators": [400, 800],
    "clf__max_depth": [None, 25],
    "clf__min_samples_leaf": [1, 3],
    "clf__max_features": ["sqrt"],
    "clf__class_weight": [None, "balanced"],
}


if __name__ == "__main__":
    R.run_from_cli("stage 03", "rf", GRID)