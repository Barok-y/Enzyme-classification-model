"""Stage 04 - LightGBM.

LightGBM handles the sparse 3-mer counts directly, which makes it the model
that benefits most from the tripeptide representation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_runner as R

GRID = {
    "screen_params": {"n_estimators": 300},
    "clf__n_estimators": [400, 800],
    "clf__learning_rate": [0.05, 0.1],
    "clf__num_leaves": [31, 63],
    "clf__min_child_samples": [10, 20],
    "clf__class_weight": [None, "balanced"],
}


if __name__ == "__main__":
    R.run_from_cli("stage 04", "lgbm", GRID)