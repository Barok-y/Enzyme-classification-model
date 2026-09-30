"""Stage 05 - Support Vector Machine (RBF kernel).

Features are standardised inside the pipeline because the distance based RBF
kernel is not scale invariant. The grid is deliberately small: an RBF SVM on
the full matrix is the most expensive model of the study, so
``ENZYME_TUNE_MAX_ROWS`` should be raised only on a machine that can afford it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_runner as R

GRID = {
    "screen_params": {},
    "clf__C": [1.0, 10.0, 100.0],
    "clf__gamma": ["scale", 0.001],
    "clf__class_weight": [None, "balanced"],
}


if __name__ == "__main__":
    R.run_from_cli("stage 05", "svm", GRID)