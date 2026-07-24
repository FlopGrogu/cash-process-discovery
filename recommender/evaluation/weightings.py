"""The 15-weighting grid shared by every evaluation script.

All analyses (intrinsic evaluation, ProReco comparison, learning curve, dataset
analysis) iterate over the same 15 measure weightings so their numbers are
directly comparable: equal + 4 single measures + 6 pairs + 4 graded mixes.
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cash.model import MEASURES


def weight_sets() -> dict:
    """15 weightings: equal + 4 single measures + 6 pairs + 4 graded mixes."""
    sets = {"equal": {x: 0.25 for x in MEASURES}}
    for x in MEASURES:
        sets[x[:4]] = {y: (1.0 if y == x else 0.0) for y in MEASURES}
    for a, b in itertools.combinations(MEASURES, 2):
        sets[f"{a[:3]}+{b[:3]}"] = {y: (1.0 if y in (a, b) else 0.0) for y in MEASURES}
    sets["mix-fit"] = {"fitness": 0.4, "precision": 0.3, "generalization": 0.1, "simplicity": 0.2}
    sets["mix-prec"] = {"fitness": 0.2, "precision": 0.4, "generalization": 0.3, "simplicity": 0.1}
    sets["mix-simp"] = {"fitness": 0.3, "precision": 0.1, "generalization": 0.2, "simplicity": 0.4}
    sets["mix-gen"] = {"fitness": 0.1, "precision": 0.2, "generalization": 0.4, "simplicity": 0.3}
    return sets
