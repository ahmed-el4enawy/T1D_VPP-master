import sys
from pathlib import Path
import sys
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / 'data' / 'inputs'
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import unittest
from scripts.dataset_baseline import window_outcomes_batch, WelfordAccumulator
import numpy as np

class TestDatasetBaseline(unittest.TestCase):
    def test_welford_accumulator(self):
        acc = WelfordAccumulator()
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        acc.update(data)
        self.assertTrue(np.isclose(acc.get_mean(), 3.0))
        self.assertTrue(np.isclose(acc.get_std(), np.std(data, ddof=1)))
        
    def test_risk_metrics(self):
        bg_test = np.array([[100.0]])
        out = window_outcomes_batch(bg_test)
        self.assertTrue(out["LBGI"][0] > 0)
        self.assertTrue(out["HBGI"][0] == 0)
        
    def test_tir_tar_tbr(self):
        win = np.array([[65]*20 + [100]*21 + [200]*20])
        self.assertTrue(win.shape[1] == 61)
        out = window_outcomes_batch(win)
        tir, tar, tbr = out["TIR"][0], out["TAR"][0], out["TBR"][0]
        self.assertTrue(np.isclose(tir + tar + tbr, 100.0))

if __name__ == '__main__':
    unittest.main()
