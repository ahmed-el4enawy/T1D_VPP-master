import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset_baseline import window_outcomes_batch, WelfordAccumulator
import numpy as np

def test_welford_accumulator():
    acc = WelfordAccumulator()
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    acc.update(data)
    assert np.isclose(acc.get_mean(), 3.0)
    assert np.isclose(acc.get_std(), np.std(data, ddof=1))
    
def test_risk_metrics():
    bg_test = np.array([[100.0]])
    out = window_outcomes_batch(bg_test)
    assert out["LBGI"][0] > 0
    assert out["HBGI"][0] == 0
    
def test_tir_tar_tbr():
    win = np.array([[65]*20 + [100]*21 + [200]*20])
    assert win.shape[1] == 61
    out = window_outcomes_batch(win)
    tir, tar, tbr = out["TIR"][0], out["TAR"][0], out["TBR"][0]
    assert np.isclose(tir + tar + tbr, 100.0)
