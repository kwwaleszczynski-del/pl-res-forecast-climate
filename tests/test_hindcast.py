from datetime import date

import numpy as np
import pandas as pd

from res_forecast.hindcast import calibrations_at, run_hindcast
from res_forecast.verify import summarize


def synthetic(days=150, seed=0):
    """Wind-like physical forecast x and generation y = 2x + noise, hourly, for two lead days."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01 01:00", periods=24 * days, freq="h", tz="UTC")
    truth = 2000 + 1500 * np.sin(np.arange(len(idx)) / 37.0) + rng.normal(0, 300, len(idx)).cumsum() * 0.02
    truth = np.clip(truth, 50, None)
    y = pd.DataFrame({"wind": 2 * truth + rng.normal(0, 150, len(idx)), "pv": np.nan}, index=idx)
    phys = pd.DataFrame({f"wind|{lead}": truth + rng.normal(0, 100 * lead, len(idx)) for lead in (1, 2)}, index=idx)
    return {"ecmwf_ifs025": phys}, y


def test_hindcast_is_calibrated_on_synthetic_data():
    phys, y = synthetic()
    rows = run_hindcast(phys, y, window=30, techs=("wind",), leads=[1, 2])
    s = summarize(rows).set_index("lead")
    assert set(rows["product"]) == {"ecmwf_ifs025"}
    assert 70 < s.loc[1, "coverage80_pct"] < 90
    assert s.loc[1, "crps_mw"] < s.loc[2, "crps_mw"]  # longer lead, noisier forecast
    assert s.loc[1, "crpss_vs_clim_pct"] > 0


def test_no_look_ahead():
    """Changing observations after an issue day must not change forecasts issued on or before it."""
    phys, y = synthetic()
    cut = pd.Timestamp("2025-04-01", tz="UTC")
    a = run_hindcast(phys, y, window=30, techs=("wind",), leads=[1], last_issue=date(2025, 3, 30))
    y2 = y.copy()
    y2.loc[y2.index >= cut, "wind"] *= 3
    b = run_hindcast(phys, y2, window=30, techs=("wind",), leads=[1], last_issue=date(2025, 3, 30))
    np.testing.assert_allclose(a["q50"].to_numpy(), b["q50"].to_numpy())


def test_multi_model_product_present_with_two_models():
    phys, y = synthetic()
    other = phys["ecmwf_ifs025"] * 1.1
    rows = run_hindcast({"ecmwf_ifs025": phys["ecmwf_ifs025"], "icon_seamless": other}, y, window=30,
                        techs=("wind",), leads=[1])
    assert {"ecmwf_ifs025", "icon_seamless", "multi"} <= set(rows["product"])


def test_calibrations_at_returns_one_fit_per_model_and_lead():
    phys, y = synthetic()
    c = calibrations_at(phys, y, date(2025, 5, 1), window=30, leads=[1, 2])
    assert set(c) == {("ecmwf_ifs025", "wind", 1), ("ecmwf_ifs025", "wind", 2)}
    assert c[("ecmwf_ifs025", "wind", 1)].coef[0] > 1.5
