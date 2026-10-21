import numpy as np
import pandas as pd
import pytest

from res_forecast.climate import (annual_table, anomalies_pct, apply_daily_wind_curve, exceedance, load_base,
                                  monthly, normals, save_base, year_to_date)


def _fields(start="1991-01-01", end="2021-12-31", n_points=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="D")
    season = 1 + 0.8 * np.sin(2 * np.pi * (idx.dayofyear - 80) / 365.25)
    ghi = pd.DataFrame(np.outer(season * 3, np.ones(n_points)) * rng.uniform(0.7, 1.3, (len(idx), n_points)),
                       index=idx, columns=range(n_points))
    wind = pd.DataFrame(rng.uniform(4, 10, (len(idx), n_points)), index=idx, columns=range(n_points))
    return {"ghi": ghi, "wind": wind}


def test_monthly_sums_and_means():
    f = _fields("2020-01-01", "2020-03-31")
    m = monthly(f)
    jan = f["ghi"].loc["2020-01"]
    assert m["ghi"].loc[pd.Period("2020-01", "M"), 0] == pytest.approx(jan[0].sum())
    assert m["wind"].loc[pd.Period("2020-02", "M"), 1] == pytest.approx(f["wind"].loc["2020-02", 1].mean())


def test_incomplete_month_is_nan():
    f = _fields("2020-01-01", "2020-02-10")
    m = monthly(f)
    assert np.isnan(m["ghi"].loc[pd.Period("2020-02", "M")]).all()


def test_normal_period_anomaly_averages_to_zero():
    m = monthly(_fields())
    a = anomalies_pct(m, normals(m))
    sel = a["ghi"][(a["ghi"].index.year >= 1991) & (a["ghi"].index.year <= 2020)]
    assert abs(sel.mean().mean()) < 1.0


def test_daily_wind_curve_interpolation():
    curve = pd.Series([0.0, 0.1, 0.5], index=[0.0, 4.0, 10.0])
    v = pd.DataFrame({0: [2.0, 10.0, np.nan]})
    out = apply_daily_wind_curve(v, curve)
    assert out[0].iloc[0] == pytest.approx(0.05)
    assert out[0].iloc[1] == pytest.approx(0.5)
    assert np.isnan(out[0].iloc[2])


def test_exceedance_and_annual_index():
    idx = pd.date_range("1991-01-01", "2021-12-31", freq="D")
    rng = np.random.default_rng(3)
    yearly = pd.Series(rng.normal(1.0, 0.05, 31), index=range(1991, 2022))
    y = pd.DataFrame({"pv_kwh_kwp": 3.0 * yearly.reindex(idx.year).to_numpy(),
                      "wind_cf": 0.25 * yearly.reindex(idx.year).to_numpy()}, index=idx)
    ann = annual_table(y)
    assert len(ann) == 31
    assert ann.loc[1991:2020, "pv_index_pct"].mean() == pytest.approx(100)
    ex = exceedance(ann["pv_kwh_kwp"])
    assert ex["p90"] < ex["p50"] < ex["p10"]
    assert 90 < ex["p90_over_p50_pct"] < 100


def test_year_to_date_rank():
    idx = pd.date_range("1991-01-01", "2022-06-30", freq="D")
    y = pd.DataFrame({"pv_kwh_kwp": 3.0, "wind_cf": 0.25}, index=idx)
    y.loc["2022", "pv_kwh_kwp"] = 4.0  # sunniest year so far
    y.loc["2022", "wind_cf"] = 0.1      # least windy
    ytd = year_to_date(y)
    assert ytd["year"] == 2022 and ytd["until"] == "2022-06-30"
    assert ytd["pv_rank"] == 1 and ytd["wind_rank"] == ytd["n_years"]
    assert ytd["pv_index_pct"] == pytest.approx(400 / 3)


def test_base_tables_round_trip(tmp_path):
    m = monthly(_fields("2019-01-01", "2020-12-31"))
    idx = pd.date_range("2019-01-01", "2020-12-31", freq="D")
    tables = {"ghi": m["ghi"], "wind": m["wind"], "data_until": "2020-12-31",
              "yields": pd.DataFrame({"pv_kwh_kwp": 3.0, "wind_cf": 0.3}, index=idx)}
    gain = pd.DataFrame(np.ones((12, 3)), index=range(1, 13), columns=range(3))
    curve = pd.Series([0.0, 0.5], index=[0.0, 10.0])
    save_base(tables, gain, curve, tmp_path)
    b = load_base(tmp_path)
    assert b["last_year"] == 2020
    pd.testing.assert_frame_equal(b["tables"]["ghi"], tables["ghi"], check_dtype=False, check_freq=False, atol=1e-4)
    assert list(b["gain"].columns) == [0, 1, 2]
