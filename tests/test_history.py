from datetime import date

import numpy as np
import pandas as pd

from res_forecast.history import load_era5_daily, load_previous_runs, split_columns


class CountingOM:
    def __init__(self):
        self.calls = []

    def previous_runs(self, grid, model, leads, start, end):
        self.calls.append(("runs", start, end))
        idx = pd.date_range(start, end, freq="h", tz="UTC")
        return pd.DataFrame({f"{p}|shortwave_radiation|1": np.ones(len(idx)) for p in grid.index}, index=idx)

    def era5(self, grid, start, end, daily=None, hourly=None):
        self.calls.append(("era5", start, end))
        idx = pd.date_range(start, end, freq="D", tz="UTC")
        return pd.DataFrame({f"{p}|{v}": np.ones(len(idx)) for p in grid.index for v in daily}, index=idx)


GRID = pd.DataFrame({"lat": [52.0], "lon": [20.0]}, index=[0])


def test_complete_months_are_cached_and_the_current_month_reused_while_fresh(tmp_path):
    om = CountingOM()
    today = date(2025, 3, 15)
    load_previous_runs(om, GRID, "ecmwf_ifs025", date(2025, 2, 1), today, tmp_path, today=today)
    assert len(om.calls) == 2  # February (complete) and March (partial)
    df = load_previous_runs(om, GRID, "ecmwf_ifs025", date(2025, 2, 1), today, tmp_path, today=today)
    assert len(om.calls) == 2  # nothing downloaded again
    assert (tmp_path / "ecmwf_ifs025_2025-02.parquet").exists()
    assert (tmp_path / "ecmwf_ifs025_2025-03_partial.parquet").exists()
    assert (df.index[1] - df.index[0]) == pd.Timedelta("1h")
    assert list(split_columns(df, "shortwave_radiation", 1).columns) == [0]


def test_cached_only_skips_missing_months_and_uses_partial_ones(tmp_path):
    om = CountingOM()
    today = date(2025, 3, 15)
    assert load_previous_runs(None, GRID, "ecmwf_ifs025", date(2025, 1, 1), today, tmp_path, today=today,
                              cached_only=True).empty
    load_previous_runs(om, GRID, "ecmwf_ifs025", date(2025, 3, 1), today, tmp_path, today=today)
    out = load_previous_runs(None, GRID, "ecmwf_ifs025", date(2025, 1, 1), today, tmp_path, today=today,
                             cached_only=True)
    assert out.index.min() == pd.Timestamp("2025-03-01", tz="UTC")


def test_era5_years(tmp_path):
    om = CountingOM()
    today = date(2025, 6, 1)
    d = load_era5_daily(om, GRID, 2023, today, tmp_path, today=today)
    assert [c[1].year for c in om.calls] == [2023, 2024, 2025]
    assert d.index.max() == pd.Timestamp("2025-05-31", tz="UTC")
    load_era5_daily(om, GRID, 2023, today, tmp_path, today=today)
    assert len(om.calls) == 3
