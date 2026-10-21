import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from res_forecast.openmeteo import BudgetExceeded, OpenMeteo, RateLimiter, _columns, _split_members, query_weight


def test_query_weight_matches_open_meteo_rule():
    assert query_weight(1, 7) == 1.0
    # Open-Meteo's own unit test: 20 years, one variable, one location -> 54.79 calls
    days = (date(2021, 1, 1) - date(2000, 1, 1)).days
    assert query_weight(1, days) == pytest.approx(54.79286, rel=1e-4)
    assert query_weight(2, days) == pytest.approx(2 * 54.79286, rel=1e-4)
    assert query_weight(102, 8, n_locations=3) == pytest.approx(3 * 10.2)


class FakeClock:
    def __init__(self, t):
        self.t = t
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def test_rate_limiter_waits_for_minute_window():
    clock = FakeClock(datetime(2026, 1, 1, 12, tzinfo=timezone.utc).timestamp())
    lim = RateLimiter(per_minute=100, per_hour=1000, per_day=5000, clock=clock, sleep=clock.sleep)
    lim.acquire(80)
    lim.acquire(15)
    assert clock.slept == []
    lim.acquire(30)  # 80 + 15 + 30 > 100 -> wait until the first entry leaves the window
    assert len(clock.slept) == 1 and 59 < clock.slept[0] <= 62


def test_rate_limiter_daily_budget_raises_when_wait_too_long():
    clock = FakeClock(datetime(2026, 1, 1, 12, tzinfo=timezone.utc).timestamp())
    lim = RateLimiter(per_minute=1e9, per_hour=1e9, per_day=100, clock=clock, sleep=clock.sleep, max_wait=3600)
    lim.acquire(90)
    with pytest.raises(BudgetExceeded):
        lim.acquire(20)


def test_rate_limiter_persists_ledger(tmp_path):
    clock = FakeClock(datetime(2026, 1, 1, 12, tzinfo=timezone.utc).timestamp())
    path = tmp_path / "ledger.json"
    RateLimiter(clock=clock, sleep=clock.sleep, ledger_path=path).acquire(42)
    again = RateLimiter(clock=clock, sleep=clock.sleep, ledger_path=path)
    assert again.spent_today() == 42


def test_columns_turn_nulls_into_nan_and_parse_unix_time():
    payload = {"hourly": {"time": [0, 3600, 7200], "a": [1.0, None, 3.0], "a_member01": [2, 2, None]}}
    times, cols = _columns(payload)
    assert (times[1] - times[0]) == pd.Timedelta("1h")
    assert times[0] == pd.Timestamp("1970-01-01", tz="UTC")
    assert np.isnan(cols["a"][1]) and np.isnan(cols["a_member01"][2])
    members = _split_members(cols, "a")
    assert members.shape == (2, 3)


class FakeResponse:
    def __init__(self, data, status=200):
        self.data, self.status_code, self.text = data, status, json.dumps(data)

    def json(self):
        return self.data

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self, data):
        self.data, self.headers, self.calls = data, {}, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return FakeResponse(self.data)


def test_previous_runs_columns_are_point_var_lead():
    payload = [{"hourly": {"time": [0, 3600], "shortwave_radiation_previous_day1": [0, 100],
                           "shortwave_radiation_previous_day2": [0, 90]}},
               {"hourly": {"time": [0, 3600], "shortwave_radiation_previous_day1": [0, 200],
                           "shortwave_radiation_previous_day2": [0, None]}}]
    session = FakeSession(payload)
    om = OpenMeteo(limiter=RateLimiter(per_day=1e9), session=session)
    pts = pd.DataFrame({"lat": [52.0, 53.0], "lon": [20.0, 21.0]}, index=[5, 9])
    df = om.previous_runs(pts, "ecmwf_ifs025", {"shortwave_radiation": [1, 2]}, date(2025, 1, 1), date(2025, 1, 1))
    assert set(df.columns) == {"5|shortwave_radiation|1", "5|shortwave_radiation|2",
                               "9|shortwave_radiation|1", "9|shortwave_radiation|2"}
    assert df["9|shortwave_radiation|1"].iloc[1] == 200
    assert np.isnan(df["9|shortwave_radiation|2"].iloc[1])
    assert session.calls[0][1]["latitude"] == "52.00,53.00"


def test_bad_request_is_not_retried():
    class BadSession(FakeSession):
        def get(self, url, params=None, timeout=None):
            self.calls.append(url)
            return FakeResponse({"error": True, "reason": "out of range"}, status=400)

    session = BadSession(None)
    om = OpenMeteo(limiter=RateLimiter(per_day=1e9), session=session)
    pts = pd.DataFrame({"lat": [52.0], "lon": [20.0]})
    with pytest.raises(ValueError, match="out of range"):
        om.era5(pts, date(2025, 1, 1), date(2025, 1, 2), daily=["shortwave_radiation_sum"])
    assert len(session.calls) == 1
