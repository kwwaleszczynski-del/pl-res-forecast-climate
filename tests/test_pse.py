from datetime import UTC, date, datetime

import pandas as pd
import pse_opendata as pse

from res_forecast.pse import fetch_generation, to_frame, to_hourly


def _raw(start_utc: str, n: int, pv=1.0, wind=2.0) -> pd.DataFrame:
    """A pse-opendata his-wlk-cal frame of ``n`` quarter-hours starting at ``start_utc``."""
    start = pd.date_range(pd.Timestamp(start_utc, tz="UTC"), periods=n, freq="15min")
    return pd.DataFrame({"start_utc": start, "end_utc": start + pd.Timedelta("15min"),
                         "business_date": date(2025, 6, 1), "pv": [pv * (i + 1) for i in range(n)], "wi": wind})


def test_frame_index_is_interval_end_in_utc():
    df = to_frame(_raw("2025-06-01 10:00", 4))
    assert df.index[0] == pd.Timestamp("2025-06-01 10:15", tz="UTC")
    assert (df.index[1] - df.index[0]) == pd.Timedelta("15min")
    assert list(df.columns) == ["pv", "wind"]


def test_to_hourly_labels_hour_end_and_needs_four_quarters():
    q = to_frame(_raw("2025-06-01 10:00", 7))  # 10:15 .. 11:45 -> one full hour (10-11)
    h = to_hourly(q)
    assert list(h.index) == [pd.Timestamp("2025-06-01 11:00", tz="UTC")]
    assert h["pv"].iloc[0] == (1 + 2 + 3 + 4) / 4
    assert h["wind"].iloc[0] == 2.0


def test_empty_frame():
    assert to_frame(pd.DataFrame()).empty


class FakeResponse:
    def __init__(self, payload):
        self.status_code, self.headers, self.reason, self.payload = 200, {}, "OK", payload

    def json(self):
        return self.payload


class FakeSession:
    """his-wlk-cal records for 1-2 June 2025 (pv = 10, wi = 20), filtered like the API."""

    def __init__(self):
        starts = pd.date_range(pd.Timestamp("2025-06-01", tz="Europe/Warsaw"), periods=2 * 96, freq="15min")
        self.records = [{"dtime_utc": f"{(t + pd.Timedelta('15min')).tz_convert('UTC'):%Y-%m-%d %H:%M:%S}",
                         "business_date": f"{t:%Y-%m-%d}", "pv": 10.0, "wi": 20.0} for t in starts]

    def get(self, url, params=None, timeout=None, headers=None):
        lo, hi = [part.split("'")[1] for part in params["$filter"].split(" and ")]
        return FakeResponse({"value": [r for r in self.records if lo <= r["business_date"] <= hi]})


def test_fetch_generation_cuts_the_month_at_the_end_date(tmp_path, monkeypatch):
    real = pse.Client
    monkeypatch.setattr(pse, "Client", lambda **kw: real(session=FakeSession(), sleep=lambda s: None, min_interval=0,
                                                         now=lambda: datetime(2025, 6, 2, 12, tzinfo=UTC), **kw))
    gen = fetch_generation(date(2025, 6, 1), date(2025, 6, 1), tmp_path)
    assert len(gen) == 24 and (gen["pv"] == 10.0).all() and (gen["wind"] == 20.0).all()
    assert gen.index[0] == pd.Timestamp("2025-05-31 23:00", tz="UTC")     # first hour of 1 June ends at 01:00 local
    assert (tmp_path / "api" / "his-wlk-cal" / "2025-06.parquet").exists()
