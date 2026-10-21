import pandas as pd

from res_forecast.pse import parse_records, to_hourly


def _records(start_utc: str, n: int, pv=1.0, wind=2.0):
    end = pd.date_range(pd.Timestamp(start_utc) + pd.Timedelta("15min"), periods=n, freq="15min")
    return [{"dtime_utc": f"{t:%Y-%m-%d %H:%M:%S}", "pv": pv * (i + 1), "wi": wind, "business_date": "2025-06-01"}
            for i, t in enumerate(end)]


def test_parse_records_index_is_interval_end_in_utc():
    df = parse_records(_records("2025-06-01 10:00", 4))
    assert df.index[0] == pd.Timestamp("2025-06-01 10:15", tz="UTC")
    assert (df.index[1] - df.index[0]) == pd.Timedelta("15min")
    assert list(df.columns) == ["pv", "wind"]


def test_to_hourly_labels_hour_end_and_needs_four_quarters():
    q = parse_records(_records("2025-06-01 10:00", 7))  # 10:15 .. 11:45 -> one full hour (10-11)
    h = to_hourly(q)
    assert list(h.index) == [pd.Timestamp("2025-06-01 11:00", tz="UTC")]
    assert h["pv"].iloc[0] == (1 + 2 + 3 + 4) / 4
    assert h["wind"].iloc[0] == 2.0


def test_empty_records():
    assert parse_records([]).empty
