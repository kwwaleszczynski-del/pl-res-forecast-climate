"""Actual wind and solar generation in the Polish power system from the public PSE reports API.

Endpoint ``his-wlk-cal`` ("wielkości podstawowe") of https://api.raporty.pse.pl/api gives 15-minute system totals,
including ``wi`` (wind) and ``pv`` (solar) in MW, from 14 Jun 2024. Data for a day are usually published one to two
days later and revised for up to about a month. The data are read with the pse-opendata client
(https://pypi.org/project/pse-opendata/), which caches whole months as Parquet files and downloads a month again until
the revision window of the report (60 days) has passed.

Hourly values are means of the four quarter-hours ending at the hour, labelled with the hour END in UTC, which is the
same convention as Open-Meteo's hourly radiation (mean over the preceding hour).

Usage:
    gen = fetch_generation(date(2024, 6, 14), date.today(), cache_dir=Path("data/pse"))   # hourly MW, columns pv, wind
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import pse_opendata as pse

ENDPOINT = "his-wlk-cal"
FIRST_DAY = date(2024, 6, 14)
log = logging.getLogger(__name__)


def to_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """pse-opendata frame -> 15-minute frame indexed by interval END (UTC) with columns pv, wind (MW)."""
    if raw.empty:
        return pd.DataFrame(columns=["pv", "wind"], dtype="float64").rename_axis("ts_utc")
    out = pd.DataFrame({"pv": raw["pv"].to_numpy(), "wind": raw["wi"].to_numpy()},
                       index=pd.DatetimeIndex(raw["end_utc"], name="ts_utc"))
    return out.sort_index()


def to_hourly(q: pd.DataFrame) -> pd.DataFrame:
    """Hourly means labelled by hour end; hours with fewer than 4 quarter-hours are dropped."""
    if q.empty:
        return q
    g = q.resample("1h", label="right", closed="right")
    out = g.mean()
    return out[g.count().min(axis=1) == 4]


def fetch_generation(start: date, end: date, cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """Hourly pv/wind (MW) for start..end, month by month through the pse-opendata cache in ``cache_dir``."""
    start = max(start, FIRST_DAY)
    client = pse.Client(cache_dir=cache_dir, refresh=refresh)
    frames = []
    for month_start in pd.date_range(date(start.year, start.month, 1), end, freq="MS"):
        m_end = (month_start + pd.offsets.MonthEnd(0)).date()
        raw = client.get(ENDPOINT, month_start.date(), m_end, fields=["pv", "wi"])
        df = to_hourly(to_frame(raw[raw["business_date"] <= end]))
        log.info("PSE %s: %d hours", f"{month_start:%Y-%m}", len(df))
        frames.append(df)
    out = pd.concat(frames).sort_index() if frames else to_hourly(to_frame(pd.DataFrame()))
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return out[(out.index > lo - pd.Timedelta(hours=2)) & (out.index <= hi + pd.Timedelta(hours=2))]
