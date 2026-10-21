"""Actual wind and solar generation in the Polish power system from the public PSE reports API.

Endpoint ``his-wlk-cal`` ("wielkości podstawowe") of https://api.raporty.pse.pl/api gives 15-minute system totals,
including ``wi`` (wind) and ``pv`` (solar) in MW, from 14 Jun 2024. ``dtime_utc`` is the END of the 15-minute
interval. Data for a day are usually published one to two days later.

Hourly values are means of the four quarter-hours ending at the hour, labelled with the hour END in UTC, which is the
same convention as Open-Meteo's hourly radiation (mean over the preceding hour).

Usage:
    gen = fetch_generation(date(2024, 6, 14), date.today(), cache_dir=Path("data/pse"))   # hourly MW, columns pv, wind
"""
from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.raporty.pse.pl/api"
ENDPOINT = "his-wlk-cal"
FIRST_DAY = date(2024, 6, 14)
log = logging.getLogger(__name__)


def parse_records(records: list[dict]) -> pd.DataFrame:
    """15-minute frame indexed by interval END (UTC) with columns pv, wind (MW)."""
    if not records:
        return pd.DataFrame(columns=["pv", "wind"], dtype="float64").rename_axis("ts_utc")
    df = pd.DataFrame.from_records(records)
    out = pd.DataFrame({
        "ts_utc": pd.to_datetime(df["dtime_utc"]).dt.tz_localize("UTC").dt.as_unit("ns"),
        "pv": pd.to_numeric(df["pv"], errors="coerce"),
        "wind": pd.to_numeric(df["wi"], errors="coerce"),
    })
    return out.drop_duplicates("ts_utc").set_index("ts_utc").sort_index()


def to_hourly(q: pd.DataFrame) -> pd.DataFrame:
    """Hourly means labelled by hour end; hours with fewer than 4 quarter-hours are dropped."""
    if q.empty:
        return q
    g = q.resample("1h", label="right", closed="right")
    out = g.mean()
    return out[g.count().min(axis=1) == 4]


def _get(session: requests.Session, url: str, params: dict | None = None, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            log.warning("PSE API request failed (%s), retrying", e)
            time.sleep(3 * 2 ** attempt)
    raise RuntimeError("unreachable")


def fetch_period(start: date, end: date, session: requests.Session | None = None) -> pd.DataFrame:
    """15-minute pv/wind for business dates start..end (inclusive), following pagination links."""
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", "pl-res-forecast-climate (open-source research project)")
    params = {
        "$filter": f"business_date ge '{start:%Y-%m-%d}' and business_date le '{end:%Y-%m-%d}'",
        "$select": "dtime_utc,pv,wi,business_date",
        "$first": 20000,
    }
    payload = _get(session, f"{BASE_URL}/{ENDPOINT}", params)
    records = list(payload.get("value", []))
    while payload.get("nextLink"):
        payload = _get(session, payload["nextLink"])
        records += payload.get("value", [])
    return parse_records(records)


def fetch_generation(start: date, end: date, cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """Hourly pv/wind (MW) for start..end; complete past months are cached as Parquet."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    start = max(start, FIRST_DAY)
    session = requests.Session()
    frames = []
    for month_start in pd.date_range(date(start.year, start.month, 1), end, freq="MS"):
        m_end = (month_start + pd.offsets.MonthEnd(0)).date()
        cache = cache_dir / f"pse_{month_start:%Y-%m}.parquet"
        complete = m_end < date.today() - pd.Timedelta(days=3)
        if cache.exists() and not refresh and complete:
            df = pd.read_parquet(cache)
        else:
            df = to_hourly(fetch_period(max(month_start.date(), FIRST_DAY), min(m_end, end), session))
            if complete:
                df.to_parquet(cache)
            log.info("PSE %s: %d hours", f"{month_start:%Y-%m}", len(df))
        frames.append(df)
    out = pd.concat(frames).sort_index() if frames else to_hourly(parse_records([]))
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return out[(out.index > lo - pd.Timedelta(hours=2)) & (out.index <= hi + pd.Timedelta(hours=2))]
