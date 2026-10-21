"""Cached loaders for the historical inputs: archived forecast runs and ERA5.

Complete months (forecast runs) and complete years (ERA5) are cached as Parquet in ``data/``; the current month or
year is cached as ``*_partial.parquet`` and fetched again once it is older than 12 hours, so the daily update only
spends a small part of the Open-Meteo budget.

Usage:
    om = OpenMeteo(ledger_path=Path("data/openmeteo_ledger.json"))
    runs = load_previous_runs(om, grid, "ecmwf_ifs025", date(2024, 6, 14), date(2026, 10, 20), Path("data/runs"))
    daily = load_era5_daily(om, grid, 1991, date(2026, 10, 15), Path("data/era5"))
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from res_forecast.config import ERA5_DAILY, ERA5_HOURLY, HINDCAST_MODELS
from res_forecast.openmeteo import OpenMeteo

log = logging.getLogger(__name__)
ERA5_LAG_DAYS = 6  # ERA5 on Open-Meteo is available with roughly a five-day delay
PARTIAL_MAX_AGE_H = 12.0  # an incomplete month/year downloaded less than this ago is reused


def _fresh(path: Path, max_age_h: float = PARTIAL_MAX_AGE_H) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_h * 3600


def _month_starts(start: date, end: date):
    for m in pd.date_range(date(start.year, start.month, 1), end, freq="MS"):
        yield m.date(), (m + pd.offsets.MonthEnd(0)).date()


def load_previous_runs(om: OpenMeteo | None, grid: pd.DataFrame, model: str, start: date, end: date, cache_dir: Path,
                       today: date | None = None, cached_only: bool = False) -> pd.DataFrame:
    """Archived runs of ``model`` (variables and leads from HINDCAST_MODELS); columns '<point>|<var>|<lead>'.

    With ``cached_only`` months that are not cached are skipped instead of downloaded.
    """
    today = today or date.today()
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for m_start, m_end in _month_starts(start, end):
        cache = cache_dir / f"{model}_{m_start:%Y-%m}.parquet"
        partial = cache_dir / f"{model}_{m_start:%Y-%m}_partial.parquet"
        complete = m_end < today - timedelta(days=1)
        if cache.exists() and complete:
            df = pd.read_parquet(cache)
        elif cached_only:
            continue
        elif not complete and _fresh(partial):
            df = pd.read_parquet(partial)
        else:
            df = om.previous_runs(grid, model, HINDCAST_MODELS[model], m_start, min(m_end, today + timedelta(days=1)))
            df.to_parquet(cache if complete else partial)
            log.info("%s %s: %d hours", model, f"{m_start:%Y-%m}", len(df))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out[(out.index >= pd.Timestamp(start, tz="UTC")) & (out.index < pd.Timestamp(end + timedelta(days=1), tz="UTC"))]


def _load_era5_years(om: OpenMeteo | None, grid: pd.DataFrame, first_year: int, end: date, cache_dir: Path, kind: str,
                     variables: list[str], today: date, cached_only: bool = False) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in range(first_year, end.year + 1):
        y_end = min(date(year, 12, 31), end)
        cache = cache_dir / f"era5_{kind}_{year}.parquet"
        partial = cache_dir / f"era5_{kind}_{year}_partial.parquet"
        complete = date(year, 12, 31) <= today - timedelta(days=ERA5_LAG_DAYS)
        if cache.exists() and complete:
            df = pd.read_parquet(cache)
        elif cached_only:
            continue
        elif not complete and _fresh(partial):
            df = pd.read_parquet(partial)
        else:
            kw = {"daily": variables} if kind == "daily" else {"hourly": variables}
            df = om.era5(grid, date(year, 1, 1), y_end, **kw)
            df.to_parquet(cache if complete else partial)
            log.info("ERA5 %s %d: %d rows", kind, year, len(df))
        frames.append(df)
    return pd.concat(frames).sort_index() if frames else pd.DataFrame()


def load_era5_daily(om: OpenMeteo | None, grid: pd.DataFrame, first_year: int, end: date, cache_dir: Path,
                    today: date | None = None, cached_only: bool = False) -> pd.DataFrame:
    """Daily ERA5 (columns '<point>|<var>') from 1 Jan ``first_year`` to ``end``; missing recent days are NaN."""
    today = today or date.today()
    end = min(end, today - timedelta(days=1))
    return _load_era5_years(om, grid, first_year, end, cache_dir, "daily", ERA5_DAILY, today, cached_only)


def load_era5_hourly(om: OpenMeteo | None, grid: pd.DataFrame, years: tuple[int, int], cache_dir: Path,
                     today: date | None = None, cached_only: bool = False) -> pd.DataFrame:
    today = today or date.today()
    return _load_era5_years(om, grid, years[0], date(years[1], 12, 31), cache_dir, "hourly", ERA5_HOURLY, today,
                            cached_only)


def split_columns(df: pd.DataFrame, var: str, lead: int | None = None) -> pd.DataFrame:
    """Select one variable (and lead) from a '<point>|<var>[|<lead>]' frame; columns become point ids."""
    suffix = f"|{var}" if lead is None else f"|{var}|{lead}"
    cols = [c for c in df.columns if c.endswith(suffix)]
    out = df[cols]
    out.columns = [int(c.split("|")[0]) for c in cols]
    return out.sort_index(axis=1)
