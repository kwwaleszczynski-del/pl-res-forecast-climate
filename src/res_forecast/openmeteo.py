"""Client for the free Open-Meteo APIs (no API key; non-commercial use, data under CC BY 4.0).

Endpoints used:
    ensemble       https://ensemble-api.open-meteo.com/v1/ensemble       live ensemble members (ECMWF IFS ENS, ICON-EPS, GEFS)
    previous runs  https://previous-runs-api.open-meteo.com/v1/forecast  archived deterministic runs, lead day 1..7
    archive        https://archive-api.open-meteo.com/v1/archive         ERA5 reanalysis
    forecast       https://api.open-meteo.com/v1/forecast                deterministic fallback when the ensemble fails

Open-Meteo counts a request as ``max(1, n_variables / 10 * max(1, n_days / 14))`` calls per location and allows
600 calls per minute, 5 000 per hour and 10 000 per day on the free tier. :class:`RateLimiter` keeps a ledger of
the weight spent (optionally persisted to disk so separate processes share it) and sleeps before a request would break
a limit; requests are split into location batches to stay under the per-minute cap.

Usage:
    om = OpenMeteo(ledger_path=Path("data/openmeteo_ledger.json"))
    ens = om.ensemble(points, "ecmwf_ifs025", ["shortwave_radiation", "wind_speed_100m"], forecast_days=8)
    prev = om.previous_runs(points, "ecmwf_ifs025", {"shortwave_radiation": [1, 2]}, date(2025, 1, 1), date(2025, 1, 31))
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
USER_AGENT = "pl-res-forecast-climate (open-source research project)"
log = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    """Raised when a request would have to wait longer than allowed for the rate limit."""


def query_weight(n_variables: int, n_days: float, n_locations: int = 1) -> float:
    """Open-Meteo's call weight: ``max(1, vars/10 * max(1, days/14))`` per location."""
    return n_locations * max(1.0, n_variables / 10 * max(1.0, n_days / 14))


class RateLimiter:
    """Sliding-window limits on weighted calls (per minute, per hour, per UTC day)."""

    def __init__(self, per_minute=500.0, per_hour=4500.0, per_day=9500.0, ledger_path: Path | None = None,
                 max_wait: float = 3 * 3600, clock=time.time, sleep=time.sleep):
        self.limits = ((60.0, per_minute), (3600.0, per_hour))
        self.per_day = per_day
        self.ledger_path = ledger_path
        self.max_wait = max_wait
        self.clock, self.sleep = clock, sleep
        self.entries: list[tuple[float, float]] = []
        if ledger_path and ledger_path.exists():
            self.entries = [tuple(e) for e in json.loads(ledger_path.read_text())]

    def _today_start(self, now: float) -> float:
        d = datetime.fromtimestamp(now, tz=timezone.utc).date()
        return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()

    def wait_time(self, weight: float, now: float) -> float:
        wait = 0.0
        for window, limit in self.limits:
            recent = [(t, w) for t, w in self.entries if t > now - window]
            used = sum(w for _, w in recent)
            if used + weight > limit:
                # wait until enough of the oldest entries leave the window
                excess = used + weight - limit
                for t, w in sorted(recent):
                    excess -= w
                    if excess <= 0:
                        wait = max(wait, t + window - now + 1.0)
                        break
                else:
                    wait = max(wait, window)
        day_start = self._today_start(now)
        if sum(w for t, w in self.entries if t >= day_start) + weight > self.per_day:
            wait = max(wait, day_start + 86400 - now + 60.0)
        return wait

    def acquire(self, weight: float) -> None:
        now = self.clock()
        wait = self.wait_time(weight, now)
        if wait > self.max_wait:
            raise BudgetExceeded(f"Open-Meteo budget: would need to wait {wait / 60:.0f} min for {weight:.0f} calls")
        if wait > 0:
            log.info("rate limit: sleeping %.0f s before a request of weight %.0f", wait, weight)
            self.sleep(wait)
            now = self.clock()
        self.entries = [(t, w) for t, w in self.entries if t > now - 2 * 86400] + [(now, weight)]
        if self.ledger_path:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            self.ledger_path.write_text(json.dumps(self.entries))

    def spent_today(self) -> float:
        start = self._today_start(self.clock())
        return sum(w for t, w in self.entries if t >= start)


@dataclass
class Ensemble:
    """Ensemble members for one model: ``values[var]`` has shape (n_points, n_members, n_times)."""
    model: str
    times: pd.DatetimeIndex
    values: dict


def _columns(payload: dict, section: str = "hourly") -> tuple[pd.DatetimeIndex, dict]:
    block = payload[section]
    times = pd.to_datetime(np.asarray(block["time"], dtype="int64"), unit="s", utc=True).as_unit("ns")
    # numpy turns JSON nulls (None) into NaN when the dtype is float
    cols = {k: np.asarray(v, dtype="float64").astype("float32") for k, v in block.items() if k != "time"}
    return times, cols


def _split_members(cols: dict, var: str) -> np.ndarray:
    """Stack ``var`` (control) and ``var_memberNN`` columns into (n_members, n_times)."""
    members = [cols[var]] if var in cols else []
    n = 1
    while f"{var}_member{n:02d}" in cols:
        members.append(cols[f"{var}_member{n:02d}"])
        n += 1
    return np.vstack(members) if members else np.empty((0, 0), dtype="float32")


class OpenMeteo:
    def __init__(self, limiter: RateLimiter | None = None, ledger_path: Path | None = None,
                 session: requests.Session | None = None, retries: int = 5, batch_weight: float = 250.0,
                 timeout: float = 120.0):
        self.limiter = limiter or RateLimiter(ledger_path=ledger_path)
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.retries, self.batch_weight, self.timeout = retries, batch_weight, timeout

    # -- transport -------------------------------------------------------------------------------------------------
    def _get(self, url: str, params: dict, weight: float) -> list[dict]:
        self.limiter.acquire(weight)
        for attempt in range(self.retries):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 400:
                    raise ValueError(f"Open-Meteo rejected the request: {r.text[:300]}")
                if r.status_code == 429:
                    raise requests.HTTPError("429 Too Many Requests", response=r)
                r.raise_for_status()
                data = r.json()
                return data if isinstance(data, list) else [data]
            except (requests.RequestException, json.JSONDecodeError) as e:
                if attempt == self.retries - 1:
                    raise
                pause = 65.0 if "429" in str(e) else 5.0 * 2 ** attempt
                log.warning("Open-Meteo request failed (%s), retry in %.0f s", e, pause)
                time.sleep(pause)
        raise RuntimeError("unreachable")

    def _batches(self, points: pd.DataFrame, per_location: float):
        size = max(1, int(self.batch_weight // per_location))
        for i in range(0, len(points), size):
            yield points.iloc[i:i + size]

    @staticmethod
    def _loc_params(points: pd.DataFrame) -> dict:
        return {"latitude": ",".join(f"{x:.2f}" for x in points["lat"]),
                "longitude": ",".join(f"{x:.2f}" for x in points["lon"])}

    # -- ensemble --------------------------------------------------------------------------------------------------
    def ensemble(self, points: pd.DataFrame, model: str, variables: list[str], forecast_days: int = 8,
                 n_members: int = 51) -> Ensemble:
        per_loc = query_weight(len(variables) * n_members, forecast_days)
        times, out = None, {v: [] for v in variables}
        for batch in self._batches(points, per_loc):
            params = {**self._loc_params(batch), "hourly": ",".join(variables), "models": model,
                      "forecast_days": forecast_days, "timeformat": "unixtime", "timezone": "GMT"}
            for payload in self._get(ENSEMBLE_URL, params, per_loc * len(batch)):
                t, cols = _columns(payload)
                times = t if times is None else times
                for v in variables:
                    out[v].append(_split_members(cols, v))
        return Ensemble(model, times, {v: np.stack(a) for v, a in out.items()})

    # -- archived deterministic runs ------------------------------------------------------------------------------
    def previous_runs(self, points: pd.DataFrame, model: str, leads: dict[str, list[int]], start: date,
                      end: date) -> pd.DataFrame:
        """Hourly values of ``var`` forecast ``lead`` days earlier; columns '<point>|<var>|<lead>'."""
        names = {f"{v}_previous_day{d}": (v, d) for v, ds in leads.items() for d in ds}
        n_days = (end - start).days + 1
        per_loc = query_weight(len(names), n_days)
        frames = []
        for batch in self._batches(points, per_loc):
            params = {**self._loc_params(batch), "hourly": ",".join(names), "models": model,
                      "start_date": f"{start:%Y-%m-%d}", "end_date": f"{end:%Y-%m-%d}",
                      "timeformat": "unixtime", "timezone": "GMT"}
            for pid, payload in zip(batch.index, self._get(PREVIOUS_RUNS_URL, params, per_loc * len(batch))):
                times, cols = _columns(payload)
                frames.append(pd.DataFrame({f"{pid}|{names[k][0]}|{names[k][1]}": v for k, v in cols.items()},
                                           index=times))
        return pd.concat(frames, axis=1)

    # -- ERA5 ------------------------------------------------------------------------------------------------------
    def era5(self, points: pd.DataFrame, start: date, end: date, daily: list[str] | None = None,
             hourly: list[str] | None = None) -> pd.DataFrame:
        """ERA5 for the points; columns '<point>|<var>'. Pass either ``daily`` or ``hourly`` variables."""
        section, variables = ("daily", daily) if daily else ("hourly", hourly)
        n_days = (end - start).days + 1
        per_loc = query_weight(len(variables), n_days)
        frames = []
        for batch in self._batches(points, per_loc):
            params = {**self._loc_params(batch), section: ",".join(variables), "models": "era5",
                      "start_date": f"{start:%Y-%m-%d}", "end_date": f"{end:%Y-%m-%d}",
                      "timeformat": "unixtime", "timezone": "GMT"}
            for pid, payload in zip(batch.index, self._get(ARCHIVE_URL, params, per_loc * len(batch))):
                times, cols = _columns(payload, section)
                frames.append(pd.DataFrame({f"{pid}|{k}": v for k, v in cols.items()}, index=times))
        return pd.concat(frames, axis=1)

    # -- deterministic fallback ------------------------------------------------------------------------------------
    def forecast(self, points: pd.DataFrame, model: str, variables: list[str], forecast_days: int = 8) -> Ensemble:
        """Single deterministic run as a one-member :class:`Ensemble` (used when the ensemble API fails)."""
        per_loc = query_weight(len(variables), forecast_days)
        times, out = None, {v: [] for v in variables}
        for batch in self._batches(points, per_loc):
            params = {**self._loc_params(batch), "hourly": ",".join(variables), "models": model,
                      "forecast_days": forecast_days, "timeformat": "unixtime", "timezone": "GMT"}
            for payload in self._get(FORECAST_URL, params, per_loc * len(batch)):
                t, cols = _columns(payload)
                times = t if times is None else times
                for v in variables:
                    out[v].append(cols[v][None, :])
        return Ensemble(model, times, {v: np.stack(a) for v, a in out.items()})
