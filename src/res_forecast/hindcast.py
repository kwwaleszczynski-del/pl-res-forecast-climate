"""Rolling-origin hindcast of the D+1..D+6 forecasts on archived model runs, with no look-ahead.

For every issue day I and lead N the calibration is fitted on the ``window`` days up to I-1 (the actual generation
of day I-1 is published by PSE by the morning of day I) using the archived lead-N forecasts for those days, and then
applied to the lead-N forecast for day I+N. :func:`as_of` first restricts the archived runs to those published by
the 08:40 UTC issue time; D+7 would need an eight-day archive and is not scored. Products evaluated:

    ecmwf / icon / gfs   one deterministic model, dressed with its own error distribution (k = 1)
    multi                the available models as a small multi-model ensemble (weighted by recent skill) with
                         spread-dependent dressing, i.e. the same code path as the live 122-member ensemble

Reference forecasts: persistence (same hour of day I-1) and a probabilistic climatology (the same hour of the
30 days up to I-1).

Usage:
    phys = load_physical(om, data_dir, grid, weights, start, end, temperature=False)  # national MW per model, lead
    phys = {m: as_of(f) for m, f in phys.items()}             # only runs published by the issue time
    hc = run_hindcast(phys, actual, window=60, k_min=0.5)     # one row per (product, tech, lead, hour)
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from res_forecast.calibration import K_MIN, crps_sample, dressing_samples, fit_mos, weighted_quantiles
from res_forecast.config import HINDCAST_MODELS, HINDCAST_START, ISSUE_HOUR_UTC, LEADS, RUN_DELAY_H
from res_forecast.history import load_previous_runs, split_columns
from res_forecast.openmeteo import OpenMeteo
from res_forecast.physical import is_daylight, physical_from_runs, solar_features

log = logging.getLogger(__name__)
TECHS = ("pv", "wind")
OUT_LEVELS = (0.1, 0.5, 0.9)
CLIM_DAYS = 30


def load_physical(om: OpenMeteo | None, data_dir: Path, grid: pd.DataFrame, weights: pd.DataFrame, start: date,
                  end: date, models=tuple(HINDCAST_MODELS), cached_only: bool = False,
                  temperature: bool = True) -> dict[str, pd.DataFrame]:
    """National physical output for each archived model ({model: frame with columns 'tech|lead'}).

    With ``temperature=False`` the PV model uses monthly climatological air temperature instead of the archived
    day-1 temperature, whose runs start after the issue time for almost every hour (used by the hindcast).
    """
    runs = {m: load_previous_runs(om, grid, m, start, end, data_dir / "runs", cached_only=cached_only) for m in models}
    runs = {m: r for m, r in runs.items() if len(r)}
    temp = None
    if temperature and "ecmwf_ifs025" in runs:
        temp = split_columns(runs["ecmwf_ifs025"], "temperature_2m", 1)
    out = {}
    for m, r in runs.items():
        out[m] = physical_from_runs(r, m, grid, weights, temp=temp)
        log.info("physical output %s: %d hours, %d series", m, len(out[m]), out[m].shape[1])
    return out


def as_of(phys: pd.DataFrame, issue_hour: float = ISSUE_HOUR_UTC, delay_h: float = RUN_DELAY_H) -> pd.DataFrame:
    """Rebuild the 'tech|N' columns from runs that were published by the issue time.

    The Previous Runs API serves for each valid hour t the run started N x 24 h before t (or the latest one before
    that). For the forecast of day I+N issued at ``issue_hour`` UTC on day I this run starts at day I + hour(t), so
    it is only out for the first hours of the day; every later hour takes the N+1 column, a run started on day I-1.
    A lead whose N+1 column is not archived is dropped.
    """
    fresh = phys.index.hour.to_numpy() + delay_h <= issue_hour
    out = {}
    for col in phys.columns:
        tech, lead = col.split("|")
        older = f"{tech}|{int(lead) + 1}"
        if older in phys.columns:
            out[col] = np.where(fresh, phys[col].to_numpy(), phys[older].to_numpy())
    return pd.DataFrame(out, index=phys.index)


def _day_numbers(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.tz_convert("UTC").normalize() - pd.Timestamp("2000-01-01", tz="UTC")).days.to_numpy()


def day_number(d: date) -> int:
    return int(_day_numbers(pd.DatetimeIndex([pd.Timestamp(d, tz="UTC")]))[0])


def _prepare(phys: dict[str, pd.DataFrame], actual: pd.DataFrame):
    idx = actual.index
    for f in phys.values():
        idx = idx.union(f.index)
    return idx, actual.reindex(idx), {m: f.reindex(idx) for m, f in phys.items()}, _day_numbers(idx)


def rolling_calibration(x: np.ndarray, y: np.ndarray, dayno: np.ndarray, tech: str, lead: int, window: int,
                        issues, inflate: float = 1.0, feats: np.ndarray | None = None):
    """Yield (issue, mos, test_rows, prediction) for consecutive issue days without look-ahead.

    At issue day I the regression uses the ``window`` days I-window .. I-1, and the error distribution uses the
    out-of-sample predictions this generator made earlier for those days (lead-``lead`` forecasts).
    """
    oos = np.full(len(x), np.nan)
    for issue in issues:
        train = (dayno >= issue - window) & (dayno <= issue - 1)
        try:
            mos = fit_mos(x[train], y[train], tech, feats=None if feats is None else feats[train],
                          oos=(oos[train], y[train]), oos_days=dayno[train], inflate=inflate)
        except ValueError:
            continue
        test = np.flatnonzero(dayno == issue + lead)
        pred = mos.predict(x[test], None if feats is None else feats[test])
        oos[test] = pred
        yield issue, mos, test, pred


def calibrations_at(phys: dict[str, pd.DataFrame], actual: pd.DataFrame, issue: date, window: int = 60,
                    inflate: float = 1.0, leads=LEADS) -> dict[tuple[str, str, int], object]:
    """Calibrations valid on ``issue`` for every (model, tech, lead) with archived runs: {(model, tech, lead): MOS}."""
    idx, actual, phys, dayno = _prepare(phys, actual)
    feats_pv = solar_features(idx)
    issue_no = day_number(issue)
    out = {}
    for m, f in phys.items():
        for tech in TECHS:
            for lead in leads:
                col = f"{tech}|{lead}"
                if col not in f.columns:
                    continue
                for i, mos, _, _ in rolling_calibration(
                        f[col].to_numpy(), actual[tech].to_numpy(), dayno, tech, lead, window,
                        range(issue_no - 2 * window, issue_no + 1), inflate, feats_pv if tech == "pv" else None):
                    if i == issue_no:
                        out[(m, tech, lead)] = mos
    return out


def run_hindcast(phys: dict[str, pd.DataFrame], actual: pd.DataFrame, window: int = 60, k_min: float = K_MIN,
                 inflate: float = 1.0, first_issue: date | None = None, last_issue: date | None = None, leads=LEADS,
                 products=("single", "multi"), techs=TECHS) -> pd.DataFrame:
    """Hourly hindcast rows: product, tech, lead, ts, y, q10, q50, q90, crps, persistence, crps_clim, daylight."""
    idx, actual, phys, dayno = _prepare(phys, actual)
    hour = idx.hour.to_numpy()
    feats_pv = solar_features(idx)
    daylight = is_daylight(idx)
    first = day_number(first_issue or HINDCAST_START)
    first = max(first, _day_numbers(actual.dropna(how="all").index[:1])[0] + window)
    last = day_number(last_issue) if last_issue else int(dayno.max())
    rows = []
    d0 = dayno.min()
    for tech in techs:
        feats = feats_pv if tech == "pv" else None
        y_all = actual[tech].to_numpy()
        Y = np.full((dayno.max() - d0 + 1, 24), np.nan)
        Y[dayno - d0, hour] = y_all
        for lead in leads:
            models = [m for m, f in phys.items() if f"{tech}|{lead}" in f.columns]
            fits = {m: {i: (mos, pred) for i, mos, _, pred in rolling_calibration(
                        phys[m][f"{tech}|{lead}"].to_numpy(), y_all, dayno, tech, lead, window,
                        range(first - window, last + 1), inflate, feats)} for m in models}
            for issue in range(first, last + 1):
                test = np.flatnonzero(dayno == issue + lead)
                fitted = {m: fits[m][issue] for m in models if issue in fits[m]}
                if len(test) == 0 or not fitted:
                    continue
                y = y_all[test]
                # references from the day x hour matrix of observations
                hrs = hour[test]
                persistence = Y[issue - 1 - d0, hrs] if issue - 1 - d0 >= 0 else np.full(len(test), np.nan)
                clim_samples = Y[max(issue - CLIM_DAYS - d0, 0):issue - d0, :][:, hrs].T     # (n_test, <=30)
                cw = np.isfinite(clim_samples).astype(float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    crps_clim = crps_sample(np.nan_to_num(clim_samples), cw + 1e-12, np.nan_to_num(y))
                crps_clim = np.where(cw.sum(axis=1) >= 10, crps_clim, np.nan)
                variants = []
                if "single" in products:
                    variants += [(m, [m]) for m in fitted]
                if "multi" in products:
                    variants.append(("multi", list(fitted)))  # the published product, even when one model is left
                for name, ms in variants:
                    members = np.vstack([fitted[m][1] for m in ms])
                    mw = np.array([fitted[m][0].weight for m in ms])
                    samples, weights = dressing_samples(members, [fitted[m][0] for m in ms],
                                                        mw if mw.sum() > 0 else None, k_min=k_min)
                    q = weighted_quantiles(samples, weights, OUT_LEVELS)
                    crps = crps_sample(samples, weights, np.nan_to_num(y))
                    ok = np.isfinite(members).any(axis=0) & np.isfinite(y)
                    rows.append(pd.DataFrame({
                        "product": name, "tech": tech, "lead": lead, "ts": idx[test], "y": y,
                        "q10": np.where(ok, q[0], np.nan), "q50": np.where(ok, q[1], np.nan),
                        "q90": np.where(ok, q[2], np.nan), "crps": np.where(ok, crps, np.nan),
                        "persistence": persistence, "crps_clim": crps_clim, "daylight": daylight[test],
                    }))
        log.info("hindcast %s done", tech)
    return pd.concat(rows, ignore_index=True)


PRODUCT_NAMES = {"multi": "Multi-model", "ecmwf_ifs025": "ECMWF IFS", "icon_seamless": "DWD ICON",
                 "gfs_seamless": "NOAA GFS"}
