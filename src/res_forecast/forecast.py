"""Live forecast: ensemble members -> national power -> calibrated and dressed fan for today .. D+7, plus KPIs.

Members of ECMWF IFS ENS (51), DWD ICON-EPS (40) and NOAA GEFS (31) are converted to national PV and wind output,
each member is calibrated with the MOS of its model's deterministic sibling for the lead day of the hour, and the
members are dressed (see calibration.py). Each model's members share a weight proportional to the inverse square of
the model's recent out-of-sample error. GEFS has no hub-height wind and is used for PV only; ICON-EPS wind ends after
about five days.

If an ensemble request fails, the deterministic run of the same model (one member) is used instead, and if that
fails too, the model is left out; the page shows which sources were used.

Usage:
    live = run_live(om, grid, weights, calibs, issue=pd.Timestamp.now(tz="UTC"))
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from res_forecast.calibration import N_RESID, dressing_samples, weighted_quantiles
from res_forecast.config import ENSEMBLE_MODELS, LEADS, TZ
from res_forecast.openmeteo import OpenMeteo
from res_forecast.physical import physical_from_ensemble, solar_features

log = logging.getLogger(__name__)
FAN_LEVELS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
DAILY_LEVELS = (0.1, 0.5, 0.9)


@dataclass
class Live:
    issue: pd.Timestamp
    times: pd.DatetimeIndex
    fan: dict                       # tech -> DataFrame (index times, columns FAN_LEVELS), MW
    daily: pd.DataFrame             # local day, tech -> energy quantiles (MWh), pv peak
    risk: pd.DataFrame              # local day -> probability of a low wind+solar day
    members: dict = field(default_factory=dict)   # model -> {"tech": n_members}
    status: dict = field(default_factory=dict)    # source -> message
    risk_threshold_mw: float = float("nan")


def fetch_members(om: OpenMeteo, grid: pd.DataFrame, weights: pd.DataFrame, status: dict, forecast_days: int = 8,
                  det_models=None) -> tuple[pd.DatetimeIndex, dict]:
    """Physical national output per member: {det_model: {"pv": (M, T), "wind": (M, T)}} on a common time axis.

    Only ensembles whose deterministic sibling is in ``det_models`` (i.e. has a calibration) are downloaded.
    """
    temp = None
    try:
        t = om.ensemble(grid, "ecmwf_ifs025_ensemble_mean", ["temperature_2m"], forecast_days, n_members=1)
        temp = (t.times, t.values["temperature_2m"][:, 0, :])
        status["temperature (ECMWF ENS mean)"] = "ok"
    except Exception as e:  # the PV model falls back to climatological temperature
        status["temperature (ECMWF ENS mean)"] = f"failed, climatology used ({type(e).__name__})"
    raw, axes = {}, []
    for model, spec in ENSEMBLE_MODELS.items():
        if det_models is not None and spec["det"] not in det_models:
            continue
        variables = spec["variables"]
        wind_var = next((v for v in variables if v.startswith("wind_speed")), None)
        try:
            ens = om.ensemble(grid, model, variables, forecast_days, n_members=spec["members"])
            status[model] = f"ok, {ens.values[variables[0]].shape[1]} members"
        except Exception as e:
            log.warning("ensemble %s failed: %s", model, e)
            try:
                ens = om.forecast(grid, spec["det"], variables, forecast_days)
                status[model] = f"ensemble failed ({type(e).__name__}), deterministic {spec['det']} used"
            except Exception as e2:
                status[model] = f"failed ({type(e2).__name__})"
                continue
        t_air = None
        if temp is not None:
            t_air = pd.DataFrame(temp[1].T, index=temp[0]).reindex(ens.times).to_numpy().T
        raw[spec["det"]] = (ens.times, physical_from_ensemble(ens, grid, weights, wind_var, t_air))
        axes.append(ens.times)
    if not raw:
        raise RuntimeError("no weather forecast could be downloaded")
    times = axes[0]
    for a in axes[1:]:
        times = times.union(a)
    out = {}
    for det, (t, phys) in raw.items():
        out[det] = {tech: pd.DataFrame(arr.T, index=t).reindex(times).to_numpy().T for tech, arr in phys.items()}
    return times, out


def lead_days(times: pd.DatetimeIndex, issue: pd.Timestamp) -> np.ndarray:
    """Local calendar-day offset of each hour (interval start) from the issue day."""
    start_local = (times - pd.Timedelta(hours=1)).tz_convert(TZ)
    issue_day = issue.tz_convert(TZ).normalize().tz_localize(None)
    return (start_local.normalize().tz_localize(None) - issue_day).days.to_numpy()


def day_fraction(times: pd.DatetimeIndex) -> np.ndarray:
    """Position of each hour (interval mid-point) within its local day, 0..1."""
    mid = (times - pd.Timedelta(minutes=30)).tz_convert(TZ)
    return ((mid.hour + mid.minute / 60) / 24).to_numpy()


def _member_set(raw: dict, calibs: dict, tech: str, lead: int, hours: np.ndarray, feats):
    """Calibrated members (M, H), their MOS list, member weights and model names for one tech and lead."""
    blocks, mos_list, w, names = [], [], [], []
    models = [m for m in raw if tech in raw[m] and (m, tech, lead) in calibs]
    for m in models:
        x = raw[m][tech][:, hours]
        if not np.isfinite(x).any():
            continue
        mos = calibs[(m, tech, lead)]
        blocks.append(mos.predict(x, None if feats is None else feats[hours]))
        mos_list += [mos] * x.shape[0]
        names += [m] * x.shape[0]
        model_w = mos.weight if np.isfinite(mos.weight) and mos.weight > 0 else 1.0
        w.append(np.full(x.shape[0], model_w / x.shape[0]))
    if not blocks:
        return None, [], None, []
    return np.vstack(blocks), mos_list, np.concatenate(w), names


def run_live(raw: dict, times: pd.DatetimeIndex, calibs: dict, issue: pd.Timestamp, actual: pd.DataFrame,
             k_min: float) -> Live:
    leads = lead_days(times, issue)
    keep = (leads >= 0) & (leads <= max(LEADS))
    times, leads = times[keep], leads[keep]
    raw = {m: {t: a[:, keep] for t, a in d.items()} for m, d in raw.items()}
    feats_pv = solar_features(times)
    fan, cal_members = {}, {}
    for tech in ("pv", "wind"):
        q = np.full((len(FAN_LEVELS), len(times)), np.nan)
        cal_members[tech] = {}
        feats = feats_pv if tech == "pv" else None
        for lead in sorted(set(leads)):
            hours = np.flatnonzero(leads == lead)
            members, mos_list, w, names = _member_set(raw, calibs, tech, max(int(lead), 1), hours, feats)
            if members is None:
                continue
            samples, sw = dressing_samples(members, mos_list, w, k_min=k_min)
            q[:, hours] = weighted_quantiles(samples, sw, FAN_LEVELS)
            cal_members[tech][int(lead)] = (hours, members, mos_list, w, np.array(names))
            # blend towards the next lead's calibration through the day, so the fan has no steps at midnight
            if lead >= 1 and lead < max(LEADS):
                nxt = _member_set(raw, calibs, tech, int(lead) + 1, hours, feats)
                if nxt[0] is not None:
                    s2, w2 = dressing_samples(nxt[0], nxt[1], nxt[2], k_min=k_min)
                    q2 = weighted_quantiles(s2, w2, FAN_LEVELS)
                    frac = day_fraction(times[hours])
                    q[:, hours] = np.where(np.isfinite(q2), (1 - frac) * q[:, hours] + frac * q2, q[:, hours])
        fan[tech] = pd.DataFrame(q.T, index=times, columns=FAN_LEVELS)
    daily = _daily(times, leads, fan, cal_members, k_min)
    risk, thr = _low_res_risk(times, leads, cal_members, actual, k_min)
    counts = {m: {t: int(a.shape[0]) for t, a in d.items()} for m, d in raw.items()}
    return Live(issue, times, fan, daily, risk, counts, risk_threshold_mw=thr)


def _dressed_daily(totals: np.ndarray, resid_q: np.ndarray, w: np.ndarray, k_min: float):
    """Samples of a daily total: member totals (M,) each dressed with its model's daily error quantiles (M, R)."""
    wm = w / w.sum()
    mean = (wm * totals).sum()
    spread2 = (wm * (totals - mean) ** 2).sum()
    sigma2 = (wm * resid_q.var(axis=1)).sum()
    k = np.sqrt(max(k_min ** 2, 1 - spread2 / sigma2)) if sigma2 > 0 and len(totals) > 1 else 1.0
    samples = np.clip(totals[:, None] + k * resid_q, 0, None).ravel()
    return samples, np.repeat(wm, resid_q.shape[1]) / resid_q.shape[1]


def _day_groups(times: pd.DatetimeIndex, leads: np.ndarray):
    local_day = (times - pd.Timedelta(hours=1)).tz_convert(TZ).normalize().tz_localize(None)
    for lead in sorted(set(leads)):
        hours = np.flatnonzero(leads == lead)
        yield int(lead), local_day[hours[0]].date(), hours


def _daily(times, leads, fan, cal_members, k_min) -> pd.DataFrame:
    rows = []
    for lead, day, hours in _day_groups(times, leads):
        row = {"day": day, "lead": lead, "hours": len(hours)}
        for tech in ("pv", "wind"):
            entry = cal_members[tech].get(lead)
            if entry is None or len(hours) < 20:
                continue
            h, members, mos_list, w, _ = entry
            ok = np.isfinite(members).all(axis=1)
            if not ok.any():
                continue
            totals = members[ok].sum(axis=1)
            resid = np.vstack([m.daily_resid_q for m, good in zip(mos_list, ok) if good]) * len(hours) / 24.0
            s, sw = _dressed_daily(totals, resid, w[ok], k_min)
            q = weighted_quantiles(s[None, :], sw[None, :], DAILY_LEVELS)[:, 0]
            for lv, v in zip(DAILY_LEVELS, q):
                row[f"{tech}_mwh_p{int(lv * 100)}"] = v
            p50 = fan[tech][0.5].to_numpy()[hours]
            i = int(np.nanargmax(p50))
            row[f"{tech}_peak_mw_p50"] = p50[i]
            row[f"{tech}_peak_mw_p10"] = fan[tech][0.1].to_numpy()[hours][i]
            row[f"{tech}_peak_mw_p90"] = fan[tech][0.9].to_numpy()[hours][i]
            row[f"{tech}_peak_time"] = times[hours][i]
        rows.append(row)
    return pd.DataFrame(rows)


def low_res_threshold(actual: pd.DataFrame, days: int = 365) -> float:
    """10th percentile of the daily mean wind+PV output (MW) over the last ``days`` days of PSE data."""
    tot = (actual["pv"] + actual["wind"]).dropna()
    tot = tot[tot.index > tot.index.max() - pd.Timedelta(days=days)]
    daily = tot.groupby((tot.index - pd.Timedelta(hours=1)).tz_convert(TZ).date).agg(["mean", "count"])
    return float(daily.loc[daily["count"] >= 20, "mean"].quantile(0.1))


def _low_res_risk(times, leads, cal_members, actual, k_min) -> tuple[pd.DataFrame, float]:
    """P(daily mean wind+PV < threshold) from the models that forecast both technologies (members paired by index)."""
    thr = low_res_threshold(actual) if actual is not None and len(actual) else float("nan")
    rows = []
    rng = np.random.default_rng(0)
    for lead, day, hours in _day_groups(times, leads):
        pv, wind = cal_members["pv"].get(lead), cal_members["wind"].get(lead)
        if pv is None or wind is None or len(hours) < 20 or not np.isfinite(thr):
            continue
        totals, resid, w = [], [], []
        for m in dict.fromkeys(wind[4]):
            iw, ip = np.flatnonzero(wind[4] == m), np.flatnonzero(pv[4] == m)
            if len(iw) != len(ip):
                continue
            tw, tp = wind[1][iw], pv[1][ip]
            ok = np.isfinite(tw).all(axis=1) & np.isfinite(tp).all(axis=1)
            if not ok.any():
                continue
            totals.append((tw[ok].sum(axis=1) + tp[ok].sum(axis=1)) / len(hours))
            # combined daily error (MWh/day -> mean MW): random pairs of the PV and wind daily error quantiles
            p_q, w_q = pv[2][ip[0]].daily_resid_q, wind[2][iw[0]].daily_resid_q
            comb = np.sort(rng.choice(p_q, 40 * N_RESID) + rng.choice(w_q, 40 * N_RESID))[20::40] / 24.0
            resid.append(np.repeat(comb[None, :], ok.sum(), axis=0))
            w.append(np.full(ok.sum(), 1.0 / ok.sum()))
        if not totals:
            continue
        s, sw = _dressed_daily(np.concatenate(totals), np.vstack(resid), np.concatenate(w), k_min)
        rows.append({"day": day, "lead": lead, "p_low": float((sw * (s < thr)).sum() / sw.sum()),
                     "mean_mw_p50": float(weighted_quantiles(s[None, :], sw[None, :], [0.5])[0, 0])})
    return pd.DataFrame(rows), thr
