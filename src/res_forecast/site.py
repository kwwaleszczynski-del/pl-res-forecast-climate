"""Static website: one self-contained index.html with the data embedded as JSON (charts drawn by Plotly.js).

Every section is optional: when a step of the daily run failed, its part of the payload is missing and the page
shows a note instead of the chart, together with the status of each data source.

Usage:
    payload = build_payload(live=live, recent=recent_rows, hindcast=summary, climate=clim, status=status)
    render(payload, Path("site"))
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from res_forecast.config import TZ
from res_forecast.geo import load_capacity, load_geojson

TEMPLATE = Path(__file__).parent / "templates" / "index.html"
REPO_URL = "https://github.com/kwwaleszczynski-del/pl-res-forecast-climate"


def _clean(o):
    """JSON-safe: NaN -> None, numpy scalars -> Python, timestamps -> ISO strings in local time."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    if o is pd.NaT:
        return None
    if isinstance(o, pd.Timestamp):
        return o.tz_convert(TZ).isoformat() if o.tzinfo else o.isoformat()
    if isinstance(o, (datetime, date, pd.Period)):
        return str(o) if isinstance(o, pd.Period) else o.isoformat()
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else round(float(o), 3)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _times(idx: pd.DatetimeIndex) -> list[str]:
    """Hour-ending timestamps -> local wall-clock strings of the hour START (what a reader expects on a time axis)."""
    return [f"{t:%Y-%m-%dT%H:%M}" for t in (idx - pd.Timedelta(hours=1)).tz_convert(TZ)]


def live_payload(live, actual: pd.DataFrame | None) -> dict:
    fan = {"times": _times(live.times)}
    for tech, df in live.fan.items():
        fan[tech] = {f"p{int(round(c * 100)):02d}": df[c].to_numpy() for c in df.columns}
    out = {"issue": live.issue, "fan": fan, "daily": live.daily.to_dict("records"),
           "risk": live.risk.to_dict("records"), "risk_threshold_mw": live.risk_threshold_mw,
           "members": live.members}
    if actual is not None and len(actual):
        a = actual[actual.index > live.times[0] - pd.Timedelta(days=3)]
        out["actual"] = {"times": _times(a.index), "pv": a["pv"].to_numpy(), "wind": a["wind"].to_numpy()}
    return out


def recent_payload(rows: pd.DataFrame, product: str, days: int = 21) -> dict:
    """D+1 hindcast of the last ``days`` days (forecast fan vs PSE) for the 'forecast vs actual' chart."""
    r = rows[(rows["product"] == product) & (rows["lead"] == 1)]
    r = r[r["ts"] > r["ts"].max() - pd.Timedelta(days=days)]
    out = {"product": product}
    for tech, g in r.groupby("tech"):
        g = g.sort_values("ts")
        out[tech] = {"times": _times(pd.DatetimeIndex(g["ts"])), "y": g["y"].to_numpy(), "q10": g["q10"].to_numpy(),
                     "q50": g["q50"].to_numpy(), "q90": g["q90"].to_numpy()}
    return out


def climate_payload(clim: dict, months: int = 36) -> dict:
    nat = clim["national"].tail(months)
    ann = clim["annual"]
    ytd = dict(clim["ytd"])
    series = ytd.pop("series")
    return {
        "national": {"month": [str(m) for m in nat.index], **{c: nat[c].to_numpy() for c in nat.columns}},
        "annual": {"year": ann.index.to_list(), **{c: ann[c].to_numpy() for c in ann.columns}},
        "ytd": ytd,
        "ytd_series": {"year": series.index.to_list(), "pv_index_pct": series["pv_index_pct"].to_numpy(),
                       "wind_index_pct": series["wind_index_pct"].to_numpy()},
        "maps": clim["maps"], "exceedance": clim["exceedance"], "data_until": clim["data_until"],
    }


def build_payload(live=None, actual=None, recent=None, recent_metrics=None, hindcast=None, hindcast_meta=None,
                  live_metrics=None, live_series=None, climate=None, status=None, stale_note=None) -> dict:
    cap = load_capacity()
    p = {
        "generated": pd.Timestamp.now(tz="UTC"),
        "repo": REPO_URL,
        "capacity_mw": {"pv": float(cap["pv_mw"].sum()), "wind": float(cap["wind_mw"].sum())},
        "status": status or {},
        "stale_note": stale_note,
        "geojson": load_geojson(),
        "names": cap["name"].to_dict(),
    }
    if live is not None:
        p["live"] = live_payload(live, actual)
    if recent is not None and len(recent):
        p["recent"] = recent_payload(recent, "multi" if (recent["product"] == "multi").any() else "ecmwf_ifs025")
    if recent_metrics is not None and len(recent_metrics):
        p["recent_metrics"] = recent_metrics.to_dict("records")
    if hindcast is not None and len(hindcast):
        p["hindcast"] = hindcast.to_dict("records")
        p["hindcast_meta"] = hindcast_meta or {}
    if live_metrics is not None and len(live_metrics):
        p["live_metrics"] = live_metrics.to_dict("records")
    if climate is not None:
        p["climate"] = climate_payload(climate)
    return _clean(p)


def render(payload: dict, site_dir: Path) -> Path:
    site_dir.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", data)
    out = site_dir / "index.html"
    out.write_text(html, encoding="utf-8")
    (site_dir / "data.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (site_dir / ".nojekyll").write_text("")
    return out
