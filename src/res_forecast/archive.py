"""Archive of issued forecasts and their live verification.

Every daily run appends its hourly quantiles to ``<archive>/forecasts/YYYY-MM.parquet`` (one file per month of the
issue date). The GitHub Actions workflow keeps this directory on a separate ``forecast-archive`` branch, so the
main branch does not grow. Once forecasts have been archived for a few days, the D+1 forecasts issued each morning
are compared with PSE generation.

Usage:
    save_forecast(Path("archive"), live)
    table, series = live_verification(Path("archive"), actual)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from res_forecast.config import TZ

log = logging.getLogger(__name__)
COLS = ["p05", "p10", "p25", "p50", "p75", "p90", "p95"]


def save_forecast(archive_dir: Path, live) -> Path:
    rows = []
    for tech, fan in live.fan.items():
        df = fan.copy()
        df.columns = COLS
        df["tech"] = tech
        rows.append(df)
    out = pd.concat(rows).rename_axis("ts").reset_index()
    out["issue_utc"] = live.issue.tz_convert("UTC")
    out["issue_date"] = live.issue.tz_convert(TZ).date().isoformat()
    path = archive_dir / "forecasts" / f"{live.issue.tz_convert(TZ):%Y-%m}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        old = old[old["issue_date"] != out["issue_date"].iloc[0]]  # a re-run on the same day replaces the earlier one
        out = pd.concat([old, out], ignore_index=True)
    out[COLS] = out[COLS].astype("float32")
    out.to_parquet(path, index=False)
    log.info("archived %d rows to %s", len(out), path)
    return path


def load_archive(archive_dir: Path) -> pd.DataFrame:
    files = sorted((archive_dir / "forecasts").glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=["ts", "tech", "issue_date", *COLS])
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def day_ahead(archive: pd.DataFrame) -> pd.DataFrame:
    """Rows forecasting the local day after the issue day (D+1)."""
    if archive.empty:
        return archive
    local_day = (pd.to_datetime(archive["ts"], utc=True) - pd.Timedelta(hours=1)).dt.tz_convert(TZ).dt.date
    issue = pd.to_datetime(archive["issue_date"]).dt.date
    lead = np.array([(d - i).days for d, i in zip(local_day, issue)])
    return archive[lead == 1]


def live_verification(archive_dir: Path, actual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metrics of the archived D+1 forecasts against PSE and the joined hourly series (empty while too short)."""
    d1 = day_ahead(load_archive(archive_dir))
    if d1.empty:
        return pd.DataFrame(), pd.DataFrame()
    d1 = d1.assign(ts=pd.to_datetime(d1["ts"], utc=True))
    obs = actual.stack().rename("y").rename_axis(["ts", "tech"]).reset_index()
    obs["ts"] = pd.to_datetime(obs["ts"], utc=True)
    j = d1.merge(obs, on=["ts", "tech"], how="inner").dropna(subset=["y", "p50"])
    if j.empty:
        return pd.DataFrame(), pd.DataFrame()
    j = j[(j["tech"] != "pv") | (j["p95"] > 0) | (j["y"] > 0)]
    levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

    def metrics(g):
        pin = np.mean([np.mean(np.maximum(lv * (g["y"] - g[c]), (lv - 1) * (g["y"] - g[c])))
                       for lv, c in zip(levels, COLS)])
        return pd.Series({
            "days": g["issue_date"].nunique(), "n_hours": len(g),
            "mae_mw": (g["p50"] - g["y"]).abs().mean(),
            "nmae_pct": 100 * (g["p50"] - g["y"]).abs().mean() / g["y"].mean(),
            "pinball_mw": pin,
            "coverage80_pct": 100 * ((g["y"] >= g["p10"]) & (g["y"] <= g["p90"])).mean(),
        })

    table = j.groupby("tech").apply(metrics, include_groups=False).reset_index()
    return table, j.sort_values(["tech", "ts"])
