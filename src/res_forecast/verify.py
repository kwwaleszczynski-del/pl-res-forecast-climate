"""Forecast verification metrics.

    MAE, RMSE, bias       of the median (P50) in MW
    nMAE                  MAE / mean observed generation of the evaluated hours
    CRPS                  continuous ranked probability score of the full predictive distribution (MW)
    CRPSS                 1 - CRPS / CRPS of the 30-day hourly climatology (positive = better than climatology)
    MAE skill             1 - MAE / MAE of persistence (same hour of the last observed day)
    coverage80, width80   share of observations inside [P10, P90] (ideal 80 %) and its mean width

PV is evaluated on daylight hours only (sun above the horizon at the centre of Poland).

Usage:
    table = summarize(hindcast_rows, by=["product", "tech", "lead"])
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _metrics(g: pd.DataFrame) -> pd.Series:
    err = g["q50"] - g["y"]
    mae = err.abs().mean()
    clim = g["crps_clim"].notna()
    pers = g["persistence"].notna()
    mae_pers = (g.loc[pers, "persistence"] - g.loc[pers, "y"]).abs().mean()
    mae_on_pers = err[pers].abs().mean()
    return pd.Series({
        "n_hours": len(g),
        "mean_obs_mw": g["y"].mean(),
        "mae_mw": mae,
        "rmse_mw": float(np.sqrt((err ** 2).mean())),
        "bias_mw": err.mean(),
        "nmae_pct": 100 * mae / g["y"].mean(),
        "crps_mw": g["crps"].mean(),
        "crpss_vs_clim_pct": 100 * (1 - g.loc[clim, "crps"].mean() / g.loc[clim, "crps_clim"].mean()),
        "mae_skill_vs_persistence_pct": 100 * (1 - mae_on_pers / mae_pers),
        "coverage80_pct": 100 * ((g["y"] >= g["q10"]) & (g["y"] <= g["q90"])).mean(),
        "width80_mw": (g["q90"] - g["q10"]).mean(),
    })


def evaluable(rows: pd.DataFrame) -> pd.DataFrame:
    ok = rows["y"].notna() & rows["q50"].notna() & rows["crps"].notna()
    daylight = (rows["tech"] != "pv") | rows["daylight"].astype(bool)
    return rows[ok & daylight]


def summarize(rows: pd.DataFrame, by=("product", "tech", "lead")) -> pd.DataFrame:
    df = evaluable(rows)
    return df.groupby(list(by)).apply(_metrics, include_groups=False).reset_index()
