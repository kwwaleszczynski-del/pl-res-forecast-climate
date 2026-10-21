"""National physical output (MW) from gridded weather: capacity-weighted sums of the point models in power.py.

Usage:
    weights = capacity_weights(grid)
    phys = physical_from_runs(runs, "ecmwf_ifs025", grid, weights)        # columns 'pv|1' .. 'wind|7'
    members = physical_from_ensemble(ens, grid, weights)                   # {'pv': (M, T), 'wind': (M, T)}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from res_forecast.config import HINDCAST_MODELS, WIND_HEIGHT, WIND_VARIABLE
from res_forecast.history import split_columns
from res_forecast.openmeteo import Ensemble
from res_forecast.power import MONTHLY_TEMP_C, hourly_mean_wind_power, pv_power, solar_geometry


CENTROID = (52.1, 19.4)  # lat, lon of Poland's centre, used for the time features of the PV calibration


def solar_features(times: pd.DatetimeIndex) -> np.ndarray:
    """(T, 2) slope features for the PV calibration: azimuth asymmetry and low-sun indicator at the country centre.

    Also returns NaN-free values at night (the PV calibration ignores night hours anyway).
    """
    g = solar_geometry(pd.DatetimeIndex(times), *CENTROID)
    elev = 90.0 - g["zenith"]
    az = np.clip((g["azimuth"] - 180.0) / 90.0, -1.5, 1.5)
    low = np.clip((20.0 - elev) / 20.0, 0.0, 1.0)
    return np.column_stack([az, low])


def is_daylight(times: pd.DatetimeIndex) -> np.ndarray:
    """Sun above the horizon at the country centre during the hour ending at ``times``."""
    return solar_geometry(pd.DatetimeIndex(times), *CENTROID)["zenith"] < 90.0


def _temp_or_climate(temp, times: pd.DatetimeIndex) -> np.ndarray:
    clim = MONTHLY_TEMP_C[(times - pd.Timedelta(minutes=30)).month.to_numpy() - 1]
    if temp is None:
        return clim
    t = np.asarray(temp, dtype=float)
    return np.where(np.isfinite(t), t, clim)


def _regular(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a gap-free hourly index so the wind interval averaging uses true neighbours."""
    full = pd.date_range(df.index.min(), df.index.max(), freq="1h")
    return df.reindex(full)


def physical_from_runs(runs: pd.DataFrame, model: str, grid: pd.DataFrame, weights: pd.DataFrame,
                       temp: pd.DataFrame | None = None) -> pd.DataFrame:
    """National physical PV and wind output (MW) for every lead of an archived-runs frame."""
    runs = _regular(runs)
    times = runs.index
    spec = HINDCAST_MODELS[model]
    out = {}
    pv_leads = spec.get("shortwave_radiation", [])
    if pv_leads:
        ghi = {lead: split_columns(runs, "shortwave_radiation", lead) for lead in pv_leads}
        tot = np.zeros((len(pv_leads), len(times)))
        for p in grid.index:
            geom = solar_geometry(times, grid.at[p, "lat"], grid.at[p, "lon"])
            t_air = _temp_or_climate(None if temp is None else temp[p].reindex(times).to_numpy(), times)
            stack = np.vstack([ghi[lead][p].to_numpy() for lead in pv_leads])
            tot += weights.at[p, "pv_mw"] * pv_power(stack, times, grid.at[p, "lat"], grid.at[p, "lon"],
                                                     temp_air=t_air, geometry=geom)
        for i, lead in enumerate(pv_leads):
            valid = ghi[lead].notna().all(axis=1).to_numpy()
            out[f"pv|{lead}"] = np.where(valid, tot[i], np.nan)
    wvar = WIND_VARIABLE.get(model)
    if wvar:
        for lead in spec.get(wvar, []):
            v = split_columns(runs, wvar, lead)
            cf = hourly_mean_wind_power(v.to_numpy().T, from_height=WIND_HEIGHT[wvar])  # (P, T)
            mw = weights.loc[v.columns, "wind_mw"].to_numpy() @ cf
            out[f"wind|{lead}"] = np.where(v.notna().all(axis=1).to_numpy(), mw, np.nan)
    return pd.DataFrame(out, index=times)


def physical_from_ensemble(ens: Ensemble, grid: pd.DataFrame, weights: pd.DataFrame, wind_variable: str | None,
                           temp: np.ndarray | None = None) -> dict:
    """National physical output per member: {'pv': (M, T), 'wind': (M, T)} in MW (missing techs omitted).

    ``temp`` is an optional (P, T) array of air temperature for the PV model.
    """
    times = ens.times
    out = {}
    if "shortwave_radiation" in ens.values:
        ghi = ens.values["shortwave_radiation"]  # (P, M, T)
        tot = np.zeros(ghi.shape[1:])
        for i, p in enumerate(grid.index):
            t_air = _temp_or_climate(None if temp is None else temp[i], times)
            tot += weights.at[p, "pv_mw"] * pv_power(ghi[i], times, grid.at[p, "lat"], grid.at[p, "lon"],
                                                     temp_air=t_air)
        out["pv"] = np.where(np.isfinite(ghi).all(axis=0), tot, np.nan)
    if wind_variable and wind_variable in ens.values:
        v = ens.values[wind_variable]  # (P, M, T)
        cf = hourly_mean_wind_power(v, from_height=WIND_HEIGHT[wind_variable])
        mw = np.einsum("p,pmt->mt", weights.loc[grid.index, "wind_mw"].to_numpy(), cf)
        out["wind"] = np.where(np.isfinite(v).all(axis=0), mw, np.nan)
    return out
