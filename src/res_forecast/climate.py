"""ERA5 resource climatology for Poland: monthly anomalies, "how was this year" and P50/P90 of annual yield.

Inputs are daily ERA5 values on the 1-degree grid (Open-Meteo archive API): the daily sum of global horizontal
irradiance (GHI) and the daily mean wind speed at 100 m. Normals are the 1991-2020 means of every calendar month.

National series weight the grid points by installed capacity (PV capacity for irradiance, wind capacity for wind),
so they describe the resource where the fleet actually is; voivodeship maps are plain area means.

Annual energy uses daily transfer functions derived from one year of hourly ERA5 run through the same physical models
as the forecast (power.py):
    PV    daily yield = daily GHI x monthly POA/AC gain of each point (kWh per kWp)
    wind  daily capacity factor = f(daily mean speed), the mean hourly fleet power curve output of all days with that
          daily mean speed (a "daily power curve", which accounts for the spread of hourly speeds within a day)
The yields are generic (fixed fleet layout of today, generic modules and turbines, no curtailment), so they are
reported both in absolute terms and as an index relative to the 1991-2020 mean; the year-to-year variability is the
robust part.

Usage:
    gain, curve = pv_gain(hourly, grid), daily_wind_curve(hourly)
    tables = base_tables(daily, grid, weights, gain, curve)     # monthly fields + national daily yields
    clim = summarize(tables, grid, weights)                       # anomalies, maps, annual P50/P90, year to date
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from res_forecast.config import NORMAL_PERIOD
from res_forecast.geo import region_means
from res_forecast.history import split_columns
from res_forecast.power import KMH_PER_MS, pv_power, solar_geometry, wind_power

MJ_PER_KWH = 3.6


def daily_fields(daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Daily GHI (kWh/m2) and mean 100 m wind (m/s) as (day x point) frames, index naive dates."""
    ghi = split_columns(daily, "shortwave_radiation_sum") / MJ_PER_KWH
    wind = split_columns(daily, "wind_speed_100m_mean") / KMH_PER_MS
    for f in (ghi, wind):
        f.index = f.index.tz_convert("UTC").tz_localize(None).normalize()
    return {"ghi": ghi, "wind": wind}


def monthly(fields: dict[str, pd.DataFrame], min_share: float = 0.9) -> dict[str, pd.DataFrame]:
    """Monthly GHI totals (kWh/m2) and mean wind (m/s); months with < ``min_share`` of days are NaN."""
    out = {}
    for name, f in fields.items():
        g = f.groupby(f.index.to_period("M"))
        days = pd.Series(g.size().index.days_in_month, index=g.size().index)
        valid = g.count().min(axis=1) >= min_share * days
        m = g.sum(min_count=1) if name == "ghi" else g.mean()
        if name == "ghi":
            m = m.div(g.count()).mul(days, axis=0)  # scale to a full month when a few days are missing
        out[name] = m.where(valid, np.nan)
    return out


def normals(monthly_fields: dict[str, pd.DataFrame], period=NORMAL_PERIOD) -> dict[str, pd.DataFrame]:
    out = {}
    for name, m in monthly_fields.items():
        sel = m[(m.index.year >= period[0]) & (m.index.year <= period[1])]
        out[name] = sel.groupby(sel.index.month).mean()
    return out


def anomalies_pct(monthly_fields: dict, norm: dict) -> dict[str, pd.DataFrame]:
    out = {}
    for name, m in monthly_fields.items():
        base = norm[name].reindex(m.index.month).set_index(m.index)
        out[name] = 100 * (m / base - 1)
    return out


def weighted(frame: pd.DataFrame, w: pd.Series) -> pd.Series:
    w = w.reindex(frame.columns).fillna(0)
    return (frame * w).sum(axis=1, min_count=1) / w.sum()


# -- daily transfer functions --------------------------------------------------------------------------------------

def pv_gain(hourly: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Monthly ratio of modelled PV yield (kWh/kWp) to GHI (kWh/m2) per point, from hourly ERA5 (12 x points)."""
    ghi = split_columns(hourly, "shortwave_radiation")
    times = ghi.index
    out = {}
    for p in grid.index:
        g = solar_geometry(times, grid.at[p, "lat"], grid.at[p, "lon"])
        y = pv_power(ghi[p].to_numpy(), times, grid.at[p, "lat"], grid.at[p, "lon"], geometry=g)
        df = pd.DataFrame({"y": y, "ghi": ghi[p].to_numpy() / 1000.0}, index=times)
        s = df.groupby(df.index.month).sum()
        out[p] = s["y"] / s["ghi"]
    return pd.DataFrame(out)


def daily_wind_curve(hourly: pd.DataFrame, step: float = 0.5) -> pd.Series:
    """Mean daily capacity factor as a function of the daily mean 100 m wind speed (m/s), pooled over points."""
    v = split_columns(hourly, "wind_speed_100m")
    cf = pd.DataFrame(wind_power(v.to_numpy(), from_height=100.0), index=v.index, columns=v.columns)
    day = v.index.tz_convert("UTC").normalize()
    v_d, cf_d = v.groupby(day).mean() / KMH_PER_MS, cf.groupby(day).mean()
    df = pd.DataFrame({"v": v_d.to_numpy().ravel(), "cf": cf_d.to_numpy().ravel()}).dropna()
    df["bin"] = (df["v"] / step).round() * step
    curve = df.groupby("bin")["cf"].agg(["mean", "count"])
    curve = curve[curve["count"] >= 5]["mean"]
    curve = pd.Series(np.maximum.accumulate(curve.to_numpy()), index=curve.index)  # monotone
    return pd.concat([pd.Series({0.0: 0.0}), curve]).groupby(level=0).last()


def apply_daily_wind_curve(v: pd.DataFrame, curve: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(np.interp(v.to_numpy(), curve.index.to_numpy(), curve.to_numpy()), index=v.index,
                        columns=v.columns).where(v.notna())


# -- annual yields ---------------------------------------------------------------------------------------------------

def daily_yields(fields: dict, gain: pd.DataFrame, curve: pd.Series, weights: pd.DataFrame) -> pd.DataFrame:
    """National daily PV yield (kWh/kWp) and wind capacity factor (capacity weighted)."""
    ghi = fields["ghi"]
    g = gain.reindex(index=ghi.index.month, columns=ghi.columns).to_numpy()
    pv = pd.DataFrame(ghi.to_numpy() * g, index=ghi.index, columns=ghi.columns)
    cf = apply_daily_wind_curve(fields["wind"], curve)
    return pd.DataFrame({"pv_kwh_kwp": weighted(pv, weights["pv_mw"]), "wind_cf": weighted(cf, weights["wind_mw"])})


def annual_table(yields: pd.DataFrame, period=NORMAL_PERIOD) -> pd.DataFrame:
    """Complete calendar years: PV specific yield, wind full-load hours and both as % of the normal-period mean."""
    g = yields.groupby(yields.index.year)
    n = g.size()
    full = n[n >= 360].index
    ann = pd.DataFrame({"pv_kwh_kwp": g["pv_kwh_kwp"].sum(), "wind_flh": g["wind_cf"].mean() * 8760}).loc[full]
    base = ann.loc[(ann.index >= period[0]) & (ann.index <= period[1])].mean()
    ann["pv_index_pct"] = 100 * ann["pv_kwh_kwp"] / base["pv_kwh_kwp"]
    ann["wind_index_pct"] = 100 * ann["wind_flh"] / base["wind_flh"]
    ann.index.name = "year"
    return ann


def exceedance(values: pd.Series) -> dict:
    """P50 / P90 / P10 (exceeded in 50 / 90 / 10 % of years), empirical and from a normal fit."""
    v = values.dropna()
    mu, sd = v.mean(), v.std(ddof=1)
    return {"n_years": int(len(v)), "mean": mu, "sd_pct": 100 * sd / mu,
            "p50": float(v.quantile(0.5)), "p90": float(v.quantile(0.1)), "p10": float(v.quantile(0.9)),
            "p90_normal": float(mu - 1.2816 * sd), "p90_over_p50_pct": float(100 * v.quantile(0.1) / v.quantile(0.5))}


def year_to_date(yields: pd.DataFrame, period=NORMAL_PERIOD) -> dict:
    """Current year from 1 Jan to the last day with data, against the same period of every earlier year."""
    last = yields.dropna().index.max()
    year = last.year
    doy_mask = (yields.index.dayofyear <= last.dayofyear)
    same = yields[doy_mask].groupby(yields.index[doy_mask].year).agg({"pv_kwh_kwp": "sum", "wind_cf": "mean"})
    same = same[same.index <= year]
    base = same.loc[(same.index >= period[0]) & (same.index <= period[1])].mean()
    cur = same.loc[year]
    rank = lambda col: int((same[col] > cur[col]).sum() + 1)  # 1 = best year
    return {"year": int(year), "until": last.date().isoformat(), "n_years": int(len(same)),
            "pv_index_pct": float(100 * cur["pv_kwh_kwp"] / base["pv_kwh_kwp"]),
            "wind_index_pct": float(100 * cur["wind_cf"] / base["wind_cf"]),
            "pv_rank": rank("pv_kwh_kwp"), "wind_rank": rank("wind_cf"),
            "series": same.assign(pv_index_pct=100 * same["pv_kwh_kwp"] / base["pv_kwh_kwp"],
                                  wind_index_pct=100 * same["wind_cf"] / base["wind_cf"])}


def base_tables(daily: pd.DataFrame, grid: pd.DataFrame, weights: pd.DataFrame, gain: pd.DataFrame,
                curve: pd.Series) -> dict:
    """Monthly point fields and national daily yields from daily ERA5 (the inputs of :func:`summarize`)."""
    fields = daily_fields(daily)
    mon = monthly(fields)
    return {"ghi": mon["ghi"], "wind": mon["wind"], "yields": daily_yields(fields, gain, curve, weights),
            "data_until": fields["ghi"].dropna(how="all").index.max().date().isoformat()}


def extend(base: dict, recent: dict) -> dict:
    """Append (and overwrite) the months/days of ``recent`` onto ``base``."""
    out = {}
    for k in ("ghi", "wind", "yields"):
        out[k] = pd.concat([base[k][~base[k].index.isin(recent[k].index)], recent[k]]).sort_index()
    out["data_until"] = max(base["data_until"], recent["data_until"])
    return out


def summarize(tables: dict, grid: pd.DataFrame, weights: pd.DataFrame) -> dict:
    """Everything the website and README show about the resource climate."""
    mon = {"ghi": tables["ghi"], "wind": tables["wind"]}
    norm = normals(mon)
    anom = anomalies_pct(mon, norm)
    national = pd.DataFrame({
        "ghi_kwh_m2": weighted(mon["ghi"], weights["pv_mw"]),
        "ghi_anom_pct": weighted(anom["ghi"], weights["pv_mw"]),
        "wind_ms": weighted(mon["wind"], weights["wind_mw"]),
        "wind_anom_pct": weighted(anom["wind"], weights["wind_mw"]),
    }).dropna(how="all")
    last_month = national.dropna().index.max()
    annual = annual_table(tables["yields"])
    ytd = year_to_date(tables["yields"])
    year_months = [m for m in mon["ghi"].index if m.year == ytd["year"] and m <= last_month]
    maps = {"month": str(last_month), "ytd_months": [str(m) for m in year_months]}
    for name in ("ghi", "wind"):
        maps[f"{name}_month"] = region_means(anom[name].loc[[last_month]], grid).iloc[0].to_dict()
        if year_months:
            cur = mon[name].loc[year_months]
            base = norm[name].loc[[m.month for m in year_months]]
            agg = (cur.sum() / base.sum()) if name == "ghi" else (cur.mean() / base.mean())
            maps[f"{name}_ytd"] = region_means(pd.DataFrame([100 * (agg - 1)]), grid).iloc[0].to_dict()
    return {
        "national": national, "annual": annual, "ytd": ytd, "maps": maps,
        "exceedance": {"pv": exceedance(annual["pv_kwh_kwp"]), "wind": exceedance(annual["wind_flh"])},
        "data_until": tables["data_until"],
    }


def save_base(tables: dict, gain: pd.DataFrame, curve: pd.Series, out_dir: Path) -> None:
    """Persist the base tables (complete years) so the daily run only downloads the current year."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in ("ghi", "wind"):
        m = tables[k].astype("float32")
        m.index, m.columns = m.index.astype(str), m.columns.astype(str)
        m.to_parquet(out_dir / f"{k}_monthly.parquet")
    y = tables["yields"].astype("float32")
    y.index.name = "day"
    y.to_parquet(out_dir / "yields_daily.parquet")
    gain.round(5).to_csv(out_dir / "pv_gain.csv", index_label="month")
    curve.round(5).rename("cf").to_csv(out_dir / "wind_daily_curve.csv", index_label="speed_ms")
    (out_dir / "meta.json").write_text(json.dumps({"data_until": tables["data_until"],
                                                   "last_year": int(tables["yields"].index.year.max())}, indent=2))


def load_base(in_dir: Path) -> dict:
    meta = json.loads((in_dir / "meta.json").read_text())
    tables = {"data_until": meta["data_until"]}
    for k in ("ghi", "wind"):
        m = pd.read_parquet(in_dir / f"{k}_monthly.parquet").astype("float64")
        m.index, m.columns = pd.PeriodIndex(m.index, freq="M"), m.columns.astype(int)
        tables[k] = m
    tables["yields"] = pd.read_parquet(in_dir / "yields_daily.parquet").astype("float64")
    gain = pd.read_csv(in_dir / "pv_gain.csv", index_col="month")
    gain.columns = gain.columns.astype(int)
    curve = pd.read_csv(in_dir / "wind_daily_curve.csv", index_col="speed_ms")["cf"]
    return {"tables": tables, "gain": gain, "curve": curve, "last_year": meta["last_year"]}
