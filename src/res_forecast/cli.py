"""Command line interface.

Usage:
    python -m res_forecast backfill               # one-off download of the history (resumable, respects API limits)
    python -m res_forecast hindcast --tune        # tune on the first year, verify D+1..D+7 on the rest
    python -m res_forecast climate                # ERA5 resource climatology (base tables in results/climate)
    python -m res_forecast update                 # daily run: live forecast, archive, website
    python -m res_forecast update --help

Cached inputs live in ./data (override with --data-dir or RES_DATA_DIR), committed results in ./results,
README charts in ./docs/img, the archive of issued forecasts in ./archive and the generated website in ./site.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from res_forecast import plots
from res_forecast.archive import COLS, live_verification, load_archive, save_forecast
from res_forecast.calibration import K_MIN
from res_forecast.climate import base_tables, daily_wind_curve, extend, load_base, pv_gain, save_base, summarize
from res_forecast.config import (CLIMATE_START_YEAR, ERA5_HOURLY_YEARS, HINDCAST_MODELS, HINDCAST_START,
                                  MULTI_MODEL_START, TEST_START, TZ)
from res_forecast.forecast import Live, fetch_members, run_live
from res_forecast.geo import build_grid, capacity_weights
from res_forecast.hindcast import TECHS, calibrations_at, load_physical, run_hindcast
from res_forecast.history import load_era5_daily, load_era5_hourly, load_previous_runs
from res_forecast.openmeteo import OpenMeteo, RateLimiter
from res_forecast.pse import fetch_generation
from res_forecast.site import build_payload, render
from res_forecast.verify import summarize as verify_summary

log = logging.getLogger("res_forecast")
DEFAULT_PARAMS = {"pv": {"window": 60, "inflate": 1.0}, "wind": {"window": 60, "inflate": 1.0}, "k_min": K_MIN}


def _om(args, max_wait: float) -> OpenMeteo:
    limiter = RateLimiter(per_day=args.daily_budget, ledger_path=args.data_dir / "openmeteo_ledger.json",
                          max_wait=max_wait)
    return OpenMeteo(limiter=limiter)


def load_params(results_dir: Path) -> dict:
    p = results_dir / "hindcast_params.json"
    return json.loads(p.read_text()) if p.exists() else DEFAULT_PARAMS


# -- backfill --------------------------------------------------------------------------------------------------------

def cmd_backfill(args):
    grid = build_grid()
    om = _om(args, max_wait=26 * 3600)
    today = date.today()
    pse = fetch_generation(HINDCAST_START, today, args.data_dir / "pse")
    log.info("PSE: %d hours %s .. %s", len(pse), pse.index.min(), pse.index.max())
    models = list(HINDCAST_MODELS)
    load_previous_runs(om, grid, models[0], HINDCAST_START, today, args.data_dir / "runs")
    load_era5_daily(om, grid, CLIMATE_START_YEAR, today, args.data_dir / "era5")
    load_era5_hourly(om, grid, ERA5_HOURLY_YEARS, args.data_dir / "era5")
    for model in models[1:]:
        load_previous_runs(om, grid, model, MULTI_MODEL_START, today, args.data_dir / "runs")
    log.info("backfill complete, %.0f weighted calls spent today", om.limiter.spent_today())


# -- hindcast --------------------------------------------------------------------------------------------------------

def _hindcast(phys, actual, params, **kw) -> pd.DataFrame:
    parts = [run_hindcast(phys, actual, window=params[t]["window"], inflate=params[t]["inflate"],
                          k_min=params["k_min"], techs=(t,), **kw) for t in TECHS]
    return pd.concat(parts, ignore_index=True)


def cmd_hindcast(args):
    grid = build_grid()
    weights = capacity_weights(grid)
    today = date.today()
    actual = fetch_generation(HINDCAST_START, today, args.data_dir / "pse")
    phys = load_physical(None, args.data_dir, grid, weights, HINDCAST_START, today, cached_only=True)
    log.info("models with archived runs: %s", ", ".join(phys))
    split = args.split
    params = json.loads(json.dumps(DEFAULT_PARAMS))
    tuning_rows = []
    if args.tune:
        base = "ecmwf_ifs025"
        for tech in TECHS:
            for window in args.windows:
                for inflate in args.inflates:
                    rows = run_hindcast({base: phys[base]}, actual, window=window, inflate=inflate, techs=(tech,),
                                        last_issue=split - timedelta(days=1), products=("single",))
                    s = verify_summary(rows)
                    tuning_rows.append({"tech": tech, "window": window, "inflate": inflate, "k_min": None,
                                        "crps_mw": s["crps_mw"].mean(), "coverage80_pct": s["coverage80_pct"].mean()})
                    log.info("tune %s window=%d inflate=%.2f: CRPS %.1f MW", tech, window, inflate,
                             tuning_rows[-1]["crps_mw"])
            best = min((r for r in tuning_rows if r["tech"] == tech), key=lambda r: r["crps_mw"])
            params[tech] = {"window": int(best["window"]), "inflate": float(best["inflate"])}
        if len(phys) > 1:
            scores = {}
            for k_min in args.k_mins:
                params["k_min"] = k_min
                rows = _hindcast(phys, actual, params, last_issue=split - timedelta(days=1), products=("multi",))
                s = verify_summary(rows)
                scores[k_min] = (s.groupby("tech")["crps_mw"].mean() / s.groupby("tech")["mean_obs_mw"].mean()).mean()
                tuning_rows.append({"tech": "both", "window": None, "inflate": None, "k_min": k_min,
                                    "crps_mw": s["crps_mw"].mean(), "coverage80_pct": s["coverage80_pct"].mean()})
                log.info("tune k_min=%.2f: relative CRPS %.4f", k_min, scores[k_min])
            params["k_min"] = float(min(scores, key=scores.get))
        args.results_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(tuning_rows).to_csv(args.results_dir / "hindcast_tuning.csv", index=False)
        (args.results_dir / "hindcast_params.json").write_text(json.dumps(
            {**params, "tuned_on": f"issues before {split}"}, indent=2))
    else:
        params = load_params(args.results_dir)
    log.info("parameters: %s", params)
    rows = _hindcast(phys, actual, params)
    rows.to_parquet(args.data_dir / "hindcast_rows.parquet")
    issue = pd.to_datetime(rows["ts"]).dt.tz_convert("UTC").dt.normalize() - pd.to_timedelta(rows["lead"], unit="D")
    test = rows[issue >= pd.Timestamp(split, tz="UTC")]
    summary = verify_summary(test)
    summary.insert(0, "period", f"{test['ts'].min():%Y-%m-%d}..{test['ts'].max():%Y-%m-%d}")
    summary.to_csv(args.results_dir / "hindcast_summary.csv", index=False, float_format="%.3f")
    by_month = verify_summary(test.assign(month=pd.to_datetime(test["ts"]).dt.strftime("%Y-%m")),
                              by=("product", "tech", "lead", "month"))
    by_month[by_month["lead"] == 1].to_csv(args.results_dir / "hindcast_d1_by_month.csv", index=False,
                                            float_format="%.3f")
    log.info("\n%s", summary[summary["lead"].isin([1, 3, 5, 7])][
        ["product", "tech", "lead", "nmae_pct", "crps_mw", "crpss_vs_clim_pct", "coverage80_pct"]].round(1).to_string())
    prod = "multi" if (summary["product"] == "multi").any() else "ecmwf_ifs025"
    plots.hindcast_skill(summary, args.img_dir / "hindcast_skill.png", products=(prod, "ecmwf_ifs025"))
    w = actual["wind"][actual.index >= pd.Timestamp(split, tz="UTC")]
    start = w.rolling("10D").std().idxmax() - pd.Timedelta(days=10)
    plots.hindcast_example(rows, prod, start.normalize(), 10, args.img_dir / "hindcast_example.png")


# -- climate ---------------------------------------------------------------------------------------------------------

def cmd_climate(args):
    grid = build_grid()
    weights = capacity_weights(grid)
    today = date.today()
    om = _om(args, max_wait=3600) if args.fetch else None
    daily = load_era5_daily(om, grid, CLIMATE_START_YEAR, today, args.data_dir / "era5", cached_only=not args.fetch)
    hourly = load_era5_hourly(om, grid, ERA5_HOURLY_YEARS, args.data_dir / "era5", cached_only=not args.fetch)
    gain, curve = pv_gain(hourly, grid), daily_wind_curve(hourly)
    years = daily.index.year
    full_years = sorted(y for y in set(years) if (years == y).sum() >= 365 and daily[years == y].notna().all(axis=1).sum() >= 365)
    last_full = max(full_years)
    base = base_tables(daily[years <= last_full], grid, weights, gain, curve)
    save_base(base, gain, curve, args.results_dir / "climate")
    recent = base_tables(daily[years > last_full], grid, weights, gain, curve) if (years > last_full).any() else None
    clim = summarize(extend(base, recent) if recent else base, grid, weights)
    _write_climate(clim, args)


def _write_climate(clim: dict, args):
    args.results_dir.mkdir(parents=True, exist_ok=True)
    clim["annual"].round(2).to_csv(args.results_dir / "climate_annual.csv")
    nat = clim["national"].copy()
    nat.index = nat.index.astype(str)
    nat.round(2).to_csv(args.results_dir / "climate_national_monthly.csv", index_label="month")
    ytd = {k: v for k, v in clim["ytd"].items() if k != "series"}
    (args.results_dir / "climate_summary.json").write_text(json.dumps(
        {"data_until": clim["data_until"], "ytd": ytd, "exceedance": clim["exceedance"], "maps": clim["maps"]},
        indent=2, default=float))
    plots.anomaly_maps(clim, args.img_dir / "climate_maps.png")
    plots.annual(clim, args.img_dir / "climate_annual.png")
    plots.monthly_anomalies(clim, args.img_dir / "climate_monthly.png")
    log.info("climate: data until %s, YTD solar %.1f %% wind %.1f %%", clim["data_until"],
             clim["ytd"]["pv_index_pct"], clim["ytd"]["wind_index_pct"])


# -- daily update ----------------------------------------------------------------------------------------------------

def _live_from_archive(archive_dir: Path) -> Live | None:
    """Last archived forecast as a fan-only Live object (used when today's forecast failed)."""
    a = load_archive(archive_dir)
    if a.empty:
        return None
    last = a[a["issue_date"] == a["issue_date"].max()]
    fan = {}
    for tech, g in last.groupby("tech"):
        g = g.set_index(pd.to_datetime(g["ts"], utc=True)).sort_index()
        fan[tech] = g[COLS].set_axis([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], axis=1).astype(float)
    times = next(iter(fan.values())).index
    return Live(pd.to_datetime(last["issue_utc"].iloc[0], utc=True), times, fan, pd.DataFrame(), pd.DataFrame())


def cmd_update(args):
    status: dict[str, str] = {}
    params = load_params(args.results_dir)
    grid = build_grid()
    weights = capacity_weights(grid)
    om = _om(args, max_wait=args.max_wait)
    now = pd.Timestamp.now(tz="UTC")
    today = now.tz_convert(TZ).date()
    max_window = max(params[t]["window"] for t in TECHS)

    actual = None
    try:
        actual = fetch_generation(today - timedelta(days=400), today, args.data_dir / "pse")
        status["PSE generation"] = f"ok, to {actual.index.max().tz_convert(TZ):%d %b %H:%M}"
    except Exception as e:
        status["PSE generation"] = f"failed ({type(e).__name__})"
        log.exception("PSE download failed")

    phys, calibs = {}, {}
    try:
        phys = load_physical(om, args.data_dir, grid, weights, today - timedelta(days=2 * max_window + 20),
                             today + timedelta(days=1), models=args.models)
        for tech in TECHS:
            c = calibrations_at(phys, actual, today, window=params[tech]["window"], inflate=params[tech]["inflate"])
            calibs.update({k: v for k, v in c.items() if k[1] == tech})
        status["calibration"] = f"ok, {len(calibs)} model/lead fits"
    except Exception as e:
        status["calibration"] = f"failed ({type(e).__name__})"
        log.exception("calibration failed")

    live, stale = None, None
    if calibs:
        try:
            cache = args.data_dir / "live_members.pkl"
            if args.reuse_members and cache.exists():
                with open(cache, "rb") as f:
                    saved = pickle.load(f)  # our own file from an earlier run today
                if saved["day"] != today.isoformat():
                    raise RuntimeError("cached ensemble members are not from today")
                times, raw, now = saved["times"], saved["raw"], saved["issue"]
                status.update(saved["status"])
            else:
                times, raw = fetch_members(om, grid, weights, status, det_models={k[0] for k in calibs})
                with open(cache, "wb") as f:
                    pickle.dump({"day": today.isoformat(), "times": times, "raw": raw, "issue": now,
                                 "status": {k: v for k, v in status.items() if k not in ("PSE generation", "calibration")}}, f)
            live = run_live(raw, times, calibs, now, actual, params["k_min"])
            live.status = status
            save_forecast(args.archive_dir, live)
        except Exception as e:
            status["live forecast"] = f"failed ({type(e).__name__})"
            log.exception("live forecast failed")
    if live is None:
        live = _live_from_archive(args.archive_dir)
        if live is not None:
            stale = f"Showing the last successful forecast, issued {live.issue.tz_convert(TZ):%d %b %H:%M}"

    recent, recent_metrics = None, None
    try:
        parts = [run_hindcast(phys, actual, window=params[t]["window"], inflate=params[t]["inflate"],
                              k_min=params["k_min"], techs=(t,), first_issue=today - timedelta(days=30),
                              last_issue=today - timedelta(days=1)) for t in TECHS]
        recent = pd.concat(parts, ignore_index=True)
        recent_metrics = verify_summary(recent)
    except Exception as e:
        status["recent verification"] = f"failed ({type(e).__name__})"
        log.exception("recent verification failed")

    live_metrics = None
    if actual is not None:
        try:
            live_metrics, _ = live_verification(args.archive_dir, actual)
            if len(live_metrics) and live_metrics["days"].min() < args.min_live_days:
                live_metrics = None
        except Exception:
            log.exception("live verification failed")

    clim = None
    try:
        base = load_base(args.results_dir / "climate")
        cur = load_era5_daily(om, grid, base["last_year"] + 1, today, args.data_dir / "era5")
        tables = base["tables"]
        if len(cur):
            tables = extend(tables, base_tables(cur, grid, weights, base["gain"], base["curve"]))
        clim = summarize(tables, grid, weights)
        status["ERA5"] = f"ok, to {clim['data_until']}"
    except Exception as e:
        status["ERA5"] = f"failed ({type(e).__name__})"
        log.exception("climatology failed")

    hindcast, meta = None, {}
    p = args.results_dir / "hindcast_summary.csv"
    if p.exists():
        hindcast = pd.read_csv(p)
        meta = {"period": hindcast["period"].iloc[0].replace("..", " to "),
                "product": "multi" if (hindcast["product"] == "multi").any() else "ecmwf_ifs025"}
    payload = build_payload(live=live, actual=actual, recent=recent, recent_metrics=recent_metrics,
                            hindcast=hindcast, hindcast_meta=meta, live_metrics=live_metrics, climate=clim,
                            status=status, stale_note=stale)
    out = render(payload, args.site_dir)
    log.info("site written to %s; status: %s", out, status)
    if args.readme_images and live is not None and stale is None:
        plots.fan(live, actual, args.img_dir / "forecast_fan.png")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="res-forecast", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("backfill", "hindcast", "climate", "update"):
        p = sub.add_parser(name)
        p.add_argument("--data-dir", type=Path, default=Path(os.environ.get("RES_DATA_DIR", "data")))
        p.add_argument("--results-dir", type=Path, default=Path("results"))
        p.add_argument("--img-dir", type=Path, default=Path("docs/img"))
        p.add_argument("--daily-budget", type=float, default=9000.0, help="max weighted Open-Meteo calls per UTC day")
        if name == "hindcast":
            p.add_argument("--tune", action="store_true", help="choose window/inflation/k_min on the tuning period")
            p.add_argument("--split", type=date.fromisoformat, default=TEST_START,
                           help="issues before this date tune the method, later ones are the test period")
            p.add_argument("--windows", type=int, nargs="+", default=[45, 60, 90])
            p.add_argument("--inflates", type=float, nargs="+", default=[1.0, 1.1, 1.2])
            p.add_argument("--k-mins", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0])
        if name == "climate":
            p.add_argument("--fetch", action="store_true", help="download missing ERA5 years (default: cache only)")
        if name == "update":
            p.add_argument("--site-dir", type=Path, default=Path("site"))
            p.add_argument("--archive-dir", type=Path, default=Path("archive"))
            p.add_argument("--max-wait", type=float, default=900.0, help="longest wait for the API rate limit, s")
            p.add_argument("--min-live-days", type=int, default=7, help="days of archive before live scores show")
            p.add_argument("--readme-images", action="store_true", help="also redraw docs/img/forecast_fan.png")
            p.add_argument("--reuse-members", action="store_true",
                           help="reuse today's downloaded ensemble members (data/live_members.pkl) instead of fetching")
            p.add_argument("--models", nargs="+", default=list(HINDCAST_MODELS),
                           help="deterministic models to calibrate (their ensembles are used live)")
    args = ap.parse_args(argv)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.data_dir / "run.log", encoding="utf-8")])
    np.seterr(all="ignore")
    {"backfill": cmd_backfill, "hindcast": cmd_hindcast, "climate": cmd_climate, "update": cmd_update}[args.cmd](args)


if __name__ == "__main__":
    main()
