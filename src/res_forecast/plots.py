"""Static charts for the README (matplotlib, same palette as the website).

Usage:
    plots.fan(live, actual, Path("docs/img/forecast_fan.png"))
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from res_forecast.config import TZ  # noqa: E402
from res_forecast.geo import load_geojson  # noqa: E402

COL = {"pv": "#eb6834", "wind": "#2a78d6", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb", "neg": "#e34948", "pos": "#2a78d6", "mid": "#f0efec"}
NAME = {"pv": "Solar", "wind": "Wind"}
plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9.5, "axes.edgecolor": COL["axis"],
    "axes.labelcolor": COL["ink2"], "xtick.color": COL["muted"], "ytick.color": COL["muted"],
    "axes.grid": True, "grid.color": COL["grid"], "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": COL["surface"], "axes.facecolor": COL["surface"],
    "savefig.facecolor": COL["surface"], "axes.titleweight": "bold", "axes.titlesize": 10.5,
    "axes.titlecolor": COL["ink"], "legend.frameon": False,
})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _local(idx) -> pd.DatetimeIndex:
    return (pd.DatetimeIndex(idx) - pd.Timedelta(hours=1)).tz_convert(TZ).tz_localize(None)


def fan(live, actual: pd.DataFrame | None, path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    t = _local(live.times)
    for ax, tech in zip(axes, ("pv", "wind")):
        q = live.fan[tech] / 1000
        c = COL[tech]
        ax.fill_between(t, q[0.1], q[0.9], color=c, alpha=0.14, lw=0, label="80 % range")
        ax.fill_between(t, q[0.25], q[0.75], color=c, alpha=0.28, lw=0, label="50 % range")
        ax.plot(t, q[0.5], color=c, lw=2, label="median")
        if actual is not None and len(actual):
            a = actual[actual.index > live.times[0] - pd.Timedelta(days=2)]
            ax.plot(_local(a.index), a[tech] / 1000, color=COL["ink"], lw=1.6, label="PSE reported")
        ax.axvline(_local([live.issue.ceil("h")])[0], color=COL["axis"], lw=1)
        ax.set_ylabel("GW")
        ax.set_ylim(bottom=0)
        ax.set_title(f"{NAME[tech]}, issued {live.issue.tz_convert(TZ):%d %b %Y %H:%M}", loc="left")
        # above the plot, right of the title, so it never covers a peak
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=4, fontsize=8.5, borderaxespad=0.1)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    fig.tight_layout()
    _save(fig, path)


def hindcast_skill(summary: pd.DataFrame, path: Path, products=("multi", "ecmwf_ifs025")):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    labels = {"multi": "multi-model", "ecmwf_ifs025": "ECMWF only"}
    products = tuple(dict.fromkeys(products))
    for tech in ("pv", "wind"):
        for prod, ls in zip(products, ("-", "--")):
            s = summary[(summary["tech"] == tech) & (summary["product"] == prod)].sort_values("lead")
            if s.empty:
                continue
            kw = dict(color=COL[tech], lw=2, ls=ls, marker="o", ms=4, label=f"{NAME[tech]}, {labels.get(prod, prod)}")
            axes[0].plot(s["lead"], s["nmae_pct"], **kw)
            axes[1].plot(s["lead"], s["crpss_vs_clim_pct"], **kw)
            axes[2].plot(s["lead"], s["coverage80_pct"], **kw)
    axes[0].set_title("nMAE of the median, %", loc="left")
    axes[1].set_title("CRPS skill vs climatology, %", loc="left")
    axes[2].set_title("Share inside the 80 % range, %", loc="left")
    axes[2].axhline(80, color=COL["ink2"], lw=1)
    axes[2].set_ylim(60, 90)
    leads = sorted(int(x) for x in summary["lead"].unique())
    for ax in axes:
        ax.set_xticks(leads, [f"D+{i}" for i in leads])
        ax.set_xlim(leads[0] - 0.3, leads[-1] + 0.3)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncols=4, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, path)


def hindcast_example(rows: pd.DataFrame, product: str, start, days: int, path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    lo = pd.Timestamp(start)
    lo = lo.tz_localize("UTC") if lo.tzinfo is None else lo.tz_convert("UTC")
    for ax, tech in zip(axes, ("pv", "wind")):
        r = rows[(rows["product"] == product) & (rows["tech"] == tech) & (rows["lead"] == 1)]
        r = r[(r["ts"] >= lo) & (r["ts"] < lo + pd.Timedelta(days=days))].sort_values("ts")
        t = _local(r["ts"])
        c = COL[tech]
        ax.fill_between(t, r["q10"] / 1000, r["q90"] / 1000, color=c, alpha=0.16, lw=0, label="80 % range")
        ax.plot(t, r["q50"] / 1000, color=c, lw=2, label="day-ahead median")
        ax.plot(t, r["y"] / 1000, color=COL["ink"], lw=1.4, label="PSE reported")
        ax.set_ylabel("GW")
        ax.set_ylim(bottom=0)
        ax.set_title(f"{NAME[tech]}: day-ahead hindcast vs PSE", loc="left")
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=3, fontsize=8.5, borderaxespad=0.1)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.tight_layout()
    _save(fig, path)


def _map(ax, values: dict, title: str, lim: float):
    cmap = LinearSegmentedColormap.from_list("div", [COL["neg"], COL["mid"], COL["pos"]])
    norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
    for f in load_geojson()["features"]:
        v = values.get(f["properties"]["code"], np.nan)
        for poly in f["geometry"]["coordinates"]:
            ring = np.asarray(poly[0])
            ax.add_patch(Polygon(ring, closed=True, facecolor=cmap(norm(v)) if np.isfinite(v) else COL["grid"],
                                 edgecolor=COL["surface"], lw=0.8))
        if np.isfinite(v):
            ring = np.asarray(f["geometry"]["coordinates"][0][0])
            ax.text(ring[:, 0].mean(), ring[:, 1].mean(), f"{v:+.0f}", ha="center", va="center", fontsize=7.5,
                    color=COL["ink"])
    ax.set_xlim(14, 24.3)
    ax.set_ylim(48.9, 55)
    ax.set_aspect(1 / np.cos(np.radians(52)))
    ax.axis("off")
    ax.set_title(title, loc="left")
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def anomaly_maps(clim: dict, path: Path):
    m = clim["maps"]
    month = pd.Period(m["month"]).strftime("%B %Y")
    panels = [("ghi_month", f"Irradiance, {month}"), ("wind_month", f"Wind at 100 m, {month}")]
    if m.get("ytd_months"):
        y = clim["ytd"]["year"]
        panels += [("ghi_ytd", f"Irradiance, {y} to date"), ("wind_ytd", f"Wind at 100 m, {y} to date")]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.3))
    for ax, (key, title) in zip(np.atleast_1d(axes), panels):
        vals = m.get(key, {})
        lim = max(5.0, float(np.nanmax(np.abs(list(vals.values())))) if vals else 5.0)
        sm = _map(ax, vals, title, np.ceil(lim))
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02)
        cb.set_label("% vs 1991-2020", fontsize=8)
        cb.outline.set_visible(False)
    fig.tight_layout()
    _save(fig, path)


def annual(clim: dict, path: Path):
    a = clim["annual"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, (col, unit, tech) in zip(axes, (("pv_kwh_kwp", "kWh/kWp", "pv"), ("wind_flh", "full-load hours", "wind"))):
        ex = clim["exceedance"][tech]
        ax.bar(a.index, a[col], color=COL[tech], width=0.6)
        for name, v, ls in (("P50", ex["p50"], "-"), ("P90", ex["p90"], ":")):
            ax.axhline(v, color=COL["ink2"], lw=1, ls=ls)
            ax.text(a.index.max() + 1.0, v, name, va="bottom", fontsize=8, color=COL["ink2"])
        ax.set_xlim(a.index.min() - 1, a.index.max() + 3.5)
        ax.set_ylim(a[col].min() * 0.92, a[col].max() * 1.03)
        ax.set_title(f"{NAME[tech]}: annual yield, {unit} (P90/P50 = {ex['p90_over_p50_pct']:.1f} %)", loc="left")
    fig.tight_layout()
    _save(fig, path)


def monthly_anomalies(clim: dict, path: Path, months: int = 36):
    n = clim["national"].tail(months)
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.4), sharex=True)
    x = n.index.to_timestamp()
    for ax, (col, title) in zip(axes, (("ghi_anom_pct", "Solar irradiance"), ("wind_anom_pct", "Wind speed at 100 m"))):
        v = n[col]
        ax.bar(x, v, width=20, color=np.where(v >= 0, COL["pos"], COL["neg"]))
        ax.axhline(0, color=COL["axis"], lw=1)
        ax.set_title(f"{title}: monthly anomaly vs 1991-2020, % (capacity weighted)", loc="left")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    _save(fig, path)
