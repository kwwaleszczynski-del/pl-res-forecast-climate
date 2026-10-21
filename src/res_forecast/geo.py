"""Poland geometry, the 1-degree point grid and installed-capacity weights.

The weather is sampled on a regular 1-degree lattice clipped to Poland. Each lattice cell is represented by one point
placed at the centroid of the Polish part of the cell, so every point lies inside the country. Installed capacity of
each voivodeship (public URE data plus a population-based estimate of rooftop PV, see tools/build_static_data.py) is
spread uniformly over the voivodeship's area and summed per cell. No individual plant locations are used.

Usage:
    from res_forecast.geo import build_grid, capacity_weights
    grid = build_grid()                 # one row per point: lat, lon, area_km2 and overlap with each voivodeship
    weights = capacity_weights(grid)    # MW of wind and PV represented by each point
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

DATA_DIR = Path(__file__).parent / "data"
LAT_RANGE = (49.0, 55.0)
LON_RANGE = (14.0, 25.0)


@dataclass(frozen=True)
class Region:
    code: str
    name: str
    rings: tuple  # outer rings as (n, 2) arrays of lon, lat


@lru_cache(maxsize=1)
def load_voivodeships() -> tuple[Region, ...]:
    geo = json.loads((DATA_DIR / "voivodeships.geojson").read_text(encoding="utf-8"))
    regions = []
    for f in geo["features"]:
        rings = tuple(np.asarray(poly[0], dtype=float) for poly in f["geometry"]["coordinates"])
        regions.append(Region(f["properties"]["code"], f["properties"]["name"], rings))
    return tuple(regions)


def load_geojson() -> dict:
    return json.loads((DATA_DIR / "voivodeships.geojson").read_text(encoding="utf-8"))


def load_capacity() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "capacity_voivodeship.csv").set_index("code")


def region_of(lon: np.ndarray, lat: np.ndarray, regions=None) -> np.ndarray:
    """Voivodeship code for each point ('' outside Poland)."""
    regions = regions or load_voivodeships()
    pts = np.column_stack([np.ravel(lon), np.ravel(lat)])
    out = np.full(len(pts), "", dtype=object)
    for r in regions:
        inside = np.zeros(len(pts), dtype=bool)
        for ring in r.rings:
            inside |= MplPath(ring).contains_points(pts)
        out[inside & (out == "")] = r.code
    return out.reshape(np.shape(lon))


@lru_cache(maxsize=4)
def build_grid(step: float = 1.0, sub: float = 0.05, min_share: float = 0.3) -> pd.DataFrame:
    """Lattice points with the area (km2) of every voivodeship inside each cell.

    Cells whose Polish part is smaller than ``min_share`` of the cell are dropped; their area is not lost for the
    capacity weights because those are normalised per voivodeship over the kept cells.
    """
    regions = load_voivodeships()
    codes = [r.code for r in regions]
    lats = np.arange(LAT_RANGE[0] + sub / 2, LAT_RANGE[1], sub)
    lons = np.arange(LON_RANGE[0] + sub / 2, LON_RANGE[1], sub)
    lon2, lat2 = np.meshgrid(lons, lats)
    reg = region_of(lon2, lat2, regions)
    sub_area = (111.32 * sub) ** 2 * np.cos(np.radians(lat2))  # km2 of each sub-cell
    cell_i = np.floor((lat2 - LAT_RANGE[0]) / step).astype(int)
    cell_j = np.floor((lon2 - LON_RANGE[0]) / step).astype(int)
    rows = []
    for (ci, cj), idx in pd.DataFrame({"i": cell_i.ravel(), "j": cell_j.ravel()}).groupby(["i", "j"]).groups.items():
        idx = np.asarray(idx)
        r = reg.ravel()[idx]
        a = sub_area.ravel()[idx]
        pl = r != ""
        if a[pl].sum() < min_share * a.sum():
            continue
        row = {
            "cell_lat": LAT_RANGE[0] + (ci + 0.5) * step,
            "cell_lon": LON_RANGE[0] + (cj + 0.5) * step,
            "lat": round(float(np.average(lat2.ravel()[idx][pl], weights=a[pl])), 2),
            "lon": round(float(np.average(lon2.ravel()[idx][pl], weights=a[pl])), 2),
            "area_km2": float(a[pl].sum()),
        }
        for c in codes:
            row[c] = float(a[r == c].sum())
        rows.append(row)
    grid = pd.DataFrame(rows).sort_values(["cell_lat", "cell_lon"], ascending=[False, True]).reset_index(drop=True)
    grid.index.name = "point"
    return grid


def overlap(grid: pd.DataFrame) -> pd.DataFrame:
    """Area of each voivodeship (columns) inside each grid cell (rows), km2."""
    return grid[[r.code for r in load_voivodeships()]]


def capacity_weights(grid: pd.DataFrame | None = None, capacity: pd.DataFrame | None = None) -> pd.DataFrame:
    """MW of wind and PV represented by each grid point (sums equal the national totals)."""
    grid = build_grid() if grid is None else grid
    capacity = load_capacity() if capacity is None else capacity
    ov = overlap(grid)
    share = ov / ov.sum(axis=0)  # fraction of each voivodeship's area that falls in each cell
    return pd.DataFrame({
        "wind_mw": share.mul(capacity.loc[ov.columns, "wind_mw"], axis=1).sum(axis=1),
        "pv_mw": share.mul(capacity.loc[ov.columns, "pv_mw"], axis=1).sum(axis=1),
    })


def region_means(values: pd.DataFrame, grid: pd.DataFrame | None = None) -> pd.DataFrame:
    """Area-weighted mean of point values (columns = points) for every voivodeship (columns = codes)."""
    grid = build_grid() if grid is None else grid
    ov = overlap(grid)
    w = ov / ov.sum(axis=0)
    v = values[grid.index]
    return pd.DataFrame(v.to_numpy() @ w.to_numpy(), index=values.index, columns=ov.columns)
