import numpy as np
import pandas as pd

from res_forecast.geo import build_grid, capacity_weights, load_capacity, region_means, region_of


def test_cities_fall_in_the_right_voivodeship():
    lon = np.array([21.01, 19.94, 18.65, 14.55, 10.0])
    lat = np.array([52.23, 50.06, 54.35, 53.43, 50.0])  # Warsaw, Krakow, Gdansk, Szczecin, Germany
    assert list(region_of(lon, lat)) == ["PL-MZ", "PL-MA", "PL-PM", "PL-ZP", ""]


def test_grid_covers_poland_and_points_lie_inside():
    g = build_grid()
    assert 40 <= len(g) <= 60
    assert 290_000 < g["area_km2"].sum() < 315_000  # Poland: 312 700 km2, minus dropped border slivers
    assert (region_of(g["lon"].to_numpy(), g["lat"].to_numpy()) != "").all()


def test_capacity_weights_preserve_national_totals():
    w = capacity_weights()
    cap = load_capacity()
    assert np.isclose(w["wind_mw"].sum(), cap["wind_mw"].sum())
    assert np.isclose(w["pv_mw"].sum(), cap["pv_mw"].sum())
    assert (w >= 0).all().all()


def test_region_means_of_a_constant_field():
    g = build_grid()
    vals = pd.DataFrame([np.full(len(g), 7.0)], columns=g.index)
    out = region_means(vals, g)
    assert out.shape == (1, 16)
    assert np.allclose(out.to_numpy(), 7.0)
