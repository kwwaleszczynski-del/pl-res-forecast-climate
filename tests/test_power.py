import numpy as np
import pandas as pd
import pvlib

from res_forecast.power import DC_AC_RATIO, hourly_mean_wind_power, pv_power, wind_power


def _clear_sky_day(lat=52.2, lon=21.0, day="2025-06-21"):
    times = pd.date_range(day, periods=24, freq="h", tz="UTC") + pd.Timedelta("1h")  # hour-ending labels
    cs = pvlib.location.Location(lat, lon).get_clearsky(times - pd.Timedelta("30min"))["ghi"].to_numpy()
    return times, cs


def test_pv_zero_at_night_and_positive_at_noon():
    times, ghi = _clear_sky_day()
    p = pv_power(ghi, times, 52.2, 21.0)
    hour = (times - pd.Timedelta("1h")).hour
    assert p[hour == 0].max() == 0
    assert p[hour == 10] > 0.5
    assert p.max() <= 1 / DC_AC_RATIO + 1e-9


def test_pv_broadcasts_over_members():
    times, ghi = _clear_sky_day()
    stack = np.vstack([ghi, 0.5 * ghi, np.zeros_like(ghi)])
    p = pv_power(stack, times, 52.2, 21.0)
    assert p.shape == (3, 24)
    np.testing.assert_allclose(p[0], pv_power(ghi, times, 52.2, 21.0))
    assert (p[1] <= p[0] + 1e-12).all() and p[2].max() == 0


def test_pv_hotter_cells_produce_less():
    times, ghi = _clear_sky_day()
    cold = pv_power(ghi, times, 52.2, 21.0, temp_air=np.full(24, 0.0)).sum()
    hot = pv_power(ghi, times, 52.2, 21.0, temp_air=np.full(24, 35.0)).sum()
    assert hot < cold


def test_wind_curve_shape():
    v_ms = np.array([0, 3, 6, 9, 12, 15, 30])
    cf = wind_power(v_ms * 3.6)
    assert cf[0] < 1e-6
    assert (np.diff(cf[:6]) > 0).all()
    assert 0.8 < cf[5] <= 1.0
    assert cf[6] < 0.05  # storm shutdown


def test_wind_shear_raises_speed_from_lower_level():
    v = np.array([25.0])  # km/h
    assert wind_power(v, from_height=80.0) > wind_power(v, from_height=100.0)


def test_hourly_mean_wind_power_averages_neighbours():
    p = hourly_mean_wind_power(np.array([[0.0, 36.0, 36.0]]))
    inst = wind_power(np.array([0.0, 36.0]))
    assert np.isclose(p[0, 1], inst.mean())
    assert np.isclose(p[0, 2], inst[1])
