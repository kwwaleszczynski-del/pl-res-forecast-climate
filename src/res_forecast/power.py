"""Weather to power: a generic PV fleet (pvlib) and a smoothed multi-turbine wind power curve.

PV: hourly GHI (mean over the preceding hour, as Open-Meteo and ERA5 report it) is split into beam and diffuse with
the Erbs model at the interval mid-point, transposed (Hay-Davies) onto a mix of orientations typical for a national
fleet (south-facing ground-mounted plus east and west roofs), corrected for incidence angle and cell temperature
(Faiman) and converted with a PVWatts-style model to AC output per kWp, with inverter clipping.

Wind: wind speed at the model level is shifted to a fleet hub height with a power-law profile and passed through a
generic modern turbine curve smoothed with a speed-dependent Gaussian (Norgaard & Holttinen's multi-turbine curve),
which mimics the spread of wind speeds over a wind farm and over the turbines represented by one grid point.

Both models are deliberately generic: the absolute level, curtailment and the fleet's growth are handled by the
statistical calibration against PSE data (see calibration.py).

Usage:
    p = pv_power(ghi, times, lat=52.2, lon=21.0)          # ghi (..., T) W/m2 -> AC kW per kWp, same shape
    cf = wind_power(speed_kmh, from_height=100.0)          # capacity factor 0..1
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

KMH_PER_MS = 3.6

# (share, tilt, azimuth): ground-mounted south plus east/west rooftops
FLEET_ORIENTATIONS = ((0.6, 25.0, 180.0), (0.2, 35.0, 100.0), (0.2, 35.0, 260.0))
DC_AC_RATIO = 1.15
GAMMA_PDC = -0.0037  # 1/K
SYSTEM_LOSSES = 0.12
ALBEDO = 0.2
# Poland, monthly mean air temperature 1991-2020 (approx.) + 4 K for daytime; used only when no forecast is given
MONTHLY_TEMP_C = np.array([-1.2, -0.1, 3.3, 8.6, 13.4, 16.9, 18.8, 18.4, 13.7, 8.7, 4.1, 0.2]) + 4.0

HUB_HEIGHT = 100.0
SHEAR_EXPONENT = 0.14
WIND_LOSSES = 0.10  # wakes, availability, electrical
_CURVE_V = np.array([0, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 25, 25.01, 40])
_CURVE_P = np.array([0, 0, 0.005, 0.03, 0.075, 0.14, 0.23, 0.345, 0.48, 0.62, 0.76, 0.88, 0.96, 1.0, 1.0, 0, 0])


def solar_geometry(times: pd.DatetimeIndex, lat: float, lon: float) -> dict:
    """Zenith, azimuth (deg) and extraterrestrial DNI at the mid-point of the hour ending at ``times``."""
    mid = times - pd.Timedelta(minutes=30)
    sp = pvlib.solarposition.get_solarposition(mid, lat, lon, method="nrel_numpy")
    return {
        "zenith": sp["apparent_zenith"].to_numpy(),
        "azimuth": sp["azimuth"].to_numpy(),
        "dni_extra": np.asarray(pvlib.irradiance.get_extra_radiation(mid.dayofyear)),
        "doy": mid.dayofyear.to_numpy(),
    }


def pv_power(ghi, times: pd.DatetimeIndex, lat: float, lon: float, temp_air=None, geometry: dict | None = None,
             orientations=FLEET_ORIENTATIONS) -> np.ndarray:
    """AC output per kWp DC for GHI with time on the last axis (any leading shape)."""
    g = geometry or solar_geometry(times, lat, lon)
    ghi = np.clip(np.nan_to_num(np.asarray(ghi, dtype=float), nan=0.0), 0, None)
    zen, az = g["zenith"], g["azimuth"]
    day = zen < 89.5
    ghi = np.where(day, ghi, 0.0)
    if temp_air is None:
        temp_air = MONTHLY_TEMP_C[(times - pd.Timedelta(minutes=30)).month.to_numpy() - 1]
    temp_air = np.asarray(temp_air, dtype=float)
    erbs = pvlib.irradiance.erbs(ghi, zen, g["doy"])
    dni = np.where(day, np.nan_to_num(np.asarray(erbs["dni"])), 0.0)
    dhi = np.where(day, np.nan_to_num(np.asarray(erbs["dhi"])), 0.0)
    out = np.zeros(np.broadcast_shapes(ghi.shape, zen.shape))
    for share, tilt, azimuth in orientations:
        aoi = np.asarray(pvlib.irradiance.aoi(tilt, azimuth, zen, az))
        beam = dni * np.clip(np.cos(np.radians(aoi)), 0, None) * np.asarray(pvlib.iam.physical(aoi))
        sky = np.asarray(pvlib.irradiance.haydavies(tilt, azimuth, dhi, dni, g["dni_extra"], zen, az))
        ground = np.asarray(pvlib.irradiance.get_ground_diffuse(tilt, ghi, ALBEDO))
        poa = np.nan_to_num(beam + sky + ground)
        t_cell = np.asarray(pvlib.temperature.faiman(poa, temp_air, wind_speed=1.0))
        dc = poa / 1000.0 * (1 + GAMMA_PDC * (t_cell - 25.0)) * (1 - SYSTEM_LOSSES)
        out += share * np.clip(dc, 0, 1 / DC_AC_RATIO)
    return out


def _smoothed_curve(v_grid: np.ndarray) -> np.ndarray:
    z, w = np.polynomial.hermite_e.hermegauss(21)
    w = w / w.sum()
    sigma = 0.6 + 0.1 * v_grid
    v = np.clip(v_grid[:, None] + sigma[:, None] * z[None, :], 0, None)
    return (np.interp(v, _CURVE_V, _CURVE_P) * w).sum(axis=1)


_V_GRID = np.linspace(0, 40, 801)
_P_GRID = _smoothed_curve(_V_GRID) * (1 - WIND_LOSSES)


def wind_power(speed_kmh, from_height: float = 100.0, hub_height: float = HUB_HEIGHT,
               shear: float = SHEAR_EXPONENT) -> np.ndarray:
    """Fleet capacity factor (0..1) for wind speed in km/h at ``from_height`` metres."""
    v = np.asarray(speed_kmh, dtype=float) / KMH_PER_MS * (hub_height / from_height) ** shear
    return np.interp(np.nan_to_num(v, nan=0.0), _V_GRID, _P_GRID)


def hourly_mean_wind_power(speed_kmh, **kw) -> np.ndarray:
    """Mean power over the hour ending at each step from instantaneous speeds at its start and end (time last)."""
    p = wind_power(speed_kmh, **kw)
    prev = np.concatenate([p[..., :1], p[..., :-1]], axis=-1)
    return 0.5 * (p + prev)
