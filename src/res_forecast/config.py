"""Models, variables and constants shared by the pipeline."""
from __future__ import annotations

from datetime import date

LEADS = list(range(1, 8))
QUANTILES = [round(q, 2) for q in [0.05 * i for i in range(1, 20)]]  # 0.05 .. 0.95
TZ = "Europe/Warsaw"

# Archived deterministic runs used for calibration and the hindcast: {model: {variable: lead days}}.
# GFS has no hub-height wind in the ensemble, so it is used for PV only; ICON's archive stops at day 6 for radiation
# and the ICON-EPS ensemble carries 80 m wind for about five days.
HINDCAST_MODELS = {
    "ecmwf_ifs025": {"shortwave_radiation": LEADS, "wind_speed_100m": LEADS, "temperature_2m": [1]},
    "icon_seamless": {"shortwave_radiation": LEADS[:6], "wind_speed_80m": LEADS[:5]},
    "gfs_seamless": {"shortwave_radiation": LEADS},
}

# Live ensembles and the deterministic model whose calibration their members use.
ENSEMBLE_MODELS = {
    "ecmwf_ifs025": {"members": 51, "variables": ["shortwave_radiation", "wind_speed_100m"], "det": "ecmwf_ifs025"},
    "icon_seamless_eps": {"members": 40, "variables": ["shortwave_radiation", "wind_speed_80m"], "det": "icon_seamless"},
    "gfs025": {"members": 31, "variables": ["shortwave_radiation"], "det": "gfs_seamless"},
}
WIND_VARIABLE = {"ecmwf_ifs025": "wind_speed_100m", "icon_seamless": "wind_speed_80m"}
WIND_HEIGHT = {"wind_speed_100m": 100.0, "wind_speed_80m": 80.0, "wind_speed_10m": 10.0}

HINDCAST_START = date(2024, 6, 14)  # first day of PSE his-wlk-cal data
MULTI_MODEL_START = date(2025, 3, 1)  # ICON and GFS runs are archived from here (keeps the download within API limits)
TEST_START = date(2025, 10, 1)  # issues before this tune the method, later ones are the reported test period
TEST_END = date(2026, 9, 30)  # last verified day of the reported twelve-month test
# As-of rule of the hindcast: forecasts are issued at 08:40 UTC (the daily workflow) and a model run counts as
# published 7 h after its initialisation (Open-Meteo states 4-6 h for global models; the rule of pl-forecast-bench).
ISSUE_HOUR_UTC = 8 + 40 / 60
RUN_DELAY_H = 7.0
CLIMATE_START_YEAR = 1991
NORMAL_PERIOD = (1991, 2020)
ERA5_DAILY = ["shortwave_radiation_sum", "wind_speed_100m_mean"]
ERA5_HOURLY = ["shortwave_radiation", "wind_speed_100m"]
ERA5_HOURLY_YEARS = (2025, 2025)  # one full year of hourly ERA5 to derive daily transfer functions
