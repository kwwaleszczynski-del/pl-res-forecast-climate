import json

import numpy as np
import pandas as pd

from res_forecast.archive import day_ahead, live_verification, load_archive, save_forecast
from res_forecast.calibration import fit_mos
from res_forecast.forecast import lead_days, low_res_threshold, run_live
from res_forecast.physical import solar_features
from res_forecast.site import build_payload, render

rng = np.random.default_rng(7)


def test_lead_days_follow_local_calendar_days_across_dst():
    issue = pd.Timestamp("2025-10-25 07:00", tz="UTC")  # Saturday morning, DST ends that night
    times = pd.date_range("2025-10-25 22:00", periods=4, freq="h", tz="UTC")  # hour ends 00:00..03:00 local
    leads = lead_days(times, issue)
    # the hour ending 00:00 local belongs to the 25th (lead 0), later hours to the 26th (lead 1)
    assert list(leads) == [0, 1, 1, 1]


def _mos(tech):
    x = rng.uniform(1, 8000, 3000)
    feats = rng.uniform(-1, 1, (3000, 2)) if tech == "pv" else None
    days = np.repeat(np.arange(125), 24)[:3000]
    y = x + rng.normal(0, 300, x.size)
    return fit_mos(x, y, tech, feats=feats, oos=(x + rng.normal(0, 300, x.size), y), oos_days=days)


def _live():
    issue = pd.Timestamp("2025-06-10 07:00", tz="UTC")
    times = pd.date_range("2025-06-10 01:00", periods=192, freq="h", tz="UTC")
    sun = np.clip(np.sin((times.hour.to_numpy() - 4) / 16 * np.pi), 0, None)
    raw = {"ecmwf_ifs025": {"pv": 9000 * sun * rng.uniform(0.6, 1.0, (51, 1)),
                            "wind": rng.uniform(1000, 6000, (51, 192))},
           "gfs_seamless": {"pv": 8000 * sun * rng.uniform(0.6, 1.0, (31, 1))}}
    calibs = {(m, t, lead): _mos(t) for m in raw for t in raw[m] for lead in range(1, 8)}
    actual = pd.DataFrame({"pv": rng.uniform(0, 9000, 24 * 400), "wind": rng.uniform(500, 7000, 24 * 400)},
                          index=pd.date_range("2024-05-07", periods=24 * 400, freq="h", tz="UTC"))
    return run_live(raw, times, calibs, issue, actual, k_min=0.5), actual


def test_run_live_produces_ordered_fan_daily_table_and_risk():
    live, _ = _live()
    for tech in ("pv", "wind"):
        q = live.fan[tech].dropna().to_numpy()
        assert len(q) > 100
        assert (np.diff(q, axis=1) >= -1e-6).all()  # quantiles are ordered
    d1 = live.daily[live.daily["lead"] == 1].iloc[0]
    assert d1["pv_mwh_p10"] <= d1["pv_mwh_p50"] <= d1["pv_mwh_p90"]
    assert d1["pv_peak_mw_p50"] > 0
    assert ((live.risk["p_low"] >= 0) & (live.risk["p_low"] <= 1)).all()
    assert live.members["ecmwf_ifs025"] == {"pv": 51, "wind": 51}


def test_low_res_threshold_is_a_low_percentile():
    _, actual = _live()
    thr = low_res_threshold(actual)
    daily = (actual["pv"] + actual["wind"]).resample("D").mean()
    assert daily.quantile(0.02) < thr < daily.median()


def test_payload_is_json_safe_and_embedded(tmp_path):
    live, actual = _live()
    payload = build_payload(live=live, actual=actual, status={"PSE generation": "ok"})
    assert payload["live"]["fan"]["pv"]["p50"]
    out = render(payload, tmp_path)
    html = out.read_text(encoding="utf-8")
    assert "__DATA__" not in html
    data = html.split('<script id="data" type="application/json">')[1].split("</script>")[0]
    assert json.loads(data)["status"]["PSE generation"] == "ok"
    assert "NaN" not in data


def test_payload_without_live_forecast_still_renders(tmp_path):
    payload = build_payload(status={"ECMWF": "failed (HTTPError)"}, stale_note="no forecast")
    html = render(payload, tmp_path).read_text(encoding="utf-8")
    assert "no forecast" in html


def test_archive_replaces_same_day_and_verifies_day_ahead(tmp_path):
    live, actual = _live()
    save_forecast(tmp_path, live)
    save_forecast(tmp_path, live)
    a = load_archive(tmp_path)
    assert a["issue_date"].nunique() == 1
    assert len(a) == sum(len(f) for f in live.fan.values())
    d1 = day_ahead(a)
    local = (pd.to_datetime(d1["ts"], utc=True) - pd.Timedelta("1h")).dt.tz_convert("Europe/Warsaw")
    assert set(local.dt.date.astype(str)) == {"2025-06-11"}
    obs = pd.DataFrame({"pv": 5000.0, "wind": 3000.0}, index=pd.to_datetime(d1["ts"].unique(), utc=True))
    table, series = live_verification(tmp_path, obs)
    assert set(table["tech"]) == {"pv", "wind"} and (table["days"] == 1).all()


def test_solar_features_are_finite():
    f = solar_features(pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC"))
    assert f.shape == (48, 2) and np.isfinite(f).all()


def test_fan_has_no_step_at_midnight():
    """Leads with different calibrations are blended through the day, so the fan is continuous at midnight."""
    issue = pd.Timestamp("2025-06-10 07:00", tz="UTC")
    times = pd.date_range("2025-06-10 01:00", periods=192, freq="h", tz="UTC")
    raw = {"ecmwf_ifs025": {"wind": np.full((51, 192), 3000.0) + rng.normal(0, 50, (51, 192))}}
    x = rng.uniform(0, 8000, 3000)
    calibs = {("ecmwf_ifs025", "wind", lead): fit_mos(x, (1 + 0.2 * lead) * x + rng.normal(0, 50, x.size), "wind")
              for lead in range(1, 8)}
    live = run_live(raw, times, calibs, issue, None, k_min=0.5)
    med = live.fan["wind"][0.5]
    local = (med.index - pd.Timedelta("1h")).tz_convert("Europe/Warsaw")
    jumps = med.diff().abs()[(local.hour == 0)].dropna()
    assert jumps.max() < 0.05 * med.mean()
