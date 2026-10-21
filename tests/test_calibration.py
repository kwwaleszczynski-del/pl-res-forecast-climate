import numpy as np
import pytest

from res_forecast.calibration import (N_RESID, crps_sample, dressed_quantiles, dressing_samples, fit_mos,
                                      weighted_quantiles)

rng = np.random.default_rng(1)


def test_fit_recovers_linear_relation_and_extrapolates_linearly():
    x = rng.uniform(0, 1000, 2000)
    y = 3.0 * x + 50 + rng.normal(0, 10, x.size)
    mos = fit_mos(x, y, "wind")
    assert mos.predict(np.array([500.0]))[0] == pytest.approx(1550, rel=0.01)
    # above the training range the curve continues with the slope at the edge (no parabola)
    edge = mos.predict(np.array([mos.x_max]))[0]
    slope = (mos.predict(np.array([mos.x_max + 100]))[0] - edge) / 100
    assert slope == pytest.approx(3.0, rel=0.05)


def test_pv_is_zero_without_sun_and_uses_features():
    x = rng.uniform(0, 1000, 3000)
    f = rng.uniform(-1, 1, (3000, 1))
    y = x * (1.0 + 0.3 * f[:, 0]) + rng.normal(0, 5, x.size)
    mos = fit_mos(x, y, "pv", feats=f)
    assert mos.predict(np.array([0.0]), np.array([[0.5]]))[0] == 0
    assert mos.predict(np.array([100.0]), np.array([[1.0]]))[0] == pytest.approx(130, rel=0.03)


def test_too_little_data_raises():
    with pytest.raises(ValueError):
        fit_mos(np.arange(10.0), np.arange(10.0), "wind")


def test_out_of_sample_errors_are_used_when_available():
    x = rng.uniform(0, 1000, 2000)
    y = x + rng.normal(0, 10, x.size)
    oos_hat = x + rng.normal(0, 200, x.size)  # earlier forecasts were much worse than the in-sample fit
    narrow = fit_mos(x, y, "wind")
    wide = fit_mos(x, y, "wind", oos=(oos_hat, y))
    assert wide.resid_sd.max() > 5 * narrow.resid_sd.max()


def test_single_member_dressing_reproduces_error_quantiles():
    x = rng.uniform(100, 1000, 4000)
    y = x + rng.normal(0, 50, x.size)
    mos = fit_mos(x, y, "wind")
    q = dressed_quantiles(np.array([[600.0]]), mos, [0.1, 0.5, 0.9])[:, 0]
    assert q[1] == pytest.approx(mos.predict(np.array([600.0]))[0], abs=10)
    assert q[2] - q[0] == pytest.approx(2 * 1.2816 * 50, rel=0.15)


def test_wide_ensemble_shrinks_the_kernel():
    x = rng.uniform(100, 1000, 4000)
    mos = fit_mos(x, x + rng.normal(0, 50, x.size), "wind")
    tight = np.full((10, 1), 600.0)
    wide = np.linspace(400, 800, 10)[:, None]
    s_t, w_t = dressing_samples(tight, mos)
    s_w, w_w = dressing_samples(wide, mos)
    assert s_t.shape == (1, 10 * N_RESID)
    spread = lambda s, w: weighted_quantiles(s, w, [0.9])[0, 0] - weighted_quantiles(s, w, [0.1])[0, 0]
    assert spread(s_w, w_w) > spread(s_t, w_t)
    assert spread(s_w, w_w) < (800 - 400) + 2 * 1.2816 * 50  # kernel shrunk, not simply added


def test_nan_members_get_zero_weight():
    mos = fit_mos(np.linspace(0, 1000, 500), np.linspace(0, 1000, 500) + rng.normal(0, 5, 500), "wind")
    members = np.array([[500.0, np.nan], [520.0, 300.0]])
    s, w = dressing_samples(members, mos)
    assert np.allclose(w.sum(axis=1), 1)
    q = weighted_quantiles(s, w, [0.5])[0]
    assert q[1] == pytest.approx(300, abs=20)


def test_weighted_quantiles_match_numpy_for_equal_weights():
    s = rng.normal(size=(3, 1001))
    w = np.ones(1001)
    q = weighted_quantiles(s, w, [0.1, 0.5, 0.9])
    # Hazen plotting positions differ slightly from numpy's default (type 7) in the tails
    np.testing.assert_allclose(q, np.quantile(s, [0.1, 0.5, 0.9], axis=1), atol=0.03)


def test_crps_matches_brute_force_and_point_mass():
    s = rng.normal(size=(1, 50))
    w = rng.uniform(0.5, 1.5, 50)
    y = np.array([0.3])
    wn = w / w.sum()
    brute = (wn * np.abs(s[0] - y)).sum() - 0.5 * (wn[:, None] * wn[None, :] * np.abs(s[0][:, None] - s[0][None, :])).sum()
    assert crps_sample(s, w, y)[0] == pytest.approx(brute)
    assert crps_sample(np.full((1, 5), 2.0), np.ones(5), np.array([5.0]))[0] == pytest.approx(3.0)
