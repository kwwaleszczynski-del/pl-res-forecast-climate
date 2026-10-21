"""Statistical calibration of physical power forecasts against PSE generation, and probabilistic dressing.

1. MOS regression (per technology, model and lead day), refitted every day on a rolling window of the latest
   ``window`` days that were already observed at issue time:

       y = x * (a + b * min(x, x_max) + c . f) + d

   x is the physical national output from power.py, f are optional time features (for PV the sun's azimuth and a
   low-sun indicator, which let the fit correct the daily profile, e.g. row shading in winter) and d is used for wind
   only. Above the largest x of the window the curve continues linearly instead of following the parabola.
2. Error model: errors y - y_hat of earlier out-of-sample forecasts of the same lead (falling back to in-sample
   residuals at the start) are grouped by the predicted level, as a share of the window's peak output, and stored as
   empirical quantiles, optionally inflated by a factor tuned on the first hindcast year.
3. Dressing: every calibrated member m gets the error distribution of its own model added, scaled by
   ``k = sqrt(max(k_min^2, 1 - s^2 / sigma^2))`` where ``s`` is the spread of the calibrated members and ``sigma``
   the error standard deviation. With one member (k = 1) this is the plain statistical error model; when the
   ensemble already spreads as much as the typical error only ``k_min`` of the kernel is kept. The mixture is clipped
   to [0, 1.1 x window peak] and summarised as quantiles.

Usage:
    mos = fit_mos(x_train, y_train, tech="wind")
    y_hat = mos.predict(x_members)                    # (..., T)
    q = dressed_quantiles(y_hat, mos, QUANTILES)      # (n_quantiles, T)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LEVEL_EDGES = np.array([0.0, 0.03, 0.1, 0.2, 0.32, 0.46, 0.62, np.inf])
N_RESID = 40  # residual quantiles kept per bin
MIN_BIN = 48
K_MIN = 0.5


@dataclass
class MOS:
    tech: str
    coef: np.ndarray            # a, b, c_1..c_k, d
    x_max: float                # largest x in training
    peak: float                 # largest observed y in training (capacity proxy)
    resid_q: np.ndarray         # (n_bins, N_RESID) error quantiles
    resid_sd: np.ndarray        # (n_bins,)
    n_train: int = 0
    daily_resid_q: np.ndarray = field(default_factory=lambda: np.zeros(N_RESID))

    def predict(self, x, feats=None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        a, b, *c, d = self.coef
        slope = a + b * np.minimum(x, self.x_max)
        if c:
            slope = slope + np.asarray(feats, dtype=float) @ np.asarray(c)
        y = np.clip(x * slope + d, 0, 1.1 * self.peak)
        if self.tech == "pv":
            y = np.where(x <= 0, 0.0, y)
        return y

    def level_bin(self, y_hat) -> np.ndarray:
        return np.clip(np.searchsorted(LEVEL_EDGES, np.asarray(y_hat) / self.peak, side="right") - 1,
                       0, len(LEVEL_EDGES) - 2)


def _residual_bins(resid: np.ndarray, bins: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    levels = (np.arange(N_RESID) + 0.5) / N_RESID
    q = np.zeros((n_bins, N_RESID))
    sd = np.zeros(n_bins)
    for b in range(n_bins):
        sel = bins == b
        lo, hi = b, b
        while sel.sum() < MIN_BIN and (lo > 0 or hi < n_bins - 1):  # widen to neighbours until enough samples
            lo, hi = max(lo - 1, 0), min(hi + 1, n_bins - 1)
            sel = (bins >= lo) & (bins <= hi)
        if sel.sum() == 0:
            continue
        q[b] = np.quantile(resid[sel], levels)
        sd[b] = resid[sel].std()
    return q, sd


def fit_mos(x: np.ndarray, y: np.ndarray, tech: str, feats: np.ndarray | None = None,
            oos: tuple[np.ndarray, np.ndarray] | None = None, oos_days: np.ndarray | None = None,
            min_oos: int = 240, inflate: float = 1.0) -> MOS:
    """Fit the regression on aligned hourly arrays (NaNs dropped) and attach an error distribution.

    ``feats`` (n, k) are extra slope features. The errors come from ``oos = (y_hat, y)``: earlier out-of-sample
    forecasts of the same lead whose outcome is already known (``oos_days`` groups them into days for the daily-total
    errors). With fewer than ``min_oos`` such hours the in-sample residuals of the fit are used instead.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    f = np.zeros((len(x), 0)) if feats is None else np.asarray(feats, dtype=float).reshape(len(x), -1)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(f).all(axis=1)
    if tech == "pv":
        ok &= x > 0
    xs, ys, fs = x[ok], y[ok], f[ok]
    if len(xs) < 48:
        raise ValueError(f"not enough training data for {tech} MOS ({len(xs)} hours)")
    cols = [xs, xs ** 2] + [xs * fs[:, j] for j in range(fs.shape[1])]
    if tech != "pv":
        cols.append(np.ones_like(xs))
    sol = np.linalg.lstsq(np.column_stack(cols), ys, rcond=None)[0]
    coef = sol if tech != "pv" else np.append(sol, 0.0)
    n_bins = len(LEVEL_EDGES) - 1
    mos = MOS(tech, coef, float(xs.max()), float(ys.max()), np.zeros((n_bins, N_RESID)), np.zeros(n_bins),
              n_train=len(xs))
    y_hat, y_obs, days = mos.predict(xs, fs), ys, None
    if oos is not None:
        oh, oy = (np.asarray(a, dtype=float) for a in oos)
        good = np.isfinite(oh) & np.isfinite(oy)
        if tech == "pv":
            good &= oh > 0
        if good.sum() >= min_oos:
            y_hat, y_obs = oh[good], oy[good]
            if oos_days is not None:
                full = np.isfinite(oh) & np.isfinite(oy)
                days = (np.asarray(oos_days)[full], oy[full] - oh[full])
    mos.resid_q, mos.resid_sd = _residual_bins(inflate * (y_obs - y_hat), mos.level_bin(y_hat), n_bins)
    if days is not None:
        di, r = days
        _, inv = np.unique(di, return_inverse=True)
        daily, counts = np.bincount(inv, weights=r), np.bincount(inv)
        daily = daily[counts >= 20] * 24.0 / counts[counts >= 20]  # scale partial days to 24 h
        if len(daily) >= 10:
            mos.daily_resid_q = inflate * np.quantile(daily, (np.arange(N_RESID) + 0.5) / N_RESID)
    return mos


def weighted_quantiles(samples: np.ndarray, weights: np.ndarray, levels) -> np.ndarray:
    """Quantiles along the last axis of ``samples`` (..., N) with weights (..., N) or (N,)."""
    order = np.argsort(samples, axis=-1)
    s = np.take_along_axis(samples, order, axis=-1)
    w = np.broadcast_to(weights, samples.shape)
    w = np.take_along_axis(w, order, axis=-1)
    cw = np.cumsum(w, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cw = (cw - 0.5 * w) / cw[..., -1:]
    out = np.empty(samples.shape[:-1] + (len(levels),))
    flat_s = s.reshape(-1, s.shape[-1])
    flat_c = cw.reshape(-1, cw.shape[-1])
    flat_o = out.reshape(-1, len(levels))
    for i in range(flat_s.shape[0]):
        flat_o[i] = np.interp(levels, flat_c[i], flat_s[i])
    return np.moveaxis(out, -1, 0)


def dressing_samples(members: np.ndarray, mos, member_weights: np.ndarray | None = None,
                     k_min: float = K_MIN) -> tuple[np.ndarray, np.ndarray]:
    """Mixture samples (T, M * N_RESID) and weights for calibrated members (M, T).

    ``mos`` is one :class:`MOS` for all members or a list with one per member (multi-model ensembles); each member
    is dressed with the residuals of its own calibration. Members that are NaN at an hour get zero weight there.
    """
    members = np.atleast_2d(np.asarray(members, dtype=float))
    M, T = members.shape
    mos_list = list(mos) if isinstance(mos, (list, tuple)) else [mos] * M
    wm = np.full(M, 1.0 / M) if member_weights is None else np.asarray(member_weights, float) / np.sum(member_weights)
    valid = np.isfinite(members)
    w_t = np.where(valid, wm[:, None], 0.0)                                   # (M, T)
    w_t = w_t / np.where(w_t.sum(axis=0) > 0, w_t.sum(axis=0), 1.0)
    m0 = np.where(valid, members, 0.0)
    mean = (w_t * m0).sum(axis=0)
    spread2 = (w_t * (m0 - mean) ** 2).sum(axis=0)
    sigma2 = np.zeros(T)
    resid = np.empty((M, T, N_RESID))
    peak = max(m.peak for m in mos_list)
    for m_obj in {id(m): m for m in mos_list}.values():
        idx = [i for i, m in enumerate(mos_list) if m is m_obj]
        sigma2 += w_t[idx].sum(axis=0) * m_obj.resid_sd[m_obj.level_bin(mean)] ** 2
        resid[idx] = m_obj.resid_q[m_obj.level_bin(m0[idx])]
    with np.errstate(divide="ignore", invalid="ignore"):
        k2 = np.where(sigma2 > 0, 1 - spread2 / sigma2, 1.0)
    k = np.sqrt(np.maximum(k_min ** 2, k2))
    k = np.where(valid.sum(axis=0) > 1, k, 1.0)
    samples = m0[:, :, None] + k[None, :, None] * resid
    samples = np.clip(samples, 0, 1.1 * peak)
    if mos_list[0].tech == "pv":
        samples = np.where(m0[:, :, None] <= 0, 0.0, samples)
    samples = np.moveaxis(samples, 0, 1).reshape(T, M * N_RESID)
    weights = np.repeat(w_t.T, N_RESID, axis=1) / N_RESID                     # (T, M * N_RESID)
    return samples, weights


def dressed_quantiles(members: np.ndarray, mos, levels, member_weights=None, k_min: float = K_MIN) -> np.ndarray:
    samples, weights = dressing_samples(members, mos, member_weights, k_min)
    return weighted_quantiles(samples, weights, levels)


def daily_total_samples(member_totals: np.ndarray, daily_resid_q: np.ndarray, member_weights=None,
                        k: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Samples of a daily energy total: member totals (M,) dressed with daily residual quantiles."""
    m = np.atleast_1d(np.asarray(member_totals, dtype=float))
    wm = np.full(len(m), 1.0 / len(m)) if member_weights is None else np.asarray(member_weights) / np.sum(member_weights)
    s = np.clip(m[:, None] + k * daily_resid_q[None, :], 0, None).ravel()
    return s, np.repeat(wm, len(daily_resid_q)) / len(daily_resid_q)


def crps_sample(samples: np.ndarray, weights: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS of weighted samples (T, N) for observations y (T,): E|X-y| - 0.5 E|X-X'|."""
    order = np.argsort(samples, axis=-1)
    s = np.take_along_axis(samples, order, axis=-1)
    w = np.take_along_axis(np.broadcast_to(weights, samples.shape), order, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = w / w.sum(axis=-1, keepdims=True)
    term1 = (w * np.abs(s - y[:, None])).sum(axis=-1)
    # E|X-X'| = 2 * sum_i w_i x_i (2 F_i - w_i - 1), with F_i the cumulative weight up to and including i
    F = np.cumsum(w, axis=-1)
    term2 = (w * s * (2 * F - w - 1)).sum(axis=-1) * 2
    return term1 - 0.5 * term2
