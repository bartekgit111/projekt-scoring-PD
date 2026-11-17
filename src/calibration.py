"""
Kalibracja PD: diagnostyka, kształt (Platt / izotoniczna), poziom (przesunięcie δ do średniego PD = 4%).

"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit, logit
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from . import config as cfg
from .metrics import metrics

EPS = 1e-6


def z(p):
    return logit(np.clip(p, EPS, 1 - EPS))


# ---------------------------------------------------------------- diagnostyka
def citl_slope(y, p):

    y, zp = np.asarray(y), z(np.asarray(p))
    citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=zp).fit().params[0]
    slope = sm.GLM(y, sm.add_constant(zp), family=sm.families.Binomial()).fit().params[1]
    return float(citl), float(slope)


def calib_report(y, p) -> dict:
    m = metrics(y, p)
    citl, slope = citl_slope(y, p)
    return {k: m[k] for k in ["DR", "średnie PD", "AUC", "Brier", "log-loss", "ECE", "ACE"]} | \
           {"CITL (intercept)": citl, "slope": slope}


def reliability(ax, y, p, label, n_bins=10, **kw):

    obs, pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    ax.plot(pred, obs, "o-", label=label, **kw)


def log_axes(ax, lo=3e-3, hi=0.9):
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("średnie PD w przedziale"); ax.set_ylabel("obserwowany DR"); ax.grid(alpha=0.3, which="both")


# ---------------------------------------------------------------- kształt
FIT = {
    "bez kalibracji": lambda y, p: None,
    "Platt": lambda y, p: LogisticRegression(C=1e6, max_iter=1000).fit(z(p).reshape(-1, 1), y),
    "izotoniczna": lambda y, p: IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(p, y),
}


def apply_cal(method, cal, p):

    p = np.asarray(p)
    if method == "bez kalibracji":
        return p
    if method == "Platt":
        return cal.predict_proba(z(p).reshape(-1, 1))[:, 1]
    return cal.predict(p)


def cv_logloss(method, y, p, k=5, seed=cfg.SEED):

    y, p = np.asarray(y), np.asarray(p)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    out = []
    for a, b in skf.split(p.reshape(-1, 1), y):
        q = apply_cal(method, FIT[method](y[a], p[a]), p[b])
        out.append(log_loss(y[b], np.clip(q, EPS, 1 - EPS)))
    return float(np.mean(out))


# ---------------------------------------------------------------- poziom
def delta_to_mean(p, target, lo=-15.0, hi=15.0, iters=100):

    zp = z(p)
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if expit(zp + mid).mean() < target else (lo, mid)
    return (lo + hi) / 2


def shift(p, delta):
    return expit(z(p) + delta)