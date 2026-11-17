"""
Wyjaśnienia modeli: SHAP (logit i LightGBM), PDP / ICE / ALE, LIME, miary stabilności.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.special import logit as _logit
from scipy.stats import spearmanr

EPS = 1e-6


def to_logodds(p):
    return _logit(np.clip(p, EPS, 1 - EPS))


# ---------------------------------------------------------------- SHAP
def shap_lgbm(pipe, X):

    Xt = pipe.named_steps["cat"].transform(X)
    contrib = pipe.named_steps["lgbm"].predict(Xt, pred_contrib=True)
    return pd.DataFrame(contrib[:, :-1], index=X.index, columns=X.columns), float(contrib[0, -1])


def shap_logit(pipe, X, X_background):

    woe, lr = pipe.named_steps["woe"], pipe.named_steps["lr"]
    W, Wb = woe.transform(X), woe.transform(X_background)
    beta = pd.Series(lr.coef_[0], index=W.columns)
    phi = (W - Wb.mean()) * beta
    return phi, float(lr.intercept_[0] + (Wb.mean() * beta).sum())


def importance_table(phi_by_model: dict) -> pd.DataFrame:

    t = pd.DataFrame({m: phi.abs().mean() for m, phi in phi_by_model.items()}).fillna(0.0)
    return t.sort_values(t.columns[-1], ascending=False)


def jaccard_topk(a: pd.Series, b: pd.Series, k=5) -> float:

    A, B = set(a.nlargest(k).index), set(b.nlargest(k).index)
    return len(A & B) / len(A | B)


def rank_agreement(a: pd.Series, b: pd.Series, k=5) -> dict:
    common = a.index.union(b.index)
    a, b = a.reindex(common, fill_value=0), b.reindex(common, fill_value=0)
    return {"Spearman": spearmanr(a, b).statistic, f"Jaccard top-{k}": jaccard_topk(a, b, k)}


# ---------------------------------------------------------------- PDP / ICE / ALE (Lab 5 - z definicji)
def quantile_grid(x: pd.Series, n=25, lo=0.02, hi=0.98):

    return np.unique(np.nanquantile(x, np.linspace(lo, hi, n)))


def ice_curves(predict_logodds, X: pd.DataFrame, feature, grid):

    out = np.empty((len(X), len(grid)))
    for j, v in enumerate(grid):
        Xv = X.copy(); Xv[feature] = v
        out[:, j] = predict_logodds(Xv)
    return out


def ale_1d(predict_logodds, X: pd.DataFrame, feature, K=20):

    x = X[feature]
    edges = np.unique(np.nanquantile(x, np.linspace(0, 1, K + 1)))
    ok = x.notna()
    idx = np.clip(np.searchsorted(edges, x[ok], side="left") - 1, 0, len(edges) - 2)
    Xo = X[ok]
    lo, hi = Xo.copy(), Xo.copy()
    lo[feature], hi[feature] = edges[idx], edges[idx + 1]
    diff = predict_logodds(hi) - predict_logodds(lo)
    local = pd.Series(diff).groupby(idx).mean().reindex(range(len(edges) - 1), fill_value=0.0)
    counts = pd.Series(idx).value_counts().reindex(range(len(edges) - 1), fill_value=0)
    ale = np.concatenate([[0.0], np.cumsum(local.values)])
    centers = (ale[:-1] + ale[1:]) / 2                      # wyśrodkowanie: średni efekt w próbie = 0
    ale = ale - np.sum(centers * counts.values) / counts.sum()
    return edges, ale, int(counts.min())


def nonadditive_share(predict_logodds, X, f1, f2, n=10):

    g1, g2 = quantile_grid(X[f1], n), quantile_grid(X[f2], n)
    P = np.empty((len(g1), len(g2)))
    for i, a in enumerate(g1):
        for j, b in enumerate(g2):
            Xv = X.copy(); Xv[f1] = a; Xv[f2] = b
            P[i, j] = predict_logodds(Xv).mean()
    Pc = P - P.mean()
    inter = Pc - Pc.mean(axis=1, keepdims=True) - Pc.mean(axis=0, keepdims=True)
    return P, g1, g2, float((inter ** 2).sum() / (Pc ** 2).sum())


# ---------------------------------------------------------------- LIME (Lab 6)
class LimeAdapter:


    def __init__(self, X_train: pd.DataFrame, cat_cols):
        self.columns = list(X_train.columns)
        self.cat_cols = [c for c in cat_cols if c in self.columns]
        self.cat_idx = [self.columns.index(c) for c in self.cat_cols]
        self.levels = {c: sorted(X_train[c].astype(str).unique()) for c in self.cat_cols}
        self.medians = X_train.drop(columns=self.cat_cols).median()

    def encode(self, X: pd.DataFrame) -> np.ndarray:
        Z = X[self.columns].copy()
        for c in self.cat_cols:
            Z[c] = Z[c].astype(str).map({v: i for i, v in enumerate(self.levels[c])}).fillna(0)
        Z[self.medians.index] = Z[self.medians.index].fillna(self.medians)
        return Z.to_numpy(dtype=float)

    def decode(self, A: np.ndarray) -> pd.DataFrame:
        Z = pd.DataFrame(A, columns=self.columns)
        for c in self.cat_cols:
            lv = np.array(self.levels[c])
            Z[c] = lv[np.clip(np.round(Z[c]).astype(int), 0, len(lv) - 1)]
        return Z

    def predict_fn(self, model):
        return lambda A: model.predict_proba(self.decode(A))


def lime_weights(exp, columns) -> pd.Series:

    label = list(exp.local_exp)[0]
    return pd.Series({columns[i]: w for i, w in exp.local_exp[label]})


def pairwise_stability(weight_list, k=5) -> dict:

    jac, spe = [], []
    for a, b in itertools.combinations(weight_list, 2):
        idx = a.index.union(b.index)
        a2, b2 = a.reindex(idx, fill_value=0), b.reindex(idx, fill_value=0)
        jac.append(jaccard_topk(a2.abs(), b2.abs(), k)); spe.append(spearmanr(a2, b2).statistic)
    return {f"Jaccard top-{k}": float(np.mean(jac)), "Spearman wag": float(np.mean(spe))}