"""
Skala ratingowa, backtest klas, próg decyzyjny z funkcji zysku i analiza kontrfaktyczna.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, binomtest

MARGIN, LGD = 0.10, 0.45
P_STAR = MARGIN / (MARGIN + LGD)

RATING_BINS = [0.00, 0.01, 0.02, 0.05, 0.08, 0.12, 0.20, 1.0 + 1e-9]
RATING_LABELS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]          # D - default


def rating(p):

    return pd.Categorical(pd.cut(np.asarray(p), bins=RATING_BINS, labels=RATING_LABELS, right=False),
                          categories=RATING_LABELS, ordered=True)


def scale_table():

    rows = [{"klasa": l, "PD od": RATING_BINS[k], "PD do": min(RATING_BINS[k + 1], 1.0)} for k, l in enumerate(RATING_LABELS)]
    rows.append({"klasa": "D", "PD od": np.nan, "PD do": np.nan})
    return pd.DataFrame(rows).set_index("klasa")


def light(p):
    return "zielone" if p >= 0.05 else ("żółte" if p >= 0.01 else "czerwone")


def backtest(y, p, r):
    y, p = np.asarray(y), np.asarray(p)
    rows = []

    for lab in RATING_LABELS:
        m = np.asarray(r == lab)
        if not m.any():
            continue

        n_k, d_k, pd_k = int(m.sum()), int(y[m].sum()), float(p[m].mean())
        pj = float(beta_dist.cdf(pd_k, d_k + 0.5, n_k - d_k + 0.5))
        rows.append({"klasa": lab, "n": n_k, "udział": m.mean(), "defaulty": d_k, "oczekiwane": n_k * pd_k,
                     "średnie PD": pd_k, "DR": d_k / n_k,
                     "p dwumianowy": binomtest(d_k, n_k, pd_k, alternative="greater").pvalue,
                     "p Jeffreysa": pj, "światło": light(pj)})

    return pd.DataFrame(rows).set_index("klasa")


def economics(y, p, prog, ead=1.0, margin=MARGIN, lgd=LGD):
    y, p = np.asarray(y), np.asarray(p)
    e = np.broadcast_to(ead, p.shape)
    acc = p < prog

    return {"próg": prog, "stopa akceptacji": acc.mean(),
            "DR wśród przyjętych": y[acc].mean() if acc.any() else np.nan,
            "DR wśród odrzuconych": y[~acc].mean() if (~acc).any() else np.nan,
            "EL przyjętych (oczekiwana)": float(np.sum(p[acc] * lgd * e[acc])),
            "strata zrealizowana": float(np.sum(lgd * e[acc & (y == 1)])),
            "zysk": float(np.sum(margin * e[acc & (y == 0)]) - np.sum(lgd * e[acc & (y == 1)]))}


# ---------------------------------------------------------------- analiza kontrfaktyczna
class Counterfactual:

    def __init__(self, pd_fn, X_train: pd.DataFrame, actionable, rho_min=0.6, lo=0.05, hi=0.95):
        self.pd_fn, self.actionable, self.lo, self.hi = pd_fn, list(actionable), lo, hi
        self.ref = {c: np.sort(X_train[c].dropna().to_numpy()) for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])}
        corr = X_train[list(self.ref)].corr(method="spearman")
        self.links = {c: {d: np.sign(corr.loc[c, d]) for d in corr.columns
                          if d != c and abs(corr.loc[c, d]) >= rho_min} for c in self.actionable}

    def pct(self, c, v):                       # wartość -> percentyl w train
        return np.searchsorted(self.ref[c], v) / len(self.ref[c])

    def val(self, c, q):                       # percentyl -> wartość w train
        return float(np.quantile(self.ref[c], np.clip(q, self.lo, self.hi)))

    def apply(self, x: pd.DataFrame, shifts: dict, consistent: bool):
        x = x.copy()
        moves = dict(shifts)

        if consistent:
            for c, s in shifts.items():
                for d, sign in self.links[c].items():
                    if d not in moves:
                        moves[d] = sign * s

        for c, s in moves.items():
            v0 = x[c].iloc[0]
            q0 = self.pct(c, v0) if pd.notna(v0) else 0.5
            x[c] = self.val(c, q0 + s)

        return x, moves

    def search(self, x: pd.DataFrame, target_pd, consistent=False, steps=np.arange(0.05, 0.91, 0.05), max_feats=2):
        for k in range(1, max_feats + 1):
            cands = []
            for feats in itertools.combinations(self.actionable, k):
                for combo in itertools.product(steps, repeat=k):
                    for signs in itertools.product([-1, 1], repeat=k):
                        shifts = {f: sg * s for f, s, sg in zip(feats, combo, signs)}
                        x2, moves = self.apply(x, shifts, consistent)
                        cands.append((feats, shifts, len(moves), x2))

            pds = self.pd_fn(pd.concat([c[3] for c in cands], ignore_index=True))

            out = [{"cechy": f, "przesunięcia": sh, "koszt": sum(map(abs, sh.values())), "PD po zmianie": float(p),
                    "zmienione (z powiązanymi)": nm, "x": x2}

                   for (f, sh, nm, x2), p in zip(cands, pds) if p < target_pd]

            if out:
                return sorted(out, key=lambda r: r["koszt"])

        return []