"""
Binning + WoE dla karty scoringowej.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class WoEBinner(BaseEstimator, TransformerMixin):
    def __init__(self, max_bins=5, init_bins=10, min_share=0.05, smoothing=0.5, u_shape=()):
        self.max_bins, self.init_bins, self.min_share = max_bins, init_bins, min_share
        self.smoothing, self.u_shape = smoothing, u_shape

    def _stats(self, codes, y, k):
        n = np.bincount(codes, minlength=k).astype(float)
        bad = np.bincount(codes, weights=y, minlength=k)
        return n, bad

    def _num_cuts(self, x, y, monotone=True):
        """
        Granice przedziałów: start od kwantyli, łączenie za małych, niemonotonicznych i nadmiarowych.
        """
        cuts = list(np.unique(np.quantile(x, np.linspace(0, 1, self.init_bins + 1)[1:-1])))
        rho = pd.Series(x).corr(pd.Series(y), method="spearman")
        direction = 1.0 if (not np.isfinite(rho) or rho == 0) else float(np.sign(rho))

        N = len(x)

        while cuts:
            codes = np.searchsorted(cuts, x, side="right"); k = len(cuts) + 1
            n, bad = self._stats(codes, y, k)
            br = bad / np.maximum(n, 1)
            small = np.where(n / N < self.min_share)[0]
            if len(small):                                   # za mały przedział -> łączymy z podobniejszym sąsiadem
                i = small[0]
                j = i - 1 if i == k - 1 else (i if i == 0 else (i - 1 if abs(br[i] - br[i - 1]) < abs(br[i] - br[i + 1]) else i))
                cuts.pop(j); continue
            if monotone:                                     # naruszenie monotoniczności
                viol = np.where(np.diff(br) * direction < 0)[0]
                if len(viol):
                    cuts.pop(viol[0]); continue
            if k > self.max_bins:                            # za dużo przedziałów -> łączymy najbardziej podobne
                cuts.pop(int(np.argmin(np.abs(np.diff(br))))); continue
            break

        return cuts

    def _table(self, codes, y, labels):
        k = len(labels); n, bad = self._stats(codes, y, k); good = n - bad; s = self.smoothing
        pg = (good + s) / (good.sum() + s * k); pb = (bad + s) / (bad.sum() + s * k)
        woe = np.log(pg / pb)

        return pd.DataFrame({"przedział": labels, "N": n.astype(int), "udział": n / n.sum(), "złe": bad.astype(int),
                             "stopa złych": bad / np.maximum(n, 1), "% dobrych": pg, "% złych": pb,
                             "WoE": woe, "IV": (pg - pb) * woe})

    def fit(self, X, y):
        X = pd.DataFrame(X); y = np.asarray(y).astype(float)

        self.cuts_, self.groups_, self.tables_, self.woe_, self.iv_ = {}, {}, {}, {}, {}

        self.has_nan_ = {}

        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                x = X[c].to_numpy(dtype=float); ok = ~np.isnan(x)
                cuts = self._num_cuts(x[ok], y[ok], monotone=c not in self.u_shape)
                self.cuts_[c] = cuts
                edges = [-np.inf] + cuts + [np.inf]
                labels = [f"[{edges[i]:.4g}; {edges[i+1]:.4g})" for i in range(len(edges) - 1)]
                codes = np.searchsorted(cuts, np.where(ok, x, 0), side="right")
                self.has_nan_[c] = bool((~ok).any())
                if self.has_nan_[c]:                         # MODYFIKACJA: braki = ostatni, osobny przedział
                    codes = np.where(ok, codes, len(labels)); labels = labels + ["brak"]
            else:
                share = X[c].value_counts(normalize=True)
                groups = {v: (v if share[v] >= self.min_share else "INNE") for v in share.index}
                labels = sorted(set(groups.values()), key=lambda g: (g == "INNE", str(g)))
                self.groups_[c] = (groups, labels)
                codes = X[c].map(groups).map({g: i for i, g in enumerate(labels)}).to_numpy()
            tab = self._table(codes, y, labels)
            self.tables_[c] = tab; self.woe_[c] = tab["WoE"].to_numpy(); self.iv_[c] = float(tab["IV"].sum())

        self.feature_names_in_ = np.array(X.columns)
        return self

    def codes(self, X):
        """
        Numer przedziału dla każdej obserwacji
        (-1 = brak przedziału w danych uczących -> WoE 0)
        """
        X = pd.DataFrame(X); out = {}

        for c in self.feature_names_in_:
            if c in self.cuts_:
                x = X[c].to_numpy(dtype=float); ok = ~np.isnan(x)
                cd = np.searchsorted(self.cuts_[c], np.where(ok, x, 0), side="right")
                nan_code = len(self.cuts_[c]) + 1 if self.has_nan_[c] else -1
                out[c] = np.where(ok, cd, nan_code)
            else:
                groups, labels = self.groups_[c]; idx = {g: i for i, g in enumerate(labels)}
                out[c] = X[c].map(lambda v: idx[groups[v]] if v in groups else -1).to_numpy()

        return pd.DataFrame(out, index=X.index)

    def transform(self, X):
        cd = self.codes(X)
        return pd.DataFrame({c: np.where(cd[c] >= 0, self.woe_[c][np.clip(cd[c], 0, None)], 0.0)
                             for c in self.feature_names_in_}, index=cd.index)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_