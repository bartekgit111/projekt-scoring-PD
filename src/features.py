"""
Przygotowanie cech

"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config as cfg

CAT_COLS = ["formaWlasnosci_Symbol", "pkd_sekcja", "schemat_wsk_bilans", "schemat_wsk_rzis"]
SIZE_COLS = {"Aktywa": "log_aktywa", "przychody": "log_przychody"}


_PKD_SECTIONS = [("A", 1, 3), ("B", 5, 9), ("C", 10, 33), ("D", 35, 35), ("E", 36, 39), ("F", 41, 43),
                 ("G", 45, 47), ("H", 49, 53), ("I", 55, 56), ("J", 58, 63), ("K", 64, 66), ("L", 68, 68),
                 ("M", 69, 75), ("N", 77, 82), ("O", 84, 84), ("P", 85, 85), ("Q", 86, 88), ("R", 90, 93),
                 ("S", 94, 96), ("T", 97, 98), ("U", 99, 99)]


def pkd_section(pkd_code: pd.Series):
    """
    Kod PKD zapisany jako liczba -> litera sekcji.
    """
    division = pkd_code.astype(int) // 100
    out = pd.Series("inne", index=pkd_code.index)
    for letter, lo, hi in _PKD_SECTIONS:
        out[(division >= lo) & (division <= hi)] = letter
    return out


def build_base_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wstępne przekształcenia
    """

    wsk = [c for c in df.columns if c.startswith("wsk_") and c not in cfg.DROP_COLS]

    parts = [df[wsk].replace([np.inf, -np.inf], np.nan)]  # inf  -> brak

    parts.append(pd.DataFrame({new: np.log1p(df[raw].clip(lower=0)) for raw, new in SIZE_COLS.items()})) # ujemne kwoty -> 0

    parts.append(pd.DataFrame({
        "formaWlasnosci_Symbol": df["formaWlasnosci_Symbol"].astype(str),
        "pkd_sekcja": pkd_section(df["pkdKod"]),
        "schemat_wsk_bilans": df["schemat_wsk_bilans"].astype(str),
        "schemat_wsk_rzis": df["schemat_wsk_rzis"].astype(str)}))
    out = pd.concat(parts, axis=1)
    return out


def select_columns(X_train, max_missing=0.5, max_mode_share=0.95):
    """
    Zwraca (lista zachowanych cech liczbowych, słownik powodów odrzucenia). Liczone TYLKO na train.
    """
    num = X_train.drop(columns=CAT_COLS)
    reasons = {}

    for c in num.columns:
        x = num[c]
        if x.isna().mean() > max_missing:
            reasons[c] = f"braki > {max_missing:.0%}"
        elif x.value_counts(normalize=True, dropna=False).iloc[0] > max_mode_share:
            reasons[c] = f"quasi-stała (> {max_mode_share:.0%} jednej wartości)"

    kept = [c for c in num.columns if c not in reasons]

    dup = num[kept].T.duplicated()

    for c in np.array(kept)[dup.to_numpy()]:
        reasons[c] = "duplikat innej kolumny"

    kept = [c for c in kept if c not in reasons]

    return kept, reasons


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Capping do percentyli [lower, upper] wyznaczonych na danych uczących (NaN pomijane).
    """

    def __init__(self, lower=0.01, upper=0.99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lo_ = X.quantile(self.lower)
        self.hi_ = X.quantile(self.upper)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.clip(lower=self.lo_, upper=self.hi_, axis=1)


class RareGrouper(BaseEstimator, TransformerMixin):
    """
    Zamiana kategorii o udziale < min_share na danych uczących na 'inne'
    """

    def __init__(self, min_share=0.02):
        self.min_share = min_share

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.keep_ = {c: set(X[c].value_counts(normalize=True)[lambda s: s >= self.min_share].index) for c in X.columns}
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for c, keep in self.keep_.items():
            X[c] = X[c].where(X[c].isin(keep), "inne")
        return X


def information_value(x, y, n_bins=10):
    """
    IV na binach kwantylowych (braki = osobny bin).
    """
    if x.dtype == object or isinstance(x.dtype, pd.CategoricalDtype) or x.dtype == "string":
        bins = x.astype(str)
    else:
        bins = pd.qcut(x, q=n_bins, duplicates="drop").astype(str)    # braki -> 'nan'

    t = pd.crosstab(bins, y)

    good = (t[0] + 0.5) / (t[0].sum() + 0.5)
    bad = (t[1] + 0.5) / (t[1].sum() + 0.5)

    return float(((good - bad) * np.log(good / bad)).sum())




class CategoryPrep(BaseEstimator, TransformerMixin):
    """Dla LightGBM: rzadkie kategorie -> 'inne', potem typ pandas 'category'
    ze stałą listą kategorii z danych uczących (nowa kategoria w valid/test -> 'inne').
    """

    def __init__(self, cat_cols=(), min_share=0.02):
        self.cat_cols, self.min_share = cat_cols, min_share

    def fit(self, X, y=None):
        cats = [c for c in self.cat_cols if c in X.columns]
        self.rare_ = RareGrouper(self.min_share).fit(X[cats])

        grouped = self.rare_.transform(X[cats])

        self.categories_ = {c: sorted(set(grouped[c]) | {"inne"}) for c in cats}

        return self

    def transform(self, X):
        X = X.copy()

        if self.categories_:
            grouped = self.rare_.transform(X[list(self.categories_)])

            for c, cats in self.categories_.items():
                X[c] = pd.Categorical(grouped[c], categories=cats)

        return X