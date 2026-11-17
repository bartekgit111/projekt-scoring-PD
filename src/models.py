"""
Wspólne procedury uczenia: CV z predykcjami out-of-fold, predykcje valid/test, zapis predykcji.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone

from . import config as cfg
from .metrics import metrics

PRED_DIR = cfg.ROOT / "outputs" / "predictions"


def cross_validate_oof(model, X, y, splits):
    """
    Uczy kopię modelu na każdym foldzie treningowym; zwraca (predykcje OOF, tabela metryk per fold
    )"""
    oof = pd.Series(np.nan, index=splits["train"], name="p")

    rows = []

    for k, f in enumerate(splits["folds"], 1):
        m = clone(model).fit(X.loc[f["train"]], y.loc[f["train"]])
        p = m.predict_proba(X.loc[f["val"]])[:, 1]
        oof.loc[f["val"]] = p
        rows.append({"fold": k, **metrics(y.loc[f["val"]], p)})

    return oof, pd.DataFrame(rows).set_index("fold")


def summarize_folds(fold_table: pd.DataFrame):
    """
    Średnia i odchylenie standardowe metryk między foldami
    """
    cols = ["AUC", "Gini", "KS", "PR-AUC", "log-loss", "Brier"]

    return fold_table[cols].agg(["mean", "std"]).T.rename(columns={"mean": "średnia", "std": "odch. std"})


def save_predictions(name, oof, p_valid, p_test, splits):
    """
    Zapis predykcji
    """
    fold_of = {i: k for k, f in enumerate(splits["folds"], 1) for i in f["val"]}

    parts = [pd.DataFrame({"id": oof.index, "zbior": "oof", "fold": [fold_of[i] for i in oof.index], "p": oof.values}),
             pd.DataFrame({"id": splits["valid"], "zbior": "valid", "fold": 0, "p": p_valid}),
             pd.DataFrame({"id": splits["test"], "zbior": "test", "fold": 0, "p": p_test})]

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(PRED_DIR / f"{name}.csv", index=False)

    return out


def load_predictions(name):
    return pd.read_csv(PRED_DIR / f"{name}.csv")