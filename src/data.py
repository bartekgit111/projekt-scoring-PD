"""
Wczytanie danych i podział s train / valid / test + foldy CV.
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import config as cfg


def load_data(path=cfg.DATA_PATH):
    """
    Wczytanie danych
    Index w ramce danych - id
    """
    df = pd.read_csv(path)
    df.index.name = "id"
    return df


def make_splits(y, seed):
    """
    Funkcja dzieli dane w proporcji 70/15/15 i tworzy 5 foldów CV na części treningowej.

    Zwraca słownik z listami identyfikatorów (indeksów DataFrame):
        -train
        -valid
        -test
        -folds
    """
    idx = y.index.to_numpy()

    # test - 15%, reszta 85%
    idx_rest, idx_test = train_test_split(idx, test_size=cfg.TEST_SIZE, stratify=y.loc[idx], random_state=seed)

    # valid - 15% z reszty, pozostałe to train
    valid_share = cfg.VALID_SIZE / (1 - cfg.TEST_SIZE)
    idx_train, idx_valid = train_test_split(idx_rest, test_size=valid_share, stratify=y.loc[idx_rest], random_state=seed)

    # 3) 5-fold CV tylko na train
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=seed)
    idx_train = np.sort(idx_train)

    folds = []

    for tr, va in skf.split(idx_train, y.loc[idx_train]):
        fold = {
            "train": idx_train[tr].tolist(),
            "val": idx_train[va].tolist()
        }
        folds.append(fold)


    return {"seed": seed,
            "train": idx_train.tolist(),
            "valid": np.sort(idx_valid).tolist(),
            "test": np.sort(idx_test).tolist(),
            "folds": folds}


def save_splits(splits: dict, path=cfg.SPLITS_PATH):
    """
    Funkcja zapisuje podzielone dane w pliku json
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits))


def load_splits(path=cfg.SPLITS_PATH):
    """
    Funkcja wczytuje podzielone dane z pliku
    """
    return json.loads(path.read_text())

