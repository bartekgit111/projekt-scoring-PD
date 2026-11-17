"""Parametry całego projektu"""


from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "zbior_9.csv"       # folder z danymi
SPLITS_PATH = ROOT / "splits" / "splits.json"   # folder z podziałem danych
FIG_DIR = ROOT / "outputs" / "figures"          # folder na wykresy
TAB_DIR = ROOT / "outputs" / "tables"           # folder na tabelki

SEED = 42               #ziarno generatora
TARGET = "default"      # zmienna objasniana

# Podział danych: 70 15 15
TEST_SIZE = 0.15
VALID_SIZE = 0.15
N_FOLDS = 5     # 5-krotna kroswalidacja

# Kolumny wykluczone z modelowania
LEAK_COLS = ["wsk_index_aktywne", "wsk_index_skr_zaw", "wsk_index_nowe"]                        # brak opisu
CONST_COLS = ["szczegolnaFormaPrawna_Symbol", "Zobowiazania_dostaw_uslug_pow12m_kapitale"]      # jedna wartość w całym zbiorze
DROP_COLS = LEAK_COLS + CONST_COLS                                                              # kolumny do usunięcia

# Cechy, dla których w karcie scoringowej dopuszczamy kształt U
U_SHAPE = ["wsk_kapital_do_aktywa", "wsk_udzial_kap_wlasnego_aktywa_1", "wsk_udzial_kap_wlasnego_aktywa_2",
           "wsk_struktura_kap_wlasnego_2", "wsk_struktura_kap_wlasnego_s_1", "wsk_mnoznik_kap_wl",
           "wsk_ogolnego_zadluzenia_2", "wsk_struktury_kapitalu"]


PD_TARGET = 0.04 #  tendencja centralna PD