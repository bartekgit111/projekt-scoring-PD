
# Instrukcja uruchomienia

## Jak uruchomić
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux / Mac
pip install -r requirements.txt
jupyter lab
```
Dane: `data/zbior_9.csv` i `data/Dane_Opis_zmiennych.pdf`.

Notebooki uruchamiamy **po kolei** z katalogu `notebooks/`. Każdy zapisuje wyniki
do `outputs/`, a kolejne je wczytują.

| Notebook               | Zawartość                                                                                                                                               |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_dane_podzial`      | struktura danych, kolumny bez opisu i stałe, podział 70/15/15 + 5-fold CV, przesiew AUC (anty-leak), schemat walidacji                                  |
| `02_eda_jakosc_v2`     | Data Quality, selekcja kolumn, capping, kategorie, IV, DR w decylach (kształt U), korelacje                                                             |
| `03_model_logit`       | binning i WoE, pula cech, selekcja krokowa na CV, regresja logistyczna, VIF, profile WoE, karta punktowa                                                |
| `04_model_blackbox`    | LightGBM: random search z kontrolą przeuczenia, ograniczenia monotoniczne, porównanie z logitem                                                         |
| `05_wyjasnialnosc_v3`  | SHAP (ważność, beeswarm, zależności), PDP/c-ICE, ALE, 2D PDP, permutacja, case studies (waterfall, kody przyczyn, what-if), LIME, wierność i stabilność |
| `06_kalibracja`        | diagnostyka pre-cal, Platt vs izotoniczna, dostosowanie do 4%, reliability post-cal                                                                     |
| `07_rating_v2`         | skala ratingowa, backtest klas, próg z funkcji zysku, tabela decyzyjna, porównanie decyzji, kontrfakt (karta)                                           |
| `08_ewaluacja_koncowa` | jednorazowa ocena na teście, audyt podgrup, porównanie modeli                                                                                           |

## Struktura repozytorium
```
data/            zbiór danych i opis zmiennych
splits/          splits.json – zapisany podział (indeksy train/valid/test i foldów)
src/             moduły: config, data, features, woe, metrics, models, explain, calibration, rating
notebooks/       notebooki 01–08 (wykonane, z wynikami)
outputs/
  figures/       wykresy (png)
  tables/        tabele wyników (csv, json)
  predictions/   predykcje OOF / valid / test, PD po kalibracji
  models/        zapisane modele i kalibratory (joblib)
prezentacja.pptx
raport.md        raport techniczny
MODEL_CARD.md    karta modelu
```

## Replikowalność
- seed 42 i wszystkie ustawienia  w `src/config.py`,
- podział zapisany w `splits/splits.json` i wczytywany przez wszystkie notebooki,
- wersje cech: `outputs/tables/02_cechy.json` (po selekcji kolumn), `03_cechy_logit.json` (pula i cechy karty),
- wszystkie dopasowania (capping, binning, WoE, selekcja, tuning) na train lub foldach treningowych; kalibracja i próg na valid; test używany jednorazowo w notebooku 08.


