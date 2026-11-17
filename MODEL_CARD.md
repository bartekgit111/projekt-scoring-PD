
## 1. Cel i zakres użycia
| Cel                 | Ocena prawdopodobieństwa niewykonania zobowiązania (PD) firmy i przypisanie klasy ratingowej AAA…CCC na potrzeby decyzji kredytowej (przyjęcie / odrzucenie).                                        |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model rekomendowany | **Karta scoringowa**: regresja logistyczna na WoE, 9 cech, 38 przedziałów, 10 parametrów; kalibracja Platta + przesunięcie log-szans do średniego PD = 4%.                                           |
| Model porównawczy   | LightGBM z ograniczeniami monotonicznymi (482 drzewa głębokości 1, 29 używanych cech), ta sama kalibracja.                                                                                           |
| Populacja           | Firmy jednej formy prawnej (kod 117), sprawozdania finansowe (bilans, RZiS); 13 sekcji PKD.                                                                                                          |
| Zdarzenie           | `default = 1`                                                                                                                                                                                        |
| Wynik               | PD po dostosowaniu do tendencji centralnej 4% → klasa ratingowa → decyzja (próg 5%).                                                                                                                 |
| Do czego NIE używać | Do oceny osób fizycznych i jednoosobowych działalności; do firm innej formy prawnej; do wyceny bez ponownej kalibracji poziomu PD; do wniosków przyczynowych („obniżenie zadłużenia obniży ryzyko”). |

## 2. Dane
- 3000 firm, DR = 13,7%; dane przekrojowe, bez dat i identyfikatora firmy, bez duplikatów.
- Podział stratyfikowany (seed 42): train 70% (2099 firm, 288 defaultów), valid 15% (451, 62), test 15% (450, 62); 5-fold CV na train.
- Wykluczone: `wsk_index_aktywne`, `wsk_index_skr_zaw`, `wsk_index_nowe` (brak opisu w słowniku danych, ryzyko przecieku), kolumny stałe (`szczegolnaFormaPrawna_Symbol`, `Zobowiazania_dostaw_uslug_pow12m_kapitale`), wskaźniki z > 50% braków (rachunek przepływów), duplikaty kolumn.
- Polityka cech: ±inf → brak; capping (1–99%, granice z train) w EDA; w karcie braki jako osobny przedział WoE; kod PKD → sekcja; kategorie < 2% → „inne”; binning monotoniczny, a dla 8 wskaźników struktury kapitału dopuszczony kształt U (`U_SHAPE`).


## 3. Wyniki
| | **Karta (logit na WoE)** | LightGBM monotoniczny |
|---|---|---|
| AUC CV (5 foldów) | **0,718 ± 0,034** | 0,708 ± 0,040 |
| AUC / Gini valid | 0,748 / 0,50 | 0,758 / 0,52 |
| AUC / Gini test [95% CI] | 0,708 / 0,42 [0,30; 0,55] | 0,730 / 0,46 [0,34; 0,59] |
| KS / PR-AUC test | 0,34 / 0,31 | 0,35 / 0,30 |
| ΔGini test (sparowany bootstrap) | – | +0,042 [−0,010; +0,091] – **nieistotna** |
| Stabilność wyjaśnień (Jaccard top-5 SHAP między foldami) | **0,77** | 0,60 |

**Kalibracja karty (test, DR 13,8%)**

| | surowe | po kalibracji kształtu (Platt) | po dostosowaniu do 4% |
|---|---|---|---|
| średnie PD | 13,0% | 12,6% | 3,6% |
| Brier / log-loss | 0,110 / 0,367 | 0,110 / 0,366 | 0,124 / 0,459 |
| calibration intercept / slope | 0,07 / 0,83 | 0,11 / 0,88 | 1,55 / 0,88 |

- Kalibracja statystyczna: karta była dobrze skalibrowana względem próby już przed kalibracją (valid: intercept −0,05, slope 0,93). Spośród dwóch metod wybrano Platta (a = 0,93, b = −0,15), bo izotoniczna przeuczała się na 451 firmach (log-loss w CV wewnątrz valid 0,403 vs 0,355). Platt jest bliski braku kalibracji (log-loss bez kalibracji 0,354) – kształt PD praktycznie nie wymagał korekty.
- Dostosowanie do 4%: δ = −1,44 w log-szansach, wyznaczone na valid. To założenie biznesowe o portfelu docelowym, a nie poprawa kalibracji. Względem próby PD jest zaniżone ok. 3,4 raza: krzywa reliability leży **powyżej** przekątnej (DR > PD), CITL ≈ 1,55. Ranking (AUC) się nie zmienia.

**Skala ratingowa i próg (test, karta)**
- Skala: AAA < 1%, AA 1–2%, A 2–5%, BBB 5–8%, BB 8–12%, B 12–20%, CCC ≥ 20%, D – stwierdzony default.
- Rozkład: AAA 18%, AA 24%, A 36%, BBB 15%, BB 4%, B 3%, CCC 1% (HHI 0,24, koncentracja w A).
- DR rośnie monotonicznie z klasą: 1% (AAA) → 11% → 13% → 23% → 29% → 38% → 67% (CCC). Backtest Jeffreysa: 3 klasy czerwone, 2 żółte, 2 zielone – skutek różnicy poziomu (PD 4% vs DR próby), a nie błędu szeregowania.
- Próg teoretyczny p* = MARGIN / (MARGIN + LGD) = 18,2% → 99% akceptacji (przy PD zaniżonym do 4% prawie nikogo nie odrzuca).
- Próg z valid 5% → akceptacja 78%, DR przyjętych 10,0% (vs 13,8%), zysk 15,9 vs 10,9 przy przyjęciu wszystkich (+45%; EAD = 1, MARGIN 10%, LGD 45%).

## 4. Interpretowalność i wyjaśnienia

- **Globalnie**: karta jest interpretowalna z konstrukcji – 9 cech, każda z 3–5 przedziałami WoE i punktami (skala PDO 20, 600 pkt przy szansach 50:1; wynik 456–622 pkt). Wszystkie β < 0, VIF < 2, kierunki zgodne z intuicją ekonomiczną. Najważniejsze cechy (średnie |SHAP| na teście): struktura kapitału, rotacja wartości niematerialnych, zysk / zobowiązania, stopa zysku ze sprzedaży, zadłużenie krótkoterminowe.
- **Struktura kapitału ma kształt U**: DR 28% przy najniższych wartościach, 5% w środku rozkładu i ponownie 12% przy bardzo wysokich (firmy z kapitałem, ale bez realnej działalności).
- **Spójność z LightGBM**: 4 z 5 najważniejszych cech wspólne (Jaccard top-5 0,67, Spearman 0,75). LightGBM jest addytywny (2D PDP: część nieaddytywna = 0), a krzywe PDP i ALE są zgodne (różnica ≤ 7% zakresu efektu).
- **Lokalnie**: dekompozycja logitu φⱼ = βⱼ(WoEⱼ − E[WoEⱼ]) – dokładna, bez przybliżeń. Wykresy kaskadowe (waterfall) dla 4 firm: `outputs/figures/05_waterfall_logit_{1-4}.png` (karta) i `05_waterfall_bb_{1-4}.png` (LightGBM). Kody przyczyn = 3 cechy najbardziej podnoszące ryzyko (`05_kody_przyczyn.csv`).
- **Wierność**: dla karty pełna (wyjaśnieniem jest sam model); SHAP LightGBM dokładny ; LIME słabo wierny (lokalne R² mediana 0,31, błąd PD do 15 pp, dla firmy bezpiecznej PD ujemne).
- **Stabilność**: LIME stabilny dopiero przy 5000 próbek (Jaccard top-5 1,00; przy 1000 – 0,83); ranking SHAP między foldami: karta 0,77, LightGBM 0,60.
- **Wpływ kalibracji**: Platt mnoży wkłady przez a = 0,94, δ zmienia tylko wartość bazową – ranking przyczyn się nie zmienia.
- **Przykład**: firma graniczna (PD ≈ p* przed dostosowaniem) – przyczyny w obu modelach: niska struktura kapitału, wysoka struktura finansowania, zerowy zysk względem zobowiązań. Kontrfakt na karcie (firma 273, BB → BBB): wzrost wskaźnika struktury kapitału z 0,47 do 0,66 (przekroczenie granicy przedziału WoE) → PD 7,4%; w praktyce dokapitalizowanie albo spłata części zobowiązań.

## 5. Ograniczenia i ryzyka
- Mała próba (412 defaultów, po 62 w valid i teście) – szerokie przedziały ufności; przewaga LightGBM nad kartą jest nieistotna.
- Dane przekrojowe – brak oceny stabilności w czasie i nieznany horyzont PD.
- Słabe rozróżnianie mikroprzedsiębiorstw (AUC 0,61) – uproszczone sprawozdania.
- Kalibracja w podgrupach niejednorodna (np. budownictwo: DR 18% przy średnim PD 3,6%).
- Wykluczenie niektórych cech może obniżać jakość, jeśli te wskaźniki były znane w dniu decyzji.
- Wskaźniki finansowe są silnie skorelowane (164 pary |ρ| > 0,9) – wkład w wyjaśnieniach dzieli się między powiązane cechy. Wyjaśnienia i kontrfakty opisują model, nie przyczynowość.

## 6. Plan monitoringu
| Obszar | Miara | Próg alarmowy | Częstotliwość |
|---|---|---|---|
| Stabilność populacji | PSI wyniku (punkty / PD) i CSI 9 cech karty | PSI > 0,10 obserwuj, > 0,25 przegląd modelu | kwartalnie |
| Jakość | AUC, Gini, KS na nowych defaultach | spadek Gini > 0,10 względem walidacji | półrocznie / rocznie |
| Kalibracja | średnie PD vs DR, Brier, test Jeffreysa per klasa | 2 kolejne okresy z czerwonym światłem w ≥ 3 klasach | rocznie |
| Założenie 4% | długoterminowy DR portfela vs 4% | DR trwale poza 3–5% → zmiana δ | rocznie |
| Skala | udział klas, koncentracja (HHI), monotoniczność DR | klasa > 40% portfela, odwrócenie DR | rocznie |
| Karta | rozkład firm w przedziałach WoE, znaki β po douczeniu | zmiana udziału przedziału > 10 pp, zmiana znaku | rocznie |
| Challenger | Gini LightGBM vs karta (sparowany bootstrap) | LightGBM istotnie lepszy w 2 okresach → rozważ zmianę | rocznie |
| Overrides | odsetek i kierunek korekt analityków | korekty w jedną stronę > 10% decyzji | kwartalnie |
