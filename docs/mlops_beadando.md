# MLOps beadandó scope

## Projekt célja

A beadandó célja egy MI-alapú ingatlanérték-becslő rendszer bemutatása, amely SQL-ben kezelt ingatlanadatokból aktuális piaci értéket és felújítás utáni piaci értéket prediktál. A predikciókhoz magyarázható kimenet készül SHAP-eredményekből és RAG-szerű, domain tudásbázisra támaszkodó szöveges indoklásból.

Ez a beadandó a diplomamunka projekt egy szűkített részét mutatja be. A döntési rangsorolás, TOPSIS scoring és portfóliópriorizálás a diplomamunka későbbi része, nem tartozik a jelen beadandó scope-jába.

---

## Scope

### Beletartozik

* SQL-alapú adatkezelés PostgreSQL és SQLAlchemy használatával.
* Aktuális ingatlanérték predikciója.
* Felújítás utáni ingatlanérték predikciója.
* SHAP-alapú lokális magyarázatok.
* RAG-szerű szöveges magyarázat domain szabályok és modellkimenetek alapján.
* Streamlit alapú megjelenítés.
* MLOps dokumentáció: verziózás, monitoring, KPI-k, retraining.

### Nem tartozik bele

* TOPSIS döntési scoring.
* Gazdasági és társadalmi döntési rangsorolás.
* Stratégiai portfóliókezelés.

---

## Architektúra

1. Az ingatlan- és KSH-adatok CSV-fájlokból PostgreSQL adatbázisba kerülnek.
2. A predikciós modell a strukturált ingatlanjellemzőkből külön telek- és épületkomponenst becsül, amelyekből strukturális score képződik.
3. A KSH települési vagy területi fajlagos ár külön piaci benchmarkként jelenik meg. A modellkimenet és a KSH benchmark aránya interpretációs mutató, nem tanítási target.
4. A felújítás utáni szcenárióban a rendszer célállapotra újraprediktálja az ingatlan strukturális score-ját, majd újrabecsüli az ingatlan piaci értékét.
5. A magyarázati réteg a predikciós eredményeket, SHAP-feature hatásokat és domain szabályokat kapcsol össze.
6. A Streamlit felület a predikciókat, a felújítási szcenáriót és a magyarázatot jeleníti meg.

---

## Training set

A tanító adatbázis SQL-ben tárolt szintetikus ingatlanadatokra épül. A `properties_synthetic` tábla az alap ingatlanadatokat tartalmazza, a `ksh_avg_prices` tábla pedig a települési és területi KSH-árakat. A training folyamat ezekből képez tanító nézetet.

A tanítóadatok tartalmazzák többek között:

* ingatlan azonosító,
* ingatlan típusa,
* település, vármegye, településtípus,
* telekméret,
* épület hasznos alapterület,
* állapot,
* becsült épületérték,
* telekérték,
* korábbi felújítási költség.

A korábbi felújítási költség nem bemeneti feature, hanem a strukturális score képzésének egyik alapadata.

Két célváltozó képződik dinamikusan, mindkettő csoporton belüli percentile rank formájában:

```text
land_structural_score =
    percentile_rank(land_value / land_area_m2
                    within county + settlement_type + property_type)

building_structural_score =
    percentile_rank((building_value + renovation_cost) / building_area_m2
                    within county + settlement_type + property_type)
```

A training sorokból kiszűrésre kerülnek a hiányos vagy irreális rekordok, például a nulla vagy hiányzó alapterület, a nulla épületérték-proxy vagy a nulla KSH benchmark.

A modell jelenlegi feature halmaza:

* property_type
* settlement_type
* county
* land_area_m2
* building_area_m2
* condition

Nem modell feature a `city`, mert a települési piaci szintet a KSH benchmark kezeli.

Nem feature az `annual_cost`, valamint a `building_value`, `land_value`, `renovation_cost`, `activation_year` és KSH benchmark mezők sem, mert ezek target, benchmark vagy a jelenlegi modellből kizárt változók.

---

## Modell

A predikciós modell egy scikit-learn pipeline:

* medián imputálás numerikus változókra,
* RandomForestRegressor regressziós modell,
* lokálisan generált modellfájl: `models/valuation_model.pkl`.

A modellfájl a training futtatásakor jön létre. A GitHub repository nem
verziózza a bináris modell artifactot, mert az újragenerálható a
`src/valuation/train_model.py` futtatásával.

A modell két komponenst tanul a tanítóadatokból: egy telekhez és egy épülethez kapcsolódó strukturális score-t.

Lakóházaknál a végső score a telek- és épületscore súlyozott kombinációja. Az auditok alapján a telekkomponens hatása jelenleg korlátozott, ezért a score-t elsősorban az épületkomponens befolyásolja.

Lakások esetén az épületscore dominál.

A KSH fajlagos áradataiból számolt `ksh_baseline_value_huf` piaci benchmarkként szolgál, amelyet a prediktált strukturális score korrigál.

```text
adjustment_factor = 0.80 + 0.40 * predicted_structural_score

predicted_market_value =
    ksh_baseline_value_huf * adjustment_factor

benchmark_delta =
    predicted_market_value - ksh_baseline_value_huf
```

---

## MLOps verziózás

### Forráskód

A forráskód Git repositoryban van tárolva.

A beadandó szempontjából releváns modulok:

* `src/valuation` – predikció és felújítás utáni értékbecslés,
* `src/data` – SQL adatbetöltés,
* `src/rag` – magyarázati réteg,
* `src/ui` – Streamlit felület.

### Adatok

Az adatverziók fájlszintű verziózása a `data/` könyvtárban történik.

Nagyobb éles rendszerben DVC vagy objektumtár használata lenne indokolt, ahol minden tanítóadat-verzió hash-sel és metaadatokkal azonosítható.

A szintetikus tanítóadatok minőségét külön statisztikai audit ellenőrizte. Az audit során az eredeti és a szintetikus adatok eloszlásai, percentilisei és főbb kapcsolatai kerültek összehasonlításra.

### Modell

A modell artifact lokálisan a `models/` könyvtárba kerül.

A beadandó GitHub repositoryban a bináris modellfájl nincs verziózva
(`models/*.pkl` szerepel a `.gitignore` fájlban). A modell reprodukálhatóságát
a verziózott training script, a verziózott bemeneti adatok és a dokumentált
futtatási lépések biztosítják.

Éles működésben minden modellhez tárolandó:

* tanítóadat verziója,
* training script commit hash,
* metrikák,
* tanítás dátuma,
* feature lista,
* Python- és csomagverziók.

### Promptok és RAG tudásbázis

A jelenlegi beadandóban a magyarázati réteg determinisztikus, RAG-szerű sablonokkal dolgozik.

A tudásbázis a `docs/rag_knowledge/` könyvtárban verziózott markdown-fájlokból áll.

Ha később LLM is bekerül, minden promptverzió külön fájlban tárolandó, például:

* `prompts/explanation_v1.md`
* `prompts/explanation_v2.md`

---

## Monitoring

### Modell KPI-k

Javasolt offline metrikák:

* MAE (Mean Absolute Error),
* RMSE (Root Mean Squared Error),
* MAPE (Mean Absolute Percentage Error),
* R²,
* predikciós hibák vármegye és ingatlantípus szerint.

### Adatminőségi KPI-k

Figyelendő:

* hiányzó értékek aránya,
* negatív vagy nulla alapterület,
* irreális éves költség,
* ismeretlen település vagy vármegye,
* KSH-áradat hiánya,
* állapotskálán kívüli érték.

### Drift KPI-k

Figyelendő eloszlásváltozások:

* alapterület eloszlás,
* állapot eloszlás,
* éves költség eloszlás,
* KSH fajlagos árak eloszlása,
* prediktált strukturális score eloszlása,
* KSH benchmark eloszlása,
* market position ratio eloszlása,
* prediktált piaci értékek eloszlása.

### Monitoring gyakoriság

Beadandó/demo környezetben:

* minden batch futás után alap adatminőségi riport,
* minden új tanítás után modellmetrika riport.

Éles környezetben:

* napi adatminőségi ellenőrzés,
* heti drift riport,
* havi modellteljesítmény-review,
* újratanítás trigger jelentősebb drift vagy romló pontosság esetén.

---

## Adatminőségi probléma kezelése

Ha adatminőségi probléma merül fel:

1. A rekord validációs figyelmeztetést kap.
2. Kritikus hiba esetén a rekord kimarad a predikcióból.
3. Nem kritikus hiba esetén imputálás vagy fallback logika fut.
4. A hiba bekerül a monitoring riportba.
5. Ismétlődő probléma esetén adatforrás oldali javítás szükséges.

---

## Retraining

Újratanítás akkor indokolt, ha:

* új ingatlanadat-verzió érkezik,
* a KSH piaci árak jelentősen változnak,
* drift detektálható a bemeneti eloszlásokban,
* a validációs MAE vagy MAPE romlik,
* új feature kerül be a modellbe.

Javasolt retraining folyamat:

1. adatverzió rögzítése,
2. training script futtatása,
3. metrikák exportálása,
4. modell artifact mentése,
5. előző modellhez viszonyított összehasonlítás,
6. jóváhagyás után modellcsere.

---

## LLM vagy magyarázati modell cseréje

Ha később LLM-alapú magyarázat kerül be:

* a promptokat verziózni kell,
* a RAG dokumentumokat verziózni kell,
* regressziós tesztkérdéseket kell fenntartani,
* a válaszokat szakmai és stilisztikai szempontból ellenőrizni kell,
* modellcsere csak összehasonlító értékelés után történhet.

---

## Futtatás

### Adatbázis indítása

```powershell
docker compose -f docker/docker-compose.yml up -d
```

### Adatok betöltése

```powershell
python src/data/load_synthetic.py
python src/data/load_ksh_avg_prices.py
```

### Modell tanítása

```powershell
python src/valuation/train_model.py
```

### Streamlit felület

```powershell
streamlit run src/ui/streamlit_app.py
```
