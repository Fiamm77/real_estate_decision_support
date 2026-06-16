# MLOps beadando scope

## Projekt celja

A beadando celja egy MI-alapu ingatlanertek-becslo rendszer bemutatasa, amely
SQL-ben kezelt ingatlanadatokbol aktualis piaci erteket es felujitas utani
piaci erteket prediktal. A predikciokhoz magyarazhato kimenet keszul SHAP
eredmenyekbol es RAG-szeru, domain tudasbazisra tamaszkodo szoveges
indoklasbol.

Ez a beadando a diplomamunka repojanak egy szukitett resze. A dontesi
rangsorolas, TOPSIS scoring es portfoliopriorizalas a diplomamunka kesobbi
resze, nem tartozik a jelen beadando scope-jaba.

## Scope

Beletartozik:

- SQL-alapu adatkezeles PostgreSQL es SQLAlchemy hasznalataval.
- Aktualis ingatlanertek predikcioja.
- Felujitas utani ingatlanertek predikcioja.
- Felujitasi koltseg becslese.
- SHAP-alapu lokalis magyarazatok.
- RAG-szeru szoveges magyarazat domain szabalyok es modellkimenetek alapjan.
- Streamlit alapu megjelenites.
- MLOps dokumentacio: verziozas, monitoring, KPI-k, retraining.

Nem tartozik bele:

- TOPSIS dontesi scoring.
- Gazdasagi es tarsadalmi dontesi rangsorolas.
- Strategiai portfoliokezeles.

## Architektura

1. Az ingatlan-, KSH- es felujitasi adatok CSV fajlokbol PostgreSQL tablaba
   kerulnek.
2. A predikcios modell a strukturalt ingatlanjellemzokbol becsul egy
   asset ertekproxyt.
3. A KSH telepulesi vagy teruleti fajlagos ar kulon piaci benchmarkkent
   jelenik meg. A modellkimenet es a KSH benchmark aranya interpretacios
   mutato, nem tanitasi target.
4. A felujitas utani szcenarioban a rendszer celallapotra szamolja a
   felujitasi koltseget, majd ujra becsli az ingatlan erteket.
5. A magyarazati reteg a predikcios eredmenyeket, SHAP feature hatasokat es
   domain szabalyokat kapcsol ossze.
6. A Streamlit felulet a predikciokat, felujitasi szcenariot es magyarazatot
   jeleniti meg.

## Training set

A tanito adatbazis SQL-ben tarolt szintetikus ingatlanadatokra epul. A
`properties_synthetic` tabla az alap ingatlanadatokat tartalmazza, a
`ksh_avg_prices` tabla pedig a telepulesi es teruleti KSH arakat. A training
folyamat ezekbol kepez tanito nezetet.

- ingatlan azonosito,
- ingatlan tipusa,
- telepules, varmegye, telepulestipus,
- telekmeret,
- epulethasznos alapterulet,
- allapot,
- becsult epuletertek,
- telekertek,
- korabbi felujitasi koltseg.

Ket celvaltozo kepzodik dinamikusan, mindketto csoporton beluli percentile
rank formajaban:

```text
land_structural_score =
    percentile_rank(land_value / land_area_m2 within county + settlement_type + property_type)

building_structural_score =
    percentile_rank((building_value + renovation_cost) / building_area_m2
                    within county + settlement_type + property_type)
```

A training sorokbol kiszuresre kerulnek a hianyos vagy irrealis rekordok,
peldaul a nulla vagy hianyzo alapterulet, a nulla epuletertek proxy, vagy a
nulla KSH benchmark.

A modell jelenlegi feature halmaza:

- `property_type`
- `settlement_type`
- `county`
- `land_area_m2`
- `building_area_m2`
- `condition`

Nem modell feature a `city`, mert a telepulesi piaci szintet a KSH benchmark
kezeli. Nem feature az `annual_cost`, valamint a `building_value`,
`land_value`, `renovation_cost`, `activation_year` es KSH baseline mezok sem,
mert ezek target, benchmark vagy az aktualis modellbol kizart mezok.

## Modell

A predikcios modell egy scikit-learn pipeline:

- median imputalas numerikus valtozokra,
- RandomForestRegressor regresszios modell,
- mentett modell fajl: `models/valuation_model.pkl`.

A modell ket komponenst tanul a DF-bol: telekhez es epülethez kapcsolodo
strukturalt percentile score-t. Lakohazaknal a vegso score a telek es epület
score sulyozott atlaga, lakasoknal az epület score dominans. A KSH
fajlagos aradatokbol szamolt `ksh_baseline_value_huf` piaci benchmark, amelyet
a prediktalt strukturalt score korrigal.

```text
adjustment_factor = 0.80 + 0.40 * predicted_structural_score
predicted_market_value = ksh_baseline_value_huf * adjustment_factor
benchmark_delta = predicted_market_value - ksh_baseline_value_huf
```

## MLOps verziozas

### Forraskod

A forraskod Git repoban van tarolva. A beadando szempontjabol kulon kezelendo
modulok:

- `src/valuation`: predikcio es felujitas utani ertekbecsles,
- `src/data`: SQL adatbetoltes,
- `src/rag`: magyarazati reteg,
- `src/ui`: Streamlit felulet.

### Adatok

Az adatverziok fajlszintu verziozasa a `data/` konyvtarban tortenik. Nagyobb
eles rendszerben DVC vagy objektumtar hasznalata lenne indokolt, ahol minden
tanito adatverzio hash-sel es metadata rekorddal azonosithato.

### Modell

A modell artifact a `models/` konyvtarba kerul. Eles mukodesben minden modellhez
tarolando:

- tanito adat verzioja,
- training script commit hash,
- metrikak,
- tanitas datuma,
- feature lista,
- Python es csomagverziok.

### Promptok es RAG tudasbazis

A jelenlegi beadandoban a magyarazati reteg determinisztikus RAG-szeru
sablonokkal dolgozik. A tudasbazis a `docs/rag_knowledge/` konyvtarban
verziozott markdown fajlokbol all.

Ha kesobb LLM is bekerul, minden prompt verziot kulon fajlban kell tarolni,
peldaul:

- `prompts/explanation_v1.md`
- `prompts/explanation_v2.md`

## Monitoring

### Modell KPI-k

Javasolt offline metrikak:

- MAE: atlagos abszolut hiba forintban,
- RMSE: nagy hibakra erzekeny regresszios mutato,
- MAPE: relativ hiba,
- R2: magyarazoero,
- predikcios hibak varmegye es ingatlantipus szerint.

### Adatminosegi KPI-k

Figyelendo:

- hianyzo ertekek aranya,
- negativ vagy nulla alapterulet,
- irrealis eves koltseg,
- ismeretlen telepules vagy varmegye,
- KSH aradat hianya,
- allapot skalan kivuli ertek.

### Drift KPI-k

Figyelendo eloszlasvaltozasok:

- alapterulet eloszlas,
- allapot eloszlas,
- eves koltseg eloszlas,
- KSH fajlagos arak eloszlasa,
- predicted asset ertek eloszlasa,
- KSH benchmark eloszlasa,
- market position ratio eloszlasa,
- prediktalt ertekek eloszlasa.

### Monitoring gyakorisag

Beadando/demo kornyezetben:

- minden batch futas utan alap adatminosegi riport,
- minden uj tanitas utan modellmetrika riport.

Eles kornyezetben:

- napi adatminosegi ellenorzes,
- heti drift riport,
- havi modell teljesitmeny review,
- ujratanitas trigger jelentosebb drift vagy romlo pontossag eseten.

## Adatminosegi problema kezelese

Ha adatminosegi problema merul fel:

1. A rekord validacios figyelmeztetest kap.
2. Kritikus hiba eseten a rekord kimarad a predikciobol.
3. Nem kritikus hiba eseten imputalas vagy fallback logika fut.
4. A hiba bekerul a monitoring riportba.
5. Ismetlodo problema eseten adatforras oldali javitas szukseges.

## Retraining

Ujratanitas akkor indokolt, ha:

- uj ingatlanadat-verzio erkezik,
- a KSH piaci arak jelentosen valtoznak,
- drift detektalhato a bemeneti eloszlasokban,
- a validacios MAE vagy MAPE romlik,
- uj feature kerul be a modellbe.

Javasolt retraining folyamat:

1. adatverzio rogzites,
2. training script futtatasa,
3. metrikak exportalasa,
4. modell artifact mentese,
5. elozo modellhez viszonyitott osszehasonlitas,
6. jovahagyas utan modellcsere.

## LLM vagy magyarazati modell csereje

Ha kesobb LLM-alapu magyarazat kerul be:

- a promptokat verziozni kell,
- a RAG dokumentumokat verziozni kell,
- regresszios tesztkerdeseket kell fenntartani,
- a valaszokat szakmai es stilusbeli szempontbol ellenorizni kell,
- modellcsere csak osszehasonlito ertekeles utan tortenhet.

## Futtatas

Adatbazis inditasa:

```powershell
docker compose -f docker/docker-compose.yml up -d
```

Adatok betoltese:

```powershell
python src/data/load_synthetic.py
python src/data/load_ksh_avg_prices
python "src/data/load_renovation data.py"
python src/data/load_ksh_social.py
```

Modell tanitasa:

```powershell
python src/valuation/train_model.py
```

Streamlit felulet:

```powershell
streamlit run src/ui/streamlit_app.py
```
