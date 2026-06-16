# AI-Assisted Real Estate Valuation System

This repository is the thesis project workspace for a real-estate decision
support system. Some earlier assignments support the broader thesis scope.

The current MLOps assignment uses only a focused subset of the repository:

- SQL-based property data handling
- current market value prediction
- renovated market value prediction
- renovation cost estimation
- SHAP and RAG-style explanation
- Streamlit presentation layer
- MLOps documentation

The portfolio decision/scoring layer is intentionally out of scope for the
current assignment and remains relevant for the thesis.

## MLOps Assignment Scope

Included modules:

- `src/data`: CSV to PostgreSQL loaders and database inspection helpers
- `src/valuation`: valuation model, renovation calculation and prediction pipeline
- `src/rag`: lightweight retrieval-augmented explanation layer
- `src/ui`: Streamlit dashboard
- `docs/mlops_beadando.md`: assignment documentation
- `docs/rag_knowledge`: versioned explanation knowledge base

Out of scope for this assignment:

- `src/decision`
- TOPSIS ranking
- economic/social scoring
- strategic portfolio prioritization
- `notebooks/decision_model*`

## Core Valuation Logic

```text
ML model output = predicted structural asset value
KSH baseline = KSH price/m2 * building area
Market position ratio = predicted structural asset value / KSH baseline
```

The model estimates a structural asset-value proxy from property attributes.
KSH price data remains a separate market benchmark. The dashboard compares the
modelled asset value with the KSH benchmark instead of training on a KSH-divided
modifier.

The renovated scenario first estimates renovation cost and target condition,
then runs the asset-value model again for the post-renovation state. A business
minimum rule prevents the renovated scenario from producing a lower asset value
than the current scenario.

## Technology Stack

- Python
- PostgreSQL
- SQLAlchemy
- scikit-learn
- SHAP
- Streamlit
- Docker

## Project Structure

```text
src/
  data/
  valuation/
  rag/
  ui/
  decision/       # thesis scope, not current assignment scope

data/
docs/
docker/
notebooks/
outputs/
```

## Run The Assignment Demo

Start PostgreSQL:

```powershell
docker compose -f docker/docker-compose.yml up -d
```

Load source data:

```powershell
python src/data/load_synthetic.py
python src/data/load_ksh_avg_prices
python "src/data/load_renovation data.py"
python src/data/load_ksh_social.py
```

`properties_synthetic` stores the base synthetic property data. The model learns
two DF-derived relative structural targets: a land percentile score and a
building percentile score within comparable county, settlement type and property
type groups. KSH values are derived from `ksh_avg_prices` as benchmark outputs;
they are not model features.

Train the valuation model:

```powershell
python src/valuation/train_model.py
```

Generate valuation outputs:

```powershell
python src/run_full_pipeline.py
```

Run the Streamlit dashboard:

```powershell
streamlit run src/ui/streamlit_app.py
```

The dashboard can also demonstrate from existing CSV exports in `outputs/`.

## Documentation

The MLOps assignment documentation is in:

```text
docs/mlops_beadando.md
```

It covers:

- project description and architecture
- training set description
- code, model, data and prompt versioning
- monitoring and logging
- model KPIs
- data quality checks
- drift detection
- retraining policy
- explanation model replacement policy
