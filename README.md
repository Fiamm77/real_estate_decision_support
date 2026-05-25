# AI-Assisted Real Estate Decision Support System

Hybrid real-estate valuation and decision-support platform.

## Current Scope

Implemented:
- KSH market-anchor valuation
- ML structural valuation model
- renovation cost calculation
- renovated value prediction
- ROI calculation
- batch decision dataset generation

Planned:
- TOPSIS decision support
- Streamlit dashboard
- FastAPI API
- strategy weighting
- portfolio simulation

---

## Core Valuation Logic

Final Value =
KSH Baseline × ML Structural Modifier

The ML model predicts structural relative value,
while KSH data provides regional market anchoring.

---

## Architecture Concept

This project represents a domain-focused implementation
of a broader snapshot/story/context-based
organizational intelligence architecture.

Core concepts:
- snapshot-based state modeling
- state transitions
- strategy-aware interpretation
- organizational memory
- decision-support pipelines

---

## Project Structure

src/
├── valuation/
├── decision/
├── api/
├── ui/
├── utils/

models/
outputs/
docs/

---

## Technology Stack

- Python
- PostgreSQL
- scikit-learn
- SHAP
- SQLAlchemy
- FastAPI
- Streamlit
- Docker

---

## Current Pipeline

1. Load synthetic property dataset
2. Predict current property value
3. Calculate renovation cost
4. Predict renovated value
5. Calculate value uplift and ROI
6. Export decision dataset

---

## Long-Term Vision

The long-term vision is a broader
organizational memory and strategic intelligence platform
built on snapshot/story/context modeling principles.