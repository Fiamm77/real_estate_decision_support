"""Shared loading helpers for valuation modules."""

import joblib
import pandas as pd
from sqlalchemy import create_engine

try:
    from .config import DB_URL
except ImportError:
    from config import DB_URL


def get_engine():
    return create_engine(DB_URL)


def read_table(table_name):
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)


def load_model(model_path):
    return joblib.load(model_path)


engine = get_engine()
