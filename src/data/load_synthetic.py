import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

# --- BASE PATH ---
BASE_DIR = Path(__file__).resolve().parents[2]

# --- PATHS ---
DATA_PATH = BASE_DIR / "data" / "synthetic" / "real_estate_synthetic.csv"

# --- LOAD DATA ---
df = pd.read_csv(DATA_PATH, sep=";")

print(f"Loaded rows: {len(df)}")

# --- POSTGRES CONNECTION ---
engine = create_engine(
    "postgresql+psycopg2://postgres:postgres@localhost:5432/real_estate"
)

# --- SAVE TABLE ---
df.to_sql(
    "properties_synthetic",
    engine,
    if_exists="replace",
    index=False
)

print("Table created successfully!")