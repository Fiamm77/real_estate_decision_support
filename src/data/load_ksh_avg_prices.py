import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

# -------------------------
# BASE PATH
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

# -------------------------
# PATH
# -------------------------
KSH_PATH = (
    BASE_DIR
    / "data"
    / "external"
    / "ksh"
    / "ksh_avg_prices.csv"
)

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(KSH_PATH, sep=";")

print(f"Loaded rows: {len(df)}")

# -------------------------
# POSTGRES CONNECTION
# -------------------------
engine = create_engine(
    "postgresql+psycopg2://postgres:postgres@localhost:5432/real_estate"
)

# -------------------------
# SAVE TABLE
# -------------------------
df.to_sql(
    "ksh_avg_prices",
    engine,
    if_exists="replace",
    index=False
)

print("KSH table created successfully!")
