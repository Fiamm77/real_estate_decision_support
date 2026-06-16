import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

# --- BASE PATH ---
BASE_DIR = Path(__file__).resolve().parents[2]

# --- PATHS ---
SYNTHETIC_DIR = BASE_DIR / "data" / "synthetic"
DATA_PATH = SYNTHETIC_DIR / "real_estate_synthetic.csv"

BASE_COLUMNS = [
    "property_id",
    "activation_year",
    "building_value",
    "land_value",
    "renovation_cost",
    "property_type",
    "land_area_m2",
    "building_area_m2",
    "condition",
    "settlement_type",
    "city",
    "county",
    "annual_cost",
]


# --- LOAD DATA ---
data_path = DATA_PATH
df = pd.read_csv(data_path, sep=";")

missing_columns = [column for column in BASE_COLUMNS if column not in df.columns]
if missing_columns:
    raise ValueError(
        "Synthetic input is missing required columns: "
        + ", ".join(missing_columns)
    )

df = df[BASE_COLUMNS].copy()

print(f"Loaded file: {data_path}")
print(f"Loaded rows: {len(df)}")
print(f"Condition values: {sorted(df['condition'].dropna().unique())}")

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
