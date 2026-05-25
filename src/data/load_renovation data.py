import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

# --- BASE PATH ---
BASE_DIR = Path(__file__).resolve().parents[2]

# --- PATHS ---
RENOVATION_DIR = BASE_DIR / "data" / "external" / "renovation"

WORKS_PATH = RENOVATION_DIR / "works.csv"
PACKAGES_PATH = RENOVATION_DIR / "packages.csv"
PACKAGE_WORKS_PATH = RENOVATION_DIR / "package_works.csv"
CONDITIONS_PATH = RENOVATION_DIR / "conditions.csv"

# --- LOAD CSV FILES ---
works_df = pd.read_csv(WORKS_PATH, sep=";")
packages_df = pd.read_csv(PACKAGES_PATH, sep=";")
package_works_df = pd.read_csv(PACKAGE_WORKS_PATH, sep=";")
conditions_df = pd.read_csv(CONDITIONS_PATH, sep=";")

print(f"Loaded works: {len(works_df)}")
print(f"Loaded packages: {len(packages_df)}")
print(f"Loaded package-work relations: {len(package_works_df)}")
print(f"Loaded conditions: {len(conditions_df)}")

# --- POSTGRES CONNECTION ---
engine = create_engine(
    "postgresql+psycopg2://postgres:postgres@localhost:5432/real_estate"
)

# --- SAVE TABLES ---
works_df.to_sql(
    "works",
    engine,
    if_exists="replace",
    index=False
)

packages_df.to_sql(
    "packages",
    engine,
    if_exists="replace",
    index=False
)

package_works_df.to_sql(
    "package_works",
    engine,
    if_exists="replace",
    index=False
)

conditions_df.to_sql(
    "conditions",
    engine,
    if_exists="replace",
    index=False
)

print("Renovation tables created successfully!")