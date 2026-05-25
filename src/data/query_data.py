import pandas as pd
from sqlalchemy import create_engine, inspect

# -------------------------
# POSTGRES CONNECTION
# -------------------------
engine = create_engine(
    "postgresql+psycopg2://postgres:postgres@localhost:5432/real_estate"
)

# -------------------------
# INSPECT DATABASE
# -------------------------
inspector = inspect(engine)

tables = inspector.get_table_names()

print("\n=== TABLES IN DATABASE ===\n")

for table in tables:

    # Row count query
    query = f"SELECT COUNT(*) AS row_count FROM {table}"

    row_count = pd.read_sql(query, engine)

    count = row_count.iloc[0]["row_count"]

    print(f"{table}: {count} rows")