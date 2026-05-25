import os

os.environ["PYTHONWARNINGS"] = "ignore"

import warnings

warnings.filterwarnings("ignore")

import logging

logging.getLogger("sklearn").setLevel(
    logging.ERROR
)

import pandas as pd

from sqlalchemy import create_engine

from predict_value import predict_property_row

from calculate_renovation_cost import (
    calculate_renovation_cost
)

# =====================================================
# DATABASE CONNECTION
# =====================================================

engine = create_engine(
    "postgresql+psycopg2://postgres:postgres@localhost:5432/real_estate"
)

# =====================================================
# LOAD SYNTHETIC DATASET
# =====================================================

df = pd.read_sql(
    "SELECT * FROM properties_synthetic",
    engine
)

print(f"Loaded rows: {len(df)}")

# =====================================================
# RUN BATCH PREDICTIONS
# =====================================================

results = []

for idx, row in df.iterrows():

    if idx % 500 == 0:
        print(f"Processed rows: {idx}")

    result = predict_property_row(row)

    results.append(result)

results = pd.DataFrame(results)

# =====================================================
# MERGE RESULTS
# =====================================================

decision_df = pd.concat(
    [df, results],
    axis=1
)

# =====================================================
# CALCULATE RENOVATION COSTS
# =====================================================

renovation_results = []

for idx, row in decision_df.iterrows():

    if idx % 500 == 0:
        print(
            f"Renovation rows processed: {idx}"
        )

    renovation_result = (
        calculate_renovation_cost(
            current_condition=row["condition"],
            target_condition=5,
            building_area_m2=row["building_area_m2"]
        )
    )

    renovation_results.append(
        renovation_result
    )

renovation_df = pd.DataFrame(
    renovation_results
)

# =====================================================
# MERGE RESULTS
# =====================================================

decision_df = pd.concat(
    [decision_df, renovation_df],
    axis=1
)

# =====================================================
# RENOVATED VALUE PREDICTIONS
# =====================================================

renovated_results = []

for idx, row in decision_df.iterrows():

    if idx % 500 == 0:
        print(
            f"Renovated predictions processed: {idx}"
        )

    renovated_prediction = (
        predict_property_value(

            city=row["city"],

            county=row["county"],

            settlement_type=row["settlement_type"],

            property_type=row["property_type"],

            land_area_m2=row["land_area_m2"],

            building_area_m2=row["building_area_m2"],

            condition=5,

            annual_cost=row["annual_cost"],

            renovation_cost=row["total_cost"]
        )
    )

    renovated_results.append(
        renovated_prediction
    )

renovated_df = pd.DataFrame(
    renovated_results
)

# =====================================================
# RENAME RENOVATED COLUMNS
# =====================================================

renovated_df = renovated_df.rename(
    columns={
        "final_value": "renovated_value"
    }
)

# =====================================================
# MERGE RENOVATED RESULTS
# =====================================================

decision_df = pd.concat(
    [decision_df, renovated_df],
    axis=1
)

# =====================================================
# VALUE UPLIFT
# =====================================================

decision_df["value_uplift"] = (
    decision_df["renovated_value"]
    - decision_df["final_value"]
)

# =====================================================
# ROI
# =====================================================

decision_df["roi"] = (
    decision_df["value_uplift"]
    / decision_df["total_cost"]
)

# =====================================================
# SAVE OUTPUT
# =====================================================

decision_df.to_csv(
    "outputs/decision_dataset.csv",
    index=False,
    sep=";",
    encoding="utf-8-sig"
)



print("\nDecision dataset generated successfully!")

print(
    "\nSaved to: outputs/decision_dataset.csv"
)

print(decision_df.head())