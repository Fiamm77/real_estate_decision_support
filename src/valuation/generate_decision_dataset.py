"""Generate decision dataset using refactored valuation and renovation modules.

The main logic is preserved but wrapped in `generate_decision_dataset()` for reuse.
"""

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

from predict_value import predict_property_frame
from calculate_renovation_cost import calculate_renovation_cost

MIN_SCORE_UPLIFT_PER_CONDITION_STEP = 0.05
ADJUSTMENT_MIN = 0.80
ADJUSTMENT_RANGE = 0.40


def generate_decision_dataset(engine=None, output_path="outputs/decision_dataset.csv"):
    if engine is None:
        engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5432/real_estate")

    df = pd.read_sql("SELECT * FROM properties_synthetic", engine)
    print(f"Loaded rows: {len(df)}")

    # batch predictions
    results = predict_property_frame(df)
    decision_df = pd.concat([df, results], axis=1)

    # renovation costs
    renovation_results = [
        calculate_renovation_cost(current_condition=row["condition"], target_condition=5, building_area_m2=row["building_area_m2"]) for _, row in decision_df.iterrows()
    ]

    renovation_df = pd.DataFrame(renovation_results)
    decision_df = pd.concat([decision_df, renovation_df], axis=1)

    # renovated predictions
    renovated_input_df = decision_df.copy()
    renovated_input_df["condition"] = 5
    renovated_df = predict_property_frame(renovated_input_df).rename(
        columns={
            "predicted_market_value": "renovated_market_value",
            "ksh_price_m2": "renovated_ksh_price_m2",
            "ksh_baseline_value_huf": "renovated_ksh_baseline_value_huf",
            "predicted_land_structural_score": "renovated_land_structural_score",
            "predicted_building_structural_score": "renovated_building_structural_score",
            "predicted_structural_score": "renovated_structural_score",
            "adjustment_factor": "renovated_adjustment_factor",
            "benchmark_delta": "renovated_benchmark_delta",
            "ksh_source_level": "renovated_ksh_source_level",
        }
    )
    decision_df = pd.concat([decision_df, renovated_df], axis=1)

    renovation_mask = (
        (decision_df["total_cost"] > 0)
        & (decision_df["renovation_possible"])
        & (decision_df["condition"] < 5)
    )
    if renovation_mask.any():
        condition_gap = (5 - decision_df["condition"]).clip(lower=0)
        minimum_score = (
            decision_df["predicted_structural_score"]
            + condition_gap * MIN_SCORE_UPLIFT_PER_CONDITION_STEP
        ).clip(upper=1)
        decision_df.loc[renovation_mask, "renovated_structural_score"] = decision_df.loc[
            renovation_mask,
            "renovated_structural_score",
        ].combine(minimum_score.loc[renovation_mask], max)
        decision_df.loc[renovation_mask, "renovated_adjustment_factor"] = (
            ADJUSTMENT_MIN
            + ADJUSTMENT_RANGE
            * decision_df.loc[renovation_mask, "renovated_structural_score"]
        )
        decision_df.loc[renovation_mask, "renovated_market_value"] = (
            decision_df.loc[renovation_mask, "renovated_ksh_baseline_value_huf"]
            * decision_df.loc[renovation_mask, "renovated_adjustment_factor"]
        ).round(0)
        decision_df.loc[renovation_mask, "renovated_benchmark_delta"] = (
            decision_df.loc[renovation_mask, "renovated_market_value"]
            - decision_df.loc[renovation_mask, "renovated_ksh_baseline_value_huf"]
        ).round(0)

    decision_df["renovated_market_value"] = decision_df[
        ["renovated_market_value", "predicted_market_value"]
    ].max(axis=1)
    decision_df["value_uplift"] = (
        decision_df["renovated_market_value"] - decision_df["predicted_market_value"]
    )
    decision_df["roi"] = 0.0
    cost_mask = decision_df["total_cost"] > 0
    decision_df.loc[cost_mask, "roi"] = (
        decision_df.loc[cost_mask, "value_uplift"]
        / decision_df.loc[cost_mask, "total_cost"]
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    decision_df.to_csv(output_path, index=False, sep=";", encoding="utf-8-sig")

    print("\nDecision dataset generated successfully!")
    print(f"\nSaved to: {output_path}")
    print(decision_df.head())


if __name__ == "__main__":
    generate_decision_dataset()
