"""Reusable decision-support pipeline functions."""

import pandas as pd

MIN_SCORE_UPLIFT_PER_CONDITION_STEP = 0.05
ADJUSTMENT_MIN = 0.80
ADJUSTMENT_RANGE = 0.40
TARGET_CONDITION_FOR_UPLIFT = 5

try:
    from .calculate_renovation_cost import calculate_renovation_cost
    from .config import (
        OUTPUT_CSV_ENCODING,
        OUTPUT_CSV_SEPARATOR,
        OUTPUT_DECISION_DATASET_PATH,
        PROPERTIES_TABLE,
        TARGET_RENOVATION_CONDITION,
    )
    from .core import read_table
    from .predict_value import predict_property_frame, predict_property_row, predict_property_value
except ImportError:
    from calculate_renovation_cost import calculate_renovation_cost
    from config import (
        OUTPUT_CSV_ENCODING,
        OUTPUT_CSV_SEPARATOR,
        OUTPUT_DECISION_DATASET_PATH,
        PROPERTIES_TABLE,
        TARGET_RENOVATION_CONDITION,
    )
    from core import read_table
    from predict_value import predict_property_frame, predict_property_row, predict_property_value


def load_properties():
    return read_table(PROPERTIES_TABLE)


def load_decision_dataset(output_path=OUTPUT_DECISION_DATASET_PATH):
    return pd.read_csv(
        output_path,
        sep=OUTPUT_CSV_SEPARATOR,
        encoding=OUTPUT_CSV_ENCODING,
    )


def _drop_existing_decision_outputs(decision_df):
    generated_columns = [
        "total_cost",
        "package_ids",
        "works",
        "renovation_possible",
        "renovated_ksh_price_m2",
        "renovated_ksh_baseline_value_huf",
        "renovated_land_structural_score",
        "renovated_building_structural_score",
        "renovated_structural_score",
        "renovated_adjustment_factor",
        "renovated_benchmark_delta",
        "renovated_market_value",
        "renovated_ksh_source_level",
        "ksh_price_m2.1",
        "ksh_baseline_value_huf.1",
        "predicted_land_structural_score.1",
        "predicted_building_structural_score.1",
        "predicted_structural_score.1",
        "adjustment_factor.1",
        "benchmark_delta.1",
        "ksh_source_level.1",
        "value_uplift",
        "roi",
    ]
    return decision_df.drop(
        columns=[column for column in generated_columns if column in decision_df],
    )


def _log_progress(index, every, message):
    if index % every == 0:
        print(f"{message}: {index}")


def build_current_valuation_df(properties_df, progress_every=500):
    return pd.concat([properties_df, predict_property_frame(properties_df)], axis=1)


def build_renovation_cost_df(
    decision_df,
    target_condition=TARGET_RENOVATION_CONDITION,
    progress_every=500,
):
    renovation_results = []

    for idx, row in decision_df.iterrows():
        _log_progress(idx, progress_every, "Renovation rows processed")
        renovation_results.append(
            calculate_renovation_cost(
                current_condition=row["condition"],
                target_condition=target_condition,
                building_area_m2=row["building_area_m2"],
            )
        )

    return pd.DataFrame(renovation_results)


def add_renovation_costs(
    decision_df,
    target_condition=TARGET_RENOVATION_CONDITION,
    progress_every=500,
):
    renovation_df = build_renovation_cost_df(
        decision_df=decision_df,
        target_condition=target_condition,
        progress_every=progress_every,
    )
    return pd.concat([decision_df, renovation_df], axis=1)


def build_renovated_valuation_df(
    decision_df,
    target_condition=TARGET_RENOVATION_CONDITION,
    progress_every=500,
):
    renovated_input_df = decision_df.copy()
    renovated_input_df["condition"] = target_condition

    return predict_property_frame(renovated_input_df).rename(
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


def add_renovated_values(
    decision_df,
    target_condition=TARGET_RENOVATION_CONDITION,
    progress_every=500,
):
    renovated_df = build_renovated_valuation_df(
        decision_df=decision_df,
        target_condition=target_condition,
        progress_every=progress_every,
    )
    return pd.concat([decision_df, renovated_df], axis=1)


def add_roi_columns(decision_df):
    decision_df = decision_df.copy()
    decision_df = apply_minimum_renovation_uplift(decision_df)
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
    return decision_df


def add_value_uplift_column(decision_df):
    decision_df = decision_df.copy()
    decision_df = apply_minimum_renovation_uplift(decision_df)
    decision_df["renovated_market_value"] = decision_df[
        ["renovated_market_value", "predicted_market_value"]
    ].max(axis=1)
    decision_df["value_uplift"] = (
        decision_df["renovated_market_value"] - decision_df["predicted_market_value"]
    )
    return decision_df


def apply_minimum_renovation_uplift(
    decision_df,
    target_condition=TARGET_CONDITION_FOR_UPLIFT,
):
    decision_df = decision_df.copy()
    renovation_mask = (
        (decision_df["total_cost"] > 0)
        & (decision_df["renovation_possible"])
        & (decision_df["condition"] < target_condition)
    )
    if not renovation_mask.any():
        return decision_df

    condition_gap = (target_condition - decision_df["condition"]).clip(lower=0)
    minimum_score = (
        decision_df["predicted_structural_score"]
        + condition_gap * MIN_SCORE_UPLIFT_PER_CONDITION_STEP
    ).clip(upper=1)

    decision_df.loc[renovation_mask, "renovated_structural_score"] = (
        decision_df.loc[renovation_mask, "renovated_structural_score"]
        .combine(minimum_score.loc[renovation_mask], max)
    )
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

    return decision_df


def generate_decision_dataset(
    target_condition=TARGET_RENOVATION_CONDITION,
    reuse_current_predictions=False,
    include_roi=False,
    progress_every=500,
):
    if reuse_current_predictions:
        decision_df = load_decision_dataset()
        decision_df = _drop_existing_decision_outputs(decision_df)
        print(f"Loaded existing decision rows: {len(decision_df)}")
    else:
        properties_df = load_properties()
        print(f"Loaded rows: {len(properties_df)}")

        decision_df = build_current_valuation_df(
            properties_df=properties_df,
            progress_every=progress_every,
        )

    decision_df = add_renovation_costs(
        decision_df=decision_df,
        target_condition=target_condition,
        progress_every=progress_every,
    )
    decision_df = add_renovated_values(
        decision_df=decision_df,
        target_condition=target_condition,
        progress_every=progress_every,
    )

    if include_roi:
        return add_roi_columns(decision_df)

    return add_value_uplift_column(decision_df)


def save_decision_dataset(
    decision_df,
    output_path=OUTPUT_DECISION_DATASET_PATH,
):
    decision_df.to_csv(
        output_path,
        index=False,
        sep=OUTPUT_CSV_SEPARATOR,
        encoding=OUTPUT_CSV_ENCODING,
    )
    return output_path
