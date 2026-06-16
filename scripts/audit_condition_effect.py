from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from valuation.config import PROPERTIES_TABLE  # noqa: E402
from valuation.core import read_table  # noqa: E402
from valuation.predict_impl import predict_property_frame  # noqa: E402


OUTPUT_DIR = ROOT_DIR / "outputs"
PROCESSED_PREDICTIONS_PATH = ROOT_DIR / "data" / "processed" / "valuation_predictions.csv"


def summarize_by_condition(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return (
        df.groupby("condition", dropna=False)[value_column]
        .agg(
            count="count",
            mean="mean",
            median="median",
            p25=lambda x: x.quantile(0.25),
            p75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )


def scenario_audit(properties_df: pd.DataFrame, source_condition: int) -> pd.DataFrame:
    scenario_df = properties_df[
        (properties_df["condition"] == source_condition)
        & (properties_df["building_area_m2"] > 0)
    ].copy()
    scenario_df = scenario_df.reset_index(drop=True)

    current_prediction = predict_property_frame(scenario_df)
    target_df = scenario_df.copy()
    target_df["condition"] = 5
    target_prediction = predict_property_frame(target_df)

    result = scenario_df[
        [
            "property_id",
            "property_type",
            "county",
            "settlement_type",
            "city",
            "building_area_m2",
            "land_area_m2",
        ]
    ].copy()
    result["scenario"] = f"{source_condition}->5"
    result["source_condition"] = source_condition
    result["target_condition"] = 5
    result["current_score"] = current_prediction["predicted_structural_score"]
    result["target_score"] = target_prediction["predicted_structural_score"]
    result["score_delta"] = result["target_score"] - result["current_score"]
    result["current_factor"] = current_prediction["adjustment_factor"]
    result["target_factor"] = target_prediction["adjustment_factor"]
    result["factor_delta"] = result["target_factor"] - result["current_factor"]
    result["current_value"] = current_prediction["predicted_market_value"]
    result["target_value"] = target_prediction["predicted_market_value"]
    result["value_delta"] = result["target_value"] - result["current_value"]
    result["value_delta_pct"] = result["value_delta"] / result["current_value"].replace(0, pd.NA)
    result["ksh_baseline"] = current_prediction["ksh_baseline_value_huf"]
    result["ksh_source_level"] = current_prediction["ksh_source_level"]
    return result


def summarize_scenarios(scenario_df: pd.DataFrame) -> pd.DataFrame:
    return (
        scenario_df.groupby(["scenario", "property_type"], dropna=False)
        .agg(
            count=("property_id", "count"),
            current_score_median=("current_score", "median"),
            target_score_median=("target_score", "median"),
            score_delta_mean=("score_delta", "mean"),
            score_delta_median=("score_delta", "median"),
            score_delta_p25=("score_delta", lambda x: x.quantile(0.25)),
            score_delta_p75=("score_delta", lambda x: x.quantile(0.75)),
            factor_delta_mean=("factor_delta", "mean"),
            factor_delta_median=("factor_delta", "median"),
            value_delta_mean=("value_delta", "mean"),
            value_delta_median=("value_delta", "median"),
            value_delta_p25=("value_delta", lambda x: x.quantile(0.25)),
            value_delta_p75=("value_delta", lambda x: x.quantile(0.75)),
            value_delta_pct_median=("value_delta_pct", "median"),
            current_value_median=("current_value", "median"),
            target_value_median=("target_value", "median"),
            ksh_baseline_median=("ksh_baseline", "median"),
        )
        .reset_index()
    )


def summarize_mapping_flatness(scenario_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, group in scenario_df.groupby("scenario"):
        median_score_delta = group["score_delta"].median()
        median_factor_delta = group["factor_delta"].median()
        rows.append(
            {
                "scenario": scenario,
                "median_score_delta": median_score_delta,
                "current_band_factor_delta": median_factor_delta,
                "current_band_value_pct": median_factor_delta
                / group["current_factor"].median(),
                "wide_060_140_factor_delta": 0.80 * median_score_delta,
                "wide_060_140_value_pct": (0.80 * median_score_delta)
                / group["current_factor"].median(),
                "narrow_080_120_factor_delta": 0.40 * median_score_delta,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.read_csv(PROCESSED_PREDICTIONS_PATH)
    properties_df = read_table(PROPERTIES_TABLE)

    score_by_condition = summarize_by_condition(
        predictions_df, "predicted_structural_score"
    )
    factor_by_condition = summarize_by_condition(predictions_df, "adjustment_factor")

    scenario_frames = [
        scenario_audit(properties_df, source_condition)
        for source_condition in [2, 3, 4]
    ]
    scenario_df = pd.concat(scenario_frames, ignore_index=True)
    scenario_summary = summarize_scenarios(scenario_df)
    flatness_summary = summarize_mapping_flatness(scenario_df)

    score_by_condition.to_csv(
        OUTPUT_DIR / "condition_score_audit.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )
    factor_by_condition.to_csv(
        OUTPUT_DIR / "condition_adjustment_factor_audit.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )
    scenario_summary.to_csv(
        OUTPUT_DIR / "condition_scenario_summary.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )
    scenario_df.to_csv(
        OUTPUT_DIR / "condition_scenario_rows.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )
    flatness_summary.to_csv(
        OUTPUT_DIR / "condition_mapping_flatness_audit.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    print("\n=== STRUCTURAL SCORE BY CONDITION ===")
    print(score_by_condition.to_string(index=False))
    print("\n=== ADJUSTMENT FACTOR BY CONDITION ===")
    print(factor_by_condition.to_string(index=False))
    print("\n=== SCENARIO SUMMARY ===")
    print(scenario_summary.to_string(index=False))
    print("\n=== MAPPING FLATNESS ===")
    print(flatness_summary.to_string(index=False))


if __name__ == "__main__":
    main()
