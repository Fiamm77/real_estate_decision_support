from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from valuation.config import PROPERTIES_TABLE  # noqa: E402
from valuation.core import read_table  # noqa: E402


OUTPUT_DIR = ROOT_DIR / "outputs"
PREDICTIONS_PATH = ROOT_DIR / "data" / "processed" / "valuation_predictions.csv"


PERCENTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]


def distribution(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False)[value_col] if group_cols else df[value_col]

    if not group_cols:
        series = df[value_col]
        return pd.DataFrame(
            [
                {
                    "group": "all",
                    "count": series.count(),
                    "min": series.min(),
                    "p01": series.quantile(0.01),
                    "p05": series.quantile(0.05),
                    "p25": series.quantile(0.25),
                    "median": series.median(),
                    "p75": series.quantile(0.75),
                    "p95": series.quantile(0.95),
                    "p99": series.quantile(0.99),
                    "max": series.max(),
                }
            ]
        )

    return (
        grouped.agg(
            count="count",
            min="min",
            p01=lambda x: x.quantile(0.01),
            p05=lambda x: x.quantile(0.05),
            p25=lambda x: x.quantile(0.25),
            median="median",
            p75=lambda x: x.quantile(0.75),
            p95=lambda x: x.quantile(0.95),
            p99=lambda x: x.quantile(0.99),
            max="max",
        )
        .reset_index()
    )


def scenario_delta(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for property_type, group in df.groupby("property_type", dropna=False):
        medians = group.groupby("condition")["predicted_structural_score"].median()
        for source_condition in [2, 3, 4]:
            if source_condition not in medians or 5 not in medians:
                continue
            rows.append(
                {
                    "property_type": property_type,
                    "scenario": f"{source_condition}->5",
                    "source_median_score": medians[source_condition],
                    "target_median_score": medians[5],
                    "median_score_delta": medians[5] - medians[source_condition],
                }
            )
    return pd.DataFrame(rows)


def add_asset_ratios(predictions_df: pd.DataFrame, properties_df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "property_id",
        "building_value",
        "land_value",
        "renovation_cost",
    ]
    merged = predictions_df.merge(
        properties_df[value_cols],
        on="property_id",
        how="left",
        suffixes=("", "_source"),
    )
    for col in ["building_value", "land_value", "renovation_cost"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)
    merged["asset_proxy"] = (
        merged["building_value"] + merged["land_value"] + merged["renovation_cost"]
    )
    merged["land_value_ratio"] = (
        merged["land_value"] / merged["asset_proxy"].replace(0, pd.NA)
    )
    merged["building_value_ratio"] = (
        (merged["building_value"] + merged["renovation_cost"])
        / merged["asset_proxy"].replace(0, pd.NA)
    )
    return merged


def sanity_checks(df: pd.DataFrame) -> pd.DataFrame:
    apartments = df[df["property_type"] == "Lakás"].copy()
    houses = df[df["property_type"] == "Lakóház"].copy()

    rows = [
        {
            "check": "apartments_land_area_zero_rate",
            "value": apartments["land_area_m2"].eq(0).mean(),
        },
        {
            "check": "apartments_land_score_neutral_rate",
            "value": apartments["predicted_land_structural_score"].round(6).eq(0.5).mean(),
        },
        {
            "check": "apartments_structural_equals_building_rate",
            "value": (
                apartments["predicted_structural_score"].round(6)
                == apartments["predicted_building_structural_score"].round(6)
            ).mean(),
        },
        {
            "check": "houses_positive_land_area_rate",
            "value": houses["land_area_m2"].gt(0).mean(),
        },
    ]

    if not houses.empty:
        expected_house_score = (
            0.3 * houses["predicted_land_structural_score"]
            + 0.7 * houses["predicted_building_structural_score"]
        )
        rows.append(
            {
                "check": "houses_structural_matches_30_70_rate",
                "value": (
                    expected_house_score.round(6)
                    == houses["predicted_structural_score"].round(6)
                ).mean(),
            }
        )
        rows.append(
            {
                "check": "houses_land_score_structural_corr",
                "value": houses["predicted_land_structural_score"].corr(
                    houses["predicted_structural_score"]
                ),
            }
        )
        rows.append(
            {
                "check": "houses_building_score_structural_corr",
                "value": houses["predicted_building_structural_score"].corr(
                    houses["predicted_structural_score"]
                ),
            }
        )
        rows.append(
            {
                "check": "houses_land_value_ratio_structural_corr",
                "value": houses["land_value_ratio"].corr(
                    houses["predicted_structural_score"]
                ),
            }
        )

    return pd.DataFrame(rows)


def comparison_samples(df: pd.DataFrame) -> pd.DataFrame:
    apartments = df[
        (df["property_type"] == "Lakás")
        & (df["building_area_m2"].between(50, 60))
    ].copy()
    houses = df[
        (df["property_type"] == "Lakóház")
        & (df["building_area_m2"].between(90, 110))
    ].copy()

    sample_rows = []
    for (county, settlement_type), apt_group in apartments.groupby(
        ["county", "settlement_type"], dropna=False
    ):
        house_group = houses[
            (houses["county"] == county)
            & (houses["settlement_type"] == settlement_type)
        ].copy()
        if house_group.empty:
            continue

        apt = apt_group.assign(_distance=(apt_group["building_area_m2"] - 55).abs()).sort_values(
            "_distance"
        ).head(3)
        house = house_group.assign(
            _distance=(house_group["building_area_m2"] - 100).abs()
        ).sort_values("_distance").head(3)

        pair = pd.concat([apt, house], ignore_index=True)
        pair["comparison_group"] = f"{county} / {settlement_type}"
        sample_rows.append(pair)
        if len(sample_rows) >= 5:
            break

    if not sample_rows:
        return pd.DataFrame()

    columns = [
        "comparison_group",
        "property_id",
        "property_type",
        "county",
        "settlement_type",
        "city",
        "building_area_m2",
        "land_area_m2",
        "ksh_baseline_value_huf",
        "predicted_land_structural_score",
        "predicted_building_structural_score",
        "predicted_structural_score",
        "adjustment_factor",
        "predicted_market_value",
        "land_value_ratio",
    ]
    return pd.concat(sample_rows, ignore_index=True)[columns]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.read_csv(PREDICTIONS_PATH)
    properties_df = read_table(PROPERTIES_TABLE)
    df = add_asset_ratios(predictions_df, properties_df)

    median_score = df["predicted_structural_score"].median()
    df["raw_factor"] = df["predicted_structural_score"] / median_score

    all_score_dist = distribution(df, [], "predicted_structural_score")
    type_score_dist = distribution(df, ["property_type"], "predicted_structural_score")
    type_condition_score_dist = distribution(
        df, ["property_type", "condition"], "predicted_structural_score"
    )
    raw_factor_dist = distribution(df, [], "raw_factor")
    raw_factor_type_dist = distribution(df, ["property_type"], "raw_factor")
    raw_factor_type_condition_dist = distribution(
        df, ["property_type", "condition"], "raw_factor"
    )
    scenario_df = scenario_delta(df)

    property_type_metrics = []
    for value_col in [
        "ksh_baseline_value_huf",
        "predicted_market_value",
        "predicted_structural_score",
        "predicted_land_structural_score",
        "predicted_building_structural_score",
        "land_value_ratio",
        "building_value_ratio",
        "adjustment_factor",
    ]:
        metric_df = distribution(df, ["property_type"], value_col)
        metric_df.insert(0, "metric", value_col)
        property_type_metrics.append(metric_df)
    property_type_metrics_df = pd.concat(property_type_metrics, ignore_index=True)

    checks_df = sanity_checks(df)
    samples_df = comparison_samples(df)

    outputs = {
        "raw_structural_score_distribution.csv": all_score_dist,
        "raw_structural_score_by_property_type.csv": type_score_dist,
        "raw_structural_score_by_property_type_condition.csv": type_condition_score_dist,
        "raw_factor_distribution.csv": raw_factor_dist,
        "raw_factor_by_property_type.csv": raw_factor_type_dist,
        "raw_factor_by_property_type_condition.csv": raw_factor_type_condition_dist,
        "condition_score_delta_summary.csv": scenario_df,
        "property_type_component_audit.csv": property_type_metrics_df,
        "property_type_sanity_checks.csv": checks_df,
        "property_type_comparison_samples.csv": samples_df,
    }

    for filename, output_df in outputs.items():
        output_df.to_csv(
            OUTPUT_DIR / filename,
            index=False,
            sep=";",
            encoding="utf-8-sig",
        )

    print("\n=== RAW STRUCTURAL SCORE: ALL ===")
    print(all_score_dist.to_string(index=False))
    print("\n=== RAW STRUCTURAL SCORE BY PROPERTY TYPE ===")
    print(type_score_dist.to_string(index=False))
    print("\n=== RAW STRUCTURAL SCORE BY PROPERTY TYPE x CONDITION ===")
    print(type_condition_score_dist.to_string(index=False))
    print("\n=== RAW FACTOR BY PROPERTY TYPE ===")
    print(raw_factor_type_dist.to_string(index=False))
    print("\n=== CONDITION DELTAS ===")
    print(scenario_df.to_string(index=False))
    print("\n=== PROPERTY TYPE COMPONENT AUDIT ===")
    print(property_type_metrics_df.to_string(index=False))
    print("\n=== SANITY CHECKS ===")
    print(checks_df.to_string(index=False))
    print("\n=== COMPARISON SAMPLES ===")
    print(samples_df.to_string(index=False))


if __name__ == "__main__":
    main()
