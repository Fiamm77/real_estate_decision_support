from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from valuation.config import KSH_AVG_PRICES_TABLE, PROPERTIES_TABLE  # noqa: E402
from valuation.core import read_table  # noqa: E402


pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def describe_series(series: pd.Series) -> pd.DataFrame:
    return series.describe(
        percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ).to_frame().T


def group_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (
        df.groupby(columns, dropna=False)
        .agg(
            n=("property_id", "count"),
            land_ratio_median=("land_value_ratio", "median"),
            land_ratio_mean=("land_value_ratio", "mean"),
            building_ratio_median=("building_value_ratio", "median"),
            land_value_median=("land_value", "median"),
            building_proxy_median=("building_proxy", "median"),
            asset_proxy_median=("asset_proxy", "median"),
            land_per_m2_median=("land_value_per_m2", "median"),
            building_per_m2_median=("building_proxy_per_m2", "median"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )


def prepare_properties() -> pd.DataFrame:
    df = read_table(PROPERTIES_TABLE).copy()
    numeric_columns = [
        "land_value",
        "building_value",
        "renovation_cost",
        "land_area_m2",
        "building_area_m2",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["renovation_cost"] = df["renovation_cost"].fillna(0)
    df["asset_proxy"] = (
        df["land_value"].fillna(0)
        + df["building_value"].fillna(0)
        + df["renovation_cost"].fillna(0)
    )
    df["building_proxy"] = df["building_value"].fillna(0) + df[
        "renovation_cost"
    ].fillna(0)

    valid_asset = df["asset_proxy"].gt(0)
    df["land_value_ratio"] = np.where(
        valid_asset, df["land_value"].fillna(0) / df["asset_proxy"], np.nan
    )
    df["building_value_ratio"] = np.where(
        valid_asset, df["building_proxy"] / df["asset_proxy"], np.nan
    )
    df["land_value_per_m2"] = np.where(
        df["land_area_m2"].gt(0), df["land_value"] / df["land_area_m2"], np.nan
    )
    df["building_proxy_per_m2"] = np.where(
        df["building_area_m2"].gt(0),
        df["building_proxy"] / df["building_area_m2"],
        np.nan,
    )
    return df


def prepare_ksh_long() -> pd.DataFrame:
    ksh = read_table(KSH_AVG_PRICES_TABLE).copy()
    for column in ["apartment_price_m2", "house_price_m2"]:
        ksh[column] = pd.to_numeric(ksh[column], errors="coerce").fillna(0)

    apartments = ksh[
        ["city", "county", "settlement_type", "apartment_price_m2"]
    ].rename(columns={"apartment_price_m2": "ksh_price_m2"})
    apartments["property_type"] = "Lakás"

    houses = ksh[["city", "county", "settlement_type", "house_price_m2"]].rename(
        columns={"house_price_m2": "ksh_price_m2"}
    )
    houses["property_type"] = "Lakóház"

    ksh_long = pd.concat([apartments, houses], ignore_index=True)
    return ksh_long[ksh_long["ksh_price_m2"].gt(0)].copy()


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    df = prepare_properties()
    print_section("DATASET")
    print("properties rows", len(df))
    print(df["property_type"].value_counts(dropna=False).to_string())

    print_section("LAND / BUILDING RATIO DISTRIBUTION")
    print(describe_series(df["land_value_ratio"]).to_string(index=False))
    print(describe_series(df["building_value_ratio"]).to_string(index=False))

    print_section("VALUE SCALE DISTRIBUTION")
    for column in ["land_value", "building_value", "renovation_cost", "asset_proxy"]:
        print(f"\n{column}")
        print(describe_series(df[column]).to_string(index=False))

    print_section("BY PROPERTY TYPE")
    print(group_summary(df, ["property_type"]).to_string(index=False))

    print_section("BY SETTLEMENT TYPE")
    print(group_summary(df, ["settlement_type"]).to_string(index=False))

    print_section("BY COUNTY TOP 20")
    print(group_summary(df, ["county"]).head(20).to_string(index=False))

    ksh_long = prepare_ksh_long()
    merged = df.merge(
        ksh_long,
        on=["city", "county", "settlement_type", "property_type"],
        how="left",
    )
    merged["ksh_baseline"] = merged["ksh_price_m2"] * merged["building_area_m2"]
    merged["asset_to_ksh_ratio"] = np.where(
        merged["ksh_baseline"].gt(0),
        merged["asset_proxy"] / merged["ksh_baseline"],
        np.nan,
    )
    valid = merged[
        merged["asset_to_ksh_ratio"].replace([np.inf, -np.inf], np.nan).notna()
    ].copy()

    print_section("KSH MATCH")
    print(
        "matched rows",
        len(valid),
        "of",
        len(merged),
        "match_rate",
        round(len(valid) / len(merged), 3),
    )

    print_section("ASSET_PROXY / KSH_BASELINE RATIO")
    print(describe_series(valid["asset_to_ksh_ratio"]).to_string(index=False))

    print_section("RATIO BY PROPERTY TYPE")
    print(
        valid.groupby("property_type")
        .agg(
            n=("property_id", "count"),
            median=("asset_to_ksh_ratio", "median"),
            mean=("asset_to_ksh_ratio", "mean"),
            p05=("asset_to_ksh_ratio", lambda x: x.quantile(0.05)),
            p25=("asset_to_ksh_ratio", lambda x: x.quantile(0.25)),
            p75=("asset_to_ksh_ratio", lambda x: x.quantile(0.75)),
            p95=("asset_to_ksh_ratio", lambda x: x.quantile(0.95)),
        )
        .reset_index()
        .to_string(index=False)
    )

    print_section("RATIO BY SETTLEMENT TYPE")
    print(
        valid.groupby("settlement_type")
        .agg(
            n=("property_id", "count"),
            median=("asset_to_ksh_ratio", "median"),
            p10=("asset_to_ksh_ratio", lambda x: x.quantile(0.10)),
            p90=("asset_to_ksh_ratio", lambda x: x.quantile(0.90)),
        )
        .reset_index()
        .sort_values("n", ascending=False)
        .to_string(index=False)
    )

    print_section("RATIO BY COUNTY TOP 20")
    print(
        valid.groupby("county")
        .agg(
            n=("property_id", "count"),
            median=("asset_to_ksh_ratio", "median"),
            p10=("asset_to_ksh_ratio", lambda x: x.quantile(0.10)),
            p90=("asset_to_ksh_ratio", lambda x: x.quantile(0.90)),
        )
        .reset_index()
        .sort_values("n", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    valid["asset_proxy_per_m2"] = np.where(
        valid["building_area_m2"].gt(0),
        valid["asset_proxy"] / valid["building_area_m2"],
        np.nan,
    )
    dispersion = (
        valid.groupby(["county", "settlement_type", "property_type"], dropna=False)
        .agg(
            n=("property_id", "count"),
            p10_asset_m2=("asset_proxy_per_m2", lambda x: x.quantile(0.10)),
            p50_asset_m2=("asset_proxy_per_m2", "median"),
            p90_asset_m2=("asset_proxy_per_m2", lambda x: x.quantile(0.90)),
            p10_ratio=("asset_to_ksh_ratio", lambda x: x.quantile(0.10)),
            p50_ratio=("asset_to_ksh_ratio", "median"),
            p90_ratio=("asset_to_ksh_ratio", lambda x: x.quantile(0.90)),
        )
        .reset_index()
    )
    dispersion = dispersion[dispersion["n"] >= 20].copy()
    dispersion["p90_p10_ratio_spread"] = (
        dispersion["p90_ratio"] - dispersion["p10_ratio"]
    )
    print_section("GROUP DISPERSION n>=20 TOP SPREAD")
    print(
        dispersion.sort_values("p90_p10_ratio_spread", ascending=False)
        .head(15)
        .to_string(index=False)
    )

    print_section("SUGGESTED DATA-DRIVEN WEIGHTS FROM RATIOS")
    print(
        "global land median",
        df["land_value_ratio"].median(),
        "building median",
        df["building_value_ratio"].median(),
    )
    print(
        df.groupby("property_type")[["land_value_ratio", "building_value_ratio"]]
        .median()
        .to_string()
    )


if __name__ == "__main__":
    main()
