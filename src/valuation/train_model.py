"""Train two-component structural-score valuation model."""

from pathlib import Path
import unicodedata

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "valuation_model.pkl"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"

DB_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/real_estate"

ADJUSTMENT_MIN = 0.80
ADJUSTMENT_RANGE = 0.40
LAND_WEIGHT_HOUSE = 0.30
BUILDING_WEIGHT_HOUSE = 0.70
NEUTRAL_LAND_SCORE = 0.50

GROUP_COLUMNS = ["county", "settlement_type", "property_type"]

LAND_FEATURES = [
    "land_area_m2",
    "property_type",
    "county",
    "settlement_type",
]

BUILDING_FEATURES = [
    "building_area_m2",
    "condition",
    "property_type",
    "county",
    "settlement_type",
]


def normalize_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def normalize_ascii(value):
    normalized = unicodedata.normalize("NFKD", normalize_text(value))
    return normalized.encode("ascii", "ignore").decode("ascii")


def normalize_county(value):
    return normalize_ascii(value).replace(" varmegye", "")


def is_house_property(value):
    return normalize_ascii(value) == "lakohaz"


def price_column_for_property_type(property_type):
    if is_house_property(property_type):
        return "house_price_m2"

    return "apartment_price_m2"


def extract_valid_price(df, column):
    prices = pd.to_numeric(df[column], errors="coerce")
    prices = prices[prices > 0]

    if prices.empty:
        return None

    return prices.mean()


def find_ksh_price(row, ksh_df):
    price_column = price_column_for_property_type(row["property_type"])
    city = normalize_text(row["city"])
    county = normalize_county(row["county"])
    settlement_type = normalize_text(row["settlement_type"])

    city_match = ksh_df[ksh_df["_city"] == city]
    price = extract_valid_price(city_match, price_column)
    if price is not None:
        return pd.Series({"ksh_price_m2": price, "ksh_source_level": "city"})

    county_settlement_match = ksh_df[
        (ksh_df["_county"] == county)
        & (ksh_df["_settlement_type"] == settlement_type)
    ]
    price = extract_valid_price(county_settlement_match, price_column)
    if price is not None:
        return pd.Series({"ksh_price_m2": price, "ksh_source_level": "county_settlement"})

    county_match = ksh_df[ksh_df["_county"] == county]
    price = extract_valid_price(county_match, price_column)
    if price is not None:
        return pd.Series({"ksh_price_m2": price, "ksh_source_level": "county"})

    price = extract_valid_price(ksh_df, price_column)
    if price is not None:
        return pd.Series({"ksh_price_m2": price, "ksh_source_level": "national"})

    return pd.Series({"ksh_price_m2": 0, "ksh_source_level": "missing"})


def percentile_rank_within_group(series):
    return series.rank(method="average", pct=True)


def build_training_frame(properties_df, ksh_df):
    ksh_df = ksh_df.copy()
    ksh_df["_city"] = ksh_df["city"].map(normalize_text)
    ksh_df["_county"] = ksh_df["county"].map(normalize_county)
    ksh_df["_settlement_type"] = ksh_df["settlement_type"].map(normalize_text)

    df = properties_df.copy()
    ksh_prices = df.apply(lambda row: find_ksh_price(row, ksh_df), axis=1)
    df = pd.concat([df, ksh_prices], axis=1)

    df["is_house"] = df["property_type"].map(is_house_property)
    df["ksh_baseline_value_huf"] = df["ksh_price_m2"] * df["building_area_m2"]

    df["land_value_per_m2"] = df["land_value"] / df["land_area_m2"]
    df.loc[df["land_area_m2"] <= 0, "land_value_per_m2"] = None

    df["building_value_proxy"] = (
        df["building_value"].fillna(0) + df["renovation_cost"].fillna(0)
    )
    df["building_value_per_m2"] = (
        df["building_value_proxy"] / df["building_area_m2"]
    )
    df.loc[df["building_area_m2"] <= 0, "building_value_per_m2"] = None

    df = df[
        (df["building_area_m2"] > 0)
        & (df["building_value_proxy"] > 0)
        & (df["ksh_baseline_value_huf"] > 0)
    ].copy()

    df["building_structural_score"] = (
        df.groupby(GROUP_COLUMNS)["building_value_per_m2"]
        .transform(percentile_rank_within_group)
        .clip(0, 1)
    )

    df["land_structural_score"] = NEUTRAL_LAND_SCORE
    house_land_mask = df["is_house"] & (df["land_area_m2"] > 0) & df["land_value_per_m2"].notna()
    df.loc[house_land_mask, "land_structural_score"] = (
        df.loc[house_land_mask]
        .groupby(GROUP_COLUMNS)["land_value_per_m2"]
        .transform(percentile_rank_within_group)
        .clip(0, 1)
    )

    df["structural_score"] = df["building_structural_score"]
    df.loc[df["is_house"], "structural_score"] = (
        LAND_WEIGHT_HOUSE * df.loc[df["is_house"], "land_structural_score"]
        + BUILDING_WEIGHT_HOUSE * df.loc[df["is_house"], "building_structural_score"]
    )
    df["structural_score"] = df["structural_score"].clip(0, 1)
    df["adjustment_factor"] = ADJUSTMENT_MIN + ADJUSTMENT_RANGE * df["structural_score"]
    df["pseudo_market_value"] = (
        df["ksh_baseline_value_huf"] * df["adjustment_factor"]
    )
    df["benchmark_delta"] = (
        df["pseudo_market_value"] - df["ksh_baseline_value_huf"]
    )

    print(f"Loaded training rows after filters: {len(df)}")
    print("\n=== STRUCTURAL SCORE TARGET DISTRIBUTIONS ===\n")
    print(
        df[
            [
                "land_structural_score",
                "building_structural_score",
                "structural_score",
                "adjustment_factor",
                "pseudo_market_value",
                "ksh_baseline_value_huf",
            ]
        ]
        .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        .T
    )

    return df


def build_model(categorical_features, numerical_features):
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numerical_transformer, numerical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=120,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_model(name, pipeline, x_test, y_test):
    predictions = pipeline.predict(x_test).clip(0, 1)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"\n=== {name.upper()} SCORE METRICS ===\n")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return predictions, {"mae": mae, "rmse": rmse, "r2": r2}


def export_feature_importance(name, pipeline):
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importance_df = pd.DataFrame(
        {
            "component": name,
            "feature": feature_names,
            "importance": pipeline.named_steps["model"].feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    return importance_df


def predict_scores(model_artifact, df):
    land_model = model_artifact["land_model"]
    building_model = model_artifact["building_model"]

    land_predictions = land_model.predict(df[LAND_FEATURES]).clip(0, 1)
    building_predictions = building_model.predict(df[BUILDING_FEATURES]).clip(0, 1)

    land_predictions = pd.Series(land_predictions, index=df.index)
    building_predictions = pd.Series(building_predictions, index=df.index)
    land_predictions.loc[~df["is_house"]] = NEUTRAL_LAND_SCORE

    structural_score = building_predictions.copy()
    structural_score.loc[df["is_house"]] = (
        LAND_WEIGHT_HOUSE * land_predictions.loc[df["is_house"]]
        + BUILDING_WEIGHT_HOUSE * building_predictions.loc[df["is_house"]]
    )

    return land_predictions, building_predictions, structural_score.clip(0, 1)


def export_audit_outputs(df, model_artifact, metrics):
    df = df.copy()
    land_pred, building_pred, structural_pred = predict_scores(model_artifact, df)
    df["predicted_land_structural_score"] = land_pred
    df["predicted_building_structural_score"] = building_pred
    df["predicted_structural_score"] = structural_pred
    df["adjustment_factor"] = ADJUSTMENT_MIN + ADJUSTMENT_RANGE * structural_pred
    df["predicted_market_value"] = (
        df["ksh_baseline_value_huf"] * df["adjustment_factor"]
    )
    df["benchmark_delta"] = (
        df["predicted_market_value"] - df["ksh_baseline_value_huf"]
    )

    audit_metrics_df = pd.DataFrame(
        [
            {"metric": f"land_score_{key}", "value": value}
            for key, value in metrics["land"].items()
        ]
        + [
            {"metric": f"building_score_{key}", "value": value}
            for key, value in metrics["building"].items()
        ]
    )

    audit_distribution_df = pd.concat(
        [
            df["predicted_land_structural_score"].describe(
                percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
            ).rename("predicted_land_structural_score"),
            df["predicted_building_structural_score"].describe(
                percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
            ).rename("predicted_building_structural_score"),
            df["predicted_structural_score"].describe(
                percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
            ).rename("predicted_structural_score"),
            df["adjustment_factor"].describe(
                percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
            ).rename("adjustment_factor"),
            df["predicted_market_value"].describe(
                percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
            ).rename("predicted_market_value"),
            df["ksh_baseline_value_huf"].describe(
                percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
            ).rename("ksh_baseline_value_huf"),
        ],
        axis=1,
    )

    sample_columns = [
        "property_id",
        "city",
        "county",
        "property_type",
        "settlement_type",
        "building_area_m2",
        "land_area_m2",
        "condition",
        "ksh_baseline_value_huf",
        "predicted_land_structural_score",
        "predicted_building_structural_score",
        "predicted_structural_score",
        "adjustment_factor",
        "predicted_market_value",
        "benchmark_delta",
        "ksh_source_level",
    ]
    audit_sample_df = df[sample_columns].head(20)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    audit_metrics_df.to_csv(
        OUTPUTS_DIR / "valuation_audit_metrics.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )
    audit_distribution_df.to_csv(
        OUTPUTS_DIR / "valuation_audit_distributions.csv",
        sep=";",
        encoding="utf-8-sig",
    )
    audit_sample_df.to_csv(
        OUTPUTS_DIR / "valuation_audit_samples.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )
    df.to_csv(PROCESSED_DIR / "valuation_predictions.csv", index=False)

    print("\n=== AUDIT DISTRIBUTIONS ===\n")
    print(audit_distribution_df)
    print("\n=== AUDIT SAMPLE ===\n")
    print(audit_sample_df)


def main():
    engine = create_engine(DB_URL)
    properties_df = pd.read_sql("SELECT * FROM properties_synthetic", engine)
    ksh_df = pd.read_sql("SELECT * FROM ksh_avg_prices", engine)
    df = build_training_frame(properties_df, ksh_df)

    land_categorical = ["property_type", "county", "settlement_type"]
    land_numerical = ["land_area_m2"]
    building_categorical = ["property_type", "county", "settlement_type"]
    building_numerical = ["building_area_m2", "condition"]

    land_train_df = df[df["is_house"]].copy()
    if land_train_df.empty:
        land_train_df = df.copy()
        land_train_df["land_structural_score"] = NEUTRAL_LAND_SCORE

    land_x_train, land_x_test, land_y_train, land_y_test = train_test_split(
        land_train_df[LAND_FEATURES],
        land_train_df["land_structural_score"],
        test_size=0.2,
        random_state=42,
    )
    building_x_train, building_x_test, building_y_train, building_y_test = train_test_split(
        df[BUILDING_FEATURES],
        df["building_structural_score"],
        test_size=0.2,
        random_state=42,
    )

    land_model = build_model(land_categorical, land_numerical)
    building_model = build_model(building_categorical, building_numerical)

    land_model.fit(land_x_train, land_y_train)
    building_model.fit(building_x_train, building_y_train)

    _, land_metrics = evaluate_model("land", land_model, land_x_test, land_y_test)
    _, building_metrics = evaluate_model(
        "building",
        building_model,
        building_x_test,
        building_y_test,
    )

    model_artifact = {
        "model_type": "two_component_structural_score",
        "land_model": land_model,
        "building_model": building_model,
        "land_features": LAND_FEATURES,
        "building_features": BUILDING_FEATURES,
        "adjustment_min": ADJUSTMENT_MIN,
        "adjustment_range": ADJUSTMENT_RANGE,
        "land_weight_house": LAND_WEIGHT_HOUSE,
        "building_weight_house": BUILDING_WEIGHT_HOUSE,
        "neutral_land_score": NEUTRAL_LAND_SCORE,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifact, MODEL_PATH)
    print(f"\nModel saved: {MODEL_PATH}")

    importance_df = pd.concat(
        [
            export_feature_importance("land", land_model),
            export_feature_importance("building", building_model),
        ],
        ignore_index=True,
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(PROCESSED_DIR / "shap_feature_importance.csv", index=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=importance_df.head(30),
        x="importance",
        y="feature",
        hue="component",
    )
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(
        PROCESSED_DIR / "shap_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    export_audit_outputs(
        df=df,
        model_artifact=model_artifact,
        metrics={"land": land_metrics, "building": building_metrics},
    )
    print("\nPredictions and audit outputs exported successfully!")


if __name__ == "__main__":
    main()
