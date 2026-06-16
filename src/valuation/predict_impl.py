"""Two-component structural-score prediction with KSH market benchmark."""

import unicodedata

import pandas as pd

try:
    from .config import (
        APARTMENT_PRICE_COLUMN,
        HOUSE_PRICE_COLUMN,
        HOUSE_PROPERTY_TYPE,
        KSH_AVG_PRICES_TABLE,
        MODEL_PATH,
    )
    from .core import load_model, read_table
except ImportError:
    from config import (
        APARTMENT_PRICE_COLUMN,
        HOUSE_PRICE_COLUMN,
        HOUSE_PROPERTY_TYPE,
        KSH_AVG_PRICES_TABLE,
        MODEL_PATH,
    )
    from core import load_model, read_table


model_artifact = load_model(MODEL_PATH)

for component in ["land_model", "building_model"]:
    try:
        model_artifact[component].named_steps["model"].set_params(n_jobs=1)
    except (AttributeError, KeyError, TypeError, ValueError):
        pass

ksh_df = read_table(KSH_AVG_PRICES_TABLE)


def _normalize_text(value):
    if value is None:
        return ""

    return str(value).strip().lower()


def _normalize_ascii(value):
    normalized = unicodedata.normalize("NFKD", _normalize_text(value))
    return normalized.encode("ascii", "ignore").decode("ascii")


def _normalize_county(value):
    return _normalize_ascii(value).replace(" varmegye", "")


def _is_house_property(value):
    return _normalize_ascii(value) == _normalize_ascii(HOUSE_PROPERTY_TYPE)


def _extract_valid_price(df, column):
    if df.empty:
        return None

    prices = pd.to_numeric(df[column], errors="coerce")
    prices = prices[prices > 0]

    if prices.empty:
        return None

    return prices.mean()


def _price_column_for_property_type(property_type):
    if _is_house_property(property_type):
        return HOUSE_PRICE_COLUMN

    return APARTMENT_PRICE_COLUMN


def _prepare_ksh_work_df():
    ksh_work_df = ksh_df.copy()
    ksh_work_df["_city"] = ksh_work_df["city"].map(_normalize_text)
    ksh_work_df["_county"] = ksh_work_df["county"].map(_normalize_county)
    ksh_work_df["_settlement_type"] = ksh_work_df["settlement_type"].map(
        _normalize_text
    )
    return ksh_work_df


def _lookup_ksh_price(city, county, settlement_type, property_type, ksh_work_df):
    price_column = _price_column_for_property_type(property_type)

    normalized_city = _normalize_text(city)
    normalized_county = _normalize_county(county)
    normalized_settlement_type = _normalize_text(settlement_type)

    if normalized_city:
        city_match = ksh_work_df[ksh_work_df["_city"] == normalized_city]
        price = _extract_valid_price(city_match, price_column)
        if price is not None:
            return {"price_m2": round(price, 0), "source_level": "city"}

    county_settlement_match = ksh_work_df[
        (ksh_work_df["_county"] == normalized_county)
        & (ksh_work_df["_settlement_type"] == normalized_settlement_type)
    ]
    price = _extract_valid_price(county_settlement_match, price_column)
    if price is not None:
        return {"price_m2": round(price, 0), "source_level": "county_settlement"}

    county_match = ksh_work_df[ksh_work_df["_county"] == normalized_county]
    price = _extract_valid_price(county_match, price_column)
    if price is not None:
        return {"price_m2": round(price, 0), "source_level": "county"}

    price = _extract_valid_price(ksh_work_df, price_column)
    if price is not None:
        return {"price_m2": round(price, 0), "source_level": "national"}

    return {"price_m2": 0, "source_level": "missing"}


def get_ksh_price(city, county, settlement_type, property_type):
    """Hierarchical KSH fallback logic used as market benchmark."""

    return _lookup_ksh_price(
        city=city,
        county=county,
        settlement_type=settlement_type,
        property_type=property_type,
        ksh_work_df=_prepare_ksh_work_df(),
    )


def _build_single_row_frame(
    county,
    settlement_type,
    property_type,
    land_area_m2,
    building_area_m2,
    condition,
):
    return pd.DataFrame(
        [
            {
                "property_type": property_type,
                "settlement_type": settlement_type,
                "county": county,
                "land_area_m2": land_area_m2,
                "building_area_m2": building_area_m2,
                "condition": condition,
            }
        ]
    )


def _predict_structural_components(properties_df):
    land_model = model_artifact["land_model"]
    building_model = model_artifact["building_model"]
    land_features = model_artifact["land_features"]
    building_features = model_artifact["building_features"]

    land_scores = pd.Series(
        land_model.predict(properties_df[land_features]).clip(0, 1),
        index=properties_df.index,
    )
    building_scores = pd.Series(
        building_model.predict(properties_df[building_features]).clip(0, 1),
        index=properties_df.index,
    )

    is_house = properties_df["property_type"].map(_is_house_property)
    neutral_land_score = model_artifact["neutral_land_score"]
    land_scores.loc[~is_house] = neutral_land_score

    structural_scores = building_scores.copy()
    structural_scores.loc[is_house] = (
        model_artifact["land_weight_house"] * land_scores.loc[is_house]
        + model_artifact["building_weight_house"] * building_scores.loc[is_house]
    )

    return land_scores.clip(0, 1), building_scores.clip(0, 1), structural_scores.clip(0, 1)


def _build_output_frame(properties_df):
    land_scores, building_scores, structural_scores = _predict_structural_components(
        properties_df
    )

    ksh_work_df = _prepare_ksh_work_df()
    ksh_results = [
        _lookup_ksh_price(
            city=row.get("city"),
            county=row.get("county"),
            settlement_type=row.get("settlement_type"),
            property_type=row.get("property_type"),
            ksh_work_df=ksh_work_df,
        )
        for _, row in properties_df.iterrows()
    ]
    result_df = pd.DataFrame(ksh_results)
    result_df = result_df.rename(
        columns={
            "price_m2": "ksh_price_m2",
            "source_level": "ksh_source_level",
        }
    )

    result_df["ksh_baseline_value_huf"] = (
        result_df["ksh_price_m2"] * properties_df["building_area_m2"].to_numpy()
    )
    result_df["predicted_land_structural_score"] = land_scores.to_numpy()
    result_df["predicted_building_structural_score"] = building_scores.to_numpy()
    result_df["predicted_structural_score"] = structural_scores.to_numpy()
    result_df["adjustment_factor"] = (
        model_artifact["adjustment_min"]
        + model_artifact["adjustment_range"] * result_df["predicted_structural_score"]
    )
    result_df["predicted_market_value"] = (
        result_df["ksh_baseline_value_huf"] * result_df["adjustment_factor"]
    )
    result_df["benchmark_delta"] = (
        result_df["predicted_market_value"] - result_df["ksh_baseline_value_huf"]
    )

    return result_df[
        [
            "predicted_market_value",
            "ksh_price_m2",
            "ksh_baseline_value_huf",
            "predicted_land_structural_score",
            "predicted_building_structural_score",
            "predicted_structural_score",
            "adjustment_factor",
            "benchmark_delta",
            "ksh_source_level",
        ]
    ].round(
        {
            "predicted_market_value": 0,
            "ksh_price_m2": 0,
            "ksh_baseline_value_huf": 0,
            "predicted_land_structural_score": 6,
            "predicted_building_structural_score": 6,
            "predicted_structural_score": 6,
            "adjustment_factor": 6,
            "benchmark_delta": 0,
        }
    )


def predict_property_value(
    city,
    county,
    settlement_type,
    property_type,
    land_area_m2,
    building_area_m2,
    condition,
    activation_year=None,
    annual_cost=None,
    renovation_cost=None,
):
    input_df = _build_single_row_frame(
        county=county,
        settlement_type=settlement_type,
        property_type=property_type,
        land_area_m2=land_area_m2,
        building_area_m2=building_area_m2,
        condition=condition,
    )
    input_df["city"] = city

    return _build_output_frame(input_df).iloc[0].to_dict()


def predict_property_row(row):
    return pd.Series(
        predict_property_value(
            city=row.get("city"),
            county=row.get("county"),
            settlement_type=row.get("settlement_type"),
            property_type=row.get("property_type"),
            land_area_m2=row.get("land_area_m2"),
            building_area_m2=row.get("building_area_m2"),
            condition=row.get("condition"),
            annual_cost=row.get("annual_cost"),
            renovation_cost=row.get("renovation_cost"),
        )
    )


def predict_property_frame(properties_df):
    """Vectorized structural-score prediction plus KSH benchmark columns."""

    return _build_output_frame(properties_df.copy())


if __name__ == "__main__":
    result = predict_property_value(
        city="Bikacs",
        county="Tolna varmegye",
        settlement_type="kozseg",
        property_type="Lakohaz",
        land_area_m2=1000,
        building_area_m2=120,
        condition=4,
    )
    print(result)
