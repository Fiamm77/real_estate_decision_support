import os

os.environ["PYTHONWARNINGS"] = "ignore"

import warnings

warnings.filterwarnings("ignore")

import logging

logging.getLogger("sklearn").setLevel(
    logging.ERROR
)

import pandas as pd
import joblib

from sqlalchemy import create_engine

# =====================================================
# DATABASE CONNECTION
# =====================================================

engine = create_engine(
    "postgresql+psycopg2://postgres:postgres@localhost:5432/real_estate"
)

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load(
    "models/valuation_model.pkl"
)

print("Model loaded successfully!")

# =====================================================
# LOAD KSH DATA
# =====================================================

ksh_df = pd.read_sql(
    "SELECT * FROM ksh_avg_prices",
    engine
)

print(f"KSH rows loaded: {len(ksh_df)}")

# =====================================================
# LOAD REFERENCE DATASET
# =====================================================

reference_df = pd.read_sql(
    "SELECT * FROM properties_synthetic",
    engine
)

# =====================================================
# CALCULATE DATASET MEAN PREDICTION
# =====================================================

REFERENCE_FEATURES = [
    "land_area_m2",
    "building_area_m2",
    "condition",
    "annual_cost",
    "renovation_cost"
]

reference_predictions = model.predict(
    reference_df[REFERENCE_FEATURES]
)

dataset_mean_prediction = (
    reference_predictions.mean()
)

print(
    f"Dataset mean prediction: "
    f"{dataset_mean_prediction:,.0f} Ft"
)

# =====================================================
# KSH LOOKUP FUNCTION
# =====================================================

def get_ksh_price(
    city,
    county,
    settlement_type,
    property_type
):
    """
    Hierarchical KSH fallback logic

    1. Exact city match
    2. County + settlement type
    3. County average
    4. National average

    Handles missing house/apartment values safely.
    """

    # =================================================
    # PROPERTY TYPE COLUMN
    # =================================================

    if property_type.lower() == "lakóház":
        price_column = "house_price_m2"
    else:
        price_column = "apartment_price_m2"

    # =================================================
    # HELPER FUNCTION
    # =================================================

    def extract_valid_price(df, column):

        if df.empty:
            return None

        prices = (
            df[column]
            .dropna()
        )

        # remove zero values
        prices = prices[
            prices > 0
        ]

        if len(prices) == 0:
            return None

        return prices.mean()

    # =================================================
    # 1. CITY LEVEL
    # =================================================

    if city is not None:

        city_match = ksh_df[
            (
                ksh_df["city"].str.lower()
                == city.lower()
            )
        ]

        price = extract_valid_price(
            city_match,
            price_column
        )

        if price is not None:

            return {
                "price_m2": round(price, 0),
                "source_level": "city"
            }

    # =================================================
    # 2. COUNTY + SETTLEMENT TYPE
    # =================================================

    county_settlement_match = ksh_df[
        (
            ksh_df["county"].str.lower()
            == county.lower()
        )
        &
        (
            ksh_df["settlement_type"].str.lower()
            == settlement_type.lower()
        )
    ]

    price = extract_valid_price(
        county_settlement_match,
        price_column
    )

    if price is not None:

        return {
            "price_m2": round(price, 0),
            "source_level": "county_settlement"
        }

    # =================================================
    # 3. COUNTY LEVEL
    # =================================================

    county_match = ksh_df[
        (
            ksh_df["county"].str.lower()
            == county.lower()
        )
    ]

    price = extract_valid_price(
        county_match,
        price_column
    )

    if price is not None:

        return {
            "price_m2": round(price, 0),
            "source_level": "county"
        }

    # =================================================
    # 4. NATIONAL LEVEL
    # =================================================

    price = extract_valid_price(
        ksh_df,
        price_column
    )

    if price is not None:

        return {
            "price_m2": round(price, 0),
            "source_level": "national"
        }

    # =================================================
    # FINAL SAFETY FALLBACK
    # =================================================

    return {
        "price_m2": 0,
        "source_level": "missing"
    }

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_property_value(
    city,
    county,
    settlement_type,
    property_type,
    land_area_m2,
    building_area_m2,
    condition,
    annual_cost,
    renovation_cost
):

    # =================================================
    # CREATE INPUT DATAFRAME
    # =================================================

    input_df = pd.DataFrame([{
        "land_area_m2": land_area_m2,
        "building_area_m2": building_area_m2,
        "condition": condition,
        "annual_cost": annual_cost,
        "renovation_cost": renovation_cost
    }])

    # =================================================
    # ML PREDICTION
    # =================================================

    predicted_value = model.predict(
        input_df
    )[0]

    # =================================================
    # STRUCTURAL MODIFIER
    # =================================================

    modifier = (
        predicted_value
        / dataset_mean_prediction
    )

    # =================================================
    # KSH LOOKUP
    # =================================================

    ksh_result = get_ksh_price(
        city=city,
        county=county,
        settlement_type=settlement_type,
        property_type=property_type
    )

    ksh_price_m2 = ksh_result["price_m2"]

    # =================================================
    # BASELINE VALUE
    # =================================================

    baseline_value = (
        ksh_price_m2
        * building_area_m2
    )

    # =================================================
    # FINAL VALUE
    # =================================================

    final_value = (
        baseline_value
        * modifier
    )

    # =================================================
    # RETURN RESULTS
    # =================================================

    return {
        "ksh_price_m2": round(ksh_price_m2, 0),
        "baseline_value": round(baseline_value, 0),
        "ml_prediction": round(predicted_value, 0),
        "modifier": round(modifier, 3),
        "final_value": round(final_value, 0),
        "ksh_source_level": ksh_result["source_level"]
    }


# =====================================================
# ROW-BASED PREDICTION WRAPPER
# =====================================================

def predict_property_row(row):

    result = predict_property_value(

        city=row.get("city"),

        county=row.get("county"),

        settlement_type=row.get("settlement_type"),

        property_type=row.get("property_type"),

        land_area_m2=row.get("land_area_m2"),

        building_area_m2=row.get("building_area_m2"),

        condition=row.get("condition"),

        annual_cost=row.get("annual_cost"),

        renovation_cost=row.get("renovation_cost")
    )

    return pd.Series(result)


# =====================================================
# MANUAL TEST
# =====================================================

if __name__ == "__main__":

    result = predict_property_value(
        city="Bikács",
        county="Tolna vármegye",
        settlement_type="község",
        property_type="Lakóház",
        land_area_m2=1000,
        building_area_m2=120,
        condition=4,
        annual_cost=200000,
        renovation_cost=0
    )

    print("\n=== PROPERTY VALUATION ===\n")

    for key, value in result.items():
        print(f"{key}: {value}")

    # =================================================

    renovated_result = predict_property_value(
        city="Felcsút",
        county="Fejér vármegye",
        settlement_type="község",
        property_type="Lakóház",
        land_area_m2=700,
        building_area_m2=130,
        condition=5,
        annual_cost=100000,
        renovation_cost=12000000
    )

    print("\n=== RENOVATED SCENARIO ===\n")

    for key, value in renovated_result.items():
        print(f"{key}: {value}")

    # =================================================

    uplift = (
        renovated_result["final_value"]
        - result["final_value"]
    )

    print("\n=== VALUE UPLIFT ===\n")

    print(f"Value uplift: {uplift:,.0f} Ft")