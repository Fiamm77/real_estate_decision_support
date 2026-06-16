"""Public valuation API.

This module keeps the original import path while delegating the implementation
to the modular valuation functions.
"""

import logging
import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
logging.getLogger("sklearn").setLevel(logging.ERROR)

try:
    from .predict_impl import (
        get_ksh_price,
        predict_property_frame,
        predict_property_row,
        predict_property_value,
    )
except ImportError:
    from predict_impl import (
        get_ksh_price,
        predict_property_frame,
        predict_property_row,
        predict_property_value,
    )


__all__ = [
    "get_ksh_price",
    "predict_property_frame",
    "predict_property_row",
    "predict_property_value",
]


if __name__ == "__main__":
    result = predict_property_value(
        city="Bik\u00e1cs",
        county="Tolna v\u00e1rmegye",
        settlement_type="k\u00f6zs\u00e9g",
        property_type="Lak\u00f3h\u00e1z",
        land_area_m2=1000,
        building_area_m2=120,
        condition=4,
    )

    print("\n=== PROPERTY VALUATION ===\n")
    for key, value in result.items():
        print(f"{key}: {value}")

    renovated_result = predict_property_value(
        city="Felcs\u00fat",
        county="Fej\u00e9r v\u00e1rmegye",
        settlement_type="k\u00f6zs\u00e9g",
        property_type="Lak\u00f3h\u00e1z",
        land_area_m2=700,
        building_area_m2=130,
        condition=5,
    )

    print("\n=== RENOVATED SCENARIO ===\n")
    for key, value in renovated_result.items():
        print(f"{key}: {value}")

    uplift = (
        renovated_result["predicted_market_value"]
        - result["predicted_market_value"]
    )

    print("\n=== VALUE UPLIFT ===\n")
    print(f"Value uplift: {uplift:,.0f} Ft")
