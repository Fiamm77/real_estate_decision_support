"""Configuration and constants for the valuation pipeline."""

DB_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/real_estate"
MODEL_PATH = "models/valuation_model.pkl"

KSH_AVG_PRICES_TABLE = "ksh_avg_prices"
PROPERTIES_TABLE = "properties_synthetic"
CONDITIONS_TABLE = "conditions"
PACKAGES_TABLE = "packages"
PACKAGE_WORKS_TABLE = "package_works"
WORKS_TABLE = "works"

OUTPUT_DECISION_DATASET_PATH = "outputs/decision_dataset.csv"
OUTPUT_CSV_SEPARATOR = ";"
OUTPUT_CSV_ENCODING = "utf-8-sig"

RENOVATION_MULTIPLIER = 4
TARGET_RENOVATION_CONDITION = 5
NON_RENOVATABLE_CONDITION = 1

HOUSE_PROPERTY_TYPE = "lak\u00f3h\u00e1z"
HOUSE_PRICE_COLUMN = "house_price_m2"
APARTMENT_PRICE_COLUMN = "apartment_price_m2"
WORK_IDS_COLUMN = "Munk\u00e1k"
WORK_NAME_COLUMN = "Munkamenet"
WORK_COST_COLUMN = "Med Ft/nm (2024)"

REFERENCE_FEATURES = [
    "property_type",
    "settlement_type",
    "county",
    "land_area_m2",
    "building_area_m2",
    "condition",
]
