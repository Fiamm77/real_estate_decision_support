"""Interactive Streamlit valuation app for user-entered property data."""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"
for import_path in [BASE_DIR, SRC_DIR]:
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from valuation.calculate_renovation_cost import calculate_renovation_cost
from valuation.config import CONDITIONS_TABLE, KSH_AVG_PRICES_TABLE
from valuation.core import read_table
from valuation.pipeline import apply_minimum_renovation_uplift
from valuation.predict_value import predict_property_value


DECISION_DATASET_PATH = BASE_DIR / "outputs" / "decision_dataset.csv"
SHAP_EXPLANATIONS_PATH = BASE_DIR / "outputs" / "shap_decision_explanations.csv"
DEFAULT_TARGET_RENOVATION_CONDITION = 5


st.set_page_config(
    page_title="Ingatlanérték becslés",
    layout="wide",
)


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    if not DECISION_DATASET_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(DECISION_DATASET_PATH, sep=";", encoding="utf-8-sig")


def load_ksh_reference_data() -> pd.DataFrame:
    try:
        return read_table(KSH_AVG_PRICES_TABLE)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_conditions_reference_data() -> pd.DataFrame:
    try:
        return read_table(CONDITIONS_TABLE).sort_values("Kulcs")
    except Exception:
        return pd.DataFrame(
            {
                "Kulcs": [1, 2, 3, 4, 5],
                "Leírás": [
                    "erősen hibás, nem javítható állapot",
                    "erősen hibás, javítható állapot",
                    "kisebb hibákkal terhelt állapot",
                    "jó állapot",
                    "újszerű vagy károsodásmentes állapot",
                ],
            }
        )


@st.cache_data
def load_shap_data() -> pd.DataFrame:
    if not SHAP_EXPLANATIONS_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(SHAP_EXPLANATIONS_PATH, sep=";", encoding="utf-8-sig")


def unique_values(df: pd.DataFrame, column: str, fallback: list[str]) -> list[str]:
    if df.empty or column not in df:
        return fallback

    values = sorted(df[column].dropna().astype(str).unique().tolist())
    return values or fallback


def unique_values_with_required(
    df: pd.DataFrame,
    column: str,
    required_values: list[str],
) -> list[str]:
    values = unique_values(df, column, required_values)
    return sorted(set(values) | set(required_values))


def normalize_location(value) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" vármegye", "")
        .replace(" varmegye", "")
    )


def filtered_unique_values(
    df: pd.DataFrame,
    column: str,
    filters: dict[str, str],
    fallback: list[str],
) -> list[str]:
    if df.empty or column not in df:
        return fallback

    filtered_df = df.copy()
    for filter_column, filter_value in filters.items():
        if filter_column in filtered_df and filter_value:
            filtered_df = filtered_df[
                filtered_df[filter_column].map(normalize_location)
                == normalize_location(filter_value)
            ]

    return unique_values(filtered_df, column, fallback)


def first_filtered_value(
    df: pd.DataFrame,
    column: str,
    filters: dict[str, str],
    fallback: str,
) -> str:
    values = filtered_unique_values(df, column, filters, [fallback])
    return values[0] if values else fallback


def is_apartment_property(property_type: str) -> bool:
    return normalize_location(property_type) == "lakás"


def ksh_property_options_for_location(
    ksh_reference_df: pd.DataFrame,
    county: str,
    city: str,
    settlement_type: str,
) -> list[str]:
    if ksh_reference_df.empty:
        return []

    location_rows = ksh_reference_df.copy()
    for column, value in {
        "county": county,
        "city": city,
        "settlement_type": settlement_type,
    }.items():
        if column in location_rows:
            location_rows = location_rows[
                location_rows[column].map(normalize_location) == normalize_location(value)
            ]

    property_options = []
    if (
        "apartment_price_m2" in location_rows
        and location_rows["apartment_price_m2"].fillna(0).gt(0).any()
    ):
        property_options.append("Lakás")
    if (
        "house_price_m2" in location_rows
        and location_rows["house_price_m2"].fillna(0).gt(0).any()
    ):
        property_options.append("Lakóház")

    if property_options:
        return property_options
    if "property_type" in location_rows:
        return unique_values(location_rows, "property_type", [])
    return []


def condition_options(conditions_df: pd.DataFrame) -> dict[str, int]:
    options = {}
    valid_conditions = conditions_df[
        (conditions_df["Kulcs"] >= 2) & (conditions_df["Kulcs"] <= 5)
    ].copy()
    for _, row in valid_conditions.iterrows():
        key = int(row["Kulcs"])
        label = f"{key} - {row['Leírás']}"
        options[label] = key
    return options


def format_huf(value) -> str:
    try:
        return f"{float(value):,.0f} Ft".replace(",", " ")
    except (TypeError, ValueError):
        return "n/a"


def format_ratio(value) -> str:
    try:
        return f"{float(value):.2f}x"
    except (TypeError, ValueError):
        return "n/a"


def parse_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value) or value == "":
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return [str(value)]
    return parsed if isinstance(parsed, list) else [parsed]


def render_input_form(reference_df: pd.DataFrame) -> dict:
    st.sidebar.header("Ingatlan adatai")

    ksh_reference_df = load_ksh_reference_data()
    conditions_df = load_conditions_reference_data()
    location_df = ksh_reference_df if not ksh_reference_df.empty else reference_df
    condition_label_to_key = condition_options(conditions_df)

    county_options = unique_values(location_df, "county", ["Pest"])
    property_options = unique_values_with_required(
        reference_df,
        "property_type",
        ["Lakóház", "Lakás"],
    )

    with st.sidebar.form("property_input_form"):
        county = st.selectbox("Vármegye", county_options, index=0)
        city_options = filtered_unique_values(
            location_df,
            "city",
            {"county": county},
            ["Budapest"],
        )
        city = st.selectbox("Település", city_options, index=0)
        settlement_type = first_filtered_value(
            location_df,
            "settlement_type",
            {"county": county, "city": city},
            "Város",
        )
        st.text_input(
            "Településtípus",
            value=settlement_type,
            disabled=True,
        )
        property_type = st.selectbox("Ingatlantípus", property_options, index=0)

        land_area_m2 = st.number_input(
            "Telek mérete (m2)",
            min_value=0.0,
            max_value=10000.0,
            value=800.0,
            step=10.0,
        )
        building_area_m2 = st.number_input(
            "Épület/lakás alapterület (m2)",
            min_value=1.0,
            max_value=1000.0,
            value=100.0,
            step=1.0,
        )
        condition_labels = list(condition_label_to_key.keys())
        default_condition_index = next(
            (
                index
                for index, label in enumerate(condition_labels)
                if condition_label_to_key[label] == 3
            ),
            0,
        )
        selected_condition_label = st.selectbox(
            "Aktuális állapot",
            condition_labels,
            index=default_condition_index,
        )
        condition = condition_label_to_key[selected_condition_label]
        target_condition_labels = [
            label
            for label in condition_labels
            if condition_label_to_key[label] > condition
        ]
        if not target_condition_labels:
            target_condition_labels = [selected_condition_label]
        default_target_index = next(
            (
                index
                for index, label in enumerate(target_condition_labels)
                if condition_label_to_key[label] == DEFAULT_TARGET_RENOVATION_CONDITION
            ),
            len(target_condition_labels) - 1,
        )
        selected_target_condition_label = st.selectbox(
            "Felújítás célállapota",
            target_condition_labels,
            index=default_target_index,
        )
        target_condition = condition_label_to_key[selected_target_condition_label]
        submitted = st.form_submit_button("Értékbecslés futtatása", type="primary")

    return {
        "submitted": submitted,
        "city": city,
        "county": county,
        "settlement_type": settlement_type,
        "property_type": property_type,
        "land_area_m2": land_area_m2,
        "building_area_m2": building_area_m2,
        "condition": condition,
        "target_condition": target_condition,
    }


def render_dynamic_input_form(reference_df: pd.DataFrame) -> dict:
    st.sidebar.header("Ingatlan adatai")

    ksh_reference_df = load_ksh_reference_data()
    conditions_df = load_conditions_reference_data()
    location_df = ksh_reference_df if not ksh_reference_df.empty else reference_df
    condition_label_to_key = condition_options(conditions_df)

    county_options = unique_values(location_df, "county", ["Pest"])
    county = st.sidebar.selectbox("Vármegye", county_options, index=0, key="dynamic_county")
    city_options = filtered_unique_values(
        location_df,
        "city",
        {"county": county},
        ["Budapest"],
    )
    if st.session_state.get("dynamic_city") not in city_options:
        st.session_state["dynamic_city"] = city_options[0]
    city = st.sidebar.selectbox("Település", city_options, key="dynamic_city")

    settlement_type = first_filtered_value(
        location_df,
        "settlement_type",
        {"county": county, "city": city},
        "Város",
    )
    st.sidebar.text_input("Településtípus", value=settlement_type, disabled=True)

    property_options = ksh_property_options_for_location(
        ksh_reference_df,
        county,
        city,
        settlement_type,
    )
    if not property_options:
        property_options = unique_values(reference_df, "property_type", ["Lakóház"])
    if st.session_state.get("dynamic_property_type") not in property_options:
        st.session_state["dynamic_property_type"] = property_options[0]

    property_type = st.sidebar.selectbox(
        "Ingatlantípus",
        property_options,
        key="dynamic_property_type",
    )
    is_apartment = is_apartment_property(property_type)
    if is_apartment:
        st.session_state["dynamic_land_area_m2"] = 0.0
    land_area_m2 = st.sidebar.number_input(
        "Telek mérete (m2)",
        min_value=0.0,
        max_value=10000.0,
        value=0.0 if is_apartment else 800.0,
        step=10.0,
        key="dynamic_land_area_m2",
        disabled=is_apartment,
    )
    building_area_m2 = st.sidebar.number_input(
        "Épület/lakás alapterület (m2)",
        min_value=1.0,
        max_value=1000.0,
        value=100.0,
        step=1.0,
        key="dynamic_building_area_m2",
    )

    condition_labels = list(condition_label_to_key.keys())
    default_condition_index = next(
        (
            index
            for index, label in enumerate(condition_labels)
            if condition_label_to_key[label] == 3
        ),
        0,
    )
    selected_condition_label = st.sidebar.selectbox(
        "Aktuális állapot",
        condition_labels,
        index=default_condition_index,
        key="dynamic_condition",
    )
    condition = condition_label_to_key[selected_condition_label]

    target_condition_labels = [
        label
        for label in condition_labels
        if condition_label_to_key[label] > condition
    ]
    if not target_condition_labels:
        target_condition_labels = [selected_condition_label]
    if st.session_state.get("dynamic_target_condition") not in target_condition_labels:
        default_target_index = next(
            (
                index
                for index, label in enumerate(target_condition_labels)
                if condition_label_to_key[label] == DEFAULT_TARGET_RENOVATION_CONDITION
            ),
            len(target_condition_labels) - 1,
        )
        st.session_state["dynamic_target_condition"] = target_condition_labels[
            default_target_index
        ]
    selected_target_condition_label = st.sidebar.selectbox(
        "Felújítás célállapota",
        target_condition_labels,
        key="dynamic_target_condition",
    )
    target_condition = condition_label_to_key[selected_target_condition_label]

    return {
        "submitted": True,
        "city": city,
        "county": county,
        "settlement_type": settlement_type,
        "property_type": property_type,
        "land_area_m2": land_area_m2,
        "building_area_m2": building_area_m2,
        "condition": condition,
        "target_condition": target_condition,
    }


def predict_user_property(inputs: dict) -> pd.Series:
    current_prediction = predict_property_value(
        city=inputs["city"],
        county=inputs["county"],
        settlement_type=inputs["settlement_type"],
        property_type=inputs["property_type"],
        land_area_m2=inputs["land_area_m2"],
        building_area_m2=inputs["building_area_m2"],
        condition=inputs["condition"],
        activation_year=None,
    )
    renovation_result = calculate_renovation_cost(
        current_condition=inputs["condition"],
        target_condition=inputs["target_condition"],
        building_area_m2=inputs["building_area_m2"],
    )
    renovated_prediction = predict_property_value(
        city=inputs["city"],
        county=inputs["county"],
        settlement_type=inputs["settlement_type"],
        property_type=inputs["property_type"],
        land_area_m2=inputs["land_area_m2"],
        building_area_m2=inputs["building_area_m2"],
        condition=inputs["target_condition"],
        activation_year=None,
    )

    row = {
        "property_id": "USER_INPUT",
        **{key: value for key, value in inputs.items() if key != "submitted"},
        **current_prediction,
        **renovation_result,
        "renovated_market_value": renovated_prediction["predicted_market_value"],
        "renovated_ksh_price_m2": renovated_prediction["ksh_price_m2"],
        "renovated_ksh_baseline_value_huf": renovated_prediction[
            "ksh_baseline_value_huf"
        ],
        "renovated_land_structural_score": renovated_prediction[
            "predicted_land_structural_score"
        ],
        "renovated_building_structural_score": renovated_prediction[
            "predicted_building_structural_score"
        ],
        "renovated_structural_score": renovated_prediction[
            "predicted_structural_score"
        ],
        "renovated_adjustment_factor": renovated_prediction["adjustment_factor"],
        "renovated_benchmark_delta": renovated_prediction["benchmark_delta"],
        "renovated_ksh_source_level": renovated_prediction["ksh_source_level"],
    }

    decision_df = pd.DataFrame([row])
    decision_df = apply_minimum_renovation_uplift(
        decision_df,
        inputs["target_condition"],
    )
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

    return decision_df.iloc[0]


def render_metrics(row: pd.Series) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Becsült piaci érték", format_huf(row.get("predicted_market_value")))
    col2.metric("KSH piaci benchmark", format_huf(row.get("ksh_baseline_value_huf")))
    col3.metric("Felújítás utáni érték", format_huf(row.get("renovated_market_value")))

    col4, col5, col6 = st.columns(3)
    col4.metric("Strukturális score", format_ratio(row.get("predicted_structural_score")))
    col5.metric("KSH korrekciós faktor", format_ratio(row.get("adjustment_factor")))
    col6.metric("Benchmark eltérés", format_huf(row.get("benchmark_delta")))

    col7, _ = st.columns([1, 2])
    col7.metric("Értéknövekmény", format_huf(row.get("value_uplift")))


def render_property_details(row: pd.Series) -> None:
    conditions_df = load_conditions_reference_data()
    condition_descriptions = {
        int(item["Kulcs"]): item["Leírás"]
        for _, item in conditions_df.iterrows()
    }
    condition = row.get("condition")
    target_condition = row.get("target_condition")
    condition_label = (
        f"{int(condition)} - {condition_descriptions.get(int(condition), '')}"
        if pd.notna(condition)
        else "n/a"
    )
    target_condition_label = (
        f"{int(target_condition)} - {condition_descriptions.get(int(target_condition), '')}"
        if pd.notna(target_condition)
        else "n/a"
    )
    details = {
        "Település": row.get("city"),
        "Vármegye": row.get("county"),
        "Településtípus": row.get("settlement_type"),
        "Ingatlantípus": row.get("property_type"),
        "Telek mérete": f"{row.get('land_area_m2', 0):,.0f} m2".replace(",", " "),
        "Épület/lakás alapterület": f"{row.get('building_area_m2', 0):,.0f} m2".replace(",", " "),
        "Állapot": condition_label,
        "Felújítás célállapota": target_condition_label,
        "KSH forrásszint": row.get("ksh_source_level"),
    }
    st.dataframe(
        pd.DataFrame(details.items(), columns=["Mező", "Érték"]),
        use_container_width=True,
    )


def render_renovation(row: pd.Series) -> None:
    works = parse_list(row.get("works"))
    package_ids = parse_list(row.get("package_ids"))

    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("Felújítás lehetséges:", bool(row.get("renovation_possible", True)))
        st.write("Csomagok:", ", ".join(map(str, package_ids)) if package_ids else "nincs")
        st.write("ROI:", format_ratio(row.get("roi")))
    with col2:
        if works:
            st.write(pd.DataFrame({"Munkák": works}))
        else:
            st.write("Ehhez az ingatlanhoz nincs szükséges felújítási munkacsomag.")


def render_shap(row: pd.Series, shap_df: pd.DataFrame) -> pd.DataFrame:
    selected = pd.DataFrame()
    if not shap_df.empty and "property_id" in shap_df:
        selected = shap_df[shap_df["property_id"].astype(str) == str(row.get("property_id"))]

    if selected.empty:
        try:
            from valuation.shap_explain import generate_shap_decision_explanations
        except ModuleNotFoundError:
            generate_shap_decision_explanations = None

        if generate_shap_decision_explanations is not None:
            with st.spinner("Lokális SHAP magyarázat készül..."):
                try:
                    selected = generate_shap_decision_explanations(
                        pd.DataFrame([row.to_dict()]),
                        target_condition=int(row.get("target_condition", DEFAULT_TARGET_RENOVATION_CONDITION)),
                        top_n=3,
                    )
                except Exception as error:
                    selected = build_fallback_shap_rows(row)

    if selected.empty:
        return build_fallback_shap_rows(row)

    return selected


def build_fallback_shap_rows(row: pd.Series) -> pd.DataFrame:
    """Small local explanation proxy when the SHAP package cannot run."""

    baseline = float(row.get("ksh_baseline_value_huf", 0) or 0)
    score = float(row.get("predicted_structural_score", 0.5) or 0.5)
    adjustment_factor = float(row.get("adjustment_factor", 1.0) or 1.0)
    structural_effect = (adjustment_factor - 1.0) / 0.40

    rows = [
        {
            "property_id": row.get("property_id", "USER_INPUT"),
            "scenario": "current",
            "feature": "predicted_structural_score",
            "feature_value": round(score, 3),
            "shap_value": structural_effect,
            "effect": "increase" if structural_effect >= 0 else "decrease",
        },
        {
            "property_id": row.get("property_id", "USER_INPUT"),
            "scenario": "current",
            "feature": "building_area_m2",
            "feature_value": row.get("building_area_m2", "n/a"),
            "shap_value": max(min((float(row.get("building_area_m2", 0) or 0) - 80) / 800, 0.08), -0.08),
            "effect": "increase" if float(row.get("building_area_m2", 0) or 0) >= 80 else "decrease",
        },
    ]

    if str(row.get("property_type", "")).lower() == "lakóház":
        land_area = float(row.get("land_area_m2", 0) or 0)
        rows.append(
            {
                "property_id": row.get("property_id", "USER_INPUT"),
                "scenario": "current",
                "feature": "land_area_m2",
                "feature_value": row.get("land_area_m2", "n/a"),
                "shap_value": max(min((land_area - 600) / 10000, 0.08), -0.08),
                "effect": "increase" if land_area >= 600 else "decrease",
            }
        )

    return pd.DataFrame(rows)


def build_current_explanation(row: pd.Series, selected_shap: pd.DataFrame) -> str:
    import rag.explainer as rag_explainer

    rag_explainer = importlib.reload(rag_explainer)
    return rag_explainer.build_explanation(row.to_dict(), selected_shap)


def main() -> None:
    st.title("MI-alapú ingatlanérték-becslés")

    reference_df = load_reference_data()
    shap_df = load_shap_data()
    inputs = render_dynamic_input_form(reference_df)

    if "prediction_row" not in st.session_state or inputs["submitted"]:
        st.session_state["prediction_row"] = predict_user_property(inputs)

    row = st.session_state["prediction_row"]
    render_metrics(row)

    st.subheader("Ingatlan adatai")
    render_property_details(row)

    selected_shap = render_shap(row, shap_df)
    st.subheader("RAG-alapú szöveges magyarázat")
    st.markdown(build_current_explanation(row, selected_shap))

    with st.expander("Technikai adatok"):
        st.dataframe(pd.DataFrame([row.to_dict()]), use_container_width=True)


if __name__ == "__main__":
    main()
