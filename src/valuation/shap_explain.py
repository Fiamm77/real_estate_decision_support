"""Property-level SHAP explanations for valuation predictions."""

import pandas as pd

try:
    import shap
except ModuleNotFoundError:
    shap = None

try:
    from .config import MODEL_PATH, REFERENCE_FEATURES
    from .core import load_model
except ImportError:
    from valuation.config import MODEL_PATH, REFERENCE_FEATURES
    from valuation.core import load_model


DEFAULT_OUTPUT_PATH = "outputs/shap_decision_explanations.csv"

FEATURE_LABELS = {
    "num__land_area_m2": "land_area_m2",
    "num__building_area_m2": "building_area_m2",
    "num__condition": "condition",
}


def _load_pipeline_and_features():
    model_artifact = load_model(MODEL_PATH)
    pipeline = (
        model_artifact["building_model"]
        if isinstance(model_artifact, dict)
        else model_artifact
    )
    features = (
        model_artifact.get("building_features", REFERENCE_FEATURES)
        if isinstance(model_artifact, dict)
        else REFERENCE_FEATURES
    )

    try:
        expected_input_features = list(pipeline.named_steps["preprocessor"].feature_names_in_)
        if expected_input_features:
            features = expected_input_features
    except (AttributeError, KeyError):
        pass

    try:
        pipeline.named_steps["model"].set_params(n_jobs=1)
    except (AttributeError, KeyError, ValueError):
        pass

    return pipeline, list(features)


def _feature_defaults(decision_df):
    defaults = {
        "property_type": "Lakóház",
        "settlement_type": "város",
        "county": "",
        "land_area_m2": 0,
        "building_area_m2": 1,
        "condition": 3,
        "activation_year": 2022,
    }
    if "activation_year" not in decision_df and "activation_year" in defaults:
        defaults["activation_year"] = 2022
    return defaults


def _build_feature_frame(decision_df, features):
    features_df = decision_df.copy()
    defaults = _feature_defaults(features_df)
    for feature in features:
        if feature not in features_df:
            features_df[feature] = defaults.get(feature, 0)
    return features_df[features].copy()


def _build_current_features(decision_df, features):
    return _build_feature_frame(decision_df, features)


def _build_renovated_features(decision_df, target_condition, features):
    features_df = _build_feature_frame(decision_df, features)
    features_df["condition"] = target_condition
    return features_df


def _explain_feature_frame(
    pipeline,
    features_df,
    property_ids,
    scenario,
    top_n,
):
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(features_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = transformed.astype(float)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(
        transformed,
        check_additivity=False,
    )

    explanations = []
    for row_index, property_id in enumerate(property_ids):
        row_values = shap_values[row_index]
        ranked_indexes = abs(row_values).argsort()[::-1][:top_n]

        for feature_index in ranked_indexes:
            feature_name = feature_names[feature_index]
            source_feature = FEATURE_LABELS.get(feature_name, feature_name)
            shap_value = row_values[feature_index]
            feature_value = (
                features_df.iloc[row_index][source_feature]
                if source_feature in features_df.columns
                else ""
            )

            explanations.append(
                {
                    "property_id": property_id,
                    "scenario": scenario,
                    "feature": source_feature,
                    "feature_value": feature_value,
                    "shap_value": shap_value,
                    "effect": "increase" if shap_value >= 0 else "decrease",
                }
            )

    return pd.DataFrame(explanations)


def generate_shap_decision_explanations(
    decision_df,
    target_condition=5,
    top_n=3,
    include_current=True,
    include_renovated=True,
):
    if shap is None:
        return pd.DataFrame()

    pipeline, features = _load_pipeline_and_features()
    property_ids = decision_df["property_id"].tolist()
    explanation_frames = []

    if include_current:
        current_features = _build_current_features(decision_df, features)
        explanation_frames.append(
            _explain_feature_frame(
                pipeline=pipeline,
                features_df=current_features,
                property_ids=property_ids,
                scenario="current",
                top_n=top_n,
            )
        )

    if include_renovated:
        renovated_features = _build_renovated_features(
            decision_df,
            target_condition=target_condition,
            features=features,
        )
        explanation_frames.append(
            _explain_feature_frame(
                pipeline=pipeline,
                features_df=renovated_features,
                property_ids=property_ids,
                scenario="renovated",
                top_n=top_n,
            )
        )

    if not explanation_frames:
        return pd.DataFrame()

    return pd.concat(explanation_frames, ignore_index=True)


def save_shap_decision_explanations(
    explanations_df,
    output_path=DEFAULT_OUTPUT_PATH,
):
    explanations_df.to_csv(
        output_path,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )
    return output_path
