"""Run the full decision-support pipeline end to end."""

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

try:
    from src.data.load_ksh_social import (
        DB_URL,
        TABLE_NAME as SOCIAL_TABLE,
        load_ksh_social_data,
    )
    from src.decision.topsis import (
        ECONOMIC_CRITERIA,
        SOCIAL_CRITERIA,
        build_dual_topsis_decision,
    )
    from src.valuation.shap_explain import (
        generate_shap_decision_explanations,
        save_shap_decision_explanations,
    )
    from src.valuation.pipeline import generate_decision_dataset, save_decision_dataset
except ImportError:
    from data.load_ksh_social import (
        DB_URL,
        TABLE_NAME as SOCIAL_TABLE,
        load_ksh_social_data,
    )
    from decision.topsis import (
        ECONOMIC_CRITERIA,
        SOCIAL_CRITERIA,
        build_dual_topsis_decision,
    )
    from valuation.shap_explain import (
        generate_shap_decision_explanations,
        save_shap_decision_explanations,
    )
    from valuation.pipeline import generate_decision_dataset, save_decision_dataset


BASE_DIR = Path(__file__).resolve().parents[1]
TOPSIS_OUTPUT_PATH = BASE_DIR / "outputs" / "decision_topsis_scores.csv"
DEFAULT_TARGET_CONDITION = 5


def _prompt_int(prompt, default, minimum=None, maximum=None):
    while True:
        raw_value = input(f"{prompt} [{default}]: ").strip()
        if not raw_value:
            return default

        try:
            value = int(raw_value)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if minimum is not None and value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue

        if maximum is not None and value > maximum:
            print(f"Please enter a value <= {maximum}.")
            continue

        return value


def _prompt_float(prompt, default, minimum=None):
    while True:
        raw_value = input(f"{prompt} [{default}]: ").strip()
        if not raw_value:
            return default

        try:
            value = float(raw_value.replace(",", "."))
        except ValueError:
            print("Please enter a number.")
            continue

        if minimum is not None and value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue

        return value


def _prompt_yes_no(prompt, default=False):
    default_label = "y" if default else "n"

    while True:
        raw_value = input(f"{prompt} [{default_label}]: ").strip().lower()
        if not raw_value:
            return default

        if raw_value in {"y", "yes", "i", "igen"}:
            return True

        if raw_value in {"n", "no", "nem"}:
            return False

        print("Please enter yes or no.")


def _prompt_weights(criteria, title):
    print(f"\n{title}")
    print("Press Enter to keep equal/default weighting.")

    weights = {}
    for criterion, direction in criteria.items():
        weights[criterion] = _prompt_float(
            f"{criterion} ({direction}) weight",
            default=1.0,
            minimum=0,
        )

    return weights


def prompt_pipeline_parameters():
    print("\n=== Full decision-support pipeline parameters ===\n")
    target_condition = _prompt_int(
        "Target renovation condition",
        default=DEFAULT_TARGET_CONDITION,
        minimum=1,
        maximum=5,
    )
    economic_weights = _prompt_weights(
        ECONOMIC_CRITERIA,
        "Economic TOPSIS criterion weights",
    )
    social_weights = _prompt_weights(
        SOCIAL_CRITERIA,
        "Social TOPSIS criterion weights",
    )

    print("\nFinal eco/social weighting")
    eco_weight = _prompt_float("Economic score weight", default=0.5, minimum=0)
    soc_weight = _prompt_float("Social score weight", default=0.5, minimum=0)
    include_shap = _prompt_yes_no(
        "Generate SHAP explanation export",
        default=False,
    )

    return {
        "target_condition": target_condition,
        "economic_weights": economic_weights,
        "social_weights": social_weights,
        "eco_weight": eco_weight,
        "soc_weight": soc_weight,
        "include_shap": include_shap,
    }


def refresh_social_indicators():
    social_df = load_ksh_social_data()
    engine = create_engine(DB_URL)
    social_df.to_sql(
        SOCIAL_TABLE,
        engine,
        if_exists="replace",
        index=False,
    )
    return social_df


def run_full_decision_pipeline(
    target_condition=DEFAULT_TARGET_CONDITION,
    eco_weight=0.5,
    soc_weight=0.5,
    economic_weights=None,
    social_weights=None,
    refresh_social=True,
    reuse_current_predictions=True,
    include_shap=False,
):
    if refresh_social:
        social_df = refresh_social_indicators()
        print(f"Social indicators refreshed: {len(social_df)} rows")
    else:
        engine = create_engine(DB_URL)
        social_df = pd.read_sql(f"SELECT * FROM {SOCIAL_TABLE}", engine)
        print(f"Social indicators loaded: {len(social_df)} rows")

    decision_df = generate_decision_dataset(
        target_condition=target_condition,
        reuse_current_predictions=reuse_current_predictions,
    )
    decision_output_path = save_decision_dataset(decision_df)
    print(f"Decision dataset saved to: {decision_output_path}")

    scored_df = build_dual_topsis_decision(
        decision_df=decision_df,
        social_df=social_df,
        eco_weight=eco_weight,
        soc_weight=soc_weight,
        economic_weights=economic_weights,
        social_weights=social_weights,
    )
    scored_df.to_csv(
        TOPSIS_OUTPUT_PATH,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )
    print(f"TOPSIS scores saved to: {TOPSIS_OUTPUT_PATH}")

    if include_shap:
        explanations_df = generate_shap_decision_explanations(
            decision_df=decision_df,
            target_condition=target_condition,
        )
        shap_output_path = save_shap_decision_explanations(explanations_df)
        print(f"SHAP explanations saved to: {shap_output_path}")

    return scored_df


def main():
    parameters = prompt_pipeline_parameters()
    scored_df = run_full_decision_pipeline(**parameters)
    print("\nFull decision-support pipeline completed!")
    print(scored_df.head())


if __name__ == "__main__":
    main()
