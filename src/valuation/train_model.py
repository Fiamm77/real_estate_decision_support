import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    classification_report,
    confusion_matrix
)

# =====================================================
# DATABASE CONNECTION
# =====================================================

engine = create_engine(
    "postgresql+psycopg2://postgres:postgres@localhost:5432/real_estate"
)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_sql(
    "SELECT * FROM properties_synthetic",
    engine
)

print(f"Loaded rows: {len(df)}")

# =====================================================
# TARGET
# =====================================================

TARGET = "building_value"

# =====================================================
# FEATURES
# =====================================================

FEATURES = [
    "land_area_m2",
    "building_area_m2",
    "condition",
    "annual_cost",
    "renovation_cost"
]

# =====================================================
# SELECT FEATURES
# =====================================================

X = df[FEATURES]
y = df[TARGET]

# =====================================================
# FEATURE TYPES
# =====================================================


numerical_features = [
    "land_area_m2",
    "building_area_m2",
    "condition",
    "annual_cost",
    "renovation_cost"
]

# =====================================================
# PREPROCESSING
# =====================================================

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features)
    ]
)

# =====================================================
# MODEL
# =====================================================

model = RandomForestRegressor(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

# =====================================================
# PIPELINE
# =====================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# TRAIN MODEL
# =====================================================

pipeline.fit(X_train, y_train)

# =====================================================
# PREDICTIONS
# =====================================================

predictions = pipeline.predict(X_test)

# =====================================================
# REGRESSION METRICS
# =====================================================

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    predictions
)

mape = mean_absolute_percentage_error(
    y_test,
    predictions
)

print("\n=== REGRESSION METRICS ===\n")

print(f"MAE: {mae:,.0f} Ft")
print(f"RMSE: {rmse:,.0f} Ft")
print(f"R2 Score: {r2:.4f}")
print(f"MAPE: {mape:.4f}")

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(
    pipeline,
    "models/valuation_model.pkl"
)

print("\nModel saved successfully!")

# =====================================================
# CLASSIFICATION EVALUATION
# =====================================================

print(y_test.min(), y_test.max())
print(predictions.min(), predictions.max())

bins = [
    -1_000_000,
    2_000_000,
    5_000_000,
    100_000_000
]

labels = [
    "low",
    "medium",
    "high"
]

# True classes

y_test_classes = pd.cut(
    y_test,
    bins=bins,
    labels=labels
)

y_test_classes = y_test_classes.astype(str)

# Predicted classes

predicted_classes = pd.cut(
    predictions,
    bins=bins,
    labels=labels
)

predicted_classes = predicted_classes.astype(str)

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\n=== CLASSIFICATION REPORT ===\n")

print(
    classification_report(
        y_test_classes,
        predicted_classes
    )
)

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test_classes,
    predicted_classes
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "data/processed/confusion_matrix.png",
    dpi=300
)

print("\nConfusion matrix saved!")

# =====================================================
# PREPROCESS TEST DATA FOR SHAP
# =====================================================

X_test_transformed = pipeline.named_steps[
    "preprocessor"
].transform(X_test)

# =====================================================
# FEATURE NAMES
# =====================================================

feature_names = (
    pipeline.named_steps["preprocessor"]
    .get_feature_names_out()
)

# =====================================================
# SHAP EXPLAINER
# =====================================================

explainer = shap.TreeExplainer(
    pipeline.named_steps["model"]
)

# =====================================================
# CALCULATE SHAP VALUES
# =====================================================

sample_size = 50

shap_values = explainer.shap_values(
    X_test_transformed[:sample_size],
    check_additivity=False
)

# =====================================================
# SHAP SUMMARY PLOT
# =====================================================

plt.figure(figsize=(12, 8))

shap.summary_plot(
    shap_values,
    X_test_transformed[:sample_size],
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

plt.savefig(
    "data/processed/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nSHAP summary plot saved!")

# =====================================================
# SHAP BAR PLOT
# =====================================================

plt.figure(figsize=(10, 6))

shap.summary_plot(
    shap_values,
    X_test_transformed[:sample_size],
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    "data/processed/shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nSHAP feature importance plot saved!")

# =====================================================
# GLOBAL FEATURE IMPORTANCE TABLE
# =====================================================

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": abs(shap_values).mean(axis=0)
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\n=== SHAP FEATURE IMPORTANCE ===\n")

print(importance_df)

# =====================================================
# EXPORT FEATURE IMPORTANCE
# =====================================================

importance_df.to_csv(
    "data/processed/shap_feature_importance.csv",
    index=False
)

print("\nSHAP feature importance exported!")

# =====================================================
# CURRENT VALUE PREDICTIONS
# =====================================================

df["predicted_value"] = pipeline.predict(X)

print("\n=== SAMPLE PREDICTIONS ===\n")

print(
    df[
        [
            "building_value",
            "predicted_value"
        ]
    ].head()
)

# =====================================================
# EXPORT PREDICTIONS
# =====================================================

df.to_csv(
    "data/processed/valuation_predictions.csv",
    index=False
)

print("\nPredictions exported successfully!")