from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "titanic.csv"
MODEL_PATH = ROOT / "best_pipeline.joblib"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def summarize_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    return {
        "confusion_matrix": cm,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "prediction": y_pred,
        "probability": y_prob,
    }


def build_preprocessor(numeric_features=None):
    if numeric_features is None:
        numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
    categorical_features = ["sex", "embarked"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def train_and_compare_models(X_train, X_test, y_train, y_test):
    preprocessor = build_preprocessor()
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=4),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200, max_depth=5),
    }

    results = {}
    for name, estimator in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        metrics = summarize_classification(pipe, X_test, y_test)
        results[name] = metrics

    # Decision tree plot
    tree_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", DecisionTreeClassifier(random_state=42, max_depth=4))])
    tree_pipe.fit(X_train, y_train)
    tree_model = tree_pipe.named_steps["model"]
    feature_names = tree_pipe.named_steps["preprocessor"].get_feature_names_out()
    plt.figure(figsize=(18, 10))
    plot_tree(tree_model, feature_names=feature_names, class_names=["No", "Yes"], filled=True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "decision_tree.png", dpi=200)
    plt.close()

    comparison = pd.DataFrame(
        {
            "Model": list(results.keys()),
            "Accuracy": [results[name]["accuracy"] for name in results],
            "Precision": [results[name]["precision"] for name in results],
            "Recall": [results[name]["recall"] for name in results],
            "F1": [results[name]["f1"] for name in results],
            "AUC": [results[name]["roc_auc"] for name in results],
        }
    )
    print("\nClassification comparison table:\n", comparison.to_string(index=False))

    return results, comparison


def imbalance_comparison(X_train, X_test, y_train, y_test):
    preprocessor = build_preprocessor()
    variants = {}

    baseline = Pipeline(steps=[("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=1000, random_state=42))])
    baseline.fit(X_train, y_train)
    variants["baseline"] = summarize_classification(baseline, X_test, y_test)

    balanced = Pipeline(steps=[("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
    balanced.fit(X_train, y_train)
    variants["balanced"] = summarize_classification(balanced, X_test, y_test)

    smote_pipe = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    smote_pipe.fit(X_train, y_train)
    variants["smote"] = summarize_classification(smote_pipe, X_test, y_test)

    imbalance_df = pd.DataFrame(
        {
            "Variant": ["baseline", "class_weight='balanced'", "SMOTE"],
            "Precision": [variants[v]["precision"] for v in ["baseline", "balanced", "smote"]],
            "Recall": [variants[v]["recall"] for v in ["baseline", "balanced", "smote"]],
            "F1": [variants[v]["f1"] for v in ["baseline", "balanced", "smote"]],
        }
    )
    print("\nImbalance handling comparison:\n", imbalance_df.to_string(index=False))
    return imbalance_df


def tune_random_forest(X_train, y_train):
    preprocessor = build_preprocessor()
    rfc = RandomForestClassifier(oob_score=True, random_state=42)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", rfc)])
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__max_features": ["sqrt", "log2"],
    }
    search = GridSearchCV(pipeline, param_grid=param_grid, cv=3, scoring="f1", n_jobs=-1)
    search.fit(X_train, y_train)
    print("\nBest Random Forest params:", search.best_params_)
    best_oob = search.best_estimator_.named_steps["model"].oob_score_
    print("Best Random Forest OOB score:", best_oob)
    return search


def regression_side_task(df):
    X = df.drop(columns=["survived", "fare"])
    y = df["fare"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = build_preprocessor(numeric_features=["pclass", "age", "sibsp", "parch"])
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    n = len(y_test)
    p = X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    residuals = y_test - preds
    plt.figure(figsize=(8, 6))
    plt.scatter(preds, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted fare")
    plt.ylabel("Residual")
    plt.title("Residual plot for fare regression")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fare_residuals.png", dpi=200)
    plt.close()

    print("\nRegression metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")
    print(f"Adjusted R^2: {adj_r2:.4f}")
    if residuals.abs().mean() > 0:
        print("Residuals do not appear to be random; there is evidence of some heteroscedasticity.")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R^2": r2,
        "Adjusted R^2": adj_r2,
    }


def save_pipeline_and_test(df):
    X = df.drop(columns=["survived"])
    y = df["survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    preprocessor = build_preprocessor()
    best_model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=5, max_features="sqrt")
    full_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", best_model)])
    full_pipeline.fit(X_train, y_train)
    joblib.dump(full_pipeline, MODEL_PATH)

    reloaded = joblib.load(MODEL_PATH)
    sample = X_test.iloc[[0]].copy()
    pred = reloaded.predict(sample)[0]
    print("\nPipeline reload check:")
    print(f"Raw input example predicted survived={pred} using reloaded pipeline.")
    return full_pipeline


def main():
    df = pd.read_csv(DATA_PATH)
    columns_to_drop = [col for col in ["deck", "cabin"] if col in df.columns]
    df = df.drop(columns=columns_to_drop).copy()
    df = df.dropna(subset=["embarked"]).copy()
    df["age"] = df["age"].fillna(df["age"].median())

    X = df.drop(columns=["survived"])
    y = df["survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("\nClass balance:")
    print(y.value_counts(normalize=True).to_string())
    print("Stratified split is used because the target is imbalanced, so each fold preserves the same survival ratio as the full dataset.")

    results, comparison = train_and_compare_models(X_train, X_test, y_train, y_test)
    imbalance_df = imbalance_comparison(X_train, X_test, y_train, y_test)
    search = tune_random_forest(X_train, y_train)
    regression_summary = regression_side_task(df)

    classification_table = comparison
    regression_table = pd.DataFrame(
        {
            "Metric": ["MAE", "RMSE", "R^2", "Adjusted R^2"],
            "Linear Regression": [regression_summary["MAE"], regression_summary["RMSE"], regression_summary["R^2"], regression_summary["Adjusted R^2"]],
        }
    )
    print("\nRegression comparison table:\n", regression_table.to_string(index=False))

    pipeline = save_pipeline_and_test(df)
    print("\nSaved complete pipeline to:", MODEL_PATH)

    best_model_name = comparison.sort_values("F1", ascending=False).iloc[0]["Model"]
    print(f"\nRecommendation: I would deploy {best_model_name} because the highest F1 and AUC indicate the best balance between recall and precision for survival prediction.")


if __name__ == "__main__":
    main()
