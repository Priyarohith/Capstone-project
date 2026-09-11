from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "titanic.csv"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def save_hist_boxplots(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(df["age"], bins=20, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Age Distribution")
    sns.boxplot(x=df["age"], ax=axes[0, 1])
    axes[0, 1].set_title("Age Boxplot")
    sns.histplot(df["fare"], bins=20, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("Fare Distribution")
    sns.boxplot(x=df["fare"], ax=axes[1, 1])
    axes[1, 1].set_title("Fare Boxplot")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "age_fare_distributions.png", dpi=200)
    plt.close()


def compute_iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return len(outliers), lower, upper


def main():
    df = sns.load_dataset("titanic")
    print("DF INFO")
    print(df.info())
    print("\nDF DESCRIBE")
    print(df.describe(include="all").to_string())
    print("\nDF SHAPE")
    print(df.shape)

    missing_rates = df.isna().mean().sort_values(ascending=False)
    print("\nMissing value percentages:\n", missing_rates[missing_rates > 0].to_string())

    df.to_csv(DATA_PATH, index=False)
    print(f"\nSaved offline fallback csv to: {DATA_PATH}")

    # threshold-based strategy
    cleaning_decision = {}
    missing_rates_named = df.isna().mean() * 100
    for col, rate in missing_rates_named.items():
        if not pd.isna(rate) and rate > 0:
            if rate < 5:
                cleaning_decision[col] = "drop rows"
            elif rate <= 30:
                cleaning_decision[col] = "impute"
            else:
                cleaning_decision[col] = "drop column"

    print("\nMissing-value strategy decisions:")
    for col, decision in cleaning_decision.items():
        print(f"- {col}: {decision} (missing rate = {missing_rates_named[col]:.2f}%)")

    cleaned = df.copy()
    columns_to_drop = [col for col in ["deck", "cabin"] if col in cleaned.columns]
    cleaned = cleaned.drop(columns=columns_to_drop)  # severe missingness; not reliable to impute
    cleaned = cleaned.dropna(subset=["embarked"]).copy()
    cleaned["age"] = cleaned["age"].fillna(cleaned["age"].median())

    # IQR outliers
    age_outliers, age_lower, age_upper = compute_iqr_outliers(cleaned["age"])
    fare_outliers, fare_lower, fare_upper = compute_iqr_outliers(cleaned["fare"])
    print(f"\nAge outliers: {age_outliers} (bounds: [{age_lower:.2f}, {age_upper:.2f}])")
    print(f"Fare outliers: {fare_outliers} (bounds: [{fare_lower:.2f}, {fare_upper:.2f}])")

    fare_mean = cleaned["fare"].mean()
    fare_median = cleaned["fare"].median()
    fare_mode = cleaned["fare"].mode().iloc[0]
    print(f"\nFare summary: mean={fare_mean:.2f}, median={fare_median:.2f}, mode={fare_mode:.2f}")
    if fare_mean > fare_median > fare_mode:
        skewness = "right-skewed"
    elif fare_mean < fare_median < fare_mode:
        skewness = "left-skewed"
    else:
        skewness = "approximately symmetric"
    print(f"Fare distribution is {skewness} because mean > median > mode.")

    # bivariate rates
    sex_survival = cleaned.groupby("sex")["survived"].mean().sort_values(ascending=False)
    pclass_survival = cleaned.groupby("pclass")["survived"].mean().sort_values(ascending=False)
    sex_pclass_survival = cleaned.groupby(["sex", "pclass"])["survived"].mean().unstack()
    print("\nSurvival rate by sex:\n", sex_survival)
    print("\nSurvival rate by pclass:\n", pclass_survival)
    print("\nSurvival rate by sex and pclass:\n", sex_pclass_survival)

    corr_columns = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr_matrix = cleaned[corr_columns].corr()
    print("\nCorrelation matrix:\n", corr_matrix.to_string())

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Titanic feature correlation heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()

    corr_pairs = []
    for i in range(len(corr_columns)):
        for j in range(i + 1, len(corr_columns)):
            corr_pairs.append((corr_columns[i], corr_columns[j], abs(corr_matrix.iloc[i, j])))
    corr_pairs.sort(key=lambda x: x[2], reverse=True)
    top_two = corr_pairs[:2]
    print("\nTop two absolute off-diagonal correlations:")
    for left, right, value in top_two:
        print(f"- {left} vs {right}: {value:.4f}")

    # multivariate charts
    sns.barplot(data=cleaned, x="sex", y="survived", estimator="mean", ci=None)
    plt.title("Survival rate by sex")
    plt.ylabel("Survival rate")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "survival_by_sex.png", dpi=200)
    plt.close()

    sns.barplot(data=cleaned, x="pclass", y="survived", estimator="mean", ci=None)
    plt.title("Survival rate by passenger class")
    plt.ylabel("Survival rate")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "survival_by_pclass.png", dpi=200)
    plt.close()

    sns.boxplot(data=cleaned, x="sex", y="fare")
    plt.title("Fare by sex")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fare_by_sex.png", dpi=200)
    plt.close()

    sns.scatterplot(data=cleaned, x="age", y="fare", hue="survived", alpha=0.6)
    plt.title("Age vs fare by survival")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "age_fare_scatter.png", dpi=200)
    plt.close()

    save_hist_boxplots(cleaned)

    # standardized check
    age_mean = cleaned["age"].mean()
    age_std = cleaned["age"].std(ddof=0)
    fare_mean = cleaned["fare"].mean()
    fare_std = cleaned["fare"].std(ddof=0)
    cleaned["age_z"] = (cleaned["age"] - age_mean) / age_std
    cleaned["fare_z"] = (cleaned["fare"] - fare_mean) / fare_std
    print(f"\nAge z-score summary: mean={cleaned['age_z'].mean():.6f}, std={cleaned['age_z'].std(ddof=0):.6f}")
    print(f"Fare z-score summary: mean={cleaned['fare_z'].mean():.6f}, std={cleaned['fare_z'].std(ddof=0):.6f}")

    plt.figure(figsize=(12, 4))
    sns.histplot(cleaned["age_z"], bins=25, kde=True, color="navy")
    plt.title("Age z-score distribution")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "age_z_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 4))
    sns.histplot(cleaned["fare_z"], bins=25, kde=True, color="darkorange")
    plt.title("Fare z-score distribution")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fare_z_distribution.png", dpi=200)
    plt.close()

    print("\nEDA complete. Saved charts to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
