from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PHYSICS_DIR = ROOT / "outputs" / "physics_baseline"
ML_ONLY_DIR = ROOT / "outputs" / "ml_only"
PIML_MANUAL_DIR = ROOT / "outputs" / "piml_manual"
PIML_DNN_DIR = ROOT / "outputs" / "piml_dnn"
AUTOML_DIR = ROOT / "outputs" / "automl"

SUMMARY_DIR = ROOT / "outputs" / "summary"
EVALUATION_DIR = ROOT / "outputs" / "evaluation"

ALL_METRICS_FILE = SUMMARY_DIR / "all_model_metrics_combined.csv"
TRAIN_RANKING_FILE = SUMMARY_DIR / "final_train_ranking.csv"
TEST_RANKING_FILE = SUMMARY_DIR / "final_test_ranking.csv"

EVALUATION_METRICS_FILE = EVALUATION_DIR / "all_model_metrics_summary.csv"
EVALUATION_TEST_RANKING_FILE = EVALUATION_DIR / "all_model_test_ranking.csv"
MISSING_SOURCES_FILE = EVALUATION_DIR / "missing_metrics_sources.csv"

METRIC_SOURCES = [
    PHYSICS_DIR / "classical_abrams_metrics.csv",
    PHYSICS_DIR / "physics_prior_wb_age_metrics.csv",
    PHYSICS_DIR / "multivariable_empirical_prior_metrics.csv",
    ML_ONLY_DIR / "ml_only_manual_metrics.csv",
    ML_ONLY_DIR / "ml_only_dnn_metrics.csv",
    AUTOML_DIR / "ml_only_automl_metrics.csv",
    PIML_MANUAL_DIR / "abrams_guided_manual_metrics.csv",
    PIML_DNN_DIR / "abrams_guided_dnn_metrics.csv",
    AUTOML_DIR / "abrams_guided_automl_metrics.csv",
    PIML_MANUAL_DIR / "multivariable_guided_manual_metrics.csv",
    PIML_DNN_DIR / "multivariable_guided_dnn_metrics.csv",
    AUTOML_DIR / "multivariable_guided_automl_metrics.csv",
]

REQUIRED_COLUMNS = {"Dataset", "Model", "R2", "RMSE", "MAE"}


def assign_category(model_name):
    if model_name == "Classical_Abrams":
        return "Classical Baseline"
    if model_name == "PhysicsPrior_WB_Age":
        return "Physics Prior"
    if model_name == "Multivariable_Empirical_Prior":
        return "Multivariable Empirical Prior"
    if model_name in {
        "MLR",
        "SVR",
        "KNN",
        "RF",
        "GBR",
        "XGBoost",
        "CatBoost",
        "DNN",
        "AutoGluon",
    }:
        return "ML-only Models"
    if model_name.startswith("AbramsGuided_"):
        return "Abrams-Guided Models"
    if model_name.startswith("MultivariableGuided_"):
        return "Multivariable Empirical-Guided Models"

    return "Other"


def assign_category_order(category):
    order = {
        "Classical Baseline": 1,
        "Physics Prior": 2,
        "Multivariable Empirical Prior": 3,
        "ML-only Models": 4,
        "Abrams-Guided Models": 5,
        "Multivariable Empirical-Guided Models": 6,
        "Other": 99,
    }
    return order.get(category, 99)


def read_metric_sources():
    metric_tables = []
    missing_sources = []

    for path in METRIC_SOURCES:
        if not path.exists():
            missing_sources.append(
                {
                    "Source_File": path.name,
                    "Expected_Path": path.relative_to(ROOT).as_posix(),
                }
            )
            continue

        table = pd.read_csv(path)
        missing_columns = REQUIRED_COLUMNS.difference(table.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing columns in {path}: {missing}")

        table["Source_File"] = path.name
        metric_tables.append(table)

    if not metric_tables:
        expected = "\n".join(
            f"  - {path.relative_to(ROOT).as_posix()}" for path in METRIC_SOURCES
        )
        raise FileNotFoundError(
            "No model metrics were found. Run the modeling scripts before the "
            f"evaluation workflow. Expected files:\n{expected}"
        )

    return metric_tables, missing_sources


def prepare_metrics(metric_tables):
    metrics = pd.concat(metric_tables, ignore_index=True)

    numeric_columns = [
        column
        for column in ["R2", "RMSE", "MAE", "MBE", "MAPE", "RRMSE", "A20"]
        if column in metrics.columns
    ]
    for column in numeric_columns:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")

    metrics["Dataset"] = metrics["Dataset"].astype(str).str.strip().str.title()
    metrics["Model"] = metrics["Model"].astype(str).str.strip()

    # Guided scripts also report their empirical prior. Dedicated prior files are
    # listed first and remain the canonical source for duplicate model rows.
    metrics = metrics.drop_duplicates(
        subset=["Dataset", "Model"],
        keep="first",
    )

    metrics["Category"] = metrics["Model"].apply(assign_category)
    metrics["Category_Order"] = metrics["Category"].apply(assign_category_order)

    front_columns = ["Category_Order", "Category", "Dataset", "Model"]
    other_columns = [
        column for column in metrics.columns if column not in front_columns
    ]
    metrics = metrics[front_columns + other_columns]

    return metrics.sort_values(
        by=["Category_Order", "Dataset", "R2"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def prepare_ranking(metrics, dataset):
    return (
        metrics[metrics["Dataset"] == dataset]
        .sort_values(
            by=["R2", "RMSE", "MAE"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )


def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    metric_tables, missing_sources = read_metric_sources()
    metrics = prepare_metrics(metric_tables)
    train_ranking = prepare_ranking(metrics, "Train")
    test_ranking = prepare_ranking(metrics, "Test")

    metrics.to_csv(ALL_METRICS_FILE, index=False)
    metrics.to_csv(EVALUATION_METRICS_FILE, index=False)
    train_ranking.to_csv(TRAIN_RANKING_FILE, index=False)
    test_ranking.to_csv(TEST_RANKING_FILE, index=False)
    test_ranking.to_csv(EVALUATION_TEST_RANKING_FILE, index=False)

    if missing_sources:
        pd.DataFrame(missing_sources).to_csv(MISSING_SOURCES_FILE, index=False)
        print(f"Missing metric sources: {len(missing_sources)}")
        print(f"See {MISSING_SOURCES_FILE.relative_to(ROOT)}")
    elif MISSING_SOURCES_FILE.exists():
        MISSING_SOURCES_FILE.unlink()

    display_columns = [
        column
        for column in ["Category", "Model", "R2", "RMSE", "MAE", "MAPE", "A20"]
        if column in metrics.columns
    ]

    print("All model metrics combined.")
    print("\nTest ranking:")
    print(test_ranking[display_columns])
    print("\nTrain ranking:")
    print(train_ranking[display_columns])


if __name__ == "__main__":
    main()
