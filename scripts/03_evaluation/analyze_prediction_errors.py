from pathlib import Path
import re
import warnings

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "outputs" / "evaluation" / "error_analysis"
MANUSCRIPT_DIR = OUTPUT_DIR / "manuscript_core"
ARCHIVE_DIR = OUTPUT_DIR / "all_models_archive"
MISSING_MODELS_FILE = OUTPUT_DIR / "missing_models.csv"

PHYSICS_DIR = ROOT / "outputs" / "physics_baseline"
ML_ONLY_DIR = ROOT / "outputs" / "ml_only"
PIML_MANUAL_DIR = ROOT / "outputs" / "piml_manual"
PIML_DNN_DIR = ROOT / "outputs" / "piml_dnn"
AUTOML_DIR = ROOT / "outputs" / "automl"

ABRAMS_MANUAL_FILE = PIML_MANUAL_DIR / "abrams_guided_manual_predictions.csv"
MULTIVARIABLE_MANUAL_FILE = (
    PIML_MANUAL_DIR / "multivariable_guided_manual_predictions.csv"
)

DPI = 600

MODEL_ORDER = [
    "Classical Abrams",
    "Abrams",
    "Multivariable Abrams",
    "MLR",
    "SVR",
    "KNN",
    "RF",
    "GBR",
    "XGBoost",
    "CatBoost",
    "DNN",
    "AutoGluon",
    "Abrams-MLR",
    "Abrams-SVR",
    "Abrams-KNN",
    "Abrams-RF",
    "Abrams-GBR",
    "Abrams-XGBoost",
    "Abrams-CatBoost",
    "Abrams-DNN",
    "Abrams-AutoGluon",
    "Multivariable-MLR",
    "Multivariable-SVR",
    "Multivariable-KNN",
    "Multivariable-RF",
    "Multivariable-GBR",
    "Multivariable-XGBoost",
    "Multivariable-CatBoost",
    "Multivariable-DNN",
    "Multivariable-AutoGluon",
]

MANUSCRIPT_MODELS = {
    "Classical Abrams",
    "Abrams",
    "Multivariable Abrams",
    "Abrams-XGBoost",
    "Multivariable-XGBoost",
    "Abrams-CatBoost",
    "Multivariable-CatBoost",
}

SEPARATE_FILE_MODELS = {
    "Classical Abrams": (
        PHYSICS_DIR / "classical_abrams_train_predictions.csv",
        PHYSICS_DIR / "classical_abrams_test_predictions.csv",
    ),
    "Abrams": (
        PHYSICS_DIR / "physics_prior_wb_age_train_predictions.csv",
        PHYSICS_DIR / "physics_prior_wb_age_test_predictions.csv",
    ),
    "Multivariable Abrams": (
        PHYSICS_DIR / "multivariable_empirical_prior_train_predictions.csv",
        PHYSICS_DIR / "multivariable_empirical_prior_test_predictions.csv",
    ),
    "MLR": (
        ML_ONLY_DIR / "mlr_train_predictions.csv",
        ML_ONLY_DIR / "mlr_test_predictions.csv",
    ),
    "SVR": (
        ML_ONLY_DIR / "svr_train_predictions.csv",
        ML_ONLY_DIR / "svr_test_predictions.csv",
    ),
    "KNN": (
        ML_ONLY_DIR / "knn_train_predictions.csv",
        ML_ONLY_DIR / "knn_test_predictions.csv",
    ),
    "RF": (
        ML_ONLY_DIR / "rf_train_predictions.csv",
        ML_ONLY_DIR / "rf_test_predictions.csv",
    ),
    "GBR": (
        ML_ONLY_DIR / "gbr_train_predictions.csv",
        ML_ONLY_DIR / "gbr_test_predictions.csv",
    ),
    "XGBoost": (
        ML_ONLY_DIR / "xgboost_train_predictions.csv",
        ML_ONLY_DIR / "xgboost_test_predictions.csv",
    ),
    "CatBoost": (
        ML_ONLY_DIR / "catboost_train_predictions.csv",
        ML_ONLY_DIR / "catboost_test_predictions.csv",
    ),
    "DNN": (
        ML_ONLY_DIR / "dnn_train_predictions.csv",
        ML_ONLY_DIR / "dnn_test_predictions.csv",
    ),
    "AutoGluon": (
        AUTOML_DIR / "autogluon_train_predictions.csv",
        AUTOML_DIR / "autogluon_test_predictions.csv",
    ),
    "Abrams-DNN": (
        PIML_DNN_DIR / "abrams_guided_dnn_train_predictions.csv",
        PIML_DNN_DIR / "abrams_guided_dnn_test_predictions.csv",
    ),
    "Abrams-AutoGluon": (
        AUTOML_DIR / "abrams_guided_automl_train_predictions.csv",
        AUTOML_DIR / "abrams_guided_automl_test_predictions.csv",
    ),
    "Multivariable-DNN": (
        PIML_DNN_DIR / "multivariable_guided_dnn_train_predictions.csv",
        PIML_DNN_DIR / "multivariable_guided_dnn_test_predictions.csv",
    ),
    "Multivariable-AutoGluon": (
        AUTOML_DIR / "multivariable_guided_automl_train_predictions.csv",
        AUTOML_DIR / "multivariable_guided_automl_test_predictions.csv",
    ),
}

ABRAMS_MANUAL_MODELS = {
    "Abrams-MLR": ["AbramsGuided_MLR"],
    "Abrams-SVR": ["AbramsGuided_SVR"],
    "Abrams-KNN": ["AbramsGuided_KNN"],
    "Abrams-RF": ["AbramsGuided_RF"],
    "Abrams-GBR": ["AbramsGuided_GBR"],
    "Abrams-XGBoost": ["AbramsGuided_XGBoost"],
    "Abrams-CatBoost": ["AbramsGuided_CatBoost"],
}

MULTIVARIABLE_MANUAL_MODELS = {
    "Multivariable-MLR": ["MultivariableGuided_MLR"],
    "Multivariable-SVR": ["MultivariableGuided_SVR"],
    "Multivariable-KNN": ["MultivariableGuided_KNN"],
    "Multivariable-RF": ["MultivariableGuided_RF"],
    "Multivariable-GBR": ["MultivariableGuided_GBR"],
    "Multivariable-XGBoost": ["MultivariableGuided_XGBoost"],
    "Multivariable-CatBoost": ["MultivariableGuided_CatBoost"],
}

PREDICTION_COLUMNS = [
    "fc_pred_abrams_guided_automl",
    "fc_pred_multivariable_guided_automl",
    "fc_pred_abrams_guided_dnn",
    "fc_pred_multivariable_guided_dnn",
    "fc_pred_physics_prior_wb_age",
    "fc_pred_multivariable_empirical_prior",
    "y_pred_final",
    "y_pred",
    "prediction",
    "predicted",
    "pred",
]

TRAIN_COLOR = "#16324f"
TEST_COLOR = "#c95a1e"
ZERO_LINE_COLOR = "#8b1e3f"


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def normalize_text(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def find_column(data, candidates):
    normalized = {column.lower(): column for column in data.columns}
    return next(
        (normalized[candidate] for candidate in candidates if candidate in normalized),
        None,
    )


def find_actual_prediction_columns(data):
    actual_column = find_column(
        data,
        ["y_true", "actual", "true", "target", "fc_mpa", "fc"],
    )
    if actual_column is None:
        raise ValueError(f"Actual-strength column not found in {list(data.columns)}")

    prediction_column = find_column(data, PREDICTION_COLUMNS)
    if prediction_column is None:
        candidates = [
            column for column in data.columns if column.lower().startswith("fc_pred")
        ]
        if candidates:
            prediction_column = sorted(candidates, key=len)[0]

    if prediction_column is None:
        raise ValueError(f"Prediction column not found in {list(data.columns)}")

    return actual_column, prediction_column


def extract_predictions(data):
    actual_column, prediction_column = find_actual_prediction_columns(data)
    actual = pd.to_numeric(data[actual_column], errors="coerce").values
    prediction = pd.to_numeric(data[prediction_column], errors="coerce").values
    valid = np.isfinite(actual) & np.isfinite(prediction)

    if not valid.any():
        raise ValueError("No finite actual/predicted pairs were found.")

    return actual[valid], prediction[valid]


def load_prediction_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path.relative_to(ROOT).as_posix()}"
        )

    return extract_predictions(pd.read_csv(path))


def load_manual_predictions(path, aliases, split):
    if not path.exists():
        raise FileNotFoundError(
            f"Combined prediction file not found: {path.relative_to(ROOT).as_posix()}"
        )

    data = pd.read_csv(path)
    split_column = find_column(data, ["split", "dataset", "subset", "partition"])
    model_column = find_column(
        data,
        ["model", "model_name", "base_model", "ml_model", "algorithm", "estimator"],
    )
    if split_column is None or model_column is None:
        raise ValueError(f"Dataset/Model columns not found in {path.name}")

    normalized_aliases = {normalize_text(alias) for alias in aliases}
    selected = data[
        (data[split_column].apply(normalize_text) == normalize_text(split))
        & (data[model_column].apply(normalize_text).isin(normalized_aliases))
    ].copy()
    if selected.empty:
        raise ValueError(f"No {split} rows for {aliases[0]} in {path.name}")

    return extract_predictions(selected)


def load_model_predictions(model_name):
    if model_name in SEPARATE_FILE_MODELS:
        train_file, test_file = SEPARATE_FILE_MODELS[model_name]
        y_train, train_prediction = load_prediction_file(train_file)
        y_test, test_prediction = load_prediction_file(test_file)
        return y_train, train_prediction, y_test, test_prediction

    if model_name in ABRAMS_MANUAL_MODELS:
        aliases = ABRAMS_MANUAL_MODELS[model_name]
        y_train, train_prediction = load_manual_predictions(
            ABRAMS_MANUAL_FILE,
            aliases,
            "Train",
        )
        y_test, test_prediction = load_manual_predictions(
            ABRAMS_MANUAL_FILE,
            aliases,
            "Test",
        )
        return y_train, train_prediction, y_test, test_prediction

    if model_name in MULTIVARIABLE_MANUAL_MODELS:
        aliases = MULTIVARIABLE_MANUAL_MODELS[model_name]
        y_train, train_prediction = load_manual_predictions(
            MULTIVARIABLE_MANUAL_FILE,
            aliases,
            "Train",
        )
        y_test, test_prediction = load_manual_predictions(
            MULTIVARIABLE_MANUAL_FILE,
            aliases,
            "Test",
        )
        return y_train, train_prediction, y_test, test_prediction

    raise ValueError(f"No prediction source is configured for {model_name}")


def style_axes(axis):
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_color("black")

    axis.grid(True, linestyle="-", alpha=0.18)
    axis.tick_params(axis="both", labelsize=10)


def plot_model_error(
    model_name,
    y_train,
    train_prediction,
    y_test,
    test_prediction,
    output_dir,
):
    train_error = y_train - train_prediction
    test_error = y_test - test_prediction

    figure, axis = plt.subplots(figsize=(8.2, 5.1))
    axis.scatter(
        np.arange(len(train_error)),
        train_error,
        s=14,
        facecolors="none",
        edgecolors=TRAIN_COLOR,
        linewidths=0.7,
        alpha=0.85,
        label="Training data",
    )
    axis.scatter(
        np.arange(len(test_error)),
        test_error,
        s=18,
        facecolors=TEST_COLOR,
        edgecolors="black",
        linewidths=0.2,
        alpha=0.80,
        label="Testing data",
    )
    axis.axhline(0, color=ZERO_LINE_COLOR, linestyle="--", linewidth=1.4)
    axis.set_xlabel("Data sample index", fontsize=11)
    axis.set_ylabel("Prediction error (MPa)", fontsize=11)
    axis.set_title(
        f"Combined error analysis: {model_name}",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    style_axes(axis)
    axis.legend(loc="upper right", fontsize=9, frameon=True)

    output_file = output_dir / f"{safe_name(model_name)}_error_combined_600dpi.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def main():
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    missing_rows = []

    for model_name in MODEL_ORDER:
        try:
            y_train, train_prediction, y_test, test_prediction = (
                load_model_predictions(model_name)
            )
            plot_model_error(
                model_name,
                y_train,
                train_prediction,
                y_test,
                test_prediction,
                ARCHIVE_DIR,
            )

            if model_name in MANUSCRIPT_MODELS:
                plot_model_error(
                    model_name,
                    y_train,
                    train_prediction,
                    y_test,
                    test_prediction,
                    MANUSCRIPT_DIR,
                )

            print(f"Completed: {model_name}")
        except (FileNotFoundError, ValueError) as error:
            missing_rows.append({"Model": model_name, "Reason": str(error)})
            print(f"Skipped {model_name}: {error}")

    missing = pd.DataFrame(missing_rows, columns=["Model", "Reason"])
    if not missing.empty:
        order = {model: index for index, model in enumerate(MODEL_ORDER)}
        missing["Order"] = missing["Model"].map(order)
        missing = missing.sort_values("Order").drop(columns="Order")
    missing.to_csv(MISSING_MODELS_FILE, index=False)

    print(f"Error analysis saved to {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
