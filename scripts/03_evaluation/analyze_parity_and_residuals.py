from pathlib import Path
import re
import warnings

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


matplotlib.use("Agg")
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "outputs" / "evaluation" / "parity_residual"
PARITY_DIR = OUTPUT_DIR / "parity_plots"
RESIDUAL_DIR = OUTPUT_DIR / "residual_plots"
TABLE_DIR = OUTPUT_DIR / "tables"

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

TRAIN_EDGE = "#2F80ED"
TEST_FACE = "#F2994A"
TEST_EDGE = "#B56B1D"
PERFECT_COLOR = "#1F4E79"
PLUS20_COLOR = "#F28E1C"
MINUS20_COLOR = "#18A999"
ZERO_COLOR = "#8E244D"
GRID_ALPHA = 0.22

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 20,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "axes.linewidth": 1.3,
    }
)


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def normalize_text(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def find_actual_prediction_columns(data):
    columns = {column.lower(): column for column in data.columns}

    actual_column = next(
        (
            columns[candidate]
            for candidate in ["y_true", "actual", "true", "target", "fc_mpa", "fc"]
            if candidate in columns
        ),
        None,
    )
    if actual_column is None:
        raise ValueError(f"Actual-strength column not found in {list(data.columns)}")

    prediction_column = next(
        (
            columns[candidate]
            for candidate in PREDICTION_COLUMNS
            if candidate in columns
        ),
        None,
    )
    if prediction_column is None:
        candidates = [
            column for column in data.columns if column.lower().startswith("fc_pred")
        ]
        if candidates:
            prediction_column = sorted(candidates, key=len)[0]

    if prediction_column is None:
        raise ValueError(f"Prediction column not found in {list(data.columns)}")

    return actual_column, prediction_column


def find_column(data, candidates):
    normalized = {column.lower(): column for column in data.columns}
    return next(
        (normalized[candidate] for candidate in candidates if candidate in normalized),
        None,
    )


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


def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def regression_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def style_axes(axis):
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("black")

    axis.grid(True, linestyle="--", alpha=GRID_ALPHA)
    axis.set_axisbelow(True)


def plot_parity(train, test, model_name, output_file):
    test_metrics = regression_metrics(test["Actual"], test["Predicted"])
    values = np.concatenate(
        [
            train["Actual"],
            train["Predicted"],
            test["Actual"],
            test["Predicted"],
        ]
    ).astype(float)
    lower_limit = min(0, np.nanmin(values)) - 2
    upper_limit = np.nanmax(values) + 5

    figure, axis = plt.subplots(figsize=(8.4, 8.0))
    axis.scatter(
        train["Actual"],
        train["Predicted"],
        s=18,
        facecolors="none",
        edgecolors=TRAIN_EDGE,
        linewidths=0.8,
        alpha=0.9,
        label="Training data",
    )
    axis.scatter(
        test["Actual"],
        test["Predicted"],
        s=18,
        facecolors=TEST_FACE,
        edgecolors=TEST_EDGE,
        linewidths=0.5,
        alpha=0.9,
        label="Testing data",
    )

    reference = np.linspace(lower_limit, upper_limit, 200)
    axis.plot(
        reference,
        reference,
        color=PERFECT_COLOR,
        linewidth=2.2,
        label="Perfect prediction",
    )
    axis.plot(
        reference,
        1.2 * reference,
        linestyle="--",
        color=PLUS20_COLOR,
        linewidth=1.7,
        label="+20%",
    )
    axis.plot(
        reference,
        0.8 * reference,
        linestyle="--",
        color=MINUS20_COLOR,
        linewidth=1.7,
        label="-20%",
    )

    axis.set_xlim(lower_limit, upper_limit)
    axis.set_ylim(lower_limit, upper_limit)
    axis.set_title(model_name, pad=12)
    axis.set_xlabel("Actual compressive strength (MPa)")
    axis.set_ylabel("Predicted compressive strength (MPa)")
    style_axes(axis)
    axis.legend(loc="upper left", frameon=True)

    annotation = (
        f"Test $R^2$ = {test_metrics['R2']:.3f}\n"
        f"Test RMSE = {test_metrics['RMSE']:.2f} MPa\n"
        f"Test MAE = {test_metrics['MAE']:.2f} MPa\n"
        f"n(train/test) = {len(train)}/{len(test)}"
    )
    axis.text(
        0.98,
        0.05,
        annotation,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "gray",
        },
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def plot_residual(train, test, model_name, output_file):
    train_residual = train["Actual"] - train["Predicted"]
    test_residual = test["Actual"] - test["Predicted"]

    figure, axis = plt.subplots(figsize=(8.4, 8.0))
    axis.scatter(
        train["Predicted"],
        train_residual,
        s=22,
        facecolors="none",
        edgecolors=TRAIN_EDGE,
        linewidths=0.9,
        alpha=0.9,
        label="Train",
    )
    axis.scatter(
        test["Predicted"],
        test_residual,
        s=22,
        facecolors=TEST_FACE,
        edgecolors=TEST_EDGE,
        linewidths=0.6,
        alpha=0.9,
        label="Test",
    )
    axis.axhline(0, linestyle="--", color=ZERO_COLOR, linewidth=1.8)
    axis.set_title(model_name, pad=12)
    axis.set_xlabel("Predicted compressive strength (MPa)")
    axis.set_ylabel("Residual (actual - predicted) (MPa)")
    style_axes(axis)
    axis.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def main():
    PARITY_DIR.mkdir(parents=True, exist_ok=True)
    RESIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    manifest_rows = []
    missing_rows = []

    for model_name in MODEL_ORDER:
        try:
            y_train, train_prediction, y_test, test_prediction = (
                load_model_predictions(model_name)
            )
            train = pd.DataFrame(
                {"Actual": y_train, "Predicted": train_prediction}
            )
            test = pd.DataFrame({"Actual": y_test, "Predicted": test_prediction})

            parity_file = PARITY_DIR / f"{safe_name(model_name)}_parity_600dpi.png"
            residual_file = (
                RESIDUAL_DIR / f"{safe_name(model_name)}_residual_600dpi.png"
            )
            plot_parity(train, test, model_name, parity_file)
            plot_residual(train, test, model_name, residual_file)

            train_metrics = regression_metrics(train["Actual"], train["Predicted"])
            test_metrics = regression_metrics(test["Actual"], test["Predicted"])
            metric_rows.append(
                {
                    "Model": model_name,
                    "Train_R2": train_metrics["R2"],
                    "Train_RMSE": train_metrics["RMSE"],
                    "Train_MAE": train_metrics["MAE"],
                    "Test_R2": test_metrics["R2"],
                    "Test_RMSE": test_metrics["RMSE"],
                    "Test_MAE": test_metrics["MAE"],
                    "n_train": len(train),
                    "n_test": len(test),
                }
            )
            manifest_rows.append(
                {
                    "Model": model_name,
                    "Parity_Plot": parity_file.relative_to(ROOT).as_posix(),
                    "Residual_Plot": residual_file.relative_to(ROOT).as_posix(),
                }
            )
            print(f"Completed: {model_name}")
        except (FileNotFoundError, ValueError) as error:
            missing_rows.append({"Model": model_name, "Reason": str(error)})
            print(f"Skipped {model_name}: {error}")

    model_order = {model: index for index, model in enumerate(MODEL_ORDER)}

    metrics = pd.DataFrame(metric_rows)
    if not metrics.empty:
        metrics["Order"] = metrics["Model"].map(model_order)
        metrics = metrics.sort_values("Order").drop(columns="Order")
    metrics.to_csv(TABLE_DIR / "model_metrics_summary.csv", index=False)

    manifest = pd.DataFrame(manifest_rows)
    if not manifest.empty:
        manifest["Order"] = manifest["Model"].map(model_order)
        manifest = manifest.sort_values("Order").drop(columns="Order")
    manifest.to_csv(TABLE_DIR / "plot_manifest.csv", index=False)

    missing = pd.DataFrame(missing_rows, columns=["Model", "Reason"])
    if not missing.empty:
        missing["Order"] = missing["Model"].map(model_order)
        missing = missing.sort_values("Order").drop(columns="Order")
    missing.to_csv(TABLE_DIR / "missing_models.csv", index=False)

    print(f"Parity and residual analysis saved to {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
