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
RETRAINING_DIR = ROOT / "outputs" / "retraining"

DATASETS = ["exp1", "exp2", "exp12"]
DPI = 600

CORE_MODELS = [
    "Classical Abrams",
    "Abrams",
    "Multivariable Abrams",
    "MLR",
    "SVR",
    "Abrams-MLR",
    "Abrams-SVR",
    "Abrams-XGBoost",
    "Abrams-CatBoost",
    "Multivariable-MLR",
    "Multivariable-SVR",
    "Multivariable-XGBoost",
    "Multivariable-CatBoost",
]

TRAIN_COLOR = "#1f77d0"
TEST_COLOR = "#e63946"
IDEAL_COLOR = "#1f4e79"
PLUS20_COLOR = "#f4a261"
MINUS20_COLOR = "#2a9d8f"
ZERO_COLOR = "#8b1e3f"
TITLE_COLOR = "#16324f"


def safe_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def regression_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def style_axes(axis):
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_color("black")

    axis.grid(True, linestyle="--", alpha=0.18)
    axis.tick_params(axis="both", labelsize=10)


def prepare_predictions(data):
    required_columns = {"Model", "Split", "y_true", "y_pred"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing prediction columns: {missing}")

    data = data.copy()
    data["Model"] = data["Model"].astype(str).str.strip()
    data["Split"] = data["Split"].astype(str).str.strip().str.title()
    data["y_true"] = pd.to_numeric(data["y_true"], errors="coerce")
    data["y_pred"] = pd.to_numeric(data["y_pred"], errors="coerce")

    return data.dropna(subset=["y_true", "y_pred"])


def plot_model(dataset, model_name, model_data, output_dir):
    train = model_data[model_data["Split"] == "Train"].copy()
    test = model_data[model_data["Split"] == "Test"].copy()
    if train.empty and test.empty:
        return None

    values = np.concatenate(
        [
            train["y_true"].values,
            train["y_pred"].values,
            test["y_true"].values,
            test["y_pred"].values,
        ]
    ).astype(float)
    span = max(float(np.max(values) - np.min(values)), 1.0)
    lower_limit = max(0.0, float(np.min(values)) - 0.05 * span)
    upper_limit = float(np.max(values)) + 0.05 * span
    reference = np.linspace(lower_limit, upper_limit, 200)

    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))

    parity_axis = axes[0]
    if not train.empty:
        parity_axis.scatter(
            train["y_true"],
            train["y_pred"],
            s=18,
            facecolors="none",
            edgecolors=TRAIN_COLOR,
            linewidths=0.8,
            alpha=0.82,
            label="Train",
        )
    if not test.empty:
        parity_axis.scatter(
            test["y_true"],
            test["y_pred"],
            s=20,
            color=TEST_COLOR,
            edgecolors="black",
            linewidths=0.2,
            alpha=0.80,
            label="Test",
        )

    parity_axis.plot(
        reference,
        reference,
        color=IDEAL_COLOR,
        linewidth=1.5,
        label="Perfect prediction",
    )
    parity_axis.plot(
        reference,
        1.2 * reference,
        "--",
        color=PLUS20_COLOR,
        linewidth=1.0,
        label="+20%",
    )
    parity_axis.plot(
        reference,
        0.8 * reference,
        "--",
        color=MINUS20_COLOR,
        linewidth=1.0,
        label="-20%",
    )
    parity_axis.set_xlim(lower_limit, upper_limit)
    parity_axis.set_ylim(lower_limit, upper_limit)
    parity_axis.set_xlabel("Measured compressive strength (MPa)", fontsize=11)
    parity_axis.set_ylabel("Predicted compressive strength (MPa)", fontsize=11)
    parity_axis.set_title(
        "Parity plot",
        fontsize=12,
        fontweight="bold",
        color=TITLE_COLOR,
    )
    style_axes(parity_axis)
    parity_axis.legend(loc="upper left", fontsize=8.5, frameon=True)

    if not test.empty:
        metrics = regression_metrics(test["y_true"], test["y_pred"])
        annotation = (
            f"Test $R^2$ = {metrics['R2']:.3f}\n"
            f"Test RMSE = {metrics['RMSE']:.2f}\n"
            f"Test MAE = {metrics['MAE']:.2f}\n"
            f"n = {len(test)}"
        )
        parity_axis.text(
            0.97,
            0.05,
            annotation,
            transform=parity_axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.8,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "0.65",
                "alpha": 0.95,
            },
        )

    residual_axis = axes[1]
    if not train.empty:
        train_residual = train["y_true"] - train["y_pred"]
        residual_axis.scatter(
            train["y_pred"],
            train_residual,
            s=18,
            facecolors="none",
            edgecolors=TRAIN_COLOR,
            linewidths=0.8,
            alpha=0.82,
            label="Train",
        )
    if not test.empty:
        test_residual = test["y_true"] - test["y_pred"]
        residual_axis.scatter(
            test["y_pred"],
            test_residual,
            s=20,
            color=TEST_COLOR,
            edgecolors="black",
            linewidths=0.2,
            alpha=0.80,
            label="Test",
        )

    residual_axis.axhline(0, linestyle="--", color=ZERO_COLOR, linewidth=1.2)
    residual_axis.set_xlabel("Predicted compressive strength (MPa)", fontsize=11)
    residual_axis.set_ylabel("Residual (actual - predicted) (MPa)", fontsize=11)
    residual_axis.set_title(
        "Residual plot",
        fontsize=12,
        fontweight="bold",
        color=TITLE_COLOR,
    )
    style_axes(residual_axis)
    residual_axis.legend(loc="upper right", fontsize=8.5, frameon=True)

    figure.suptitle(
        f"{dataset.upper()} | Combined literature-experimental retraining | {model_name}",
        fontsize=13.5,
        fontweight="bold",
        color=TITLE_COLOR,
        y=0.99,
    )

    output_file = (
        output_dir
        / f"{dataset}_{safe_name(model_name)}_parity_residual_600dpi.png"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close(figure)

    return output_file


def main():
    missing_rows = []
    manifest_rows = []

    for dataset in DATASETS:
        dataset_dir = RETRAINING_DIR / dataset
        prediction_file = (
            dataset_dir / f"{dataset}_combined_retraining_predictions_long.csv"
        )
        output_dir = dataset_dir / "figures_parity_residual"

        if not prediction_file.exists():
            missing_rows.append(
                {
                    "Dataset": dataset,
                    "Reason": f"Missing {prediction_file.relative_to(ROOT).as_posix()}",
                }
            )
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        predictions = prepare_predictions(pd.read_csv(prediction_file))

        for model_name in CORE_MODELS:
            model_data = predictions[predictions["Model"] == model_name].copy()
            if model_data.empty:
                continue

            output_file = plot_model(
                dataset,
                model_name,
                model_data,
                output_dir,
            )
            if output_file is not None:
                manifest_rows.append(
                    {
                        "Dataset": dataset,
                        "Model": model_name,
                        "Figure": output_file.relative_to(ROOT).as_posix(),
                    }
                )

    RETRAINING_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows, columns=["Dataset", "Model", "Figure"]).to_csv(
        RETRAINING_DIR / "parity_residual_manifest.csv",
        index=False,
    )
    pd.DataFrame(missing_rows, columns=["Dataset", "Reason"]).to_csv(
        RETRAINING_DIR / "parity_residual_missing.csv",
        index=False,
    )

    print(f"Parity and residual figures saved under {RETRAINING_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
