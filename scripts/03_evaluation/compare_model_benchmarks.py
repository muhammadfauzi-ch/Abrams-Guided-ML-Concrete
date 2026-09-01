from pathlib import Path
import warnings

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

SUMMARY_FILE = ROOT / "outputs" / "evaluation" / "all_model_metrics_summary.csv"
OUTPUT_DIR = ROOT / "outputs" / "evaluation" / "benchmark_comparison"

DPI = 600

DISPLAY_NAMES = {
    "Classical_Abrams": "Classical Abrams",
    "PhysicsPrior_WB_Age": "Abrams",
    "Multivariable_Empirical_Prior": "Multivariable Abrams",
    "MLR": "MLR",
    "SVR": "SVR",
    "KNN": "KNN",
    "RF": "RF",
    "GBR": "GBR",
    "XGBoost": "XGBoost",
    "CatBoost": "CatBoost",
    "DNN": "DNN",
    "AutoGluon": "AutoGluon",
    "AbramsGuided_MLR": "Abrams-MLR",
    "AbramsGuided_SVR": "Abrams-SVR",
    "AbramsGuided_KNN": "Abrams-KNN",
    "AbramsGuided_RF": "Abrams-RF",
    "AbramsGuided_GBR": "Abrams-GBR",
    "AbramsGuided_XGBoost": "Abrams-XGBoost",
    "AbramsGuided_CatBoost": "Abrams-CatBoost",
    "AbramsGuided_DNN": "Abrams-DNN",
    "AbramsGuided_AutoGluon": "Abrams-AutoGluon",
    "MultivariableGuided_MLR": "Multivariable-MLR",
    "MultivariableGuided_SVR": "Multivariable-SVR",
    "MultivariableGuided_KNN": "Multivariable-KNN",
    "MultivariableGuided_RF": "Multivariable-RF",
    "MultivariableGuided_GBR": "Multivariable-GBR",
    "MultivariableGuided_XGBoost": "Multivariable-XGBoost",
    "MultivariableGuided_CatBoost": "Multivariable-CatBoost",
    "MultivariableGuided_DNN": "Multivariable-DNN",
    "MultivariableGuided_AutoGluon": "Multivariable-AutoGluon",
}

SIMPLE_MODELS = [
    "Classical_Abrams",
    "PhysicsPrior_WB_Age",
    "Multivariable_Empirical_Prior",
    "MLR",
    "SVR",
    "AbramsGuided_MLR",
    "AbramsGuided_SVR",
    "MultivariableGuided_MLR",
    "MultivariableGuided_SVR",
]

ABRAMS_MODELS = [
    "AbramsGuided_MLR",
    "AbramsGuided_SVR",
    "AbramsGuided_KNN",
    "AbramsGuided_RF",
    "AbramsGuided_GBR",
    "AbramsGuided_XGBoost",
    "AbramsGuided_CatBoost",
    "AbramsGuided_DNN",
    "AbramsGuided_AutoGluon",
]

MULTIVARIABLE_MODELS = [
    "MultivariableGuided_MLR",
    "MultivariableGuided_SVR",
    "MultivariableGuided_KNN",
    "MultivariableGuided_RF",
    "MultivariableGuided_GBR",
    "MultivariableGuided_XGBoost",
    "MultivariableGuided_CatBoost",
    "MultivariableGuided_DNN",
    "MultivariableGuided_AutoGluon",
]

FULL_BENCHMARK_MODELS = [
    "Classical_Abrams",
    "PhysicsPrior_WB_Age",
    "Multivariable_Empirical_Prior",
    "MLR",
    "SVR",
    *ABRAMS_MODELS,
    *MULTIVARIABLE_MODELS,
]

COLOR_CLASSICAL = "#d55e5e"
COLOR_PRIOR = "#f4a261"
COLOR_ML = "#6fa8dc"
COLOR_ABRAMS_GUIDED = "#7fbf7b"
COLOR_MULTIVARIABLE_GUIDED = "#f6c177"
COLOR_AUTOML = "#8ecae6"


def infer_color(model_name):
    if model_name == "Classical_Abrams":
        return COLOR_CLASSICAL
    if model_name in {
        "PhysicsPrior_WB_Age",
        "Multivariable_Empirical_Prior",
    }:
        return COLOR_PRIOR
    if model_name == "AutoGluon":
        return COLOR_AUTOML
    if model_name.startswith("AbramsGuided_"):
        return COLOR_ABRAMS_GUIDED
    if model_name.startswith("MultivariableGuided_"):
        return COLOR_MULTIVARIABLE_GUIDED

    return COLOR_ML


def read_test_metrics(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics summary not found: {path}. Run 01_metrics_summary.py first."
        )

    metrics = pd.read_csv(path)
    required_columns = {"Dataset", "Model", "R2", "RMSE", "MAE"}
    missing_columns = required_columns.difference(metrics.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing columns in {path}: {missing}")

    for column in ["R2", "RMSE", "MAE"]:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")

    metrics = metrics[
        metrics["Dataset"].astype(str).str.lower().str.strip() == "test"
    ].copy()
    metrics = metrics.dropna(subset=["Model", "R2", "RMSE", "MAE"])
    metrics = metrics.sort_values(
        by=["Model", "R2", "RMSE", "MAE"],
        ascending=[True, False, True, True],
    )
    metrics = metrics.drop_duplicates(subset=["Model"], keep="first")

    metrics["Display_Model"] = metrics["Model"].map(DISPLAY_NAMES).fillna(
        metrics["Model"]
    )
    metrics["Bar_Color"] = metrics["Model"].apply(infer_color)

    return metrics


def select_models(metrics, model_order):
    available = metrics[metrics["Model"].isin(model_order)].copy()
    available["Order"] = available["Model"].map(
        {model: index for index, model in enumerate(model_order)}
    )
    available = available.sort_values("Order").reset_index(drop=True)

    missing = [model for model in model_order if model not in set(available["Model"])]
    return available, missing


def style_axis(axis, ylabel, title):
    axis.set_title(title, fontsize=19, fontweight="bold", pad=10)
    axis.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    axis.grid(axis="y", linestyle="-", alpha=0.22)
    axis.tick_params(axis="both", labelsize=11)

    for spine in axis.spines.values():
        spine.set_linewidth(1.1)


def add_value_labels(axis, bars, values, metric, vertical, upper_limit):
    for bar, value in zip(bars, values):
        label = f"{value:.4f}" if metric == "R2" else f"{value:.2f}"
        offset = 0.008 if metric == "R2" else max(values) * 0.015
        y_position = min(bar.get_height() + offset, upper_limit * 0.965)

        axis.text(
            bar.get_x() + bar.get_width() / 2,
            y_position,
            label,
            ha="center",
            va="bottom",
            fontsize=8 if vertical else 9,
            rotation=90 if vertical else 0,
            clip_on=True,
        )


def plot_metric_panel(axis, data, metric, ylabel, title, vertical_labels):
    ascending = metric != "R2"
    ordered = data.sort_values(metric, ascending=ascending).reset_index(drop=True)
    positions = np.arange(len(ordered))

    bars = axis.bar(
        positions,
        ordered[metric],
        color=ordered["Bar_Color"],
        edgecolor="black",
        linewidth=0.6,
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        ordered["Display_Model"],
        rotation=45,
        ha="right",
        fontsize=10,
    )
    style_axis(axis, ylabel, title)

    if metric == "R2":
        upper_limit = max(1.02, ordered[metric].max() + 0.05)
    else:
        upper_limit = ordered[metric].max() * 1.12

    axis.set_ylim(0, upper_limit)
    add_value_labels(
        axis,
        bars,
        ordered[metric].values,
        metric,
        vertical_labels,
        upper_limit,
    )


def make_three_panel_plot(data, figure_title, output_file, vertical_labels=False):
    if data.empty:
        print(f"Skipping {output_file.name}: no requested models are available.")
        return

    figure, axes = plt.subplots(1, 3, figsize=(22, 8.5))
    plot_metric_panel(
        axes[0],
        data,
        "R2",
        r"Test $R^2$",
        "Higher is better",
        vertical_labels,
    )
    plot_metric_panel(
        axes[1],
        data,
        "RMSE",
        "Test RMSE (MPa)",
        "Lower is better",
        vertical_labels,
    )
    plot_metric_panel(
        axes[2],
        data,
        "MAE",
        "Test MAE (MPa)",
        "Lower is better",
        vertical_labels,
    )

    figure.suptitle(figure_title, fontsize=23, fontweight="bold", y=0.98)
    legend_elements = [
        Patch(facecolor=COLOR_ML, edgecolor="black", label="ML-only"),
        Patch(
            facecolor=COLOR_ABRAMS_GUIDED,
            edgecolor="black",
            label="Abrams-guided",
        ),
        Patch(
            facecolor=COLOR_MULTIVARIABLE_GUIDED,
            edgecolor="black",
            label="Multivariable-guided",
        ),
        Patch(facecolor=COLOR_AUTOML, edgecolor="black", label="AutoML"),
        Patch(facecolor=COLOR_PRIOR, edgecolor="black", label="Physics prior"),
        Patch(
            facecolor=COLOR_CLASSICAL,
            edgecolor="black",
            label="Classical baseline",
        ),
    ]
    figure.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=6,
        fontsize=11,
        frameon=True,
    )

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.90])
    plt.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file.relative_to(ROOT)}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_metrics = read_test_metrics(SUMMARY_FILE)

    figure_specs = [
        (
            SIMPLE_MODELS,
            "Comparison of baseline, prior, and simple guided models on the test set",
            OUTPUT_DIR / "Figure_B1_Simple_Model_Comparison_600dpi.png",
            False,
        ),
        (
            ABRAMS_MODELS,
            "Comparison of Abrams-guided model family on the test set",
            OUTPUT_DIR / "Figure_B2_Abrams_Family_Comparison_600dpi.png",
            False,
        ),
        (
            MULTIVARIABLE_MODELS,
            "Comparison of multivariable-guided model family on the test set",
            OUTPUT_DIR / "Figure_B3_Multivariable_Family_Comparison_600dpi.png",
            False,
        ),
        (
            FULL_BENCHMARK_MODELS,
            "Benchmark comparison across predictive performance metrics on the test set",
            OUTPUT_DIR / "Figure_B4_Full_Benchmark_Comparison_600dpi.png",
            True,
        ),
    ]

    for model_order, title, output_file, vertical_labels in figure_specs:
        selected, missing = select_models(test_metrics, model_order)
        if missing:
            print(f"Missing from {output_file.name}: {', '.join(missing)}")
        make_three_panel_plot(
            selected,
            title,
            output_file,
            vertical_labels=vertical_labels,
        )


if __name__ == "__main__":
    main()
