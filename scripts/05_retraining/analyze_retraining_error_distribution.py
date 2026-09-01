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
RETRAINING_DIR = ROOT / "outputs" / "retraining"

DATASETS = ["exp1", "exp2", "exp12"]
FAMILIES = ["ML-only", "Abrams-guided", "Multivariable-guided"]
METRICS = ["R2", "RMSE", "MAE"]
DPI = 600

TITLE_COLOR = "#16324f"
FAMILY_COLORS = {
    "ML-only": "#3f8efc",
    "Abrams-guided": "#ff7f50",
    "Multivariable-guided": "#27b3a7",
}
METRIC_LABELS = {
    "R2": r"$R^2$",
    "RMSE": "RMSE (MPa)",
    "MAE": "MAE (MPa)",
}


def safe_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def style_axes(axis):
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_color("black")

    axis.grid(True, linestyle="--", alpha=0.20)
    axis.tick_params(axis="both", labelsize=10)


def prepare_summary(path):
    data = pd.read_csv(path)
    required_columns = {"Family", "Model", *METRICS}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing columns in {path.name}: {missing}")

    data = data.copy()
    data["Family"] = data["Family"].astype(str).str.strip()
    data["Model"] = data["Model"].astype(str).str.strip()
    for metric in METRICS:
        data[metric] = pd.to_numeric(data[metric], errors="coerce")

    data = data[~data["Model"].isin(["", "nan", "None"])]
    return data.dropna(subset=METRICS)


def set_metric_limits(axis, values, metric):
    values = np.asarray(values, dtype=float)
    minimum = np.nanmin(values)
    maximum = np.nanmax(values)

    if metric == "R2":
        if minimum >= 0:
            lower_limit = max(0.0, minimum - 0.06)
            upper_limit = min(1.02, maximum + 0.06)
        else:
            padding = 0.08 * (maximum - minimum + 1e-9)
            lower_limit = minimum - padding
            upper_limit = maximum + padding
    else:
        padding = 0.12 * (maximum - minimum + 1e-9)
        lower_limit = max(0.0, minimum - 0.05 * (maximum + 1e-9))
        upper_limit = maximum + max(padding, 0.3)

    if upper_limit <= lower_limit:
        upper_limit = lower_limit + (0.1 if metric == "R2" else 1.0)

    axis.set_ylim(lower_limit, upper_limit)


def add_value_labels(axis, bars, values, metric):
    lower_limit, upper_limit = axis.get_ylim()
    span = upper_limit - lower_limit
    offset = 0.015 * span

    for bar, value in zip(bars, values):
        if metric == "R2" and value < 0:
            y_position = max(value - offset, lower_limit + 0.04 * span)
            vertical_alignment = "top"
        else:
            y_position = min(value + offset, upper_limit - 0.04 * span)
            vertical_alignment = "bottom"

        axis.text(
            bar.get_x() + bar.get_width() / 2,
            y_position,
            f"{value:.3f}",
            ha="center",
            va=vertical_alignment,
            fontsize=8.5,
            rotation=90,
        )


def plot_family(dataset, family, summary, output_dir):
    selected = summary[summary["Family"] == family].copy()
    if selected.empty:
        return None

    selected = selected.sort_values("Model").reset_index(drop=True)
    figure, axes = plt.subplots(1, 3, figsize=(15.8, 5.1))

    for axis, metric in zip(axes, METRICS):
        values = selected[metric].values
        positions = np.arange(len(selected))
        bars = axis.bar(
            positions,
            values,
            color=FAMILY_COLORS[family],
            edgecolor="#2f2f2f",
            linewidth=0.8,
            alpha=0.9,
        )

        set_metric_limits(axis, values, metric)
        add_value_labels(axis, bars, values, metric)
        axis.set_xticks(positions)
        axis.set_xticklabels(selected["Model"], rotation=50, ha="right")
        axis.set_title(
            METRIC_LABELS[metric],
            fontsize=13,
            fontweight="bold",
            color=TITLE_COLOR,
            pad=8,
        )
        axis.set_ylabel(METRIC_LABELS[metric], fontsize=11)
        style_axes(axis)

        if metric == "R2":
            axis.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)

    figure.suptitle(
        f"{dataset.upper()} | Combined retraining | {family}",
        fontsize=15,
        fontweight="bold",
        color=TITLE_COLOR,
        y=0.98,
    )

    output_file = (
        output_dir
        / f"{dataset}_{safe_name(family)}_metrics_comparison_600dpi.png"
    )
    plt.tight_layout(rect=[0.02, 0.10, 1, 0.93])
    plt.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close(figure)

    return output_file


def main():
    manifest_rows = []
    missing_rows = []

    for dataset in DATASETS:
        dataset_dir = RETRAINING_DIR / dataset
        summary_file = (
            dataset_dir / f"{dataset}_combined_retraining_test_summary.csv"
        )
        if not summary_file.exists():
            missing_rows.append(
                {
                    "Dataset": dataset,
                    "Reason": f"Missing {summary_file.relative_to(ROOT).as_posix()}",
                }
            )
            continue

        summary = prepare_summary(summary_file)
        output_dir = dataset_dir / "figures_metric_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)

        for family in FAMILIES:
            output_file = plot_family(dataset, family, summary, output_dir)
            if output_file is not None:
                manifest_rows.append(
                    {
                        "Dataset": dataset,
                        "Family": family,
                        "Figure": output_file.relative_to(ROOT).as_posix(),
                    }
                )

    RETRAINING_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows, columns=["Dataset", "Family", "Figure"]).to_csv(
        RETRAINING_DIR / "metric_comparison_manifest.csv",
        index=False,
    )
    pd.DataFrame(missing_rows, columns=["Dataset", "Reason"]).to_csv(
        RETRAINING_DIR / "metric_comparison_missing.csv",
        index=False,
    )

    print(f"Metric comparison figures saved under {RETRAINING_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
