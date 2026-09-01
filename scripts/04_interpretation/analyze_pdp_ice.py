import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _interpretation_common import (
    DISPLAY_LABELS,
    MANUSCRIPT_MODELS,
    PDP_DIR,
    load_artifacts,
    load_model_frame,
    pdp_ice_values,
    safe_name,
)


warnings.filterwarnings("ignore")

DPI = 600
GRID_POINTS = 30
PDP_FEATURES = ["Age_day", "W_over_B", "Cement_kgm3"]
CORE_DIR = PDP_DIR / "manuscript_core"


def feature_grid(data, feature):
    values = pd.to_numeric(data[feature], errors="coerce").dropna()
    return np.unique(np.quantile(values, np.linspace(0.05, 0.95, GRID_POINTS)))


def plot_pdp_ice(model_name, artifact, output_dir):
    train = load_model_frame(artifact, split="train")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(PDP_FEATURES), figsize=(15.5, 4.8))
    curve_tables = []

    for axis, feature in zip(axes, PDP_FEATURES):
        grid = feature_grid(train, feature)
        sample, ice, pdp = pdp_ice_values(artifact, train, feature, grid)

        axis.plot(grid, ice.T, color="#9fb7c9", linewidth=0.65, alpha=0.35)
        axis.plot(grid, pdp, color="#8b1e3f", linewidth=2.4, label="PDP")
        axis.set_xlabel(DISPLAY_LABELS.get(feature, feature))
        axis.set_ylabel("Predicted compressive strength (MPa)")
        axis.set_title(DISPLAY_LABELS.get(feature, feature), fontweight="bold")
        axis.grid(True, alpha=0.2)
        axis.legend(frameon=False)

        ymin, ymax = axis.get_ylim()
        rug_height = 0.025 * (ymax - ymin)
        rug_y = ymin + rug_height
        for value in np.quantile(sample[feature], np.linspace(0.05, 0.95, 10)):
            axis.plot(
                [value, value],
                [rug_y, rug_y + rug_height],
                color="#202020",
                linewidth=0.9,
            )

        sample_ids = np.arange(len(sample))
        block = pd.DataFrame(
            {
                "Model": model_name,
                "Feature": feature,
                "GridValue": np.tile(grid, len(sample)),
                "IceSample": np.repeat(sample_ids, len(grid)),
                "ICEPrediction": ice.reshape(-1),
                "PDP": np.tile(pdp, len(sample)),
            }
        )
        curve_tables.append(block)

    fig.suptitle(f"PDP and ICE: {model_name}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    stem = safe_name(model_name)
    fig.savefig(
        output_dir / f"{stem}_pdp_ice_600dpi.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    pd.concat(curve_tables, ignore_index=True).to_csv(
        output_dir / f"{stem}_pdp_ice_data.csv", index=False
    )


def run_models(model_names, output_dir):
    artifacts = load_artifacts(model_names)
    if not artifacts:
        raise FileNotFoundError(
            "No requested interpretation artifacts are available. "
            "Run 01_shap_analysis.py first."
        )

    missing = [name for name in model_names if name not in artifacts]
    for model_name, artifact in artifacts.items():
        print(f"Calculating PDP/ICE: {model_name}")
        plot_pdp_ice(model_name, artifact, output_dir)

    pd.DataFrame({"MissingModel": missing}).to_csv(
        Path(output_dir) / "missing_models.csv", index=False
    )
    return artifacts


def main():
    run_models(MANUSCRIPT_MODELS, CORE_DIR)
    print(f"Output directory: {CORE_DIR}")


if __name__ == "__main__":
    main()
