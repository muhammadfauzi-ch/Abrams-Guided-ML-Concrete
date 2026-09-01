import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _interpretation_common import (
    DISPLAY_LABELS,
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
OUTPUT_DIR = PDP_DIR / "all_models"


def plot_model(model_name, artifact):
    train = load_model_frame(artifact, split="train")
    fig, axes = plt.subplots(1, len(PDP_FEATURES), figsize=(15.5, 4.8))
    data_blocks = []

    for axis, feature in zip(axes, PDP_FEATURES):
        values = train[feature].dropna()
        grid = np.unique(
            np.quantile(values, np.linspace(0.05, 0.95, GRID_POINTS))
        )
        sample, ice, pdp = pdp_ice_values(artifact, train, feature, grid)

        axis.plot(grid, ice.T, color="#9fb7c9", linewidth=0.6, alpha=0.3)
        axis.plot(grid, pdp, color="#8b1e3f", linewidth=2.3, label="PDP")
        axis.set_xlabel(DISPLAY_LABELS.get(feature, feature))
        axis.set_ylabel("Predicted compressive strength (MPa)")
        axis.set_title(DISPLAY_LABELS.get(feature, feature), fontweight="bold")
        axis.grid(True, alpha=0.2)
        axis.legend(frameon=False)

        data_blocks.append(
            pd.DataFrame(
                {
                    "Model": model_name,
                    "Feature": feature,
                    "GridValue": np.tile(grid, len(sample)),
                    "IceSample": np.repeat(np.arange(len(sample)), len(grid)),
                    "ICEPrediction": ice.reshape(-1),
                    "PDP": np.tile(pdp, len(sample)),
                }
            )
        )

    fig.suptitle(f"PDP and ICE: {model_name}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    stem = safe_name(model_name)
    fig.savefig(
        OUTPUT_DIR / f"{stem}_pdp_ice_600dpi.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)
    pd.concat(data_blocks, ignore_index=True).to_csv(
        OUTPUT_DIR / f"{stem}_pdp_ice_data.csv", index=False
    )


def main():
    artifacts = load_artifacts()
    if not artifacts:
        raise FileNotFoundError(
            "No interpretation model artifacts are available. "
            "Run 01_shap_analysis.py first."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    completed = []
    failures = []
    for model_name, artifact in artifacts.items():
        print(f"Calculating PDP/ICE: {model_name}")
        try:
            plot_model(model_name, artifact)
            completed.append(
                {"Model": model_name, "Family": artifact["family"]}
            )
        except Exception as exc:
            failures.append({"Model": model_name, "Error": str(exc)})
            print(f"[FAILED] {model_name}: {exc}")

    pd.DataFrame(completed).to_csv(
        OUTPUT_DIR / "pdp_ice_run_summary.csv", index=False
    )
    pd.DataFrame(failures, columns=["Model", "Error"]).to_csv(
        OUTPUT_DIR / "pdp_ice_run_failures.csv", index=False
    )
    print(f"PDP/ICE completed for {len(completed)} models.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
