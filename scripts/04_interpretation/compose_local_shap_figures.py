import math
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _interpretation_common import (
    DISPLAY_LABELS,
    INTERPRETATION_DIR,
    MANUSCRIPT_MODELS,
    SHAP_DIR,
    safe_name,
)


warnings.filterwarnings("ignore")

DPI = 600
LOCAL_DIR = SHAP_DIR / "local"
COMPOSITE_DIR = INTERPRETATION_DIR / "composites" / "shap_local"


def available_models():
    models = {}
    for shap_path in sorted(SHAP_DIR.glob("*_shap_values.csv")):
        model_name = shap_path.stem.removesuffix("_shap_values")
        x_path = SHAP_DIR / f"{model_name}_X_explain.csv"
        base_path = SHAP_DIR / f"{model_name}_base_values.csv"
        if x_path.exists() and base_path.exists():
            models[model_name] = (shap_path, x_path, base_path)
    return models


def representative_rows(shap_values):
    total = shap_values.abs().sum(axis=1)
    ordered = total.sort_values().index
    return [
        ("Low total impact", int(ordered[0])),
        ("Median total impact", int(ordered[len(ordered) // 2])),
        ("High total impact", int(ordered[-1])),
    ]


def plot_local_model(model_name, paths):
    shap_values = pd.read_csv(paths[0])
    explained = pd.read_csv(paths[1])
    base_values = pd.read_csv(paths[2])
    if len(shap_values) == 0:
        raise ValueError("SHAP file is empty.")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), sharex=False)
    export_blocks = []
    for axis, (sample_label, row_index) in zip(
        axes, representative_rows(shap_values)
    ):
        contributions = shap_values.iloc[row_index].sort_values(
            key=np.abs, ascending=True
        )
        colors = np.where(contributions >= 0, "#d9487d", "#3478b8")
        axis.barh(
            [DISPLAY_LABELS.get(feature, feature) for feature in contributions.index],
            contributions.values,
            color=colors,
            edgecolor="black",
            linewidth=0.35,
        )
        axis.axvline(0, color="dimgray", linestyle="--", linewidth=0.9)
        prediction = base_values.loc[row_index, "ReconstructedPrediction"]
        source_row = explained.loc[row_index, "SourceRow"]
        axis.set_title(
            f"{sample_label}\nrow {source_row:g}; prediction {prediction:.2f} MPa",
            fontsize=10,
            fontweight="bold",
        )
        axis.set_xlabel("SHAP contribution (MPa)")
        axis.grid(axis="x", linestyle="--", alpha=0.2)

        block = pd.DataFrame(
            {
                "Model": model_name,
                "SampleClass": sample_label,
                "SourceRow": source_row,
                "BaseValue": base_values.loc[row_index, "BaseValue"],
                "ReconstructedPrediction": prediction,
                "Feature": contributions.index,
                "FeatureValue": [
                    explained.loc[row_index, feature]
                    for feature in contributions.index
                ],
                "SHAPValue": contributions.values,
            }
        )
        export_blocks.append(block)

    fig.suptitle(f"Local SHAP explanations: {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = LOCAL_DIR / f"{safe_name(model_name)}_SHAP_local_600dpi.png"
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    pd.concat(export_blocks, ignore_index=True).to_csv(
        LOCAL_DIR / f"{safe_name(model_name)}_SHAP_local_data.csv", index=False
    )
    return output_path


def family_name(model_name):
    if model_name.startswith("AbramsGuided_"):
        return "abrams_guided"
    if model_name.startswith("MultivariableGuided_"):
        return "multivariable_guided"
    return "ml_only"


def build_composite(models, figures, output_path, title):
    columns = 2
    rows = math.ceil(len(models) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(15, 5.5 * rows))
    axes = np.atleast_1d(axes).reshape(-1)
    for axis, model_name in zip(axes, models):
        axis.imshow(mpimg.imread(figures[model_name]))
        axis.set_title(model_name, fontsize=10, fontweight="bold")
        axis.axis("off")
    for axis in axes[len(models) :]:
        axis.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    models = available_models()
    if not models:
        raise FileNotFoundError(
            f"No complete SHAP raw outputs found in {SHAP_DIR}. "
            "Run 01_shap_analysis.py first."
        )
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    COMPOSITE_DIR.mkdir(parents=True, exist_ok=True)

    figures = {}
    for model_name, paths in models.items():
        figures[model_name] = plot_local_model(model_name, paths)
        print(f"Created local SHAP figure: {model_name}")

    manuscript = [name for name in MANUSCRIPT_MODELS if name in figures]
    if manuscript:
        build_composite(
            manuscript,
            figures,
            COMPOSITE_DIR / "manuscript_models_local_600dpi.png",
            "Local SHAP explanations for selected guided models",
        )

    families = {}
    for model_name in figures:
        families.setdefault(family_name(model_name), []).append(model_name)
    for family, names in families.items():
        build_composite(
            sorted(names),
            figures,
            COMPOSITE_DIR / f"{safe_name(family)}_local_600dpi.png",
            f"Local SHAP explanations: {family.replace('_', ' ').title()}",
        )

    print(f"Local SHAP composites saved to: {COMPOSITE_DIR}")


if __name__ == "__main__":
    main()
