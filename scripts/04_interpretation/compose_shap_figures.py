import math
import warnings

import matplotlib

matplotlib.use("Agg")

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
OUTPUT_DIR = INTERPRETATION_DIR / "composites" / "shap"


def available_models():
    return {
        path.stem.removesuffix("_shap_values"): path
        for path in sorted(SHAP_DIR.glob("*_shap_values.csv"))
    }


def family_name(model_name):
    if model_name.startswith("AbramsGuided_"):
        return "abrams_guided"
    if model_name.startswith("MultivariableGuided_"):
        return "multivariable_guided"
    return "ml_only"


def importance_panel(axis, model_name, values):
    importance = values.abs().mean().sort_values()
    axis.barh(
        [DISPLAY_LABELS.get(feature, feature) for feature in importance.index],
        importance.values,
        color="#1f4e79",
        edgecolor="black",
        linewidth=0.4,
    )
    axis.set_title(model_name, fontsize=10, fontweight="bold")
    axis.set_xlabel("Mean |SHAP| (MPa)")
    axis.grid(axis="x", linestyle="--", alpha=0.2)


def signed_panel(axis, model_name, values):
    signed = values.mean().sort_values()
    colors = np.where(signed >= 0, "#d9487d", "#3478b8")
    axis.barh(
        [DISPLAY_LABELS.get(feature, feature) for feature in signed.index],
        signed.values,
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )
    axis.axvline(0, color="dimgray", linestyle="--", linewidth=0.9)
    axis.set_title(model_name, fontsize=10, fontweight="bold")
    axis.set_xlabel("Mean signed SHAP value (MPa)")
    axis.grid(axis="x", linestyle="--", alpha=0.2)


def build_composite(models, files, panel_function, title, output_path):
    columns = 2
    rows = math.ceil(len(models) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(12.5, 4.3 * rows))
    axes = np.atleast_1d(axes).reshape(-1)

    for axis, model_name in zip(axes, models):
        panel_function(axis, model_name, pd.read_csv(files[model_name]))
    for axis in axes[len(models) :]:
        axis.axis("off")

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def write_pair(models, files, stem, title):
    build_composite(
        models,
        files,
        importance_panel,
        f"{title}: mean absolute contribution",
        OUTPUT_DIR / f"{stem}_importance_600dpi.png",
    )
    build_composite(
        models,
        files,
        signed_panel,
        f"{title}: mean signed contribution",
        OUTPUT_DIR / f"{stem}_mean_signed_600dpi.png",
    )


def main():
    files = available_models()
    if not files:
        raise FileNotFoundError(
            f"No SHAP values found in {SHAP_DIR}. Run 01_shap_analysis.py first."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manuscript = [name for name in MANUSCRIPT_MODELS if name in files]
    if manuscript:
        write_pair(
            manuscript,
            files,
            "manuscript_models",
            "Selected guided models",
        )

    families = {}
    for model_name in files:
        families.setdefault(family_name(model_name), []).append(model_name)
    for family, models in families.items():
        write_pair(
            sorted(models),
            files,
            safe_name(family),
            family.replace("_", " ").title(),
        )

    print(f"SHAP composite figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
