import math
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from _interpretation_common import (
    INTERPRETATION_DIR,
    MANUSCRIPT_MODELS,
    SHAP_DIR,
    safe_name,
)


warnings.filterwarnings("ignore")

DPI = 600
OUTPUT_DIR = INTERPRETATION_DIR / "composites" / "shap_dependence"


def available_figures():
    return {
        path.stem.removesuffix("_SHAP_dependence_600dpi"): path
        for path in sorted(SHAP_DIR.glob("*_SHAP_dependence_600dpi.png"))
    }


def family_name(model_name):
    if model_name.startswith("AbramsGuided_"):
        return "abrams_guided"
    if model_name.startswith("MultivariableGuided_"):
        return "multivariable_guided"
    return "ml_only"


def build_composite(models, figures, output_path, title):
    columns = 2
    rows = math.ceil(len(models) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(14, 10.5 * rows))
    axes = np.atleast_1d(axes).reshape(-1)

    for axis, model_name in zip(axes, models):
        axis.imshow(mpimg.imread(figures[model_name]))
        axis.set_title(model_name, fontsize=11, fontweight="bold")
        axis.axis("off")
    for axis in axes[len(models) :]:
        axis.axis("off")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    figures = available_figures()
    if not figures:
        raise FileNotFoundError(
            "No SHAP dependence figures were found. "
            "Run 02_export_shap_raw_for_dependence.py first."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manuscript = [name for name in MANUSCRIPT_MODELS if name in figures]
    if manuscript:
        build_composite(
            manuscript,
            figures,
            OUTPUT_DIR / "manuscript_models_dependence_600dpi.png",
            "SHAP dependence for selected guided models",
        )

    families = {}
    for model_name in figures:
        families.setdefault(family_name(model_name), []).append(model_name)
    for family, models in families.items():
        build_composite(
            sorted(models),
            figures,
            OUTPUT_DIR / f"{safe_name(family)}_dependence_600dpi.png",
            f"SHAP dependence: {family.replace('_', ' ').title()}",
        )

    print(f"Dependence composites saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
