import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from _interpretation_common import DISPLAY_LABELS, SHAP_DIR, safe_name


warnings.filterwarnings("ignore")

DPI = 600
TOP_FEATURES = 4
RAW_DIR = SHAP_DIR / "dependence_raw"

SHAP_CMAP = LinearSegmentedColormap.from_list(
    "blue_pink_shap",
    ["#1e88ff", "#7b61ff", "#d946ef", "#ff2d7a"],
)


def paired_shap_files():
    pairs = []
    for shap_path in sorted(SHAP_DIR.glob("*_shap_values.csv")):
        model_name = shap_path.stem.removesuffix("_shap_values")
        x_path = SHAP_DIR / f"{model_name}_X_explain.csv"
        if x_path.exists():
            pairs.append((model_name, shap_path, x_path))
    return pairs


def load_pair(shap_path, x_path):
    shap_values = pd.read_csv(shap_path)
    explained = pd.read_csv(x_path)
    source_rows = explained.pop("SourceRow") if "SourceRow" in explained else pd.Series(
        np.arange(len(explained)), name="SourceRow"
    )

    if len(shap_values) != len(explained):
        raise ValueError("SHAP values and explained features have different row counts.")
    if list(shap_values.columns) != list(explained.columns):
        raise ValueError("SHAP and explained-feature columns are not aligned.")
    return shap_values, explained, source_rows


def interaction_feature(explained, feature):
    correlations = explained.corr(numeric_only=True)[feature].abs().drop(feature)
    correlations = correlations.replace([np.inf, -np.inf], np.nan).dropna()
    if correlations.empty:
        return feature
    return correlations.idxmax()


def tidy_dependence_data(model_name, shap_values, explained, source_rows):
    importance_rank = (
        shap_values.abs().mean().rank(method="first", ascending=False).astype(int)
    )
    blocks = []
    for feature in shap_values.columns:
        color_feature = interaction_feature(explained, feature)
        blocks.append(
            pd.DataFrame(
                {
                    "Model": model_name,
                    "SourceRow": source_rows.to_numpy(),
                    "Feature": feature,
                    "FeatureValue": explained[feature].to_numpy(),
                    "SHAPValue": shap_values[feature].to_numpy(),
                    "InteractionFeature": color_feature,
                    "InteractionValue": explained[color_feature].to_numpy(),
                    "ImportanceRank": int(importance_rank[feature]),
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)


def style_axis(axis):
    axis.axhline(0, color="dimgray", linestyle="--", linewidth=1)
    axis.grid(True, linestyle="--", alpha=0.22)
    for spine in axis.spines.values():
        spine.set_linewidth(1)
        spine.set_color("black")


def plot_dependence(model_name, shap_values, explained):
    importance = shap_values.abs().mean().sort_values(ascending=False)
    selected = importance.head(TOP_FEATURES).index
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))

    for axis, feature in zip(axes.flat, selected):
        color_feature = interaction_feature(explained, feature)
        scatter = axis.scatter(
            explained[feature],
            shap_values[feature],
            c=explained[color_feature],
            cmap=SHAP_CMAP,
            s=20,
            alpha=0.78,
            edgecolors="black",
            linewidth=0.15,
        )
        axis.set_xlabel(DISPLAY_LABELS.get(feature, feature))
        axis.set_ylabel(f"SHAP value for {DISPLAY_LABELS.get(feature, feature)} (MPa)")
        axis.set_title(DISPLAY_LABELS.get(feature, feature), fontweight="bold")
        style_axis(axis)
        colorbar = fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label(DISPLAY_LABELS.get(color_feature, color_feature))

    for axis in axes.flat[len(selected) :]:
        axis.axis("off")

    fig.suptitle(f"SHAP dependence: {model_name}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(
        SHAP_DIR / f"{safe_name(model_name)}_SHAP_dependence_600dpi.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    pairs = paired_shap_files()
    if not pairs:
        raise FileNotFoundError(
            f"No paired SHAP files found in {SHAP_DIR}. Run 01_shap_analysis.py first."
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_models = []
    for model_name, shap_path, x_path in pairs:
        shap_values, explained, source_rows = load_pair(shap_path, x_path)
        tidy = tidy_dependence_data(
            model_name, shap_values, explained, source_rows
        )
        tidy.to_csv(
            RAW_DIR / f"{safe_name(model_name)}_dependence_raw.csv", index=False
        )
        all_models.append(tidy)
        plot_dependence(model_name, shap_values, explained)
        print(f"Exported SHAP dependence data: {model_name}")

    pd.concat(all_models, ignore_index=True).to_csv(
        RAW_DIR / "all_models_dependence_raw.csv", index=False
    )
    print(f"Output directory: {RAW_DIR}")


if __name__ == "__main__":
    main()
