import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from _interpretation_common import (
    DISPLAY_LABELS,
    SHAP_DIR,
    load_artifacts,
    load_model_frame,
    predict_artifact,
    safe_name,
    train_interpretation_models,
)


warnings.filterwarnings("ignore")

DPI = 600
N_BACKGROUND = 120
N_EXPLAIN = 180

SHAP_CMAP = LinearSegmentedColormap.from_list(
    "blue_pink_shap",
    ["#1e88ff", "#7b61ff", "#d946ef", "#ff2d7a"],
)


def make_predict_function(artifact):
    features = artifact["features"]

    def predict(values):
        frame = pd.DataFrame(values, columns=features)
        return predict_artifact(artifact, frame)

    return predict


def explain_model(artifact):
    import shap

    train = load_model_frame(artifact, split="train")
    test = load_model_frame(artifact, split="test")
    features = artifact["features"]

    background = train[features].sample(
        min(N_BACKGROUND, len(train)), random_state=42
    )
    explain = test[features].sample(
        min(N_EXPLAIN, len(test)), random_state=42
    )
    source_rows = explain.index.to_numpy()
    explain = explain.reset_index(drop=True)

    explainer = shap.Explainer(
        make_predict_function(artifact),
        background,
        algorithm="permutation",
        feature_names=features,
    )
    explanation = explainer(
        explain,
        max_evals=2 * len(features) + 1,
        silent=True,
    )
    values = np.asarray(explanation.values)
    if values.ndim != 2 or values.shape != explain.shape:
        raise ValueError(
            f"Unexpected SHAP shape {values.shape}; expected {explain.shape}."
        )

    base_values = np.asarray(explanation.base_values).reshape(-1)
    if len(base_values) == 1:
        base_values = np.repeat(base_values, len(explain))
    return explain, source_rows, values, base_values


def save_raw_outputs(model_name, explain, source_rows, values, base_values):
    stem = safe_name(model_name)
    pd.DataFrame(values, columns=explain.columns).to_csv(
        SHAP_DIR / f"{stem}_shap_values.csv", index=False
    )

    x_export = explain.copy()
    x_export.insert(0, "SourceRow", source_rows)
    x_export.to_csv(SHAP_DIR / f"{stem}_X_explain.csv", index=False)

    pd.DataFrame(
        {
            "SourceRow": source_rows,
            "BaseValue": base_values,
            "SHAPSum": values.sum(axis=1),
            "ReconstructedPrediction": base_values + values.sum(axis=1),
        }
    ).to_csv(SHAP_DIR / f"{stem}_base_values.csv", index=False)


def plot_summary(model_name, explain, values):
    import shap

    labels = [DISPLAY_LABELS.get(feature, feature) for feature in explain.columns]
    shap.summary_plot(
        values,
        explain,
        feature_names=labels,
        cmap=SHAP_CMAP,
        show=False,
    )
    plt.title(f"SHAP summary: {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        SHAP_DIR / f"{safe_name(model_name)}_SHAP_summary_600dpi.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close()


def plot_importance(model_name, features, values):
    importance = pd.Series(
        np.abs(values).mean(axis=0), index=features
    ).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(
        [DISPLAY_LABELS.get(feature, feature) for feature in importance.index],
        importance.values,
        color="#1f4e79",
        edgecolor="black",
        linewidth=0.6,
    )
    offset = max(float(importance.max()) * 0.015, 1e-9)
    for bar, value in zip(bars, importance.values):
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("Mean absolute SHAP value (MPa)")
    ax.set_title(f"SHAP feature importance: {model_name}", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        SHAP_DIR / f"{safe_name(model_name)}_SHAP_bar_600dpi.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    print("Training manual models for interpretation...")
    manifest = train_interpretation_models()
    artifacts = load_artifacts()

    completed = []
    failures = []
    for model_name in manifest["Model"]:
        artifact = artifacts.get(model_name)
        if artifact is None:
            failures.append({"Model": model_name, "Error": "Artifact not found"})
            continue

        print(f"Calculating SHAP values: {model_name}")
        try:
            explain, source_rows, values, base_values = explain_model(artifact)
            save_raw_outputs(
                model_name, explain, source_rows, values, base_values
            )
            plot_summary(model_name, explain, values)
            plot_importance(model_name, explain.columns, values)
            completed.append(
                {
                    "Model": model_name,
                    "Family": artifact["family"],
                    "ExplainedRows": len(explain),
                    "Features": len(explain.columns),
                }
            )
        except Exception as exc:
            failures.append({"Model": model_name, "Error": str(exc)})
            print(f"[FAILED] {model_name}: {exc}")

    pd.DataFrame(completed).to_csv(
        SHAP_DIR / "shap_run_summary.csv", index=False
    )
    pd.DataFrame(failures, columns=["Model", "Error"]).to_csv(
        SHAP_DIR / "shap_run_failures.csv", index=False
    )

    print(f"SHAP analysis completed for {len(completed)} models.")
    print(f"Output directory: {SHAP_DIR}")


if __name__ == "__main__":
    main()
