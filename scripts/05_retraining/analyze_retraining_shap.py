from pathlib import Path
import json
import re
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap, Normalize


matplotlib.use("Agg")
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
RETRAINING_DIR = ROOT / "outputs" / "retraining"
OUTPUT_DIR = RETRAINING_DIR / "shap"

DATASETS = ["exp1", "exp2", "exp12"]

DPI = 600
N_BACKGROUND = 120
N_EXPLAIN = 180
TOP_DEPENDENCE_FEATURES = 4
RANDOM_STATE = 42

DISPLAY_LABELS = {
    "Age_day": "Age",
    "Cement_kgm3": "Cement",
    "Water_kgm3": "Water",
    "W_over_B": "W/B",
    "SCM1_kgm3": "SCM1",
    "SCM2_kgm3": "SCM2",
    "FineAgg_kgm3": "Fine aggregate",
    "CoarseAgg_kgm3": "Coarse aggregate",
    "SP_kgm3": "SP",
    "Fiber_kgm3": "Fiber",
}

TREE_MODEL_NAMES = {
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "XGBRegressor",
    "CatBoostRegressor",
}

SHAP_COLORMAP = LinearSegmentedColormap.from_list(
    "blue_pink_shap",
    ["#1e88ff", "#7b61ff", "#d946ef", "#ff2d7a"],
)


def safe_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def repo_relative(path):
    return path.relative_to(ROOT).as_posix()


def style_axes(axis):
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_color("black")

    axis.grid(True, linestyle="--", alpha=0.25)
    axis.tick_params(axis="both", labelsize=10)


def prepare_powerlaw_features(data, features):
    transformed = data[features].copy().astype(float)
    zero_allowed = {
        "SCM1_kgm3": 1e-6,
        "SCM2_kgm3": 1e-6,
        "SP_kgm3": 1e-6,
        "Fiber_kgm3": 1e-6,
    }

    for feature, epsilon in zero_allowed.items():
        if feature in transformed.columns:
            transformed[feature] = transformed[feature].clip(lower=0.0) + epsilon

    return transformed


def prior_design_matrix(data, prior_type, features):
    if prior_type == "classical_abrams":
        return np.column_stack(
            [np.ones(len(data)), data["W_over_B"].values]
        )
    if prior_type == "abrams_age":
        return np.column_stack(
            [
                np.ones(len(data)),
                np.log(data["W_over_B"].values),
                np.log(data["Age_day"].values),
            ]
        )
    if prior_type == "multivariable":
        transformed = prepare_powerlaw_features(data, features)
        log_features = [
            np.log(transformed[feature].values) for feature in features
        ]
        return np.column_stack([np.ones(len(data)), *log_features])

    raise ValueError(f"Unknown prior type: {prior_type}")


def predict_prior(artifact, data):
    features = artifact["features"]
    design_matrix = prior_design_matrix(data, artifact["prior_type"], features)
    return np.exp(design_matrix @ artifact["coefficients"])


def predict_artifact(artifact, data):
    artifact_type = artifact["artifact_type"]

    if artifact_type == "prior":
        return predict_prior(artifact, data)
    if artifact_type == "ml":
        return artifact["model"].predict(data[artifact["features"]])
    if artifact_type == "guided":
        prior_prediction = predict_prior(artifact["prior"], data)
        residual_prediction = artifact["residual_model"].predict(
            data[artifact["features"]]
        )
        return prior_prediction + residual_prediction

    raise ValueError(f"Unknown artifact type: {artifact_type}")


def read_artifact_manifest(dataset):
    manifest_file = (
        RETRAINING_DIR
        / dataset
        / f"{dataset}_model_artifacts.csv"
    )
    if not manifest_file.exists():
        raise FileNotFoundError(f"Missing {repo_relative(manifest_file)}")

    manifest = pd.read_csv(manifest_file)
    required_columns = {"Family", "Model", "Artifact", "Features"}
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing columns in {manifest_file.name}: {missing}")

    return manifest, manifest_file


def read_feature_tables(dataset):
    dataset_dir = RETRAINING_DIR / dataset
    train_file = dataset_dir / f"{dataset}_train.csv"
    test_file = dataset_dir / f"{dataset}_test.csv"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"Missing train/test feature tables under {repo_relative(dataset_dir)}"
        )

    return pd.read_csv(train_file), pd.read_csv(test_file), train_file, test_file


def get_predict_function(artifact, features):
    def predict_function(values):
        data = pd.DataFrame(values, columns=features)
        return np.asarray(predict_artifact(artifact, data), dtype=float)

    return predict_function


def normalize_shap_values(values):
    if isinstance(values, list):
        values = values[0]

    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP value shape: {values.shape}")

    return values


def calculate_shap_values(artifact, background, explain):
    if artifact["artifact_type"] == "ml":
        model = artifact["model"]
        if model.__class__.__name__ in TREE_MODEL_NAMES:
            explainer = shap.TreeExplainer(model)
            return normalize_shap_values(explainer.shap_values(explain))

    predict_function = get_predict_function(artifact, list(explain.columns))

    try:
        explainer = shap.Explainer(predict_function, background)
        explanation = explainer(explain)
        return normalize_shap_values(explanation.values)
    except Exception:
        explainer = shap.KernelExplainer(predict_function, background)
        values = explainer.shap_values(explain, nsamples=100)
        return normalize_shap_values(values)


def choose_color_feature(feature_data, feature):
    correlations = {}

    for candidate in feature_data.columns:
        if candidate == feature:
            continue
        correlations[candidate] = abs(
            pd.to_numeric(feature_data[feature], errors="coerce").corr(
                pd.to_numeric(feature_data[candidate], errors="coerce")
            )
        )

    correlations = pd.Series(correlations, dtype=float).dropna()
    if correlations.empty:
        return feature

    return correlations.idxmax()


def plot_shap_bar(model_name, shap_values, features, output_file):
    importance = pd.DataFrame(shap_values, columns=features).abs().mean()
    importance = importance.sort_values(ascending=True)

    figure, axis = plt.subplots(figsize=(8.0, 6.0))
    bars = axis.barh(
        [DISPLAY_LABELS.get(feature, feature) for feature in importance.index],
        importance.values,
        color="#1f4e79",
        edgecolor="black",
        linewidth=0.8,
    )
    maximum = max(importance.max(), 1e-12)
    for bar, value in zip(bars, importance.values):
        axis.text(
            value + maximum * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=10,
        )

    axis.set_xlabel("Mean absolute SHAP value", fontsize=12, fontweight="bold")
    axis.set_title(
        f"SHAP feature importance: {model_name}",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    style_axes(axis)
    plt.tight_layout()
    plt.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def plot_shap_summary(shap_values, explain, output_file):
    plt.figure()
    shap.summary_plot(
        shap_values,
        explain,
        feature_names=[DISPLAY_LABELS.get(feature, feature) for feature in explain],
        cmap=SHAP_COLORMAP,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_shap_dependence(shap_values, explain, output_file):
    shap_table = pd.DataFrame(shap_values, columns=explain.columns)
    importance = shap_table.abs().mean().sort_values(ascending=False)
    top_features = importance.index[:TOP_DEPENDENCE_FEATURES].tolist()
    if not top_features:
        return False

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for axis, feature in zip(axes, top_features):
        color_feature = choose_color_feature(explain, feature)
        if color_feature == feature:
            color_values = np.zeros(len(explain))
        else:
            color_values = pd.to_numeric(
                explain[color_feature],
                errors="coerce",
            ).values

        finite = np.isfinite(color_values)
        minimum = np.nanmin(color_values) if finite.any() else 0.0
        maximum = np.nanmax(color_values) if finite.any() else 1.0
        if minimum == maximum:
            maximum = minimum + 1e-9

        scatter = axis.scatter(
            explain[feature],
            shap_table[feature],
            c=color_values,
            cmap=SHAP_COLORMAP,
            norm=Normalize(vmin=minimum, vmax=maximum),
            s=18,
            alpha=0.78,
            edgecolors="black",
            linewidth=0.15,
        )
        axis.axhline(0, linestyle="--", color="dimgray", linewidth=1.1)
        axis.set_xlabel(DISPLAY_LABELS.get(feature, feature), fontsize=11)
        axis.set_ylabel(
            f"SHAP value for {DISPLAY_LABELS.get(feature, feature)}",
            fontsize=11,
        )
        axis.set_title(
            f"Dependence: {DISPLAY_LABELS.get(feature, feature)}",
            fontsize=12,
            fontweight="bold",
        )
        style_axes(axis)

        colorbar = plt.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label(DISPLAY_LABELS.get(color_feature, color_feature), fontsize=10)

    for axis in axes[len(top_features):]:
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close(figure)

    return True


def run_dataset(dataset, manifest_rows):
    artifact_manifest, manifest_file = read_artifact_manifest(dataset)
    train, test, train_file, test_file = read_feature_tables(dataset)
    dataset_output = OUTPUT_DIR / dataset
    dataset_output.mkdir(parents=True, exist_ok=True)

    dataset_metadata = {
        "dataset": dataset,
        "artifact_manifest": repo_relative(manifest_file),
        "train_file": repo_relative(train_file),
        "test_file": repo_relative(test_file),
        "n_train": len(train),
        "n_test": len(test),
        "n_background_max": N_BACKGROUND,
        "n_explain_max": N_EXPLAIN,
    }
    metadata_file = dataset_output / f"{dataset}_shap_metadata.json"
    metadata_file.write_text(
        json.dumps(dataset_metadata, indent=2),
        encoding="utf-8",
    )

    for row in artifact_manifest.itertuples(index=False):
        model_name = str(row.Model)
        family = str(row.Family)
        artifact_file = ROOT / str(row.Artifact)

        try:
            features = json.loads(row.Features)
            missing_features = set(features).difference(train.columns).union(
                set(features).difference(test.columns)
            )
            if missing_features:
                missing = ", ".join(sorted(missing_features))
                raise ValueError(f"Missing features: {missing}")
            if not artifact_file.exists():
                raise FileNotFoundError(f"Missing {repo_relative(artifact_file)}")

            background_pool = train[features].dropna()
            explain_pool = test[features].dropna()
            background = background_pool.sample(
                min(N_BACKGROUND, len(background_pool)),
                random_state=RANDOM_STATE,
            )
            explain = explain_pool.sample(
                min(N_EXPLAIN, len(explain_pool)),
                random_state=RANDOM_STATE,
            )
            if background.empty or explain.empty:
                raise ValueError("Background or explanation data is empty.")

            artifact = joblib.load(artifact_file)
            shap_values = calculate_shap_values(artifact, background, explain)

            model_output = dataset_output / safe_name(model_name)
            model_output.mkdir(parents=True, exist_ok=True)
            shap_file = model_output / f"{safe_name(model_name)}_shap_values.csv"
            explain_file = model_output / f"{safe_name(model_name)}_X_explain.csv"
            summary_file = model_output / f"{safe_name(model_name)}_SHAP_summary_600dpi.png"
            bar_file = model_output / f"{safe_name(model_name)}_SHAP_bar_600dpi.png"
            dependence_file = (
                model_output / f"{safe_name(model_name)}_SHAP_dependence_600dpi.png"
            )

            pd.DataFrame(shap_values, columns=features).to_csv(
                shap_file,
                index=False,
            )
            explain.to_csv(explain_file, index=False)
            plot_shap_summary(shap_values, explain, summary_file)
            plot_shap_bar(model_name, shap_values, features, bar_file)
            dependence_created = plot_shap_dependence(
                shap_values,
                explain,
                dependence_file,
            )

            manifest_rows.append(
                {
                    "Dataset": dataset,
                    "Family": family,
                    "Model": model_name,
                    "Status": "ok",
                    "Artifact": repo_relative(artifact_file),
                    "SHAP_Values": repo_relative(shap_file),
                    "X_Explain": repo_relative(explain_file),
                    "Summary_Figure": repo_relative(summary_file),
                    "Bar_Figure": repo_relative(bar_file),
                    "Dependence_Figure": (
                        repo_relative(dependence_file) if dependence_created else ""
                    ),
                }
            )
            print(f"Completed {dataset}: {model_name}")
        except Exception as error:
            manifest_rows.append(
                {
                    "Dataset": dataset,
                    "Family": family,
                    "Model": model_name,
                    "Status": f"failed: {error}",
                    "Artifact": str(row.Artifact),
                }
            )
            print(f"Skipped {dataset}: {model_name} ({error})")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for dataset in DATASETS:
        try:
            run_dataset(dataset, manifest_rows)
        except (FileNotFoundError, ValueError) as error:
            manifest_rows.append(
                {
                    "Dataset": dataset,
                    "Family": "",
                    "Model": "",
                    "Status": f"failed: {error}",
                    "Artifact": "",
                }
            )
            print(f"Skipped {dataset}: {error}")

    pd.DataFrame(manifest_rows).to_csv(
        OUTPUT_DIR / "retraining_shap_manifest.csv",
        index=False,
    )

    print(f"SHAP analysis saved to {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
