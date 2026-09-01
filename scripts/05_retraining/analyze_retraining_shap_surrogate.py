from pathlib import Path
import json
import re
import warnings

import matplotlib
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


matplotlib.use("Agg")
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
RETRAINING_DIR = ROOT / "outputs" / "retraining"
OUTPUT_DIR = RETRAINING_DIR / "shap_surrogate"

DATASETS = ["exp1", "exp2", "exp12"]

RANDOM_STATE = 42
N_BACKGROUND = 120
N_EXPLAIN = 180
TOP_DEPENDENCE_FEATURES = 4
DPI = 600

TARGET = "fc_MPa"

FULL_FEATURES = [
    "Age_day",
    "Cement_kgm3",
    "Water_kgm3",
    "W_over_B",
    "SCM1_kgm3",
    "SCM2_kgm3",
    "FineAgg_kgm3",
    "CoarseAgg_kgm3",
    "SP_kgm3",
    "Fiber_kgm3",
]

GUIDED_FEATURES = [
    "Age_day",
    "Cement_kgm3",
    "W_over_B",
    "SCM1_kgm3",
    "SCM2_kgm3",
    "FineAgg_kgm3",
    "SP_kgm3",
    "Fiber_kgm3",
]

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

SURROGATE_PARAMETERS = {
    "n_estimators": 400,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

SHAP_COLORMAP = LinearSegmentedColormap.from_list(
    "blue_pink_shap",
    ["#1e88ff", "#7b61ff", "#d946ef", "#ff2d7a"],
)


def safe_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def repo_relative(path):
    return path.relative_to(ROOT).as_posix()


def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def regression_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def style_axes(axis):
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_color("black")

    axis.grid(True, linestyle="--", alpha=0.25)
    axis.tick_params(axis="both", labelsize=10)


def feature_set(family, model_name):
    if family == "Prior":
        if model_name == "Classical Abrams":
            return ["W_over_B"]
        if model_name == "Abrams":
            return ["W_over_B", "Age_day"]
        return GUIDED_FEATURES
    if family == "ML-only":
        return FULL_FEATURES

    return GUIDED_FEATURES


def read_retraining_tables(dataset):
    dataset_dir = RETRAINING_DIR / dataset
    train_file = dataset_dir / f"{dataset}_train.csv"
    test_file = dataset_dir / f"{dataset}_test.csv"
    prediction_file = (
        dataset_dir / f"{dataset}_combined_retraining_predictions_long.csv"
    )

    missing = [
        path for path in [train_file, test_file, prediction_file] if not path.exists()
    ]
    if missing:
        paths = ", ".join(repo_relative(path) for path in missing)
        raise FileNotFoundError(f"Missing retraining files: {paths}")

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    predictions = pd.read_csv(prediction_file)

    return train, test, predictions, train_file, test_file, prediction_file


def prepare_features(train, test):
    required_columns = set(FULL_FEATURES + [TARGET])
    for data, split in [(train, "Train"), (test, "Test")]:
        missing_columns = required_columns.difference(data.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing {split} columns: {missing}")

    train = train[FULL_FEATURES + [TARGET]].copy()
    test = test[FULL_FEATURES + [TARGET]].copy()
    train["Split"] = "Train"
    test["Split"] = "Test"
    train["Row_ID"] = np.arange(len(train))
    test["Row_ID"] = np.arange(len(test))

    return pd.concat([train, test], ignore_index=True)


def prepare_predictions(predictions):
    required_columns = {"Family", "Model", "Split", "y_true", "y_pred"}
    missing_columns = required_columns.difference(predictions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing prediction columns: {missing}")

    predictions = predictions.copy()
    predictions["Family"] = predictions["Family"].astype(str).str.strip()
    predictions["Model"] = predictions["Model"].astype(str).str.strip()
    predictions["Split"] = predictions["Split"].astype(str).str.strip().str.title()
    predictions["y_true"] = pd.to_numeric(predictions["y_true"], errors="coerce")
    predictions["y_pred"] = pd.to_numeric(predictions["y_pred"], errors="coerce")

    if "Row_ID" not in predictions.columns:
        predictions["Row_ID"] = predictions.groupby(
            ["Family", "Model", "Split"],
            sort=False,
        ).cumcount()
    predictions["Row_ID"] = pd.to_numeric(predictions["Row_ID"], errors="coerce")

    return predictions.dropna(subset=["Row_ID", "y_true", "y_pred"])


def merge_features_predictions(features, predictions):
    merged = predictions.merge(
        features,
        on=["Split", "Row_ID"],
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise ValueError("Feature and prediction tables did not match.")

    mismatch = np.abs(merged["y_true"] - merged[TARGET])
    if (mismatch > 1e-8).any():
        raise ValueError("Prediction rows are not aligned with the feature tables.")

    return merged


def choose_color_feature(feature_data, feature):
    correlations = {}

    for candidate in feature_data.columns:
        if candidate == feature:
            continue
        correlations[candidate] = abs(
            feature_data[feature].corr(feature_data[candidate])
        )

    correlations = pd.Series(correlations, dtype=float).dropna()
    if correlations.empty:
        return feature

    return correlations.idxmax()


def plot_shap_bar(model_name, shap_values, features, output_file):
    importance = pd.DataFrame(shap_values, columns=features).abs().mean()
    importance = importance.sort_values(ascending=True)

    figure, axis = plt.subplots(figsize=(8, 6))
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

    axis.set_xlabel("Mean absolute surrogate SHAP value", fontsize=12, fontweight="bold")
    axis.set_title(model_name, fontsize=14, fontweight="bold", pad=10)
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
            color_values = explain[color_feature].values

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
            f"Surrogate SHAP value for {DISPLAY_LABELS.get(feature, feature)}",
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


def plot_abrams_relationship(model_name, data, output_file):
    data = data.dropna(subset=["W_over_B", TARGET, "y_pred"]).copy()
    if data.empty:
        return False

    data = data.sort_values("W_over_B").reset_index(drop=True)
    window = max(5, min(25, len(data) // 8))
    smoothed = data["y_pred"].rolling(
        window=window,
        center=True,
        min_periods=1,
    ).mean()

    figure, axis = plt.subplots(figsize=(7.2, 5.6))
    axis.scatter(
        data["W_over_B"],
        data[TARGET],
        s=24,
        facecolors="none",
        edgecolors="#7F8C8D",
        linewidths=0.8,
        alpha=0.85,
        label="Actual",
    )
    axis.scatter(
        data["W_over_B"],
        data["y_pred"],
        s=24,
        facecolors="#E74C3C",
        edgecolors="#7a1d1d",
        linewidths=0.35,
        alpha=0.55,
        label="Predicted",
    )
    axis.plot(
        data["W_over_B"],
        smoothed,
        color="#8E244D",
        linewidth=2.2,
        label="Smoothed prediction",
    )
    axis.fill_between(
        data["W_over_B"],
        smoothed * 0.90,
        smoothed * 1.10,
        color="#C9D8EA",
        alpha=0.65,
    )
    axis.set_xlabel("W/B")
    axis.set_ylabel("Compressive strength (MPa)")
    axis.set_title(model_name, fontsize=14, fontweight="bold", pad=10)
    style_axes(axis)
    axis.legend(loc="upper right")

    median_age = np.nanmedian(data["Age_day"])
    axis.text(
        0.98,
        0.98,
        f"Median age = {median_age:.0f} day\nn = {len(data)}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#999999",
            "linewidth": 1.0,
            "alpha": 0.92,
        },
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close(figure)

    return True


def run_model(dataset, family, model_name, model_data, dataset_output):
    features = feature_set(family, model_name)
    train = model_data[model_data["Split"] == "Train"].copy()
    test = model_data[model_data["Split"] == "Test"].copy()

    X_train = train[features].dropna()
    X_test = test[features].dropna()
    if len(X_train) < 20 or len(X_test) < 10:
        raise ValueError("Too few rows for a train/test surrogate analysis.")

    y_train = train.loc[X_train.index, "y_pred"]
    y_test = test.loc[X_test.index, "y_pred"]

    surrogate = RandomForestRegressor(**SURROGATE_PARAMETERS)
    surrogate.fit(X_train, y_train)

    train_prediction = surrogate.predict(X_train)
    test_prediction = surrogate.predict(X_test)
    train_metrics = regression_metrics(y_train, train_prediction)
    test_metrics = regression_metrics(y_test, test_prediction)

    model_output = dataset_output / safe_name(model_name)
    model_output.mkdir(parents=True, exist_ok=True)

    fidelity_file = model_output / f"{safe_name(model_name)}_surrogate_fidelity.csv"
    pd.DataFrame(
        [
            {"Split": "Train", **train_metrics},
            {"Split": "Test", **test_metrics},
        ]
    ).to_csv(fidelity_file, index=False)

    background = X_train.sample(
        min(N_BACKGROUND, len(X_train)),
        random_state=RANDOM_STATE,
    )
    explain = X_test.sample(
        min(N_EXPLAIN, len(X_test)),
        random_state=RANDOM_STATE,
    )
    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(explain)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)

    shap_file = model_output / f"{safe_name(model_name)}_shap_values.csv"
    explain_file = model_output / f"{safe_name(model_name)}_X_explain.csv"
    summary_file = model_output / f"{safe_name(model_name)}_SHAP_summary_600dpi.png"
    bar_file = model_output / f"{safe_name(model_name)}_SHAP_bar_600dpi.png"
    dependence_file = (
        model_output / f"{safe_name(model_name)}_SHAP_dependence_600dpi.png"
    )
    abrams_file = model_output / f"{safe_name(model_name)}_Abrams_plot_600dpi.png"

    pd.DataFrame(shap_values, columns=features).to_csv(shap_file, index=False)
    explain.to_csv(explain_file, index=False)
    plot_shap_summary(shap_values, explain, summary_file)
    plot_shap_bar(model_name, shap_values, features, bar_file)
    dependence_created = plot_shap_dependence(
        shap_values,
        explain,
        dependence_file,
    )
    abrams_created = plot_abrams_relationship(model_name, model_data, abrams_file)

    return {
        "Dataset": dataset,
        "Family": family,
        "Model": model_name,
        "Status": "ok",
        "Test_Fidelity_R2": test_metrics["R2"],
        "Test_Fidelity_RMSE": test_metrics["RMSE"],
        "Output_Directory": repo_relative(model_output),
        "Fidelity_CSV": repo_relative(fidelity_file),
        "SHAP_Values": repo_relative(shap_file),
        "X_Explain": repo_relative(explain_file),
        "Summary_Figure": repo_relative(summary_file),
        "Bar_Figure": repo_relative(bar_file),
        "Dependence_Figure": (
            repo_relative(dependence_file) if dependence_created else ""
        ),
        "Abrams_Figure": repo_relative(abrams_file) if abrams_created else "",
    }


def run_dataset(dataset, manifest_rows):
    train, test, predictions, train_file, test_file, prediction_file = (
        read_retraining_tables(dataset)
    )
    features = prepare_features(train, test)
    predictions = prepare_predictions(predictions)
    merged = merge_features_predictions(features, predictions)

    dataset_output = OUTPUT_DIR / dataset
    dataset_output.mkdir(parents=True, exist_ok=True)
    merged_file = dataset_output / f"{dataset}_merged_features_predictions.csv"
    merged.to_csv(merged_file, index=False)

    metadata = {
        "dataset": dataset,
        "train_file": repo_relative(train_file),
        "test_file": repo_relative(test_file),
        "prediction_file": repo_relative(prediction_file),
        "merged_file": repo_relative(merged_file),
        "n_train": len(train),
        "n_test": len(test),
        "method": "Random-forest surrogate fitted to saved model predictions",
        "interpretation": (
            "Surrogate SHAP explains the surrogate approximation, not the original "
            "model. Test fidelity metrics must be considered when interpreting it."
        ),
    }
    metadata_file = dataset_output / f"{dataset}_surrogate_shap_metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    model_groups = merged[["Family", "Model"]].drop_duplicates()
    for row in model_groups.itertuples(index=False):
        family = str(row.Family)
        model_name = str(row.Model)
        model_data = merged[
            (merged["Family"] == family) & (merged["Model"] == model_name)
        ].copy()

        try:
            result = run_model(
                dataset,
                family,
                model_name,
                model_data,
                dataset_output,
            )
            manifest_rows.append(result)
            print(f"Completed {dataset}: {model_name}")
        except Exception as error:
            manifest_rows.append(
                {
                    "Dataset": dataset,
                    "Family": family,
                    "Model": model_name,
                    "Status": f"failed: {error}",
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
                }
            )
            print(f"Skipped {dataset}: {error}")

    pd.DataFrame(manifest_rows).to_csv(
        OUTPUT_DIR / "retraining_surrogate_shap_manifest.csv",
        index=False,
    )

    print(f"Surrogate SHAP analysis saved to {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
