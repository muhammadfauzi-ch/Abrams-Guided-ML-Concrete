import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from _interpretation_common import (
    DISPLAY_LABELS,
    FULL_FEATURES,
    INTERPRETATION_DIR,
    SPLIT_DIR,
    TARGET,
    read_split,
)


warnings.filterwarnings("ignore")

OUTPUT_DIR = INTERPRETATION_DIR / "feature_reduction_exploratory"
TOP_K = 6
RANDOM_STATE = 42
DPI = 600


def build_model():
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def shap_ranking(model, features):
    values = np.asarray(shap.TreeExplainer(model).shap_values(features))
    ranking = pd.DataFrame(
        {
            "Feature": features.columns,
            "MeanAbsSHAP": np.abs(values).mean(axis=0),
        }
    ).sort_values("MeanAbsSHAP", ascending=False, ignore_index=True)
    ranking["Rank"] = np.arange(1, len(ranking) + 1)
    ranking["Selected"] = ranking["Rank"] <= TOP_K
    return ranking


def plot_ranking(ranking):
    ordered = ranking.sort_values("MeanAbsSHAP")
    colors = np.where(ordered["Selected"], "#1f77b4", "#c9d1d9")
    fig, axis = plt.subplots(figsize=(8.3, 5.8))
    axis.barh(
        [DISPLAY_LABELS.get(name, name) for name in ordered["Feature"]],
        ordered["MeanAbsSHAP"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    axis.set_xlabel("Mean absolute SHAP value (MPa)")
    axis.set_title("Exploratory SHAP-guided feature ranking", fontweight="bold")
    axis.grid(axis="x", linestyle="--", alpha=0.22)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "shap_feature_ranking_600dpi.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


def comparison_table(y_train, y_test, predictions):
    rows = []
    for scenario, (train_pred, test_pred) in predictions.items():
        for split, observed, predicted in (
            ("Train", y_train, train_pred),
            ("Test", y_test, test_pred),
        ):
            rows.append(
                {
                    "Scenario": scenario,
                    "Dataset": split,
                    **metrics(observed, predicted),
                }
            )
    return pd.DataFrame(rows)


def plot_comparison(results):
    test = results.loc[results["Dataset"] == "Test"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for axis, metric in zip(axes, ["R2", "RMSE", "MAE"]):
        bars = axis.bar(
            test["Scenario"],
            test[metric],
            color=["#4f9df7", "#e76f51"],
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, value in zip(bars, test[metric]):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axis.set_title(metric, fontweight="bold")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Random forest before and after feature reduction", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(
        OUTPUT_DIR / "feature_reduction_test_metrics_600dpi.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = read_split(SPLIT_DIR / "train_full.csv", FULL_FEATURES)
    test = read_split(SPLIT_DIR / "test_full.csv", FULL_FEATURES)

    x_train = train[FULL_FEATURES]
    x_test = test[FULL_FEATURES]
    y_train = train[TARGET]
    y_test = test[TARGET]

    full_model = build_model()
    full_model.fit(x_train, y_train)
    ranking = shap_ranking(full_model, x_train)
    selected = ranking.loc[ranking["Selected"], "Feature"].tolist()

    reduced_model = build_model()
    reduced_model.fit(x_train[selected], y_train)

    results = comparison_table(
        y_train,
        y_test,
        {
            "All features": (
                full_model.predict(x_train),
                full_model.predict(x_test),
            ),
            "Selected features": (
                reduced_model.predict(x_train[selected]),
                reduced_model.predict(x_test[selected]),
            ),
        },
    )

    ranking.to_csv(OUTPUT_DIR / "shap_feature_ranking.csv", index=False)
    results.to_csv(OUTPUT_DIR / "feature_reduction_metrics.csv", index=False)
    train[selected + [TARGET]].to_csv(
        OUTPUT_DIR / "train_selected_features.csv", index=False
    )
    test[selected + [TARGET]].to_csv(
        OUTPUT_DIR / "test_selected_features.csv", index=False
    )

    metadata = {
        "scope": "exploratory; does not replace the main modeling pipeline",
        "model": "RandomForestRegressor",
        "n_estimators": 300,
        "random_state": RANDOM_STATE,
        "selection_rule": f"top_{TOP_K}_mean_absolute_shap",
        "selected_features": selected,
        "removed_features": [name for name in FULL_FEATURES if name not in selected],
    }
    with (OUTPUT_DIR / "feature_reduction_metadata.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, indent=2)

    joblib.dump(full_model, OUTPUT_DIR / "rf_all_features.joblib")
    joblib.dump(reduced_model, OUTPUT_DIR / "rf_selected_features.joblib")
    plot_ranking(ranking)
    plot_comparison(results)

    print("Exploratory feature-reduction analysis completed.")
    print(f"Selected features: {selected}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
