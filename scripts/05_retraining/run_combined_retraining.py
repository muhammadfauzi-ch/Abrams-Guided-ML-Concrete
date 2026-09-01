from pathlib import Path
import json
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


ROOT = Path(__file__).resolve().parents[2]

LITERATURE_FILE = ROOT / "data" / "processed" / "literature_train_ready.csv"
EXPERIMENT_FILES = {
    "exp1": ROOT / "data" / "processed" / "finaldataexp1.csv",
    "exp2": ROOT / "data" / "processed" / "finaldataexp2.csv",
    "exp12": ROOT / "data" / "processed" / "finaldataexp12.csv",
}

OUTPUT_DIR = ROOT / "outputs" / "retraining"

RANDOM_STATE = 42
TEST_SIZE = 0.20
SCENARIO = "Combined literature-experimental retraining analysis"

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

ZERO_ALLOWED_FEATURES = {
    "SCM1_kgm3": 1e-6,
    "SCM2_kgm3": 1e-6,
    "SP_kgm3": 1e-6,
    "Fiber_kgm3": 1e-6,
}


def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = y_true != 0

    mape = np.nan
    if valid.any():
        mape = np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100

    rmse = calculate_rmse(y_true, y_pred)

    return {
        "R2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
        "RMSE": rmse,
        "MAE": mean_absolute_error(y_true, y_pred),
        "MBE": np.mean(y_true - y_pred),
        "MAPE": mape,
        "RRMSE": rmse / np.mean(y_true) * 100,
        "A20": np.mean((y_pred >= 0.8 * y_true) & (y_pred <= 1.2 * y_true)),
    }


def safe_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def repo_relative(path):
    return path.relative_to(ROOT).as_posix()


def read_dataset(path, source_tag):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {repo_relative(path)}")

    data = pd.read_csv(path)
    required_columns = set(FULL_FEATURES + [TARGET])
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing columns in {repo_relative(path)}: {missing}")

    data = data[FULL_FEATURES + [TARGET]].copy()
    for column in FULL_FEATURES + [TARGET]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Missing SCM, admixture, and fiber values indicate that the constituent was
    # not used. Core mixture variables must remain observed and physically valid.
    for column in ZERO_ALLOWED_FEATURES:
        data[column] = data[column].fillna(0.0).clip(lower=0.0)

    observed_columns = [
        "Age_day",
        "Cement_kgm3",
        "Water_kgm3",
        "W_over_B",
        "FineAgg_kgm3",
        "CoarseAgg_kgm3",
        TARGET,
    ]
    positive_columns = [
        "Age_day",
        "Cement_kgm3",
        "Water_kgm3",
        "W_over_B",
        "FineAgg_kgm3",
        TARGET,
    ]
    data = data.dropna(subset=observed_columns)
    data = data[(data[positive_columns] > 0).all(axis=1)]
    data = data[data["CoarseAgg_kgm3"] >= 0].copy()
    data["SourceTag"] = source_tag

    return data.reset_index(drop=True)


def prepare_merged_dataset(experiment_tag):
    literature = read_dataset(LITERATURE_FILE, "literature")
    experiment_file = EXPERIMENT_FILES[experiment_tag]
    experiment = read_dataset(experiment_file, experiment_tag)

    merged = pd.concat([literature, experiment], ignore_index=True)

    return merged, LITERATURE_FILE, experiment_file


def build_stratify_labels(data):
    source = data["SourceTag"].astype(str)
    age = data["Age_day"].round().astype(int).astype(str)
    labels = source + "__age_" + age

    counts = labels.value_counts()
    rare = labels.isin(counts[counts < 2].index)
    labels = labels.where(~rare, source + "__other_age")

    if (labels.value_counts() < 2).any():
        labels = source

    return labels


def prepare_powerlaw_features(data):
    transformed = data[GUIDED_FEATURES].copy().astype(float)

    for feature in GUIDED_FEATURES:
        if feature in ZERO_ALLOWED_FEATURES:
            transformed[feature] = (
                transformed[feature].clip(lower=0.0) + ZERO_ALLOWED_FEATURES[feature]
            )

    return transformed


def prior_design_matrix(data, prior_type):
    if prior_type == "classical_abrams":
        return np.column_stack(
            [
                np.ones(len(data)),
                data["W_over_B"].values,
            ]
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
        powerlaw_data = prepare_powerlaw_features(data)
        log_features = [
            np.log(powerlaw_data[feature].values) for feature in GUIDED_FEATURES
        ]
        return np.column_stack([np.ones(len(data)), *log_features])

    raise ValueError(f"Unknown prior type: {prior_type}")


def fit_prior(data, target, prior_type):
    design_matrix = prior_design_matrix(data, prior_type)
    log_strength = np.log(np.asarray(target, dtype=float))
    coefficients, *_ = np.linalg.lstsq(
        design_matrix,
        log_strength,
        rcond=None,
    )

    return {
        "artifact_type": "prior",
        "prior_type": prior_type,
        "coefficients": coefficients,
        "features": prior_features(prior_type),
    }


def prior_features(prior_type):
    if prior_type == "classical_abrams":
        return ["W_over_B"]
    if prior_type == "abrams_age":
        return ["W_over_B", "Age_day"]
    if prior_type == "multivariable":
        return GUIDED_FEATURES

    raise ValueError(f"Unknown prior type: {prior_type}")


def predict_prior(artifact, data):
    design_matrix = prior_design_matrix(data, artifact["prior_type"])
    return np.exp(design_matrix @ artifact["coefficients"])


def build_ml_models():
    models = {
        "MLR": LinearRegression(),
        "SVR": SVR(
            kernel="rbf",
            C=100,
            epsilon=0.1,
            gamma="scale",
        ),
        "KNN": KNeighborsRegressor(
            n_neighbors=5,
            weights="distance",
        ),
        "RF": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GBR": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
        )

    if CatBoostRegressor is not None:
        models["CatBoost"] = CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            verbose=0,
            random_seed=RANDOM_STATE,
        )

    return models


def fit_ml_model(model, X_train, y_train, features):
    fitted_model = clone(model)
    fitted_model.fit(X_train[features], y_train)

    return {
        "artifact_type": "ml",
        "model": fitted_model,
        "features": list(features),
    }


def fit_guided_model(prior_type, residual_model, X_train, y_train):
    prior = fit_prior(X_train, y_train, prior_type)
    prior_prediction = predict_prior(prior, X_train)
    residual = np.asarray(y_train, dtype=float) - prior_prediction

    fitted_residual_model = clone(residual_model)
    fitted_residual_model.fit(X_train[GUIDED_FEATURES], residual)

    return {
        "artifact_type": "guided",
        "prior": prior,
        "residual_model": fitted_residual_model,
        "features": GUIDED_FEATURES,
    }


def predict_artifact(artifact, data):
    if artifact["artifact_type"] == "prior":
        return predict_prior(artifact, data)
    if artifact["artifact_type"] == "ml":
        return artifact["model"].predict(data[artifact["features"]])
    if artifact["artifact_type"] == "guided":
        prior_prediction = predict_prior(artifact["prior"], data)
        residual_prediction = artifact["residual_model"].predict(
            data[artifact["features"]]
        )
        return prior_prediction + residual_prediction

    raise ValueError(f"Unknown artifact type: {artifact['artifact_type']}")


def append_metrics(rows, dataset, family, model_name, split, y_true, y_pred):
    rows.append(
        {
            "Dataset": dataset,
            "Scenario": SCENARIO,
            "Family": family,
            "Model": model_name,
            "Split": split,
            "N": len(y_true),
            **regression_metrics(y_true, y_pred),
        }
    )


def append_predictions(
    rows,
    dataset,
    family,
    model_name,
    split,
    split_data,
    y_true,
    y_pred,
):
    rows.append(
        pd.DataFrame(
            {
                "Dataset": dataset,
                "Scenario": SCENARIO,
                "Family": family,
                "Model": model_name,
                "Split": split,
                "Row_ID": np.arange(len(split_data)),
                "SourceTag": split_data["SourceTag"].values,
                "y_true": np.asarray(y_true, dtype=float),
                "y_pred": np.asarray(y_pred, dtype=float),
            }
        )
    )


def save_artifact(artifact, dataset_dir, family, model_name):
    model_dir = dataset_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_file = model_dir / f"{safe_name(family)}__{safe_name(model_name)}.joblib"
    joblib.dump(artifact, artifact_file)

    return artifact_file


def evaluate_artifact(
    artifact,
    dataset_tag,
    family,
    model_name,
    train,
    test,
    metrics_rows,
    prediction_rows,
):
    train_prediction = predict_artifact(artifact, train)
    test_prediction = predict_artifact(artifact, test)

    append_metrics(
        metrics_rows,
        dataset_tag,
        family,
        model_name,
        "Train",
        train[TARGET],
        train_prediction,
    )
    append_metrics(
        metrics_rows,
        dataset_tag,
        family,
        model_name,
        "Test",
        test[TARGET],
        test_prediction,
    )
    append_predictions(
        prediction_rows,
        dataset_tag,
        family,
        model_name,
        "Train",
        train,
        train[TARGET],
        train_prediction,
    )
    append_predictions(
        prediction_rows,
        dataset_tag,
        family,
        model_name,
        "Test",
        test,
        test[TARGET],
        test_prediction,
    )


def build_wide_metrics(metrics):
    train = metrics[metrics["Split"] == "Train"].copy()
    test = metrics[metrics["Split"] == "Test"].copy()
    keys = ["Dataset", "Scenario", "Family", "Model"]
    values = ["N", "R2", "RMSE", "MAE", "MBE", "MAPE", "RRMSE", "A20"]

    train = train[keys + values].rename(
        columns={column: f"{column}_Train" for column in values}
    )
    test = test[keys + values].rename(
        columns={column: f"{column}_Test" for column in values}
    )

    return pd.merge(train, test, on=keys, how="outer")


def run_dataset(experiment_tag):
    dataset_dir = OUTPUT_DIR / experiment_tag
    dataset_dir.mkdir(parents=True, exist_ok=True)

    merged, literature_file, experiment_file = prepare_merged_dataset(experiment_tag)
    stratify_labels = build_stratify_labels(merged)
    train, test = train_test_split(
        merged,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_labels,
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    merged.to_csv(dataset_dir / f"merged_literature_{experiment_tag}.csv", index=False)
    train.to_csv(dataset_dir / f"{experiment_tag}_train.csv", index=False)
    test.to_csv(dataset_dir / f"{experiment_tag}_test.csv", index=False)

    metrics_rows = []
    prediction_rows = []
    artifact_rows = []

    prior_specs = {
        "Classical Abrams": "classical_abrams",
        "Abrams": "abrams_age",
        "Multivariable Abrams": "multivariable",
    }
    for model_name, prior_type in prior_specs.items():
        artifact = fit_prior(train, train[TARGET], prior_type)
        artifact_file = save_artifact(artifact, dataset_dir, "Prior", model_name)
        evaluate_artifact(
            artifact,
            experiment_tag,
            "Prior",
            model_name,
            train,
            test,
            metrics_rows,
            prediction_rows,
        )
        artifact_rows.append(
            {
                "Dataset": experiment_tag,
                "Family": "Prior",
                "Model": model_name,
                "Artifact": repo_relative(artifact_file),
                "Features": json.dumps(artifact["features"]),
            }
        )

    ml_models = build_ml_models()
    for model_name, model in ml_models.items():
        artifact = fit_ml_model(model, train, train[TARGET], FULL_FEATURES)
        artifact_file = save_artifact(artifact, dataset_dir, "ML-only", model_name)
        evaluate_artifact(
            artifact,
            experiment_tag,
            "ML-only",
            model_name,
            train,
            test,
            metrics_rows,
            prediction_rows,
        )
        artifact_rows.append(
            {
                "Dataset": experiment_tag,
                "Family": "ML-only",
                "Model": model_name,
                "Artifact": repo_relative(artifact_file),
                "Features": json.dumps(FULL_FEATURES),
            }
        )

    for model_name, model in ml_models.items():
        guided_name = f"Abrams-{model_name}"
        artifact = fit_guided_model("abrams_age", model, train, train[TARGET])
        artifact_file = save_artifact(
            artifact,
            dataset_dir,
            "Abrams-guided",
            guided_name,
        )
        evaluate_artifact(
            artifact,
            experiment_tag,
            "Abrams-guided",
            guided_name,
            train,
            test,
            metrics_rows,
            prediction_rows,
        )
        artifact_rows.append(
            {
                "Dataset": experiment_tag,
                "Family": "Abrams-guided",
                "Model": guided_name,
                "Artifact": repo_relative(artifact_file),
                "Features": json.dumps(GUIDED_FEATURES),
            }
        )

    for model_name, model in ml_models.items():
        guided_name = f"Multivariable-{model_name}"
        artifact = fit_guided_model("multivariable", model, train, train[TARGET])
        artifact_file = save_artifact(
            artifact,
            dataset_dir,
            "Multivariable-guided",
            guided_name,
        )
        evaluate_artifact(
            artifact,
            experiment_tag,
            "Multivariable-guided",
            guided_name,
            train,
            test,
            metrics_rows,
            prediction_rows,
        )
        artifact_rows.append(
            {
                "Dataset": experiment_tag,
                "Family": "Multivariable-guided",
                "Model": guided_name,
                "Artifact": repo_relative(artifact_file),
                "Features": json.dumps(GUIDED_FEATURES),
            }
        )

    metrics = pd.DataFrame(metrics_rows).round(6)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    artifact_manifest = pd.DataFrame(artifact_rows)

    train_summary = metrics[metrics["Split"] == "Train"].copy()
    train_summary = train_summary.sort_values(["Family", "Model"]).reset_index(drop=True)
    test_summary = metrics[metrics["Split"] == "Test"].copy()
    test_summary = test_summary.sort_values(["Family", "Model"]).reset_index(drop=True)

    metrics.to_csv(
        dataset_dir / f"{experiment_tag}_combined_retraining_metrics_long.csv",
        index=False,
    )
    predictions.to_csv(
        dataset_dir / f"{experiment_tag}_combined_retraining_predictions_long.csv",
        index=False,
    )
    train_summary.to_csv(
        dataset_dir / f"{experiment_tag}_combined_retraining_train_summary.csv",
        index=False,
    )
    test_summary.to_csv(
        dataset_dir / f"{experiment_tag}_combined_retraining_test_summary.csv",
        index=False,
    )
    build_wide_metrics(metrics).to_csv(
        dataset_dir / f"{experiment_tag}_combined_retraining_metrics_wide.csv",
        index=False,
    )
    artifact_manifest.to_csv(
        dataset_dir / f"{experiment_tag}_model_artifacts.csv",
        index=False,
    )

    metadata = {
        "scenario_name": SCENARIO,
        "literature_file": repo_relative(literature_file),
        "experimental_file": repo_relative(experiment_file),
        "n_total": int(len(merged)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "full_features": FULL_FEATURES,
        "guided_features": GUIDED_FEATURES,
        "models_included": sorted(metrics["Model"].unique().tolist()),
    }
    metadata_file = dataset_dir / f"{experiment_tag}_run_metadata.json"
    metadata_file.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Completed {experiment_tag}: {dataset_dir.relative_to(ROOT)}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for experiment_tag in EXPERIMENT_FILES:
        run_dataset(experiment_tag)


if __name__ == "__main__":
    main()
