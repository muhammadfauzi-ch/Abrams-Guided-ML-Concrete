from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "outputs" / "splits"
INTERPRETATION_DIR = ROOT / "outputs" / "interpretation"
MODEL_DIR = INTERPRETATION_DIR / "models"
SHAP_DIR = INTERPRETATION_DIR / "shap"
PDP_DIR = INTERPRETATION_DIR / "pdp_ice"

TARGET = "fc_MPa"
RANDOM_STATE = 42

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

DISPLAY_LABELS = {
    "Age_day": "Age (day)",
    "Cement_kgm3": "Cement (kg m$^{-3}$)",
    "Water_kgm3": "Water (kg m$^{-3}$)",
    "W_over_B": "Water-to-binder ratio (-)",
    "SCM1_kgm3": "SCM1 (kg m$^{-3}$)",
    "SCM2_kgm3": "SCM2 (kg m$^{-3}$)",
    "FineAgg_kgm3": "Fine aggregate (kg m$^{-3}$)",
    "CoarseAgg_kgm3": "Coarse aggregate (kg m$^{-3}$)",
    "SP_kgm3": "SP (kg m$^{-3}$)",
    "Fiber_kgm3": "Fiber (kg m$^{-3}$)",
}

MANUSCRIPT_MODELS = [
    "AbramsGuided_XGBoost",
    "MultivariableGuided_XGBoost",
    "AbramsGuided_CatBoost",
    "MultivariableGuided_CatBoost",
]


def safe_name(name):
    return name.replace(" ", "_").replace("/", "_")


def read_split(path, features):
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}. Run the data-splitting stage first."
        )

    data = pd.read_csv(path)
    required = features + [TARGET]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    data = data[required].copy()
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=required).reset_index(drop=True)


def prepare_abrams_data(data):
    valid = (
        (data[TARGET] > 0)
        & (data["W_over_B"] > 0)
        & (data["Age_day"] > 0)
    )
    return data.loc[valid].reset_index(drop=True)


def prepare_multivariable_data(data):
    data = filter_multivariable_rows(data)
    for feature in GUIDED_FEATURES:
        if feature in ZERO_ALLOWED_FEATURES:
            data[feature] = (
                data[feature].clip(lower=0) + ZERO_ALLOWED_FEATURES[feature]
            )
    return data.reset_index(drop=True)


def filter_multivariable_rows(data):
    data = data.loc[data[TARGET] > 0].copy()
    for feature in GUIDED_FEATURES:
        if feature not in ZERO_ALLOWED_FEATURES:
            data = data.loc[data[feature] > 0].copy()
    return data.reset_index(drop=True)


def fit_abrams_prior(data):
    design = np.column_stack(
        [
            np.ones(len(data)),
            np.log(data["W_over_B"].to_numpy()),
            np.log(data["Age_day"].to_numpy()),
        ]
    )
    coefficients, *_ = np.linalg.lstsq(
        design, np.log(data[TARGET].to_numpy()), rcond=None
    )
    return coefficients


def predict_abrams_prior(data, coefficients):
    b0, b1, b2 = coefficients
    return np.exp(
        b0
        + b1 * np.log(data["W_over_B"].to_numpy())
        + b2 * np.log(data["Age_day"].to_numpy())
    )


def multivariable_design(data):
    terms = [np.log(data[feature].to_numpy()) for feature in GUIDED_FEATURES]
    return np.column_stack([np.ones(len(data)), *terms])


def fit_multivariable_prior(data):
    coefficients, *_ = np.linalg.lstsq(
        multivariable_design(data),
        np.log(data[TARGET].to_numpy()),
        rcond=None,
    )
    return coefficients


def predict_multivariable_prior(data, coefficients):
    return np.exp(multivariable_design(data) @ coefficients)


def build_manual_models():
    return {
        "MLR": LinearRegression(),
        "SVR": SVR(kernel="rbf", C=100, epsilon=0.1, gamma="scale"),
        "KNN": KNeighborsRegressor(n_neighbors=5, weights="distance"),
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
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
        ),
        "CatBoost": CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            verbose=0,
            random_seed=RANDOM_STATE,
        ),
    }


def save_artifact(model_name, family, features, estimator, prior=None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_name": model_name,
        "family": family,
        "features": list(features),
        "estimator": estimator,
        "prior": prior,
    }
    path = MODEL_DIR / f"{safe_name(model_name)}.joblib"
    joblib.dump(artifact, path)
    return path


def train_interpretation_models():
    full_train = read_split(SPLIT_DIR / "train_full.csv", FULL_FEATURES)
    guided_train = read_split(SPLIT_DIR / "train_reduced.csv", GUIDED_FEATURES)

    records = []
    for model_name, estimator in build_manual_models().items():
        estimator.fit(full_train[FULL_FEATURES], full_train[TARGET])
        path = save_artifact(
            model_name, "ML-only", FULL_FEATURES, estimator
        )
        records.append(_artifact_record(model_name, "ML-only", FULL_FEATURES, path))

    abrams_train = prepare_abrams_data(guided_train)
    abrams_coefficients = fit_abrams_prior(abrams_train)
    abrams_residual = (
        abrams_train[TARGET].to_numpy()
        - predict_abrams_prior(abrams_train, abrams_coefficients)
    )
    for short_name, estimator in build_manual_models().items():
        model_name = f"AbramsGuided_{short_name}"
        estimator.fit(abrams_train[GUIDED_FEATURES], abrams_residual)
        prior = {"kind": "abrams", "coefficients": abrams_coefficients}
        path = save_artifact(
            model_name, "Abrams-guided", GUIDED_FEATURES, estimator, prior
        )
        records.append(
            _artifact_record(model_name, "Abrams-guided", GUIDED_FEATURES, path)
        )

    multivariable_train = prepare_multivariable_data(guided_train)
    multivariable_coefficients = fit_multivariable_prior(multivariable_train)
    multivariable_residual = (
        multivariable_train[TARGET].to_numpy()
        - predict_multivariable_prior(
            multivariable_train, multivariable_coefficients
        )
    )
    for short_name, estimator in build_manual_models().items():
        model_name = f"MultivariableGuided_{short_name}"
        estimator.fit(
            multivariable_train[GUIDED_FEATURES], multivariable_residual
        )
        prior = {
            "kind": "multivariable",
            "coefficients": multivariable_coefficients,
        }
        path = save_artifact(
            model_name,
            "Multivariable-guided",
            GUIDED_FEATURES,
            estimator,
            prior,
        )
        records.append(
            _artifact_record(
                model_name,
                "Multivariable-guided",
                GUIDED_FEATURES,
                path,
            )
        )

    manifest = pd.DataFrame(records)
    manifest.to_csv(MODEL_DIR / "model_manifest.csv", index=False)
    return manifest


def _artifact_record(model_name, family, features, path):
    return {
        "Model": model_name,
        "Family": family,
        "Features": "|".join(features),
        "Artifact": path.relative_to(ROOT).as_posix(),
    }


def load_artifacts(model_names=None):
    manifest_path = MODEL_DIR / "model_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Interpretation model artifacts were not found. "
            "Run 01_shap_analysis.py first."
        )

    manifest = pd.read_csv(manifest_path)
    if model_names is not None:
        manifest = manifest.loc[manifest["Model"].isin(model_names)]

    artifacts = {}
    for row in manifest.itertuples(index=False):
        path = ROOT / Path(row.Artifact)
        if path.exists():
            artifacts[row.Model] = joblib.load(path)
    return artifacts


def prepare_prediction_frame(data, artifact):
    features = artifact["features"]
    frame = data[features].copy()
    for feature in features:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    if frame.isna().any().any():
        raise ValueError("Prediction data contain missing or non-numeric values.")

    prior = artifact.get("prior")
    if prior and prior["kind"] == "multivariable":
        for feature, epsilon in ZERO_ALLOWED_FEATURES.items():
            frame[feature] = frame[feature].clip(lower=0) + epsilon
    return frame


def predict_artifact(artifact, data):
    frame = prepare_prediction_frame(data, artifact)
    prediction = np.asarray(artifact["estimator"].predict(frame), dtype=float)

    prior = artifact.get("prior")
    if prior is None:
        return prediction
    if prior["kind"] == "abrams":
        return prediction + predict_abrams_prior(frame, prior["coefficients"])
    if prior["kind"] == "multivariable":
        return prediction + predict_multivariable_prior(
            frame, prior["coefficients"]
        )
    raise ValueError(f"Unsupported prior type: {prior['kind']}")


def load_model_frame(artifact, split="train"):
    suffix = "full" if artifact["family"] == "ML-only" else "reduced"
    frame = read_split(
        SPLIT_DIR / f"{split}_{suffix}.csv", artifact["features"]
    )
    if artifact["family"] == "Abrams-guided":
        frame = prepare_abrams_data(frame)
    elif artifact["family"] == "Multivariable-guided":
        frame = filter_multivariable_rows(frame)
    return frame


def pdp_ice_values(artifact, data, feature, grid, max_ice_samples=120):
    sample = data[artifact["features"]].sample(
        min(max_ice_samples, len(data)), random_state=RANDOM_STATE
    )
    curves = []
    for value in grid:
        changed = sample.copy()
        changed[feature] = value
        curves.append(predict_artifact(artifact, changed))
    ice = np.column_stack(curves)
    return sample, ice, ice.mean(axis=0)
