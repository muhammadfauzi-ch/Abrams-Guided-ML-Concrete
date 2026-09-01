from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
APPLICATION_DIR = ROOT / "outputs" / "application"

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
OPTIONAL_ZERO_FEATURES = [
    "SCM1_kgm3",
    "SCM2_kgm3",
    "SP_kgm3",
    "Fiber_kgm3",
]
CORE_FEATURES = [
    "Age_day",
    "Cement_kgm3",
    "Water_kgm3",
    "W_over_B",
    "FineAgg_kgm3",
    "CoarseAgg_kgm3",
]

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

SCENARIOS = {
    "A": {
        "label": "Literature-trained models",
        "candidate_file": PROCESSED_DIR / "literature_train_ready.csv",
        "manifest": ROOT
        / "outputs"
        / "interpretation"
        / "models"
        / "model_manifest.csv",
    },
    "C": {
        "label": "Literature + Exp1/Exp2 retraining",
        "candidate_file": ROOT
        / "outputs"
        / "retraining"
        / "exp12"
        / "merged_literature_exp12.csv",
        "manifest": ROOT
        / "outputs"
        / "retraining"
        / "exp12"
        / "exp12_model_artifacts.csv",
    },
}


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")


def repo_relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def _fallback_scenario_c_candidates():
    paths = [
        PROCESSED_DIR / "literature_train_ready.csv",
        PROCESSED_DIR / "finaldataexp12.csv",
    ]
    missing = [repo_relative(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing candidate datasets: {missing}")

    blocks = []
    for path, source in zip(paths, ["literature", "exp12"]):
        block = pd.read_csv(path)
        block["CandidateSource"] = source
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True), " + ".join(
        repo_relative(path) for path in paths
    )


def load_candidates(scenario):
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    candidate_path = SCENARIOS[scenario]["candidate_file"]
    if candidate_path.exists():
        data = pd.read_csv(candidate_path)
        source_description = repo_relative(candidate_path)
    elif scenario == "C":
        data, source_description = _fallback_scenario_c_candidates()
    else:
        raise FileNotFoundError(
            f"Candidate dataset not found: {repo_relative(candidate_path)}"
        )

    missing = [feature for feature in FULL_FEATURES if feature not in data.columns]
    if missing:
        raise ValueError(f"Candidate dataset is missing features: {missing}")

    data = data.copy()
    for feature in FULL_FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
    for feature in OPTIONAL_ZERO_FEATURES:
        data[feature] = data[feature].fillna(0.0).clip(lower=0.0)

    data = data.dropna(subset=CORE_FEATURES)
    positive = [
        "Age_day",
        "Cement_kgm3",
        "Water_kgm3",
        "W_over_B",
        "FineAgg_kgm3",
    ]
    data = data.loc[(data[positive] > 0).all(axis=1)]
    data = data.loc[data["CoarseAgg_kgm3"] >= 0].copy()

    if "CandidateSource" not in data:
        if "SourceTag" in data:
            data["CandidateSource"] = data["SourceTag"].astype(str)
        else:
            data["CandidateSource"] = scenario

    data = data.drop_duplicates(subset=FULL_FEATURES).reset_index(drop=True)
    data.insert(0, "Candidate_ID", [f"{scenario}-{i + 1:05d}" for i in range(len(data))])
    if data.empty:
        raise ValueError(f"No physically valid candidates remain for scenario {scenario}.")
    return data, source_description


def load_model_manifest(scenario):
    manifest_path = SCENARIOS[scenario]["manifest"]
    if not manifest_path.exists():
        prerequisite = (
            "01_shap_analysis.py"
            if scenario == "A"
            else "05_retraining/01_run_retraining.py"
        )
        raise FileNotFoundError(
            f"Model manifest not found: {repo_relative(manifest_path)}. "
            f"Run {prerequisite} first."
        )

    manifest = pd.read_csv(manifest_path)
    required = {"Family", "Model", "Artifact"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(
            f"{manifest_path.name} is missing columns: {sorted(missing)}"
        )
    return manifest, manifest_path


def artifact_features(row, artifact):
    features = artifact.get("features")
    if features is not None:
        return list(features)

    raw = getattr(row, "Features", None)
    if pd.isna(raw):
        raise ValueError("Artifact feature metadata are missing.")
    if str(raw).lstrip().startswith("["):
        return list(json.loads(raw))
    return [value for value in str(raw).split("|") if value]


def load_scenario_artifacts(scenario):
    manifest, manifest_path = load_model_manifest(scenario)
    loaded = []
    failures = []
    for row in manifest.itertuples(index=False):
        artifact_path = ROOT / Path(str(row.Artifact))
        try:
            if not artifact_path.exists():
                raise FileNotFoundError(repo_relative(artifact_path))
            artifact = joblib.load(artifact_path)
            loaded.append(
                {
                    "scenario": scenario,
                    "family": str(row.Family),
                    "model": str(row.Model),
                    "artifact": artifact,
                    "features": artifact_features(row, artifact),
                    "artifact_path": artifact_path,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "Scenario": scenario,
                    "Family": str(row.Family),
                    "Model": str(row.Model),
                    "Artifact": str(row.Artifact),
                    "Error": str(exc),
                }
            )
    return loaded, failures, manifest_path


def _multivariable_frame(data, features):
    transformed = data[features].copy().astype(float)
    for feature in OPTIONAL_ZERO_FEATURES:
        if feature in transformed:
            transformed[feature] = transformed[feature].clip(lower=0.0) + 1e-6
    return transformed


def _predict_retraining_prior(prior, data):
    prior_type = prior["prior_type"]
    coefficients = np.asarray(prior["coefficients"], dtype=float)
    if prior_type == "classical_abrams":
        design = np.column_stack([np.ones(len(data)), data["W_over_B"]])
    elif prior_type == "abrams_age":
        design = np.column_stack(
            [
                np.ones(len(data)),
                np.log(data["W_over_B"]),
                np.log(data["Age_day"]),
            ]
        )
    elif prior_type == "multivariable":
        transformed = _multivariable_frame(data, GUIDED_FEATURES)
        terms = [np.log(transformed[feature]) for feature in GUIDED_FEATURES]
        design = np.column_stack([np.ones(len(data)), *terms])
    else:
        raise ValueError(f"Unsupported prior type: {prior_type}")
    return np.exp(design @ coefficients)


def _predict_retraining_artifact(artifact, data):
    artifact_type = artifact["artifact_type"]
    if artifact_type == "prior":
        return _predict_retraining_prior(artifact, data)
    if artifact_type == "ml":
        return artifact["model"].predict(data[artifact["features"]])
    if artifact_type == "guided":
        prior = _predict_retraining_prior(artifact["prior"], data)
        residual = artifact["residual_model"].predict(
            data[artifact["features"]]
        )
        return prior + residual
    raise ValueError(f"Unsupported artifact type: {artifact_type}")


def _predict_interpretation_artifact(artifact, data):
    features = artifact["features"]
    model_data = data[features].copy()
    prior = artifact.get("prior")
    if prior and prior["kind"] == "multivariable":
        model_data = _multivariable_frame(model_data, features)

    prediction = np.asarray(artifact["estimator"].predict(model_data), dtype=float)
    if prior is None:
        return prediction
    coefficients = np.asarray(prior["coefficients"], dtype=float)
    if prior["kind"] == "abrams":
        b0, b1, b2 = coefficients
        prior_prediction = np.exp(
            b0
            + b1 * np.log(model_data["W_over_B"])
            + b2 * np.log(model_data["Age_day"])
        )
    elif prior["kind"] == "multivariable":
        terms = [np.log(model_data[feature]) for feature in features]
        design = np.column_stack([np.ones(len(model_data)), *terms])
        prior_prediction = np.exp(design @ coefficients)
    else:
        raise ValueError(f"Unsupported prior kind: {prior['kind']}")
    return prediction + np.asarray(prior_prediction)


def predict_model(model_record, data):
    missing = set(model_record["features"]).difference(data.columns)
    if missing:
        raise ValueError(f"Prediction data are missing features: {sorted(missing)}")

    artifact = model_record["artifact"]
    if "artifact_type" in artifact:
        prediction = _predict_retraining_artifact(artifact, data)
    elif "estimator" in artifact:
        prediction = _predict_interpretation_artifact(artifact, data)
    else:
        raise ValueError("Unsupported application artifact format.")

    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    if len(prediction) != len(data):
        raise ValueError("Prediction length does not match candidate rows.")
    if not np.isfinite(prediction).all():
        raise ValueError("Model produced non-finite predictions.")
    return prediction


def model_file_stem(record):
    return safe_name(f"{record['family']}__{record['model']}")


def physically_consistent_grid(baseline, feature, values):
    rows = []
    for value in values:
        row = baseline.copy()
        row[feature] = value
        binder = row["Cement_kgm3"] + row["SCM1_kgm3"] + row["SCM2_kgm3"]
        if feature == "W_over_B":
            row["Water_kgm3"] = value * binder
        elif feature == "Cement_kgm3":
            row["W_over_B"] = row["Water_kgm3"] / binder
        rows.append(row)
    return pd.DataFrame(rows, columns=FULL_FEATURES)
