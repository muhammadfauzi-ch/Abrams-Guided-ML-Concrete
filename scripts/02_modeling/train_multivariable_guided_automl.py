from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = ROOT / "outputs" / "splits" / "train_full.csv"
TEST_FILE = ROOT / "outputs" / "splits" / "test_full.csv"

OUTPUT_DIR = ROOT / "outputs" / "automl"
MODEL_DIR = OUTPUT_DIR / "multivariable_guided_autogluon_model"

METRICS_FILE = OUTPUT_DIR / "multivariable_guided_automl_metrics.csv"
TRAIN_PREDICTIONS_FILE = (
    OUTPUT_DIR / "multivariable_guided_automl_train_predictions.csv"
)
TEST_PREDICTIONS_FILE = (
    OUTPUT_DIR / "multivariable_guided_automl_test_predictions.csv"
)
LEADERBOARD_FILE = OUTPUT_DIR / "multivariable_guided_automl_leaderboard.csv"
FEATURE_IMPORTANCE_FILE = (
    OUTPUT_DIR / "multivariable_guided_automl_feature_importance.csv"
)
PRIOR_PARAMETERS_FILE = OUTPUT_DIR / "multivariable_guided_prior_params.csv"

TARGET = "fc_MPa"
RESIDUAL_TARGET = "residual_target"


def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mbe(y_true, y_pred):
    return np.mean(np.asarray(y_true) - np.asarray(y_pred))


def calculate_mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid = y_true != 0

    return np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100


def calculate_rrmse(y_true, y_pred):
    return calculate_rmse(y_true, y_pred) / np.mean(y_true) * 100


def calculate_a20(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean((y_pred >= 0.8 * y_true) & (y_pred <= 1.2 * y_true))


def regression_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MBE": calculate_mbe(y_true, y_pred),
        "MAPE": calculate_mape(y_true, y_pred),
        "RRMSE": calculate_rrmse(y_true, y_pred),
        "A20": calculate_a20(y_true, y_pred),
    }


def load_datasets():
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    feature_columns = [column for column in train.columns if column != TARGET]
    required_columns = feature_columns + [TARGET]

    for column in required_columns:
        train[column] = pd.to_numeric(train[column], errors="coerce")
        test[column] = pd.to_numeric(test[column], errors="coerce")

    train = train.dropna(subset=required_columns).copy()
    test = test.dropna(subset=required_columns).copy()

    return train, test, feature_columns


def select_positive_prior_features(train, test, feature_columns):
    positive_features = [
        feature
        for feature in feature_columns
        if (train[feature] > 0).all() and (test[feature] > 0).all()
    ]

    if not positive_features:
        raise ValueError(
            "No strictly positive numeric features are available for the "
            "multivariable prior."
        )

    return positive_features


def fit_multivariable_prior(train, test, feature_columns):
    # Multivariable empirical relationship:
    # ln(fc) = b0 + sum(bi ln(xi))
    eps = 1e-8
    prior_features = select_positive_prior_features(
        train,
        test,
        feature_columns,
    )

    train_log_features = np.log(train[prior_features].clip(lower=eps))
    test_log_features = np.log(test[prior_features].clip(lower=eps))
    train_log_strength = np.log(train[TARGET].clip(lower=eps))

    model = LinearRegression()
    model.fit(train_log_features, train_log_strength)

    train["ln_fc"] = train_log_strength
    test["ln_fc"] = np.log(test[TARGET].clip(lower=eps))

    train["ln_fc_prior"] = model.predict(train_log_features)
    test["ln_fc_prior"] = model.predict(test_log_features)

    train["fc_prior"] = np.exp(train["ln_fc_prior"])
    test["fc_prior"] = np.exp(test["ln_fc_prior"])

    return model, prior_features


def export_prior_parameters(prior_model, prior_features):
    parameters = pd.DataFrame(
        {
            "Feature": ["Intercept"] + prior_features,
            "Coefficient": [prior_model.intercept_] + list(prior_model.coef_),
        }
    )
    parameters.to_csv(PRIOR_PARAMETERS_FILE, index=False)


def prepare_residual_datasets(train, test, feature_columns):
    train_residual = train[feature_columns].copy()
    test_residual = test[feature_columns].copy()

    train_residual["fc_prior"] = train["fc_prior"]
    test_residual["fc_prior"] = test["fc_prior"]

    train_residual[RESIDUAL_TARGET] = train[TARGET] - train["fc_prior"]
    test_residual[RESIDUAL_TARGET] = test[TARGET] - test["fc_prior"]

    return train_residual, test_residual


def train_or_load_autogluon(train_data):
    predictor_file = MODEL_DIR / "predictor.pkl"

    if predictor_file.exists():
        return TabularPredictor.load(str(MODEL_DIR))

    predictor = TabularPredictor(
        label=RESIDUAL_TARGET,
        path=str(MODEL_DIR),
        problem_type="regression",
        eval_metric="root_mean_squared_error",
    )
    predictor.fit(
        train_data=train_data,
        presets="best_quality",
        time_limit=1800,
        verbosity=2,
    )

    return predictor


def build_metrics(train, test, train_prediction, test_prediction):
    return pd.DataFrame(
        [
            {
                "Dataset": "Train",
                "Model": "MultivariableGuided_AutoGluon",
                **regression_metrics(train[TARGET].values, train_prediction),
            },
            {
                "Dataset": "Test",
                "Model": "MultivariableGuided_AutoGluon",
                **regression_metrics(test[TARGET].values, test_prediction),
            },
        ]
    ).round(4)


def build_prediction_output(data, residual_prediction, final_prediction):
    output = data.copy()
    output["residual_target"] = data[TARGET] - data["fc_prior"]
    output["residual_pred"] = residual_prediction
    output["fc_pred_multivariable_guided_automl"] = final_prediction

    return output


def export_model_diagnostics(predictor, test_data):
    leaderboard = predictor.leaderboard(test_data, silent=True)
    leaderboard.to_csv(LEADERBOARD_FILE, index=False)

    feature_importance = predictor.feature_importance(test_data).reset_index()
    feature_importance = feature_importance.rename(columns={"index": "Feature"})
    feature_importance.to_csv(FEATURE_IMPORTANCE_FILE, index=False)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train, test, feature_columns = load_datasets()

    prior_model, prior_features = fit_multivariable_prior(
        train,
        test,
        feature_columns,
    )
    export_prior_parameters(prior_model, prior_features)

    # AutoML estimates the deviation that is not explained by the empirical prior.
    train_residual, test_residual = prepare_residual_datasets(
        train,
        test,
        feature_columns,
    )
    predictor = train_or_load_autogluon(train_residual)

    train_residual_prediction = predictor.predict(
        train_residual.drop(columns=[RESIDUAL_TARGET])
    ).values
    test_residual_prediction = predictor.predict(
        test_residual.drop(columns=[RESIDUAL_TARGET])
    ).values

    train_prediction = np.clip(
        train["fc_prior"].values + train_residual_prediction,
        0,
        None,
    )
    test_prediction = np.clip(
        test["fc_prior"].values + test_residual_prediction,
        0,
        None,
    )

    metrics = build_metrics(
        train,
        test,
        train_prediction,
        test_prediction,
    )
    train_output = build_prediction_output(
        train,
        train_residual_prediction,
        train_prediction,
    )
    test_output = build_prediction_output(
        test,
        test_residual_prediction,
        test_prediction,
    )

    metrics.to_csv(METRICS_FILE, index=False)
    train_output.to_csv(TRAIN_PREDICTIONS_FILE, index=False)
    test_output.to_csv(TEST_PREDICTIONS_FILE, index=False)
    export_model_diagnostics(predictor, test_residual)

    print("Multivariable-guided AutoML completed.")
    print(metrics)


if __name__ == "__main__":
    main()
