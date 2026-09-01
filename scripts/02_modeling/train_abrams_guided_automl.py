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
MODEL_DIR = OUTPUT_DIR / "abrams_guided_autogluon_model"

METRICS_FILE = OUTPUT_DIR / "abrams_guided_automl_metrics.csv"
TRAIN_PREDICTIONS_FILE = (
    OUTPUT_DIR / "abrams_guided_automl_train_predictions.csv"
)
TEST_PREDICTIONS_FILE = (
    OUTPUT_DIR / "abrams_guided_automl_test_predictions.csv"
)
LEADERBOARD_FILE = OUTPUT_DIR / "abrams_guided_automl_leaderboard.csv"
FEATURE_IMPORTANCE_FILE = (
    OUTPUT_DIR / "abrams_guided_automl_feature_importance.csv"
)
PRIOR_PARAMETERS_FILE = OUTPUT_DIR / "abrams_guided_prior_params.csv"

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


def load_dataset(path):
    data = pd.read_csv(path)

    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data.dropna().copy()


def abrams_design_matrix(data):
    eps = 1e-8

    return pd.DataFrame(
        {
            "ln_W_over_B": np.log(data["W_over_B"].clip(lower=eps)),
            "ln_Age_day": np.log(data["Age_day"].clip(lower=eps)),
        },
        index=data.index,
    )


def fit_abrams_prior(train):
    # Abrams-age empirical relationship:
    # ln(fc) = b0 + b1 ln(W/B) + b2 ln(Age)
    eps = 1e-8
    design_matrix = abrams_design_matrix(train)
    log_strength = np.log(train[TARGET].clip(lower=eps))

    model = LinearRegression()
    model.fit(design_matrix, log_strength)

    return model


def predict_abrams_prior(data, model):
    design_matrix = abrams_design_matrix(data)
    return np.exp(model.predict(design_matrix))


def prepare_residual_dataset(data):
    feature_columns = [column for column in data.columns if column != TARGET]
    residual_data = data[feature_columns].copy()
    residual_data[RESIDUAL_TARGET] = data[TARGET] - data["fc_prior"]

    return residual_data


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
                "Model": "AbramsGuided_AutoGluon",
                **regression_metrics(train[TARGET], train_prediction),
            },
            {
                "Dataset": "Test",
                "Model": "AbramsGuided_AutoGluon",
                **regression_metrics(test[TARGET], test_prediction),
            },
        ]
    )


def build_prediction_output(data, residual_prediction, final_prediction):
    output = data.copy()
    output["residual_pred"] = residual_prediction
    output["fc_pred_abrams_guided_automl"] = final_prediction

    return output


def export_feature_importance(predictor, test_data):
    feature_importance = predictor.feature_importance(test_data).reset_index()
    feature_importance = feature_importance.rename(columns={"index": "Feature"})
    feature_importance.to_csv(FEATURE_IMPORTANCE_FILE, index=False)


def export_prior_parameters(prior_model):
    prior_parameters = pd.DataFrame(
        {
            "Parameter": [
                "Intercept",
                "Coef_ln_W_over_B",
                "Coef_ln_Age_day",
            ],
            "Value": [
                prior_model.intercept_,
                prior_model.coef_[0],
                prior_model.coef_[1],
            ],
        }
    )
    prior_parameters.to_csv(PRIOR_PARAMETERS_FILE, index=False)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train = load_dataset(TRAIN_FILE)
    test = load_dataset(TEST_FILE)

    prior_model = fit_abrams_prior(train)
    train["fc_prior"] = predict_abrams_prior(train, prior_model)
    test["fc_prior"] = predict_abrams_prior(test, prior_model)

    # AutoML estimates the deviation that is not explained by the Abrams prior.
    train_residual = prepare_residual_dataset(train)
    test_residual = prepare_residual_dataset(test)

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

    leaderboard = predictor.leaderboard(test_residual, silent=True)
    leaderboard.to_csv(LEADERBOARD_FILE, index=False)

    export_feature_importance(predictor, test_residual)
    export_prior_parameters(prior_model)

    print("Abrams-guided AutoML completed.")
    print(metrics)


if __name__ == "__main__":
    main()
