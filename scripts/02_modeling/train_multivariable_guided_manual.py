from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = ROOT / "outputs" / "splits" / "train_reduced.csv"
TEST_FILE = ROOT / "outputs" / "splits" / "test_reduced.csv"

OUTPUT_DIR = ROOT / "outputs" / "piml_manual"
PHYSICS_DIR = ROOT / "outputs" / "physics_baseline"

PREDICTIONS_FILE = OUTPUT_DIR / "multivariable_guided_manual_predictions.csv"
METRICS_FILE = OUTPUT_DIR / "multivariable_guided_manual_metrics.csv"
RESIDUAL_DIAGNOSTICS_FILE = (
    OUTPUT_DIR / "multivariable_guided_manual_residual_diagnostics.csv"
)
PRIOR_COEFFICIENTS_FILE = (
    PHYSICS_DIR / "multivariable_guided_prior_coefficients.csv"
)

TARGET = "fc_MPa"

FEATURES = [
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


def prepare_numeric_data(data, columns):
    data = data.copy()

    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data.dropna(subset=columns)


def prepare_powerlaw_data(data):
    data = data.dropna(subset=[TARGET] + FEATURES).copy()
    data = data[data[TARGET] > 0]

    for feature in FEATURES:
        if feature in ZERO_ALLOWED_FEATURES:
            data[feature] = (
                data[feature].clip(lower=0) + ZERO_ALLOWED_FEATURES[feature]
            )
        else:
            data = data[data[feature] > 0]

    return data


def powerlaw_design_matrix(data):
    log_features = [np.log(data[feature].values) for feature in FEATURES]
    return np.column_stack([np.ones(len(data)), *log_features])


def fit_multivariable_prior(train):
    # Multivariable empirical relationship:
    # ln(fc) = b0 + sum(bi ln(xi))
    design_matrix = powerlaw_design_matrix(train)
    log_strength = np.log(train[TARGET].values)

    coefficients, *_ = np.linalg.lstsq(
        design_matrix,
        log_strength,
        rcond=None,
    )
    return coefficients


def predict_multivariable_prior(data, coefficients):
    log_prediction = powerlaw_design_matrix(data) @ coefficients
    return np.exp(log_prediction)


def export_prior_coefficients(coefficients):
    coefficient_table = pd.DataFrame(
        {
            "Term": ["Intercept_lnA"] + [f"ln_{feature}" for feature in FEATURES],
            "Coefficient": coefficients,
        }
    ).round(6)
    coefficient_table.to_csv(PRIOR_COEFFICIENTS_FILE, index=False)

    scale = float(np.exp(coefficients[0]))
    log_terms = " + ".join(
        f"({coefficients[index + 1]:.6f})*ln({feature})"
        for index, feature in enumerate(FEATURES)
    )
    power_terms = " * ".join(
        f"({feature})^({coefficients[index + 1]:.6f})"
        for index, feature in enumerate(FEATURES)
    )

    print("\nMultivariable empirical prior:")
    print(f"ln(fc) = {coefficients[0]:.6f} + {log_terms}")
    print(f"fc = {scale:.6f} * {power_terms}")


def build_residual_models():
    return {
        "MultivariableGuided_MLR": LinearRegression(),
        "MultivariableGuided_SVR": SVR(
            kernel="rbf",
            C=100,
            epsilon=0.1,
            gamma="scale",
        ),
        "MultivariableGuided_KNN": KNeighborsRegressor(
            n_neighbors=5,
            weights="distance",
        ),
        "MultivariableGuided_RF": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
        "MultivariableGuided_GBR": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
        "MultivariableGuided_XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            objective="reg:squarederror",
        ),
        "MultivariableGuided_CatBoost": CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            verbose=0,
            random_seed=42,
        ),
    }


def prediction_block(
    dataset,
    model_name,
    y_true,
    prior_prediction,
    residual_true,
    residual_prediction,
    final_prediction,
):
    return pd.DataFrame(
        {
            "Dataset": dataset,
            "Model": model_name,
            "y_true": y_true,
            "fc_pred_multivariable_prior": prior_prediction,
            "residual_true": residual_true,
            "residual_pred": residual_prediction,
            "y_pred_final": final_prediction,
            "final_error": y_true - final_prediction,
        }
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PHYSICS_DIR.mkdir(parents=True, exist_ok=True)

    required_columns = FEATURES + [TARGET]
    train = prepare_numeric_data(pd.read_csv(TRAIN_FILE), required_columns)
    test = prepare_numeric_data(pd.read_csv(TEST_FILE), required_columns)

    train = prepare_powerlaw_data(train)
    test = prepare_powerlaw_data(test)

    X_train = train[FEATURES].copy()
    X_test = test[FEATURES].copy()
    y_train = train[TARGET].values
    y_test = test[TARGET].values

    prior_coefficients = fit_multivariable_prior(train)
    export_prior_coefficients(prior_coefficients)

    prior_train_prediction = predict_multivariable_prior(
        train,
        prior_coefficients,
    )
    prior_test_prediction = predict_multivariable_prior(
        test,
        prior_coefficients,
    )

    train_residual = y_train - prior_train_prediction
    test_residual = y_test - prior_test_prediction

    metrics_rows = [
        {
            "Dataset": "Train",
            "Model": "Multivariable_Empirical_Prior",
            **regression_metrics(y_train, prior_train_prediction),
        },
        {
            "Dataset": "Test",
            "Model": "Multivariable_Empirical_Prior",
            **regression_metrics(y_test, prior_test_prediction),
        },
    ]
    prediction_blocks = []
    residual_diagnostics = []

    for model_name, model in build_residual_models().items():
        print(f"Training {model_name}")
        model.fit(X_train, train_residual)

        train_residual_prediction = model.predict(X_train)
        test_residual_prediction = model.predict(X_test)

        train_prediction = prior_train_prediction + train_residual_prediction
        test_prediction = prior_test_prediction + test_residual_prediction

        metrics_rows.extend(
            [
                {
                    "Dataset": "Train",
                    "Model": model_name,
                    **regression_metrics(y_train, train_prediction),
                },
                {
                    "Dataset": "Test",
                    "Model": model_name,
                    **regression_metrics(y_test, test_prediction),
                },
            ]
        )

        prediction_blocks.extend(
            [
                prediction_block(
                    "Train",
                    model_name,
                    y_train,
                    prior_train_prediction,
                    train_residual,
                    train_residual_prediction,
                    train_prediction,
                ),
                prediction_block(
                    "Test",
                    model_name,
                    y_test,
                    prior_test_prediction,
                    test_residual,
                    test_residual_prediction,
                    test_prediction,
                ),
            ]
        )

        residual_diagnostics.append(
            {
                "Model": model_name,
                "Train_residual_true_mean": np.mean(train_residual),
                "Train_residual_pred_mean": np.mean(train_residual_prediction),
                "Test_residual_true_mean": np.mean(test_residual),
                "Test_residual_pred_mean": np.mean(test_residual_prediction),
                "Train_final_error_mean": np.mean(y_train - train_prediction),
                "Test_final_error_mean": np.mean(y_test - test_prediction),
            }
        )

    metrics = pd.DataFrame(metrics_rows).round(4)
    predictions = pd.concat(prediction_blocks, ignore_index=True)
    residual_diagnostics = pd.DataFrame(residual_diagnostics).round(6)

    metrics.to_csv(METRICS_FILE, index=False)
    predictions.to_csv(PREDICTIONS_FILE, index=False)
    residual_diagnostics.to_csv(RESIDUAL_DIAGNOSTICS_FILE, index=False)

    print("Multivariable-guided manual models completed.")
    print(metrics)


if __name__ == "__main__":
    main()
