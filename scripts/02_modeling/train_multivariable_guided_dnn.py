from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = ROOT / "outputs" / "splits" / "train_reduced.csv"
TEST_FILE = ROOT / "outputs" / "splits" / "test_reduced.csv"

OUTPUT_DIR = ROOT / "outputs" / "piml_dnn"
PHYSICS_DIR = ROOT / "outputs" / "physics_baseline"

METRICS_FILE = OUTPUT_DIR / "multivariable_guided_dnn_metrics.csv"
TRAIN_PREDICTIONS_FILE = (
    OUTPUT_DIR / "multivariable_guided_dnn_train_predictions.csv"
)
TEST_PREDICTIONS_FILE = (
    OUTPUT_DIR / "multivariable_guided_dnn_test_predictions.csv"
)
MODEL_FILE = OUTPUT_DIR / "multivariable_guided_dnn_model.keras"
PRIOR_COEFFICIENTS_FILE = (
    PHYSICS_DIR / "multivariable_guided_dnn_prior_coefficients.csv"
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
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return np.mean(y_true - y_pred)


def calculate_mape(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    valid = y_true != 0

    return np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100


def calculate_rrmse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    return np.sqrt(np.mean((y_true - y_pred) ** 2)) / np.mean(y_true) * 100


def calculate_a20(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

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


def build_residual_model(n_features):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )

    return model


def build_prediction_output(
    data,
    y_true,
    prior_prediction,
    residual_true,
    residual_prediction,
    final_prediction,
):
    output = data[FEATURES].copy()
    output[TARGET] = y_true
    output["fc_pred_multivariable_prior"] = prior_prediction
    output["residual_true"] = residual_true
    output["residual_pred"] = residual_prediction
    output["y_pred_final"] = final_prediction

    return output


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PHYSICS_DIR.mkdir(parents=True, exist_ok=True)

    required_columns = FEATURES + [TARGET]
    train = prepare_numeric_data(pd.read_csv(TRAIN_FILE), required_columns)
    test = prepare_numeric_data(pd.read_csv(TEST_FILE), required_columns)

    train = prepare_powerlaw_data(train)
    test = prepare_powerlaw_data(test)

    X_train = train[FEATURES].values
    X_test = test[FEATURES].values
    y_train = train[TARGET].values.reshape(-1, 1)
    y_test = test[TARGET].values.reshape(-1, 1)

    prior_coefficients = fit_multivariable_prior(train)
    export_prior_coefficients(prior_coefficients)

    prior_train_prediction = predict_multivariable_prior(
        train,
        prior_coefficients,
    ).reshape(-1, 1)
    prior_test_prediction = predict_multivariable_prior(
        test,
        prior_coefficients,
    ).reshape(-1, 1)

    # The DNN learns the residual that remains after the empirical prior.
    train_residual = y_train - prior_train_prediction
    test_residual = y_test - prior_test_prediction

    feature_scaler = StandardScaler()
    residual_scaler = StandardScaler()

    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    train_residual_scaled = residual_scaler.fit_transform(train_residual)
    test_residual_scaled = residual_scaler.transform(test_residual)

    model = build_residual_model(X_train_scaled.shape[1])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=30,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X_train_scaled,
        train_residual_scaled,
        validation_split=0.2,
        epochs=300,
        batch_size=32,
        verbose=0,
        callbacks=callbacks,
    )

    train_residual_scaled_prediction = model.predict(
        X_train_scaled,
        verbose=0,
    )
    test_residual_scaled_prediction = model.predict(
        X_test_scaled,
        verbose=0,
    )

    train_residual_prediction = residual_scaler.inverse_transform(
        train_residual_scaled_prediction
    ).reshape(-1, 1)
    test_residual_prediction = residual_scaler.inverse_transform(
        test_residual_scaled_prediction
    ).reshape(-1, 1)

    train_prediction = (prior_train_prediction + train_residual_prediction).reshape(-1)
    test_prediction = (prior_test_prediction + test_residual_prediction).reshape(-1)

    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    metrics = pd.DataFrame(
        [
            {
                "Dataset": "Train",
                "Model": "Multivariable_Empirical_Prior",
                **regression_metrics(y_train, prior_train_prediction.reshape(-1)),
            },
            {
                "Dataset": "Test",
                "Model": "Multivariable_Empirical_Prior",
                **regression_metrics(y_test, prior_test_prediction.reshape(-1)),
            },
            {
                "Dataset": "Train",
                "Model": "MultivariableGuided_DNN",
                **regression_metrics(y_train, train_prediction),
            },
            {
                "Dataset": "Test",
                "Model": "MultivariableGuided_DNN",
                **regression_metrics(y_test, test_prediction),
            },
        ]
    ).round(4)

    train_output = build_prediction_output(
        train,
        y_train,
        prior_train_prediction.reshape(-1),
        train_residual.reshape(-1),
        train_residual_prediction.reshape(-1),
        train_prediction,
    )
    test_output = build_prediction_output(
        test,
        y_test,
        prior_test_prediction.reshape(-1),
        test_residual.reshape(-1),
        test_residual_prediction.reshape(-1),
        test_prediction,
    )

    metrics.to_csv(METRICS_FILE, index=False)
    train_output.to_csv(TRAIN_PREDICTIONS_FILE, index=False)
    test_output.to_csv(TEST_PREDICTIONS_FILE, index=False)
    model.save(MODEL_FILE)

    print("Multivariable-guided DNN completed.")
    print(metrics)


if __name__ == "__main__":
    main()
