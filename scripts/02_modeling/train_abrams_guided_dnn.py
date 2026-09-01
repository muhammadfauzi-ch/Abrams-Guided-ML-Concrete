from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = ROOT / "outputs" / "splits" / "train_reduced.csv"
TEST_FILE = ROOT / "outputs" / "splits" / "test_reduced.csv"

OUTPUT_DIR = ROOT / "outputs" / "piml_dnn"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "fc_MPa"
WB_COL = "W_over_B"
AGE_COL = "Age_day"

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


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def regression_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MBE": np.mean(y_true - y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
    }


def load_dataset(path):
    data = pd.read_csv(path)

    for col in FEATURES + [TARGET]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=FEATURES + [TARGET])
    data = data[(data[WB_COL] > 0) & (data[AGE_COL] > 0)]

    return data


def fit_abrams_prior(data):
    # Abrams-age empirical relationship:
    # ln(fc) = b0 + b1 ln(W/B) + b2 ln(Age)
    y = np.log(data[TARGET].values)

    X = np.column_stack(
        [
            np.ones(len(data)),
            np.log(data[WB_COL].values),
            np.log(data[AGE_COL].values),
        ]
    )

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def predict_abrams_prior(data, beta):
    b0, b1, b2 = beta

    return np.exp(
        b0
        + b1 * np.log(data[WB_COL].values)
        + b2 * np.log(data[AGE_COL].values)
    )


def build_model(n_features):
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


def main():

    train = load_dataset(TRAIN_FILE)
    test = load_dataset(TEST_FILE)

    X_train = train[FEATURES].values
    X_test = test[FEATURES].values

    y_train = train[TARGET].values.reshape(-1, 1)
    y_test = test[TARGET].values.reshape(-1, 1)

    beta = fit_abrams_prior(train)

    physics_train = predict_abrams_prior(train, beta).reshape(-1, 1)
    physics_test = predict_abrams_prior(test, beta).reshape(-1, 1)

    # DNN learns unexplained deviation from the empirical prior.
    residual_train = y_train - physics_train

    x_scaler = StandardScaler()
    residual_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    residual_scaled = residual_scaler.fit_transform(residual_train)

    model = build_model(X_train_scaled.shape[1])

    model.fit(
        X_train_scaled,
        residual_scaled,
        validation_split=0.2,
        epochs=300,
        batch_size=32,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=30,
                restore_best_weights=True,
            )
        ],
    )

    train_residual = residual_scaler.inverse_transform(
        model.predict(X_train_scaled, verbose=0)
    )

    test_residual = residual_scaler.inverse_transform(
        model.predict(X_test_scaled, verbose=0)
    )

    train_prediction = (physics_train + train_residual).ravel()
    test_prediction = (physics_test + test_residual).ravel()

    metrics = pd.DataFrame(
        [
            {
                "Dataset": "Train",
                "Model": "AbramsGuided_DNN",
                **regression_metrics(y_train.ravel(), train_prediction),
            },
            {
                "Dataset": "Test",
                "Model": "AbramsGuided_DNN",
                **regression_metrics(y_test.ravel(), test_prediction),
            },
        ]
    )

    metrics.to_csv(
        OUTPUT_DIR / "abrams_guided_dnn_metrics.csv",
        index=False,
    )

    model.save(
        OUTPUT_DIR / "abrams_guided_dnn_model.keras"
    )

    print("Abrams-guided DNN completed.")
    print(metrics)


if __name__ == "__main__":
    main()
