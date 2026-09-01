from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


np.random.seed(42)
tf.random.set_seed(42)


def calculate_rmse(y_true, y_pred):

    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


def calculate_mbe(y_true, y_pred):

    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)

    return np.mean(
        y_true - y_pred
    )


def calculate_mape(y_true, y_pred):

    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)

    valid = y_true != 0

    return np.mean(
        np.abs(
            (y_true[valid] - y_pred[valid])
            / y_true[valid]
        )
    ) * 100


def calculate_rrmse(y_true, y_pred):

    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)

    return (
        np.sqrt(
            np.mean(
                (y_true - y_pred) ** 2
            )
        )
        / np.mean(y_true)
    ) * 100


def calculate_a20(y_true, y_pred):

    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)

    return np.mean(
        (y_pred >= 0.8 * y_true) &
        (y_pred <= 1.2 * y_true)
    )


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


ROOT = Path(__file__).resolve().parents[2]

train_file = ROOT / "outputs" / "splits" / "train_full.csv"
test_file = ROOT / "outputs" / "splits" / "test_full.csv"

output_dir = ROOT / "outputs" / "ml_only"

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


metrics_file = output_dir / "ml_only_dnn_metrics.csv"
train_pred_file = output_dir / "dnn_train_predictions.csv"
test_pred_file = output_dir / "dnn_test_predictions.csv"
model_file = output_dir / "dnn_keras_model.keras"


target_column = "fc_MPa"


train_df = pd.read_csv(
    train_file
)

test_df = pd.read_csv(
    test_file
)


feature_columns = [
    column
    for column in train_df.columns
    if column != target_column
]


required_columns = (
    feature_columns +
    [target_column]
)


for column in required_columns:

    train_df[column] = pd.to_numeric(
        train_df[column],
        errors="coerce"
    )

    test_df[column] = pd.to_numeric(
        test_df[column],
        errors="coerce"
    )


train_df = train_df.dropna(
    subset=required_columns
).copy()

test_df = test_df.dropna(
    subset=required_columns
).copy()


X_train = train_df[feature_columns].values
X_test = test_df[feature_columns].values

y_train = train_df[target_column].values.reshape(-1, 1)
y_test = test_df[target_column].values.reshape(-1, 1)


x_scaler = StandardScaler()
y_scaler = StandardScaler()


X_train_scaled = x_scaler.fit_transform(
    X_train
)

X_test_scaled = x_scaler.transform(
    X_test
)


y_train_scaled = y_scaler.fit_transform(
    y_train
)

y_test_scaled = y_scaler.transform(
    y_test
)


model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(
            shape=(
                X_train_scaled.shape[1],
            )
        ),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            0.10
        ),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            0.10
        ),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            1
        ),
    ]
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"],
)


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=30,
        restore_best_weights=True,
    )
]


model.fit(
    X_train_scaled,
    y_train_scaled,
    validation_split=0.2,
    epochs=300,
    batch_size=32,
    verbose=0,
    callbacks=callbacks,
)


train_prediction_scaled = model.predict(
    X_train_scaled,
    verbose=0
)

test_prediction_scaled = model.predict(
    X_test_scaled,
    verbose=0
)


train_prediction = y_scaler.inverse_transform(
    train_prediction_scaled
).reshape(-1)


test_prediction = y_scaler.inverse_transform(
    test_prediction_scaled
).reshape(-1)


y_train_true = y_train.reshape(-1)
y_test_true = y_test.reshape(-1)


metrics = pd.DataFrame(
    [
        {
            "Dataset": "Train",
            "Model": "DNN",
            **regression_metrics(
                y_train_true,
                train_prediction
            ),
        },
        {
            "Dataset": "Test",
            "Model": "DNN",
            **regression_metrics(
                y_test_true,
                test_prediction
            ),
        },
    ]
).round(4)


train_output = train_df[feature_columns].copy()
train_output[target_column] = y_train_true
train_output["fc_pred_dnn"] = train_prediction


test_output = test_df[feature_columns].copy()
test_output[target_column] = y_test_true
test_output["fc_pred_dnn"] = test_prediction


metrics.to_csv(
    metrics_file,
    index=False
)

train_output.to_csv(
    train_pred_file,
    index=False
)

test_output.to_csv(
    test_pred_file,
    index=False
)

model.save(
    model_file
)


print("ML-only DNN completed.")
print(metrics)