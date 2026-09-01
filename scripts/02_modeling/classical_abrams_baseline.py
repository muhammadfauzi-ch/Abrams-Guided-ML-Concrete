from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


warnings.filterwarnings("ignore")


def calculate_rmse(y_true, y_pred):
    return np.sqrt(
        mean_squared_error(y_true, y_pred)
    )


def calculate_mbe(y_true, y_pred):
    return np.mean(
        np.array(y_true) - np.array(y_pred)
    )


def calculate_mape(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    valid = y_true != 0

    return np.mean(
        np.abs(
            (y_true[valid] - y_pred[valid]) /
            y_true[valid]
        )
    ) * 100


def calculate_rrmse(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return (
        np.sqrt(
            np.mean((y_true - y_pred) ** 2)
        )
        / np.mean(y_true)
    ) * 100


def calculate_a20(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

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

train_file = ROOT / "outputs" / "splits" / "train_reduced.csv"
test_file = ROOT / "outputs" / "splits" / "test_reduced.csv"

output_dir = ROOT / "outputs" / "physics_baseline"
output_dir.mkdir(
    parents=True,
    exist_ok=True
)


train_pred_file = (
    output_dir /
    "classical_abrams_train_predictions.csv"
)

test_pred_file = (
    output_dir /
    "classical_abrams_test_predictions.csv"
)

metrics_file = (
    output_dir /
    "classical_abrams_metrics.csv"
)

coeff_file = (
    output_dir /
    "classical_abrams_coefficients.csv"
)


target_column = "fc_MPa"
wb_column = "W_over_B"


train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)


required_columns = [
    target_column,
    wb_column,
]


for column in required_columns:

    train_data[column] = pd.to_numeric(
        train_data[column],
        errors="coerce"
    )

    test_data[column] = pd.to_numeric(
        test_data[column],
        errors="coerce"
    )


train_data = train_data.dropna(
    subset=required_columns
).copy()

test_data = test_data.dropna(
    subset=required_columns
).copy()


train_data = train_data[
    (train_data[target_column] > 0) &
    (train_data[wb_column] > 0)
]

test_data = test_data[
    (test_data[target_column] > 0) &
    (test_data[wb_column] > 0)
]


# Classical Abrams relationship:
#
# fc = A / B^(W/B)
#
# Linearized form:
# ln(fc) = ln(A) - (W/B)ln(B)

x_train = train_data[wb_column].values
y_train = train_data[target_column].values


log_strength = np.log(
    y_train
)


design_matrix = np.column_stack(
    [
        np.ones(len(x_train)),
        x_train,
    ]
)


coefficients, *_ = np.linalg.lstsq(
    design_matrix,
    log_strength,
    rcond=None
)


log_A = float(coefficients[0])
negative_log_B = float(coefficients[1])


A = float(np.exp(log_A))
B = float(np.exp(-negative_log_B))


def predict_classical_abrams(data):

    return A / (
        B ** data[wb_column].values
    )


train_prediction = predict_classical_abrams(
    train_data
)

test_prediction = predict_classical_abrams(
    test_data
)


train_metrics = regression_metrics(
    train_data[target_column].values,
    train_prediction
)

test_metrics = regression_metrics(
    test_data[target_column].values,
    test_prediction
)


metrics = pd.DataFrame(
    [
        {
            "Dataset": "Train",
            "Model": "Classical_Abrams",
            **train_metrics,
        },
        {
            "Dataset": "Test",
            "Model": "Classical_Abrams",
            **test_metrics,
        },
    ]
).round(4)


coefficients_table = pd.DataFrame(
    [
        {
            "Term": "A",
            "Coefficient": A,
        },
        {
            "Term": "B",
            "Coefficient": B,
        },
        {
            "Term": "lnA",
            "Coefficient": log_A,
        },
        {
            "Term": "-lnB",
            "Coefficient": negative_log_B,
        },
    ]
).round(6)


train_output = train_data.copy()
train_output[
    "fc_pred_classical_abrams"
] = train_prediction


test_output = test_data.copy()
test_output[
    "fc_pred_classical_abrams"
] = test_prediction


metrics.to_csv(
    metrics_file,
    index=False
)

coefficients_table.to_csv(
    coeff_file,
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


print("Classical Abrams baseline completed.")
print(f"A = {A:.6f}")
print(f"B = {B:.6f}")
print(metrics)