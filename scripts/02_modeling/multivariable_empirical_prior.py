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
    "multivariable_empirical_prior_train_predictions.csv"
)

test_pred_file = (
    output_dir /
    "multivariable_empirical_prior_test_predictions.csv"
)

metrics_file = (
    output_dir /
    "multivariable_empirical_prior_metrics.csv"
)

coeff_file = (
    output_dir /
    "multivariable_empirical_prior_coefficients.csv"
)


target_column = "fc_MPa"


feature_columns = [
    "Age_day",
    "Cement_kgm3",
    "W_over_B",
    "SCM1_kgm3",
    "SCM2_kgm3",
    "FineAgg_kgm3",
    "SP_kgm3",
    "Fiber_kgm3",
]


zero_allowed_columns = {
    "SCM1_kgm3": 1e-6,
    "SCM2_kgm3": 1e-6,
    "SP_kgm3": 1e-6,
    "Fiber_kgm3": 1e-6,
}


def prepare_numeric_data(data, columns):

    data = data.copy()

    for column in columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    return data.dropna(
        subset=columns
    )


def prepare_powerlaw_data(
    data,
    target_column,
    feature_columns,
    zero_allowed_columns,
):

    data = data.copy()

    required_columns = (
        feature_columns +
        [target_column]
    )

    data = data.dropna(
        subset=required_columns
    )


    data = data[
        data[target_column] > 0
    ]


    for column in feature_columns:

        if column in zero_allowed_columns:

            data[column] = (
                data[column]
                .clip(lower=0)
                + zero_allowed_columns[column]
            )

        else:

            data = data[
                data[column] > 0
            ]

    return data


def fit_powerlaw_model(
    data,
    target_column,
    feature_columns,
):

    log_strength = np.log(
        data[target_column].values
    )


    feature_matrix = [
        np.ones(len(data))
    ]


    for column in feature_columns:

        feature_matrix.append(
            np.log(
                data[column].values
            )
        )


    design_matrix = np.column_stack(
        feature_matrix
    )


    coefficients, *_ = np.linalg.lstsq(
        design_matrix,
        log_strength,
        rcond=None
    )

    return coefficients


def predict_powerlaw(
    data,
    coefficients,
    feature_columns,
):

    feature_matrix = [
        np.ones(len(data))
    ]


    for column in feature_columns:

        feature_matrix.append(
            np.log(
                data[column].values
            )
        )


    design_matrix = np.column_stack(
        feature_matrix
    )


    log_prediction = (
        design_matrix @ coefficients
    )


    return np.exp(
        log_prediction
    )


train_data = pd.read_csv(
    train_file
)

test_data = pd.read_csv(
    test_file
)


required_columns = (
    feature_columns +
    [target_column]
)


train_data = prepare_numeric_data(
    train_data,
    required_columns
)

test_data = prepare_numeric_data(
    test_data,
    required_columns
)


train_data = prepare_powerlaw_data(
    train_data,
    target_column,
    feature_columns,
    zero_allowed_columns
)

test_data = prepare_powerlaw_data(
    test_data,
    target_column,
    feature_columns,
    zero_allowed_columns
)


# Multivariable empirical prior:
#
# ln(fc) = b0 + b1 ln(Age)
#              + b2 ln(Cement)
#              + b3 ln(W/B)
#              + b4 ln(SCM1)
#              + b5 ln(SCM2)
#              + b6 ln(FineAgg)
#              + b7 ln(SP)
#              + b8 ln(Fiber)
#
# Equivalent power-law form:
# fc = A × Age^b1 × Cement^b2 ×
#      (W/B)^b3 × SCM1^b4 × ...


coefficients = fit_powerlaw_model(
    train_data,
    target_column,
    feature_columns
)


b0 = float(
    coefficients[0]
)

A = float(
    np.exp(b0)
)


train_prediction = predict_powerlaw(
    train_data,
    coefficients,
    feature_columns
)


test_prediction = predict_powerlaw(
    test_data,
    coefficients,
    feature_columns
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
            "Model": "Multivariable_Empirical_Prior",
            **train_metrics,
        },
        {
            "Dataset": "Test",
            "Model": "Multivariable_Empirical_Prior",
            **test_metrics,
        },
    ]
).round(4)


coefficient_table = pd.DataFrame(
    {
        "Term": (
            ["Intercept_lnA"] +
            [
                f"ln_{column}"
                for column in feature_columns
            ]
        ),
        "Coefficient": coefficients,
    }
).round(6)


train_output = train_data.copy()

train_output[
    "fc_pred_multivariable_empirical_prior"
] = train_prediction


test_output = test_data.copy()

test_output[
    "fc_pred_multivariable_empirical_prior"
] = test_prediction


metrics.to_csv(
    metrics_file,
    index=False
)

coefficient_table.to_csv(
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


print("Multivariable empirical prior completed.")
print(f"A = {A:.6f}")
print(metrics)