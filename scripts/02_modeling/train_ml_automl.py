from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from autogluon.tabular import TabularPredictor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)


warnings.filterwarnings("ignore")


def calculate_rmse(y_true, y_pred):

    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


def calculate_mbe(y_true, y_pred):

    return np.mean(
        np.array(y_true) -
        np.array(y_pred)
    )


def calculate_mape(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    valid = y_true != 0

    return np.mean(
        np.abs(
            (y_true[valid] - y_pred[valid])
            / y_true[valid]
        )
    ) * 100


def calculate_rrmse(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return (
        np.sqrt(
            np.mean(
                (y_true - y_pred) ** 2
            )
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

train_file = ROOT / "outputs" / "splits" / "train_full.csv"
test_file = ROOT / "outputs" / "splits" / "test_full.csv"


output_dir = ROOT / "outputs" / "automl"
model_dir = output_dir / "autogluon_model"

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

model_dir.mkdir(
    parents=True,
    exist_ok=True
)


metrics_file = output_dir / "ml_only_automl_metrics.csv"
train_pred_file = output_dir / "autogluon_train_predictions.csv"
test_pred_file = output_dir / "autogluon_test_predictions.csv"
leaderboard_file = output_dir / "autogluon_leaderboard.csv"
feature_importance_file = output_dir / "autogluon_feature_importance.csv"


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


predictor = TabularPredictor(
    label=target_column,
    path=str(model_dir),
    problem_type="regression",
    eval_metric="root_mean_squared_error",
).fit(
    train_data=train_df,
    presets="best_quality",
    time_limit=1800,
    verbosity=2,
)


train_prediction = predictor.predict(
    train_df
).values


test_prediction = predictor.predict(
    test_df
).values


y_train = train_df[target_column].values
y_test = test_df[target_column].values


metrics = pd.DataFrame(
    [
        {
            "Dataset": "Train",
            "Model": "AutoGluon",
            **regression_metrics(
                y_train,
                train_prediction
            ),
        },
        {
            "Dataset": "Test",
            "Model": "AutoGluon",
            **regression_metrics(
                y_test,
                test_prediction
            ),
        },
    ]
).round(4)


train_output = train_df.copy()
train_output[
    "fc_pred_autogluon"
] = train_prediction


test_output = test_df.copy()
test_output[
    "fc_pred_autogluon"
] = test_prediction


leaderboard = predictor.leaderboard(
    test_df,
    silent=True
)


feature_importance = (
    predictor
    .feature_importance(test_df)
    .reset_index()
    .rename(
        columns={
            "index": "Feature"
        }
    )
)


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

leaderboard.to_csv(
    leaderboard_file,
    index=False
)

feature_importance.to_csv(
    feature_importance_file,
    index=False
)


print("ML-only AutoGluon completed.")
print(metrics)