from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)

from xgboost import XGBRegressor
from catboost import CatBoostRegressor


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

train_file = (
    ROOT /
    "outputs" /
    "splits" /
    "train_full.csv"
)

test_file = (
    ROOT /
    "outputs" /
    "splits" /
    "test_full.csv"
)


output_dir = (
    ROOT /
    "outputs" /
    "ml_only"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


metrics_file = (
    output_dir /
    "ml_only_manual_metrics.csv"
)

ranking_file = (
    output_dir /
    "ml_only_manual_test_summary.csv"
)


train_df = pd.read_csv(
    train_file
)

test_df = pd.read_csv(
    test_file
)


target_column = "fc_MPa"

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


X_train = train_df[feature_columns]
X_test = test_df[feature_columns]

y_train = train_df[target_column].values
y_test = test_df[target_column].values


models = {

    "MLR": LinearRegression(),

    "SVR": SVR(
        kernel="rbf",
        C=100,
        epsilon=0.1,
        gamma="scale"
    ),

    "KNN": KNeighborsRegressor(
        n_neighbors=5,
        weights="distance"
    ),

    "RF": RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ),

    "GBR": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        objective="reg:squarederror"
    ),

    "CatBoost": CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=0,
        random_seed=42
    ),
}


metrics_rows = []


for model_name, model in models.items():

    print(f"Training {model_name}")

    model.fit(
        X_train,
        y_train
    )


    train_prediction = model.predict(
        X_train
    )

    test_prediction = model.predict(
        X_test
    )


    train_metrics = regression_metrics(
        y_train,
        train_prediction
    )

    test_metrics = regression_metrics(
        y_test,
        test_prediction
    )


    metrics_rows.extend(
        [
            {
                "Dataset": "Train",
                "Model": model_name,
                **train_metrics,
            },
            {
                "Dataset": "Test",
                "Model": model_name,
                **test_metrics,
            },
        ]
    )


    train_output = X_train.copy()
    train_output[target_column] = y_train
    train_output[
        f"fc_pred_{model_name.lower()}"
    ] = train_prediction


    test_output = X_test.copy()
    test_output[target_column] = y_test
    test_output[
        f"fc_pred_{model_name.lower()}"
    ] = test_prediction


    train_output.to_csv(
        output_dir /
        f"{model_name.lower()}_train_predictions.csv",
        index=False
    )

    test_output.to_csv(
        output_dir /
        f"{model_name.lower()}_test_predictions.csv",
        index=False
    )


metrics_df = pd.DataFrame(
    metrics_rows
).round(4)


metrics_df.to_csv(
    metrics_file,
    index=False
)


test_ranking = (
    metrics_df[
        metrics_df["Dataset"] == "Test"
    ]
    .sort_values(
        by="R2",
        ascending=False
    )
    .reset_index(drop=True)
)


test_ranking.to_csv(
    ranking_file,
    index=False
)


print("ML-only manual benchmarking completed.")
print(metrics_df)

print(test_ranking[
    [
        "Model",
        "R2",
        "RMSE",
        "MAE",
        "MAPE",
        "A20",
    ]
])