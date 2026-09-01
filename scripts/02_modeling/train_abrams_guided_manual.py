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

train_file = ROOT / "outputs" / "splits" / "train_reduced.csv"
test_file = ROOT / "outputs" / "splits" / "test_reduced.csv"


output_dir = ROOT / "outputs" / "piml_manual"
physics_dir = ROOT / "outputs" / "physics_baseline"

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

physics_dir.mkdir(
    parents=True,
    exist_ok=True
)


prediction_file = (
    output_dir /
    "abrams_guided_manual_predictions.csv"
)

metrics_file = (
    output_dir /
    "abrams_guided_manual_metrics.csv"
)

residual_file = (
    output_dir /
    "abrams_guided_manual_residual_diagnostics.csv"
)

physics_coeff_file = (
    physics_dir /
    "abrams_guided_prior_wb_age_coefficients.csv"
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

wb_column = "W_over_B"
age_column = "Age_day"


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


def prepare_physics_data(data):

    data = data.copy()

    required = [
        target_column,
        wb_column,
        age_column,
    ]

    data = data.dropna(
        subset=required
    )

    data = data[
        (data[target_column] > 0) &
        (data[wb_column] > 0) &
        (data[age_column] > 0)
    ]

    return data


def fit_physics_prior(data):

    # Physics prior:
    #
    # ln(fc) = b0 + b1 ln(W/B) + b2 ln(Age)
    #
    # fc = A × (W/B)^b1 × Age^b2

    log_strength = np.log(
        data[target_column].values
    )


    design_matrix = np.column_stack(
        [
            np.ones(len(data)),
            np.log(data[wb_column].values),
            np.log(data[age_column].values),
        ]
    )


    coefficients, *_ = np.linalg.lstsq(
        design_matrix,
        log_strength,
        rcond=None
    )

    return coefficients


def predict_physics_prior(
    data,
    coefficients,
):

    b0, b1, b2 = coefficients

    log_prediction = (
        b0
        + b1 * np.log(data[wb_column].values)
        + b2 * np.log(data[age_column].values)
    )

    return np.exp(log_prediction)


train_df = pd.read_csv(
    train_file
)

test_df = pd.read_csv(
    test_file
)


required_columns = (
    feature_columns +
    [target_column]
)


train_df = prepare_numeric_data(
    train_df,
    required_columns
)

test_df = prepare_numeric_data(
    test_df,
    required_columns
)


train_df = prepare_physics_data(
    train_df
)

test_df = prepare_physics_data(
    test_df
)


X_train = train_df[feature_columns]
X_test = test_df[feature_columns]

y_train = train_df[target_column].values
y_test = test_df[target_column].values


physics_coefficients = fit_physics_prior(
    train_df
)


b0, b1, b2 = physics_coefficients

A = np.exp(b0)


coefficients = pd.DataFrame(
    [
        {
            "Term": "Intercept_lnA",
            "Coefficient": b0,
        },
        {
            "Term": "ln_W_over_B",
            "Coefficient": b1,
        },
        {
            "Term": "ln_Age_day",
            "Coefficient": b2,
        },
        {
            "Term": "A_exp_b0",
            "Coefficient": A,
        },
    ]
)


coefficients.to_csv(
    physics_coeff_file,
    index=False
)


physics_train_prediction = predict_physics_prior(
    train_df,
    physics_coefficients
)

physics_test_prediction = predict_physics_prior(
    test_df,
    physics_coefficients
)


train_residual = (
    y_train -
    physics_train_prediction
)

test_residual = (
    y_test -
    physics_test_prediction
)


models = {

    "AbramsGuided_MLR": LinearRegression(),

    "AbramsGuided_SVR": SVR(
        kernel="rbf",
        C=100,
        epsilon=0.1,
        gamma="scale"
    ),

    "AbramsGuided_KNN": KNeighborsRegressor(
        n_neighbors=5,
        weights="distance"
    ),

    "AbramsGuided_RF": RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ),

    "AbramsGuided_GBR": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "AbramsGuided_XGBoost": XGBRegressor(
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

    "AbramsGuided_CatBoost": CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=0,
        random_seed=42
    ),
}


metrics_rows = []
prediction_blocks = []
residual_blocks = []


metrics_rows.append(
    {
        "Dataset": "Train",
        "Model": "PhysicsPrior_WB_Age",
        **regression_metrics(
            y_train,
            physics_train_prediction
        ),
    }
)

metrics_rows.append(
    {
        "Dataset": "Test",
        "Model": "PhysicsPrior_WB_Age",
        **regression_metrics(
            y_test,
            physics_test_prediction
        ),
    }
)


for model_name, model in models.items():

    print(f"Training {model_name}")

    model.fit(
        X_train,
        train_residual
    )


    residual_train_prediction = model.predict(
        X_train
    )

    residual_test_prediction = model.predict(
        X_test
    )


    final_train_prediction = (
        physics_train_prediction +
        residual_train_prediction
    )

    final_test_prediction = (
        physics_test_prediction +
        residual_test_prediction
    )


    metrics_rows.append(
        {
            "Dataset": "Train",
            "Model": model_name,
            **regression_metrics(
                y_train,
                final_train_prediction
            ),
        }
    )

    metrics_rows.append(
        {
            "Dataset": "Test",
            "Model": model_name,
            **regression_metrics(
                y_test,
                final_test_prediction
            ),
        }
    )


    prediction_blocks.extend(
        [
            pd.DataFrame(
                {
                    "Dataset": "Train",
                    "Model": model_name,
                    "y_true": y_train,
                    "fc_pred_physics_prior": physics_train_prediction,
                    "residual_true": train_residual,
                    "residual_pred": residual_train_prediction,
                    "y_pred_final": final_train_prediction,
                    "error": y_train - final_train_prediction,
                }
            ),

            pd.DataFrame(
                {
                    "Dataset": "Test",
                    "Model": model_name,
                    "y_true": y_test,
                    "fc_pred_physics_prior": physics_test_prediction,
                    "residual_true": test_residual,
                    "residual_pred": residual_test_prediction,
                    "y_pred_final": final_test_prediction,
                    "error": y_test - final_test_prediction,
                }
            ),
        ]
    )


    residual_blocks.append(
        {
            "Model": model_name,
            "Train_residual_mean": np.mean(train_residual),
            "Train_prediction_residual_mean": np.mean(residual_train_prediction),
            "Test_residual_mean": np.mean(test_residual),
            "Test_prediction_residual_mean": np.mean(residual_test_prediction),
            "Train_error_mean": np.mean(
                y_train - final_train_prediction
            ),
            "Test_error_mean": np.mean(
                y_test - final_test_prediction
            ),
        }
    )


metrics_df = pd.DataFrame(
    metrics_rows
).round(4)

prediction_df = pd.concat(
    prediction_blocks,
    ignore_index=True
)

residual_df = pd.DataFrame(
    residual_blocks
).round(6)


metrics_df.to_csv(
    metrics_file,
    index=False
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

residual_df.to_csv(
    residual_file,
    index=False
)


print("Abrams-guided manual model completed.")
print(metrics_df)