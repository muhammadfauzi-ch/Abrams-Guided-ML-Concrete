from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]

input_file = ROOT / "data" / "processed" / "literature_train_ready.csv"

output_dir = ROOT / "outputs" / "splits"
output_dir.mkdir(parents=True, exist_ok=True)


data = pd.read_csv(input_file)

target_column = "fc_MPa"


full_features = [
    "Age_day",
    "Cement_kgm3",
    "Water_kgm3",
    "W_over_B",
    "SCM1_kgm3",
    "SCM2_kgm3",
    "FineAgg_kgm3",
    "CoarseAgg_kgm3",
    "SP_kgm3",
    "Fiber_kgm3",
]


reduced_features = [
    "Age_day",
    "Cement_kgm3",
    "W_over_B",
    "SCM1_kgm3",
    "SCM2_kgm3",
    "FineAgg_kgm3",
    "SP_kgm3",
    "Fiber_kgm3",
]


full_features = [
    feature for feature in full_features
    if feature in data.columns
]

reduced_features = [
    feature for feature in reduced_features
    if feature in data.columns
]


def create_split(features, name):

    dataset = data[
        features + [target_column]
    ].copy()

    X = dataset[features]
    y = dataset[target_column]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )


    train_data = X_train.copy()
    train_data[target_column] = y_train

    test_data = X_test.copy()
    test_data[target_column] = y_test


    train_data.to_csv(
        output_dir / f"train_{name}.csv",
        index=False
    )

    test_data.to_csv(
        output_dir / f"test_{name}.csv",
        index=False
    )


create_split(
    full_features,
    "full"
)

create_split(
    reduced_features,
    "reduced"
)


print("Dataset split completed.")
print(f"Full features: {len(full_features)}")
print(f"Reduced features: {len(reduced_features)}")