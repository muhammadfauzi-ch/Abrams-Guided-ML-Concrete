from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

input_file = ROOT / "data" / "processed" / "literature_master_all.csv"

output_dir = ROOT / "data" / "processed"
qc_dir = ROOT / "outputs" / "qc"

output_dir.mkdir(parents=True, exist_ok=True)
qc_dir.mkdir(parents=True, exist_ok=True)


output_file = output_dir / "literature_train_ready.csv"
qc_report_file = qc_dir / "literature_qc_report.csv"


df = pd.read_csv(input_file)

print(f"Original dataset shape: {df.shape}")


# Standardize column names
df.columns = [column.strip() for column in df.columns]


numeric_columns = [
    "Age_day",
    "fc_MPa",
    "Cement_kgm3",
    "Water_kgm3",
    "W_over_B",
    "SCM1_kgm3",
    "SCM2_kgm3",
    "FineAgg_kgm3",
    "CoarseAgg_kgm3",
    "SP_kgm3",
    "Slump_mm",
    "Fiber_kgm3",
    "Fiber_vol_pct",
]


categorical_columns = [
    "Water_type",
    "SCM1_name",
    "SCM2_name",
    "Fiber_shape",
]


text_columns = [
    "Notes",
]


zero_fill_columns = [
    "SCM1_kgm3",
    "SCM2_kgm3",
    "SP_kgm3",
    "Fiber_kgm3",
    "Fiber_vol_pct",
]


required_columns = [
    "Study_ID",
    "Mix_ID",
    "Age_day",
    "fc_MPa",
    "Cement_kgm3",
    "Water_kgm3",
    "W_over_B",
]


# Convert numerical variables
for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# Handle missing values based on variable characteristics
for column in categorical_columns:
    if column in df.columns:
        df[column] = df[column].fillna("Unknown")


for column in text_columns:
    if column in df.columns:
        df[column] = df[column].fillna("")


for column in zero_fill_columns:
    if column in df.columns:
        df[column] = df[column].fillna(0)


# Remove records without essential mix design information
rows_before_cleaning = len(df)

df = df.dropna(
    subset=required_columns
)

rows_after_cleaning = len(df)


# Remove duplicated records
duplicates_before = int(df.duplicated().sum())

df = df.drop_duplicates()

duplicates_after = int(df.duplicated().sum())


# Generate quality control summary
qc_report = pd.DataFrame(
    {
        "column": df.columns,
        "dtype": [str(df[column].dtype) for column in df.columns],
        "missing_after_cleaning": [
            int(df[column].isna().sum())
            for column in df.columns
        ],
    }
)


qc_report.to_csv(
    qc_report_file,
    index=False
)

df.to_csv(
    output_file,
    index=False
)


print(f"Cleaned dataset shape: {df.shape}")
print(
    f"Rows removed due to missing required fields: "
    f"{rows_before_cleaning - rows_after_cleaning}"
)
print(
    f"Duplicates removed: {duplicates_before - duplicates_after}"
)

print(f"Saved dataset: {output_file}")
print(f"Saved QC report: {qc_report_file}")