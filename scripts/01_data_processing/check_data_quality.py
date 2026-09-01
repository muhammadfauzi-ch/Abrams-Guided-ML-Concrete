from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

processed_dir = ROOT / "data" / "processed"
output_dir = ROOT / "outputs" / "qc"

output_dir.mkdir(parents=True, exist_ok=True)


dataset_files = {
    "literature": processed_dir / "literature_master_all.csv",
    "exp1": processed_dir / "exp1_harmonized.csv",
    "exp2": processed_dir / "exp2_harmonized.csv",
}


summary_file = output_dir / "data_check_summary.csv"
schema_file = output_dir / "data_check_schema_comparison.csv"
missing_file = output_dir / "data_check_missing_values.csv"


def load_dataset(file_path):
    return pd.read_csv(file_path)


def summarize_dataset(dataset_name, data):
    return {
        "Dataset": dataset_name,
        "Rows": len(data),
        "Columns": len(data.columns),
        "Duplicate_Rows": int(data.duplicated().sum()),
    }


def collect_schema_information(dataset_name, data):
    records = []

    for column in data.columns:
        records.append(
            {
                "Dataset": dataset_name,
                "Column": column,
                "Dtype": str(data[column].dtype),
                "NonNull_Count": int(data[column].notna().sum()),
                "Null_Count": int(data[column].isna().sum()),
            }
        )

    return records


def collect_missing_information(dataset_name, data):
    records = []

    for column in data.columns:
        missing_count = int(data[column].isna().sum())
        missing_percentage = (
            missing_count / len(data) * 100
            if len(data) > 0
            else 0
        )

        records.append(
            {
                "Dataset": dataset_name,
                "Column": column,
                "Null_Count": missing_count,
                "Null_Percent": round(missing_percentage, 2),
            }
        )

    return records


def main():

    summary_records = []
    schema_records = []
    missing_records = []

    loaded_datasets = {}

    for name, file_path in dataset_files.items():

        if not file_path.exists():
            print(f"Missing dataset: {file_path}")
            continue

        data = load_dataset(file_path)

        loaded_datasets[name] = data

        print(
            f"{name} loaded | Shape: {data.shape}"
        )

        summary_records.append(
            summarize_dataset(name, data)
        )

        schema_records.extend(
            collect_schema_information(name, data)
        )

        missing_records.extend(
            collect_missing_information(name, data)
        )


    pd.DataFrame(summary_records).to_csv(
        summary_file,
        index=False
    )

    pd.DataFrame(schema_records).to_csv(
        schema_file,
        index=False
    )

    pd.DataFrame(missing_records).to_csv(
        missing_file,
        index=False
    )


    print("\nDataset quality check completed.")
    print(f"Saved: {summary_file}")
    print(f"Saved: {schema_file}")
    print(f"Saved: {missing_file}")


if __name__ == "__main__":
    main()