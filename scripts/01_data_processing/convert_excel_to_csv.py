from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

raw_dir = ROOT / "data" / "raw"
processed_dir = ROOT / "data" / "processed"

processed_dir.mkdir(parents=True, exist_ok=True)


# Input datasets
literature_file = raw_dir / "1. Data Literature.xlsx"
exp1_file = raw_dir / "2. Data Exp 1.xlsx"
exp2_file = raw_dir / "3. Data Exp 2.xlsx"


def convert_excel(input_file, sheet_name, output_file):
    """Convert selected Excel sheet into CSV format."""

    data = pd.read_excel(
        input_file,
        sheet_name=sheet_name
    )

    data.to_csv(
        output_file,
        index=False
    )

    print(
        f"{output_file.name} saved | Shape: {data.shape}"
    )


def main():

    convert_excel(
        literature_file,
        "Master_All",
        processed_dir / "literature_master_all.csv"
    )

    convert_excel(
        exp1_file,
        "Harmonized_Data",
        processed_dir / "exp1_harmonized.csv"
    )

    convert_excel(
        exp2_file,
        "Harmonized_Data",
        processed_dir / "exp2_harmonized.csv"
    )

    print("Excel conversion completed.")


if __name__ == "__main__":
    main()