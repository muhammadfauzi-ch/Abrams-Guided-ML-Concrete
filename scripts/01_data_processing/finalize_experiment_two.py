from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data" / "processed" / "exp2_harmonized.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "finaldataexp2.csv"
OUTPUT_XLSX = ROOT / "data" / "processed" / "finaldataexp2.xlsx"


TARGET_COLUMNS = [
    "Study_ID",
    "DOI",
    "Domain_Tag",
    "Mix_ID",
    "Water_type",
    "Age_day",
    "fc_MPa",
    "Cement_kgm3",
    "Water_kgm3",
    "W_over_B",
    "SCM1_name",
    "SCM1_kgm3",
    "SCM2_name",
    "SCM2_kgm3",
    "FineAgg_kgm3",
    "CoarseAgg_kgm3",
    "SP_kgm3",
    "Slump_mm",
    "Fiber_kgm3",
    "Fiber_vol_pct",
    "Fiber_shape",
    "Notes",
]


def read_data(path):

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def convert_numeric_columns(data, columns):

    for column in columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    return data


def build_notes(row):

    note_columns = [
        "Description",
        "Mix_number_status",
        "Mix_family",
        "Temperature_condition",
        "Chemical_Admixture",
        "Chemical_Dosage_mL",
        "Assumption_note",
        "Mix_number_decoded",
        "Code_meaning_source",
        "Original_Type",
        "Replicate_No",
        "Data_quality_flag",
    ]

    notes = []

    for column in note_columns:
        if column in row.index:
            value = row[column]

            if pd.notna(value) and str(value).strip():
                notes.append(
                    f"{column}={value}"
                )

    if "Chemical_Dosage_mL" in row.index:
        dosage = row["Chemical_Dosage_mL"]

        if pd.notna(dosage) and float(dosage) != 0:
            notes.append(
                "SP_kgm3 left blank because admixture is only available in mL dosage"
            )

    return " | ".join(dict.fromkeys(notes))


def create_base_dataset(data):

    output = pd.DataFrame(index=data.index)

    output["Study_ID"] = "EXP2_FINAL"
    output["DOI"] = np.nan
    output["Domain_Tag"] = "Experimental_Exp2"

    output["Mix_ID"] = data.get(
        "Specimen_ID",
        data.get("Mix_number", np.nan)
    )

    output["Water_type"] = np.nan

    output["Cement_kgm3"] = pd.to_numeric(
        data.get("Cement_kgm3", np.nan),
        errors="coerce"
    )

    output["Water_kgm3"] = pd.to_numeric(
        data.get("Water_kgm3", np.nan),
        errors="coerce"
    )

    scm_content = pd.to_numeric(
        data.get("SCM_kgm3", 0),
        errors="coerce"
    ).fillna(0)

    binder_content = (
        output["Cement_kgm3"] +
        scm_content
    )

    output["W_over_B"] = np.where(
        (binder_content.notna()) & (binder_content != 0),
        output["Water_kgm3"] / binder_content,
        np.nan
    )

    output["SCM1_name"] = data.get(
        "SCM_type",
        np.nan
    )

    output["SCM1_kgm3"] = scm_content
    output["SCM2_name"] = np.nan
    output["SCM2_kgm3"] = 0.0

    output["FineAgg_kgm3"] = pd.to_numeric(
        data.get("FA_kgm3", np.nan),
        errors="coerce"
    )

    output["CoarseAgg_kgm3"] = pd.to_numeric(
        data.get("CA_kgm3", np.nan),
        errors="coerce"
    )

    output["SP_kgm3"] = np.nan
    output["Slump_mm"] = np.nan
    output["Fiber_kgm3"] = 0.0
    output["Fiber_vol_pct"] = 0.0
    output["Fiber_shape"] = np.nan

    output["Notes"] = data.apply(
        build_notes,
        axis=1
    )

    return output


def convert_strength_to_long_format(data, base_dataset):

    records = []

    strength_columns = {
        28: "fc_28_MPa",
        56: "fc_56_MPa",
    }

    for age, column in strength_columns.items():

        if column in data.columns:

            temp = base_dataset.copy()

            temp["Age_day"] = age
            temp["fc_MPa"] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

            temp = temp[
                temp["fc_MPa"].notna()
            ]

            records.append(temp)


    if not records:
        raise ValueError(
            "No compressive strength columns were found."
        )

    return pd.concat(
        records,
        ignore_index=True
    )


def apply_schema(data):

    for column in TARGET_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan

    return data[TARGET_COLUMNS]


def main():

    data = read_data(INPUT_PATH)

    data = convert_numeric_columns(
        data,
        [
            "fc_28_MPa",
            "fc_56_MPa",
            "Water_kgm3",
            "Cement_kgm3",
            "SCM_kgm3",
            "CA_kgm3",
            "FA_kgm3",
            "Chemical_Dosage_mL",
        ]
    )

    base_dataset = create_base_dataset(data)

    final_dataset = convert_strength_to_long_format(
        data,
        base_dataset
    )

    final_dataset = apply_schema(
        final_dataset
    )


    final_dataset.to_csv(
        OUTPUT_CSV,
        index=False
    )

    with pd.ExcelWriter(
        OUTPUT_XLSX,
        engine="openpyxl"
    ) as writer:

        final_dataset.to_excel(
            writer,
            sheet_name="finaldataexp2",
            index=False
        )


    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_XLSX}")
    print(f"Dataset shape: {final_dataset.shape}")


if __name__ == "__main__":
    main()