from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

literature_path = ROOT / "data" / "processed" / "literature_train_ready.csv"
exp1_path = ROOT / "data" / "processed" / "finaldataexp1.csv"
exp2_path = ROOT / "data" / "processed" / "finaldataexp2.csv"

output_exp12 = ROOT / "data" / "processed" / "finaldataexp12.csv"
output_all = ROOT / "data" / "processed" / "finaldata_all.csv"
output_summary = ROOT / "data" / "processed" / "finaldata_sources_summary.csv"


target_columns = [
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


def read_dataset(path, name):

    if not path.exists():
        raise FileNotFoundError(
            f"{name} dataset not found: {path}"
        )

    return pd.read_csv(path)


def apply_schema(data, source_name):

    for column in target_columns:
        if column not in data.columns:
            data[column] = pd.NA

    data = data[target_columns].copy()

    data["Source_File"] = source_name

    return data


def main():

    literature = read_dataset(
        literature_path,
        "Literature"
    )

    exp1 = read_dataset(
        exp1_path,
        "Exp1"
    )

    exp2 = read_dataset(
        exp2_path,
        "Exp2"
    )


    literature = apply_schema(
        literature,
        "literature_train_ready.csv"
    )

    exp1 = apply_schema(
        exp1,
        "finaldataexp1.csv"
    )

    exp2 = apply_schema(
        exp2,
        "finaldataexp2.csv"
    )


    exp12 = pd.concat(
        [exp1, exp2],
        ignore_index=True
    )

    all_dataset = pd.concat(
        [literature, exp1, exp2],
        ignore_index=True
    )


    exp12 = exp12.drop_duplicates()
    all_dataset = all_dataset.drop_duplicates()


    exp12.to_csv(
        output_exp12,
        index=False
    )

    all_dataset.to_csv(
        output_all,
        index=False
    )


    summary = pd.DataFrame(
        {
            "Dataset": [
                "Literature",
                "Exp1",
                "Exp2",
                "Exp1+Exp2",
                "Literature+Exp1+Exp2",
            ],
            "Rows": [
                len(literature),
                len(exp1),
                len(exp2),
                len(exp12),
                len(all_dataset),
            ],
        }
    )

    summary.to_csv(
        output_summary,
        index=False
    )


    print(f"Saved: {output_exp12}")
    print(f"Saved: {output_all}")
    print(f"Saved: {output_summary}")

    print(
        f"Dataset sizes | Literature: {literature.shape}, "
        f"Exp1: {exp1.shape}, "
        f"Exp2: {exp2.shape}, "
        f"Combined: {all_dataset.shape}"
    )


if __name__ == "__main__":
    main()