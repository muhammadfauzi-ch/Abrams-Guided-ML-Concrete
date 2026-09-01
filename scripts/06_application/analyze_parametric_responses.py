import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _application_common import (
    APPLICATION_DIR,
    DISPLAY_LABELS,
    FULL_FEATURES,
    SCENARIOS,
    load_candidates,
    load_scenario_artifacts,
    model_file_stem,
    physically_consistent_grid,
    predict_model,
    repo_relative,
)


warnings.filterwarnings("ignore")

DPI = 600
GRID_POINTS = 30
RESPONSE_FEATURES = ["Age_day", "W_over_B", "Cement_kgm3"]
OUTPUT_DIR = APPLICATION_DIR / "parametric_response"
COLORS = {
    "Age_day": "#1e88e5",
    "W_over_B": "#fb8c00",
    "Cement_kgm3": "#e53935",
}


def response_grid(candidates, feature):
    values = candidates[feature].dropna()
    low, high = np.quantile(values, [0.05, 0.95])
    if np.isclose(low, high):
        return np.array([low])
    return np.linspace(low, high, GRID_POINTS)


def calculate_curves(record, candidates, baseline):
    blocks = []
    for feature in RESPONSE_FEATURES:
        grid = response_grid(candidates, feature)
        counterfactuals = physically_consistent_grid(baseline, feature, grid)
        prediction = predict_model(record, counterfactuals)
        block = counterfactuals.copy()
        block.insert(0, "GridValue", grid)
        block.insert(0, "Feature", feature)
        block.insert(0, "Model", record["model"])
        block.insert(0, "Family", record["family"])
        block["Predicted_fc_MPa"] = prediction
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def plot_curves(record, curves, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    for axis, feature in zip(axes, RESPONSE_FEATURES):
        subset = curves.loc[curves["Feature"] == feature]
        axis.plot(
            subset["GridValue"],
            subset["Predicted_fc_MPa"],
            color=COLORS[feature],
            linewidth=2.3,
        )
        axis.set_xlabel(DISPLAY_LABELS[feature])
        axis.set_ylabel("Predicted compressive strength (MPa)")
        axis.set_title(f"Response to {DISPLAY_LABELS[feature]}", fontweight="bold")
        axis.grid(True, linestyle="--", alpha=0.22)

    label = f"{record['family']} | {record['model']}"
    fig.suptitle(f"Parametric response: {label}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def run_scenario(scenario, run_rows, failure_rows):
    candidates, candidate_source = load_candidates(scenario)
    records, artifact_failures, manifest_path = load_scenario_artifacts(scenario)
    failure_rows.extend(artifact_failures)

    scenario_dir = OUTPUT_DIR / f"scenario_{scenario}"
    figure_dir = scenario_dir / "figures"
    table_dir = scenario_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    baseline = candidates[FULL_FEATURES].median(numeric_only=True)
    binder = (
        baseline["Cement_kgm3"]
        + baseline["SCM1_kgm3"]
        + baseline["SCM2_kgm3"]
    )
    baseline["W_over_B"] = baseline["Water_kgm3"] / binder
    pd.DataFrame([baseline]).to_csv(
        table_dir / "parametric_baseline_mix.csv", index=False
    )

    for record in records:
        stem = model_file_stem(record)
        try:
            curves = calculate_curves(record, candidates, baseline)
            curve_path = table_dir / f"{stem}_parametric_response.csv"
            figure_path = figure_dir / f"{stem}_parametric_response_600dpi.png"
            curves.to_csv(curve_path, index=False)
            plot_curves(record, curves, figure_path)
            run_rows.append(
                {
                    "Scenario": scenario,
                    "ScenarioLabel": SCENARIOS[scenario]["label"],
                    "Family": record["family"],
                    "Model": record["model"],
                    "CandidateSource": candidate_source,
                    "ArtifactManifest": repo_relative(manifest_path),
                    "Artifact": repo_relative(record["artifact_path"]),
                    "CurveData": repo_relative(curve_path),
                    "Figure": repo_relative(figure_path),
                    "Status": "completed",
                }
            )
            print(f"Completed scenario {scenario}: {record['family']} | {record['model']}")
        except Exception as exc:
            failure_rows.append(
                {
                    "Scenario": scenario,
                    "Family": record["family"],
                    "Model": record["model"],
                    "Artifact": repo_relative(record["artifact_path"]),
                    "Error": str(exc),
                }
            )
            print(f"[FAILED] scenario {scenario} | {record['model']}: {exc}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_rows = []
    failure_rows = []

    for scenario in SCENARIOS:
        try:
            run_scenario(scenario, run_rows, failure_rows)
        except Exception as exc:
            failure_rows.append(
                {
                    "Scenario": scenario,
                    "Family": "",
                    "Model": "",
                    "Artifact": "",
                    "Error": str(exc),
                }
            )
            print(f"[FAILED] scenario {scenario}: {exc}")

    pd.DataFrame(run_rows).to_csv(
        OUTPUT_DIR / "parametric_response_manifest.csv", index=False
    )
    pd.DataFrame(
        failure_rows,
        columns=["Scenario", "Family", "Model", "Artifact", "Error"],
    ).to_csv(OUTPUT_DIR / "parametric_response_failures.csv", index=False)

    if not run_rows:
        raise RuntimeError(
            "No parametric response was generated. Check the failure report and prerequisites."
        )
    print(f"Parametric-response outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
