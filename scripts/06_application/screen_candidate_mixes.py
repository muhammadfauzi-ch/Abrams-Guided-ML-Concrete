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
    predict_model,
    repo_relative,
)


warnings.filterwarnings("ignore")

DPI = 600
OUTPUT_DIR = APPLICATION_DIR / "mix_optimization"


def rank_candidates(candidates, prediction, ranking_basis="Predicted_fc_MPa"):
    ranked = candidates[["Candidate_ID", "CandidateSource", *FULL_FEATURES]].copy()
    ranked["Predicted_fc_MPa"] = np.asarray(prediction, dtype=float)
    ranked = ranked.sort_values(
        "Predicted_fc_MPa", ascending=False, ignore_index=True
    )
    ranked.insert(0, "Rank", np.arange(1, len(ranked) + 1))
    ranked["RankingBasis"] = ranking_basis
    return ranked


def plot_top_candidates(ranked, title, output_path):
    top = ranked.head(10).copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    plot_data = top.sort_values("Rank", ascending=False)
    axes[0].barh(
        plot_data["Rank"].astype(str),
        plot_data["Predicted_fc_MPa"],
        color="#377eb8",
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0].set_xlabel("Predicted compressive strength (MPa)")
    axes[0].set_ylabel("Candidate rank")
    axes[0].set_title("Top ten observed candidate mixtures", fontweight="bold")
    axes[0].grid(axis="x", linestyle="--", alpha=0.22)

    axes[1].scatter(
        ranked["Age_day"],
        ranked["W_over_B"],
        s=16,
        color="#c5cbd3",
        alpha=0.55,
        label="Candidate set",
    )
    scatter = axes[1].scatter(
        top["Age_day"],
        top["W_over_B"],
        c=top["Predicted_fc_MPa"],
        cmap="viridis",
        s=75,
        edgecolors="black",
        linewidth=0.5,
        label="Top ten",
    )
    first = top.iloc[0]
    axes[1].scatter(
        first["Age_day"],
        first["W_over_B"],
        marker="*",
        s=230,
        color="#d62728",
        edgecolors="black",
        linewidth=0.7,
        label="Rank one",
        zorder=5,
    )
    axes[1].set_xlabel(DISPLAY_LABELS["Age_day"])
    axes[1].set_ylabel(DISPLAY_LABELS["W_over_B"])
    axes[1].set_title("Location in the observed design space", fontweight="bold")
    axes[1].grid(linestyle="--", alpha=0.22)
    axes[1].legend(frameon=False)
    colorbar = fig.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("Predicted compressive strength (MPa)")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def build_consensus(candidates, predictions):
    if not predictions:
        return None

    scores = []
    raw_predictions = []
    for model_key, values in predictions.items():
        series = pd.Series(values, index=candidates["Candidate_ID"])
        scores.append(series.rank(method="average", pct=True).rename(model_key))
        raw_predictions.append(series.rename(model_key))

    score_table = pd.concat(scores, axis=1)
    prediction_table = pd.concat(raw_predictions, axis=1)
    consensus = candidates[
        ["Candidate_ID", "CandidateSource", *FULL_FEATURES]
    ].set_index("Candidate_ID")
    consensus["MeanModelPercentile"] = score_table.mean(axis=1)
    consensus["ModelCount"] = score_table.notna().sum(axis=1)
    consensus["Predicted_fc_MPa"] = prediction_table.mean(axis=1)
    consensus = consensus.sort_values(
        ["MeanModelPercentile", "Predicted_fc_MPa"],
        ascending=[False, False],
    ).reset_index()
    consensus.insert(0, "Rank", np.arange(1, len(consensus) + 1))
    consensus["RankingBasis"] = "MeanModelPercentile"
    return consensus


def top_one_summary(scenario, family, model, ranked, artifact=""):
    top = ranked.iloc[0]
    row = {
        "Scenario": scenario,
        "ScenarioLabel": SCENARIOS[scenario]["label"],
        "Family": family,
        "Model": model,
        "Artifact": artifact,
        "Candidate_ID": top["Candidate_ID"],
        "CandidateSource": top["CandidateSource"],
        "Predicted_fc_MPa": top["Predicted_fc_MPa"],
        "RankingBasis": top["RankingBasis"],
    }
    if "MeanModelPercentile" in top:
        row["MeanModelPercentile"] = top["MeanModelPercentile"]
        row["ModelCount"] = top["ModelCount"]
    row.update({feature: top[feature] for feature in FULL_FEATURES})
    return row


def run_scenario(scenario, summary_rows, run_rows, failure_rows):
    candidates, candidate_source = load_candidates(scenario)
    records, artifact_failures, manifest_path = load_scenario_artifacts(scenario)
    failure_rows.extend(artifact_failures)

    scenario_dir = OUTPUT_DIR / f"scenario_{scenario}"
    all_dir = scenario_dir / "all_candidates"
    top_dir = scenario_dir / "top10"
    figure_dir = scenario_dir / "figures"
    for directory in [all_dir, top_dir, figure_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    model_predictions = {}
    for record in records:
        stem = model_file_stem(record)
        try:
            prediction = predict_model(record, candidates)
            ranked = rank_candidates(candidates, prediction)
            ranked["Scenario"] = scenario
            ranked["Family"] = record["family"]
            ranked["Model"] = record["model"]

            all_path = all_dir / f"{stem}_all_candidates.csv"
            top_path = top_dir / f"{stem}_top10.csv"
            figure_path = figure_dir / f"{stem}_top10_600dpi.png"
            ranked.to_csv(all_path, index=False)
            ranked.head(10).to_csv(top_path, index=False)
            plot_top_candidates(
                ranked,
                f"Strength-based candidate screening: {record['family']} | {record['model']}",
                figure_path,
            )

            key = f"{record['family']} | {record['model']}"
            model_predictions[key] = prediction
            summary_rows.append(
                top_one_summary(
                    scenario,
                    record["family"],
                    record["model"],
                    ranked,
                    repo_relative(record["artifact_path"]),
                )
            )
            run_rows.append(
                {
                    "Scenario": scenario,
                    "Family": record["family"],
                    "Model": record["model"],
                    "CandidateSource": candidate_source,
                    "ArtifactManifest": repo_relative(manifest_path),
                    "AllCandidates": repo_relative(all_path),
                    "Top10": repo_relative(top_path),
                    "Figure": repo_relative(figure_path),
                    "Status": "completed",
                }
            )
            print(f"Completed scenario {scenario}: {key}")
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

    consensus = build_consensus(candidates, model_predictions)
    if consensus is not None:
        consensus["Scenario"] = scenario
        consensus["Family"] = "Consensus"
        consensus["Model"] = "All available models"
        all_path = all_dir / "Consensus__All_available_models_all_candidates.csv"
        top_path = top_dir / "Consensus__All_available_models_top10.csv"
        figure_path = figure_dir / "Consensus__All_available_models_top10_600dpi.png"
        consensus.to_csv(all_path, index=False)
        consensus.head(10).to_csv(top_path, index=False)
        plot_top_candidates(
            consensus,
            f"Consensus candidate screening: scenario {scenario}",
            figure_path,
        )
        summary_rows.append(
            top_one_summary(
                scenario,
                "Consensus",
                "All available models",
                consensus,
            )
        )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    run_rows = []
    failure_rows = []

    for scenario in SCENARIOS:
        try:
            run_scenario(scenario, summary_rows, run_rows, failure_rows)
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

    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "top1_recommendations.csv", index=False
    )
    pd.DataFrame(run_rows).to_csv(
        OUTPUT_DIR / "mix_optimization_manifest.csv", index=False
    )
    pd.DataFrame(
        failure_rows,
        columns=["Scenario", "Family", "Model", "Artifact", "Error"],
    ).to_csv(OUTPUT_DIR / "mix_optimization_failures.csv", index=False)

    if not run_rows:
        raise RuntimeError(
            "No candidate ranking was generated. Check the failure report and prerequisites."
        )
    print(f"Candidate-screening outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
