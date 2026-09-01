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
    repo_relative,
    safe_name,
)


warnings.filterwarnings("ignore")

DPI = 600
INPUT_DIR = APPLICATION_DIR / "mix_optimization"
OUTPUT_DIR = APPLICATION_DIR / "final_recommendations"
COMPOSITION_FEATURES = [
    "Cement_kgm3",
    "Water_kgm3",
    "SCM1_kgm3",
    "SCM2_kgm3",
    "FineAgg_kgm3",
    "CoarseAgg_kgm3",
    "SP_kgm3",
    "Fiber_kgm3",
]
COLORS = {
    "Cement_kgm3": "#d62728",
    "Water_kgm3": "#1f77b4",
    "SCM1_kgm3": "#2ca02c",
    "SCM2_kgm3": "#bcbd22",
    "FineAgg_kgm3": "#ff7f0e",
    "CoarseAgg_kgm3": "#8c564b",
    "SP_kgm3": "#9467bd",
    "Fiber_kgm3": "#17becf",
}


def read_ranking(path):
    data = pd.read_csv(path)
    required = {
        "Rank",
        "Predicted_fc_MPa",
        "Candidate_ID",
        "Age_day",
        "W_over_B",
        "Family",
        "Model",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    numeric = ["Rank", "Predicted_fc_MPa", *FULL_FEATURES]
    for column in numeric:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["Rank", "Predicted_fc_MPa", *FULL_FEATURES])
    if data.empty:
        raise ValueError("Ranking file contains no valid candidates.")
    return data.sort_values("Rank", ignore_index=True)


def composition_values(top):
    return [float(top[feature]) for feature in COMPOSITION_FEATURES]


def draw_composition(axis, top):
    values = composition_values(top)
    positions = np.arange(len(COMPOSITION_FEATURES))
    bars = axis.barh(
        positions,
        values,
        color=[COLORS[feature] for feature in COMPOSITION_FEATURES],
        edgecolor="black",
        linewidth=0.55,
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(
        [DISPLAY_LABELS[feature].split(" (")[0] for feature in COMPOSITION_FEATURES]
    )
    axis.invert_yaxis()
    axis.set_xlabel("Content (kg m$^{-3}$)")
    axis.set_title("Recommended candidate composition", fontweight="bold")
    axis.grid(axis="x", linestyle="--", alpha=0.2)

    limit = max(values) if values else 1.0
    offset = max(limit * 0.012, 0.02)
    for bar, value in zip(bars, values):
        axis.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            fontsize=8,
        )


def draw_design_space(axis, ranking, top):
    axis.scatter(
        ranking["Age_day"],
        ranking["W_over_B"],
        s=18,
        color="#b8bec7",
        alpha=0.55,
        label="Observed candidates",
    )
    axis.scatter(
        top["Age_day"],
        top["W_over_B"],
        marker="*",
        s=240,
        color="#d62728",
        edgecolors="black",
        linewidth=0.7,
        label="Rank-one candidate",
        zorder=5,
    )
    axis.set_xlabel(DISPLAY_LABELS["Age_day"])
    axis.set_ylabel(DISPLAY_LABELS["W_over_B"])
    axis.set_title("Location in the observed design space", fontweight="bold")
    axis.grid(linestyle="--", alpha=0.2)
    axis.legend(frameon=False)


def recommendation_text(top):
    binder = sum(float(top[feature]) for feature in [
        "Cement_kgm3", "SCM1_kgm3", "SCM2_kgm3"
    ])
    lines = [
        f"Candidate: {top['Candidate_ID']}",
        f"Predicted strength: {top['Predicted_fc_MPa']:.2f} MPa",
        f"Age: {top['Age_day']:.0f} day",
        f"W/B: {top['W_over_B']:.3f}",
        f"Binder: {binder:.1f} kg m$^{{-3}}$",
    ]
    if "MeanModelPercentile" in top and pd.notna(top["MeanModelPercentile"]):
        lines.append(f"Consensus percentile: {top['MeanModelPercentile']:.3f}")
    return "\n".join(lines)


def save_figures(ranking, scenario, family, model, output_dir):
    top = ranking.iloc[0]
    label = f"{family} | {model}"
    stem = safe_name(label)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.7))
    draw_composition(axes[0], top)
    draw_design_space(axes[1], ranking, top)
    axes[1].text(
        0.03,
        0.04,
        recommendation_text(top),
        transform=axes[1].transAxes,
        va="bottom",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.92},
    )
    fig.suptitle(
        f"Rank-one observed candidate: {label}\n{SCENARIOS[scenario]['label']}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    full_path = output_dir / f"{stem}_final_recommendation_600dpi.png"
    fig.savefig(full_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.0, 5.7))
    draw_composition(axis, top)
    axis.text(
        0.98,
        0.04,
        recommendation_text(top),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.92},
    )
    fig.tight_layout()
    composition_path = output_dir / f"{stem}_composition_600dpi.png"
    fig.savefig(composition_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.8, 5.7))
    draw_design_space(axis, ranking, top)
    axis.text(
        0.03,
        0.04,
        recommendation_text(top),
        transform=axis.transAxes,
        va="bottom",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.92},
    )
    fig.tight_layout()
    design_path = output_dir / f"{stem}_design_space_600dpi.png"
    fig.savefig(design_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return full_path, composition_path, design_path


def process_file(path):
    ranking = read_ranking(path)
    scenario = str(ranking.iloc[0]["Scenario"])
    family = str(ranking.iloc[0]["Family"])
    model = str(ranking.iloc[0]["Model"])
    output_dir = OUTPUT_DIR / f"scenario_{scenario}"
    output_dir.mkdir(parents=True, exist_ok=True)
    full, composition, design = save_figures(
        ranking, scenario, family, model, output_dir
    )

    top = ranking.iloc[0]
    return {
        "Scenario": scenario,
        "Family": family,
        "Model": model,
        "Candidate_ID": top["Candidate_ID"],
        "Predicted_fc_MPa": top["Predicted_fc_MPa"],
        "SourceRanking": repo_relative(path),
        "FullFigure": repo_relative(full),
        "CompositionFigure": repo_relative(composition),
        "DesignSpaceFigure": repo_relative(design),
        **{feature: top[feature] for feature in FULL_FEATURES},
    }


def main():
    candidate_files = sorted(
        INPUT_DIR.glob("scenario_*/all_candidates/*_all_candidates.csv")
    )
    if not candidate_files:
        raise FileNotFoundError(
            f"No candidate rankings found in {INPUT_DIR}. "
            "Run 02_mix_optimization_all_models.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    failures = []
    for path in candidate_files:
        try:
            summaries.append(process_file(path))
            print(f"Created final recommendation from: {path.name}")
        except Exception as exc:
            failures.append({"SourceRanking": repo_relative(path), "Error": str(exc)})
            print(f"[FAILED] {path.name}: {exc}")

    pd.DataFrame(summaries).to_csv(
        OUTPUT_DIR / "final_recommendations_summary.csv", index=False
    )
    pd.DataFrame(failures, columns=["SourceRanking", "Error"]).to_csv(
        OUTPUT_DIR / "final_recommendation_failures.csv", index=False
    )
    if not summaries:
        raise RuntimeError("No final recommendation figure was generated.")
    print(f"Final recommendation figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
