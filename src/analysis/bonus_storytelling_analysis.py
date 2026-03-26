"""
TableTalk - Bonus: Storytelling Audio Analysis
Analyzes features that distinguish narrative/storytelling speech from conversational speech.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats


# Heuristic: label "storytelling" vs "conversational"
# RAVDESS "strong intensity" recordings are more performative/theatrical
#  used as a proxy for storytelling delivery


def assign_storytelling_label(row: pd.Series) -> str:
    """
    Heuristic labelling for storytelling vs conversational speech.
    Strong intensity + dramatic/suspense/urgency tones → storytelling proxy.
    """
    narrative_tone = str(row.get("narrative_tone", ""))
    intensity      = str(row.get("intensity", ""))

    storytelling_tones = {"suspense", "dramatic_emphasis", "urgency"}
    if intensity == "strong" or narrative_tone in storytelling_tones:
        return "storytelling"
    return "conversational"


# Feature analysis

ANALYSIS_FEATURES = [
    "duration_sec",
    "pitch_mean_hz",
    "pitch_std_hz",
    "pitch_range_hz",
    "rms_energy_mean",
    "rms_energy_std",
    "silence_ratio",
    "speech_ratio",
    "tempo_bpm",
    "spectral_centroid_mean",
    "zcr_mean",
    "voiced_fraction",
]


def analyze_storytelling_features(df: pd.DataFrame, output_dir: str = "./outputs") -> dict:
    """
    Compare storytelling vs conversational speech across acoustic features.
    Returns statistical summary with effect sizes.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Assign labels
    from src.models.task2_classification import NARRATIVE_MAP
    if "narrative_tone" not in df.columns:
        df["narrative_tone"] = df["emotion_label"].map(NARRATIVE_MAP)

    df["speech_type"] = df.apply(assign_storytelling_label, axis=1)
    print(f"\n[Bonus] Speech type distribution:\n{df['speech_type'].value_counts().to_string()}\n")

    story = df[df["speech_type"] == "storytelling"]
    conv  = df[df["speech_type"] == "conversational"]

    available = [f for f in ANALYSIS_FEATURES if f in df.columns]
    summary_rows = []

    for feat in available:
        s_vals = story[feat].dropna().values
        c_vals = conv[feat].dropna().values
        if len(s_vals) < 3 or len(c_vals) < 3:
            continue

        t_stat, p_val = stats.ttest_ind(s_vals, c_vals, equal_var=False)
        # Cohen's d effect size
        pooled_std = np.sqrt((np.std(s_vals)**2 + np.std(c_vals)**2) / 2)
        cohens_d   = (np.mean(s_vals) - np.mean(c_vals)) / pooled_std if pooled_std > 0 else 0.0

        summary_rows.append({
            "feature":         feat,
            "storytelling_mean": round(np.mean(s_vals), 5),
            "storytelling_std":  round(np.std(s_vals),  5),
            "conv_mean":         round(np.mean(c_vals), 5),
            "conv_std":          round(np.std(c_vals),  5),
            "t_statistic":       round(t_stat,    4),
            "p_value":           round(p_val,     6),
            "cohens_d":          round(cohens_d,  4),
            "significant":       p_val < 0.05,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("cohens_d", key=abs, ascending=False)
    summary_df.to_csv(out / "storytelling_analysis.csv", index=False)

    print("Feature Discriminability (sorted by |Cohen's d|):")
    print(summary_df[["feature","storytelling_mean","conv_mean","p_value","cohens_d","significant"]]
          .to_string(index=False))

    # ── Plots ──
    _plot_feature_distributions(df, available[:6], out)
    _plot_effect_sizes(summary_df, out)

    return summary_df.to_dict("records")


# Visualisation helpers

def _plot_feature_distributions(df: pd.DataFrame, features: list, out: Path):
    """Violin plots comparing storytelling vs conversational for key features."""
    n   = len(features)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    palette = {"storytelling": "#E05C5C", "conversational": "#5C9BE0"}

    for ax, feat in zip(axes, features):
        sns.violinplot(data=df, x="speech_type", y=feat,
                       palette=palette, inner="box", ax=ax)
        ax.set_title(feat.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Storytelling vs Conversational Speech — Feature Distributions",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out / "storytelling_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Bonus] Distribution plot saved → {out / 'storytelling_distributions.png'}")


def _plot_effect_sizes(summary_df: pd.DataFrame, out: Path):
    """Horizontal bar chart of Cohen's d effect sizes."""
    df = summary_df.copy()
    df["color"] = df["cohens_d"].apply(lambda d: "#E05C5C" if d > 0 else "#5C9BE0")

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(df["feature"], df["cohens_d"], color=df["color"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline( 0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.axvline(-0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_xlabel("Cohen's d  (positive = higher in storytelling)", fontsize=11)
    ax.set_title("Effect Size: Storytelling vs Conversational Speech", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    # Significance markers
    for bar, (_, row) in zip(bars, df.iterrows()):
        if row["significant"]:
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    "*", va="center", fontsize=12, color="darkgreen")

    plt.tight_layout()
    plt.savefig(out / "storytelling_effect_sizes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Bonus] Effect size plot saved → {out / 'storytelling_effect_sizes.png'}")


# Entry point

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TableTalk Storytelling Analysis")
    parser.add_argument("--csv",        type=str, default="./outputs/features_dataset.csv")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    analyze_storytelling_features(df, args.output_dir)