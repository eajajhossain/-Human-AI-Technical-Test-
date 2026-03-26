"""
TableTalk - Task 4: Narrative Audio Retrieval System
Prototype system that retrieves voice recordings based on natural-language queries
about narrative characteristics (tone, energy, duration, pacing, etc.).
"""

import re
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


# Data model

@dataclass
class Recording:
    """A single audio recording with metadata and extracted features."""
    filepath:         str
    filename:         str
    emotion_label:    str         = "unknown"
    narrative_tone:   str         = "unknown"
    duration_sec:     float       = 0.0
    rms_energy_mean:  float       = 0.0
    pitch_mean_hz:    float       = 0.0
    pitch_range_hz:   float       = 0.0
    speech_ratio:     float       = 1.0
    silence_ratio:    float       = 0.0
    tempo_bpm:        float       = 0.0
    spectral_centroid_mean: float = 0.0
    transcript:       str         = ""
    actor:            str         = ""
    intensity:        str         = ""

    def to_dict(self) -> dict:
        return asdict(self)


# Query parser
# Converts natural-language queries into structured filter rules


# Energy thresholds (tuned for RAVDESS RMS values ~0.01 – 0.15)
ENERGY_THRESHOLDS = {
    "low":    (0.000, 0.030),
    "medium": (0.030, 0.070),
    "high":   (0.070, 1.000),
}

TONE_ALIASES = {
    "calm":         "calm_description",
    "calm narration": "calm_description",
    "quiet":        "calm_description",
    "soft":         "calm_description",
    "suspense":     "suspense",
    "tense":        "suspense",
    "fearful":      "suspense",
    "scary":        "suspense",
    "urgent":       "urgency",
    "urgency":      "urgency",
    "intense":      "urgency",
    "angry":        "urgency",
    "dramatic":     "dramatic_emphasis",
    "dramatic emphasis": "dramatic_emphasis",
    "emotional":    "dramatic_emphasis",
    "dialogue":     "character_dialogue",
    "character dialogue": "character_dialogue",
    "happy":        "character_dialogue",
    "conversational": "character_dialogue",
}


def parse_query(query: str) -> dict:
    """
    Parse a natural-language retrieval query into filter rules.

    Supported patterns:
      - "calm narration longer than 4 seconds"
      - "high-energy speech"
      - "dramatic dialogue"
      - "suspense shorter than 3s"
      - "calm narration between 3 and 6 seconds with low energy"
      - "urgency with high pitch"
      - keyword: calm | suspense | urgency | dramatic | dialogue
      - energy:  low | medium | high
      - duration: longer/shorter than N seconds, between N and M seconds
    """
    q = query.lower().strip()
    filters = {}

    #  Narrative tone 
    for alias, tone in TONE_ALIASES.items():
        if alias in q:
            filters["narrative_tone"] = tone
            break

    #  Energy level 
    for level, (lo, hi) in ENERGY_THRESHOLDS.items():
        if f"{level}-energy" in q or f"{level} energy" in q or f"{level} rms" in q:
            filters["energy_range"] = (lo, hi)
            break

    #  Duration filters 
    m = re.search(r"longer than (\d+\.?\d*)\s*s", q)
    if m:
        filters["min_duration"] = float(m.group(1))

    m = re.search(r"shorter than (\d+\.?\d*)\s*s", q)
    if m:
        filters["max_duration"] = float(m.group(1))

    m = re.search(r"between (\d+\.?\d*) and (\d+\.?\d*)\s*s", q)
    if m:
        filters["min_duration"] = float(m.group(1))
        filters["max_duration"] = float(m.group(2))

    #  Pitch 
    if "high pitch" in q or "high-pitch" in q:
        filters["min_pitch"] = 200.0
    elif "low pitch" in q or "low-pitch" in q:
        filters["max_pitch"] = 150.0

    # Pacing / silence 
    if "slow" in q or "slow paced" in q or "pauses" in q:
        filters["min_silence_ratio"] = 0.30
    elif "fast" in q or "fast paced" in q:
        filters["max_silence_ratio"] = 0.20

    #  Transcript keyword search 
    m = re.search(r'transcript[:\s]+"?([^"]+)"?', q)
    if m:
        filters["transcript_keyword"] = m.group(1).strip()

    return filters


# Retrieval engine

class NarrativeRetrievalSystem:
    """
    Filters and ranks narrative audio recordings based on structured queries.
    Supports exact filters and similarity-based ranking.
    """

    def __init__(self):
        self.recordings: list[Recording] = []

    #  Index building 
    def build_index_from_csv(self, features_csv: str,
                              transcripts_csv: Optional[str] = None,
                              predicted_tones_csv: Optional[str] = None) -> None:
        """Load recordings from the feature CSV produced by Task 1."""
        df = pd.read_csv(features_csv)

        # Merge transcripts if available
        if transcripts_csv and Path(transcripts_csv).exists():
            tr = pd.read_csv(transcripts_csv)[["filename", "transcript"]]
            df = df.merge(tr, on="filename", how="left")
        else:
            df["transcript"] = ""

        # Merge predicted tones if available
        if predicted_tones_csv and Path(predicted_tones_csv).exists():
            pt = pd.read_csv(predicted_tones_csv)[["filename", "narrative_tone"]]
            df = df.merge(pt, on="filename", how="left")

        # Map emotion -> tone where narrative_tone is missing
        from src.models.task2_classification import NARRATIVE_MAP
        if "narrative_tone" not in df.columns:
            df["narrative_tone"] = df["emotion_label"].map(NARRATIVE_MAP)
        else:
            mask = df["narrative_tone"].isna()
            df.loc[mask, "narrative_tone"] = df.loc[mask, "emotion_label"].map(NARRATIVE_MAP)

        for _, row in df.iterrows():
            rec = Recording(
                filepath               = str(row.get("filepath", "")),
                filename               = str(row.get("filename", "")),
                emotion_label          = str(row.get("emotion_label", "unknown")),
                narrative_tone         = str(row.get("narrative_tone", "unknown")),
                duration_sec           = float(row.get("duration_sec", 0)),
                rms_energy_mean        = float(row.get("rms_energy_mean", 0)),
                pitch_mean_hz          = float(row.get("pitch_mean_hz", 0)),
                pitch_range_hz         = float(row.get("pitch_range_hz", 0)),
                speech_ratio           = float(row.get("speech_ratio", 1)),
                silence_ratio          = float(row.get("silence_ratio", 0)),
                tempo_bpm              = float(row.get("tempo_bpm", 0)),
                spectral_centroid_mean = float(row.get("spectral_centroid_mean", 0)),
                transcript             = str(row.get("transcript", "")),
                actor                  = str(row.get("actor", "")),
                intensity              = str(row.get("intensity", "")),
            )
            self.recordings.append(rec)

        print(f"[Task4] Index built with {len(self.recordings)} recordings.")

    #  Query interface
    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Execute a natural-language query and return the top_k matching recordings.

        Args:
            query: natural-language query string
            top_k: number of results to return

        Returns:
            list of dicts with match metadata + scores
        """
        filters = parse_query(query)
        print(f"\n[Query] '{query}'")
        print(f"[Filters] {json.dumps(filters, indent=2)}")

        candidates = self._apply_filters(self.recordings, filters)
        ranked     = self._rank(candidates, filters)
        results    = ranked[:top_k]

        print(f"[Results] {len(results)} / {len(self.recordings)} recordings matched\n")
        return results

    #  Filter application
    @staticmethod
    def _apply_filters(recordings: list[Recording], filters: dict) -> list[Recording]:
        out = []
        for rec in recordings:
            # Narrative tone
            if "narrative_tone" in filters:
                if rec.narrative_tone != filters["narrative_tone"]:
                    continue

            # Duration
            if "min_duration" in filters and rec.duration_sec < filters["min_duration"]:
                continue
            if "max_duration" in filters and rec.duration_sec > filters["max_duration"]:
                continue

            # Energy
            if "energy_range" in filters:
                lo, hi = filters["energy_range"]
                if not (lo <= rec.rms_energy_mean <= hi):
                    continue

            # Pitch
            if "min_pitch" in filters and rec.pitch_mean_hz < filters["min_pitch"]:
                continue
            if "max_pitch" in filters and rec.pitch_mean_hz > filters["max_pitch"]:
                continue

            # Silence / pacing
            if "min_silence_ratio" in filters and rec.silence_ratio < filters["min_silence_ratio"]:
                continue
            if "max_silence_ratio" in filters and rec.silence_ratio > filters["max_silence_ratio"]:
                continue

            # Transcript keyword
            if "transcript_keyword" in filters:
                kw = filters["transcript_keyword"].lower()
                if kw not in rec.transcript.lower():
                    continue

            out.append(rec)
        return out

    #  Ranking 
    @staticmethod
    def _rank(recordings: list[Recording], filters: dict) -> list[dict]:
        """
        Rank by relevance score. Higher energy recordings are ranked first
        within the same tone; duration is a secondary sort key.
        """
        scored = []
        for rec in recordings:
            score = 0.0
            # Higher energy -> more engaging/dramatic ->ranked higher by default
            score += rec.rms_energy_mean * 10.0
            # Prefer recordings with transcripts
            if rec.transcript:
                score += 0.5
            scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "rank":           i + 1,
                "filename":       rec.filename,
                "narrative_tone": rec.narrative_tone,
                "emotion_label":  rec.emotion_label,
                "duration_sec":   round(rec.duration_sec, 2),
                "rms_energy":     round(rec.rms_energy_mean, 5),
                "pitch_hz":       round(rec.pitch_mean_hz, 1),
                "silence_ratio":  round(rec.silence_ratio, 3),
                "transcript":     rec.transcript[:80] + "..." if len(rec.transcript) > 80 else rec.transcript,
                "filepath":       rec.filepath,
                "score":          round(score, 4),
            }
            for i, (score, rec) in enumerate(scored)
        ]

    #  Demo queries 
    def run_demo_queries(self, output_dir: str = "./outputs"):
        """Run a set of pre-defined example queries and save results."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        demo_queries = [
            "calm narration longer than 4 seconds",
            "high-energy speech",
            "dramatic emphasis",
            "suspense shorter than 3 seconds",
            "urgency with high pitch",
            "slow paced calm description",
            "character dialogue",
            "calm narration between 3 and 6 seconds",
        ]

        all_results = {}
        for q in demo_queries:
            results = self.search(q, top_k=5)
            all_results[q] = results
            if results:
                df = pd.DataFrame(results)
                print(df[["rank","filename","narrative_tone","duration_sec",
                           "rms_energy","pitch_hz"]].to_string(index=False))
            else:
                print("  No results found.")

        # Save all demo results
        with open(out / "retrieval_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[Task4] Demo results saved → {out / 'retrieval_results.json'}")
        return all_results


# Entry point

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TableTalk Audio Retrieval System")
    parser.add_argument("--features_csv",    type=str, default="./outputs/features_dataset.csv")
    parser.add_argument("--transcripts_csv", type=str, default="./outputs/transcripts.csv")
    parser.add_argument("--output_dir",      type=str, default="./outputs")
    parser.add_argument("--query",           type=str, default=None,
                        help="Single query to execute (optional; runs demos if not provided)")
    args = parser.parse_args()

    system = NarrativeRetrievalSystem()
    system.build_index_from_csv(args.features_csv, args.transcripts_csv)

    if args.query:
        results = system.search(args.query)
        print(pd.DataFrame(results).to_string(index=False))
    else:
        system.run_demo_queries(args.output_dir)