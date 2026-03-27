"""
TableTalk - Task 1: Audio Processing Pipeline
Processes RAVDESS emotional speech recordings and extracts ML-ready features.
"""

import os
import re
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
import librosa.display
from pathlib import Path
from tqdm import tqdm


# RAVDESS filename parser
# Format: 03-01-06-01-02-01-12.wav
# [Modality]-[VocalChannel]-[Emotion]-[Intensity]-[Statement]-[Repetition]-[Actor]
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

INTENSITY_MAP = {"01": "normal", "02": "strong"}
STATEMENT_MAP = {"01": "kids", "02": "dogs"}  # RAVDESS sentences


def parse_ravdess_filename(filepath: str) -> dict:
    """Extract RAVDESS metadata from filename."""
    name = Path(filepath).stem
    parts = name.split("-")
    if len(parts) != 7:
        return {}
    return {
        "modality": parts[0],
        "vocal_channel": parts[1],
        "emotion_code": parts[2],
        "emotion_label": EMOTION_MAP.get(parts[2], "unknown"),
        "intensity": INTENSITY_MAP.get(parts[3], "unknown"),
        "statement": STATEMENT_MAP.get(parts[4], "unknown"),
        "repetition": parts[5],
        "actor": parts[6],
    }

# Feature extraction

def extract_features(filepath: str, sr: int = 22050) -> dict:
    """
    Extract a rich set of acoustic features from a single audio file.

    Returns:
        dict with all extracted features + metadata
    """
    try:
        y, sr = librosa.load(filepath, sr=sr, mono=True)
    except Exception as e:
        print(f"  [ERROR] Could not load {filepath}: {e}")
        return {}

    features = {}

    #  Duration
    features["duration_sec"] = round(librosa.get_duration(y=y, sr=sr), 4)

    #  MFCCs (13 coefficients → mean & std each) 
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = round(float(np.mean(mfccs[i])), 6)
        features[f"mfcc_{i+1}_std"]  = round(float(np.std(mfccs[i])),  6)

    #  Delta MFCCs (capture temporal dynamics) 
    delta_mfccs = librosa.feature.delta(mfccs)
    for i in range(13):
        features[f"delta_mfcc_{i+1}_mean"] = round(float(np.mean(delta_mfccs[i])), 6)

    #  Pitch (F0 via YIN) 
    f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    f0_voiced = f0[f0 > 0]
    features["pitch_mean_hz"]   = round(float(np.mean(f0_voiced))   if len(f0_voiced) else 0.0, 4)
    features["pitch_std_hz"]    = round(float(np.std(f0_voiced))    if len(f0_voiced) else 0.0, 4)
    features["pitch_range_hz"]  = round(float(np.ptp(f0_voiced))    if len(f0_voiced) else 0.0, 4)
    features["voiced_fraction"] = round(float(len(f0_voiced) / len(f0)), 4)

    # Spectral features 
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"] = round(float(np.mean(spectral_centroid)), 4)
    features["spectral_centroid_std"]  = round(float(np.std(spectral_centroid)),  4)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features["spectral_bandwidth_mean"] = round(float(np.mean(spectral_bandwidth)), 4)

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["spectral_rolloff_mean"] = round(float(np.mean(spectral_rolloff)), 4)

    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features["spectral_contrast_mean"] = round(float(np.mean(spectral_contrast)), 4)

    #  Zero Crossing Rate 
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr_mean"] = round(float(np.mean(zcr)), 6)
    features["zcr_std"]  = round(float(np.std(zcr)),  6)

    #  RMS Energy 
    rms = librosa.feature.rms(y=y)
    features["rms_energy_mean"] = round(float(np.mean(rms)), 6)
    features["rms_energy_std"]  = round(float(np.std(rms)),  6)
    features["rms_energy_max"]  = round(float(np.max(rms)),  6)

    #  Chroma features 
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features["chroma_mean"] = round(float(np.mean(chroma)), 6)
    features["chroma_std"]  = round(float(np.std(chroma)),  6)

    # Tempo / Pacing 
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features["tempo_bpm"] = round(float(np.mean(tempo)), 4) if hasattr(tempo, "__len__") else round(float(tempo), 4)

    #  Pauses (silence ratio) 
    intervals = librosa.effects.split(y, top_db=25)
    total_speech_samples = sum(end - start for start, end in intervals)
    features["speech_ratio"]   = round(total_speech_samples / max(len(y), 1), 4)
    features["silence_ratio"]  = round(1.0 - features["speech_ratio"], 4)
    features["num_segments"]   = len(intervals)

    return features


# Normalize audio (returns normalized y array)

def normalize_audio(y: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to [-1, 1]."""
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y


def segment_audio(y: np.ndarray, sr: int, top_db: int = 25):
    """
    Split audio into non-silent segments.
    Returns list of (start_sec, end_sec, y_segment) tuples.
    """
    intervals = librosa.effects.split(y, top_db=top_db)
    segments = []
    for start, end in intervals:
        seg = y[start:end]
        if len(seg) / sr >= 0.3:  # keep segments >= 0.3s
            segments.append((start / sr, end / sr, seg))
    return segments



# Main pipeline

def process_dataset(data_dir: str, output_csv: str = "features_dataset.csv",
                    max_files: int = 200) -> pd.DataFrame:
    """
    Walk data_dir for .wav files, extract features, and save to CSV.

    Args:
        data_dir:   root directory containing RAVDESS .wav files
        output_csv: path to save the resulting CSV
        max_files:  cap on number of files to process (use None for all)

    Returns:
        pd.DataFrame with one row per recording
    """
    data_dir = Path(data_dir)
    wav_files = [
        f for f in Path(data_dir).rglob("*.wav")
        if "Audio_Speech" in str(f)
    ]
    if max_files:
        wav_files = wav_files[:max_files]

    print(f"\n[TableTalk] Found {len(wav_files)} .wav files in '{data_dir}'")
    print(f"[TableTalk] Extracting features...\n")

    records = []
    for fpath in tqdm(wav_files, desc="Processing"):
        meta = parse_ravdess_filename(str(fpath))
        feats = extract_features(str(fpath))

        if not feats:
            continue

        row = {
            "filepath": str(fpath),
            "filename": fpath.name,
            **meta,
            **feats,
        }
        records.append(row)

    df = pd.DataFrame(records)

    # Save
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[TableTalk] Dataset saved → {out_path}  ({len(df)} rows, {len(df.columns)} columns)")

    # Quick summary
    if "emotion_label" in df.columns:
        print("\nEmotion distribution:")
        print(df["emotion_label"].value_counts().to_string())

    return df



# Demo / entry point


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TableTalk Audio Processing Pipeline")
    parser.add_argument("--data_dir",    type=str, default="./data/ravdess",
                        help="Root directory with RAVDESS .wav files")
    parser.add_argument("--output_csv",  type=str, default="./outputs/features_dataset.csv")
    parser.add_argument("--max_files",   type=int, default=200)
    args = parser.parse_args()

    df = process_dataset(args.data_dir, args.output_csv, args.max_files)
    print("\nSample rows:")
    print(df[["filename", "emotion_label", "duration_sec",
              "pitch_mean_hz", "rms_energy_mean", "speech_ratio"]].head(10).to_string(index=False))