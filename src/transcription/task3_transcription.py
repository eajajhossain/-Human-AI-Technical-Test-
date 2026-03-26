"""
TableTalk - Task 3: AI-Based Transcription
Uses OpenAI Whisper to transcribe narrative audio recordings and measures accuracy.
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import jiwer                   # pip install jiwer  — for WER/CER metrics
import whisper

# RAVDESS ground-truth sentences
# RAVDESS uses exactly 2 statements repeated across actors

RAVDESS_STATEMENTS = {
    "01": "kids are talking by the door",
    "02": "dogs are sitting by the door",
}


def get_ground_truth(filepath: str) -> str | None:
    """Return RAVDESS ground-truth transcript from filename."""
    name   = Path(filepath).stem
    parts  = name.split("-")
    if len(parts) != 7:
        return None
    stmt_code = parts[4]
    return RAVDESS_STATEMENTS.get(stmt_code)


# Whisper transcription

class TableTalkTranscriber:
    def __init__(self, model_size: str = "base"):
        """
        Args:
            model_size: whisper model size — tiny | base | small | medium | large
                        'base' gives good accuracy/speed balance for English.
        """
        print(f"[Task3] Loading Whisper '{model_size}' model...")
        self.model      = whisper.load_model(model_size)
        self.model_size = model_size
        print("[Task3] Model loaded.")

    def transcribe_file(self, filepath: str) -> dict:
        """
        Transcribe a single audio file.

        Returns:
            dict with keys: filepath, transcript, language, segments, duration
        """
        result = self.model.transcribe(
            str(filepath),
            language="en",
            task="transcribe",
            fp16=False,         # safer on CPU
            verbose=False,
        )
        return {
            "filepath":   str(filepath),
            "filename":   Path(filepath).name,
            "transcript": result["text"].strip(),
            "language":   result.get("language", "en"),
            "segments":   result.get("segments", []),
        }

    def transcribe_dataset(self, data_dir: str, output_dir: str = "./outputs",
                            max_files: int = 100) -> pd.DataFrame:
        """
        Transcribe all .wav files in data_dir, compute accuracy where
        ground-truth is available, and save results.
        """
        data_dir   = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        wav_files = sorted(data_dir.rglob("*.wav"))[:max_files]
        print(f"\n[Task3] Transcribing {len(wav_files)} files with Whisper ({self.model_size})...\n")

        records = []
        for fpath in tqdm(wav_files, desc="Transcribing"):
            try:
                result = self.transcribe_file(str(fpath))
                gt     = get_ground_truth(str(fpath))

                row = {
                    "filename":         result["filename"],
                    "transcript":       result["transcript"],
                    "ground_truth":     gt or "",
                    "has_ground_truth": gt is not None,
                }
                records.append(row)

                # Save individual transcript
                txt_path = output_dir / "transcripts" / (fpath.stem + ".txt")
                txt_path.parent.mkdir(parents=True, exist_ok=True)
                txt_path.write_text(
                    f"File: {fpath.name}\n"
                    f"Transcript: {result['transcript']}\n"
                    f"Ground truth: {gt or 'N/A'}\n"
                )

            except Exception as e:
                print(f"  [ERROR] {fpath.name}: {e}")

        df = pd.DataFrame(records)

        #  Compute metrics where ground truth is available 
        df_gt = df[df["has_ground_truth"]].copy()
        if len(df_gt) > 0:
            metrics = self._compute_metrics(df_gt)
            self._print_metrics(metrics, len(df_gt))
            self._save_metrics(metrics, output_dir)
        else:
            print("[Task3] No ground-truth data available for accuracy measurement.")
            metrics = {}

        df.to_csv(output_dir / "transcripts.csv", index=False)
        print(f"\n[Task3] Transcripts saved → {output_dir / 'transcripts.csv'}")

        return df, metrics

    # Accuracy metrics
   
    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase and strip punctuation for fair WER comparison."""
        import re
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _compute_metrics(self, df: pd.DataFrame) -> dict:
        hypotheses  = [self._normalize(t) for t in df["transcript"]]
        references  = [self._normalize(t) for t in df["ground_truth"]]

        # Per-sentence exact match
        exact_matches = sum(h == r for h, r in zip(hypotheses, references))
        exact_acc     = exact_matches / len(df)

        # Word Error Rate
        wer = jiwer.wer(references, hypotheses)

        # Character Error Rate
        cer = jiwer.cer(references, hypotheses)

        # Word accuracy
        word_acc = max(0.0, 1.0 - wer)

        return {
            "num_samples":   len(df),
            "exact_match":   round(exact_acc, 4),
            "word_accuracy": round(word_acc, 4),
            "wer":           round(wer, 4),
            "cer":           round(cer, 4),
        }

    @staticmethod
    def _print_metrics(metrics: dict, n: int):
        print(f"\n{'─'*45}")
        print(f"  Transcription Accuracy  ({n} samples)")
        print(f"{'─'*45}")
        print(f"  Exact Match Accuracy : {metrics['exact_match']*100:.2f}%")
        print(f"  Word Accuracy (1-WER): {metrics['word_accuracy']*100:.2f}%")
        print(f"  Word Error Rate (WER): {metrics['wer']*100:.2f}%")
        print(f"  Char Error Rate (CER): {metrics['cer']*100:.2f}%")
        print(f"{'─'*45}\n")

    @staticmethod
    def _save_metrics(metrics: dict, output_dir: Path):
        path = output_dir / "transcription_metrics.json"
        path.write_text(json.dumps(metrics, indent=2))
        print(f"[Task3] Metrics saved → {path}")


# Segment-level transcription with timestamps

def transcribe_with_timestamps(filepath: str, model_size: str = "base") -> list[dict]:
    """
    Return word-level timestamps for a single file.
    Useful for generating captions / accessibility outputs.
    """
    model  = whisper.load_model(model_size)
    result = model.transcribe(
        str(filepath),
        language="en",
        word_timestamps=True,
        fp16=False,
        verbose=False,
    )

    captions = []
    for seg in result.get("segments", []):
        captions.append({
            "start": round(seg["start"], 3),
            "end":   round(seg["end"],   3),
            "text":  seg["text"].strip(),
        })

    return captions


def export_srt(captions: list[dict], out_path: str):
    """Export captions as .srt subtitle file (accessibility)."""
    lines = []
    for i, cap in enumerate(captions, 1):
        start = _sec_to_srt(cap["start"])
        end   = _sec_to_srt(cap["end"])
        lines.append(f"{i}\n{start} --> {end}\n{cap['text']}\n")
    Path(out_path).write_text("\n".join(lines))
    print(f"[Task3] SRT caption file saved → {out_path}")


def _sec_to_srt(seconds: float) -> str:
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


# Entry point

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TableTalk Transcription Pipeline")
    parser.add_argument("--data_dir",    type=str, default="./data/ravdess")
    parser.add_argument("--output_dir",  type=str, default="./outputs")
    parser.add_argument("--model_size",  type=str, default="base",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--max_files",   type=int, default=100)
    args = parser.parse_args()

    transcriber = TableTalkTranscriber(model_size=args.model_size)
    df, metrics = transcriber.transcribe_dataset(
        args.data_dir, args.output_dir, args.max_files
    )

    print("\nSample transcripts:")
    print(df[["filename", "transcript", "ground_truth"]].head(10).to_string(index=False))