"""
TableTalk - run_all.py
End-to-end pipeline runner. Executes Tasks 1 → 2 → 3 → 4 → Bonus in sequence.

Usage:
    python run_all.py --data_dir ./data/ravdess

With a custom Whisper model size:
    python run_all.py --data_dir ./data/ravdess --whisper_model small
"""

import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="TableTalk Full Pipeline")
    parser.add_argument("--data_dir",      type=str, required=True,
                        help="Root directory of RAVDESS .wav files")
    parser.add_argument("--output_dir",    type=str, default="./outputs")
    parser.add_argument("--max_files",     type=int, default=200,
                        help="Max audio files to process per task")
    parser.add_argument("--whisper_model", type=str, default="base",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--skip_transcription", action="store_true",
                        help="Skip Whisper transcription (faster demo run)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    # TASK 1: Audio Processing
    
    print("\n" + "═"*60)
    print("  TASK 1 — Audio Processing Pipeline")
    print("═"*60)
    t0 = time.time()
    from src.audio.task1_audio_processing import process_dataset
    features_csv = str(out / "features_dataset.csv")
    df = process_dataset(args.data_dir, features_csv, args.max_files)
    print(f"  ✓ Completed in {time.time()-t0:.1f}s  →  {features_csv}")

    # TASK 2: Tone Classification
    
    print("\n" + "═"*60)
    print("  TASK 2 — Narrative Tone Classification")
    print("═"*60)
    t0 = time.time()
    from src.models.task2_classification import load_data, evaluate_models
    df2 = load_data(features_csv)
    evaluate_models(df2, args.output_dir)
    print(f"  ✓ Completed in {time.time()-t0:.1f}s")

    # TASK 3: Transcription

    transcripts_csv = str(out / "transcripts.csv")
    if not args.skip_transcription:
        print("\n" + "═"*60)
        print("  TASK 3 — AI-Based Transcription  (Whisper)")
        print("═"*60)
        t0 = time.time()
        from src.transcription.task3_transcription import TableTalkTranscriber
        transcriber = TableTalkTranscriber(model_size=args.whisper_model)
        transcriber.transcribe_dataset(args.data_dir, args.output_dir, args.max_files)
        print(f"  ✓ Completed in {time.time()-t0:.1f}s  →  {transcripts_csv}")
    else:
        print("\n[run_all] Skipping Task 3 (--skip_transcription flag set)")

 
    # TASK 4: Retrieval System

    print("\n" + "═"*60)
    print("  TASK 4 — Narrative Audio Retrieval System")
    print("═"*60)
    t0 = time.time()
    from src.retrieval.task4_retrieval import NarrativeRetrievalSystem
    system = NarrativeRetrievalSystem()
    system.build_index_from_csv(features_csv,
                                 transcripts_csv if not args.skip_transcription else None)
    system.run_demo_queries(args.output_dir)
    print(f"  ✓ Completed in {time.time()-t0:.1f}s")

   
    # BONUS: Storytelling Analysis

    print("\n" + "═"*60)
    print("  BONUS — Storytelling Audio Analysis")
    print("═"*60)
    t0 = time.time()
    from src.analysis.bonus_storytelling_analysis import analyze_storytelling_features
    import pandas as pd
    df_full = pd.read_csv(features_csv)
    analyze_storytelling_features(df_full, args.output_dir)
    print(f"  ✓ Completed in {time.time()-t0:.1f}s")

    print("\n" + "═"*60)
    print(f"  ALL TASKS COMPLETE  —  Total: {time.time()-t_total:.1f}s")
    print(f"  Outputs → {out.resolve()}")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()