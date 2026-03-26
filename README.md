# -Human-AI-Technical-Test-
Machine Learning for Narrative Voice Classification and Retrieval in Interactive Storytelling Systems

# TableTalk — Narrative Voice Processing & Classification

An end-to-end machine learning pipeline for organizing, classifying, transcribing, and retrieving narrative voice recordings for tabletop storytelling applications.

Built for the TableTalk GSoC technical test using the **RAVDESS Emotional Speech Dataset**.

---

## 🚀 Key Highlights

* Full audio ML pipeline: feature extraction → classification → transcription → retrieval
* Best model: **SVM (RBF) with F1 ≈ 0.83**
* Whisper-based transcription with **~99% accuracy (dataset-dependent)**
* Natural-language audio retrieval system
* Statistical analysis of storytelling vs conversational speech

---

## 📁 Project Structure

```
tabletalk/
│
├── README.md
├── requirements.txt
├── technical_report.md
├── run_all.py
│
├── data/
│   └── ravdess/                # Dataset (not included in repo)
│
├── src/
│   ├── audio/
│   │   └── task1_audio_processing.py
│   │
│   ├── models/
│   │   └── task2_classification.py
│   │
│   ├── transcription/
│   │   └── task3_transcription.py
│   │
│   ├── retrieval/
│   │   └── task4_retrieval.py
│   │
│   └── analysis/
│       └── bonus_storytelling_analysis.py
│
├── notebooks/
│   └── pipeline_demo.ipynb
│
└── outputs/
    ├── features_dataset.csv
    ├── best_classifier.pkl
    ├── classification_report.txt
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── transcripts.csv
    ├── transcription_metrics.json
    ├── transcripts/
    ├── retrieval_results.json
    ├── storytelling_analysis.csv
    ├── storytelling_distributions.png
    └── storytelling_effect_sizes.png
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> Whisper requires PyTorch:

```bash
pip install torch torchvision torchaudio
```

---

### 2. Download Dataset

Download RAVDESS:
https://zenodo.org/record/1188976

Extract into:

```
data/ravdess/
```

---

## ▶️ Running the Pipeline

### Full Pipeline

```bash
python run_all.py --data_dir ./data/ravdess
```

### Optional flags

| Flag                   | Description                          |
| ---------------------- | ------------------------------------ |
| `--max_files`          | Limit dataset size                   |
| `--whisper_model`      | tiny / base / small / medium / large |
| `--skip_transcription` | Skip Whisper step                    |

---

## 🧠 Model Performance

Evaluated on 200 samples:

| Model               | Test Accuracy | Test F1   |
| ------------------- | ------------- | --------- |
| Logistic Regression | 0.775         | 0.761     |
| Random Forest       | 0.825         | 0.807     |
| Gradient Boosting   | 0.825         | 0.810     |
| **SVM (RBF)**       | **0.850**     | **0.831** |

👉 **Best Model: SVM (RBF)**

### Observations

* Calm narration is easiest to classify
* Urgency and suspense overlap due to similar acoustic patterns
* Energy (RMS) is the strongest discriminative feature

---

## 🎙️ Transcription Performance

Using Whisper `base`:

| Metric                  | Score  |
| ----------------------- | ------ |
| Exact Match Accuracy    | ~99%   |
| Word Accuracy (1 − WER) | ~99.8% |
| WER                     | ~0.17% |
| CER                     | ~0.09% |

⚠️ Note: High accuracy is due to fixed sentences in RAVDESS.

---

## 🔍 Retrieval System

Supports natural-language queries combining:

* Narrative tone
* Duration
* Energy
* Pitch
* Pacing

### Example Queries

```python
system.search("calm narration longer than 4 seconds")
system.search("urgency with high pitch")
system.search("character dialogue")
```

---

## 📊 Storytelling Analysis (Bonus)

Key findings:

* Storytelling speech shows higher energy variability
* RMS energy is the strongest discriminator (Cohen’s d ≈ 0.97)
* Pitch and duration are less significant than expected

---

## ⚠️ Limitations

* RAVDESS uses fixed sentences → limited generalization
* Small dataset (~200 samples used)
* Emotion → narrative tone mapping is heuristic
* No deep learning embeddings used

---

## 🚀 Future Work

* Use pretrained audio embeddings (wav2vec2, HuBERT)
* Add semantic retrieval (FAISS + embeddings)
* Train on real storytelling datasets
* Deploy as real-time API (FastAPI)

---

## 📦 Key Dependencies

| Library              | Purpose          |
| -------------------- | ---------------- |
| librosa              | Audio processing |
| scikit-learn         | ML models        |
| openai-whisper       | Transcription    |
| jiwer                | WER/CER metrics  |
| pandas / numpy       | Data processing  |
| matplotlib / seaborn | Visualization    |
