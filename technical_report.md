# TableTalk Human-AI Technical Test

### Narrative Voice Processing and Classification

**Dataset:** RAVDESS Emotional Speech Dataset
**Author:** [KAZI EAJAJ HOSSAIN]



## 1. Overview

This report presents an end-to-end machine learning pipeline for processing, classifying, transcribing, and retrieving narrative voice recordings for tabletop storytelling applications.

The system consists of four core components:

1. Audio feature extraction
2. Narrative tone classification
3. Speech-to-text transcription
4. Content-based audio retrieval

Additionally, a statistical analysis is performed to distinguish storytelling narration from conversational speech.

All experiments are conducted on the RAVDESS dataset, which provides controlled emotional speech recordings suitable for modeling narrative tone.



## 2. Dataset

**RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**

* 24 professional actors
* 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
* ~1,440 speech recordings
* Duration: ~3–5 seconds
* Sample rate: 22,050 Hz

### Emotion → Narrative Tone Mapping

| Emotion        | Narrative Tone     |
| -------------- | ------------------ |
| neutral, calm  | calm_description   |
| happy          | character_dialogue |
| sad, surprised | dramatic_emphasis  |
| angry, disgust | urgency            |
| fearful        | suspense           |

This mapping enables supervised learning for narrative tone classification.



## 3. Task 1 — Audio Processing Pipeline

Each audio file is:

* Loaded at 22,050 Hz
* Peak normalized
* Segmented into non-silent regions

### Extracted Features

* **MFCC (13 + delta + delta²)** → spectral representation
* **Pitch (YIN)** → mean, std, range
* **RMS Energy** → loudness dynamics
* **Spectral features** → centroid, bandwidth, rolloff
* **Zero Crossing Rate** → noisiness
* **Pacing features** → silence ratio, tempo
* **Chroma features** → tonal distribution

The final dataset contains ~70–80 features per sample and is stored as a structured CSV.



## 4. Task 2 — Narrative Tone Classification

### Models Evaluated

Using an 80/20 stratified split:

| Model               | Test Accuracy | Test F1   |
| ------------------- | ------------- | --------- |
| Logistic Regression | 0.775         | 0.761     |
| Random Forest       | 0.825         | 0.807     |
| Gradient Boosting   | 0.825         | 0.810     |
| **SVM (RBF)**       | **0.850**     | **0.831** |

### Key Result

👉 **Best Model: SVM (RBF), F1 ≈ 0.83**

### Insights

* `calm_description` is the easiest class (high precision & recall)
* `urgency` and `suspense` show overlap due to similar energy patterns
* **RMS energy and MFCC features are the most discriminative**

### Discussion

The classification task is non-trivial due to overlapping acoustic characteristics between emotional states. Despite this, the model achieves strong performance using classical ML methods.



## 5. Task 3 — AI-Based Transcription

### Approach

OpenAI Whisper (`base`) is used for speech-to-text transcription.

### Results

| Metric                  | Score  |
| ----------------------- | ------ |
| Exact Match Accuracy    | ~99%   |
| Word Accuracy (1 − WER) | ~99.8% |
| WER                     | ~0.17% |
| CER                     | ~0.09% |

### Important Note

The high accuracy is primarily due to the **fixed sentence structure in RAVDESS**:

* “Kids are talking by the door”
* “Dogs are sitting by the door”

👉 This simplifies the ASR task and inflates performance compared to real-world data.



## 6. Task 4 — Narrative Audio Retrieval System

### Design

A rule-based retrieval system converts natural-language queries into structured filters:

| Feature  | Example                 |
| -------- | ----------------------- |
| Tone     | "calm narration"        |
| Duration | "longer than 4 seconds" |
| Energy   | "high-energy speech"    |
| Pitch    | "high pitch"            |
| Pacing   | "slow paced"            |

### Example Results

* "calm narration longer than 4 seconds" → valid matches returned
* "urgency with high pitch" → filtered correctly
* "high-energy speech" → initially returned 0 results (threshold tuning required)

### Insight

The system demonstrates effective multi-constraint filtering but is sensitive to threshold selection (e.g., energy range).



## 7. Bonus — Storytelling Analysis

### Method

* Split into storytelling vs conversational speech
* Applied Welch’s t-test and Cohen’s d

### Key Findings

| Feature          | Effect            |
| ---------------- | ----------------- |
| RMS Energy Mean  | Strong (d ≈ 0.94) |
| RMS Energy Std   | Strong (d ≈ 0.97) |
| Tempo            | Moderate          |
| Pitch / Duration | Weak              |

### Interpretation

* **Energy is the strongest discriminator of storytelling speech**
* Pitch variation is less significant than expected
* Storytelling relies more on intensity than pitch


## 8. System Pipeline Summary


Audio → Feature Extraction → Classification → Transcription → Retrieval


This modular pipeline enables:

* scalable processing
* flexible querying
* integration into real-time systems



## 9. Limitations

* Fixed sentence dataset limits generalization
* Small dataset (~200 samples used)
* Heuristic emotion-to-tone mapping
* No deep learning embeddings



## 10. Future Work

* Pretrained audio embeddings (wav2vec2, HuBERT)
* Semantic retrieval (FAISS + embeddings)
* Real storytelling dataset fine-tuning
* API deployment (FastAPI)



## 11. Conclusion

This project demonstrates a complete ML pipeline for narrative audio understanding, achieving:

* Strong classification performance (F1 ≈ 0.83)
* Near-perfect transcription (dataset-dependent)
* Functional natural-language retrieval system

The system highlights the importance of **feature engineering and system design** in practical ML applications.


