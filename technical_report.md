# TableTalk Human-AI Technical Test

#Machine Learning for Narrative Voice Classification and Retrieval in Interactive Storytelling Systems

**Dataset:** RAVDESS Emotional Speech Dataset
**Author:** KAZI EAJAJ HOSSAIN

---

# 1. Overview

This project presents an end-to-end machine learning system for **processing, classifying, transcribing, and retrieving narrative voice recordings** for interactive storytelling applications.

Unlike a standard ML pipeline, this system introduces an **adaptive retrieval engine** that:

* Parses natural-language queries
* Converts them into structured constraints
* Applies filtering and ranking
* Uses **multi-stage fallback to ensure meaningful results**

---

# 2. Dataset

**RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**

* 24 professional actors
* 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
* Speech duration: ~3–5 seconds
* Sample rate: 22,050 Hz

### Subset Used

* **Speech-only subset (Audio_Speech_Actors_01-24)**
* **200 samples used in experiments** 

---

## Emotion → Narrative Tone Mapping

| Emotion        | Narrative Tone     |
| -------------- | ------------------ |
| neutral, calm  | calm_description   |
| happy          | character_dialogue |
| sad, surprised | dramatic_emphasis  |
| angry, disgust | urgency            |
| fearful        | suspense           |

---

# 3. Task 1 — Audio Processing Pipeline

Each audio file is:

* Loaded at 22,050 Hz
* Normalized
* Processed into structured features

### Extracted Features (~70 per sample)

* MFCC (13 + delta features)
* Pitch (mean, std, range)
* RMS Energy (mean, std, max)
* Spectral features (centroid, bandwidth, rolloff)
* Zero Crossing Rate
* Silence / speech ratio
* Tempo

### Output

* **200 samples × 70 features dataset** 

---

# 4. Task 2 — Narrative Tone Classification

### Experimental Setup

* 80/20 train-test split
* 5 narrative tone classes

### Results

| Model               | Test Accuracy | Test F1    |
| ------------------- | ------------- | ---------- |
| Logistic Regression | 0.8000        | 0.7874     |
| Random Forest       | **0.8750**    | **0.8838** |
| Gradient Boosting   | 0.8500        | 0.8494     |
| SVM (RBF)           | 0.8500        | 0.8371     |

👉 **Best Model: Random Forest (F1 = 0.8838)** 

---

## Key Insights

* `urgency` and `suspense` are highly separable
* `dramatic_emphasis` shows moderate confusion
* **RMS energy is the strongest discriminative feature**

---

# 5. Task 3 — AI-Based Transcription

### Approach

* OpenAI Whisper (`base`) used for speech-to-text

### Results (200 samples)

| Metric                | Score  |
| --------------------- | ------ |
| Exact Match Accuracy  | 97.50% |
| Word Accuracy (1-WER) | 99.50% |
| Word Error Rate (WER) | 0.50%  |
| Char Error Rate (CER) | 0.41%  |



### Observation

High accuracy is due to **fixed sentence structure** in RAVDESS:

* “Kids are talking by the door”
* “Dogs are sitting by the door”

---

# 6. Task 4 — Adaptive Retrieval System

## System Design

The retrieval system converts natural-language queries into structured filters:

| Feature  | Example                 |
| -------- | ----------------------- |
| Tone     | "calm narration"        |
| Duration | "longer than 4 seconds" |
| Energy   | "high-energy speech"    |
| Pitch    | "high pitch"            |
| Pacing   | "slow paced"            |

---

## Retrieval Pipeline

1. Query parsing
2. Constraint extraction
3. Filtering
4. Ranking
5. **Fallback (adaptive relaxation)**

---

## Multi-Stage Fallback Strategy

When strict constraints fail:

1. Relax energy constraints
2. Relax duration constraints
3. Relax tone constraints

👉 Ensures:

* No empty results
* Closest possible matches
* Robust user experience

---

## Example Outputs

### Query: "calm narration longer than 4 seconds"

* Returns recordings with duration ~4.0–4.14 seconds
* Correct tone: `calm_description`

---

### Query: "high-energy speech"

* Returns `urgency` samples
* RMS energy ~0.05–0.09

---

### Query: "suspense shorter than 3 seconds"

* No strict matches
* Fallback triggered
* Returns closest results (~3.6–3.8 seconds)

👉 Demonstrates **adaptive constraint handling**

---

# 7. Bonus — Storytelling Analysis

### Method

* Split into storytelling vs conversational speech
* Welch’s t-test + Cohen’s d

### Key Findings

| Feature         | Effect Size (d) |
| --------------- | --------------- |
| Duration        | ~0.87           |
| RMS Energy Std  | ~0.82           |
| RMS Energy Mean | ~0.81           |



---

## Interpretation

* Storytelling speech has **higher energy variability**
* Energy is more important than pitch
* Duration also plays a significant role

---

# 8. System Pipeline Summary

Audio
→ Feature Extraction
→ Classification
→ Transcription
→ **Adaptive Retrieval**

---

# 9. Limitations

* Fixed sentence dataset limits generalization
* Small dataset (200 samples)
* Heuristic mapping (emotion → tone)
* No deep learning embeddings

---

# 10. Future Work

* Use pretrained embeddings (wav2vec2, HuBERT)
* Add semantic retrieval (FAISS)
* Train on real-world storytelling datasets
* Deploy API (FastAPI)

---

# 11. Conclusion

This project demonstrates:

* Strong classification performance (**F1 = 0.8838**)
* High transcription accuracy (dataset-dependent)
* A **robust adaptive retrieval system**

The key contribution is not just modeling, but **system design for real-world usability**, particularly:

👉 handling imperfect queries through adaptive constraint relaxation
👉 ensuring meaningful results under strict conditions

---
