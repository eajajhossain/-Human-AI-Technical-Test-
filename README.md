# # TableTalk Human-AI Technical Test

Machine Learning system for **classification, transcription, and intelligent retrieval of narrative voice recordings** for interactive storytelling applications.

Built for the TableTalk GSoC 2026 Technical Test using the **RAVDESS Emotional Speech Dataset**.

---

# 🚀 What This Project Actually Does

This is **not just a machine learning pipeline**.

👉 It is an **adaptive retrieval system** that:

* Understands natural-language queries
* Converts them into structured constraints
* Applies intelligent filtering + ranking
* Uses **multi-stage fallback to avoid empty results**

---

# 🔥 Key Highlights

* End-to-end pipeline: audio → features → classification → transcription → retrieval
* **Best Model: Random Forest (F1 = 0.8838)** 
* Whisper transcription with **97.5% exact match accuracy** 
* Natural-language query-based audio retrieval
* Adaptive multi-stage constraint relaxation (core innovation)
* Statistical storytelling vs conversational speech analysis

---

# 🧠 System Pipeline

Audio Input
→ Feature Extraction (librosa)
→ Narrative Tone Classification (ML)
→ Transcription (Whisper)
→ **Adaptive Retrieval Engine**

---

# 📊 Dataset & Processing

* Dataset: **RAVDESS (Speech subset only)**
* Total samples used: **200 audio files** 
* Features extracted: **70 acoustic features per file** 

### Emotion Distribution

* Calm: 32
* Happy: 32
* Sad: 24
* Angry: 24
* Fearful: 24
* Disgust: 24
* Surprised: 24
* Neutral: 16 

---

# 🧠 Model Performance

| Model               | Test Accuracy | Test F1    |
| ------------------- | ------------- | ---------- |
| Logistic Regression | 0.8000        | 0.7874     |
| Random Forest       | **0.8750**    | **0.8838** |
| Gradient Boosting   | 0.8500        | 0.8494     |
| SVM (RBF)           | 0.8500        | 0.8371     |

👉 **Selected Model: Random Forest** 

---

# 🎙️ Transcription Performance

Using Whisper (`base`):

| Metric                | Score  |
| --------------------- | ------ |
| Exact Match Accuracy  | 97.50% |
| Word Accuracy (1-WER) | 99.50% |
| Word Error Rate (WER) | 0.50%  |
| Char Error Rate (CER) | 0.41%  |

👉 Evaluated on **200 samples** 

---

# 🔍 Retrieval System (Core Innovation)

The system supports natural-language queries like:

```python
system.search("calm narration longer than 4 seconds")
system.search("high-energy speech")
system.search("suspense shorter than 3 seconds")
system.search("urgency with high pitch")
```

---

## ⚙️ How Retrieval Works

### Step 1: Query Parsing

Extracts:

* Narrative tone
* Energy constraints
* Pitch constraints
* Duration constraints
* Silence / pacing

---

### Step 2: Filtering + Ranking

* Applies structured filters
* Ranks results using energy, pitch, duration, and tone

---

### Step 3: Multi-Stage Fallback (IMPORTANT)

If strict filters fail:

1. Relax energy constraints
2. Relax duration constraints
3. Relax tone constraints

👉 Guarantees:

* No empty results
* Closest possible matches
* Real-world search behavior

---

# 🎯 Example Outputs

## Query: **"high-energy speech"**

| Tone    | Energy | Duration |
| ------- | ------ | -------- |
| urgency | 0.092  | 4.44s    |
| urgency | 0.065  | 4.50s    |

---

## Query: **"calm narration longer than 4 seconds"**

| Tone             | Duration |
| ---------------- | -------- |
| calm_description | 4.14s    |
| calm_description | 4.10s    |

---

## Query: **"suspense shorter than 3 seconds"**

```text
Fallback triggered → duration constraint relaxed
Returns closest matches (~3.6–3.8 seconds)
```

👉 Demonstrates **robust retrieval under constraint failure**

---

# 📊 Storytelling Analysis (Bonus)

Key findings:

* Storytelling speech has **higher energy variability**
* RMS energy is strongest discriminator (**Cohen’s d ≈ 0.81**) 
* Duration also highly significant (**d ≈ 0.87**) 
* Pitch is less significant than expected

---

# 📁 Project Structure

```
tabletalk/
│
├── run_all.py
├── requirements.txt
├── technical_report.md
│
├── data/
│   └── ravdess/
│
├── src/
│   ├── audio/
│   ├── models/
│   ├── transcription/
│   ├── retrieval/
│   └── analysis/
│
├── notebooks/
│   └── pipeline_demo.ipynb
│
└── outputs/
```

---

# ⚙️ Setup

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio
```

---

# 📦 Dataset Setup

Download RAVDESS:
https://zenodo.org/record/1188976

Use ONLY:

```
Audio_Speech_Actors_01-24
```

---

# ▶️ Run Full Pipeline

```bash
python run_all.py --data_dir ./data/ravdess/Audio_Speech_Actors_01-24 --max_files 200
```

---

# ⚠️ Limitations

* Dataset uses fixed sentences → limited real-world generalization
* Small dataset (200 samples)
* Emotion → narrative tone mapping is heuristic
* No deep audio embeddings used

---

# 🚀 Future Improvements

* Use pretrained embeddings (wav2vec2 / HuBERT)
* Add semantic search (FAISS)
* Train on real storytelling datasets
* Deploy as API (FastAPI)

---

# 📌 Final Takeaway

This project demonstrates:

👉 **System-level thinking beyond standard ML pipelines**
👉 **Adaptive retrieval under real-world constraints**
👉 **Robust handling of imperfect queries using fallback strategies**
