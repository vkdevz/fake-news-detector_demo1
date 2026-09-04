# TruthLens: An Explainable AI-Based Fake News Detection and Evidence Verification System

> **B.Tech Final-Year Capstone Project — Computer Science & Engineering**  
> *Evidence before belief.*

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6.svg)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Project Overview

**TruthLens** moves beyond the traditional `News -> ML Model -> Real/Fake` paradigm. Standard text classifiers evaluate only vocabulary patterns, easily mistaking formal fake news for truth and flagged sensational real news as fake.

TruthLens introduces a **hybrid, evidence-grounded verification architecture** combining:
1. **Classical & Modern ML Text Classifiers** (Stylistic/linguistic priors)
2. **Atomic Claim Extraction & Decomposition** (Separating factual assertions from opinions/satire)
3. **Multi-Source Evidence Retrieval** (Fact-check APIs, live web search, official archives)
4. **Source Credibility & Syndication Deduplication** (Preventing syndicate echo-chambers)
5. **NLI Stance & Contradiction Detection** (Supports, Contradicts, Neutral, Outdated)
6. **Temporal Consistency & Chronological Reasoning** (Distinguishing outdated truths from fake news)
7. **Contextual Baseline Distortion Analysis** (Detecting misleading percentages e.g. 1 to 3 cases = 200%)
8. **11-Category Epistemic Verdict Engine** with traceable citations and prompt-injection defenses.

---

## 2. System Architecture

```text
USER INPUT (News Text, URL, Claim)
    ↓
INGESTION & SSRF SECURITY GUARD
    ↓
TEXT CLEANING & LANGUAGE IDENTIFICATION (EN, HI, Hinglish)
    ↓
CLAIM EXTRACTION & ATOMIC DECOMPOSITION
    ↓
LINGUISTIC ML CLASSIFIER (TF-IDF + Calibrated Linear SVM)
    ↓
MULTI-QUERY GENERATION & FACT-CHECK RETRIEVAL
    ↓
WEB EVIDENCE RETRIEVAL & WIRE DEDUPLICATION
    ↓
SOURCE CREDIBILITY & FRESHNESS RANKING
    ↓
NLI CONTRADICTION & STANCE DETECTION
    ↓
TEMPORAL VALIDATION & CHRONOLOGICAL REASONING
    ↓
CONTEXT DISTORTION & MISSING BASELINE ANALYSIS
    ↓
HYBRID VERDICT ENGINE (11 Epistemic Verdicts)
    ↓
EXPLAINABLE VERDICT & CITED AUDIT TRAIL
```

---

## 3. Technology Stack

- **Backend**: Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2.0, HTTPX, BeautifulSoup4
- **Machine Learning & NLP**: Scikit-Learn, NumPy, Pandas, Joblib, Regex / NLTK
- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Lucide Icons
- **Database**: SQLite (10-table relational schema with complete audit trail)
- **External Providers**: Google Fact Check Tools API, DuckDuckGo / SerpAPI scraper, plus curated offline benchmark archives for 100% reliable college viva presentations.

---

## 4. Quickstart Guide

### Prerequisites
- Python 3.12+
- Node.js v18+ and npm

### 4.1 Clone & Setup Environment
```bash
git clone <repository_url>
cd fake-news-detector

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt
```

### 4.2 Train ML Baseline Models (Reproducible)
```bash
python scripts/preprocess_data.py
python scripts/train_baseline.py
python scripts/seed_demo_data.py
```

### 4.3 Run Backend Server
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation will be live at: `http://127.0.0.1:8000/docs`

### 4.4 Run Frontend Application
In a separate terminal window:
```bash
cd frontend
npm install
npm run dev
```
Open `http://127.0.0.1:5173` in your browser.

---

## 5. Running Automated Tests

```bash
# Run all unit, integration, and adversarial tests
pytest -v
```
All 20 test cases will execute and validate SSRF security, text cleaning, claim decomposition, contradiction detection, and hybrid verdicts.

---

## 6. Academic Documentation Index

All formal thesis documentation is located in the `/docs` directory:
- [System Architecture](docs/ARCHITECTURE.md)
- [Methodology & Equations](docs/METHODOLOGY.md)
- [Dataset Specifications](docs/DATASETS.md)
- [REST API Reference](docs/API.md)
- [Experimental Results](docs/EXPERIMENTS.md)
- [Academic Limitations](docs/LIMITATIONS.md)
- [B.Tech Viva Voce Guide (25+ Q&A)](docs/VIVA_NOTES.md)
- [18-Chapter Report Template](docs/PROJECT_REPORT_TEMPLATE.md)

---

## 7. License

Released under the MIT License. Developed for academic evaluation, research comparison, and final-year B.Tech capstone presentation.
