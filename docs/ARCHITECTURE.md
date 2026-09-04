# TruthLens System Architecture

## 1. Executive Architecture Overview

**TruthLens** is an explainable, evidence-driven verification system engineered to solve the fundamental limitation of traditional fake news detectors: **reliance on closed-vocabulary stylistic classification without external factual grounding**.

The system implements a multi-stage hybrid pipeline:
1. Ingestion & Preprocessing (Text, URL, Single Claim)
2. Linguistic & Structural Normalization (Language identification: English, Hindi, Hinglish)
3. Atomic Claim Extraction & Decomposition
4. Linguistic Machine Learning Prior Estimation (TF-IDF + Calibrated Linear SVM / Naive Bayes)
5. Multi-Strategy Query Generation
6. Dual-Mode Retrieval (Google Fact Check Tools API + Web Evidence Retrieval + Curated Ground Truth Benchmark)
7. Source Evaluation, Credibility Scoring & Syndication Deduplication
8. Natural Language Inference (NLI) & Contradiction Detection
9. Temporal Consistency & Chronological Reasoning
10. Context Distortion & Statistical Cherry-Picking Analysis
11. Hybrid Verdict Engine (11 Granular Epistemic Verdicts)
12. Grounded Explanation Generation & Traceable Citations

---

## 2. Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Frontend as React / Vite Frontend
    participant API as FastAPI Gateway
    participant Ingestion as Ingestion & Safety
    participant Claims as Claim Extraction & Decomposition
    participant ML as Classical ML Classifier
    participant Retrieval as Evidence Retrieval Engine
    participant Analysis as Stance & Temporal Reasoner
    participant Hybrid as Hybrid Verdict Engine
    participant DB as SQLite Audit Store

    User->>Frontend: Submit Article / URL / Claim
    Frontend->>API: POST /api/verify/{mode}
    API->>Ingestion: Ingest, Sanitize, Check SSRF
    Ingestion->>Claims: Extract Sentences & Decompose Atomic Claims
    Claims->>ML: Compute Stylistic Prior P(Fake|Text)
    Claims->>Retrieval: Multi-Query Search (Exact, Debunk, Entities)
    Retrieval->>Retrieval: Deduplicate Syndicated Wire Stories
    Retrieval->>Analysis: Cross-Examine Evidence Against Claim
    Analysis->>Analysis: Detect Negation & Classify Stance (SUPPORTS/CONTRADICTS)
    Analysis->>Analysis: Evaluate Chronology (CURRENT/HISTORICAL/OUTDATED)
    Analysis->>Hybrid: Aggregate Stance Weights + Prior + Temporal Validity
    Hybrid->>API: 11-Category Calibrated Verdict & Explanation
    API->>DB: Persist Request, Claims, Evidences, Audit Verdict
    API->>Frontend: Full Structured Inspection Response
    Frontend->>User: Interactive Inspection Dashboard
```

---

## 3. Component Details & Design Contracts

### 3.1 Ingestion & Security Module
- **SSRF Prevention**: Strict RFC1918 private subnet checks, IPv6 loopbacks, and cloud metadata (`169.254.169.254`) filtering before executing external GET requests.
- **Prompt-Injection Defense**: Pre-filters untrusted scraped web content using regex heuristics to defuse instructions aimed at altering fact-checking evaluations.

### 3.2 Claim Extraction & Decomposition Engine
- Parses complex syntax into atomic propositions using dependency and coordinating conjunction boundaries.
- Segregates opinions, rhetorical inquiries, and satire from empirical claims.

### 3.3 Hybrid Verdict Fusion Matrix
Instead of a naive binary output ($y \in \{0, 1\}$), the engine synthesizes evidence along three orthogonal axes:
- **Corroboration Score**: $S = \sum_{i} W_i \cdot \mathbb{I}(\text{stance}_i = \text{SUPPORTS})$
- **Contradiction Score**: $C = \sum_{j} W_j \cdot \mathbb{I}(\text{stance}_j = \text{CONTRADICTS})$
- **Temporal Freshness**: Identifies whether the claim describes a past reality that is now outdated.

Output Verdicts:
`SUPPORTED`, `LIKELY_TRUE`, `PARTIALLY_TRUE`, `MISLEADING`, `UNSUPPORTED`, `LIKELY_FALSE`, `FALSE`, `OUTDATED`, `UNVERIFIABLE`, `SATIRE`, `OPINION`.
