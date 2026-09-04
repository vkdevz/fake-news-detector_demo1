# TruthLens: Initial Baseline Evaluation & Performance Measurements

**Date of Measurement**: 2026-09-04 16:36:57  
**Commit / State**: Initial Working System Baseline  
**Environment**: Localhost (macOS, Python 3.12.6, FastAPI, SQLite, React 18, Vite 5)  

---

## 1. Demo Cases Empirical Baseline

Measured by executing real API calls (`POST /api/verify/{mode}`) against `http://127.0.0.1:8000`:

| Case ID | Scenario Title | Input Mode | Expected Verdict | Actual Initial Verdict | Initial Confidence | Latency (ms) | Claims | Evidence Count | Match Status |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `sample-clearly-true` | Clearly True Event (ISRO Aditya-L1) | `claim` | `SUPPORTED` | **SUPPORTED** | 0.98 (HIGH) | 20 ms | 1 | 2 | **PASS** |
| `sample-clearly-false` | Clearly False Viral Hoax (NASA Mars Aliens) | `claim` | `FALSE` | **FALSE** | 0.98 (HIGH) | 5 ms | 1 | 2 | **PASS** |
| `sample-misleading-stat` | Misleading Statistical Surge (200% Crime) | `claim` | `MISLEADING` | **MISLEADING** | 0.88 (HIGH) | 5 ms | 1 | 1 | **PASS** |
| `sample-outdated-fact` | Outdated Historical Fact (Queen Elizabeth II) | `claim` | `OUTDATED` | **OUTDATED** | 0.92 (HIGH) | 5 ms | 1 | 1 | **PASS** |
| `sample-conflicting-sources` | Conflicting Evidence Dispute (Grid Failure) | `claim` | `PARTIALLY_TRUE` | **UNSUPPORTED** | 0.50 (LOW) | 6 ms | 1 | 2 | **FAIL (GAP IDENTIFIED)** |
| `sample-unverifiable-rumor` | Unverifiable Speculation (Private Dinner) | `claim` | `UNVERIFIABLE` | **UNVERIFIABLE** | 0.35 (LOW) | 4 ms | 1 | 0 | **PASS** |

---

## 2. Multi-Claim Article Baseline

Measured on compound text:
```text
"The European Space Agency confirmed that the Euclid telescope captured new deep-field cosmic images. 
However, viral reports claim that NASA discovered alien cities on Mars.
Furthermore, municipal crime rates in the test city rose by 200 percent according to official statistics."
```

**Results**:
- Claims Extracted: 3
- Claim 1 (ESA Euclid True Event): Actual Verdict = **FALSE** (0.95 HIGH) — **FAILURE (Cross-Contamination Bug)**
- Claim 2 (NASA Mars Aliens Hoax): Actual Verdict = **FALSE** (0.95 HIGH) — **CORRECT**
- Claim 3 (200% Crime Statistic): Actual Verdict = **FALSE** (0.95 HIGH) — **FAILURE (Cross-Contamination Bug)**
- Overall Verdict: **FALSE** (0.95)

**Baseline Finding**: In multi-claim mode, Claim 2's fact check was broadcast to all claims, erroneously corrupting the verdicts of Claim 1 and Claim 3.

---

## 3. Initial Test Suite Baseline

- Command: `pytest -v`
- Total Tests: 20 passed
- Execution Time: 1.63 seconds
- Warnings: 4 (FastAPI testclient StarletteDeprecationWarning, anyio blocking portal warning, Pydantic ConfigDict warning)
