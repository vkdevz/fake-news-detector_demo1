# TruthLens: Current System Audit Report

**Date**: 2026-09-04  
**Audit Type**: Full Architectural, Algorithmic, and Codebase Verification  
**Auditor**: Lead System Architect & QA Engineer  

---

## 1. Audit Summary Matrix

| Component / Subsystem | Status | Evidence | Identified Problems & Gaps | Severity | Recommended Action |
|---|:---:|---|---|:---:|---|
| **FastAPI Backend Core** | `WORKING` | `backend/main.py`, `/api/health`, `/api/verify/*` | Deprecation warnings on `testclient` with Starlette 1.6; CORS open to all. | LOW | Update test client imports and refine CORS configuration. |
| **SQLite Database & Models** | `WORKING` | `backend/app/database/models.py` (10 tables) | Tables exist and store sessions, but ClaimEvidence and SearchQuery tables are not populated during every verification run. | MEDIUM | Ensure all 10 relational entities (claims, evidences, search queries) are written during pipeline execution. |
| **ML Baseline Pipeline** | `WORKING` | `scripts/train_baseline.py`, 4 models saved in `backend/models` | Benchmark corpus was synthetic expansion (288 rows); models get 100% accuracy on small in-domain data, hiding domain shift vulnerabilities. | HIGH | Add out-of-domain evaluation and implement an explicit ModelRegistry with metadata. |
| **Claim Extraction** | `PARTIAL` | `backend/app/claims/extractor.py` | Relies on basic regex and POS heuristics; does not extract numbers/statistics; misses multi-clause splits with conjunctions other than "and <verb>". | HIGH | Enhance extraction of numbers, currency, locations, and multi-part predicates; return normalized claim separately. |
| **Fact-Check Provider** | `PARTIAL` | `backend/app/retrieval/factcheck_provider.py` | Global `all_fact_checks` list in orchestrator was passed to every claim regardless of semantic match, causing false matches across unrelated claims! | CRITICAL | Associate fact-checks strictly with the specific claim that matched them. |
| **Web Evidence Retrieval** | `PARTIAL` | `backend/app/retrieval/web_search_provider.py` | Only uses local demo archive when `SERPAPI_API_KEY` is missing; does not perform real free live web queries for arbitrary user inputs. | HIGH | Add DuckDuckGo / open search fallback so live search actually functions without paid API keys. |
| **Source Evaluation** | `WORKING` | `backend/app/evidence/source_evaluator.py` | Scores authority (.gov, .edu = 0.95+), freshness, and relevance. | LOW | Include independence factor in composite evidence score. |
| **Source Deduplication** | `PARTIAL` | `backend/app/evidence/deduplicator.py` | Flags `is_syndicated`, but the hybrid verdict engine does not discount syndicated duplicate weights, and the UI doesn't display cluster counts. | HIGH | Connect syndication discount to the hybrid decision engine and display cluster counts in UI. |
| **Contradiction / Stance** | `PARTIAL` | `backend/app/analysis/contradiction_detector.py` | Limited hard-coded keyword lists; missed affirmative reporting verbs ("suggested", "attributed"), classifying Case 5 as NEUTRAL instead of SUPPORTS. | HIGH | Expand lexical & semantic entailment verbs and negation polarity logic. |
| **Temporal Reasoner** | `WORKING` | `backend/app/analysis/temporal_reasoner.py` | Successfully detects outdated historical claims (e.g. Queen Elizabeth II). | MEDIUM | Add support for future predictions ("will launch next year" -> FUTURE/NOT_YET_VERIFIABLE). |
| **Context Distortion** | `WORKING` | `backend/app/analysis/context_analyzer.py` | Flags percentage surge without baseline (200% on 1 to 3 incidents). | MEDIUM | Expand to detect selective quotes and unverified causation. |
| **Hybrid Verdict Engine** | `PARTIAL` | `backend/app/verification/hybrid_engine.py` | Hard-coded weight thresholds; did not handle mixed evidence when both support and contradiction were moderate, falling back to UNSUPPORTED. | HIGH | Implement multi-factor fusion with configurable weights, explicit conflicting state, and calibrated confidence. |
| **Explainability Engine** | `WORKING` | `backend/app/explanation/explainer.py` | Synthesizes grounded citations [1], [2] referencing real retrieved excerpts. | MEDIUM | Add claim-level citation mapping and developer audit trace. |
| **Security (SSRF & Injections)**| `WORKING` | `backend/app/core/security.py` | Blocks loopback, private subnets (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 169.254.169.254), filters prompt injection phrases. | LOW | Add IPv6 private checks, redirect validation, and URL length bounds. |
| **React / Vite Frontend** | `WORKING` | `frontend/src/` (Vite 5, React 18, Tailwind) | Clean academic dark theme, but lacks Live vs Offline toggle indicator, Evidence Cluster counter, and Developer Debug view. | MEDIUM | Add Live/Offline toggle, syndicate cluster visualizer, and Developer Debug trace drawer. |
| **Automated Test Suite** | `WORKING` | `backend/tests/` (20 passed in 1.63s) | Good coverage of baseline behaviors; lacks tests for multi-claim isolation, syndication discounting, and future predictions. | MEDIUM | Expand test suite to 30+ tests covering newly hardened edge cases. |

---

## 2. Detailed Bug & Vulnerability Findings

### Bug 1: Multi-Claim Fact-Check Cross-Contamination (Critical)
- **Location**: `backend/app/verification/pipeline_orchestrator.py:196`
- **Root Cause**: `all_fact_checks` aggregated all fact-checks across the entire article. In the claim evaluation loop, `all_fact_checks` was passed into `compute_claim_verdict` for Claim 1. Because Claim 2 had a matching fact-check ("NASA aliens debunked = False"), Claim 1 erroneously inherited this "False" fact check and was declared `FALSE`!
- **Impact**: Multi-claim articles containing one false claim caused completely innocent true claims to be branded `FALSE`.
- **Fix**: Filter `fact_checks` by `claim_id` and ensure similarity is re-evaluated per claim.

### Bug 2: Conflicting Evidence Missed in Case 5 (High)
- **Location**: `backend/app/analysis/contradiction_detector.py`
- **Root Cause**: `SUPPORT_PHRASES` lacked words like "suggested", "preliminary findings", "attributed to". In the Conflicting Evidence demo case ("electrical grid failure caused by cyber intrusion"), the supporting excerpt was classified as `NEUTRAL`, leaving only the contradictory report. The overall verdict fell through to `UNSUPPORTED` instead of `PARTIALLY_TRUE` with conflicting evidence flags.
- **Fix**: Add affirmative assertion and reporting verbs; implement explicit conflicting evidence threshold.

### Bug 3: Web Retrieval Missing Free Live Search (High)
- **Location**: `backend/app/retrieval/web_search_provider.py`
- **Root Cause**: When `SERPAPI_API_KEY` is not provided, the service only queried `self.evidence_archive`. An arbitrary live news claim submitted by a user returned 0 evidence sources.
- **Fix**: Integrate DuckDuckGo / open search scraping so live search operates out of the box without paid credentials, while preserving local archive fallback.

### Bug 4: Syndicated Duplicates Counted as Independent in Decision Formula (Medium)
- **Location**: `backend/app/verification/hybrid_engine.py`
- **Root Cause**: `deduplicate_evidence_items` flagged `is_syndicated`, but `hybrid_engine.py` did not discount the weight of syndicated duplicates in $\sum W_i$, allowing syndicated repetitions to skew the score.
- **Fix**: Multiply duplicate instances by an independence discount factor ($0.40$).

---

## 3. Post-Hardening Verification & Resolution Matrix

All identified high and critical severity findings have been addressed:

| Item | Status | Hardening Resolution | Automated Test / Verification |
|---|:---:|---|---|
| **Fact-Check Isolation** | `RESOLVED` | `pipeline_orchestrator.py` filters `relevant_fcs` by matching `claim_id` | Tested: Multi-claim article isolates ESA Euclid (`UNVERIFIABLE 0.35`) from Mars alien debunk (`FALSE 0.95`). |
| **Conflicting Evidence** | `RESOLVED` | Expanded `SUPPORT_PHRASES` with reporting verbs; implemented conflicting evidence threshold ($w_{sup} \ge 0.20 \land w_{con} \ge 0.20$) | Verified: Case 5 yields `PARTIALLY_TRUE (0.55)` with `has_conflicting_evidence: True`. |
| **Free Live Web Search** | `RESOLVED` | Integrated DuckDuckGo HTML open search provider in `web_search_provider.py` | Verified: Arbitrary unindexed claims retrieve live organic articles with source evaluation and SSRF defenses. |
| **Syndication Weighting** | `RESOLVED` | Implemented cluster IDs and $0.40$ independence discount in `deduplicator.py` and `hybrid_engine.py` | Tested: `test_syndicated_reprint_clustering_and_discount` verifies 3 identical reports get clustered with discounted scores. |
| **Claim Atomicity** | `RESOLVED` | Upgraded `decomposer.py` to split compound coordinate clauses; extracted currency, percentage, and counts in `normalizer.py` | Tested: `test_coordinate_clause_claim_decomposition` and `test_number_and_statistics_extraction`. |
| **Security Hardening** | `RESOLVED` | Added IPv6 loopback (`[::1]`), metadata hostname, port filtering, and max payload size checks in `security.py` | Tested: `test_ssrf_hardened_targets` and `test_payload_size_validation`. |
| **Model Registry** | `RESOLVED` | Fully implemented `ModelMetadata` store with active model switching and `predict_prior()` | Tested: Naive Bayes, Logistic Regression, SVM comparative baseline and transformer comparison. |
| **Ablation Study** | `RESOLVED` | Created `scripts/run_ablation_study.py` evaluating 5 variants (M1 to M5) on hold-out dataset with Brier calibration scores | Verified: Results exported to `data/processed/ablation_study.json` and documented in `docs/EXPERIMENTS.md`. |
| **Frontend Polish** | `RESOLVED` | Added Live Web/Archive indicator in Navbar, Cluster count & syndicate badges in EvidenceSection, and expandable Developer Audit Trace in VerifyPage | Verified: `tsc && vite build` built cleanly; visual UI tested via browser subagent. |
| **Regression Suite** | `RESOLVED` | Expanded pytest suite to 27 unit, integration, adversarial, and hardened feature tests | Verified: 27 passed in 1.85s. |

