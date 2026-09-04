# TruthLens: B.Tech Final-Year Capstone Project Report Template

This document provides the standard 18-chapter outline for the official undergraduate engineering capstone report.

---

## Table of Contents

- **Certificate of Authenticity**
- **Declaration by Candidates**
- **Acknowledgements**
- **Abstract**
- **List of Figures**
- **List of Tables**
- **Abbreviations & Notations**

---

### Chapter 1: Introduction
1.1 Background & Digital Disinformation Landscape  
1.2 Impact on Democratic Institutions & Public Health  
1.3 Motivation: Limitations of Superficial Black-Box AI  
1.4 Project Objectives & Scope  
1.5 Thesis Organization  

### Chapter 2: Literature Survey
2.1 Traditional Lexical & Stylistic Text Classification  
2.2 Deep Learning & Pre-trained Language Models (BERT, RoBERTa)  
2.3 Automated Fact-Checking & Knowledge Graph Approaches  
2.4 Stance Detection & Natural Language Inference  
2.5 Research Gaps Identified in Existing Literature  

### Chapter 3: Problem Statement & Proposed Architecture
3.1 Formal Problem Definition  
3.2 Research Hypotheses  
3.3 Proposed Evidence-Grounded Hybrid Architecture  
3.4 Functional & Non-Functional Requirements  

### Chapter 4: System Design & Modular Architecture
4.1 Architectural Block Diagram & Sequence Flows  
4.2 Ingestion & Security Protection Layer  
4.3 Claim Extraction & Atomic Decomposition Engine  
4.4 External Retrieval & Provider Abstractions  
4.5 Stance Analysis & Contradiction Detection  
4.6 Temporal Reasoning & Contextual Distortion Module  
4.7 Explainable Hybrid Verdict Scoring Engine  

### Chapter 5: Database Design & Provenance Modeling
5.1 Relational Schema & Entity-Relationship (ER) Diagram  
5.2 Traceability Lifecycle (Verdict to Source Provenance)  
5.3 SQLite Implementation & Scalability to PostgreSQL  

### Chapter 6: Dataset Engineering & Preprocessing
6.1 Academic Datasets (ISOT, LIAR, FakeNewsNet)  
6.2 The TruthLens Benchmark Corpus  
6.3 Text Cleaning, Stopword Handling, & Tokenization  
6.4 Stratified Train-Validation-Test Splitting  

### Chapter 7: Machine Learning Baseline Development
7.1 Feature Engineering: Sublinear TF-IDF n-grams  
7.2 Multinomial Naive Bayes Model  
7.3 Logistic Regression with L2 Regularization  
7.4 Calibrated Linear Support Vector Classifier (LinearSVC)  
7.5 Random Forest Ensemble  
7.6 Model Persistence & Active Registry  

### Chapter 8: Claim Extraction & Normalization
8.1 Sentence Segmentation & Syntactic Boundary Parsing  
8.2 Atomic Proposition Decomposition  
8.3 Categorization: Factual vs. Opinion vs. Satire  
8.4 Named Entity & Temporal Expression Normalization  

### Chapter 9: External Evidence Retrieval & Ranking
9.1 Multi-Query Generation Strategies  
9.2 Google Fact Check Claim Search Integration  
9.3 Web Evidence Scraping with SSRF Guards  
9.4 Source Authority, Freshness, & Relevance Scoring  
9.5 Syndicated Wire Copy Deduplication  

### Chapter 10: Stance Detection & Contradiction Engine
10.1 Natural Language Inference Formulation  
10.2 Lexical & Semantic Negation Recognition  
10.3 Stance Classification: Supports, Contradicts, Neutral, Outdated  
10.4 Conflicting Source Disagreement Handling  

### Chapter 11: Temporal Context & Timeline Reasoning
11.1 Chronological Analysis: Claim vs. Event vs. Publication  
11.2 Superseded Historical Facts vs. Deliberate Misinformation  
11.3 Interactive Timeline Synthesis  

### Chapter 12: Contextual Distortion & Misleading Statistics
12.1 Proportional vs. Absolute Magnitude Distortions  
12.2 Missing Denominator & Baseline Detection  
12.3 Cherry-Picked Time Windows  

### Chapter 13: Hybrid Verdict Engine & Calibration
13.1 Evidence-Weighted Multi-Factor Fusion Mathematics  
13.2 11 Epistemic Verdict Classifications  
13.3 Calibrated Confidence Estimation  
13.4 Epistemic Uncertainty Handling  

### Chapter 14: Explainability & Narrative Citations
14.1 Grounded Citation Synthesis  
14.2 Prompt-Injection Defenses for Untrusted Web Content  
14.3 Human-Auditable Verification Trace  

### Chapter 15: User Interface & Frontend Implementation
15.1 Academic Design System & Dark Aesthetic  
15.2 Interactive Execution Stepper  
15.3 Verification Results Dashboard & Visualizers  
15.4 Demonstration Benchmark Scenarios  

### Chapter 16: Experimental Results & Analysis
16.1 Evaluation Metrics: Accuracy, Precision, Recall, F1, ROC-AUC  
16.2 Baseline Model Comparative Performance  
16.3 Robustness on Adversarial & Sensational Real News  
16.4 Ablation Studies (Classifier vs. Hybrid Engine)  

### Chapter 17: Testing & Security Validation
17.1 Unit Testing Framework  
17.2 Integration & API Validation  
17.3 Adversarial Penetration Testing (SSRF, Malicious Prompts)  

### Chapter 18: Conclusion, Limitations & Future Work
18.1 Summary of Contributions  
18.2 Critical Examination of Limitations  
18.3 Future Enhancements  
18.4 Final Concluding Remarks  

---

### References (IEEE Format)
1. E. Ferrara et al., "The rise of social bots," *Communications of the ACM*, vol. 59, no. 7, pp. 96-104, 2016.
2. W. Y. Wang, "'Liar, liar pants on fire': A new benchmark dataset for fake news detection," in *Proc. 55th Annual Meeting of the Association for Computational Linguistics (ACL)*, 2017, pp. 422-426.
3. H. Ahmed, I. Traore, and S. Saad, "Detecting opinion spams and fake news using text classification," *Security and Privacy*, vol. 1, no. 1, p. e9, 2018.
4. K. Shu, A. Sliva, S. Wang, J. Tang, and H. Liu, "Fake news detection on social media: A data mining perspective," *ACM SIGKDD Explorations Newsletter*, vol. 19, no. 1, pp. 22-36, 2017.
5. J. Thorne et al., "FEVER: A large-scale dataset for fact extraction and verification," in *Proc. 2018 Conference of the North American Chapter of the ACL (NAACL)*, 2018, pp. 809-819.
