# TruthLens: Academic Limitations & Future Work

An honest appraisal of limitations is essential for an academically rigorous B.Tech final-year project:

## 1. Information Latency During Breaking News
When a catastrophic event or breaking crisis occurs, credible investigative reporting takes time to publish. During the initial hours (the "zero-evidence window"), retrieval queries may yield no verified documentation. TruthLens intentionally flags such claims as `UNVERIFIABLE` rather than making a premature determination.

## 2. Low-Resource & Vernacular Linguistic Variations
While TruthLens supports English, Hindi (Devanagari script), and Hinglish (code-mixed Latin), deep semantic parsing and claim extraction for regional languages (e.g. Tamil, Telugu, Bengali) requires specialized multilingual LLM tokenizers and localized fact-checking registries.

## 3. Paywalled & Gated Reporting
Many credible investigative journalism outlets (e.g., The Wall Street Journal, Financial Times) host content behind subscriber paywalls. Search snippet extractors only capture public abstract text, which can occasionally truncate nuanced evidence.

## 4. Nuanced Satire & Parody Detection
Sarcasm, deadpan satire, and cultural parody without overt linguistic markers (such as "The Onion" or "Parody") remain challenging for automated heuristic extractors without external knowledge graphs of authorial intent.

## 5. Future Work
- Integration of local quantized open-weights LLMs (Llama 3 / Mistral) via Ollama/vLLM for zero-cost offline semantic entailment.
- Optical Character Recognition (OCR) pipeline for social media screenshots and meme verification.
- Graph neural network (GNN) integration to map entity relationship networks across news articles.
