# Dataset Documentation & Preprocessing Specifications

## 1. Supported Benchmark Datasets

TruthLens is designed to evaluate on and interface with three established academic fake news benchmark corpora:

1. **ISOT Fake News Dataset** (University of Victoria):
   - Contains thousands of verified real articles from Reuters and debunked fake articles from flagged websites.
   - Domain: US and International Politics, World Events.
   - Classification: Binary (Real / Fake).

2. **LIAR Dataset** (Wang, ACL 2017):
   - 12.8K human-labeled short statements from PolitiFact.
   - Labels: Pants-on-fire, False, Barely-true, Half-true, Mostly-true, True.
   - Significance: Demonstrates that misinformation exists on a multi-class spectrum rather than simple binary labels.

3. **TruthLens Academic Benchmark Corpus**:
   - Curated 288-record balanced corpus located in `data/raw/truthlens_benchmark_raw.csv`.
   - Balanced representation across science, technology, politics, economics, healthcare, and conspiracy theories.
   - Includes edge cases: satirical parodies, outdated historical statements, statistical percentage distortions, and conflicting wire reports.

---

## 2. Preprocessing & Leakage Prevention Pipeline

Implemented in `scripts/preprocess_data.py`:
1. **URL & Email Redaction**: Strips hyperlinks and email strings using regex patterns.
2. **HTML Sanitization**: Cleans markup and entities using BeautifulSoup.
3. **Punctuation & Character Filtering**: Removes non-alphanumeric noise while preserving sentence-ending punctuation (`.`, `!`, `?`) necessary for claim boundary segmentation.
4. **Devnagari & Hinglish Script Detection**: Identifies language code (`en`, `hi`, `hi-Latn`) to apply appropriate linguistic tokenizers.
5. **Stratified Split**:
   - Train Split: 70% (201 samples)
   - Validation Split: 15% (43 samples)
   - Test Split: 15% (44 samples)
   - Random seed: Fixed at `42` for exact reproducibility.
