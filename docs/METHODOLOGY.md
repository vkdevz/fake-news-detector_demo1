# TruthLens Academic Methodology

## 1. Problem Formulation & Theoretical Motivation

Automated misinformation detection has traditionally been cast as a binary supervised classification problem:

$$\hat{y} = f_\theta(\mathbf{x}) \in \{0, 1\}$$

where $\mathbf{x}$ represents the surface text tokens of an article, and $\hat{y}$ is a binary label (Real or Fake).

This formulation suffers from three major flaws:
1. **Stylistic Decoupling**: A factual report written with sensational phrasing is misclassified as fake, while a completely fabricated rumor written in sober journalistic prose is misclassified as genuine.
2. **Epistemic Incompleteness**: Binary Real/Fake ignores essential nuances: partially true claims, outdated truths, satirical parody, and subjective opinions.
3. **Black-Box Opacity**: Classifiers output confidence scores without providing external, falsifiable evidence that can be audited by humans.

TruthLens addresses this by reformulating verification as an **evidence-grounded entailment problem**:

$$\text{Verdict} = \mathcal{F}\left( P_{\text{ML}}(\mathbf{x}), \{ (c_k, \mathcal{E}_k, \mathcal{S}_k, \mathcal{T}_k) \}_{k=1}^K \right)$$

where:
- $c_k$: Atomic claim decomposed from article $\mathbf{x}$.
- $\mathcal{E}_k$: Retrieved external evidence corpus for claim $c_k$.
- $\mathcal{S}_k$: Stance and NLI relationships between claim $c_k$ and retrieved evidence $\mathcal{E}_k$.
- $\mathcal{T}_k$: Temporal alignment and chronological validity score.

---

## 2. Mathematical Scoring Functions

### 2.1 Evidence Relevance, Authority & Independence Weighting
For each retrieved evidence document $e_i \in \mathcal{E}_k$:

$$W_i = \text{Relevance}(c_k, e_i) \times \text{Authority}(e_i) \times \text{Freshness}(e_i) \times \text{Independence}(e_i)$$

- **Relevance** ($[0.3, 0.98]$): Semantic keyword and lexical overlap between atomic proposition predicate and document excerpt.
- **Authority** ($[0.1, 0.98]$): Domain classification hierarchy:
  - Official government/institutional registries (`.gov`, `.mil`, `nasa.gov`, `who.int`, `isro.gov.in`): $0.95 - 0.98$
  - Accredited fact-check organizations (Snopes, Reuters, PolitiFact, BOOM Live): $0.90 - 0.94$
  - Mainstream peer-reviewed/established news agencies (BBC, AP, Reuters): $0.85 - 0.92$
  - General blogs / unverified domains: $0.35 - 0.50$
  - Satirical domains (The Onion, Babylon Bee): $0.15$
- **Freshness** ($[0.4, 1.0]$): Temporal decay function based on elapsed years from publication.
- **Independence** ($[0.4, 1.0]$): Source independence discount factor. Primary unique reports receive $1.0$, while syndicated reprints sharing identical title/content hashes receive a $0.40$ discount to prevent wire repetitions from artificially inflating confidence.

### 2.2 Aggregate Support & Contradiction Weighting
For claim $c_k$:

$$S_k = \sum_{e_i \in \mathcal{E}_k, \text{stance}_i = \text{SUPPORTS}} W_i \cdot \text{conf}_i$$

$$C_k = \sum_{e_j \in \mathcal{E}_k, \text{stance}_j = \text{CONTRADICTS}} W_j \cdot \text{conf}_j$$

### 2.3 Evidence-Weighted Hybrid Decision Engine
The overall verdict synthesizes the linguistic prior $P_{\text{ML}}$ with the evidence density and stance distribution:

$$\text{Final Score} = (1 - \alpha) P_{\text{ML}} + \alpha \left( \frac{C_k}{S_k + C_k} \right)$$

where $\alpha \in [0.65, 0.85]$ when authoritative external evidence is retrieved, ensuring factual external documentation predominates over surface text style. When evidence is contested ($S_k \ge 0.20 \land C_k \ge 0.20$), the system halts binary output and explicitly flags `has_conflicting_evidence: True` with `PARTIALLY_TRUE` or `UNVERIFIABLE`.

---

## 3. Decision Matrix for 11 Epistemic Verdicts

| Priority | Condition | Assigned Verdict |
|---|---|---|
| 1 | $\text{claim\_type} = \text{satire}$ | `SATIRE` |
| 2 | $\text{claim\_type} = \text{opinion}$ | `OPINION` |
| 3 | $\mathcal{T}_k = \text{OUTDATED}$ | `OUTDATED` |
| 4 | Fact-check match rating = "False" / "Pants on Fire" | `FALSE` |
| 5 | Fact-check match rating = "Correct" / "True" | `SUPPORTED` |
| 6 | Missing statistical baseline detected | `MISLEADING` |
| 7 | $C_k \ge 0.60$ and $C_k / (S_k + C_k) \ge 0.70$ | `FALSE` / `LIKELY_FALSE` |
| 8 | $S_k \ge 0.60$ and $S_k / (S_k + C_k) \ge 0.70$ | `SUPPORTED` / `LIKELY_TRUE` |
| 9 | $S_k \ge 0.40$ and $C_k \ge 0.40$ | `PARTIALLY_TRUE` |
| 10 | $\sum W_i < 0.40$ (No credible evidence) | `UNVERIFIABLE` |
| 11 | Fallback with linguistic prior | `UNSUPPORTED` / Modulated by $P_{\text{ML}}$ |
