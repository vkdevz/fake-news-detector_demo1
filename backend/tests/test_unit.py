import pytest
from backend.app.ml.preprocessor import clean_news_text, detect_language, extract_sentences
from backend.app.claims.normalizer import extract_entities_and_dates, normalize_claim_text
from backend.app.claims.decomposer import decompose_complex_claim
from backend.app.claims.extractor import classify_claim_type_and_verifiability
from backend.app.evidence.source_evaluator import evaluate_domain_authority, evaluate_freshness, calculate_evidence_relevance
from backend.app.evidence.deduplicator import deduplicate_evidence_items
from backend.app.analysis.contradiction_detector import classify_evidence_relationship
from backend.app.analysis.temporal_reasoner import analyze_temporal_context
from backend.app.analysis.context_analyzer import analyze_claim_context
from backend.app.verification.hybrid_engine import compute_claim_verdict
from backend.app.database.schemas import VerdictEnum

def test_text_cleaning():
    raw = "BREAKING: Check this https://example.com/test & contact info@news.org! Shocking report..."
    cleaned = clean_news_text(raw)
    assert "https" not in cleaned
    assert "info@news.org" not in cleaned
    assert "Shocking report" in cleaned

def test_language_detection():
    assert detect_language("This is a verified news story from London.") == "en"
    assert detect_language("यह एक सत्यापित समाचार है।") == "hi"
    assert detect_language("Yeh news bilkul sach hai aur confirmed bhi hai.") == "hi-Latn"

def test_claim_decomposition():
    compound = "NASA discovered aliens on Mars and confirmed extraterrestrial life."
    decomposed = decompose_complex_claim(compound)
    assert len(decomposed) == 2
    assert "NASA discovered aliens on Mars" in decomposed[0]
    assert "NASA confirmed extraterrestrial life" in decomposed[1]

def test_claim_classification_types():
    satire_type, sat_ver = classify_claim_type_and_verifiability("The Onion reports congress votes to replace currency with cheese.")
    assert satire_type == "satire"
    assert sat_ver == "unverifiable"
    
    op_type, op_ver = classify_claim_type_and_verifiability("I personally believe this is the most wonderful policy in history.")
    assert op_type == "opinion"
    assert op_ver == "unverifiable"
    
    fact_type, fact_ver = classify_claim_type_and_verifiability("ISRO launched the satellite into orbit yesterday.")
    assert fact_type == "factual"
    assert fact_ver == "verifiable"

def test_evidence_scoring():
    gov_score = evaluate_domain_authority("https://www.nasa.gov/press-release")
    blog_score = evaluate_domain_authority("https://random-rumor-blog.com/post")
    assert gov_score >= 0.95
    assert blog_score < 0.70

def test_source_deduplication():
    evs = [
        {"id": "1", "title": "Associated Press: Moon landing anniversary celebrated", "url": "https://siteA.com/ap-news"},
        {"id": "2", "title": "Associated Press: Moon landing anniversary celebrated", "url": "https://siteB.com/ap-news"},
    ]
    deduped = deduplicate_evidence_items(evs)
    assert len(deduped) == 1
    assert deduped[0]["is_syndicated"] is True

def test_contradiction_detection():
    claim = "NASA discovered alien cities on Mars"
    debunk = "NASA scientists confirmed they have found no evidence of alien civilizations or cities on Mars."
    rel, conf, _ = classify_evidence_relationship(claim, debunk)
    assert rel == "CONTRADICTS"
    assert conf >= 0.85

def test_temporal_reasoning():
    claim = {"text": "Queen Elizabeth II reigns as the current monarch of Britain", "dates": []}
    ev = [{"excerpt": "Her Majesty Queen Elizabeth II passed away peacefully on 8 September 2022 and was succeeded by King Charles III.", "publisher": "The Royal Household"}]
    res = analyze_temporal_context(claim, ev)
    assert res["temporal_status"] == "OUTDATED"
    assert res["is_outdated"] is True

def test_statistical_context_analysis():
    claim = "Municipal crime increased 200 percent"
    ev = [{"excerpt": "Incidents increased from 1 to 3 cases, which remains at historic lows against the overall baseline."}]
    res = analyze_claim_context(claim, ev)
    assert res["is_misleading"] is True
    assert "MISSING_STATISTICAL_BASELINE" in res["context_flags"]

def test_hybrid_verdict_unverifiable():
    claim = {"text": "Secret aliens live in undisclosed underground bases", "claim_type": "factual", "verifiability": "verifiable"}
    res = compute_claim_verdict(
        claim=claim,
        evidences=[],
        fact_checks=[],
        temporal_data={},
        context_data={},
        ml_prior={"fake_probability": 0.50, "confidence": 0.50}
    )
    assert res["verdict"] == VerdictEnum.UNVERIFIABLE
    assert res["confidence_label"] == "LOW"
