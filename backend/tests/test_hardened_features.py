import pytest
from backend.app.claims.decomposer import decompose_complex_claim
from backend.app.claims.normalizer import extract_numbers_and_statistics
from backend.app.claims.extractor import classify_claim_type_and_verifiability
from backend.app.evidence.deduplicator import deduplicate_evidence_items
from backend.app.verification.hybrid_engine import compute_claim_verdict
from backend.app.core.security import is_safe_url, sanitize_untrusted_text, validate_payload_size

def test_coordinate_clause_claim_decomposition():
    compound = "Company X launched Product Y in Delhi and sales increased by 200%."
    atomic_claims = decompose_complex_claim(compound)
    assert len(atomic_claims) >= 2
    assert any("product y" in c.lower() for c in atomic_claims)
    assert any("200%" in c or "sales increased" in c.lower() for c in atomic_claims)

def test_number_and_statistics_extraction():
    text = "The government allocated $500 million for solar projects, achieving 35.5% efficiency and 150,000 installations."
    numbers = extract_numbers_and_statistics(text)
    assert any("500 million" in n or "$500 million" in n for n in numbers)
    assert any("35.5%" in n for n in numbers)
    assert any("150,000" in n for n in numbers)

def test_future_prediction_classification():
    prediction = "Electric vehicle adoption will reach 80% by 2045."
    claim_type, verifiability = classify_claim_type_and_verifiability(prediction)
    assert claim_type == "prediction"
    assert verifiability == "unverifiable"

def test_syndicated_reprint_clustering_and_discount():
    reprints = [
        {"id": "ev-1", "title": "ISRO launches meteorological satellite into geostationary orbit", "url": "https://thehindu.com/news/isro-satellite"},
        {"id": "ev-2", "title": "ISRO launches meteorological satellite into geostationary orbit", "url": "https://ndtv.com/india-news/isro-satellite"},
        {"id": "ev-3", "title": "ISRO launches meteorological satellite into geostationary orbit", "url": "https://timesofindia.com/city/isro-satellite"},
    ]
    # Test primary collapse
    primaries = deduplicate_evidence_items(reprints, keep_syndicates=False)
    assert len(primaries) == 1
    assert primaries[0]["is_syndicated"] is True
    assert primaries[0]["syndicate_count"] == 3
    assert len(primaries[0]["syndicated_domains"]) == 3

    # Test syndicated independence scoring
    all_clustered = deduplicate_evidence_items(reprints, keep_syndicates=True)
    assert len(all_clustered) == 3
    assert all_clustered[0]["independence_score"] == 1.0
    assert all_clustered[1]["independence_score"] == 0.40
    assert all_clustered[2]["independence_score"] == 0.40

def test_conflicting_evidence_hybrid_verdict():
    # Source A says yes, Source B says no
    evidences = [
        {
            "id": "ev-1", "title": "Officials confirm preliminary event report", "url": "https://reuters.com/a",
            "relationship": "SUPPORTS", "authority_score": 0.90, "freshness_score": 0.90,
            "relevance_score": 0.85, "stance_confidence": 0.85, "temporal_status": "CURRENT",
            "independence_score": 1.0
        },
        {
            "id": "ev-2", "title": "Spokesperson denies allegations as unverified rumor", "url": "https://bbc.com/b",
            "relationship": "CONTRADICTS", "authority_score": 0.90, "freshness_score": 0.90,
            "relevance_score": 0.85, "stance_confidence": 0.85, "temporal_status": "CURRENT",
            "independence_score": 1.0
        }
    ]
    claim_dict = {
        "text": "Authorities confirmed the disputed event took place.",
        "claim_type": "factual",
        "verifiability": "verifiable"
    }
    result = compute_claim_verdict(
        claim=claim_dict,
        evidences=evidences,
        fact_checks=[],
        temporal_data={"temporal_status": "CURRENT", "is_outdated": False},
        context_data={"is_misleading": False},
        ml_prior={"fake_probability": 0.50}
    )
    assert result["has_conflicting_evidence"] is True
    assert result["verdict"] in ["PARTIALLY_TRUE", "UNVERIFIABLE"]
    assert result["confidence"] <= 0.65  # Confidence is appropriately dampened for conflicting reports

def test_ssrf_hardened_targets():
    # IPv6 loopback
    safe_v6, _ = is_safe_url("http://[::1]/admin")
    assert safe_v6 is False

    # Metadata hostname
    safe_meta, _ = is_safe_url("http://metadata.google.internal/computeMetadata/v1/")
    assert safe_meta is False

    # Disallowed port
    safe_port, _ = is_safe_url("http://google.com:22/ssh")
    assert safe_port is False

def test_payload_size_validation():
    valid, _ = validate_payload_size("Normal news article content", max_chars=1000)
    assert valid is True

    huge_text = "A" * 1050
    invalid, msg = validate_payload_size(huge_text, max_chars=1000)
    assert invalid is False
    assert "exceeds" in msg
