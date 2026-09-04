from typing import Dict, Any, List, Tuple
from backend.app.database.schemas import VerdictEnum

def calculate_confidence_label(confidence: float) -> str:
    if confidence >= 0.80:
        return "HIGH"
    elif confidence >= 0.55:
        return "MEDIUM"
    return "LOW"

def compute_claim_verdict(
    claim: Dict[str, Any],
    evidences: List[Dict[str, Any]],
    fact_checks: List[Dict[str, Any]],
    temporal_data: Dict[str, Any],
    context_data: Dict[str, Any],
    ml_prior: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Deterministically computes an explainable verdict for an individual atomic claim
    by fusing ML linguistic priors with retrieved evidence, source independence, and stance scores.
    """
    claim_type = claim.get("claim_type", "factual")
    verifiability = claim.get("verifiability", "verifiable")
    is_outdated = temporal_data.get("is_outdated", False)
    temporal_status = temporal_data.get("temporal_status", "CURRENT")
    is_misleading = context_data.get("is_misleading", False)
    
    # 1. Non-factual / Non-verifiable bypass
    if claim_type == "satire":
        return {
            "verdict": VerdictEnum.SATIRE,
            "confidence": 0.95,
            "confidence_label": "HIGH",
            "support_score": 0.0,
            "contradiction_score": 0.0,
            "reason": "Content contains prominent markers of satirical parody or humor rather than factual assertion."
        }
        
    if claim_type == "opinion":
        return {
            "verdict": VerdictEnum.OPINION,
            "confidence": 0.90,
            "confidence_label": "HIGH",
            "support_score": 0.0,
            "contradiction_score": 0.0,
            "reason": "Statement expresses a personal value judgment or subjective viewpoint rather than an empirically verifiable fact."
        }

    # 2. Future predictions / unverified projections
    if temporal_status == "FUTURE":
        return {
            "verdict": VerdictEnum.UNVERIFIABLE,
            "confidence": 0.50,
            "confidence_label": "LOW",
            "support_score": 0.10,
            "contradiction_score": 0.10,
            "reason": temporal_data.get("explanation", "Claim references a future event or projection that cannot be empirically verified at present.")
        }

    # 3. Outdated historical assertion
    if is_outdated or temporal_status == "OUTDATED":
        return {
            "verdict": VerdictEnum.OUTDATED,
            "confidence": 0.92,
            "confidence_label": "HIGH",
            "support_score": 0.20,
            "contradiction_score": 0.80,
            "reason": temporal_data.get("explanation", "Claim reflects historical conditions that were once accurate but have been superseded by subsequent events.")
        }

    # 4. Direct Fact-Check match (authoritative signal filtered strictly for THIS claim)
    claim_id = claim.get("claim_id")
    matching_fact_checks = [fc for fc in fact_checks if not fc.get("claim_id") or fc.get("claim_id") == claim_id]
    
    if matching_fact_checks:
        top_fc = matching_fact_checks[0]
        rating = top_fc.get("rating", "").lower()
        sim = top_fc.get("semantic_similarity", 0.0)
        
        if sim >= 0.35:
            if any(term in rating for term in ["false", "pants on fire", "fake", "incorrect"]):
                return {
                    "verdict": VerdictEnum.FALSE,
                    "confidence": min(0.98, max(0.85, sim + 0.15)),
                    "confidence_label": "HIGH",
                    "support_score": 0.05,
                    "contradiction_score": 0.95,
                    "reason": f"Directly debunked by accredited fact-checker {top_fc.get('publisher')} with rating '{top_fc.get('rating')}'."
                }
            elif any(term in rating for term in ["true", "correct", "accurate"]):
                return {
                    "verdict": VerdictEnum.SUPPORTED,
                    "confidence": min(0.98, max(0.85, sim + 0.15)),
                    "confidence_label": "HIGH",
                    "support_score": 0.95,
                    "contradiction_score": 0.05,
                    "reason": f"Directly corroborated by accredited fact-checker {top_fc.get('publisher')} with rating '{top_fc.get('rating')}'."
                }
            elif "misleading" in rating:
                return {
                    "verdict": VerdictEnum.MISLEADING,
                    "confidence": 0.88,
                    "confidence_label": "HIGH",
                    "support_score": 0.40,
                    "contradiction_score": 0.60,
                    "reason": f"Classified as Misleading by {top_fc.get('publisher')}: {top_fc.get('summary')}"
                }

    # 5. Aggregate Weighted Stance Scores from Retrieved Evidence with Independence Discounting
    support_weight = 0.0
    contradict_weight = 0.0
    total_relevance = 0.0
    
    for ev in evidences:
        rel = ev.get("relevance_score", 0.50)
        auth = ev.get("authority_score", 0.50)
        freshness = ev.get("freshness_score", 0.70)
        independence = ev.get("independence_score", 1.0)
        stance = ev.get("relationship", "NEUTRAL")
        conf = ev.get("stance_confidence", 0.50)
        
        # Composite evidence weight
        weight = rel * auth * freshness * independence * conf
        total_relevance += rel
        
        if stance == "SUPPORTS":
            support_weight += weight
        elif stance == "CONTRADICTS":
            contradict_weight += weight
        elif stance == "PARTIALLY_SUPPORTS":
            support_weight += weight * 0.5
            contradict_weight += weight * 0.2

    # 6. Check if evidence is insufficient
    if not evidences or total_relevance < 0.35:
        fake_prob = ml_prior.get("fake_probability", 0.50)
        return {
            "verdict": VerdictEnum.UNVERIFIABLE,
            "confidence": 0.35,
            "confidence_label": "LOW",
            "support_score": 0.10,
            "contradiction_score": 0.10,
            "reason": "Insufficient authoritative external documentation exists to either corroborate or refute this claim."
        }

    # Normalize weights
    total_w = support_weight + contradict_weight
    support_ratio = (support_weight / total_w) if total_w > 0 else 0.0
    contradict_ratio = (contradict_weight / total_w) if total_w > 0 else 0.0
    
    # 7. Misleading statistical context
    if is_misleading:
        return {
            "verdict": VerdictEnum.MISLEADING,
            "confidence": 0.88,
            "confidence_label": "HIGH",
            "support_score": round(support_ratio, 2),
            "contradiction_score": round(contradict_ratio, 2),
            "reason": context_data.get("details", "Statement presents literal figures in a statistically distorted or cherry-picked context.")
        }

    # 8. Conflicting / Disputed Evidence (both sides have meaningful evidence)
    if (support_weight >= 0.20 and contradict_weight >= 0.20) or (support_ratio >= 0.28 and contradict_ratio >= 0.28 and total_w >= 0.35):
        return {
            "verdict": VerdictEnum.PARTIALLY_TRUE,
            "confidence": 0.55,
            "confidence_label": "MEDIUM",
            "support_score": round(support_ratio, 2),
            "contradiction_score": round(contradict_ratio, 2),
            "has_conflicting_evidence": True,
            "reason": "Conflicting evidence detected across independent authorities; core assertion is contested between sources."
        }

    # 9. Strong Contradiction
    if contradict_ratio >= 0.65 and contradict_weight >= 0.30:
        conf = min(0.96, 0.75 + (contradict_weight * 0.15))
        return {
            "verdict": VerdictEnum.FALSE if conf >= 0.88 else VerdictEnum.LIKELY_FALSE,
            "confidence": round(conf, 2),
            "confidence_label": calculate_confidence_label(conf),
            "support_score": round(support_ratio, 2),
            "contradiction_score": round(contradict_ratio, 2),
            "reason": "Authoritative reporting and primary documentation directly contradict the claim's predicate."
        }

    # 10. Strong Support
    if support_ratio >= 0.65 and support_weight >= 0.30:
        conf = min(0.96, 0.75 + (support_weight * 0.15))
        return {
            "verdict": VerdictEnum.SUPPORTED if conf >= 0.88 else VerdictEnum.LIKELY_TRUE,
            "confidence": round(conf, 2),
            "confidence_label": calculate_confidence_label(conf),
            "support_score": round(support_ratio, 2),
            "contradiction_score": round(contradict_ratio, 2),
            "reason": "High-authority primary and secondary sources consistently substantiate the assertion."
        }

    # 11. Fallback with ML Prior Modulation
    fake_prob = ml_prior.get("fake_probability", 0.50)
    if fake_prob >= 0.75:
        return {
            "verdict": VerdictEnum.LIKELY_FALSE,
            "confidence": 0.62,
            "confidence_label": "MEDIUM",
            "support_score": round(support_ratio, 2),
            "contradiction_score": round(contradict_ratio, 2),
            "reason": "Stylistic presentation aligns with deceptive patterns, coupled with absent corroborating evidence."
        }
    elif fake_prob <= 0.25:
        return {
            "verdict": VerdictEnum.LIKELY_TRUE,
            "confidence": 0.62,
            "confidence_label": "MEDIUM",
            "support_score": round(support_ratio, 2),
            "contradiction_score": round(contradict_ratio, 2),
            "reason": "Stylistic presentation aligns with credible reportage, though external evidence density is moderate."
        }
        
    return {
        "verdict": VerdictEnum.UNSUPPORTED,
        "confidence": 0.50,
        "confidence_label": "LOW",
        "support_score": round(support_ratio, 2),
        "contradiction_score": round(contradict_ratio, 2),
        "reason": "The claim is not substantiated by authoritative reporting within the retrieved corpus."
    }

def compute_overall_verdict(
    claims_data: List[Dict[str, Any]],
    ml_prior: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Synthesizes overall article verdict by aggregating atomic claim determinations.
    """
    if not claims_data:
        fake_prob = ml_prior.get("fake_probability", 0.50)
        verdict = VerdictEnum.LIKELY_FALSE if fake_prob > 0.65 else (VerdictEnum.LIKELY_TRUE if fake_prob < 0.35 else VerdictEnum.UNVERIFIABLE)
        return {
            "verdict": verdict,
            "confidence": round(ml_prior.get("confidence", 0.50), 2),
            "confidence_label": calculate_confidence_label(ml_prior.get("confidence", 0.50)),
            "ml_probability": fake_prob,
            "ml_verdict": ml_prior.get("label"),
            "support_score": round(1.0 - fake_prob, 2),
            "contradiction_score": round(fake_prob, 2),
            "explanation": f"Linguistic ML model indicates {ml_prior.get('label')} ({fake_prob:.1%}) based on stylistic tokens.",
            "has_conflicting_evidence": False,
            "is_outdated": False,
            "is_misleading": False
        }

    verdict_counts = {}
    total_conf = 0.0
    has_false = False
    has_misleading = False
    has_outdated = False
    has_supported = False
    has_conflicting_individual = False
    
    for c in claims_data:
        v = c["verdict"]
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
        total_conf += c.get("confidence", 0.5)
        
        if v in (VerdictEnum.FALSE, VerdictEnum.LIKELY_FALSE):
            has_false = True
        if v == VerdictEnum.MISLEADING:
            has_misleading = True
        if v == VerdictEnum.OUTDATED:
            has_outdated = True
        if v in (VerdictEnum.SUPPORTED, VerdictEnum.LIKELY_TRUE):
            has_supported = True
        if v == VerdictEnum.PARTIALLY_TRUE:
            has_conflicting_individual = True

    avg_conf = total_conf / len(claims_data)
    has_conflicting = (has_false and has_supported) or has_conflicting_individual
    
    # Precedence: Conflicting > FALSE / LIKELY_FALSE > MISLEADING > OUTDATED > SUPPORTED
    if has_conflicting:
        overall_v = VerdictEnum.PARTIALLY_TRUE
        expl = "Conflicting evidence detected across reporting: central assertions are mixed, with some elements verified while other reports dispute them."
        avg_conf = min(avg_conf, 0.65) # Cap confidence on conflicting sources
    elif has_false:
        overall_v = VerdictEnum.FALSE if verdict_counts.get(VerdictEnum.FALSE, 0) > 0 else VerdictEnum.LIKELY_FALSE
        expl = "One or more central factual propositions are directly contradicted by authoritative sources."
    elif has_misleading:
        overall_v = VerdictEnum.MISLEADING
        expl = "The content contains factual elements framed in a statistically distorted or cherry-picked manner."
    elif has_outdated:
        overall_v = VerdictEnum.OUTDATED
        expl = "The content presents historical information that has been superseded by subsequent events."
    elif has_supported:
        overall_v = VerdictEnum.SUPPORTED if verdict_counts.get(VerdictEnum.SUPPORTED, 0) > 0 else VerdictEnum.LIKELY_TRUE
        expl = "Primary claims are corroborated by verified external sources and accredited documentation."
    else:
        overall_v = claims_data[0]["verdict"]
        expl = claims_data[0]["reason"]

    return {
        "verdict": overall_v,
        "confidence": round(avg_conf, 2),
        "confidence_label": calculate_confidence_label(avg_conf),
        "ml_probability": ml_prior.get("fake_probability"),
        "ml_verdict": ml_prior.get("label"),
        "support_score": round(sum(c.get("support_score", 0) for c in claims_data) / len(claims_data), 2),
        "contradiction_score": round(sum(c.get("contradiction_score", 0) for c in claims_data) / len(claims_data), 2),
        "explanation": expl,
        "has_conflicting_evidence": has_conflicting,
        "is_outdated": has_outdated,
        "is_misleading": has_misleading
    }
