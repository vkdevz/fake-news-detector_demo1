from typing import List, Dict, Any

def generate_evidence_explanation(
    overall_verdict: Dict[str, Any],
    claims: List[Dict[str, Any]],
    evidences: List[Dict[str, Any]],
    fact_checks: List[Dict[str, Any]]
) -> str:
    """
    Generates a structured, evidence-grounded explanation.
    Never fabricates facts or citations; strictly synthesizes verified retrieved evidence.
    """
    verdict = overall_verdict.get("verdict")
    conf_label = overall_verdict.get("confidence_label", "MEDIUM")
    base_expl = overall_verdict.get("explanation", "")
    
    parts = []
    parts.append(f"**Verification Summary**: The analyzed input received an overall evaluation of **{verdict}** with **{conf_label}** confidence.")
    parts.append(base_expl)
    
    # 1. Mention Fact Check findings if present
    if fact_checks:
        fc = fact_checks[0]
        parts.append(f"• **Fact-Check Finding**: {fc.get('publisher')} reviewed a matching claim and assigned a rating of *\"{fc.get('rating')}\"*.")

    # 2. Cite top evidence sources
    if evidences:
        parts.append("• **Key Evidence Grounding**:")
        for idx, ev in enumerate(evidences[:3], 1):
            pub = ev.get("publisher", "Independent Source")
            stance = ev.get("relationship", "DOCUMENTED")
            excerpt = ev.get("excerpt", "")
            # Shorten excerpt to 140 chars
            short_excerpt = excerpt[:140] + ("..." if len(excerpt) > 140 else "")
            parts.append(f"  [{idx}] *{pub}* ({stance}): \"{short_excerpt}\"")

    # 3. Methodological note
    parts.append("\n*Audit Note*: TruthLens verifies claims against public evidence and source credibility metrics. It does not claim absolute philosophical certainty.")

    return "\n\n".join(parts)
