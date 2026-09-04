import re
from datetime import datetime, timezone
from typing import Dict, Any, List

def analyze_temporal_context(claim: Dict[str, Any], evidences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates chronological consistency:
    - Claim dates
    - Evidence publication dates
    - Distinguishes CURRENT, HISTORICAL, OUTDATED, FUTURE, UNKNOWN
    - Generates timeline events for visualization
    """
    current_year = datetime.now(timezone.utc).year
    claim_dates = claim.get("dates", [])
    claim_text = claim.get("text", "")
    
    temporal_status = "CURRENT"
    explanation = "Claim addresses ongoing or modern facts."
    timeline_events = []
    
    # 1. Check for future date expressions
    for d in claim_dates:
        year_match = re.match(r'^(20[3-9]\d)$', str(d))
        if year_match:
            year = int(year_match.group(1))
            if year > current_year:
                temporal_status = "FUTURE"
                explanation = f"Claim references future projections ({year}) which cannot be definitively verified in present time."
                
    # 2. Check for historical keywords or past monarch/leader mentions
    lower_text = claim_text.lower()
    if any(term in lower_text for term in ["queen elizabeth ii", "soviet union", "ussr", "ancient", "in 19", "in 201"]):
        # Check if evidence mentions succession, death, or dissolution
        for ev in evidences:
            ev_excerpt = ev.get("excerpt", "").lower()
            if any(term in ev_excerpt for term in ["died", "passed away", "succeeded", "dissolved", "former"]):
                temporal_status = "OUTDATED"
                explanation = "Statement was historically true during a prior era but is outdated under current conditions."
                break
        if temporal_status != "OUTDATED":
            temporal_status = "HISTORICAL"
            explanation = "Claim refers to historical events."

    # 3. Build Timeline Events
    timeline_events.append({
        "event_type": "CLAIM_SUBMITTED",
        "title": "Claim Ingestion",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "description": claim_text[:120] + ("..." if len(claim_text) > 120 else "")
    })
    
    for ev in evidences:
        pub_date = ev.get("publication_date")
        if pub_date:
            timeline_events.append({
                "event_type": "EVIDENCE_PUBLISHED",
                "title": ev.get("publisher", "Source Coverage"),
                "date": pub_date,
                "description": ev.get("title", "")
            })
            
    timeline_events.append({
        "event_type": "CURRENT_STATUS",
        "title": f"Status: {temporal_status}",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "description": explanation
    })

    return {
        "temporal_status": temporal_status,
        "is_outdated": temporal_status == "OUTDATED",
        "explanation": explanation,
        "timeline_events": timeline_events
    }
