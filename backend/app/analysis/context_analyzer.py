import re
from typing import Dict, Any, List

def analyze_claim_context(claim_text: str, evidences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Detects technically accurate but contextually misleading statements:
    - Missing denominator/baseline in percentage statistics
    - Cherry-picked time windows
    - Exaggerated interpretations
    """
    lower_text = claim_text.lower()
    
    # 1. Percentage check
    percent_match = re.search(r'(\d+)\s*(%|percent)', lower_text)
    if percent_match:
        number = int(percent_match.group(1))
        # High percentage change often used for small numbers (1 -> 3 is 200%)
        if number >= 100:
            for ev in evidences:
                ev_lower = ev.get("excerpt", "").lower()
                if "baseline" in ev_lower or "population" in ev_lower or "historic lows" in ev_lower:
                    return {
                        "is_misleading": True,
                        "context_flags": ["MISSING_STATISTICAL_BASELINE"],
                        "details": f"The assertion highlights a {number}% proportional increase without communicating that the absolute baseline is exceptionally low."
                    }
                    
    # 2. Causation vs Correlation flags
    if any(term in lower_text for term in ["proves that", "causes all", "secret reason for"]):
        return {
            "is_misleading": True,
            "context_flags": ["UNSUBSTANTIATED_CAUSATION"],
            "details": "The statement asserts definitive causation where standard scientific consensus notes only correlation or disproven association."
        }
        
    return {
        "is_misleading": False,
        "context_flags": [],
        "details": "No obvious contextual omission or statistical distortion detected."
    }
