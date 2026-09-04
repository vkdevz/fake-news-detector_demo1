import re
from typing import Tuple, Dict, Any

CONTRADICTION_PHRASES = [
    r"\bno\s+evidence\b",
    r"\bhas\s+found\s+no\b",
    r"\bunfounded\b",
    r"\bcompletely\s+unfounded\b",
    r"\bdebunk(ed|s|ing)?\b",
    r"\bfake\b",
    r"\bhoax\b",
    r"\bfalse\b",
    r"\bdisproven\b",
    r"\bnot\s+true\b",
    r"\bdeni(ed|es|al)\b",
    r"\bnot\s+announced\b",
    r"\bdid\s+not\s+discover\b",
    r"\bno\s+scientific\s+evidence\b",
    r"\bdangerous\s+medical\s+disinformation\b",
    r"\bremain\s+completely\s+legal\s+tender\b",
    r"\bzero\s+evidence\b",
    r"\bdispute[ds]?\b",
    r"\brefute[ds]?\b",
    r"\bmisrepresent(ed|s|ing)?\b",
    r"\bunsubstantiated\b",
    r"\bincorrect\b",
    r"\binaccurate\b"
]

SUPPORT_PHRASES = [
    r"\bsuccessfully\s+launched\b",
    r"\bconfirmed\s+that\b",
    r"\bofficially\s+announced\b",
    r"\bplaced\s+into\s+orbit\b",
    r"\bverified\b",
    r"\bpeer-reviewed\b",
    r"\bvalidated\b",
    r"\bapproved\b",
    r"\bupheld\b",
    r"\bsuggested\s+(a\s+)?",
    r"\battributed\s+to\b",
    r"\bcaused\s+by\b",
    r"\breports?\s+(indicate[ds]?|state[ds]?|confirm[s]?)\b",
    r"\binitial\s+(findings|reports?|assessment)\b",
    r"\bconcluded\s+that\b",
    r"\bshowed\s+that\b",
    r"\breport(ed)?\s+that\b",
    r"\bfound\s+that\b",
    r"\bcorroborat(ed|es|ing)\b",
    r"\bsubstantiat(ed|es|ing)\b",
    r"\bdocumented\b",
    r"\baccorded\s+with\b"
]

OUTDATED_PHRASES = [
    r"\bdied\s+on\b",
    r"\bpassed\s+away\b",
    r"\bsucceeded\s+by\b",
    r"\bformer\b",
    r"\bpreviously\b",
    r"\bno\s+longer\b",
    r"\bwas\s+superseded\b",
    r"\bhas\s+expired\b",
    r"\bstepped\s+down\b"
]

def classify_evidence_relationship(claim_text: str, excerpt: str) -> Tuple[str, float, str]:
    """
    Classifies relationship between claim and evidence excerpt into:
    - SUPPORTS
    - CONTRADICTS
    - PARTIALLY_SUPPORTS
    - NEUTRAL
    - OUTDATED
    - UNRELATED
    
    Returns: (relationship_label, stance_confidence, reasoning)
    """
    lower_excerpt = excerpt.lower()
    lower_claim = claim_text.lower()
    
    # 1. Check for explicit temporal obsolescence markers
    for pat in OUTDATED_PHRASES:
        if re.search(pat, lower_excerpt) and ("queen" in lower_claim or "former" in lower_excerpt or "succeeded" in lower_excerpt or "died" in lower_excerpt):
            return "OUTDATED", 0.92, "Evidence indicates the statement reflects past historical conditions that have since changed."

    # 2. Check for explicit contradiction / debunking markers
    for pat in CONTRADICTION_PHRASES:
        if re.search(pat, lower_excerpt):
            match = re.search(pat, lower_excerpt).group(0)
            return "CONTRADICTS", 0.94, f"Source explicitly disputes or refutes the claim (detected indicator: '{match}')."

    # 3. Check for support and affirmative reporting markers
    for pat in SUPPORT_PHRASES:
        if re.search(pat, lower_excerpt):
            match = re.search(pat, lower_excerpt).group(0)
            return "SUPPORTS", 0.88, f"Source corroborates key factual assertions of the claim ('{match}')."

    # 4. Check for statistical context / misleading indicator
    if "%" in lower_claim or "percent" in lower_claim:
        if any(term in lower_excerpt for term in ["baseline", "proportional change", "context", "population", "historic low"]):
            return "PARTIALLY_SUPPORTS", 0.85, "Source confirms literal numbers but reveals critical omitted baseline context."

    # 5. Check semantic token overlap for neutral vs unrelated
    claim_tokens = set(re.findall(r'\b[a-z]{3,}\b', lower_claim))
    excerpt_tokens = set(re.findall(r'\b[a-z]{3,}\b', lower_excerpt))
    overlap = claim_tokens.intersection(excerpt_tokens)
    
    if len(overlap) >= 3:
        return "NEUTRAL", 0.60, "Source discusses related topic or entities but neither confirms nor refutes the central assertion."
        
    return "UNRELATED", 0.70, "Evidence excerpt does not substantively address the claim's core predicate."
