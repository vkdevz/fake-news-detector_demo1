import re
from typing import List
from backend.app.claims.normalizer import normalize_claim_text

ACTION_VERBS = {
    "confirmed", "discovered", "announced", "revealed", "reported", "stated",
    "ordered", "signed", "proved", "claimed", "decided", "passed", "launched",
    "increased", "decreased", "rose", "fell", "dropped", "surged", "held"
}

def decompose_complex_claim(claim_text: str) -> List[str]:
    """
    Decomposes a compound/complex claim into constituent atomic propositions.
    Examples:
    1. 'NASA discovered aliens on Mars and confirmed extraterrestrial life.' ->
       ['NASA discovered aliens on Mars', 'NASA confirmed extraterrestrial life']
    2. 'Company X launched Product Y in Delhi and sales increased by 200%.' ->
       ['Company X launched Product Y in Delhi', 'Sales increased by 200%']
    """
    text = normalize_claim_text(claim_text)
    
    # Check for coordinate clause with separate subject: "... and <noun> <verb> ..."
    coord_clause_match = re.search(r'^(.*?)\s+(?:and|while)\s+([a-zA-Z]{3,}\s+(?:increased|decreased|rose|fell|dropped|surged|announced|reported|confirmed|died|passed away|failed|occurred|began)\b.*)$', text, re.IGNORECASE)
    if coord_clause_match:
        part1 = coord_clause_match.group(1).strip()
        part2 = coord_clause_match.group(2).strip()
        if len(part1.split()) >= 3 and len(part2.split()) >= 3:
            return [part1, part2.capitalize()]

    # Check for compound predicate sharing same subject: "... and <verb> ..."
    compound_match = re.search(r'^(.*?)\s+(?:and|while)\s+([a-z]+)\s+(.*)$', text, re.IGNORECASE)
    if compound_match:
        part1 = compound_match.group(1).strip()
        verb = compound_match.group(2).lower()
        part2_rest = compound_match.group(3).strip()
        
        if (verb in ACTION_VERBS or verb.endswith("ed") or verb.endswith("ing")) and len(part1.split()) >= 3 and len(part2_rest.split()) >= 2:
            subject_match = re.match(r'^([A-Z][A-Za-z0-9\s]+?)(?=\s+(?:discovered|announced|confirmed|revealed|held|ruled|signed|launched|found|is|was|were|has|have)\b|\s+[a-z]+ed\b)', part1)
            subject = subject_match.group(1).strip() if subject_match else part1.split()[0]
            
            claim1 = part1
            claim2 = f"{subject} {verb} {part2_rest}"
            return [claim1, claim2]

    return [text]
