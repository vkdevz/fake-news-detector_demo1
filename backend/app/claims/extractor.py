import uuid
import re
from typing import List, Dict, Any
from backend.app.ml.preprocessor import extract_sentences
from backend.app.claims.normalizer import extract_entities_and_dates, normalize_claim_text
from backend.app.claims.decomposer import decompose_complex_claim

# Subjective opinion markers
OPINION_MARKERS = [
    r"\bi\s+think\b", r"\bi\s+feel\b", r"\bin\s+my\s+opinion\b", r"\bi\s+believe\b",
    r"\bit\s+seems\s+to\s+me\b", r"\bi\s+personally\b", r"\bprobably\b", r"\bmaybe\b",
    r"\barguably\b", r"\bwonderful\b", r"\bterrible\b", r"\bhilarious\b"
]

# Future prediction markers
PREDICTION_MARKERS = [
    r"\bwill\s+likely\b", r"\bprojected\s+to\b", r"\bexpected\s+to\b", 
    r"\bin\s+the\s+coming\s+decades\b", r"\bby\s+20[3-9]\d\b", r"\bwill\s+soon\b"
]

# Satire markers
SATIRE_MARKERS = [
    r"\bonion\b", r"\bsatire\b", r"\bparody\b", r"\bhumor\b", r"\bfictional\b",
    r"\bborowitz\b", r"\bjust\s+kidding\b"
]

def classify_claim_type_and_verifiability(text: str) -> tuple[str, str]:
    """
    Classifies claim into:
    - claim_type: factual, opinion, prediction, satire, rhetorical
    - verifiability: verifiable, unverifiable, ambiguous
    """
    lower = text.lower()
    
    # 1. Satire detection
    for pat in SATIRE_MARKERS:
        if re.search(pat, lower):
            return "satire", "unverifiable"
            
    # 2. Opinion detection
    for pat in OPINION_MARKERS:
        if re.search(pat, lower):
            return "opinion", "unverifiable"
            
    # 3. Prediction detection
    for pat in PREDICTION_MARKERS:
        if re.search(pat, lower):
            return "prediction", "unverifiable"
            
    # 4. Rhetorical question
    if text.strip().endswith("?"):
        return "rhetorical", "unverifiable"

    # 5. Check if it makes an empirical factual assertion
    words = text.split()
    if len(words) < 4:
        return "rhetorical", "unverifiable"

    # Check for presence of verifiable markers: numbers, named entities, action verbs
    has_number = bool(re.search(r'\d+', text))
    has_named_entity = bool(re.search(r'\b[A-Z][a-z]+\b', text))
    has_action_verb = bool(re.search(r'\b(discovered|announced|passed|signed|banned|confirmed|killed|arrested|won|lost|increased|decreased|built|launched|ordered)\b', lower))

    if (has_named_entity or has_action_verb or has_number):
        return "factual", "verifiable"
    else:
        return "factual", "ambiguous"

def extract_claims_from_content(content: str, max_claims: int = 6) -> List[Dict[str, Any]]:
    """
    Extracts, decomposes, and normalizes atomic claims from input content.
    """
    sentences = extract_sentences(content)
    if not sentences:
        # Fallback to single sentence if punctuation is missing
        sentences = [content.strip()]
        
    extracted_claims = []
    order = 1
    
    for sent in sentences[:max_claims * 2]:
        # Decompose compound statements
        atomic_texts = decompose_complex_claim(sent)
        
        for atomic in atomic_texts:
            if len(atomic.split()) < 3:
                continue
                
            claim_type, verifiability = classify_claim_type_and_verifiability(atomic)
            ner_data = extract_entities_and_dates(atomic)
            normalized = normalize_claim_text(atomic)
            
            # Simple subject / predicate heuristic
            tokens = atomic.split()
            subject = " ".join(tokens[:3]) if len(tokens) >= 3 else tokens[0]
            predicate = " ".join(tokens[3:]) if len(tokens) > 3 else ""
            
            claim_obj = {
                "claim_id": str(uuid.uuid4()),
                "claim_order": order,
                "text": atomic.strip(),
                "original_text": sent.strip(),
                "normalized_text": normalized,
                "subject": subject,
                "predicate": predicate,
                "claim_type": claim_type,
                "verifiability": verifiability,
                "entities": ner_data.get("entities", []),
                "dates": ner_data.get("dates", []),
                "locations": ner_data.get("locations", []),
                "numbers": ner_data.get("numbers", [])
            }
            extracted_claims.append(claim_obj)
            order += 1
            
            if len(extracted_claims) >= max_claims:
                break
                
        if len(extracted_claims) >= max_claims:
            break
            
    return extracted_claims
