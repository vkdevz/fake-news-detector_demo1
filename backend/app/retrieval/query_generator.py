from typing import List, Dict, Any
import re
from backend.app.claims.normalizer import normalize_claim_text

def generate_retrieval_queries(claim: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generates diverse search queries for a claim:
    1. exact claim query
    2. paraphrased query
    3. entity + event query
    4. official source query
    5. contradiction / debunk query
    """
    text = claim.get("text", "")
    entities = claim.get("entities", [])
    dates = claim.get("dates", [])
    locations = claim.get("locations", [])
    
    clean_text = normalize_claim_text(text)
    # Strip terminal punctuation
    clean_text = re.sub(r'[.!?]+$', '', clean_text).strip()
    
    queries = []
    
    # 1. Exact Claim Query
    queries.append({
        "query_type": "exact",
        "query_text": f'"{clean_text}"' if len(clean_text.split()) <= 8 else clean_text
    })
    
    # 2. Entity + Action / Event Query
    if entities:
        ent_str = " ".join(entities[:3])
        # Find key verb or subject in claim
        verb_match = re.search(r'\b(discovered|announced|passed|signed|banned|confirmed|killed|arrested|won|lost|increased|decreased|built|launched|ordered)\b', text, re.IGNORECASE)
        verb_str = verb_match.group(0) if verb_match else ""
        loc_str = " ".join(locations[:2]) if locations else ""
        query_text = f"{ent_str} {verb_str} {loc_str}".strip()
        if len(query_text.split()) >= 2:
            queries.append({
                "query_type": "entity_event",
                "query_text": query_text
            })

    # 3. Contradiction / Debunk Query
    queries.append({
        "query_type": "contradiction",
        "query_text": f"{clean_text} (fact check OR debunked OR hoax OR true or false)"
    })
    
    # 4. Official / Trusted Domain Query
    if entities:
        queries.append({
            "query_type": "official",
            "query_text": f"{clean_text} (site:gov OR site:org OR site:edu OR site:who.int OR site:nasa.gov)"
        })

    # 5. Temporal Query (if dates present)
    if dates:
        queries.append({
            "query_type": "temporal",
            "query_text": f"{clean_text} {dates[0]}"
        })

    return queries
