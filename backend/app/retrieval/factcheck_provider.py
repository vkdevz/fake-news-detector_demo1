import os
import json
import logging
import httpx
from typing import List, Dict, Any, Optional
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

def calculate_token_similarity(text1: str, text2: str) -> float:
    """
    Computes Jaccard word-level similarity between two text strings.
    """
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)

class FactCheckProvider:
    def __init__(self):
        self.api_key = settings.GOOGLE_FACT_CHECK_API_KEY
        self.local_cache = self._load_local_fact_checks()

    def _load_local_fact_checks(self) -> List[Dict[str, Any]]:
        local_path = os.path.join(settings.DATA_DIR, "demo", "fact_checks.json")
        if os.path.exists(local_path):
            try:
                with open(local_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Error loading local fact checks: %s", e)
        return []

    async def search_fact_checks(self, claim_text: str, claim_id: str = "", threshold: float = 0.20) -> List[Dict[str, Any]]:
        results = []
        
        # 1. Try Google Fact Check Tools API if API key is provided
        if self.api_key:
            try:
                url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search"
                params = {"query": claim_text, "key": self.api_key, "pageSize": 5}
                async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS) as client:
                    resp = await client.get(url, params=params)
                    if resp.status_code == 200:
                        data = resp.json()
                        claims = data.get("claims", [])
                        for item in claims:
                            matched_text = item.get("text", "")
                            reviews = item.get("claimReview", [{}])
                            first_review = reviews[0] if reviews else {}
                            
                            sim = calculate_token_similarity(claim_text, matched_text)
                            if sim >= threshold:
                                results.append({
                                    "claim_id": claim_id,
                                    "matched_claim": matched_text,
                                    "rating": first_review.get("textualRating", "Unknown"),
                                    "publisher": first_review.get("publisher", {}).get("name", "FactCheck Organization"),
                                    "review_date": first_review.get("reviewDate"),
                                    "url": first_review.get("url"),
                                    "summary": f"Review by {first_review.get('publisher', {}).get('name', 'FactCheck')}",
                                    "semantic_similarity": round(sim, 3)
                                })
            except Exception as e:
                logger.warning("Google Fact Check API query failed: %s. Falling back to local cache.", e)

        # 2. Check local curated fact check database
        for item in self.local_cache:
            # Check keywords or direct token similarity
            matched = item["matched_claim"]
            sim = calculate_token_similarity(claim_text, matched)
            
            # Check keyword match
            keywords = item.get("keywords", [])
            claim_lower = claim_text.lower()
            keyword_hits = sum(1 for kw in keywords if kw in claim_lower)
            if keywords and keyword_hits >= 2:
                sim = max(sim, 0.40 + (keyword_hits * 0.10))
                
            if sim >= threshold:
                results.append({
                    "claim_id": claim_id,
                    "matched_claim": item["matched_claim"],
                    "rating": item["rating"],
                    "publisher": item["publisher"],
                    "review_date": item.get("review_date"),
                    "url": item.get("url"),
                    "summary": item.get("summary"),
                    "semantic_similarity": round(min(1.0, sim), 3)
                })

        # Sort by highest semantic similarity
        results.sort(key=lambda x: x["semantic_similarity"], reverse=True)
        return results[:3]

fact_check_provider = FactCheckProvider()
