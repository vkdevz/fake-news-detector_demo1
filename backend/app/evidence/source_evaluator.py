from datetime import datetime, timezone
from urllib.parse import urlparse
import re
from typing import Dict, Any

HIGH_AUTHORITY_DOMAINS = {
    "nasa.gov": 0.98,
    "who.int": 0.98,
    "isro.gov.in": 0.98,
    "cdc.gov": 0.98,
    "nih.gov": 0.98,
    "cancer.gov": 0.98,
    "rbi.org.in": 0.98,
    "reuters.com": 0.94,
    "apnews.com": 0.94,
    "bbc.com": 0.92,
    "snopes.com": 0.92,
    "politifact.com": 0.92,
    "factcheck.org": 0.92,
    "boomlive.in": 0.90,
    "pib.gov.in": 0.95,
    "royal.uk": 0.95,
}

def evaluate_domain_authority(url: str, publisher: str = "") -> float:
    """
    Assigns an authority score between 0.1 and 1.0 based on domain credibility,
    top-level domain extension (.gov, .edu), and known fact-checking registries.
    """
    if not url:
        return 0.50
        
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
            
        # 1. Exact match in high-authority list
        if domain in HIGH_AUTHORITY_DOMAINS:
            return HIGH_AUTHORITY_DOMAINS[domain]
            
        # 2. Institutional domain suffixes
        if domain.endswith(".gov") or domain.endswith(".gov.in") or domain.endswith(".mil"):
            return 0.95
        if domain.endswith(".edu") or domain.endswith(".ac.in"):
            return 0.90
        if domain.endswith(".org"):
            return 0.75
            
        # 3. Known satire publications
        if any(sat in domain for sat in ["theonion", "babylonbee", "newsthump", "fakingnews"]):
            return 0.15
            
        # 4. Standard commercial news outlet baseline
        return 0.65
    except Exception:
        return 0.50

def evaluate_freshness(publication_date_str: str) -> float:
    """
    Evaluates freshness on a 0.0 to 1.0 scale:
    - Within 1 year: 0.90 - 1.0
    - 1-3 years: 0.70 - 0.90
    - 3-5 years: 0.50 - 0.70
    - > 5 years: 0.30 - 0.50
    """
    if not publication_date_str:
        return 0.60
        
    try:
        # Match YYYY-MM-DD or YYYY
        match = re.search(r'(\d{4})(?:-(\d{2})-(\d{2}))?', publication_date_str)
        if match:
            year = int(match.group(1))
            current_year = datetime.now(timezone.utc).year
            diff = max(0, current_year - year)
            if diff == 0:
                return 1.0
            elif diff <= 1:
                return 0.90
            elif diff <= 3:
                return 0.75
            elif diff <= 5:
                return 0.60
            else:
                return 0.40
    except Exception:
        pass
        
    return 0.60

def calculate_evidence_relevance(claim_text: str, excerpt: str) -> float:
    """
    Calculates semantic lexical relevance between the claim and the evidence excerpt.
    """
    claim_tokens = set(re.findall(r'\b\w{3,}\b', claim_text.lower()))
    excerpt_tokens = set(re.findall(r'\b\w{3,}\b', excerpt.lower()))
    
    if not claim_tokens or not excerpt_tokens:
        return 0.30
        
    overlap = claim_tokens.intersection(excerpt_tokens)
    ratio = len(overlap) / len(claim_tokens)
    
    # Scale between 0.30 and 0.98
    relevance = min(0.98, max(0.30, 0.30 + (ratio * 0.68)))
    return round(relevance, 3)
