import os
import json
import logging
import re
import httpx
from urllib.parse import urlparse, parse_qs, unquote
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from backend.app.core.config import settings
from backend.app.core.security import is_safe_url, sanitize_untrusted_text
from backend.app.evidence.source_evaluator import evaluate_domain_authority, evaluate_freshness

logger = logging.getLogger(__name__)

class EvidenceRetrievalService:
    def __init__(self):
        self.evidence_archive = self._load_evidence_archive()

    def _load_evidence_archive(self) -> List[Dict[str, Any]]:
        archive_path = os.path.join(settings.DATA_DIR, "demo", "evidence_archive.json")
        if os.path.exists(archive_path):
            try:
                with open(archive_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Error loading demo evidence archive: %s", e)
        return []

    async def _search_duckduckgo_open(self, query: str, max_results: int = 4) -> List[Dict[str, Any]]:
        """
        Free, keyless live web search fallback using DuckDuckGo HTML endpoint.
        """
        results = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }
        try:
            async with httpx.AsyncClient(timeout=4.0, follow_redirects=True) as client:
                resp = await client.post(
                    "https://html.duckduckgo.com/html/",
                    data={"q": query},
                    headers=headers
                )
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    raw_results = soup.find_all("div", class_="result")
                    for r in raw_results[:max_results]:
                        a_title = r.find("a", class_="result__a")
                        snippet_elem = r.find("a", class_="result__snippet")
                        if not a_title:
                            continue
                        href = a_title.get("href", "")
                        if "uddg=" in href:
                            parsed_q = parse_qs(urlparse(href).query)
                            href = unquote(parsed_q.get("uddg", [href])[0])
                        
                        safe, _ = is_safe_url(href)
                        if not safe:
                            continue
                            
                        title = a_title.get_text(strip=True)
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                        domain = urlparse(href).netloc.lower()
                        if domain.startswith("www."):
                            domain = domain[4:]
                            
                        # Extract year if visible in snippet
                        year_match = re.search(r'\b(20[12]\d)\b', snippet)
                        pub_date = year_match.group(1) if year_match else None
                        authority = evaluate_domain_authority(href, domain)
                        freshness = evaluate_freshness(pub_date) if pub_date else 0.70
                        
                        results.append({
                            "id": f"ev-live-ddg-{len(results)+1}",
                            "title": title,
                            "url": href,
                            "publisher": domain.capitalize(),
                            "domain": domain,
                            "source_type": "PRIMARY" if authority >= 0.90 else "SECONDARY",
                            "authority_score": authority,
                            "freshness_score": freshness,
                            "publication_date": pub_date,
                            "excerpt": sanitize_untrusted_text(snippet)
                        })
        except Exception as e:
            logger.debug("DuckDuckGo open search fallback error: %s", e)
        return results

    async def retrieve_evidence(self, queries: List[Dict[str, str]], claim_text: str) -> List[Dict[str, Any]]:
        evidences = []
        seen_urls = set()

        # 1. Check offline/benchmark curated evidence archive first for exact or high-relevance matches
        claim_words = set(re.findall(r'\b\w{4,}\b', claim_text.lower()))
        for doc in self.evidence_archive:
            keywords = [k.lower() for k in doc.get("keywords", [])]
            overlap = sum(1 for kw in keywords if kw in claim_words or any(kw in w for w in claim_words))
            if overlap >= 2 or (keywords and any(kw in claim_text.lower() for kw in keywords if len(kw) > 4)):
                evidences.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "publisher": doc.get("publisher", "Official Source"),
                    "domain": doc.get("domain", "source.org"),
                    "source_type": doc.get("source_type", "PRIMARY"),
                    "authority_score": doc.get("authority_score", 0.90),
                    "freshness_score": doc.get("freshness_score", 0.85),
                    "publication_date": doc.get("publication_date"),
                    "excerpt": sanitize_untrusted_text(doc.get("excerpt", ""))
                })
                seen_urls.add(doc["url"])

        # 2. If SerpAPI key is available, execute live search query
        if settings.SERPAPI_API_KEY and queries:
            try:
                top_query = queries[0]["query_text"]
                async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS) as client:
                    resp = await client.get(
                        "https://serpapi.com/search.json",
                        params={"q": top_query, "api_key": settings.SERPAPI_API_KEY, "num": 4}
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        for item in data.get("organic_results", []):
                            link = item.get("link")
                            safe, _ = is_safe_url(link)
                            if safe and link not in seen_urls:
                                domain = urlparse(link).netloc
                                authority = evaluate_domain_authority(link, domain)
                                evidences.append({
                                    "id": f"ev-live-serp-{len(evidences)+1}",
                                    "title": item.get("title", ""),
                                    "url": link,
                                    "publisher": domain,
                                    "domain": domain,
                                    "source_type": "PRIMARY" if authority >= 0.90 else "SECONDARY",
                                    "authority_score": authority,
                                    "publication_date": item.get("date"),
                                    "excerpt": sanitize_untrusted_text(item.get("snippet", ""))
                                })
                                seen_urls.add(link)
            except Exception as e:
                logger.warning("SerpAPI search failed: %s", e)

        # 3. If no/low evidence retrieved and we have queries, query live DuckDuckGo open search
        if len(evidences) < 2 and queries:
            search_query = queries[0]["query_text"]
            live_ddg = await self._search_duckduckgo_open(search_query, max_results=4)
            for item in live_ddg:
                if item["url"] not in seen_urls:
                    evidences.append(item)
                    seen_urls.add(item["url"])

        return evidences

evidence_retrieval_service = EvidenceRetrievalService()
