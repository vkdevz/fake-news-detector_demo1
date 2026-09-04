import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import httpx
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.security import is_safe_url, sanitize_untrusted_text, validate_payload_size
from backend.app.ml.preprocessor import clean_news_text, detect_language
from backend.app.ml.model_registry import model_registry
from backend.app.claims.extractor import extract_claims_from_content
from backend.app.retrieval.query_generator import generate_retrieval_queries
from backend.app.retrieval.factcheck_provider import fact_check_provider
from backend.app.retrieval.web_search_provider import evidence_retrieval_service
from backend.app.evidence.source_evaluator import (
    evaluate_domain_authority, evaluate_freshness, calculate_evidence_relevance
)
from backend.app.evidence.deduplicator import deduplicate_evidence_items
from backend.app.analysis.contradiction_detector import classify_evidence_relationship
from backend.app.analysis.temporal_reasoner import analyze_temporal_context
from backend.app.analysis.context_analyzer import analyze_claim_context
from backend.app.verification.hybrid_engine import compute_claim_verdict, compute_overall_verdict
from backend.app.explanation.explainer import generate_evidence_explanation
from backend.app.database.models import (
    VerificationRequest, Article, Claim, Evidence, Source, FactCheck, ClaimEvidence, Verdict, VerificationRun, SearchQuery
)

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    async def process_verification(
        self,
        input_type: str, # "text", "url", "claim"
        raw_input: str,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        pipeline_steps = []
        
        # 1. Pipeline Step: Content Ingestion
        step1_start = time.time()
        pipeline_steps.append({"step_key": "ingestion", "label": "Reading content", "status": "in_progress"})
        
        valid_size, size_err = validate_payload_size(raw_input)
        if not valid_size:
            pipeline_steps[-1]["status"] = "failed"
            pipeline_steps[-1]["details"] = size_err
            raise ValueError(f"Payload Size Error: {size_err}")

        content_text = raw_input
        url_source = None
        
        if input_type == "url":
            safe, err_msg = is_safe_url(raw_input)
            if not safe:
                pipeline_steps[-1]["status"] = "failed"
                pipeline_steps[-1]["details"] = err_msg
                raise ValueError(f"URL Security Error: {err_msg}")
            
            try:
                async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS, follow_redirects=True) as client:
                    resp = await client.get(raw_input, headers={"User-Agent": "TruthLens Research Bot 1.0"})
                    if resp.status_code != 200:
                        raise ValueError(f"Failed to fetch URL. HTTP status: {resp.status_code}")
                    
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for element in soup(["script", "style", "nav", "footer", "header"]):
                        element.decompose()
                    paragraphs = [p.get_text() for p in soup.find_all("p")]
                    content_text = " ".join(paragraphs)
                    if not content_text.strip():
                        content_text = soup.get_text()
                    url_source = raw_input
            except Exception as e:
                pipeline_steps[-1]["status"] = "failed"
                pipeline_steps[-1]["details"] = str(e)
                raise ValueError(f"URL Fetch Error: {str(e)}")

        cleaned_text = clean_news_text(content_text)
        language = detect_language(content_text)
        pipeline_steps[-1]["status"] = "completed"
        pipeline_steps[-1]["details"] = f"Ingested {len(cleaned_text.split())} words in '{language}' language."
        pipeline_steps[-1]["duration_ms"] = int((time.time() - step1_start) * 1000)

        # 2. Pipeline Step: Claim Extraction & Decomposition
        step2_start = time.time()
        pipeline_steps.append({"step_key": "claim_extraction", "label": "Extracting & decomposing claims", "status": "in_progress"})
        
        extracted_claims = extract_claims_from_content(cleaned_text, max_claims=settings.MAX_CLAIMS_PER_ARTICLE)
        if not extracted_claims:
            extracted_claims = [{
                "claim_id": "c-fallback",
                "claim_order": 1,
                "text": cleaned_text[:300],
                "normalized_text": cleaned_text[:300],
                "subject": "Claim",
                "predicate": "",
                "claim_type": "factual",
                "verifiability": "verifiable",
                "entities": [],
                "dates": [],
                "locations": []
            }]
            
        pipeline_steps[-1]["status"] = "completed"
        pipeline_steps[-1]["details"] = f"Identified {len(extracted_claims)} atomic claims."
        pipeline_steps[-1]["duration_ms"] = int((time.time() - step2_start) * 1000)

        # 3. Pipeline Step: Running ML Classifier (Linguistic Prior)
        step3_start = time.time()
        pipeline_steps.append({"step_key": "ml_classification", "label": "Running ML classifier", "status": "in_progress"})
        
        ml_prior = model_registry.predict_prior(cleaned_text)
        pipeline_steps[-1]["status"] = "completed"
        pipeline_steps[-1]["details"] = f"Prior: {ml_prior['label']} ({ml_prior['fake_probability']:.1%} probability via {ml_prior['model_type']})."
        pipeline_steps[-1]["duration_ms"] = int((time.time() - step3_start) * 1000)

        # 4. Pipeline Step: Searching Fact-Checks (ISOLATED STRICTLY PER CLAIM)
        step4_start = time.time()
        pipeline_steps.append({"step_key": "fact_check_search", "label": "Searching fact-check databases", "status": "in_progress"})
        
        all_fact_checks = []
        for claim in extracted_claims:
            fcs = await fact_check_provider.search_fact_checks(claim["text"], claim_id=claim["claim_id"])
            all_fact_checks.extend(fcs)
            
        pipeline_steps[-1]["status"] = "completed"
        pipeline_steps[-1]["details"] = f"Retrieved {len(all_fact_checks)} matching fact-check records."
        pipeline_steps[-1]["duration_ms"] = int((time.time() - step4_start) * 1000)

        # 5. Pipeline Step: Retrieving External Web Evidence & Multi-Query Generation
        step5_start = time.time()
        pipeline_steps.append({"step_key": "evidence_retrieval", "label": "Retrieving external evidence", "status": "in_progress"})
        
        raw_evidences = []
        all_queries = []
        for claim in extracted_claims:
            queries = generate_retrieval_queries(claim)
            for q in queries:
                q["claim_id"] = claim["claim_id"]
                all_queries.append(q)
                
            claim_evs = await evidence_retrieval_service.retrieve_evidence(queries, claim["text"])
            for ev in claim_evs:
                ev["claim_id"] = claim["claim_id"]
                raw_evidences.append(ev)

        # Deduplicate syndicated sources and compute independence clustering
        deduped_evidences = deduplicate_evidence_items(raw_evidences)
        cluster_count = len(set(e.get("cluster_id") for e in deduped_evidences if e.get("cluster_id")))
        
        pipeline_steps[-1]["status"] = "completed"
        pipeline_steps[-1]["details"] = f"Retrieved {len(deduped_evidences)} sources across {cluster_count or len(deduped_evidences)} independent clusters."
        pipeline_steps[-1]["duration_ms"] = int((time.time() - step5_start) * 1000)

        # 6. Pipeline Step: Contradiction & Stance Detection
        step6_start = time.time()
        pipeline_steps.append({"step_key": "contradiction_check", "label": "Checking contradictions & stances", "status": "in_progress"})
        
        classified_evidences = []
        for ev in deduped_evidences:
            target_claim = next((c for c in extracted_claims if c["claim_id"] == ev.get("claim_id")), extracted_claims[0])
            relationship, stance_conf, reason = classify_evidence_relationship(target_claim["text"], ev.get("excerpt", ""))
            
            auth_score = evaluate_domain_authority(ev.get("url", ""), ev.get("publisher", ""))
            freshness_score = evaluate_freshness(ev.get("publication_date", ""))
            relevance_score = calculate_evidence_relevance(target_claim["text"], ev.get("excerpt", ""))
            
            ev_copy = dict(ev)
            ev_copy["relationship"] = relationship
            ev_copy["stance_confidence"] = stance_conf
            ev_copy["reasoning"] = reason
            ev_copy["authority_score"] = auth_score
            ev_copy["freshness_score"] = freshness_score
            ev_copy["relevance_score"] = relevance_score
            classified_evidences.append(ev_copy)

        pipeline_steps[-1]["status"] = "completed"
        pipeline_steps[-1]["details"] = "Classified evidence stances and cross-examined assertions."
        pipeline_steps[-1]["duration_ms"] = int((time.time() - step6_start) * 1000)

        # 7. Pipeline Step: Temporal Context & Context Analysis
        step7_start = time.time()
        pipeline_steps.append({"step_key": "temporal_context", "label": "Analyzing temporal & statistical context", "status": "in_progress"})
        
        all_timelines = []
        claim_verdict_results = []
        source_counts = {}
        
        for claim in extracted_claims:
            claim_id = claim["claim_id"]
            relevant_evs = [ev for ev in classified_evidences if ev.get("claim_id") == claim_id]
            # STRICT PER-CLAIM FACT-CHECK FILTERING (fixes cross-contamination bug)
            relevant_fcs = [fc for fc in all_fact_checks if fc.get("claim_id") == claim_id]
            
            # Run temporal & context reasoners
            temporal_res = analyze_temporal_context(claim, relevant_evs)
            context_res = analyze_claim_context(claim["text"], relevant_evs)
            all_timelines.extend(temporal_res.get("timeline_events", []))
            
            # Calculate individual claim verdict
            c_verdict = compute_claim_verdict(
                claim=claim,
                evidences=relevant_evs,
                fact_checks=relevant_fcs,
                temporal_data=temporal_res,
                context_data=context_res,
                ml_prior=ml_prior
            )
            
            supp_cnt = sum(1 for e in relevant_evs if e["relationship"] == "SUPPORTS")
            contra_cnt = sum(1 for e in relevant_evs if e["relationship"] == "CONTRADICTS")
            
            claim_verdict_results.append({
                "claim_id": claim_id,
                "claim_text": claim["text"],
                "verdict": c_verdict["verdict"],
                "confidence": c_verdict["confidence"],
                "confidence_label": c_verdict["confidence_label"],
                "support_score": c_verdict["support_score"],
                "contradiction_score": c_verdict["contradiction_score"],
                "reason": c_verdict["reason"],
                "supporting_evidence_count": supp_cnt,
                "contradicting_evidence_count": contra_cnt,
                "evidences": relevant_evs
            })

        for ev in classified_evidences:
            domain = ev.get("domain", "other")
            source_counts[domain] = source_counts.get(domain, 0) + 1

        pipeline_steps[-1]["status"] = "completed"
        pipeline_steps[-1]["details"] = "Evaluated chronological validity and contextual baselines."
        pipeline_steps[-1]["duration_ms"] = int((time.time() - step7_start) * 1000)

        # 8. Pipeline Step: Hybrid Verdict Generation & Grounded Explanation
        step8_start = time.time()
        pipeline_steps.append({"step_key": "verdict_generation", "label": "Synthesizing hybrid explainable verdict", "status": "in_progress"})
        
        overall_verdict_data = compute_overall_verdict(claim_verdict_results, ml_prior)
        explanation_narrative = generate_evidence_explanation(
            overall_verdict=overall_verdict_data,
            claims=extracted_claims,
            evidences=classified_evidences,
            fact_checks=all_fact_checks
        )
        overall_verdict_data["explanation"] = explanation_narrative

        pipeline_steps[-1]["status"] = "completed"
        pipeline_steps[-1]["details"] = f"Verdict: {overall_verdict_data['verdict'].value} ({overall_verdict_data['confidence_label']} confidence)."
        pipeline_steps[-1]["duration_ms"] = int((time.time() - step8_start) * 1000)

        # Persist full relational audit chain to SQLite Database
        request_id = f"tl-{int(time.time()*1000)}"
        if db:
            try:
                db_req = VerificationRequest(
                    id=request_id,
                    input_type=input_type,
                    raw_input=raw_input,
                    status="COMPLETED",
                    completed_at=datetime.now(timezone.utc)
                )
                db.add(db_req)
                
                db_art = Article(
                    request_id=request_id,
                    title=cleaned_text[:80],
                    url=url_source,
                    cleaned_text=cleaned_text,
                    language=language,
                    word_count=len(cleaned_text.split())
                )
                db.add(db_art)
                
                # Persist claims
                claim_db_map = {}
                for c in extracted_claims:
                    db_claim = Claim(
                        id=c["claim_id"],
                        article_id=db_art.id,
                        claim_order=c.get("claim_order", 1),
                        text=c["text"],
                        normalized_text=c.get("normalized_text"),
                        subject=c.get("subject"),
                        predicate=c.get("predicate"),
                        claim_type=c.get("claim_type", "factual"),
                        verifiability=c.get("verifiability", "verifiable"),
                        entities=c.get("entities", []),
                        dates=c.get("dates", []),
                        locations=c.get("locations", [])
                    )
                    db.add(db_claim)
                    claim_db_map[c["claim_id"]] = db_claim
                
                # Persist evidence and claim_evidence
                for ev in classified_evidences:
                    db_ev = Evidence(
                        id=ev["id"],
                        title=ev.get("title", "Evidence Source"),
                        url=ev.get("url", ""),
                        publisher=ev.get("publisher"),
                        excerpt=ev.get("excerpt", ""),
                        authority_score=ev.get("authority_score", 0.5),
                        freshness_score=ev.get("freshness_score", 0.5)
                    )
                    db.merge(db_ev)
                    
                    if ev.get("claim_id") in claim_db_map:
                        db_ce = ClaimEvidence(
                            claim_id=ev["claim_id"],
                            evidence_id=ev["id"],
                            relationship_label=ev.get("relationship", "NEUTRAL"),
                            relevance_score=ev.get("relevance_score", 0.5),
                            stance_confidence=ev.get("stance_confidence", 0.5),
                            reasoning=ev.get("reasoning")
                        )
                        db.add(db_ce)

                # Persist overall verdict
                db_verdict = Verdict(
                    request_id=request_id,
                    verdict=overall_verdict_data["verdict"].value,
                    confidence=overall_verdict_data["confidence"],
                    confidence_label=overall_verdict_data["confidence_label"],
                    ml_probability=overall_verdict_data.get("ml_probability"),
                    ml_verdict=overall_verdict_data.get("ml_verdict"),
                    support_score=overall_verdict_data["support_score"],
                    contradiction_score=overall_verdict_data["contradiction_score"],
                    explanation=overall_verdict_data["explanation"]
                )
                db.add(db_verdict)
                db.commit()
            except Exception as e:
                logger.error("DB persistence error: %s", e)
                db.rollback()

        return {
            "request_id": request_id,
            "input_type": input_type,
            "raw_input": raw_input,
            "status": "COMPLETED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "language": language,
            "pipeline_steps": pipeline_steps,
            "overall_verdict": overall_verdict_data,
            "claims": claim_verdict_results,
            "all_evidence": classified_evidences,
            "fact_checks": all_fact_checks,
            "timeline_events": all_timelines,
            "source_distribution": source_counts,
            "is_demo_mode": False
        }

orchestrator = PipelineOrchestrator()
