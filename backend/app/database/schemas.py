from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class VerdictEnum(str, Enum):
    SUPPORTED = "SUPPORTED"
    LIKELY_TRUE = "LIKELY_TRUE"
    PARTIALLY_TRUE = "PARTIALLY_TRUE"
    MISLEADING = "MISLEADING"
    UNSUPPORTED = "UNSUPPORTED"
    LIKELY_FALSE = "LIKELY_FALSE"
    FALSE = "FALSE"
    OUTDATED = "OUTDATED"
    UNVERIFIABLE = "UNVERIFIABLE"
    SATIRE = "SATIRE"
    OPINION = "OPINION"

class RelationshipEnum(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    PARTIALLY_SUPPORTS = "PARTIALLY_SUPPORTS"
    NEUTRAL = "NEUTRAL"
    OUTDATED = "OUTDATED"
    UNRELATED = "UNRELATED"

class TemporalStatusEnum(str, Enum):
    CURRENT = "CURRENT"
    HISTORICAL = "HISTORICAL"
    OUTDATED = "OUTDATED"
    FUTURE = "FUTURE"
    UNKNOWN = "UNKNOWN"

class TextVerificationRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=25000, description="Article or claim text to verify")

class URLVerificationRequest(BaseModel):
    url: str = Field(..., description="Public HTTP/HTTPS article URL")

class ClaimVerificationRequest(BaseModel):
    claim: str = Field(..., min_length=5, max_length=1000, description="Single factual claim")

class PipelineStepStatus(BaseModel):
    step_key: str
    label: str
    status: str # "pending", "in_progress", "completed", "failed", "skipped"
    details: Optional[str] = None
    duration_ms: Optional[int] = None

class AtomicClaimSchema(BaseModel):
    claim_id: str
    claim_order: int
    text: str
    original_text: Optional[str] = None
    normalized_text: Optional[str] = None
    subject: Optional[str] = None
    predicate: Optional[str] = None
    claim_type: str = "factual" # factual, opinion, prediction, satire, rhetorical
    verifiability: str = "verifiable" # verifiable, unverifiable, ambiguous
    entities: List[str] = []
    dates: List[str] = []
    locations: List[str] = []
    numbers: List[str] = []

class EvidenceItemSchema(BaseModel):
    id: str
    claim_id: str
    title: str
    url: str
    publisher: Optional[str] = None
    domain: Optional[str] = None
    source_type: str = "SECONDARY" # PRIMARY, SECONDARY, ACADEMIC, FACT_CHECKER
    authority_score: float = 0.5
    freshness_score: float = 0.5
    publication_date: Optional[str] = None
    relationship: RelationshipEnum
    relevance_score: float
    stance_confidence: float
    excerpt: str
    reasoning: Optional[str] = None
    temporal_status: TemporalStatusEnum = TemporalStatusEnum.CURRENT
    cluster_id: Optional[str] = None
    is_primary_in_cluster: bool = True
    independence_score: float = 1.0

class FactCheckItemSchema(BaseModel):
    matched_claim: str
    rating: str
    publisher: str
    review_date: Optional[str] = None
    url: Optional[str] = None
    summary: Optional[str] = None
    semantic_similarity: float

class ClaimVerdictSchema(BaseModel):
    claim_id: str
    claim_text: str
    verdict: VerdictEnum
    confidence: float
    confidence_label: str # HIGH, MEDIUM, LOW
    support_score: float
    contradiction_score: float
    reason: str
    supporting_evidence_count: int
    contradicting_evidence_count: int
    evidences: List[EvidenceItemSchema] = []

class OverallVerdictSchema(BaseModel):
    verdict: VerdictEnum
    confidence: float
    confidence_label: str
    ml_probability: Optional[float] = None
    ml_verdict: Optional[str] = None
    support_score: float
    contradiction_score: float
    explanation: str
    has_conflicting_evidence: bool = False
    is_outdated: bool = False
    is_misleading: bool = False

class FullVerificationResponse(BaseModel):
    request_id: str
    input_type: str
    raw_input: str
    status: str
    created_at: str
    language: str
    pipeline_steps: List[PipelineStepStatus]
    overall_verdict: OverallVerdictSchema
    claims: List[ClaimVerdictSchema]
    all_evidence: List[EvidenceItemSchema]
    fact_checks: List[FactCheckItemSchema]
    timeline_events: List[Dict[str, Any]] = []
    source_distribution: Dict[str, int] = {}
    is_demo_mode: bool = False
