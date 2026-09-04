import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Text, Float, Integer, Boolean, DateTime, ForeignKey, Enum as SQLEnum, JSON
)
from sqlalchemy.orm import relationship
from backend.app.database.connection import Base

def generate_uuid() -> str:
    return str(uuid.uuid4())

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class VerificationRequest(Base):
    __tablename__ = "verification_requests"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    input_type = Column(String(20), nullable=False) # "text", "url", "claim"
    raw_input = Column(Text, nullable=False)
    status = Column(String(20), default="PENDING") # PENDING, PROCESSING, COMPLETED, FAILED
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    article = relationship("Article", back_populates="request", uselist=False, cascade="all, delete-orphan")
    runs = relationship("VerificationRun", back_populates="request", cascade="all, delete-orphan")
    verdicts = relationship("Verdict", back_populates="request", cascade="all, delete-orphan")

class Article(Base):
    __tablename__ = "articles"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    request_id = Column(String(36), ForeignKey("verification_requests.id"), nullable=False)
    title = Column(String(500), nullable=True)
    url = Column(Text, nullable=True)
    cleaned_text = Column(Text, nullable=False)
    language = Column(String(10), default="en")
    word_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=utc_now)
    
    # Relationships
    request = relationship("VerificationRequest", back_populates="article")
    claims = relationship("Claim", back_populates="article", cascade="all, delete-orphan")

class Claim(Base):
    __tablename__ = "claims"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    article_id = Column(String(36), ForeignKey("articles.id"), nullable=True)
    claim_order = Column(Integer, default=1)
    text = Column(Text, nullable=False)
    original_text = Column(Text, nullable=True)
    normalized_text = Column(Text, nullable=True)
    subject = Column(String(255), nullable=True)
    predicate = Column(String(255), nullable=True)
    claim_type = Column(String(50), default="factual") # factual, opinion, prediction, satire, rhetorical
    verifiability = Column(String(50), default="verifiable") # verifiable, unverifiable, ambiguous
    entities = Column(JSON, default=list) # List of extracted entities
    dates = Column(JSON, default=list) # Temporal expressions
    locations = Column(JSON, default=list) # Extracted locations
    numbers = Column(JSON, default=list) # Extracted numbers and statistics
    created_at = Column(DateTime, default=utc_now)
    
    # Relationships
    article = relationship("Article", back_populates="claims")
    claim_evidences = relationship("ClaimEvidence", back_populates="claim", cascade="all, delete-orphan")
    fact_checks = relationship("FactCheck", back_populates="claim", cascade="all, delete-orphan")
    search_queries = relationship("SearchQuery", back_populates="claim", cascade="all, delete-orphan")

class Source(Base):
    __tablename__ = "sources"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    domain = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=True)
    source_type = Column(String(50), default="SECONDARY") # PRIMARY, SECONDARY, ACADEMIC, FACT_CHECKER, UNKNOWN
    authority_score = Column(Float, default=0.5) # 0.0 to 1.0
    bias_rating = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=utc_now)
    
    # Relationships
    evidences = relationship("Evidence", back_populates="source")

class Evidence(Base):
    __tablename__ = "evidences"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    source_id = Column(String(36), ForeignKey("sources.id"), nullable=True)
    title = Column(String(500), nullable=False)
    url = Column(Text, nullable=False)
    canonical_url = Column(Text, nullable=True)
    publisher = Column(String(255), nullable=True)
    publication_date = Column(DateTime, nullable=True)
    excerpt = Column(Text, nullable=False)
    freshness_score = Column(Float, default=0.5)
    directness_score = Column(Float, default=0.5)
    created_at = Column(DateTime, default=utc_now)
    
    # Relationships
    source = relationship("Source", back_populates="evidences")
    claim_evidences = relationship("ClaimEvidence", back_populates="evidence", cascade="all, delete-orphan")

class FactCheck(Base):
    __tablename__ = "fact_checks"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    claim_id = Column(String(36), ForeignKey("claims.id"), nullable=False)
    matched_claim = Column(Text, nullable=False)
    rating = Column(String(100), nullable=False) # e.g. "False", "Pants on Fire", "Correct"
    publisher = Column(String(255), nullable=False)
    url = Column(Text, nullable=True)
    review_date = Column(DateTime, nullable=True)
    semantic_similarity = Column(Float, default=0.0)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)
    
    # Relationships
    claim = relationship("Claim", back_populates="fact_checks")

class ClaimEvidence(Base):
    __tablename__ = "claim_evidences"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    claim_id = Column(String(36), ForeignKey("claims.id"), nullable=False)
    evidence_id = Column(String(36), ForeignKey("evidences.id"), nullable=False)
    relationship_label = Column(String(50), nullable=False) # SUPPORTS, CONTRADICTS, PARTIALLY_SUPPORTS, NEUTRAL, OUTDATED, UNRELATED
    relevance_score = Column(Float, default=0.5)
    stance_confidence = Column(Float, default=0.5)
    reasoning = Column(Text, nullable=True)
    temporal_status = Column(String(50), default="CURRENT") # CURRENT, HISTORICAL, OUTDATED, FUTURE, UNKNOWN
    created_at = Column(DateTime, default=utc_now)
    
    # Relationships
    claim = relationship("Claim", back_populates="claim_evidences")
    evidence = relationship("Evidence", back_populates="claim_evidences")

class SearchQuery(Base):
    __tablename__ = "search_queries"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    claim_id = Column(String(36), ForeignKey("claims.id"), nullable=False)
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50), default="exact") # exact, paraphrased, entity_event, official, contradiction
    results_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=utc_now)
    
    # Relationships
    claim = relationship("Claim", back_populates="search_queries")

class Verdict(Base):
    __tablename__ = "verdicts"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    request_id = Column(String(36), ForeignKey("verification_requests.id"), nullable=False)
    claim_id = Column(String(36), ForeignKey("claims.id"), nullable=True) # None if whole article verdict
    verdict = Column(String(50), nullable=False) # SUPPORTED, LIKELY_TRUE, PARTIALLY_TRUE, MISLEADING, UNSUPPORTED, LIKELY_FALSE, FALSE, OUTDATED, UNVERIFIABLE, SATIRE, OPINION
    confidence = Column(Float, nullable=False) # 0.0 to 1.0
    confidence_label = Column(String(20), nullable=False) # HIGH, MEDIUM, LOW
    ml_probability = Column(Float, nullable=True) # Classical ML prior
    ml_verdict = Column(String(20), nullable=True)
    support_score = Column(Float, default=0.0)
    contradiction_score = Column(Float, default=0.0)
    explanation = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utc_now)
    
    # Relationships
    request = relationship("VerificationRequest", back_populates="verdicts")

class VerificationRun(Base):
    __tablename__ = "verification_runs"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    request_id = Column(String(36), ForeignKey("verification_requests.id"), nullable=False)
    step_name = Column(String(100), nullable=False)
    status = Column(String(20), default="COMPLETED") # IN_PROGRESS, COMPLETED, FAILED, SKIPPED
    details = Column(JSON, default=dict)
    duration_ms = Column(Integer, default=0)
    created_at = Column(DateTime, default=utc_now)
    
    # Relationships
    request = relationship("VerificationRequest", back_populates="runs")
