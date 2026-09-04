from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, List

from backend.app.database.connection import get_db
from backend.app.database.models import VerificationRequest, Article, Verdict

router = APIRouter(prefix="/verification", tags=["Verification History"])

@router.get("/{req_id}")
def get_verification_by_id(req_id: str, db: Session = Depends(get_db)):
    req = db.query(VerificationRequest).filter(VerificationRequest.id == req_id).first()
    if not req:
        raise HTTPException(status_code=404, detail="Verification request not found")
        
    article = db.query(Article).filter(Article.request_id == req_id).first()
    verdict = db.query(Verdict).filter(Verdict.request_id == req_id).first()
    
    return {
        "request_id": req.id,
        "input_type": req.input_type,
        "raw_input": req.raw_input,
        "status": req.status,
        "created_at": req.created_at.isoformat() if req.created_at else None,
        "article": {
            "title": article.title if article else None,
            "language": article.language if article else "en",
            "word_count": article.word_count if article else 0
        } if article else None,
        "verdict": {
            "verdict": verdict.verdict if verdict else None,
            "confidence": verdict.confidence if verdict else None,
            "confidence_label": verdict.confidence_label if verdict else None,
            "ml_probability": verdict.ml_probability if verdict else None,
            "explanation": verdict.explanation if verdict else None
        } if verdict else None
    }
