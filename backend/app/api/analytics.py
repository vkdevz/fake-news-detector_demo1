import os
import json
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict, Any

from backend.app.database.connection import get_db
from backend.app.database.models import VerificationRequest, Verdict
from backend.app.core.config import settings

router = APIRouter(prefix="/analytics", tags=["Analytics & Evaluation"])

@router.get("")
def get_analytics(db: Session = Depends(get_db)):
    total_requests = db.query(VerificationRequest).count()
    
    # Verdict counts
    verdict_counts_query = (
        db.query(Verdict.verdict, func.count(Verdict.id))
        .group_by(Verdict.verdict)
        .all()
    )
    verdict_distribution = {v: count for v, count in verdict_counts_query}
    
    # Load ML evaluation metrics
    eval_path = os.path.join(settings.DATA_DIR, "processed", "model_evaluation.json")
    model_eval = {}
    if os.path.exists(eval_path):
        try:
            with open(eval_path, "r") as f:
                model_eval = json.load(f)
        except Exception:
            pass
            
    return {
        "total_verifications": total_requests,
        "verdict_distribution": verdict_distribution,
        "ml_model_evaluation": model_eval,
        "system_status": "ONLINE",
        "retrieval_mode": settings.DEFAULT_RETRIEVAL_MODE
    }
