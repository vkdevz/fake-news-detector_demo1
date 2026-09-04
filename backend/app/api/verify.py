from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, List

from backend.app.database.connection import get_db
from backend.app.database.schemas import (
    TextVerificationRequest, URLVerificationRequest, ClaimVerificationRequest, FullVerificationResponse
)
from backend.app.verification.pipeline_orchestrator import orchestrator
from backend.app.database.models import VerificationRequest, Article, Verdict

router = APIRouter(prefix="/verify", tags=["Verification"])

@router.post("/text", response_model=FullVerificationResponse)
async def verify_text(payload: TextVerificationRequest, db: Session = Depends(get_db)):
    try:
        res = await orchestrator.process_verification(
            input_type="text",
            raw_input=payload.text,
            db=db
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/url", response_model=FullVerificationResponse)
async def verify_url(payload: URLVerificationRequest, db: Session = Depends(get_db)):
    try:
        res = await orchestrator.process_verification(
            input_type="url",
            raw_input=str(payload.url),
            db=db
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/claim", response_model=FullVerificationResponse)
async def verify_claim(payload: ClaimVerificationRequest, db: Session = Depends(get_db)):
    try:
        res = await orchestrator.process_verification(
            input_type="claim",
            raw_input=payload.claim,
            db=db
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
