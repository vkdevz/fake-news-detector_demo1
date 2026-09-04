# TruthLens REST API Specification

TruthLens exposes a clean REST API built on FastAPI, returning validated Pydantic v2 models.

Base URL: `http://127.0.0.1:8000/api`

---

## 1. Endpoints

### 1.1 Verify Claim
- **Method**: `POST`
- **Path**: `/api/verify/claim`
- **Request Body**:
```json
{
  "claim": "ISRO successfully launched the Aditya-L1 spacecraft to study the Sun."
}
```
- **Response**: `200 OK`
```json
{
  "request_id": "tl-1725451234567",
  "input_type": "claim",
  "status": "COMPLETED",
  "overall_verdict": {
    "verdict": "SUPPORTED",
    "confidence": 0.98,
    "confidence_label": "HIGH",
    "support_score": 0.95,
    "contradiction_score": 0.05,
    "explanation": "Primary claims are corroborated by verified external sources and accredited documentation.",
    "has_conflicting_evidence": false,
    "is_outdated": false,
    "is_misleading": false
  },
  "claims": [ ... ],
  "all_evidence": [ ... ],
  "pipeline_steps": [ ... ]
}
```

### 1.2 Verify News Text
- **Method**: `POST`
- **Path**: `/api/verify/text`
- **Request Body**:
```json
{
  "text": "Paste full multi-paragraph article body here..."
}
```

### 1.3 Verify Article URL
- **Method**: `POST`
- **Path**: `/api/verify/url`
- **Request Body**:
```json
{
  "url": "https://example.com/verified-news-story"
}
```
*Note: Evaluated against SSRF filters. Private IP addresses (127.0.0.1, 10.0.0.0/8, 169.254.169.254) are rejected with HTTP 400.*

### 1.4 System Analytics & Model Benchmark
- **Method**: `GET`
- **Path**: `/api/analytics`
- **Response**:
```json
{
  "total_verifications": 7,
  "verdict_distribution": {
    "SUPPORTED": 2,
    "FALSE": 1,
    "MISLEADING": 1,
    "OUTDATED": 1,
    "UNSUPPORTED": 1,
    "UNVERIFIABLE": 1
  },
  "ml_model_evaluation": { ... },
  "system_status": "ONLINE",
  "retrieval_mode": "hybrid"
}
```

### 1.5 Academic Benchmark Samples
- **Method**: `GET`
- **Path**: `/api/demo-samples`
- **Response**: Array of 6 labeled benchmark edge cases.

### 1.6 Health Check
- **Method**: `GET`
- **Path**: `/api/health`
- **Response**:
```json
{
  "status": "healthy",
  "service": "TruthLens",
  "version": "1.0.0",
  "database": "sqlite_connected"
}
```
