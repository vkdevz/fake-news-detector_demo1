from fastapi import APIRouter
from typing import List, Dict, Any

router = APIRouter(prefix="/demo-samples", tags=["Demonstration Scenarios"])

DEMO_SCENARIOS = [
    {
        "id": "sample-clearly-true",
        "title": "Clearly True Event",
        "category": "Science & Space",
        "expected_verdict": "SUPPORTED",
        "input_type": "claim",
        "text": "ISRO successfully launched the Aditya-L1 spacecraft to study the Sun from Lagrange point L1.",
        "description": "Corroborated by official primary ISRO space agency press releases and verified launch data."
    },
    {
        "id": "sample-clearly-false",
        "title": "Clearly False Viral Hoax",
        "category": "Conspiracy & Space",
        "expected_verdict": "FALSE",
        "input_type": "claim",
        "text": "SHOCKING: NASA discovered alien civilizations on Mars and has been secretly hiding underground cities beneath Martian craters for decades!",
        "description": "Contradicted directly by NASA Astrobiology and accredited fact-checking debunks."
    },
    {
        "id": "sample-misleading-stat",
        "title": "Misleading Statistical Surge",
        "category": "Public Policy & Stats",
        "expected_verdict": "MISLEADING",
        "input_type": "claim",
        "text": "Municipal police reports confirmed that regional violent crime increased 200 percent this year.",
        "description": "Technically true percentage calculation (1 incident to 3), but contextually misleading without baseline counts."
    },
    {
        "id": "sample-outdated-fact",
        "title": "Outdated Historical Fact",
        "category": "World History & Monarchy",
        "expected_verdict": "OUTDATED",
        "input_type": "claim",
        "text": "Queen Elizabeth II is the current reigning monarch of the United Kingdom.",
        "description": "Historically true during her reign, but superseded following the succession of King Charles III."
    },
    {
        "id": "sample-conflicting-sources",
        "title": "Conflicting Evidence Dispute",
        "category": "Infrastructure & Security",
        "expected_verdict": "PARTIALLY_TRUE",
        "input_type": "claim",
        "text": "Initial reports indicate a major regional electrical grid failure was caused by a foreign cyber intrusion.",
        "description": "Preliminary regional regulatory reporting attributed outage to cyber trips, whereas federal agency CISA verified severe freezing rain icing."
    },
    {
        "id": "sample-unverifiable-rumor",
        "title": "Unverifiable Speculation",
        "category": "Diplomacy & Rumors",
        "expected_verdict": "UNVERIFIABLE",
        "input_type": "claim",
        "text": "Foreign ambassadors held a private unminuted dinner discussing covert maritime trade boundaries.",
        "description": "Zero public evidence or primary documentation exists; demonstrates system uncertainty rather than falsely branding it Fake."
    }
]

@router.get("", response_model=List[Dict[str, Any]])
def get_demo_samples():
    return DEMO_SCENARIOS
