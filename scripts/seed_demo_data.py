import os
import sys
import asyncio
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.app.database.connection import SessionLocal
from backend.app.verification.pipeline_orchestrator import orchestrator
from backend.app.api.demo import DEMO_SCENARIOS

async def seed_data():
    db = SessionLocal()
    print(f"Seeding {len(DEMO_SCENARIOS)} benchmark scenarios into TruthLens database...")
    
    for item in DEMO_SCENARIOS:
        print(f"Processing: {item['title']}...")
        try:
            res = await orchestrator.process_verification(
                input_type=item["input_type"],
                raw_input=item["text"],
                db=db
            )
            print(f"  -> Generated Verdict: {res['overall_verdict']['verdict'].value} ({res['overall_verdict']['confidence_label']})")
        except Exception as e:
            print(f"  -> Error: {e}")
            
    db.close()
    print("Database seeding completed successfully!")

if __name__ == "__main__":
    asyncio.run(seed_data())
