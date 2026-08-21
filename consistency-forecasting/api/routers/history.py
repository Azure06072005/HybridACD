from fastapi import APIRouter
from pydantic import BaseModel
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEMO_DIR = os.path.join(PROJECT_ROOT, "demo")
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
for p in [SRC_PATH, PROJECT_ROOT, DEMO_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from demo_utils import get_history, clear_history

router = APIRouter(prefix="/api/history", tags=["history"])

@router.get("/")
def read_history():
    df = get_history()
    return df.to_dict(orient="records")

@router.delete("/")
def delete_history():
    clear_history()
    return {"status": "cleared"}
