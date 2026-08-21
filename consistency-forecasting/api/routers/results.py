from fastapi import APIRouter

router = APIRouter(prefix="/api/results", tags=["results"])

@router.get("/")
def get_results():
    return {
        "gpt-4o-mini": {"display": "GPT-4o-mini", "b": {"avs": 0.0307, "bs": 0.205}, "h": {"avs": 0.0007, "bs": 0.200}},
        "gemini-2.5-flash": {"display": "Gemini 2.5 Flash", "b": {"avs": 0.1116, "bs": 0.141}, "h": {"avs": 0.0087, "bs": 0.130}},
        "mistral-medium": {"display": "Mistral Medium", "b": {"avs": 0.0792, "bs": 0.202}, "h": {"avs": 0.0087, "bs": 0.167}},
        "mistral-small": {"display": "Mistral Small", "b": {"avs": 0.0740, "bs": 0.211}, "h": {"avs": 0.0023, "bs": 0.202}},
        "minimax": {"display": "MiniMax M3", "b": {"avs": 0.1266, "bs": 0.123}, "h": {"avs": 0.0005, "bs": 0.116}}
    }
