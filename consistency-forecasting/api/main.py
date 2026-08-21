from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import uvicorn

from api.routers import history, results, forecast

app = FastAPI(title="HybridACD Demo API")

app.include_router(history.router)
app.include_router(results.router)
app.include_router(forecast.router)

# Mount a static directory to serve CSS, JS, and HTML
static_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Frontend not found</h1><p>Please ensure index.html exists in the frontend folder.</p>"

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
