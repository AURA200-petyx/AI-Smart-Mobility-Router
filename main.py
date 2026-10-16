import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from ai_router import AIRouter

app = FastAPI(title="AI Smart Mobility Router")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=templates_dir)

# Гибкая модель, которая не упадет при любых входящих данных от клиента
class RouteRequest(BaseModel):
    grid: Optional[List[Any]] = []
    points: List[Dict[str, Any]]
    mode: Optional[str] = "sequence"
    optimization_mode: Optional[str] = "sequence"

class AssistantRequest(BaseModel):
    message: str
    current_lat: Optional[float] = None
    current_lon: Optional[float] = None
    action: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        return templates.TemplateResponse(request, "index.html", {"request": request})
    except TypeError:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Template Error: {str(e)}")

@app.post("/api/analyze-route")
async def analyze_route(data: RouteRequest):
    try:
        result = AIRouter.analyze_and_route(
            grid=data.grid, 
            points=data.points, 
            optimization_mode=data.optimization_mode,
            mode=data.mode
        )
        return result
    except Exception as e:
        print(f"[ERROR in /api/analyze-route]: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/api/ai-assistant")
async def ai_assistant(data: AssistantRequest):
    return AIRouter.process_natural_language_query(
        message=data.message, 
        lat=data.current_lat, 
        lon=data.current_lon, 
        action=data.action
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)