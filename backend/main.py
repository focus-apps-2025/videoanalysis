from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from pydantic import BaseModel
from typing import List, Optional
import os
import sys
from datetime import datetime
from bson import ObjectId
import json
import numpy as np
from datetime import datetime as dt
import traceback


from pymongo import MongoClient
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from unified_media_analyzer import UnifiedMediaAnalyzer

load_dotenv()

app = FastAPI(title="CitNow Analyzer API")
def clean_results(obj):
    """Recursively convert non-JSON-serializable types to JSON-safe types."""
    if isinstance(obj, dict):
        return {key: clean_results(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_results(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dt):
        return obj.isoformat()
    elif isinstance(obj, ObjectId):
        return str(obj)
    else:
        return obj
# Enable CORS for React
origins = [
    "http://localhost:3000",
    "https://focusvideoanalylis.netlify.app",
    "https://videoanalysis-e55w.onrender.com"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))
db = client[os.getenv("MONGODB_DB_NAME", "citnow_analyzer")]
results_collection = db["analysis_results"]

class AnalysisRequest(BaseModel):
    citnow_url: str
    transcription_language: str = "auto"
    target_language: str = "ta"

class AnalysisResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    result_id: Optional[str] = None
    results: Optional[dict] = None

@app.post("/analyze")
async def analyze_video(request: AnalysisRequest):
    try:
        analyzer = UnifiedMediaAnalyzer(target_language=request.target_language)
        if os.path.exists("trained_models.pkl"):
            analyzer.load_models("trained_models.pkl")

        results = analyzer.process_video(
            request.citnow_url,
            transcription_language=request.transcription_language,
            target_language_short=request.target_language
        )

                # Add timestamp
        results["created_at"] = dt.utcnow()
        
        # Insert into MongoDB
        result = results_collection.insert_one(results)
        result_id = str(result.inserted_id)

        # Clean results for JSON serialization
        cleaned_results = clean_results(results)

        return {
            "success": True,
            "message": "Analysis completed",
            "result_id": result_id,
            "results": cleaned_results  #  Now defined and safe
        }
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results",response_class=ORJSONResponse)
async def get_all_results():
    try:
        results = list(results_collection.find().sort("created_at", -1))
        for r in results:
            r["_id"] = str(r["_id"])
        return results
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{result_id}")
async def get_result(result_id: str):
    try:
        result = results_collection.find_one({"_id": ObjectId(result_id)})
        if not result:
            raise HTTPException(status_code=404, detail="Not found")
        result["_id"] = str(result["_id"])
        return clean_results(result)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete("/results/{result_id}")
async def delete_result(result_id: str):
    try:
        res = results_collection.delete_one({"_id": ObjectId(result_id)})
        if res.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Result not found")
        # 204 means “No Content” on success
        return
    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
