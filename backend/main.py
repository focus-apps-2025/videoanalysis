#main.py - Updated with Stop/Delete and Production Features
import os
import sys
import io as _io
import logging
import contextlib
import traceback
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as dt
from typing import List, Optional
from enum import Enum


import hashlib
# --- NEW IMPORTS FOR STRUCTURED DOWNLOAD ---
import zipfile
import shutil
import tempfile
from fastapi.responses import FileResponse, Response # Added Response to imports
# --- END NEW IMPORTS ---

import pandas as pd
from bson import ObjectId
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Request, Form # Ensure Form is imported
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, JSONResponse
from pydantic import BaseModel
from pymongo import MongoClient

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from unified_media_analyzer import UnifiedMediaAnalyzer

load_dotenv()

# -----------------------------
# Logging
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("citnow_analyzer")

# -----------------------------
# Configuration
# -----------------------------
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", "2"))
PROCESS_TIMEOUT_SECONDS = int(os.getenv("PROCESS_TIMEOUT_SECONDS", "900"))

# --- NEW CONFIG FOR STRUCTURED DOWNLOAD ---
BULK_RESULTS_BASE_DIR = os.getenv("BULK_RESULTS_BASE_DIR", "bulk_analysis_reports")
os.makedirs(BULK_RESULTS_BASE_DIR, exist_ok=True)
logger.info(f"Bulk analysis reports will be stored in: {BULK_RESULTS_BASE_DIR}")
# --- END NEW CONFIG ---

# -----------------------------
# Globals
# -----------------------------
analyzer: Optional[UnifiedMediaAnalyzer] = None
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
batch_cancellation_flags = {}  # Track cancellation requests

request_cache = {}
CACHE_TIMEOUT = 30 

# -----------------------------
# Enums
# -----------------------------
class BatchStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STOPPING = "stopping"

# -----------------------------
# FastAPI lifespan
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer
    logger.info("Initializing UnifiedMediaAnalyzer at startup...")
    analyzer = UnifiedMediaAnalyzer()

    if os.path.exists("trained_models.pkl"):
        try:
            analyzer.load_models("trained_models.pkl")
            logger.info("Loaded trained_models.pkl")
        except Exception:
            logger.exception("Failed to load trained_models.pkl")

    try:
        analyzer.load_pretrained_models()
        logger.info("Pre-loaded essential models.")
    except Exception:
        logger.exception("Could not pre-load all models (continuing)")

    yield

    # Cleanup cancellation flags
    batch_cancellation_flags.clear()
    logger.info("Shutting down executor and closing DB client.")
    try:
        executor.shutdown(wait=False)
    except Exception:
        logger.exception("Error shutting down executor")

app = FastAPI(title="CitNow Analyzer API", lifespan=lifespan)

# -----------------------------
# CORS
# -----------------------------
origins = [
    "http://localhost:3000", 
    "http://localhost:3001",
    "https://focusvideoanalylis.netlify.app" 
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# MongoDB
# -----------------------------
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "citnow_analyzer")

try:
    client = MongoClient(MONGODB_URI)
    db = client[MONGODB_DB_NAME]
    results_collection = db["analysis_results"]
    batch_collection = db["batch_jobs"]
    excel_data_collection = db["excel_upload_data"]
    
    # Create indexes for better performance
    results_collection.create_index([("batch_id", 1)])
    results_collection.create_index([("created_at", -1)])
    batch_collection.create_index([("status", 1)])
    batch_collection.create_index([("created_at", -1)])
    logger.info("MongoDB connected and indexes created")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    raise

# -----------------------------
# Pydantic models
# -----------------------------
class AnalysisRequest(BaseModel):
    citnow_url: str
    transcription_language: str = "auto"
    target_language: str = "en"

class AnalysisResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    result_id: Optional[str] = None
    results: Optional[dict] = None

class BatchCreateResponse(BaseModel):
    success: bool
    batch_id: str
    total_urls: int
    message: str

class BatchStatusResponse(BaseModel):
    batch_id: str
    status: str
    total_urls: int
    processed_urls: int
    failed_urls: int
    progress_percentage: float
    current_url: Optional[str] = None
    can_cancel: bool = False

# -----------------------------
# Helpers
# -----------------------------
def clean_results(obj):
    if isinstance(obj, dict):
        return {key: clean_results(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_results(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    elif hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    elif isinstance(obj, dt):
        return obj.isoformat()
    elif isinstance(obj, ObjectId):
        return str(obj)
    else:
        return str(obj)

def is_batch_cancelled(batch_id: str) -> bool:
    """Check if batch processing should be cancelled"""
    return batch_cancellation_flags.get(batch_id, False)

# --- NEW HELPER FOR STRUCTURED DOWNLOAD ---
def _sanitize_path_segment(name: str) -> str:
    """Sanitizes a string to be a safe filename or directory name."""
    if not name:
        return "unknown_dealer"
    # Replace any character that's not alphanumeric, underscore, or hyphen with an underscore
    safe_name = re.sub(r'[^\w\-\.]', '_', name)
    # Trim leading/trailing underscores and limit length to avoid filesystem issues
    return safe_name.strip('_')[:100]
# --- END NEW HELPER ---

# -----------------------------
# Process a single video (threadpool) with stdout/stderr suppression
# -----------------------------
async def process_single_video(url, transcription_language, target_language, timeout=PROCESS_TIMEOUT_SECONDS):
    global analyzer
    loop = asyncio.get_running_loop()
    if analyzer is None:
        raise RuntimeError("Analyzer not initialized")

    async with semaphore:
        def blocking():
            # --- CRITICAL FIX: REMOVE THIS REDUNDANT LINE ---
            # analyzer.target_language = target_language # <--- REMOVE THIS LINE!!!
            # --- END CRITICAL FIX ---
            
            # Silence analyzer prints to prevent flooding the terminal
            with contextlib.redirect_stdout(_io.StringIO()), contextlib.redirect_stderr(_io.StringIO()):
                return analyzer.process_video(
                    url,
                    transcription_language=transcription_language,
                    target_language_short=target_language  # This parameter is correctly used by analyzer.process_video
                )
        try:
            task = loop.run_in_executor(executor, blocking)
            results = await asyncio.wait_for(task, timeout=timeout)
            return results, None
        except Exception as e:
            logger.warning("Processing failed for %s: %s", url, e)
            return None, str(e)

# -----------------------------
# /analyze endpoint (single)
# -----------------------------
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(request: AnalysisRequest):
    global analyzer
    if analyzer is None:
        raise HTTPException(status_code=500, detail="Analyzer not initialized")

    try:
        results, error = await process_single_video(request.citnow_url, request.transcription_language, request.target_language)
        if error:
            raise HTTPException(status_code=500, detail=error)

        results["created_at"] = dt.utcnow()
        
        # Clean results BEFORE saving to MongoDB and returning
        cleaned = clean_results(results)
        
        # Save cleaned results to MongoDB
        res = results_collection.insert_one(cleaned.copy())  # Use copy to avoid modifying
        
        # Add the ID to response but ensure it's a string
        response_data = cleaned.copy()
        response_data["result_id"] = str(res.inserted_id)
        
        return {
            "success": True, 
            "message": "Analysis completed", 
            "result_id": str(res.inserted_id), 
            "results": response_data
        }
        
    except Exception as e:
        logger.exception("Error in /analyze")
        raise HTTPException(status_code=500, detail=str(e))
# -----------------------------
# Bulk analyze (create job)
# -----------------------------
@app.post("/bulk-analyze", response_model=BatchCreateResponse)
async def create_bulk_analysis(background_tasks: BackgroundTasks, file: UploadFile = File(...), transcription_language: str = "auto", target_language: str = Form("en")):
    try:
        # Validate file type
        if not file.filename or not file.filename.lower().endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are supported")

        contents = await file.read()
        df = pd.read_excel(_io.BytesIO(contents))
        logger.info("Excel file loaded with %d rows", len(df))

        # Find URL column
        url_column = None
        for col in df.columns:
            if "video" in col.lower() and "url" in col.lower():
                url_column = col
                break
        if not url_column:
            # Try other common column names
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['url', 'link', 'video']):
                    url_column = col
                    break
        
        if not url_column:
            raise HTTPException(status_code=400, detail="Excel file must contain a column with video URLs (named 'Video URL', 'URL', etc.)")

        urls = df[url_column].dropna().astype(str).unique().tolist()
        # Filter valid URLs
        urls = [url for url in urls if url.startswith(('http://', 'https://'))]
        
        if not urls:
            raise HTTPException(status_code=400, detail="No valid URLs found in the Excel file")

        logger.info("Found %d unique URLs to process", len(urls))

        batch_job = {
            "status": BatchStatus.PENDING,
            "total_urls": len(urls),
            "processed_urls": 0,
            "failed_urls": 0,
            "urls": urls,
            "transcription_language": transcription_language,
            "target_language": target_language,
            "original_filename": file.filename,
            "created_at": dt.utcnow(),
            "updated_at": dt.utcnow()
        }
        
        # Insert and get the batch ID
        inserted = batch_collection.insert_one(batch_job)
        batch_id = str(inserted.inserted_id)
        
        logger.info(f"Created new batch: {batch_id} with {len(urls)} URLs")

        # Clear any previous cancellation flag
        batch_cancellation_flags[batch_id] = False

        # Store excel data in chunks
        store_excel_data_in_chunks(batch_id, file.filename, df)

        # Start background processing
        background_tasks.add_task(process_batch_urls_async, batch_id, urls, transcription_language, target_language)

        return {
            "success": True, 
            "batch_id": batch_id, 
            "total_urls": len(urls), 
            "message": f"Batch processing started for {len(urls)} URLs"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error creating bulk job")
        raise HTTPException(status_code=500, detail=f"Error processing Excel file: {str(e)}")

def store_excel_data_in_chunks(batch_id: str, filename: str, df: pd.DataFrame):
    try:
        records = df.to_dict("records")
        chunk_size = 1000
        total_chunks = (len(records) + chunk_size - 1) // chunk_size
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i + chunk_size]
            excel_chunk_doc = {
                "batch_id": batch_id,
                "filename": filename,
                "uploaded_at": dt.utcnow(),
                "chunk_index": i // chunk_size,
                "total_chunks": total_chunks,
                "data": chunk,
                "total_rows": len(records)
            }
            excel_data_collection.insert_one(excel_chunk_doc)
        logger.info("Stored Excel data rows=%d chunks=%d", len(records), total_chunks)
    except Exception:
        logger.exception("Could not store Excel data")

# -----------------------------
# Batch processing with cancellation support
# -----------------------------
async def process_batch_urls_async(batch_id: str, urls: List[str], transcription_language: str, target_language: str):
    """Process batch URLs with proper cancellation and no duplicates"""
    global analyzer
    if analyzer is None:
        logger.error("Analyzer not initialized for batch processing")
        batch_collection.update_one(
            {"_id": ObjectId(batch_id)}, 
            {"$set": {
                "status": BatchStatus.FAILED, 
                "error": "Analyzer not initialized", 
                "updated_at": dt.utcnow()
            }}
        )
        return

    try:
        batch_collection.update_one(
            {"_id": ObjectId(batch_id)}, 
            {"$set": {
                "status": BatchStatus.PROCESSING, 
                "started_at": dt.utcnow(), 
                "updated_at": dt.utcnow()
            }}
        )
        logger.info("Starting batch %s (%d URLs)", batch_id, len(urls))

        for index, url in enumerate(urls):
            # Check for cancellation before each URL
            if is_batch_cancelled(batch_id):
                logger.info(f"Batch {batch_id} cancellation detected, stopping...")
                batch_collection.update_one(
                    {"_id": ObjectId(batch_id)}, 
                    {"$set": {
                        "status": BatchStatus.CANCELLED, 
                        "updated_at": dt.utcnow()
                    }}
                )
                batch_cancellation_flags.pop(batch_id, None)
                return
            
            # Process single URL
            await process_single_batch_url(
                batch_id, 
                url, 
                index + 1,  # Order starts from 1
                transcription_language, 
                target_language
            )
            
            # Small yield to event loop
            await asyncio.sleep(0.1)

        # Clear cancellation flag on successful completion
        batch_cancellation_flags.pop(batch_id, None)
        
        # Mark completed
        batch_collection.update_one(
            {"_id": ObjectId(batch_id)}, 
            {"$set": {
                "status": BatchStatus.COMPLETED, 
                "completed_at": dt.utcnow(), 
                "updated_at": dt.utcnow()
            }}
        )
        logger.info("Batch %s completed", batch_id)

    except Exception as e:
        logger.exception("Batch processing failed")
        batch_cancellation_flags.pop(batch_id, None)
        batch_collection.update_one(
            {"_id": ObjectId(batch_id)}, 
            {"$set": {
                "status": BatchStatus.FAILED, 
                "error": str(e), 
                "updated_at": dt.utcnow()
            }}
        )

async def process_single_batch_url(batch_id: str, url: str, order: int, transcription_language: str, target_language: str):
    """
    Process a single URL in the batch, with checks to gracefully handle
    cancellation or deletion of the parent batch.
    """
    global analyzer # Analyzer is a global singleton

    # --- Initial check for cancellation/deletion ---
    # Perform this check early to avoid starting heavy processing for a dead batch.
    current_batch_doc = batch_collection.find_one({"_id": ObjectId(batch_id)})
    if not current_batch_doc:
        logger.info(f"Batch {batch_id} URL {order}: Batch already deleted. Skipping processing.")
        return False
    if current_batch_doc.get("status") in [BatchStatus.CANCELLED, BatchStatus.STOPPING]:
        logger.info(f"Batch {batch_id} URL {order}: Batch already cancelled. Skipping processing.")
        return False

    try:
        if analyzer is None:
            logger.error("Analyzer not initialized within process_single_batch_url.")
            raise RuntimeError("Analyzer not initialized.")

        # The analyzer.process_video call itself is a long-running, blocking operation
        # executed in a separate thread. It cannot be interrupted mid-execution.
        # We handle the outcome *after* it completes.
        results, error = await process_single_video(url, transcription_language, target_language)

        # --- Re-check batch status AFTER the long-running process_single_video completes ---
        # This is CRITICAL. The batch status might have changed during the video analysis.
        current_batch_doc = batch_collection.find_one({"_id": ObjectId(batch_id)})

        if not current_batch_doc:
            logger.info(f"Batch {batch_id} URL {order}: Completed processing for URL, but batch was DELETED. Discarding results.")
            # No further updates to DB for this batch.
            return False 
        
        current_status = current_batch_doc.get("status")
        if current_status in [BatchStatus.CANCELLED, BatchStatus.STOPPING]:
            logger.info(f"Batch {batch_id} URL {order}: Completed processing for URL, but batch was CANCELLED. Discarding results and not updating counts.")
            # No further updates to DB for this batch.
            return False

        # --- If the batch is still active and valid, proceed to save results and update progress ---
        if error:
            # If there was an error during video processing, log it to the batch as a failure
            logger.warning(f"Batch {batch_id} URL {order} failed during video processing: {error}")
            error_doc = {
                "batch_id": batch_id,
                "original_url": url,
                "error": error,
                "processing_order": order,
                "target_language": target_language,
                "status": "failed",
                "created_at": dt.utcnow()
            }
            results_collection.insert_one(error_doc)
            
            batch_collection.update_one(
                {"_id": ObjectId(batch_id)}, 
                {
                    "$inc": {"failed_urls": 1}, 
                    "$set": {"current_url": url, "updated_at": dt.utcnow()}
                }
            )
            return False # Mark as failed for this URL

        # If no error and batch is active, save results and update success count
        results["created_at"] = dt.utcnow()
        results["batch_id"] = batch_id
        results["processing_order"] = order
        results["original_url"] = url
        results["target_language_used"] = target_language  
        
        cleaned = clean_results(results)
        results_collection.insert_one(cleaned)

        batch_collection.update_one(
            {"_id": ObjectId(batch_id)}, 
            {
                "$inc": {"processed_urls": 1}, 
                "$set": {"current_url": url, "updated_at": dt.utcnow()}
                # Remove this: ,"target_language": target_language  # No, this should not be updated per URL
            }
        )
        
        logger.info(f"Batch {batch_id}: Successfully processed URL {order} for language {target_language}")
        return True

    except Exception as e:
        # Catch any unexpected errors during the processing of this single URL
        logger.exception(f"Batch {batch_id} URL {order}: Unexpected error during processing.")
        
        # Check again if batch is still active before logging a failure
        current_batch_doc = batch_collection.find_one({"_id": ObjectId(batch_id)})
        if current_batch_doc and current_batch_doc.get("status") not in [BatchStatus.CANCELLED, BatchStatus.STOPPING]:
            error_doc = {
                "batch_id": batch_id,
                "original_url": url,
                "error": str(e),
                "processing_order": order,
                "target_language": target_language,
                "status": "failed",
                "created_at": dt.utcnow()
            }
            results_collection.insert_one(error_doc)
            
            batch_collection.update_one(
                {"_id": ObjectId(batch_id)}, 
                {
                    "$inc": {"failed_urls": 1}, 
                    "$set": {"current_url": url, "updated_at": dt.utcnow()}
                }
            )
        else:
            logger.info(f"Batch {batch_id} URL {order}: Error occurred, but batch was already inactive/deleted. No further DB updates for this URL.")
        
        return False

# -----------------------------
# Batch Control Endpoints (NEW)
# -----------------------------
@app.post("/bulk-cancel/{batch_id}")
async def cancel_bulk_processing(batch_id: str):
    """Cancel an ongoing batch processing job"""
    try:
        # ✅ FIX: Better validation and error handling
        if not batch_id or len(batch_id.strip()) == 0:
            raise HTTPException(status_code=400, detail="Batch ID is required")
        
        # ✅ FIX: Check if it's a valid ObjectId format
        if len(batch_id) != 24:
            raise HTTPException(status_code=400, detail="Invalid batch ID format")
        
        try:
            object_id = ObjectId(batch_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid batch ID format")
        
        # Check if batch exists
        batch = batch_collection.find_one({"_id": object_id})
        if not batch:
            # ✅ FIX: Provide more helpful error message
            raise HTTPException(
                status_code=404, 
                detail=f"Batch not found. The batch may have been completed, deleted, or never existed."
            )
        
        current_status = batch.get("status")
        
        # ✅ FIX: More detailed status checking
        if current_status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED]:
            return {
                "success": False, 
                "message": f"Cannot cancel batch with status: {current_status}",
                "current_status": current_status
            }
        
        # Set cancellation flag
        batch_cancellation_flags[batch_id] = True
        
        # Update batch status
        update_result = batch_collection.update_one(
            {"_id": object_id}, 
            {"$set": {
                "status": BatchStatus.STOPPING,
                "cancelled_at": dt.utcnow(),
                "updated_at": dt.utcnow()
            }}
        )
        
        if update_result.modified_count == 0:
            logger.warning(f"Batch {batch_id} status update failed")
        
        logger.info(f"Batch {batch_id} cancellation requested (status: {current_status})")
        return {
            "success": True, 
            "message": "Batch cancellation initiated",
            "batch_id": batch_id,
            "previous_status": current_status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error cancelling batch {batch_id}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.delete("/bulk-job/{batch_id}")
async def delete_bulk_job(batch_id: str):
    """Delete a batch job and all its associated data"""
    try:
        # ✅ FIX: Validate batch_id first
        if not batch_id or len(batch_id.strip()) == 0:
            raise HTTPException(status_code=400, detail="Batch ID is required")
        
        if len(batch_id) != 24:
            raise HTTPException(status_code=400, detail="Invalid batch ID format")
        
        try:
            object_id = ObjectId(batch_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid batch ID format")
        
        # Check if batch exists first
        batch = batch_collection.find_one({"_id": object_id})
        if not batch:
            raise HTTPException(
                status_code=404, 
                detail=f"Batch not found. It may have been already deleted."
            )
        
        # Set cancellation flag first to stop any ongoing processing
        batch_cancellation_flags[batch_id] = True
        
        # Delete all results for this batch
        delete_results = results_collection.delete_many({"batch_id": batch_id})
        
        # Delete excel data chunks
        delete_excel_data = excel_data_collection.delete_many({"batch_id": batch_id})
        
        # Delete the batch job itself
        delete_batch = batch_collection.delete_one({"_id": object_id})
        
        if delete_batch.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Batch not found during deletion")
        
        # Clean up cancellation flag
        batch_cancellation_flags.pop(batch_id, None)
        
        logger.info(f"Deleted batch {batch_id}: {delete_results.deleted_count} results, {delete_excel_data.deleted_count} excel chunks")
        
        return {
            "success": True,
            "message": f"Batch and {delete_results.deleted_count} results deleted successfully",
            "deleted_results": delete_results.deleted_count,
            "deleted_excel_chunks": delete_excel_data.deleted_count,
            "batch_id": batch_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting batch {batch_id}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@app.get("/bulk-batches")
async def list_all_batches(limit: int = 50, status: Optional[str] = None):
    """List all batch jobs for debugging and monitoring"""
    try:
        query = {}
        if status:
            query["status"] = status
            
        batches = list(batch_collection.find(query)
                          .sort("created_at", -1)
                          .limit(min(limit, 100)))
        
        result = []
        for batch in batches:
            result.append({
                "batch_id": str(batch["_id"]),
                "status": batch.get("status"),
                "total_urls": batch.get("total_urls", 0),
                "processed_urls": batch.get("processed_urls", 0),
                "failed_urls": batch.get("failed_urls", 0),
                "created_at": batch.get("created_at"),
                "updated_at": batch.get("updated_at"),
                "filename": batch.get("original_filename", "Unknown")
            })
        
        return {
            "success": True,
            "total_batches": len(result),
            "batches": result
        }
        
    except Exception as e:
        logger.exception("Error listing batches")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@app.post("/bulk-stop-all")
async def stop_all_processing():
    """Stop all ongoing batch processing jobs"""
    try:
        # Find all processing batches
        processing_batches = batch_collection.find({
            "status": {"$in": [BatchStatus.PROCESSING, BatchStatus.PENDING]}
        })
        
        stopped_count = 0
        for batch in processing_batches:
            batch_id = str(batch["_id"])
            batch_cancellation_flags[batch_id] = True
            
            batch_collection.update_one(
                {"_id": batch["_id"]}, 
                {"$set": {
                    "status": BatchStatus.STOPPING,
                    "cancelled_at": dt.utcnow(),
                    "updated_at": dt.utcnow()
                }}
            )
            stopped_count += 1
        
        return {"success": True, "message": f"Stopping {stopped_count} batch(es)"}
        
    except Exception as e:
        logger.exception("Error stopping all batches")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Status & results endpoints
# -----------------------------
@app.get("/bulk-status/{batch_id}", response_model=BatchStatusResponse)
async def get_bulk_status(batch_id: str):
    try:
        # Validate batch_id format
        if not batch_id or len(batch_id) != 24:
            raise HTTPException(status_code=400, detail="Invalid batch ID format")
        
        # Convert to ObjectId with proper error handling
        try:
            object_id = ObjectId(batch_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid batch ID format")
        
        batch = batch_collection.find_one({"_id": object_id})
        if not batch:
            logger.warning(f"Batch not found: {batch_id}")
            raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")
        
        processed = batch.get("processed_urls", 0)
        total = batch.get("total_urls", 0)
        progress = (processed / total * 100) if total > 0 else 0
        current_status = batch.get("status", "unknown")
        
        # Determine if batch can be cancelled
        can_cancel = current_status in [BatchStatus.PENDING, BatchStatus.PROCESSING]
        
        return {
            "batch_id": batch_id,
            "status": current_status,
            "total_urls": total,
            "processed_urls": processed,
            "failed_urls": batch.get("failed_urls", 0),
            "progress_percentage": round(progress, 2),
            "current_url": batch.get("current_url"),
            "can_cancel": can_cancel
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in /bulk-status for batch {batch_id}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/bulk-results/{batch_id}")
async def get_bulk_results(batch_id: str):
    try:
        batch = batch_collection.find_one({"_id": ObjectId(batch_id)})
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")
        results = list(results_collection.find({"batch_id": batch_id}).sort("created_at", -1))
        for r in results:
            r["_id"] = str(r["_id"])
        return {"batch_id": batch_id, "status": batch.get("status"), "total_processed": len(results), "results": [clean_results(r) for r in results]}
    except Exception:
        logger.exception("Error in /bulk-results")
        raise HTTPException(status_code=500, detail="Internal server error")

# --- NEW ENDPOINT FOR STRUCTURED DOWNLOAD ---
@app.get("/bulk-download/{batch_id}/structured")
async def download_structured_results(batch_id: str, response: Response): # Added response: Response
    """
    Downloads all analysis results for a given batch, organized into folders
    by dealer, and provided as a ZIP file.
    """
    try:
        # Validate batch_id
        if not ObjectId.is_valid(batch_id):
            raise HTTPException(status_code=400, detail="Invalid batch ID format")
        object_id = ObjectId(batch_id)

        batch_doc = batch_collection.find_one({"_id": object_id})
        if not batch_doc:
            raise HTTPException(status_code=404, detail="Batch not found.")
        
        # Create a temporary directory for this download operation
        temp_dir = tempfile.mkdtemp()
        batch_output_root = os.path.join(temp_dir, batch_id) # Unique folder for this batch download
        os.makedirs(batch_output_root, exist_ok=True)
        
        results_cursor = results_collection.find({"batch_id": batch_id})
        
        # Fetch results and organize them
        num_results = 0
        for result in results_cursor:
            num_results += 1
            # Extract dealer info
            dealership = result.get("citnow_metadata", {}).get("dealership")
            sanitized_dealer_name = _sanitize_path_segment(dealership)
            
            # Create dealer-specific subdirectory
            dealer_dir = os.path.join(batch_output_root, sanitized_dealer_name)
            os.makedirs(dealer_dir, exist_ok=True)
            
            # Get a sanitized filename part from the original URL
            original_url = result.get("original_url", "unknown_url")
            # Use URL hash or a part of URL to make filename unique and safe
            url_hash = hashlib.md5(original_url.encode()).hexdigest()[:8]
            # Try to get the last segment of the URL for a more readable name
            url_segment_match = re.search(r'[^/]+(?=\.mp4$|$)', original_url)
            base_filename = url_segment_match.group(0) if url_segment_match else url_hash
            safe_base_filename = _sanitize_path_segment(base_filename)
            
            report_name_prefix = f"analysis_{safe_base_filename}"
            
            # Generate .json report
            json_filename = f"{report_name_prefix}_{str(result['_id'])}.json"
            json_path = os.path.join(dealer_dir, json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(clean_results(result), f, ensure_ascii=False, indent=2)

            # Generate .txt report using the UnifiedMediaAnalyzer instance
            if analyzer is None:
                raise RuntimeError("UnifiedMediaAnalyzer not initialized for report generation.")
            
            # The generate_comprehensive_report method uses results['target_language']
            # which is already stored in the DB for each result.
            # No need to set global analyzer state (analyzer.target_language) here.
            
            txt_report_content = analyzer.generate_comprehensive_report(clean_results(result))
            txt_filename = f"{report_name_prefix}_{str(result['_id'])}.txt"
            txt_path = os.path.join(dealer_dir, txt_filename)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(txt_report_content)

        if num_results == 0:
            shutil.rmtree(temp_dir) # Clean up empty temp dir
            raise HTTPException(status_code=404, detail="No analysis results found for this batch.")

        # Create a ZIP archive of the entire batch folder
        zip_filename = f"batch_{batch_id}_structured_reports.zip"
        zip_filepath = os.path.join(temp_dir, zip_filename)
        
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(batch_output_root):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Add file to zip, preserving directory structure relative to temp_dir
                    zipf.write(file_path, os.path.relpath(file_path, temp_dir))

        # Set Content-Disposition header for the filename
        response.headers["Content-Disposition"] = f"attachment; filename=\"{zip_filename}\""

        # Return the ZIP file as a FastAPI FileResponse
        # The `background=BackgroundTask(lambda: shutil.rmtree(temp_dir))` ensures cleanup
        # after the file has been sent to the client.
        return FileResponse(
            path=zip_filepath,
            filename=zip_filename, # This filename is also passed here, but header takes precedence
            media_type="application/zip",
            background=BackgroundTask(lambda: shutil.rmtree(temp_dir))
        )

    except HTTPException:
        # If an HTTPException was raised, just re-raise it
        raise
    except Exception as e:
        logger.exception(f"Error generating structured download for batch {batch_id}")
        # Ensure temporary directory is cleaned up even on unexpected errors
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Failed to generate structured reports: {str(e)}")

# --- END NEW ENDPOINT ---

@app.get("/bulk-excel-data/{batch_id}")
async def get_bulk_excel_data(batch_id: str, chunk: int = 0):
    try:
        excel_data = excel_data_collection.find_one({"batch_id": batch_id, "chunk_index": chunk})
        if not excel_data:
            excel_data = excel_data_collection.find_one({"batch_id": batch_id})
            if not excel_data:
                raise HTTPException(status_code=404, detail="Excel data not found for this batch")
        return {"batch_id": batch_id, "filename": excel_data.get("filename"), "uploaded_at": excel_data.get("uploaded_at"), "total_rows": excel_data.get("total_rows"), "chunk_index": excel_data.get("chunk_index", 0), "total_chunks": excel_data.get("total_chunks", 1), "data": excel_data.get("data", [])}
    except Exception:
        logger.exception("Error in /bulk-excel-data")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/results", response_class=ORJSONResponse)
async def get_all_results(batch_id: Optional[str] = None, limit: int = 100):
    """
    Fetches individual analysis results.
    Can be filtered by batch_id if needed, but primarily for all individual results.
    """
    try:
        limit = max(1, min(limit, 1000))
        query = {}
        if batch_id:
            query["batch_id"] = batch_id
        results = list(results_collection.find(query).sort("created_at", -1).limit(limit))
        for r in results:
            r["_id"] = str(r["_id"])
        return results
    except Exception:
        logger.exception("Error in /results")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/results/{result_id}")
async def get_result(result_id: str):
    try:
        result = results_collection.find_one({"_id": ObjectId(result_id)})
        if not result:
            raise HTTPException(status_code=404, detail="Not found")
        result["_id"] = str(result["_id"])
        return clean_results(result)
    except Exception:
        logger.exception("Error in /results/{id}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/results/{result_id}")
async def delete_result(result_id: str):
    try:
        delete_operation = results_collection.delete_one({"_id": ObjectId(result_id)})
        if delete_operation.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Result not found")
        return JSONResponse(status_code=200, content={"success": True, "message": "Result deleted successfully"})
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error deleting result")
        raise HTTPException(status_code=500, detail="Internal server error")

# -----------------------------
# Health Check (NEW)
# -----------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Test database connection
        client.admin.command('ping')
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "healthy",
        "timestamp": dt.utcnow().isoformat(),
        "database": db_status,
        "active_batches": len([b for b in batch_cancellation_flags if not batch_cancellation_flags[b]]),
        "analyzer_ready": analyzer is not None
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "CitNow Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "single_analysis": "/analyze",
            "bulk_analysis": "/bulk-analyze", 
            "bulk_status": "/bulk-status/{batch_id}",
            "bulk_cancel": "/bulk-cancel/{batch_id}",
            "bulk_delete": "/bulk-job/{batch_id}",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
