from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from botocore.exceptions import ClientError
from typing import Optional, Dict, Any, List
import os
import uuid
import time
import pandas as pd
import re
import lancedb
import psutil
import json
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from utils.secrets_manager import load_secrets_into_env
load_secrets_into_env()

from config import GROQ_API_KEY, OPENAI_API_KEY, EMBEDDING_MODEL_NAME, LANCEDB_PATH

from db.ingestor import Ingestor
from retrieval.retriever import Retriever
from llm.conversational import get_conversational_answer
from preprocess.document_loader import preprocess_text
from llm.summarizer import summarizer
from db.session_logger import SessionLogger
from router import (
    rule_gate, call_llm_router, route_and_answer,
     TAU_ROUTER_HIGH,
    reply_greeting, reply_handoff, reply_safety, reply_oos, reply_not_found, reply_chitchat
)

# Centralized S3 and secret-based paths
DATA_DIR = os.getenv("DATA_DIR", "s3://droogbucket/data")
CHATDB_PATH = os.getenv("CHATDB_PATH", "s3://droogbucket/lancedb/chatdb")
LANCEDB_PATH = os.getenv("LANCEDB_PATH", "s3://droogbucket/lancedb")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

OPENAI_API_KEY = OPENAI_API_KEY

def get_unified_knowledge_base_docs(db_path):
    """Get all documents in the unified knowledge base"""
    try:
        db = lancedb.connect(db_path)
        if "unified_knowledge_base" not in db.table_names():
            return []
        
        table = db.open_table("unified_knowledge_base")
        df = table.to_pandas()
        
        if df.empty:
            return []
        
        # Get unique documents with stats
        docs = df.groupby('doc_id').agg({
            'file_hash': 'first',
            'doc_version': 'first',
            'chunk_id': 'count',
            'page': 'max'
        }).reset_index()
        
        docs.columns = ['doc_id', 'file_hash', 'doc_version', 'chunk_count', 'max_page']
        return docs.to_dict('records')
    except Exception as e:
        print(f"Error getting KB docs: {e}")
        return []

def sanitize_table_name(name):
    name_no_ext = os.path.splitext(name)[0]
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', name_no_ext)

def get_existing_tables(db_path):
    db = lancedb.connect(db_path)
    return set(db.table_names())

app = FastAPI(
    title="Hybrid RAG Chatbot API",
    description="API endpoints for document ingestion, session management, chat, and comprehensive analytics.",
    version="1.0.0"
)

origins = [
    "http://localhost:3000",       
    "http://localhost:3001",
    "http://13.239.35.20"
           
]

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models - response-models
class DocumentIngestRequest(BaseModel):
    force_reindex: Optional[bool] = False

class SessionCreateRequest(BaseModel):
    llm_model: str
    retrieval_mode: str
    top_k_dense: int
    top_k_sparse: int
    rrf_k: int
    top_k_final: int
    doc_filter: Optional[str] = None

class ChatRequest(BaseModel):
    session_id: str
    query: str
    llm_model: Optional[str] = None  # Use session default if not provided
    retrieval_mode: Optional[str] = None  # Use session default if not provided
    top_k_dense: Optional[int] = None
    top_k_sparse: Optional[int] = None
    rrf_k: Optional[int] = None
    top_k_final: Optional[int] = None

class SessionUpdateRequest(BaseModel):
    llm_model: Optional[str] = None
    retrieval_mode: Optional[str] = None
    top_k_dense: Optional[int] = None
    top_k_sparse: Optional[int] = None
    rrf_k: Optional[int] = None
    top_k_final: Optional[int] = None
    doc_filter: Optional[List[str]] = None

# Enhanced session storage with configuration
sessions = {}
retrievers = {}
session_logger = SessionLogger(CHATDB_PATH)

# --- HTTP Exception global handler ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

# Enhanced Root Endpoint
@app.get("/")
async def root():
    kb_docs = get_unified_knowledge_base_docs(LANCEDB_PATH)
    endpoints = [
        {"method": "GET", "path": "/health", "description": "Check system and database health"},
        {"method": "GET", "path": "/knowledge-base", "description": "Get unified knowledge base information"},
        {"method": "GET", "path": "/lancedb/vector-tables", "description": "List all vector DB tables"},
        {"method": "GET", "path": "/lancedb/session-tables", "description": "List all session DB tables"},
        {"method": "GET", "path": "/lancedb/table/{db_type}/{table_name}", "description": "Get table info for a specific table"},
        {"method": "GET", "path": "/documents", "description": "List all documents available in the data folder"},
        {"method": "POST", "path": "/documents/ingest", "description": "Ingest documents into the unified knowledge base"},
        {"method": "POST", "path": "/documents/upload", "description": "Upload PDF documents"},
        {"method": "GET", "path": "/documents/chunks", "description": "Get chunks from unified knowledge base"},
        {"method": "POST", "path": "/sessions", "description": "Create a new session for chat"},
        {"method": "PUT", "path": "/sessions/{session_id}", "description": "Update session configuration"},
        {"method": "DELETE", "path": "/sessions/{session_id}", "description": "Delete a session"},
        {"method": "GET", "path": "/sessions/{session_id}", "description": "Get current session details"},
        {"method": "GET", "path": "/sessions/history/{session_id}", "description": "Get session chat history"},
        {"method": "GET", "path": "/sessions/{session_id}/goal-set", "description": "Export session as goal set for RAGAS evaluation"},
        {"method": "POST", "path": "/chat", "description": "Send a chat query and get response"},
        {"method": "GET", "path": "/analytics/sessions/{session_id}/metrics", "description": "Get analytics metrics for a session"},
        {"method": "GET", "path": "/analytics/system", "description": "Get current system and application performance metrics"},
        {"method": "POST", "path": "/sessions/{session_id}/end", "description": "End a session and log summary"}
    ]

    return {
        "msg": "RAG Chatbot API connected",
        "version": "1.0.0",
        "features": ["unified_knowledge_base", "document_filtering", "enhanced_routing_metrics"],
        "endpoints": endpoints,
        "active_sessions": len(sessions),
        "knowledge_base": {
            "total_documents": len(kb_docs),
            "available": len(kb_docs) > 0
        }
    }

# --- Health Check ---
@app.get("/health")
async def health_check():
    try:
        vector_db = lancedb.connect(LANCEDB_PATH)
        chat_db = lancedb.connect(CHATDB_PATH)
        kb_docs = get_unified_knowledge_base_docs(LANCEDB_PATH)
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "databases": {
                "vector_db": "connected",
                "chat_db": "connected",
                "vector_tables": len(vector_db.table_names()),
                "chat_tables": len(chat_db.table_names())
            },
            "knowledge_base": {
                "documents": len(kb_docs),
                "available": len(kb_docs) > 0
            },
            "active_sessions": len(sessions),
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# --- Knowledge Base Endpoint ---
@app.get("/knowledge-base")
def get_knowledge_base_info():
    try:
        kb_docs = get_unified_knowledge_base_docs(LANCEDB_PATH)
        
        if not kb_docs:
            return {
                "status": "empty",
                "documents": [],
                "total_documents": 0,
                "message": "Knowledge base is empty. Upload documents to get started."
            }
        
        return {
            "status": "available",
            "documents": kb_docs,
            "total_documents": len(kb_docs),
            "total_chunks": sum(doc["chunk_count"] for doc in kb_docs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving knowledge base info: {str(e)}")

# LanceDB Table Endpoints (unchanged)
@app.get("/lancedb/vector-tables")
def list_vector_tables():
    try:
        db = lancedb.connect(LANCEDB_PATH)
        tables = db.table_names()
        
        table_info = []
        for table_name in tables:
            try:
                table = db.open_table(table_name)
                df = table.to_pandas()
                table_info.append({
                    "name": table_name,
                    "row_count": len(df),
                    "columns": list(df.columns) if not df.empty else []
                })
            except Exception as e:
                table_info.append({
                    "name": table_name,
                    "error": str(e)
                })
        
        return {
            "tables": table_info,
            "total_tables": len(tables)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing vector tables: {str(e)}")

@app.get("/lancedb/session-tables")
def list_session_tables():
    try:
        db = lancedb.connect(CHATDB_PATH)
        tables = db.table_names()
        
        table_info = []
        for table_name in tables:
            try:
                table = db.open_table(table_name)
                df = table.to_pandas()
                table_info.append({
                    "name": table_name,
                    "row_count": len(df),
                    "columns": list(df.columns) if not df.empty else []
                })
            except Exception as e:
                table_info.append({
                    "name": table_name,
                    "error": str(e)
                })
        
        return {
            "tables": table_info,
            "total_tables": len(tables)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing session tables: {str(e)}")

@app.get("/lancedb/table/{db_type}/{table_name}")
def get_table_info(db_type: str, table_name: str, limit: Optional[int] = 10):
    if db_type not in ["vector", "session"]:
        raise HTTPException(status_code=400, detail="db_type must be 'vector' or 'session'")
    
    try:
        db_path = LANCEDB_PATH if db_type == "vector" else CHATDB_PATH
        db = lancedb.connect(db_path)
        
        if table_name not in db.table_names():
            raise HTTPException(status_code=404, detail="Table not found")
        
        table = db.open_table(table_name)
        df = table.to_pandas()
        
        return {
            "table_name": table_name,
            "db_type": db_type,
            "total_rows": len(df),
            "columns": list(df.columns) if not df.empty else [],
            "sample_rows": df.head(limit).to_dict("records") if not df.empty else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving table info: {str(e)}")


from urllib.parse import urlparse
import boto3

def list_s3_pdf_files(bucket, prefix):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".pdf") and not key.endswith("/"):
                files.append(key[len(prefix):] if key.startswith(prefix) else key)
    return files

@app.post("/documents/ingest")
def ingest_documents(request: DocumentIngestRequest, background_tasks: BackgroundTasks):
    try:
        # Detect S3 or local
        pdf_files = []
        s3_mode = DATA_DIR.startswith("s3://")
        if s3_mode:
            parsed = urlparse(DATA_DIR)
            bucket = parsed.netloc
            prefix = parsed.path.lstrip("/")
            pdf_files = list_s3_pdf_files(bucket, prefix)
        else:
            pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            raise HTTPException(status_code=404, detail="No PDF files found in data directory.")
        
        # Always use unified knowledge base
        table_name = "unified_knowledge_base"
        existing_tables = get_existing_tables(LANCEDB_PATH)
        
        if table_name in existing_tables and not request.force_reindex:
            kb_docs = get_unified_knowledge_base_docs(LANCEDB_PATH)
            return {
                "status": "already_indexed",
                "table_name": table_name,
                "existing_documents": len(kb_docs),
                "message": "Knowledge base already exists. Use force_reindex=true to reindex all documents."
            }
        
        ingestor = Ingestor(db_path=LANCEDB_PATH, table_name=table_name)
        
        def ingest_all_files():
            if s3_mode:
                for pdf_key in pdf_files:
                    # For S3, pass the S3 URI to ingestor
                    file_path = f"s3://{bucket}/{prefix}{pdf_key}" if not pdf_key.startswith(prefix) else f"s3://{bucket}/{pdf_key}"
                    try:
                        ingestor.run(file_path)
                    except Exception as e:
                        print(f"Error processing {pdf_key}: {e}")
            else:
                for pdf_file in pdf_files:
                    file_path = os.path.join(DATA_DIR, pdf_file)
                    try:
                        ingestor.run(file_path)
                    except Exception as e:
                        print(f"Error processing {pdf_file}: {e}")

        background_tasks.add_task(ingest_all_files)
        
        return {
            "status": "ingestion_started",
            "table_name": table_name,
            "files_to_process": len(pdf_files),
            "force_reindex": request.force_reindex,
            "files": pdf_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting ingestion: {str(e)}")
    

def clean_chunk(chunk):
    """Clean chunk data for JSON serialization"""
    cleaned = {}
    for k, v in chunk.items():
        if isinstance(v, (np.int32, np.int64)):
            cleaned[k] = int(v)
        elif isinstance(v, (np.float32, np.float64)):
            cleaned[k] = float(v)
        else:
            cleaned[k] = v
    return cleaned

@app.get("/documents/chunks")
def get_document_chunks(limit: Optional[int] = 20, doc_filter: Optional[str] = None):
    try:
        table_name = "unified_knowledge_base"
        existing_tables = get_existing_tables(LANCEDB_PATH)

        if table_name not in existing_tables:
            raise HTTPException(status_code=404, detail="Knowledge base is not indexed.")
        
        db = lancedb.connect(LANCEDB_PATH)
        table = db.open_table(table_name)
        df = table.to_pandas()
        
        if df.empty:
            return {
                "table_name": table_name,
                "summary": {"total_chunks": 0},
                "chunks": [],
                "total_returned": 0
            }
        
        # Apply document filter if provided
        if doc_filter:
            doc_ids = [d.strip() for d in doc_filter.split(",")]
            df = df[df["doc_id"].isin(doc_ids)]
        
        if "chunk_id" not in df.columns:
            df["chunk_id"] = range(len(df))
        
        df["chunk_id"] = df["chunk_id"].astype(int)
        df["length"] = df["text"].apply(len)
        
        # Get chunk params safely
        chunk_size = int(df.get("chunk_size", pd.Series([512])).iloc[0]) if "chunk_size" in df.columns else 512
        chunk_overlap = int(df.get("chunk_overlap", pd.Series([50])).iloc[0]) if "chunk_overlap" in df.columns else 50
        
        summary = {
            "total_chunks": int(len(df)),
            "avg_length": float(round(df['length'].mean(), 2)),
            "max_length": int(df['length'].max()),
            "min_length": int(df['length'].min()),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_overlap_ratio": float(round(chunk_overlap / chunk_size, 2)) if chunk_size > 0 else 0
        }
        
        if "doc_id" in df.columns:
            summary["unique_documents"] = int(df["doc_id"].nunique())
        
        # Return cleaned chunks
        raw_chunks = df[["chunk_id", "text", "length", "doc_id"]].head(limit).to_dict("records")
        chunks_data = [clean_chunk(chunk) for chunk in raw_chunks]
        
        return {
            "table_name": table_name,
            "summary": summary,
            "chunks": chunks_data,
            "total_returned": len(chunks_data),
            "filter_applied": doc_filter is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chunks: {str(e)}")

# Session Endpoints 
@app.post("/sessions")
def create_session(request: SessionCreateRequest):
    try:
        session_id = str(uuid.uuid4())
        table_name = "unified_knowledge_base"
        
        # Check if knowledge base exists
        existing_tables = get_existing_tables(LANCEDB_PATH)
        if table_name not in existing_tables:
            kb_docs = get_unified_knowledge_base_docs(LANCEDB_PATH)
            if not kb_docs:
                raise HTTPException(status_code=400, detail="Knowledge base is empty. Please upload and ingest documents first.")
        
        retriever = Retriever(db_path=LANCEDB_PATH, table_name=table_name)
        retrievers[session_id] = retriever
        
        # Enhanced session storage with configuration
        sessions[session_id] = {
            "table_name": table_name,
            "turn_id": 0,
            "summary": "",
            "created_at": time.time(),
            "last_activity": time.time(),
            "llm_model": request.llm_model,
            "retrieval_mode": request.retrieval_mode,
            "top_k_dense": request.top_k_dense,
            "top_k_sparse": request.top_k_sparse,
            "rrf_k": request.rrf_k,
            "top_k_final": request.top_k_final,
            "doc_filter": request.doc_filter,
            "latency_logs": []
        }
        
        return {
            "session_id": session_id,
            "table_name": table_name,
            "configuration": {
                "llm_model": request.llm_model,
                "retrieval_mode": request.retrieval_mode,
                "top_k_dense": request.top_k_dense,
                "top_k_sparse": request.top_k_sparse,
                "rrf_k": request.rrf_k,
                "top_k_final": request.top_k_final,
                "doc_filter": request.doc_filter
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")

@app.put("/sessions/{session_id}")
def update_session_config(session_id: str, request: SessionUpdateRequest):
    """Update session configuration"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    try:
        session_info = sessions[session_id]
        
        # Update only provided fields
        updates = {}
        if request.llm_model is not None:
            session_info["llm_model"] = request.llm_model
            updates["llm_model"] = request.llm_model
        if request.retrieval_mode is not None:
            session_info["retrieval_mode"] = request.retrieval_mode
            updates["retrieval_mode"] = request.retrieval_mode
        if request.top_k_dense is not None:
            session_info["top_k_dense"] = request.top_k_dense
            updates["top_k_dense"] = request.top_k_dense
        if request.top_k_sparse is not None:
            session_info["top_k_sparse"] = request.top_k_sparse
            updates["top_k_sparse"] = request.top_k_sparse
        if request.rrf_k is not None:
            session_info["rrf_k"] = request.rrf_k
            updates["rrf_k"] = request.rrf_k
        if request.top_k_final is not None:
            session_info["top_k_final"] = request.top_k_final
            updates["top_k_final"] = request.top_k_final
        if request.doc_filter is not None:
            session_info["doc_filter"] = request.doc_filter
            updates["doc_filter"] = request.doc_filter
        
        session_info["last_activity"] = time.time()
        
        return {
            "session_id": session_id,
            "updates_applied": updates,
            "current_configuration": {
                "llm_model": session_info["llm_model"],
                "retrieval_mode": session_info["retrieval_mode"],
                "top_k_dense": session_info["top_k_dense"],
                "top_k_sparse": session_info["top_k_sparse"],
                "rrf_k": session_info["rrf_k"],
                "top_k_final": session_info["top_k_final"],
                "doc_filter": session_info["doc_filter"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating session: {str(e)}")

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id in sessions:
        try:
            session_logger.log_session_summary(session_id)
            session_info = sessions.pop(session_id)
            retrievers.pop(session_id, None)
            
            return {
                "status": "deleted",
                "session_id": session_id,
                "total_turns": session_info["turn_id"],
                "session_duration": round(time.time() - session_info["created_at"], 2)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")
    else:
        raise HTTPException(status_code=404, detail="Session not found.")

@app.get("/sessions/{session_id}")
def get_session_info(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    try:
        session_info = sessions[session_id]
        buffer_df = session_logger.get_buffer(session_id)
        items_df = session_logger.get_items(session_id)
        
        return {
            "session_id": session_id,
            "session": session_info,
            "buffer": buffer_df.to_dict("records") if not buffer_df.empty else [],
            "items": items_df.to_dict("records") if not items_df.empty else [],
            "conversation_count": len(buffer_df) if not buffer_df.empty else 0,
            "configuration": {
                "llm_model": session_info.get("llm_model", "Groq"),
                "retrieval_mode": session_info.get("retrieval_mode", "hybrid"),
                "top_k_dense": session_info.get("top_k_dense", 10),
                "top_k_sparse": session_info.get("top_k_sparse", 10),
                "rrf_k": session_info.get("rrf_k", 60),
                "top_k_final": session_info.get("top_k_final", 10),
                "doc_filter": session_info.get("doc_filter", None)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session info: {str(e)}")

import numpy as np
import json
@app.get("/sessions/history/{session_id}")
def get_session_history(session_id: str):
    def clean_for_json(obj):
        """Recursively clean data structure for JSON serialization"""
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        elif isinstance(obj, (np.floating, float)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return clean_for_json(obj.tolist())
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    try:
        items_df = session_logger.get_items(session_id)
        buffer_df = session_logger.get_buffer(session_id)

        # Convert to dict and clean
        items_list = []
        if not items_df.empty:
            raw_items = items_df.to_dict("records")
            items_list = clean_for_json(raw_items)

        buffer_list = []
        if not buffer_df.empty:
            raw_buffer = buffer_df.to_dict("records")
            buffer_list = clean_for_json(raw_buffer)

        # Test JSON serialization before returning
        response_data = {
            "session_id": session_id,
            "items": items_list,
            "buffer": buffer_list,
            "total_items": len(items_list)
        }
        
        # Validate JSON serialization
        json.dumps(response_data)
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session history: {str(e)}")



@app.get("/sessions/{session_id}/goal-set")
def export_session_goal_set(session_id: str):
    """Export session as goal set for RAGAS evaluation"""
    try:
        items_df = session_logger.get_items(session_id)
        
        if items_df.empty:
            raise HTTPException(status_code=404, detail="No chat history found for this session.")
        
        # Load knowledge base for chunk mapping
        try:
            db = lancedb.connect(LANCEDB_PATH)
            table = db.open_table("unified_knowledge_base")
            kb_df = table.to_pandas()
            kb_df["chunk_id"] = kb_df["chunk_id"].astype(str)
        except Exception as e:
            kb_df = pd.DataFrame()
        
        # Prepare goal set records
        goal_set = []
        for idx, row in items_df.iterrows():
            # Parse chunk ids as list of strings
            chunk_ids = []
            if pd.notna(row.get("retrieved_chunk_ids", "")):
                chunk_ids = [cid.strip() for cid in str(row["retrieved_chunk_ids"]).split(",") if cid.strip().isdigit()]
            
            # Map to chunk texts
            retrieved_contexts = []
            if not kb_df.empty and chunk_ids:
                match = kb_df[kb_df["chunk_id"].isin(chunk_ids)]
                retrieved_contexts = match["text"].tolist()
            
            goal_set.append({
                "question": row.get("user_query", ""),
                "answer": row.get("bot_response", ""),
                "retrieved_contexts": retrieved_contexts,
                "reference": "",  # Add reference answer if available
                "turn_id": row.get("turn_id", idx),
                "session_id": session_id
            })
        
        return {
            "session_id": session_id,
            "goal_set": goal_set,
            "total_records": len(goal_set),
            "format": "RAGAS compatible"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting goal set: {str(e)}")

# Chat Endpoint
@app.post("/chat")
def chat(request: ChatRequest):
    if request.session_id not in sessions or request.session_id not in retrievers:
        raise HTTPException(status_code=404, detail="Session not found.")

    try:
        retriever = retrievers[request.session_id]
        session_info = sessions[request.session_id]

        # Use session defaults if parameters not provided in request
        llm_model = "Groq"
        retrieval_mode =  "hybrid"
        top_k_dense =  10
        top_k_sparse = 10
        rrf_k = 60
        top_k_final = 10
        doc_filter =  None

        cleaned_input = preprocess_text(request.query)
        past_turns_summary = session_info.get("summary", "")

        # Track timing
        query_times = {}
        total_start = time.perf_counter()

        def answer_with_llm(top_texts, query, model_type, past_summary, chunk_ids=None):
            return get_conversational_answer(
                top_texts, query, model_type, past_summary, chunk_ids=chunk_ids
            )

        model_type = "openai" if llm_model == "OpenAI" else "groq"
        route_result = route_and_answer(
            cleaned_input,
            past_turns_summary,
            retriever,
            model_type,
            answer_with_llm,
            retrieval_mode=retrieval_mode,
            k_dense=top_k_dense,
            k_sparse=top_k_sparse,
            rrf_k=rrf_k,
            top_k_final=top_k_final,
            doc_filter=doc_filter
        )

        # Extract results
        answer = route_result["answer"]
        intent_info = route_result["intent"]
        retrieval_results = route_result["retrieval_results"]
        retrieval_strength_val = route_result["retrieval_strength"]
        used_rag = route_result["used_rag"]
        answer_decision = route_result.get("answer_decision", "")
        top_scores_router = route_result.get("top_scores", [])

        # Add router/main LLM token/cost info to query_times
        for k in [
            "router_input_tokens", "router_output_tokens", "router_total_tokens", "router_cost",
            "main_input_tokens", "main_output_tokens", "main_total_tokens", "main_llm_cost"
        ]:
            query_times[k] = route_result.get(k, 0)

        # Timing info
        if "timing_info" in route_result:
            query_times.update(route_result["timing_info"])
        if "retrieval_timings" in route_result:
            query_times.update(route_result["retrieval_timings"])
        query_times["intent_routing_time"] = query_times.get("rule_routing_time", 0) + query_times.get("llm_routing_time", 0)

        # Prepare chunks information
        retrieved_chunks = []
        chunk_ids = []
        if used_rag and retrieval_results is not None:
            retrieved_chunks = retrieval_results.to_dict("records")
            chunk_ids = [int(cid) for cid in retrieval_results["chunk_id"] if cid is not None]

        # --- Maintain rolling turn history ---
        current_turn = f"User: {cleaned_input}\nAssistant: {answer}"
        if "turn_history" not in session_info:
            session_info["turn_history"] = []
        session_info["turn_history"].append(current_turn)
        if len(session_info["turn_history"]) > 5:
            session_info["turn_history"] = session_info["turn_history"][-5:]

        # --- Summarizer ---
        summary_result = summarizer(
            last_turns=session_info["turn_history"],
            past_summary=session_info.get("summary", "")
        )
        session_info["summary"] = summary_result["summary"]
        query_times["summarizer_input_tokens"] = summary_result.get("input_tokens", 0)
        query_times["summarizer_output_tokens"] = summary_result.get("output_tokens", 0)
        query_times["summarizer_total_tokens"] = summary_result.get("total_tokens", 0)
        query_times["summarizer_cost"] = summary_result.get("cost", 0.0)

        # --- Compute total cost for this turn
        query_times["total_turn_cost"] = (
            query_times.get("router_cost", 0)
            + query_times.get("main_llm_cost", 0)
            + query_times.get("summarizer_cost", 0)
        )
        query_times["total_time"] = time.perf_counter() - total_start

        # System metrics
        query_times["query"] = cleaned_input
        query_times["response"] = answer
        query_times["intent_labels"] = intent_info.get("labels", [])
        query_times["intent_confidence"] = intent_info.get("confidence", 0)
        query_times["slots"] = intent_info.get("slots", {})
        query_times["retrieval_strength"] = retrieval_strength_val
        query_times["retrieved_chunk_ids"] = chunk_ids
        query_times["used_rag"] = used_rag
        query_times["answer_decision"] = answer_decision

        # Update session state
        session_info["latency_logs"].append(query_times)
        session_info["turn_id"] += 1
        session_info["last_activity"] = time.time()

        # Log buffer and items
        session_logger.log_to_buffer(
            session_id=request.session_id,
            turn_id=session_info["turn_id"],
            user_query=cleaned_input,
            bot_response=answer
        )
        session_logger.log_to_items(
                session_id=request.session_id,
                turn_id=session_info["turn_id"],
                user_query=cleaned_input,
                bot_response=answer,
                summary=session_info["summary"],
                metrics=query_times,
                retrieval_type=retrieval_mode,
                llm_model=llm_model,
                top_k_dense=top_k_dense,
                top_k_sparse=top_k_sparse,
                top_k_combined=top_k_final,
            )

        return {
            "session_id": request.session_id,
            "turn_id": session_info["turn_id"],
            "query": cleaned_input,
            "answer": answer,
            "intent_info": intent_info,
            "retrieval_strength": retrieval_strength_val,
            "used_rag": used_rag,
            "answer_decision": answer_decision,
            "top_scores": top_scores_router,
            "timing_metrics": query_times,
            "retrieved_chunks": retrieved_chunks,
            "retrieved_chunk_ids": chunk_ids,
            "configuration_used": {
                "llm_model": llm_model,
                "retrieval_mode": retrieval_mode,
                "top_k_dense": top_k_dense,
                "top_k_sparse": top_k_sparse,
                "rrf_k": rrf_k,
                "top_k_final": top_k_final,
                "doc_filter": doc_filter
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/analytics/sessions/{session_id}/metrics")
def get_session_metrics(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    try:
        session_info = sessions[session_id]
        latency_logs = session_info.get("latency_logs", [])
        
        if not latency_logs:
            return {
                "session_id": session_id,
                "metrics": {},
                "message": "No metrics available yet. Start chatting to generate metrics."
            }
        
        # Calculate metrics
        df = pd.DataFrame(latency_logs)
        
        # Get retrieval mode specific columns
        mode = session_info.get("retrieval_mode", "hybrid")
        base_columns = ["intent_routing_time"]
        
        if mode == "hybrid":
            columns = base_columns + ["embedding_time", "dense_search_time", "sparse_search_time",
                       "fusion_time", "retrieval_time", "generation_time", "total_time"]
        elif mode == "dense":
            columns = base_columns + ["embedding_time", "dense_search_time", "retrieval_time", "generation_time", "total_time"]
        else:
            columns = base_columns + ["embedding_time", "sparse_search_time", "retrieval_time", "generation_time", "total_time"]
        
        # Calculate statistics
        latest = latency_logs[-1]
        avg_metrics = {}
        for col in columns:
            if col in df.columns:
                avg_metrics[col] = round(df[col].mean(), 3)
        
        # Calculate additional metrics
        total_queries = len(latency_logs)
        total_runtime_sec = sum(log.get("total_time", 0) for log in latency_logs)
        throughput_qps = round(total_queries / total_runtime_sec, 3) if total_runtime_sec > 0 else 0
        
        # Intent distribution
        all_intents = []
        for log in latency_logs:
            all_intents.extend(log.get("intent_labels", []))
        intent_distribution = pd.Series(all_intents).value_counts().to_dict() if all_intents else {}
        
        # Get top scores and format them
        top_scores = latest.get("top_scores", [])
        formatted_scores = [round(x, 3) for x in top_scores]
        avg_score = sum(top_scores) / len(top_scores) if top_scores else 0
        
        return {
            "session_id": session_id,
            "retrieval_mode": mode,
            "total_queries": total_queries,
            "throughput_qps": throughput_qps,
            "latest_query": {
                "intent_labels": latest.get("intent_labels", []),
                "intent_confidence": round(latest.get("intent_confidence", 0), 2),
                "retrieval_strength": round(latest.get("retrieval_strength", 0), 2),
                "used_rag": latest.get("used_rag", False),
                "top_scores": formatted_scores,
                "avg_retrieval_strength": round(avg_score, 4),
                "timing": {k: round(latest.get(k, 0), 3) for k in columns}
            },
            "average_metrics": avg_metrics,
            "intent_distribution": intent_distribution,
            "rag_usage": {
                "total_rag_queries": sum(1 for log in latency_logs if log.get("used_rag", False)),
                "rag_percentage": round((sum(1 for log in latency_logs if log.get("used_rag", False)) / total_queries * 100), 2) if total_queries > 0 else 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session metrics: {str(e)}")

@app.get("/analytics/system")
def get_system_metrics():
    """Get current system performance metrics"""
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        ram_total = memory.total / (1024 ** 3)
        ram_used = memory.used / (1024 ** 3)
        ram_available = memory.available / (1024 ** 3)
        ram_percent = memory.percent
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_total = disk.total / (1024 ** 3)
        disk_used = disk.used / (1024 ** 3)
        disk_free = disk.free / (1024 ** 3)
        disk_percent = disk.percent
        
        # Application metrics
        total_sessions = len(sessions)
        total_queries = sum(len(s.get("latency_logs", [])) for s in sessions.values())
        total_runtime_sec = sum(
            sum(log.get("total_time", 0) for log in s.get("latency_logs", []))
            for s in sessions.values()
        )
        overall_throughput = round(total_queries / total_runtime_sec, 3) if total_runtime_sec > 0 else 0
        
        return {
            "timestamp": time.time(),
            "system_metrics": {
                "cpu_usage_percent": round(cpu_percent, 2),
                "cpu_frequency_mhz": round(cpu_freq.current, 2) if cpu_freq else 0,
                "ram_total_gb": round(ram_total, 2),
                "ram_used_gb": round(ram_used, 2),
                "ram_available_gb": round(ram_available, 2),
                "ram_usage_percent": round(ram_percent, 2),
                "disk_total_gb": round(disk_total, 2),
                "disk_used_gb": round(disk_used, 2),
                "disk_free_gb": round(disk_free, 2),
                "disk_usage_percent": round(disk_percent, 2)
            },
            "application_metrics": {
                "active_sessions": total_sessions,
                "total_queries_processed": total_queries,
                "overall_throughput_qps": overall_throughput
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving system metrics: {str(e)}")

# --- Session Management Endpoints ---
@app.post("/sessions/{session_id}/end")
def end_session(session_id: str):
    try:
        session_logger.log_session_summary(session_id)
        if session_id in sessions:
            session_info = sessions[session_id]
            return {
                "message": "Session ended and summary logged.",
                "session_id": session_id,
                "total_turns": session_info.get("turn_id", 0),
                "duration_seconds": round(time.time() - session_info.get("created_at", time.time()), 2)
            }
        else:
            return {
                "message": "Session summary logged (session was not active).",
                "session_id": session_id
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ending session: {str(e)}")
    

    from botocore.exceptions import ClientError

@app.post("/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    try:
        # Parse S3 URI
        parsed = urlparse(DATA_DIR)
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        
        s3_client = boto3.client("s3")
        uploaded_files = []
        
        for file in files:
            if not file.filename or not file.filename.lower().endswith('.pdf'):
                uploaded_files.append({
                    "filename": file.filename or "unknown",
                    "status": "skipped",
                    "message": "Only PDF files are allowed"
                })
                continue
            
            content = await file.read()
            s3_key = f"{prefix}{file.filename}"
            
            try:
                # Check if file exists in S3
                s3_client.head_object(Bucket=bucket, Key=s3_key)
                uploaded_files.append({
                    "filename": file.filename,
                    "status": "exists",
                    "message": "File already exists in S3",
                    "s3_location": f"s3://{bucket}/{s3_key}"
                })
                continue
            except ClientError as e:
                if e.response['Error']['Code'] != '404':
                    uploaded_files.append({
                        "filename": file.filename,
                        "status": "error",
                        "message": f"S3 error: {str(e)}"
                    })
                    continue
                # File doesn't exist, proceed with upload
            
            try:
                # Upload to S3
                s3_client.put_object(
                    Bucket=bucket,
                    Key=s3_key,
                    Body=content,
                    ContentType='application/pdf'
                )
                
                uploaded_files.append({
                    "filename": file.filename,
                    "status": "uploaded",
                    "file_size_bytes": len(content),
                    "file_size_mb": round(len(content) / (1024 * 1024), 2),
                    "s3_location": f"s3://{bucket}/{s3_key}"
                })
            except Exception as upload_error:
                uploaded_files.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": f"Upload failed: {str(upload_error)}"
                })
        
        # Summary statistics
        total_uploaded = len([f for f in uploaded_files if f["status"] == "uploaded"])
        total_exists = len([f for f in uploaded_files if f["status"] == "exists"])
        total_errors = len([f for f in uploaded_files if f["status"] == "error"])
        total_skipped = len([f for f in uploaded_files if f["status"] == "skipped"])
        
        return {
            "status": "completed",
            "files": uploaded_files,
            "summary": {
                "total_files_processed": len(uploaded_files),
                "uploaded": total_uploaded,
                "already_exists": total_exists,
                "errors": total_errors,
                "skipped": total_skipped
            },
            "storage_type": "s3",
            "storage_location": DATA_DIR,
            "message": "Use /documents/ingest to add uploaded files to knowledge base"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading documents: {str(e)}")
