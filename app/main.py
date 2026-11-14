"""FastAPI Server for Enterprise Contract & Vendor Risk Automation Engine ($0 Free-Tier Edition).

Provides RESTful endpoints for contract intelligence, multi-stage RAG risk auditing,
and throttled batch evaluations.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.gemini_client import gemini_client
from app.pipeline import contract_pipeline
from app.policy_rag import policy_retriever
from app.schemas import (
    BatchEvaluateRequest,
    BatchEvaluateResponse,
    CompanyPolicy,
    ContractAnalysisRequest,
    ContractAnalysisResponse,
)

logger = logging.getLogger("enterprise_contract_ai.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle manager."""
    logger.info("Initializing Enterprise Contract AI Pipeline...")
    logger.info(f"Active Provider Mode: {settings.provider_mode}")
    logger.info(f"Models: Extraction={settings.model_extraction}, Audit={settings.model_audit}, Embeddings={settings.model_embedding}")

    # Initialize or load persistent FAISS policy index
    try:
        await policy_retriever.initialize()
        logger.info(f"FAISS Policy Index ready with {len(policy_retriever.get_all_policies())} policies.")
    except Exception as e:
        logger.error(f"Failed to initialize FAISS policy store on startup: {e}")

    yield

    logger.info("Enterprise Contract AI Pipeline server shutting down cleanly.")


app = FastAPI(
    title="Enterprise Contract & Vendor Risk Automation Engine",
    description="Production-grade BPA and Document Intelligence pipeline operating on $0 Free-Tier Models with GCP Vertex AI compatibility.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Streamlit UI and enterprise web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
async def root() -> Dict[str, Any]:
    """Root info endpoint."""
    return {
        "service": "Enterprise Contract & Vendor Risk Automation Engine",
        "edition": "$0 Free-Tier Edition (Zero Billing Required)",
        "docs_url": "/docs",
        "health_url": "/health",
        "provider_mode": settings.provider_mode,
        "models": {
            "extraction": settings.model_extraction,
            "audit": settings.model_audit,
            "embedding": settings.model_embedding,
        },
    }


@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Health check verifying model configuration, vector store, and rate limiter status."""
    policy_count = len(policy_retriever.get_all_policies())
    return {
        "status": "healthy",
        "provider_mode": settings.provider_mode,
        "is_configured": settings.is_configured,
        "is_live_gemini": gemini_client.is_live,
        "vector_store": {
            "type": "Local FAISS (faiss-cpu)",
            "index_loaded": policy_retriever.index is not None,
            "policy_count": policy_count,
            "cost": "$0.00 (Persistent Disk Index)",
        },
        "models": {
            "stage_1_extraction": settings.model_extraction,
            "stage_2_embeddings": settings.model_embedding,
            "stage_3_audit": settings.model_audit,
        },
        "rate_limiter": {
            "rpm_target": settings.rate_limit_rpm,
            "min_interval_seconds": settings.rate_limit_min_interval,
            "max_retries": settings.max_retries,
        },
    }


@app.post("/analyze-contract", response_model=ContractAnalysisResponse, tags=["Contract Intelligence"])
async def analyze_contract(request: ContractAnalysisRequest) -> ContractAnalysisResponse:
    """Analyzes a vendor contract through the multi-stage BPA pipeline."""
    try:
        response = await contract_pipeline.analyze_contract(
            contract_text=request.contract_text,
            filename=request.filename or "contract.txt",
            contract_id=request.contract_id,
        )
        return response
    except Exception as e:
        logger.error(f"Error during contract analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Contract analysis pipeline error: {str(e)}",
        )


@app.post("/analyze-contract/upload", response_model=ContractAnalysisResponse, tags=["Contract Intelligence"])
async def upload_and_analyze_contract(file: UploadFile = File(...)) -> ContractAnalysisResponse:
    """Accepts a file upload (.txt, .md, .pdf), extracts text, and runs the analysis pipeline."""
    import io
    try:
        content_bytes = await file.read()
        filename_lower = (file.filename or "").lower()

        if filename_lower.endswith(".pdf"):
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(content_bytes))
                contract_text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
            except Exception as pdf_err:
                logger.error(f"Failed to extract text from PDF: {pdf_err}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Could not parse uploaded PDF document: {pdf_err}",
                )
        else:
            contract_text = content_bytes.decode("utf-8", errors="replace")

        if len(contract_text.strip()) < 20:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file contains insufficient text content to analyze.",
            )

        response = await contract_pipeline.analyze_contract(
            contract_text=contract_text,
            filename=file.filename or "uploaded_contract.txt",
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process uploaded contract file: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File processing error: {str(e)}",
        )


@app.post("/batch-evaluate", response_model=BatchEvaluateResponse, tags=["LLMOps Evaluation"])
async def batch_evaluate(request: BatchEvaluateRequest) -> BatchEvaluateResponse:
    """Batch evaluates multiple contracts sequentially with rate-limit throttling to prevent 429 errors."""
    start_time = time.perf_counter()
    results: List[ContractAnalysisResponse] = []
    successful = 0

    logger.info(f"Starting batch evaluation of {len(request.contracts)} contracts with {request.delay_between_calls_seconds}s interval...")

    for i, contract_req in enumerate(request.contracts):
        try:
            logger.info(f"Processing contract {i+1}/{len(request.contracts)}: '{contract_req.filename}'...")
            res = await contract_pipeline.analyze_contract(
                contract_text=contract_req.contract_text,
                filename=contract_req.filename or f"contract_{i+1}.txt",
                contract_id=contract_req.contract_id,
            )
            results.append(res)
            successful += 1
        except Exception as e:
            logger.error(f"Failed to analyze contract {contract_req.filename}: {e}")

        # Sleep between contracts in batch to strictly honor free-tier RPM
        if i < len(request.contracts) - 1:
            logger.info(f"Waiting {request.delay_between_calls_seconds}s before next contract in batch...")
            await asyncio.sleep(request.delay_between_calls_seconds)

    total_duration = time.perf_counter() - start_time
    return BatchEvaluateResponse(
        total_contracts=len(request.contracts),
        successful_contracts=successful,
        results=results,
        total_batch_duration_seconds=round(total_duration, 2),
    )


@app.get("/policies", response_model=List[CompanyPolicy], tags=["Compliance Policies"])
async def list_policies() -> List[CompanyPolicy]:
    """Returns the list of 10 corporate compliance policies indexed in the vector store."""
    return policy_retriever.get_all_policies()


@app.post("/reindex-policies", tags=["Compliance Policies"])
async def reindex_policies() -> Dict[str, Any]:
    """Forces re-indexing of corporate policies in FAISS."""
    try:
        await policy_retriever.initialize(force_reindex=True)
        return {
            "status": "success",
            "message": f"Successfully re-indexed {len(policy_retriever.get_all_policies())} policies into local FAISS vector store.",
            "index_size": policy_retriever.index.ntotal if policy_retriever.index else 0,
        }
    except Exception as e:
        logger.error(f"Failed to re-index policies: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Re-indexing error: {str(e)}",
        )
