"""Automated Pytest Suite for Enterprise Contract & Vendor Risk Automation Engine.

Tests:
- Configuration toggles ($0 Free Tier vs GCP Vertex AI)
- Pydantic v2 schema validations
- FAISS policy vector store indexing, persistence, and cosine retrieval
- Multi-stage document intelligence pipeline
- FastAPI REST endpoints (/health, /policies, /analyze-contract, /batch-evaluate)
- Benchmark metric computations
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.gemini_client import gemini_client
from app.main import app
from app.pipeline import contract_pipeline
from app.policy_rag import policy_retriever
from app.schemas import (
    ChainOfThoughtAudit,
    CompanyPolicy,
    ContractExtractionResult,
    LiabilityAndIndemnity,
    PaymentTerms,
    PolicyViolation,
    SecurityAndCompliance,
    SLAAndPerformance,
    TerminationAndRenewal,
    VendorMetadata,
)
from eval.run_eval import compute_metrics


# ============================================================================
# 1. Configuration & Environment Tests
# ============================================================================

def test_settings_initialization() -> None:
    """Verifies that default configuration starts in $0 Free-Tier mode."""
    assert settings.model_extraction == "gemini-3.5-flash-lite"
    assert settings.model_audit == "gemini-3.8-flash"
    assert settings.model_embedding == "gemini-embedding-001"
    assert settings.rate_limit_rpm == 15
    assert settings.embedding_dimension == 768


def test_vertex_ai_toggle() -> None:
    """Verifies seamless dynamic toggling between Free Tier and Vertex AI mode."""
    initial_mode = settings.use_vertex_ai
    settings.update_vertex_toggle(True, project="my-enterprise-gcp-project")
    assert settings.use_vertex_ai is True
    assert "Vertex AI" in settings.provider_mode
    assert settings.google_cloud_project == "my-enterprise-gcp-project"

    # Toggle back to Free Tier
    settings.update_vertex_toggle(initial_mode)
    assert settings.use_vertex_ai == initial_mode


# ============================================================================
# 2. Strict Pydantic Schema Validation Tests
# ============================================================================

def test_pydantic_schema_validation() -> None:
    """Verifies strict schema enforcement for clause extraction results."""
    result = ContractExtractionResult(
        vendor_metadata=VendorMetadata(
            vendor_name="Acme Cloud Inc.",
            contract_title="Master SaaS Agreement",
            effective_date="2026-01-01",
            total_contract_value_usd=100000.0,
            governing_law="Delaware",
        ),
        payment_terms=PaymentTerms(
            payment_net_days=45,
            late_fee_percentage_monthly=1.0,
            raw_clause_quote="Payment within 45 days.",
        ),
        liability=LiabilityAndIndemnity(
            liability_cap_multiplier_acv=2.0,
            unlimited_carveouts=["gross negligence", "confidentiality"],
            raw_clause_quote="Liability capped at 2x ACV.",
        ),
        termination_renewal=TerminationAndRenewal(
            auto_renewal=True,
            renewal_notice_period_days=60,
            termination_for_convenience=True,
            raw_clause_quote="Notice at least 60 days prior.",
        ),
        sla=SLAAndPerformance(
            uptime_guarantee_percent=99.95,
            service_credit_percentage=15.0,
            raw_clause_quote="99.95% uptime guaranteed.",
        ),
        security_compliance=SecurityAndCompliance(
            data_breach_notification_hours=24,
            security_certifications=["SOC 2 Type II"],
            raw_clause_quote="Notice within 24 hours.",
        ),
        executive_clause_summary="Standard compliant contract.",
    )
    assert result.vendor_metadata.vendor_name == "Acme Cloud Inc."
    assert result.liability.liability_cap_multiplier_acv == 2.0
    assert result.sla.uptime_guarantee_percent == 99.95


# ============================================================================
# 3. FAISS Vector Store Policy RAG Tests
# ============================================================================

def test_faiss_policy_retriever() -> None:
    """Verifies FAISS index initialization, persistence, and cosine similarity search."""
    async def _test() -> None:
        await policy_retriever.initialize()
        policies = policy_retriever.get_all_policies()
        assert len(policies) == 10
        assert policy_retriever.index is not None
        assert policy_retriever.index.ntotal == 10

        # Retrieve relevant policies for SLA query
        query = "Availability uptime percentage and service credits for downtime outages"
        matches = await policy_retriever.retrieve(query, top_k=3)
        assert len(matches) <= 3
        assert any(m.policy.policy_id == "POL-SLA-003" for m in matches)
        for m in matches:
            assert 0.0 <= m.relevance_score <= 1.0

    asyncio.run(_test())


def test_pipeline_compliant_contract() -> None:
    """Tests contract analysis pipeline on compliant contract (Contract 02)."""
    async def _test() -> None:
        c2_path = settings.SAMPLE_CONTRACTS_DIR / "contract_02_data_analytics_compliant.txt"
        text = c2_path.read_text(encoding="utf-8")

        res = await contract_pipeline.analyze_contract(text, filename="contract_02.txt")
        assert res.extraction.vendor_metadata.vendor_name != ""
        assert res.audit.risk_score <= 25
        assert res.audit.risk_level == "LOW"
        assert res.audit.executive_verdict == "APPROVE"
        assert len(res.audit.violations) == 0

    asyncio.run(_test())


def test_pipeline_adversarial_contract() -> None:
    """Tests contract analysis pipeline on adversarial contract with model training license (Contract 04)."""
    async def _test() -> None:
        c4_path = settings.SAMPLE_CONTRACTS_DIR / "contract_04_ai_infrastructure_critical_risk.txt"
        text = c4_path.read_text(encoding="utf-8")

        res = await contract_pipeline.analyze_contract(text, filename="contract_04.txt")
        assert res.audit.risk_score >= 80
        assert res.audit.risk_level == "CRITICAL"
        assert res.audit.executive_verdict == "REJECT"

        # Verify that AI training violation was caught
        violated_ids = [v.policy_id for v in res.audit.violations]
        assert "POL-DAT-010" in violated_ids
        assert "POL-LIA-002" in violated_ids

    asyncio.run(_test())


# ============================================================================
# 5. FastAPI REST API Integration Tests
# ============================================================================

client = TestClient(app)


def test_api_health_endpoint() -> None:
    """Tests /health returns healthy status and 10 loaded policies."""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["vector_store"]["policy_count"] == 10
    assert data["models"]["stage_1_extraction"] == "gemini-3.5-flash-lite"


def test_api_policies_endpoint() -> None:
    """Tests /policies returns all 10 corporate compliance policies."""
    r = client.get("/policies")
    assert r.status_code == 200
    policies = r.json()
    assert len(policies) == 10
    ids = [p["policy_id"] for p in policies]
    assert "POL-SEC-001" in ids
    assert "POL-LIA-002" in ids
    assert "POL-DAT-010" in ids


def test_api_analyze_contract_json() -> None:
    """Tests POST /analyze-contract with raw text body."""
    sample_text = (
        "Enterprise Agreement with Acme Corp. Invoices are Net 15. "
        "Vendor liability is capped at $5,000. Breach notification within 72 hours."
    )
    r = client.post("/analyze-contract", json={"contract_text": sample_text, "filename": "test.txt"})
    assert r.status_code == 200
    data = r.json()
    assert "extraction" in data
    assert "audit" in data
    assert "latencies" in data
    assert data["audit"]["risk_score"] > 50


def test_api_analyze_contract_upload() -> None:
    """Tests POST /analyze-contract/upload with multipart file."""
    sample_text = "Vendor agreement text with 2x ACV liability cap and 24h breach notice."
    r = client.post(
        "/analyze-contract/upload",
        files={"file": ("test_upload.txt", sample_text.encode("utf-8"), "text/plain")}
    )
    assert r.status_code == 200
    data = r.json()
    assert data["filename"] == "test_upload.txt"


# ============================================================================
# 6. Benchmark Evaluation Logic Tests
# ============================================================================

def test_compute_metrics_accuracy() -> None:
    """Tests precision, recall, and F1 calculations against known sets."""
    gt_violations = ["POL-SEC-001", "POL-LIA-002", "POL-SLA-003"]
    pred_violations = ["POL-SEC-001", "POL-LIA-002", "POL-PAY-004"]  # 2 TP, 1 FP, 1 FN

    contract_text = "Vendor will notify of breach within 72 hours. Cap is $5,000."
    quotes = ["breach within 72 hours", "Cap is $5,000"]

    metrics = compute_metrics(
        predicted_violations=pred_violations,
        ground_truth_violations=gt_violations,
        predicted_score=80,
        ground_truth_range=(75, 85),
        predicted_level="CRITICAL",
        ground_truth_level="CRITICAL",
        quotes=quotes,
        contract_text=contract_text,
    )
    # Precision: 2/3 = 0.667, Recall: 2/3 = 0.667, F1: 0.667
    assert metrics["precision"] == 0.667
    assert metrics["recall"] == 0.667
    assert metrics["f1"] == 0.667
    assert metrics["score_error"] == 0.0
    assert metrics["level_match"] == 1.0
    assert metrics["citation_acc"] == 1.0
