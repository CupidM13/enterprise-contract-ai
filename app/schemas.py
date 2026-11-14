"""Strict Pydantic v2 schemas for contract clause extraction and risk auditing.

These schemas guarantee deterministic typed JSON outputs from both:
- gemini-3.5-flash-lite (Stage 1: Contract Extraction)
- gemini-3.8-flash (Stage 3: Chain-of-Thought Risk Audit)
"""

from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


# ============================================================================
# Stage 1: Contract Clause Extraction Schemas
# ============================================================================

class VendorMetadata(BaseModel):
    """Metadata regarding contracting parties and governing terms."""
    vendor_name: str = Field(..., description="Legal name of the vendor or service provider.")
    customer_name: Optional[str] = Field(None, description="Legal name of customer or enterprise entity.")
    contract_title: str = Field(..., description="Document or agreement title.")
    effective_date: Optional[str] = Field(None, description="Effective commencement date of agreement (YYYY-MM-DD or as stated).")
    contract_term_months: Optional[int] = Field(None, description="Initial term duration in months.")
    total_contract_value_usd: Optional[float] = Field(None, description="Total contract or estimated annual fee in USD.")
    governing_law: Optional[str] = Field(None, description="Governing law jurisdiction (e.g., Delaware, California, New York).")
    jurisdiction_venue: Optional[str] = Field(None, description="Agreed jurisdiction / legal dispute forum.")


class PaymentTerms(BaseModel):
    """Extracted payment, billing, and penalty terms."""
    payment_net_days: Optional[int] = Field(None, description="Payment term window in net days (e.g. 15, 30, 45, 60).")
    billing_cadence: Optional[str] = Field(None, description="Billing frequency: monthly, quarterly, annual upfront, or milestone.")
    late_fee_percentage_monthly: Optional[float] = Field(None, description="Monthly percentage interest/penalty charged on overdue invoices.")
    early_payment_discount: Optional[str] = Field(None, description="Discount terms for early invoice settlement (e.g., 2% Net 10).")
    raw_clause_quote: str = Field(..., description="Verbatim quote from the contract covering payment terms.")


class LiabilityAndIndemnity(BaseModel):
    """Extracted liability caps, exclusions, and indemnification obligations."""
    liability_cap_multiplier_acv: Optional[float] = Field(None, description="Cap expressed as multiplier of annual contract value (e.g. 1.0x, 2.0x, or None if fixed dollar amount).")
    liability_cap_fixed_usd: Optional[float] = Field(None, description="Fixed dollar liability cap amount in USD (if explicitly stated, e.g. 5000).")
    is_cap_unlimited: bool = Field(False, description="True if liability is completely uncapped.")
    unlimited_carveouts: List[str] = Field(default_factory=list, description="Carve-outs from cap (e.g., gross negligence, IP infringement, breach of confidentiality).")
    has_mutual_indemnity: bool = Field(False, description="Whether vendor provides mutual indemnification for IP infringement.")
    raw_clause_quote: str = Field(..., description="Verbatim quote defining limitation of liability and indemnification.")


class TerminationAndRenewal(BaseModel):
    """Extracted renewal, termination for convenience, and notice clauses."""
    auto_renewal: bool = Field(False, description="Whether the agreement automatically renews into subsequent terms.")
    renewal_notice_period_days: Optional[int] = Field(None, description="Days of advance notice required prior to renewal to prevent extension.")
    renewal_term_months: Optional[int] = Field(None, description="Length of renewal term in months if renewed.")
    termination_for_convenience: bool = Field(False, description="Whether customer has the contractual right to terminate without cause.")
    convenience_notice_days: Optional[int] = Field(None, description="Advance notice required in days to terminate for convenience.")
    raw_clause_quote: str = Field(..., description="Verbatim quote governing term, renewal, and termination.")


class SLAAndPerformance(BaseModel):
    """Extracted service level agreements, uptime guarantees, and remediation credits."""
    uptime_guarantee_percent: Optional[float] = Field(None, description="Guaranteed percentage uptime (e.g., 99.9, 99.95, 99.0).")
    service_credit_percentage: Optional[float] = Field(None, description="Maximum fee credit percentage granted to customer for SLA breach.")
    critical_incident_response_hours: Optional[float] = Field(None, description="Target response time in hours for Severity 1 outages.")
    has_scheduled_maintenance_exclusion: bool = Field(True, description="Whether scheduled maintenance windows are excluded from downtime.")
    raw_clause_quote: str = Field(..., description="Verbatim quote governing SLA performance and remedies.")


class SecurityAndCompliance(BaseModel):
    """Extracted data privacy, breach notification, audit rights, and security certifications."""
    data_breach_notification_hours: Optional[int] = Field(None, description="Maximum hours within which vendor must notify customer of a confirmed security incident/breach.")
    audit_rights_granted: bool = Field(False, description="Whether customer has the right to conduct independent security audits.")
    security_certifications: List[str] = Field(default_factory=list, description="Certified standards explicitly cited (e.g., SOC 2 Type II, ISO 27001, HIPAA, GDPR).")
    encryption_standards: Optional[str] = Field(None, description="Encryption cited for data at rest and in transit (e.g. AES-256, TLS 1.3).")
    subprocessor_advance_notice_days: Optional[int] = Field(None, description="Advance notice days provided to customer before engaging a new subprocessor.")
    customer_data_ownership_retained: bool = Field(True, description="Whether customer retains all proprietary rights and data ownership.")
    ai_training_on_customer_data_permitted: bool = Field(False, description="Whether vendor reserves the right to train AI/ML models on customer data.")
    raw_clause_quote: str = Field(..., description="Verbatim quote covering security, data handling, and confidentiality.")


class ContractExtractionResult(BaseModel):
    """Comprehensive structured extraction result produced by gemini-3.5-flash-lite."""
    vendor_metadata: VendorMetadata
    payment_terms: PaymentTerms
    liability: LiabilityAndIndemnity
    termination_renewal: TerminationAndRenewal
    sla: SLAAndPerformance
    security_compliance: SecurityAndCompliance
    executive_clause_summary: str = Field(..., description="2-3 sentence executive synopsis of key commercial and operational commitments.")
    extraction_model: str = Field("gemini-3.5-flash-lite", description="Model ID used for extraction.")


# ============================================================================
# Stage 2: Policy RAG Schemas
# ============================================================================

class CompanyPolicy(BaseModel):
    """Enterprise policy benchmark rule stored in corporate RAG index."""
    policy_id: str = Field(..., description="Unique policy identifier (e.g., POL-SEC-001).")
    title: str = Field(..., description="Human-readable policy title.")
    category: Literal["SLA", "LIABILITY", "PAYMENT", "DATA_PRIVACY_SECURITY", "TERMINATION", "GOVERNANCE"] = Field(...)
    rule_summary: str = Field(..., description="Concise statement of mandatory enterprise standard.")
    minimum_requirement: str = Field(..., description="Exact threshold or mandatory contractual condition.")
    severity_if_violated: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] = Field(...)
    redline_guidance: str = Field(..., description="Standard procurement guidance on required redline language.")


class RetrievedPolicyMatch(BaseModel):
    """A retrieved corporate policy paired with vector similarity score."""
    policy: CompanyPolicy
    relevance_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0).")


# ============================================================================
# Stage 3: Chain-of-Thought Risk Audit Schemas
# ============================================================================

class PolicyViolation(BaseModel):
    """Specific compliance violation or risk discrepancy detected during audit."""
    policy_id: str = Field(..., description="Identifier of the breached company policy (e.g., POL-SEC-001).")
    policy_title: str = Field(..., description="Title of breached policy.")
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] = Field(...)
    category: Literal["SLA", "LIABILITY", "PAYMENT", "DATA_PRIVACY_SECURITY", "TERMINATION", "GOVERNANCE"] = Field(...)
    contract_quote: str = Field(..., description="Verbatim quote directly cited from the contract exhibiting the violation.")
    violation_reason: str = Field(..., description="Explanation of why the contract term violates corporate policy.")
    recommended_redline: str = Field(..., description="Exact redline clause replacement suggested for legal procurement.")


class ChainOfThoughtAudit(BaseModel):
    """Deep reasoning risk audit produced by gemini-3.8-flash."""
    executive_summary: str = Field(..., description="Concise assessment of overall risk profile and deal posture.")
    risk_score: int = Field(..., ge=0, le=100, description="Overall risk score between 0 (safe/compliant) and 100 (unacceptable dealbreaker).")
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(...)
    executive_verdict: Literal["APPROVE", "APPROVE_WITH_CONDITIONS", "REVISE_AND_RESUBMIT", "REJECT"] = Field(...)
    auditor_reasoning_steps: List[str] = Field(..., description="Sequential chain-of-thought analysis steps examining clauses against policies.")
    violations: List[PolicyViolation] = Field(default_factory=list, description="List of all detected policy breaches.")
    compliant_policies: List[str] = Field(default_factory=list, description="IDs of policies that were verified as fully met.")
    key_negotiation_points: List[str] = Field(default_factory=list, description="Top strategic negotiation leverage points for legal counsel.")
    audit_model: str = Field("gemini-3.8-flash", description="Model ID used for deep reasoning audit.")


# ============================================================================
# End-to-End Pipeline & API Schemas
# ============================================================================

class ContractAnalysisRequest(BaseModel):
    """Request payload to analyze a vendor contract."""
    contract_text: str = Field(..., min_length=20, description="Full raw text of vendor contract agreement.")
    filename: Optional[str] = Field("uploaded_contract.txt", description="Name of contract file.")
    contract_id: Optional[str] = Field(None, description="Optional tracking identifier.")


class StageLatencies(BaseModel):
    """Per-stage execution latency metrics in milliseconds."""
    extraction_ms: float
    rag_retrieval_ms: float
    audit_ms: float
    total_ms: float


class ContractAnalysisResponse(BaseModel):
    """Complete end-to-end response returned by the BPA pipeline."""
    contract_id: str
    filename: str
    extraction: ContractExtractionResult
    retrieved_policies: List[Dict[str, Any]]
    audit: ChainOfThoughtAudit
    latencies: StageLatencies
    model_routing: Dict[str, str]
    timestamp: str


class BatchEvaluateRequest(BaseModel):
    """Request to batch evaluate multiple contracts with safe rate-limiting delay."""
    contracts: List[ContractAnalysisRequest]
    delay_between_calls_seconds: float = Field(4.0, ge=1.0, le=30.0, description="Inter-call delay to stay within free-tier RPM.")


class BatchEvaluateResponse(BaseModel):
    """Response containing batch evaluation results."""
    total_contracts: int
    successful_contracts: int
    results: List[ContractAnalysisResponse]
    total_batch_duration_seconds: float
