"""Multi-Stage Document Intelligence and Risk Automation Pipeline.

Orchestrates:
- Stage 1: Fast Clause Extraction using gemini-3.5-flash-lite (Few-Shot + Pydantic schema)
- Stage 2: Policy RAG retrieval via local FAISS Vector Store (zero cloud vector DB cost)
- Stage 3: Deep Chain-of-Thought Risk Audit using gemini-3.8-flash (citations + redlines)
"""

from __future__ import annotations

import datetime
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from app.config import settings
from app.gemini_client import gemini_client
from app.policy_rag import policy_retriever
from app.schemas import (
    ChainOfThoughtAudit,
    ContractAnalysisResponse,
    ContractExtractionResult,
    StageLatencies,
)

logger = logging.getLogger("enterprise_contract_ai.pipeline")


# ============================================================================
# Prompts & System Instructions
# ============================================================================

EXTRACTION_SYSTEM_INSTRUCTION = """You are a senior enterprise procurement counsel and contract intelligence AI.
Your objective is to thoroughly analyze the provided vendor contract agreement and extract structured clauses with extreme precision.

Follow these strict extraction principles:
1. ACCURACY: Extract exact numbers (fees, net payment days, liability multipliers, uptime percentages, breach notification hours, renewal notice days).
2. QUOTES: In every clause category, provide the EXACT verbatim quote from the contract text as evidence.
3. COMPLETENESS: Fill all fields in the schema. If a term is silent or omitted, indicate appropriate defaults (e.g., auto_renewal=False, termination_for_convenience=False).
"""

FEW_SHOT_EXTRACTION_PROMPT = """Analyze the following vendor contract agreement and extract all structured commercial, liability, SLA, termination, and security clauses according to the required schema.

### FEW-SHOT REFERENCE EXAMPLE:
Input Contract Snippet:
"Provider agrees to host the ACME Suite for $100,000/yr. Invoices are payable within 30 days (Net 30) with 1.5% late fee per month. This Agreement auto-renews for 12-month periods unless notice is given 30 days prior. Uptime is 99.5% with 5% credit. Provider liability shall not exceed 1x annual fees ($100k). Provider shall notify customer of breach within 48 hours. Governed by New York law."

Expected Extraction Guidance:
- vendor_name: "Provider"
- total_contract_value_usd: 100000.0
- payment_net_days: 30
- late_fee_percentage_monthly: 1.5
- auto_renewal: True
- renewal_notice_period_days: 30
- uptime_guarantee_percent: 99.5
- liability_cap_multiplier_acv: 1.0
- data_breach_notification_hours: 48
- governing_law: "New York"
- raw_clause_quote: Extract actual contract sentence for each section.

### VENDOR CONTRACT TO ANALYZE:
{contract_text}
"""

AUDIT_SYSTEM_INSTRUCTION = """You are an elite Enterprise General Counsel and Chief Risk Officer.
Your objective is to conduct a rigorous, deterministic Chain-of-Thought risk audit of a vendor contract by cross-referencing extracted contract clauses and contract text against the Enterprise Corporate Policies.

EVALUATION METHODOLOGY:
1. CHAIN-OF-THOUGHT REASONING:
   - For every corporate policy, examine whether the contract satisfies, exceeds, or violates the mandatory requirement.
   - Record your analytical reasoning step-by-step in `auditor_reasoning_steps`.

2. VERBATIM QUOTE CITATIONS:
   - For every detected violation, cite the EXACT verbatim quote from the contract text illustrating the non-compliant term.
   - Do NOT paraphrase quotes; extract the actual contract phrasing.

3. DETERMINISTIC RISK SCORING (0 to 100):
   - Baseline clean score: 0-15 (fully compliant or trivial formatting notes).
   - MEDIUM risk violation (e.g. Net 30 vs Net 45, missing convenience clause): +8 to +12 points each.
   - HIGH risk violation (e.g. 1.0x ACV liability, 48h breach notice, 99.0% SLA, SOC 2 Type I): +15 to +20 points each.
   - CRITICAL risk violation (e.g. liability cap < $10k or $100, AI training on customer data, 72h+ or 5-day breach notice, offshore venue): +25 to +35 points each.
   - Risk Levels:
     * 0 - 25: LOW (Verdict: APPROVE)
     * 26 - 50: MEDIUM (Verdict: REVISE_AND_RESUBMIT or APPROVE_WITH_CONDITIONS)
     * 51 - 75: HIGH (Verdict: REVISE_AND_RESUBMIT)
     * 76 - 100: CRITICAL (Verdict: REJECT)

4. ACTIONABLE REDLINES:
   - Provide explicit, turnkey redline clauses ready for procurement counsel to paste into the contract mark-up.
"""

AUDIT_PROMPT_TEMPLATE = """Evaluate this vendor contract against the retrieved corporate policies.

### EXTRACTED CONTRACT SUMMARY & CLAUSES:
- Vendor: {vendor_name}
- Total Annual Value: ${total_value} USD
- Governing Law: {governing_law}
- Payment Terms: Net {net_days} days | Late Fee: {late_fee}% | Discount: {discount}
- Liability Cap: Multiplier={liability_multiplier}x ACV | Fixed=${liability_fixed} USD | Unlimited Exclusions={carveouts}
- Term & Renewal: Auto-renewal={auto_renew} | Notice={renewal_notice} days | Convenience Termination={convenience}
- SLA: Uptime={uptime}% | Service Credits={credits}%
- Security & Compliance: Breach Notice={breach_hours} hours | Audits={audits} | Certs={certs} | AI Model Training Permitted={ai_training}

### FULL CONTRACT TEXT (FOR EXACT VERBATIM CITATIONS):
{contract_text}

### RETRIEVED ENTERPRISE POLICIES:
{policies_context}

Execute your step-by-step Chain-of-Thought audit and return the complete structured ChainOfThoughtAudit.
"""


# ============================================================================
# Multi-Stage Pipeline Orchestrator
# ============================================================================

class ContractAnalysisPipeline:
    """Enterprise Contract BPA & Risk Automation Pipeline."""

    def __init__(self) -> None:
        self.client = gemini_client
        self.retriever = policy_retriever

    async def analyze_contract(
        self,
        contract_text: str,
        filename: str = "vendor_contract.txt",
        contract_id: Optional[str] = None,
    ) -> ContractAnalysisResponse:
        """Executes the complete multi-stage analysis pipeline.

        Stage 1: Extract clauses using gemini-3.5-flash-lite
        Stage 2: Retrieve relevant corporate policies from FAISS
        Stage 3: Deep Chain-of-Thought risk audit using gemini-3.8-flash
        """
        contract_id = contract_id or f"CTR-{uuid.uuid4().hex[:8].upper()}"
        pipeline_start = time.perf_counter()

        logger.info(f"[{contract_id}] Starting Contract Intelligence Pipeline for '{filename}' ({len(contract_text)} chars)...")

        # --------------------------------------------------------------------
        # Stage 1: Fast Clause Extraction (gemini-3.5-flash-lite)
        # --------------------------------------------------------------------
        t0 = time.perf_counter()
        extraction_prompt = FEW_SHOT_EXTRACTION_PROMPT.format(contract_text=contract_text)

        extraction_result: ContractExtractionResult = await self.client.generate_structured(
            model=settings.model_extraction,
            prompt=extraction_prompt,
            schema=ContractExtractionResult,
            system_instruction=EXTRACTION_SYSTEM_INSTRUCTION,
        )
        extraction_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"[{contract_id}] Stage 1 (Extraction) finished in {extraction_ms:.1f}ms via {settings.model_extraction}")

        # --------------------------------------------------------------------
        # Stage 2: Policy RAG Retrieval (Local FAISS Index)
        # --------------------------------------------------------------------
        t1 = time.perf_counter()

        # Build query representing key extracted commercial risks
        rag_query = (
            f"Vendor {extraction_result.vendor_metadata.vendor_name}. "
            f"Payment Net {extraction_result.payment_terms.payment_net_days} days. "
            f"Liability cap multiplier {extraction_result.liability.liability_cap_multiplier_acv}x fixed ${extraction_result.liability.liability_cap_fixed_usd}. "
            f"SLA uptime {extraction_result.sla.uptime_guarantee_percent}%. "
            f"Breach notification {extraction_result.security_compliance.data_breach_notification_hours} hours. "
            f"Auto renewal {extraction_result.termination_renewal.auto_renewal} notice {extraction_result.termination_renewal.renewal_notice_period_days} days. "
            f"Governing law {extraction_result.vendor_metadata.governing_law}."
        )

        # Retrieve top relevant policies
        retrieved_matches = await self.retriever.retrieve(query_text=rag_query, top_k=settings.rag_top_k)

        # For enterprise compliance completeness, ensure all 10 corporate policies are available during audit
        all_policies = self.retriever.get_all_policies()
        rag_retrieval_ms = (time.perf_counter() - t1) * 1000.0
        logger.info(f"[{contract_id}] Stage 2 (Policy RAG) retrieved {len(retrieved_matches)} targeted policies in {rag_retrieval_ms:.1f}ms")

        # Format policies context block for Stage 3
        policies_context_blocks = []
        for p in all_policies:
            policies_context_blocks.append(
                f"- Policy ID: {p.policy_id}\n"
                f"  Title: {p.title} [{p.category}]\n"
                f"  Standard Rule: {p.rule_summary}\n"
                f"  Mandatory Requirement: {p.minimum_requirement}\n"
                f"  Severity if Breached: {p.severity_if_violated}\n"
                f"  Redline Guidance: {p.redline_guidance}"
            )
        policies_context_str = "\n\n".join(policies_context_blocks)

        # --------------------------------------------------------------------
        # Stage 3: Chain-of-Thought Risk Audit (gemini-3.8-flash)
        # --------------------------------------------------------------------
        t2 = time.perf_counter()
        audit_prompt = AUDIT_PROMPT_TEMPLATE.format(
            vendor_name=extraction_result.vendor_metadata.vendor_name,
            total_value=extraction_result.vendor_metadata.total_contract_value_usd or "Unspecified",
            governing_law=extraction_result.vendor_metadata.governing_law or "Unspecified",
            net_days=extraction_result.payment_terms.payment_net_days or "Unspecified",
            late_fee=extraction_result.payment_terms.late_fee_percentage_monthly or "0.0",
            discount=extraction_result.payment_terms.early_payment_discount or "None",
            liability_multiplier=extraction_result.liability.liability_cap_multiplier_acv or "None",
            liability_fixed=extraction_result.liability.liability_cap_fixed_usd or "None",
            carveouts=", ".join(extraction_result.liability.unlimited_carveouts) if extraction_result.liability.unlimited_carveouts else "None",
            auto_renew=extraction_result.termination_renewal.auto_renewal,
            renewal_notice=extraction_result.termination_renewal.renewal_notice_period_days or "Unspecified",
            convenience=extraction_result.termination_renewal.termination_for_convenience,
            uptime=extraction_result.sla.uptime_guarantee_percent or "None",
            credits=extraction_result.sla.service_credit_percentage or "None",
            breach_hours=extraction_result.security_compliance.data_breach_notification_hours or "Unspecified",
            audits=extraction_result.security_compliance.audit_rights_granted,
            certs=", ".join(extraction_result.security_compliance.security_certifications) if extraction_result.security_compliance.security_certifications else "None",
            ai_training=extraction_result.security_compliance.ai_training_on_customer_data_permitted,
            contract_text=contract_text,
            policies_context=policies_context_str,
        )

        audit_result: ChainOfThoughtAudit = await self.client.generate_structured(
            model=settings.model_audit,
            prompt=audit_prompt,
            schema=ChainOfThoughtAudit,
            system_instruction=AUDIT_SYSTEM_INSTRUCTION,
        )
        audit_ms = (time.perf_counter() - t2) * 1000.0
        logger.info(f"[{contract_id}] Stage 3 (Audit) completed in {audit_ms:.1f}ms via {settings.model_audit}. Score={audit_result.risk_score}")

        total_ms = (time.perf_counter() - pipeline_start) * 1000.0

        # Construct final response
        response = ContractAnalysisResponse(
            contract_id=contract_id,
            filename=filename,
            extraction=extraction_result,
            retrieved_policies=[
                {
                    "policy_id": m.policy.policy_id,
                    "title": m.policy.title,
                    "category": m.policy.category,
                    "relevance_score": m.relevance_score,
                }
                for m in retrieved_matches
            ],
            audit=audit_result,
            latencies=StageLatencies(
                extraction_ms=round(extraction_ms, 2),
                rag_retrieval_ms=round(rag_retrieval_ms, 2),
                audit_ms=round(audit_ms, 2),
                total_ms=round(total_ms, 2),
            ),
            model_routing={
                "stage_1_extraction": settings.model_extraction,
                "stage_2_vector_rag": f"{settings.model_embedding} + FAISS IndexFlatIP",
                "stage_3_cot_audit": settings.model_audit,
            },
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

        return response


# Global pipeline singleton
contract_pipeline = ContractAnalysisPipeline()
