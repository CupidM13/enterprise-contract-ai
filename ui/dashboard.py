"""Streamlit Enterprise Dashboard ($0 Free-Tier Edition).

Interactive Web Application to:
- Upload or select vendor contracts
- Inspect structured clause extractions (gemini-3.5-flash-lite)
- View local FAISS Policy RAG vector matches (gemini-embedding-001)
- Audit risk scores, verbatim quote citations & redlines (gemini-3.8-flash)
- Monitor stage latencies and $0.00 free-tier quota telemetry
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.gemini_client import gemini_client
from app.pipeline import contract_pipeline
from app.policy_rag import policy_retriever
from app.schemas import ContractAnalysisResponse

# Page configuration
st.set_page_config(
    page_title="Enterprise Contract & Risk Automation ($0 Free Tier)",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .badge-free {
        background-color: #0d9488;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-vertex {
        background-color: #4f46e5;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .violation-card-critical {
        border-left: 5px solid #ef4444;
        background: #fef2f2;
        padding: 14px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .violation-card-high {
        border-left: 5px solid #f97316;
        background: #fff7ed;
        padding: 14px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .violation-card-medium {
        border-left: 5px solid #eab308;
        background: #fefce8;
        padding: 14px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .violation-card-low {
        border-left: 5px solid #3b82f6;
        background: #eff6ff;
        padding: 14px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .quote-box {
        font-family: monospace;
        background-color: #f1f5f9;
        border-left: 3px solid #64748b;
        padding: 8px 12px;
        font-size: 0.88rem;
        margin: 6px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Helper to run async code inside Streamlit
def run_async(coro):
    return asyncio.run(coro)


# ============================================================================
# Sidebar Configuration
# ============================================================================

st.sidebar.image("https://img.shields.io/badge/Google_AI_Studio-Free_Tier_$0-0d9488?style=for-the-badge&logo=google", use_container_width=True)
st.sidebar.title("⚙️ Engine Settings")

# Model Mode Switcher
st.sidebar.subheader("Environment Toggle")
provider_choice = st.sidebar.radio(
    "Active Environment:",
    options=["Google AI Studio (100% Free Tier)", "Google Cloud Vertex AI (Enterprise)"],
    index=1 if settings.use_vertex_ai else 0,
)

if provider_choice == "Google Cloud Vertex AI (Enterprise)":
    settings.use_vertex_ai = True
    gcp_project = st.sidebar.text_input("GCP Project ID:", value=settings.google_cloud_project or "")
    gcp_region = st.sidebar.text_input("GCP Location:", value=settings.google_cloud_location)
    if gcp_project:
        settings.update_vertex_toggle(True, project=gcp_project)
        gemini_client.refresh_client()
else:
    settings.use_vertex_ai = False
    api_key_input = st.sidebar.text_input(
        "Gemini API Key:",
        value=settings.gemini_api_key,
        type="password",
        help="Get a free Gemini API key from Google AI Studio (aistudio.google.com). No credit card required.",
    )
    if api_key_input != settings.gemini_api_key:
        settings.update_api_key(api_key_input)
        gemini_client.refresh_client()
        st.sidebar.success("API Key updated!")

st.sidebar.markdown("---")

# Model Routing Info
st.sidebar.subheader("🎯 Free-Tier Model Routing")
st.sidebar.markdown(f"**Stage 1 (Extraction):** `{settings.model_extraction}`")
st.sidebar.markdown(f"**Stage 2 (Embeddings):** `{settings.model_embedding}`")
st.sidebar.markdown(f"**Stage 3 (Risk Audit):** `{settings.model_audit}`")

# Vector Store Status
st.sidebar.markdown("---")
st.sidebar.subheader("🗂️ Local FAISS Policy RAG")
policies = policy_retriever.get_all_policies()
st.sidebar.info(f"Loaded **{len(policies)} corporate policies** into local FAISS vector index (`IndexFlatIP`). Disk-cached for $0 cloud storage.")

if st.sidebar.button("🔄 Force Re-Index Policies"):
    with st.spinner("Re-embedding policies with gemini-embedding-001..."):
        run_async(policy_retriever.initialize(force_reindex=True))
        st.sidebar.success("FAISS index rebuilt successfully!")

st.sidebar.markdown("---")
st.sidebar.caption("Enterprise Contract & Vendor Risk Automation Engine • $0 Free-Tier Edition • Production Ready")


# ============================================================================
# Main Header
# ============================================================================

col_head_1, col_head_2 = st.columns([3, 1])
with col_head_1:
    st.title("⚖️ Enterprise Contract & Vendor Risk Automation")
    st.markdown(
        "Production-grade **Business Process Automation (BPA)** & **Document Intelligence Pipeline** "
        "engineered to run **100% free with zero billing required**, with enterprise GCP Vertex AI compatibility."
    )
with col_head_2:
    st.markdown("<br>", unsafe_allow_html=True)
    if settings.use_vertex_ai:
        st.markdown('<span class="badge-vertex">VERTEX AI MODE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-free">100% FREE TIER ($0)</span>', unsafe_allow_html=True)


# ============================================================================
# Tabs Navigation
# ============================================================================

tab_audit, tab_benchmarks, tab_policies, tab_architecture = st.tabs(
    ["📋 Contract Risk Audit", "📊 LLMOps Benchmark", "🛡️ Corporate Policy Library", "🏗️ Free-Tier Architecture"]
)


# ============================================================================
# Tab 1: Interactive Contract Risk Audit
# ============================================================================

with tab_audit:
    st.subheader("1. Select or Upload Vendor Contract")

    # Sample contract selector or file upload
    sample_files = sorted(list(settings.SAMPLE_CONTRACTS_DIR.glob("contract_*.txt")))
    
    # Load ground truth metadata if available to show friendly names and difficulty
    gt_file = settings.SAMPLE_CONTRACTS_DIR / "ground_truth.json"
    gt_meta_map = {}
    if gt_file.exists():
        try:
            gt_data = json.loads(gt_file.read_text(encoding="utf-8"))
            for c in gt_data.get("contracts", []):
                gt_meta_map[c["contract_file"]] = f"[{c.get('difficulty', 'MODERATE')}] {c['vendor_name']} — {c.get('sector', 'Enterprise')} ({c['contract_file']})"
        except Exception:
            pass

    display_options = ["-- Upload Custom Contract --"]
    file_map = {"-- Upload Custom Contract --": None}

    for f in sample_files:
        label = gt_meta_map.get(f.name, f.name)
        display_options.append(label)
        file_map[label] = f

    col_input_1, col_input_2 = st.columns([2, 1])

    with col_input_1:
        selected_label = st.selectbox("Choose a synthetic vendor contract scenario (15 Enterprise Scenarios):", display_options, index=1 if sample_files else 0)

    contract_content = ""
    active_filename = "contract.txt"

    selected_file = file_map.get(selected_label)
    if selected_file is not None:
        contract_content = selected_file.read_text(encoding="utf-8")
        active_filename = selected_file.name
    else:
        uploaded_file = st.file_uploader("Upload contract (.txt, .md, .pdf):", type=["txt", "md", "pdf"])
        if uploaded_file:
            if uploaded_file.name.lower().endswith(".pdf"):
                try:
                    import io, pypdf
                    reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
                    contract_content = "\n\n".join([page.extract_text() or "" for page in reader.pages])
                    active_filename = uploaded_file.name
                    st.success(f"Successfully extracted {len(reader.pages)} pages from `{uploaded_file.name}` via pypdf!")
                except Exception as e:
                    st.error(f"Error parsing PDF: {e}")
            else:
                contract_content = uploaded_file.read().decode("utf-8", errors="replace")
                active_filename = uploaded_file.name

    with st.expander("📄 View Contract Raw Text", expanded=False):
        st.text_area("Contract Text", value=contract_content, height=220, disabled=True)

    # Analyze Button
    if st.button("🚀 Run Multi-Stage Risk Audit (Extract -> RAG -> Audit)", type="primary", use_container_width=True):
        if not contract_content.strip():
            st.error("Please provide or select a vendor contract to analyze.")
        else:
            with st.status("Executing Multi-Stage Intelligence Pipeline...", expanded=True) as status:
                st.write(f"⚡ **Stage 1**: Extracting commercial clauses via `{settings.model_extraction}` (Few-Shot Pydantic)...")
                t0 = time.perf_counter()

                # Run end-to-end pipeline
                analysis_response: ContractAnalysisResponse = run_async(
                    contract_pipeline.analyze_contract(
                        contract_text=contract_content,
                        filename=active_filename,
                    )
                )

                st.write(f"🔎 **Stage 2**: Querying local FAISS vector store via `{settings.model_embedding}` embeddings...")
                st.write(f"🧠 **Stage 3**: Conducting Chain-of-Thought risk audit via `{settings.model_audit}`...")
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

            # Store in session state
            st.session_state["latest_analysis"] = analysis_response

    # Display Analysis Results if available
    if "latest_analysis" in st.session_state:
        res: ContractAnalysisResponse = st.session_state["latest_analysis"]
        audit = res.audit
        extraction = res.extraction

        st.markdown("---")
        st.subheader("2. Executive Audit Verdict & Risk Score")

        # Top Executive Cards
        c1, c2, c3, c4 = st.columns(4)

        # Risk Score Color
        score = audit.risk_score
        if score >= 75:
            score_color = "#ef4444"  # Red
        elif score >= 50:
            score_color = "#f97316"  # Orange
        elif score >= 25:
            score_color = "#eab308"  # Yellow
        else:
            score_color = "#10b981"  # Green

        with c1:
            st.metric(
                label="Risk Score (0-100)",
                value=f"{audit.risk_score} / 100",
                delta=f"Severity: {audit.risk_level}",
                delta_color="inverse" if audit.risk_score > 25 else "normal",
            )

        with c2:
            st.metric(
                label="Executive Verdict",
                value=audit.executive_verdict.replace("_", " "),
            )

        with c3:
            st.metric(
                label="Total Policy Violations",
                value=len(audit.violations),
                delta=f"{len(audit.compliant_policies)} Verified Compliant",
            )

        with c4:
            st.metric(
                label="Total Pipeline Latency",
                value=f"{res.latencies.total_ms:.0f} ms",
                delta=f"${0.00:.2f} API Cost",
            )

        st.info(f"**Executive Summary:** {audit.executive_summary}")

        # Chain of Thought Reasoning Steps
        with st.expander("🧠 View Auditor Chain-of-Thought Reasoning Steps (gemini-3.8-flash)", expanded=False):
            for step in audit.auditor_reasoning_steps:
                st.markdown(f"- {step}")

        # Policy Violations & Redlines
        st.subheader(f"3. Policy Violations & Actionable Redlines ({len(audit.violations)})")

        if not audit.violations:
            st.success("🎉 Zero policy violations detected! This contract complies with all corporate standards.")
        else:
            for idx, v in enumerate(audit.violations, 1):
                card_class = f"violation-card-{v.severity.lower()}"
                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <h4>#{idx} [{v.severity}] {v.policy_id}: {v.policy_title}</h4>
                        <p><strong>Category:</strong> <code>{v.category}</code></p>
                        <p><strong>Violation Analysis:</strong> {v.violation_reason}</p>
                        <div class="quote-box"><strong>Verbatim Contract Quote Citation:</strong><br>"{v.contract_quote}"</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander(f"📝 View Recommended Redline for #{idx} ({v.policy_id})", expanded=True):
                    st.code(v.recommended_redline, language="markdown")

            # Generate downloadable executive legal redline report
            report_lines = [
                f"# Executive Contract Risk Audit Report: {extraction.vendor_metadata.vendor_name}",
                f"**Contract Document:** `{res.filename}` | **Generated:** `{res.timestamp}`",
                f"**Executive Verdict:** `{audit.executive_verdict}` | **Risk Score:** `{audit.risk_score} / 100` ({audit.risk_level})",
                f"**Annual Contract Value (ACV):** `${extraction.vendor_metadata.total_contract_value_usd:,.2f} USD`" if extraction.vendor_metadata.total_contract_value_usd else "",
                "",
                "## 1. Executive Summary",
                audit.executive_summary,
                "",
                "## 2. Key Negotiation Leverage Points",
                "\n".join([f"- {p}" for p in audit.key_negotiation_points]),
                "",
                "## 3. Detected Policy Violations & Actionable Redlines",
            ]
            for idx, v in enumerate(audit.violations, 1):
                report_lines.extend([
                    f"### #{idx}. [{v.severity}] {v.policy_id}: {v.policy_title} ({v.category})",
                    f"- **Non-Compliance Reason:** {v.violation_reason}",
                    f"> **Verbatim Contract Quote Citation:** \"{v.contract_quote}\"",
                    "",
                    "**Recommended Contract Redline:**",
                    "```markdown",
                    v.recommended_redline,
                    "```",
                    "",
                ])
            report_text = "\n".join(report_lines)

            st.download_button(
                label="📥 Download Executive Redline & Audit Report (.md)",
                data=report_text,
                file_name=f"{extraction.vendor_metadata.vendor_name.replace(' ', '_')}_redline_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # Extracted Structured Clauses
        st.subheader("4. Extracted Structured Clauses (gemini-3.5-flash-lite)")
        col_cl_1, col_cl_2 = st.columns(2)

        with col_cl_1:
            st.markdown("#### 🏢 Commercial & Metadata")
            st.json(extraction.vendor_metadata.model_dump())

            st.markdown("#### 💳 Payment & Penalties")
            st.json(extraction.payment_terms.model_dump())

            st.markdown("#### 🛡️ Liability & Indemnity")
            st.json(extraction.liability.model_dump())

        with col_cl_2:
            st.markdown("#### ⏱️ SLA & Performance")
            st.json(extraction.sla.model_dump())

            st.markdown("#### 🔄 Term & Renewal")
            st.json(extraction.termination_renewal.model_dump())

            st.markdown("#### 🔒 Security & Compliance")
            st.json(extraction.security_compliance.model_dump())

        # Stage Telemetry & Routing
        st.subheader("5. Pipeline Telemetry & Free-Tier Quota Tracking")
        tel_col_1, tel_col_2, tel_col_3, tel_col_4 = st.columns(4)
        tel_col_1.metric("Stage 1 (Extraction)", f"{res.latencies.extraction_ms:.1f} ms", f"Model: {res.model_routing['stage_1_extraction']}")
        tel_col_2.metric("Stage 2 (Policy RAG)", f"{res.latencies.rag_retrieval_ms:.1f} ms", "FAISS IndexFlatIP")
        tel_col_3.metric("Stage 3 (CoT Audit)", f"{res.latencies.audit_ms:.1f} ms", f"Model: {res.model_routing['stage_3_cot_audit']}")
        tel_col_4.metric("Total API Cost", "$0.00", "Google AI Studio Free Tier")


# ============================================================================
# Tab 2: LLMOps Benchmark & Eval Dashboard
# ============================================================================

with tab_benchmarks:
    st.subheader("LLMOps Rigorous Benchmark: Naive Zero-Shot vs Production Pipeline (15 Contracts)")
    st.markdown(
        "Evaluates **15 diverse enterprise contracts** across 4 difficulty tiers: "
        "`EASY`, `MODERATE`, `HARD_BURIED`, and `ADVERSARIAL`. "
        "The benchmark demonstrates why multi-stage Few-Shot extraction + local FAISS Policy RAG + CoT reasoning "
        "dramatically outperforms standard zero-shot prompting while operating at **$0.00 free-tier expense**."
    )

    col_btn_1, col_btn_2 = st.columns([1, 3])
    with col_btn_1:
        run_bench = st.button("▶️ Run 15-Contract Benchmark (Safe Throttled)", type="primary")

    if run_bench:
        with st.spinner("Executing throttled evaluation benchmark across all 15 contracts..."):
            from eval.run_eval import run_benchmark
            run_async(run_benchmark(throttle_delay_seconds=0.2))
            st.success("15-Contract Benchmark completed and saved to `eval/benchmark_results.json`!")

    # Load results from disk
    if settings.EVAL_RESULTS_PATH.exists():
        bench_data = json.loads(settings.EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
        summary = bench_data.get("summary", {})
        base = summary.get("naive_zero_shot", {})
        pipe = summary.get("production_pipeline", {})
        tiers = bench_data.get("difficulty_breakdown", {})
        scorecards = bench_data.get("contract_scorecards", [])

        st.markdown(f"### 1. Executive Performance Comparison ({bench_data.get('total_contracts_evaluated', 15)} Contracts)")

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Violation Detection F1", f"{pipe.get('f1', 0.0):.3f}", f"+{(pipe.get('f1', 0.0) - base.get('f1', 0.0))*100:.1f}% vs Zero-Shot")
        b2.metric("Violation Recall (Sensitivity)", f"{pipe.get('recall', 0.0):.3f}", f"+{(pipe.get('recall', 0.0) - base.get('recall', 0.0))*100:.1f}% vs Zero-Shot")
        b3.metric("Violation Precision", f"{pipe.get('precision', 0.0):.3f}", f"+{(pipe.get('precision', 0.0) - base.get('precision', 0.0))*100:.1f}% vs Zero-Shot")
        b4.metric("Quote Citation Acc.", f"{pipe.get('citation_accuracy', 0.0)*100:.1f}%", f"+{(pipe.get('citation_accuracy', 0.0) - base.get('citation_accuracy', 0.0))*100:.1f}% vs Zero-Shot")

        st.markdown("### 2. Overall Metric Breakdown")
        st.markdown(
            f"""
            | Benchmark Metric | Naive Zero-Shot Baseline | Production Pipeline (Few-Shot+RAG+CoT) | Improvement (Delta) |
            | :--- | :---: | :---: | :---: |
            | **Violation Detection F1 Score** | `{base.get('f1', 0):.3f}` | **`{pipe.get('f1', 0):.3f}`** | **`+{(pipe.get('f1', 0) - base.get('f1', 0))*100:.1f}%`** |
            | **Violation Recall (Sensitivity)** | `{base.get('recall', 0):.3f}` | **`{pipe.get('recall', 0):.3f}`** | **`+{(pipe.get('recall', 0) - base.get('recall', 0))*100:.1f}%`** |
            | **Violation Precision** | `{base.get('precision', 0):.3f}` | **`{pipe.get('precision', 0):.3f}`** | `+{(pipe.get('precision', 0) - base.get('precision', 0))*100:.1f}%` |
            | **Verbatim Quote Citation Acc.** | `{base.get('citation_accuracy', 0)*100:.1f}%` | **`{pipe.get('citation_accuracy', 0)*100:.1f}%`** | **`+{(pipe.get('citation_accuracy', 0) - base.get('citation_accuracy', 0))*100:.1f}%`** |
            | **Risk Level Classification Acc.** | `{base.get('level_accuracy', 0)*100:.1f}%` | **`{pipe.get('level_accuracy', 0)*100:.1f}%`** | **`+{(pipe.get('level_accuracy', 0) - base.get('level_accuracy', 0))*100:.1f}%`** |
            | **Risk Score MAE (Error)** | `{base.get('score_mae', 0):.1f} pts` | **`{pipe.get('score_mae', 0):.1f} pts`** | **`{base.get('score_mae', 0) - pipe.get('score_mae', 0):.1f} pts lower error`** |
            | **Mean Execution Latency** | `{base.get('avg_latency_ms', 0):.0f} ms` | `{pipe.get('avg_latency_ms', 0):.0f} ms` | Multi-stage orchestration |
            | **API Cloud Cost** | **`$0.00`** | **`$0.00`** | **100% Free Tier ($0.00)** |
            """
        )

        if tiers:
            st.markdown("### 3. Performance Breakdown by Difficulty Tier")
            tier_rows = []
            for t_name, t_data in tiers.items():
                gain = (t_data['pipeline_recall'] - t_data['baseline_recall']) * 100
                tier_rows.append(
                    f"| **{t_name}** | `{t_data['count']}` | `{t_data['baseline_f1']:.3f}` | **`{t_data['pipeline_f1']:.3f}`** | `{t_data['baseline_recall']:.3f}` | **`{t_data['pipeline_recall']:.3f}`** | **`+{gain:.1f}%`** |"
                )
            st.markdown(
                "| Difficulty Tier | Contracts | Baseline F1 | Pipeline F1 | Baseline Recall | Pipeline Recall | Recall Gain |\n"
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n" + "\n".join(tier_rows)
            )

        if scorecards:
            with st.expander("📋 View Individual Contract Scorecards (15 Contracts)", expanded=False):
                st.dataframe(scorecards, use_container_width=True)
    else:
        st.info("No benchmark results found on disk. Click 'Run 15-Contract Benchmark' above to generate the metrics.")


# ============================================================================
# Tab 3: Corporate Policy Library
# ============================================================================

with tab_policies:
    st.subheader(f"Enterprise Compliance Policies ({len(policies)} Rules)")
    st.markdown("These policies are embedded once with `gemini-embedding-001` into a local persistent FAISS index.")

    for p in policies:
        with st.expander(f"📌 [{p.severity_if_violated}] {p.policy_id}: {p.title}", expanded=False):
            st.markdown(f"**Category:** `{p.category}` | **Severity:** `{p.severity_if_violated}`")
            st.markdown(f"**Rule Summary:** {p.rule_summary}")
            st.markdown(f"**Minimum Requirement:** `{p.minimum_requirement}`")
            st.markdown(f"**Standard Redline Guidance:** {p.redline_guidance}")


# ============================================================================
# Tab 4: Free-Tier Architecture Blueprint
# ============================================================================

with tab_architecture:
    st.subheader("Production $0 Free-Tier Architecture & Model Routing")

    st.markdown(
        """
        ```mermaid
        flowchart TD
            A[Vendor Contract Text / PDF] --> B[Stage 1: Few-Shot Clause Extraction]
            B -->|Model: gemini-3.5-flash-lite| C[Strict Pydantic ContractExtractionResult]
            C --> D[Stage 2: Policy RAG Retrieval]
            E[(Local FAISS Index<br/>gemini-embedding-001<br/>data/faiss_index/)] -->|Zero Cloud DB Cost| D
            D -->|Top-K Matching Corporate Rules| F[Stage 3: Deep Chain-of-Thought Audit]
            F -->|Model: gemini-3.8-flash| G[Strict Pydantic ChainOfThoughtAudit]
            G --> H[Actionable Executive Deliverables]
            H --> H1[Risk Score 0-100 & Risk Level]
            H --> H2[Verbatim Contract Quote Citations]
            H --> H3[Procurement-Ready Legal Redlines]
            H --> H4[Executive Decision: APPROVE / REJECT]
        ```
        """
    )

    st.markdown("### Google AI Studio Free-Tier Quota Allocation")
    st.markdown(
        """
        | Pipeline Component | Free-Tier Model | RPM Quota | TPM Quota | RPD Quota | Free-Tier Cost |
        | :--- | :--- | :---: | :---: | :---: | :---: |
        | **Stage 1 (Extraction)** | `gemini-3.5-flash-lite` | 15 RPM | 1,000,000 TPM | 1,500 RPD | **$0.00** |
        | **Stage 2 (Embeddings)** | `gemini-embedding-001` | 15 RPM | 1,000,000 TPM | 1,500 RPD | **$0.00** |
        | **Stage 3 (CoT Audit)** | `gemini-3.8-flash` | 15 RPM | 1,000,000 TPM | 1,500 RPD | **$0.00** |
        | **Vector Database** | Local FAISS (`faiss-cpu`) | Unlimited | Unlimited | Unlimited | **$0.00** |
        """
    )
