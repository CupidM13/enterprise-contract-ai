"""Streamlit Enterprise Dashboard ($0 Free-Tier Edition).

Interactive Web Application to:
- Upload or select vendor contracts across 4 enterprise difficulty tiers
- Inspect structured clause extractions (gemini-3.5-flash-lite)
- View local FAISS Policy RAG vector matches (gemini-embedding-001)
- Audit risk scores, verbatim quote citations & redlines (gemini-3.8-flash)
- Explore interactive Plotly risk dials, benchmark comparisons & telemetry
- Download executive Markdown redline reports and structured JSON audits
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
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

# Custom Styling (Theme-Adaptive for Light and Dark modes)
st.markdown(
    """
    <style>
    /* Global Card & Container Styling */
    .app-card {
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 16px;
        transition: all 0.2s ease-in-out;
    }
    .app-card:hover {
        border-color: rgba(13, 148, 136, 0.5);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    /* Badges */
    .badge-free {
        background-color: #0d9488;
        color: #ffffff !important;
        padding: 5px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
        letter-spacing: 0.5px;
    }
    .badge-vertex {
        background-color: #4f46e5;
        color: #ffffff !important;
        padding: 5px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
        letter-spacing: 0.5px;
    }
    .badge-tier-easy {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-tier-mod {
        background-color: rgba(234, 179, 8, 0.15);
        color: #eab308;
        border: 1px solid #eab308;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-tier-hard {
        background-color: rgba(249, 115, 22, 0.15);
        color: #f97316;
        border: 1px solid #f97316;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-tier-adv {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Violation Cards with Left Accent */
    .violation-card-critical {
        border-left: 5px solid #ef4444;
        background: rgba(239, 68, 68, 0.08);
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 14px;
        border-top: 1px solid rgba(239, 68, 68, 0.2);
        border-right: 1px solid rgba(239, 68, 68, 0.2);
        border-bottom: 1px solid rgba(239, 68, 68, 0.2);
    }
    .violation-card-high {
        border-left: 5px solid #f97316;
        background: rgba(249, 115, 22, 0.08);
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 14px;
        border-top: 1px solid rgba(249, 115, 22, 0.2);
        border-right: 1px solid rgba(249, 115, 22, 0.2);
        border-bottom: 1px solid rgba(249, 115, 22, 0.2);
    }
    .violation-card-medium {
        border-left: 5px solid #eab308;
        background: rgba(234, 179, 8, 0.08);
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 14px;
        border-top: 1px solid rgba(234, 179, 8, 0.2);
        border-right: 1px solid rgba(234, 179, 8, 0.2);
        border-bottom: 1px solid rgba(234, 179, 8, 0.2);
    }
    .violation-card-low {
        border-left: 5px solid #3b82f6;
        background: rgba(59, 130, 246, 0.08);
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 14px;
        border-top: 1px solid rgba(59, 130, 246, 0.2);
        border-right: 1px solid rgba(59, 130, 246, 0.2);
        border-bottom: 1px solid rgba(59, 130, 246, 0.2);
    }

    /* Verbatim Quote Highlight Callout */
    .quote-box {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        background: rgba(0, 0, 0, 0.05);
        border-left: 4px solid #64748b;
        padding: 10px 14px;
        font-size: 0.88rem;
        margin: 10px 0;
        border-radius: 0 6px 6px 0;
        line-height: 1.45;
    }

    /* Quick Guide Banner */
    .workflow-guide {
        background: linear-gradient(90deg, rgba(13, 148, 136, 0.12), rgba(79, 70, 229, 0.12));
        border: 1px solid rgba(13, 148, 136, 0.25);
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 20px;
        font-size: 0.92rem;
    }

    /* Clause Grid Card */
    .clause-card {
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .clause-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .clause-value {
        font-size: 1.05rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Helper to run async coroutines safely inside Streamlit
def run_async(coro):
    return asyncio.run(coro)


# Helper to render a Plotly Risk Gauge Meter
def create_risk_gauge(score: int, level: str) -> go.Figure:
    """Generate an executive half-donut risk gauge with color zones."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"<b>Contract Risk Dial</b><br><span style='font-size:0.85em;color:#64748b'>Severity: {level}</span>", "font": {"size": 18}},
            number={"suffix": " / 100", "font": {"size": 32, "weight": "bold"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": "#1e293b", "thickness": 0.28},
                "steps": [
                    {"range": [0, 25], "color": "rgba(16, 185, 129, 0.4)"},   # Low (Green)
                    {"range": [25, 50], "color": "rgba(234, 179, 8, 0.4)"},   # Medium (Yellow)
                    {"range": [50, 75], "color": "rgba(249, 115, 22, 0.4)"},  # High (Orange)
                    {"range": [75, 100], "color": "rgba(239, 68, 68, 0.5)"}, # Critical (Red)
                ],
                "threshold": {
                    "line": {"color": "#ef4444" if score >= 75 else "#f97316" if score >= 50 else "#10b981", "width": 4},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=25, r=25, t=40, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, sans-serif"),
    )
    return fig


# Helper to render Benchmark Bar Charts
def create_benchmark_bar_chart(base: dict, pipe: dict) -> go.Figure:
    """Bar chart comparing baseline vs pipeline."""
    metrics = ["Violation F1", "Recall (Sensitivity)", "Precision", "Quote Citation"]
    base_vals = [
        base.get("f1", 0.0) * 100,
        base.get("recall", 0.0) * 100,
        base.get("precision", 0.0) * 100,
        base.get("citation_accuracy", 0.0) * 100,
    ]
    pipe_vals = [
        pipe.get("f1", 0.0) * 100,
        pipe.get("recall", 0.0) * 100,
        pipe.get("precision", 0.0) * 100,
        pipe.get("citation_accuracy", 0.0) * 100,
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=metrics,
            y=base_vals,
            name="Naive Zero-Shot",
            marker_color="#94a3b8",
            text=[f"{v:.1f}%" for v in base_vals],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            x=metrics,
            y=pipe_vals,
            name="Production Pipeline (Few-Shot + RAG + CoT)",
            marker_color="#0d9488",
            text=[f"{v:.1f}%" for v in pipe_vals],
            textposition="auto",
        )
    )
    fig.update_layout(
        barmode="group",
        title="<b>Production Pipeline vs. Naive Zero-Shot Performance (%)</b>",
        yaxis=dict(title="Score (%)", range=[0, 115]),
        height=360,
        margin=dict(l=20, r=20, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_tier_breakdown_chart(tiers: dict) -> go.Figure:
    """Grouped bar chart showing recall breakdown across difficulty tiers."""
    categories = list(tiers.keys())
    base_recalls = [tiers[t]["baseline_recall"] * 100 for t in categories]
    pipe_recalls = [tiers[t]["pipeline_recall"] * 100 for t in categories]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=categories,
            y=base_recalls,
            name="Zero-Shot Recall",
            marker_color="#cbd5e1",
            text=[f"{v:.1f}%" for v in base_recalls],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            x=categories,
            y=pipe_recalls,
            name="Pipeline Recall",
            marker_color="#6366f1",
            text=[f"{v:.1f}%" for v in pipe_recalls],
            textposition="auto",
        )
    )
    fig.update_layout(
        barmode="group",
        title="<b>Recall Across Difficulty Tiers: The Buried-Clause Blind Spot</b>",
        yaxis=dict(title="Recall (%)", range=[0, 115]),
        height=340,
        margin=dict(l=20, r=20, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ============================================================================
# Sidebar Configuration
# ============================================================================

st.sidebar.image("https://img.shields.io/badge/Google_AI_Studio-Free_Tier_$0-0d9488?style=for-the-badge&logo=google", use_container_width=True)
st.sidebar.title("⚙️ Engine Control Panel")

# Environment Switcher
st.sidebar.subheader("🌐 Cloud Environment")
provider_choice = st.sidebar.radio(
    "Active Provider:",
    options=["Google AI Studio (100% Free Tier)", "Google Cloud Vertex AI (Enterprise)"],
    index=1 if settings.use_vertex_ai else 0,
    help="Toggle between 100% free Google AI Studio and enterprise GCP Vertex AI.",
)

if provider_choice == "Google Cloud Vertex AI (Enterprise)":
    settings.use_vertex_ai = True
    gcp_project = st.sidebar.text_input("GCP Project ID:", value=settings.google_cloud_project or "", placeholder="e.g. enterprise-prod-ai")
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
        placeholder="AIzaSy...",
        help="Free key from Google AI Studio (aistudio.google.com). No billing required. Offline fallback engine is enabled if unconfigured.",
    )
    if api_key_input != settings.gemini_api_key:
        settings.update_api_key(api_key_input)
        gemini_client.refresh_client()
        st.sidebar.success("API Key updated!")

st.sidebar.markdown("---")

# Tiered Model Routing Info
st.sidebar.subheader("🎯 Active Model Routing")
st.sidebar.markdown(
    f"""
    - **Stage 1 (Extraction):**  
      `{settings.model_extraction}` *(Fast & Schema Strict)*
    - **Stage 2 (Embeddings):**  
      `{settings.model_embedding}` *(768-dim Vector Embeddings)*
    - **Stage 3 (Risk Audit):**  
      `{settings.model_audit}` *(Deep Chain-of-Thought)*
    """
)

# Local FAISS Vector Store Status
st.sidebar.markdown("---")
st.sidebar.subheader("🗂️ Local FAISS Vector Store")
policies = policy_retriever.get_all_policies()
st.sidebar.success(f"🟢 **{len(policies)} Corporate Rules** indexed locally (`IndexFlatIP`). $0.00 cloud storage cost.")

if st.sidebar.button("🔄 Force Re-Index Policies", use_container_width=True):
    with st.spinner("Re-embedding policies with gemini-embedding-001..."):
        run_async(policy_retriever.initialize(force_reindex=True))
        st.sidebar.success("FAISS index refreshed from disk!")

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
        st.markdown('<span class="badge-vertex">VERTEX AI ENTERPRISE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-free">100% FREE TIER ($0/MO)</span>', unsafe_allow_html=True)

# User-Friendly Workflow Guide Banner
st.markdown(
    """
    <div class="workflow-guide">
        💡 <strong>Quick Workflow:</strong>
        <strong>1.</strong> Select an enterprise contract scenario or upload a PDF/TXT contract ➔
        <strong>2.</strong> Click <strong>"Run Multi-Stage Risk Audit"</strong> ➔
        <strong>3.</strong> Review Executive Risk Dial, Verbatim Citations & Download Redline Report.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Tabs Navigation
# ============================================================================

tab_audit, tab_benchmarks, tab_policies, tab_architecture = st.tabs(
    ["📋 Contract Risk Audit", "📊 LLMOps Benchmark (15 Contracts)", "🛡️ Corporate Policy Library", "🏗️ Free-Tier Architecture"]
)


# ============================================================================
# Tab 1: Interactive Contract Risk Audit
# ============================================================================

with tab_audit:
    st.markdown("### 1. Select or Upload Vendor Contract")

    # Load ground truth metadata to enhance scenario picker
    gt_file = settings.SAMPLE_CONTRACTS_DIR / "ground_truth.json"
    gt_contracts_by_file = {}
    if gt_file.exists():
        try:
            gt_data = json.loads(gt_file.read_text(encoding="utf-8"))
            for c in gt_data.get("contracts", []):
                gt_contracts_by_file[c["contract_file"]] = c
        except Exception:
            pass

    # Three clean input options: Pre-built Enterprise Scenarios, File Upload, or Direct Paste
    input_mode = st.radio(
        "Choose Contract Input Source:",
        ["🏢 Curated Enterprise Scenarios (15 Cases across 4 Difficulty Tiers)", "📤 Upload Document (.pdf, .txt, .md)", "✍️ Paste Contract Text"],
        horizontal=True,
    )

    contract_content = ""
    active_filename = "contract.txt"
    selected_meta = None

    if "Curated Enterprise Scenarios" in input_mode:
        # Filter scenarios by difficulty
        diff_filter = st.radio(
            "Filter Scenarios by Difficulty:",
            options=["All (15)", "🟢 Easy (3)", "🟡 Moderate (4)", "🔴 Hard / Buried (4)", "🟣 Adversarial (4)"],
            horizontal=True,
            index=0,
        )

        sample_files = sorted(list(settings.SAMPLE_CONTRACTS_DIR.glob("contract_*.txt")))
        filtered_files = []

        for f in sample_files:
            meta = gt_contracts_by_file.get(f.name, {})
            diff = meta.get("difficulty", "MODERATE")
            if diff_filter == "All (15)":
                filtered_files.append(f)
            elif "Easy" in diff_filter and diff == "EASY":
                filtered_files.append(f)
            elif "Moderate" in diff_filter and diff == "MODERATE":
                filtered_files.append(f)
            elif "Hard" in diff_filter and diff == "HARD_BURIED":
                filtered_files.append(f)
            elif "Adversarial" in diff_filter and diff == "ADVERSARIAL":
                filtered_files.append(f)

        # Build dropdown options
        scenario_options = {}
        for f in filtered_files:
            meta = gt_contracts_by_file.get(f.name, {})
            tier_tag = meta.get("difficulty", "MODERATE")
            vendor = meta.get("vendor_name", f.stem)
            sector = meta.get("sector", "Enterprise")
            display_label = f"[{tier_tag}] {vendor} — {sector}"
            scenario_options[display_label] = f

        if scenario_options:
            selected_label = st.selectbox(
                "Select Contract Scenario to Audit:",
                options=list(scenario_options.keys()),
                index=0,
            )
            selected_file = scenario_options[selected_label]
            contract_content = selected_file.read_text(encoding="utf-8")
            active_filename = selected_file.name
            selected_meta = gt_contracts_by_file.get(selected_file.name)

            # Scenario Summary Card
            if selected_meta:
                diff_class = (
                    "badge-tier-easy" if selected_meta.get("difficulty") == "EASY"
                    else "badge-tier-mod" if selected_meta.get("difficulty") == "MODERATE"
                    else "badge-tier-hard" if selected_meta.get("difficulty") == "HARD_BURIED"
                    else "badge-tier-adv"
                )
                acv_str = f"${selected_meta.get('acv_usd', 0):,.2f} USD" if selected_meta.get('acv_usd') else "N/A"
                exp_verdict = selected_meta.get("expected_executive_verdict", "REJECT")
                verdict_color = "#10b981" if exp_verdict == "APPROVE" else "#f97316" if exp_verdict == "CONDITIONAL_APPROVAL" else "#ef4444"

                st.markdown(
                    f"""
                    <div class="app-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <h4 style="margin:0;">🏢 {selected_meta.get('vendor_name')}</h4>
                                <span style="font-size:0.88rem; color:#64748b;">Domain: {selected_meta.get('sector')} • Document: <code>{active_filename}</code></span>
                            </div>
                            <div>
                                <span class="{diff_class}">{selected_meta.get('difficulty')} TIER</span>
                            </div>
                        </div>
                        <div style="display:flex; gap:24px; margin-top:12px; font-size:0.9rem;">
                            <div><strong>Annual Value:</strong> {acv_str}</div>
                            <div><strong>Expected Verdict:</strong> <span style="color:{verdict_color}; font-weight:700;">{exp_verdict}</span></div>
                            <div><strong>Expected Violations:</strong> {len(selected_meta.get('expected_violated_policy_ids', []))} rules</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning("No contracts match the selected difficulty filter.")

    elif "Upload Document" in input_mode:
        uploaded_file = st.file_uploader(
            "Upload Contract (.pdf, .txt, .md):",
            type=["pdf", "txt", "md"],
            help="Supports standard digital PDFs and text files. Text is parsed cleanly without third-party cloud upload.",
        )
        if uploaded_file:
            active_filename = uploaded_file.name
            if uploaded_file.name.lower().endswith(".pdf"):
                try:
                    import pypdf
                    reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
                    contract_content = "\n\n".join([page.extract_text() or "" for page in reader.pages])
                    st.success(f"✅ Successfully extracted {len(reader.pages)} pages from `{uploaded_file.name}` via local pypdf!")
                except Exception as e:
                    st.error(f"Error parsing PDF document: {e}")
            else:
                contract_content = uploaded_file.read().decode("utf-8", errors="replace")
                st.success(f"✅ Loaded `{uploaded_file.name}` ({len(contract_content):,} characters)!")

    else:  # Paste Contract Text
        contract_content = st.text_area(
            "Paste Vendor Contract Text Here:",
            height=200,
            placeholder="Paste standard SaaS Master Services Agreement (MSA), Terms of Service, or SLA text...",
        )
        active_filename = "pasted_contract.txt"

    # Collapsible Raw Contract Inspector
    if contract_content:
        words = len(contract_content.split())
        chars = len(contract_content)
        with st.expander(f"📄 Inspect Raw Contract Text ({words:,} words • {chars:,} characters)", expanded=False):
            st.text_area("Contract Text Preview", value=contract_content, height=220, disabled=True)

    # Primary Action Button
    st.markdown("<br>", unsafe_allow_html=True)
    run_audit_col1, run_audit_col2 = st.columns([2, 1])
    with run_audit_col1:
        run_audit = st.button("🚀 Run Multi-Stage Risk Audit (Extract ➔ RAG ➔ CoT Audit)", type="primary", use_container_width=True)
    with run_audit_col2:
        if st.button("🧹 Clear Results Cache", use_container_width=True):
            if "latest_analysis" in st.session_state:
                del st.session_state["latest_analysis"]
            st.rerun()

    if run_audit:
        if not contract_content.strip():
            st.error("⚠️ Please provide or select a vendor contract before executing the audit.")
        else:
            with st.status("Executing Multi-Stage Intelligence Pipeline...", expanded=True) as status:
                st.write(f"⚡ **Stage 1 (Extraction):** Parsing clauses via `{settings.model_extraction}` (Few-Shot Pydantic)...")
                t0 = time.perf_counter()

                # Run end-to-end pipeline
                analysis_response: ContractAnalysisResponse = run_async(
                    contract_pipeline.analyze_contract(
                        contract_text=contract_content,
                        filename=active_filename,
                    )
                )

                st.write(f"🔎 **Stage 2 (Policy RAG):** Vector matching against local FAISS via `{settings.model_embedding}`...")
                st.write(f"🧠 **Stage 3 (CoT Audit):** Deep chain-of-thought risk evaluation via `{settings.model_audit}`...")
                status.update(label=f"✅ Risk Audit Complete! (Took {analysis_response.latencies.total_ms:.0f} ms)", state="complete", expanded=False)

            st.session_state["latest_analysis"] = analysis_response

    # ========================================================================
    # Display Audit Deliverables
    # ========================================================================
    if "latest_analysis" in st.session_state:
        res: ContractAnalysisResponse = st.session_state["latest_analysis"]
        audit = res.audit
        extraction = res.extraction

        st.markdown("---")
        st.markdown("### 2. Executive Audit Verdict & Risk Score")

        # Top Executive Cards Layout: Visual Gauge + Executive Metrics
        gauge_col, metrics_col = st.columns([1, 1])

        with gauge_col:
            # Interactive Plotly Risk Meter
            fig_gauge = create_risk_gauge(audit.risk_score, audit.risk_level)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with metrics_col:
            st.markdown("<br>", unsafe_allow_html=True)
            # Executive Verdict Banner
            verdict_badge = audit.executive_verdict
            if verdict_badge == "APPROVE":
                st.success(f"### ✅ Executive Verdict: **APPROVE**\nContract complies with corporate policy thresholds.")
            elif verdict_badge == "CONDITIONAL_APPROVAL":
                st.warning(f"### ⚠️ Executive Verdict: **CONDITIONAL APPROVAL**\nApproval contingent upon vendor accepting mandated redlines.")
            else:
                st.error(f"### 🛑 Executive Verdict: **REJECT OR REDLINE**\nContract exhibits critical liability, security, or commercial violations.")

            # Summary metrics grid
            m1, m2, m3 = st.columns(3)
            m1.metric("Policy Violations", len(audit.violations), help="Total number of enterprise policy breaches detected.")
            m2.metric("Compliant Rules", len(audit.compliant_policies), help="Corporate rules verified fully compliant.")
            m3.metric("Pipeline Cost", "$0.00", help="100% Free-Tier Google AI Studio API allocation.")

        # Executive Summary Box
        st.info(f"**Executive Summary:** {audit.executive_summary}")

        # Negotiation Leverage Points
        if audit.key_negotiation_points:
            with st.expander("💡 View Key Negotiation Leverage Points for Procurement", expanded=True):
                for p in audit.key_negotiation_points:
                    st.markdown(f"- 🎯 **{p}**")

        # Chain of Thought Reasoning Steps
        with st.expander("🧠 View Auditor Chain-of-Thought Reasoning Steps (gemini-3.8-flash)", expanded=False):
            for step in audit.auditor_reasoning_steps:
                st.markdown(f"- {step}")

        # ====================================================================
        # Section 3: Policy Violations & Actionable Redlines
        # ====================================================================
        st.markdown("---")
        st.markdown(f"### 3. Policy Violations & Actionable Redlines ({len(audit.violations)} Breaches)")

        if not audit.violations:
            st.success("🎉 **Zero policy violations detected!** This vendor contract fully complies with all corporate security, SLA, liability, and procurement standards.")
        else:
            # Interactive Severity Filter
            sev_filter = st.radio(
                "Filter Violations by Severity:",
                options=["All Breaches", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
                horizontal=True,
                index=0,
            )

            filtered_violations = [
                v for v in audit.violations
                if sev_filter == "All Breaches" or v.severity == sev_filter
            ]

            if not filtered_violations:
                st.info(f"No violations found with severity `{sev_filter}`.")

            for idx, v in enumerate(filtered_violations, 1):
                card_class = f"violation-card-{v.severity.lower()}"
                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h4 style="margin:0;">#{idx} [{v.severity}] {v.policy_id}: {v.policy_title}</h4>
                            <span style="font-size:0.8rem; background:rgba(0,0,0,0.08); padding:3px 8px; border-radius:4px; font-weight:600;">{v.category}</span>
                        </div>
                        <p style="margin:8px 0 4px 0;"><strong>Violation Analysis:</strong> {v.violation_reason}</p>
                        <div class="quote-box">
                            <strong>📌 Verbatim Contract Quote Citation:</strong><br>
                            "{v.contract_quote}"
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander(f"📝 Recommended Contract Redline for #{idx} ({v.policy_id})", expanded=True):
                    st.markdown("**Suggested Procurement Redline:**")
                    st.code(v.recommended_redline, language="markdown")

            # Report Export Buttons
            st.markdown("<br>", unsafe_allow_html=True)
            exp_col1, exp_col2 = st.columns(2)

            # Generate Downloadable Markdown Report
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

            with exp_col1:
                st.download_button(
                    label="📥 Download Executive Redline Report (.md)",
                    data=report_text,
                    file_name=f"{extraction.vendor_metadata.vendor_name.replace(' ', '_')}_redline_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            with exp_col2:
                audit_json = res.model_dump_json(indent=2)
                st.download_button(
                    label="📥 Export Full Audit Data (.json)",
                    data=audit_json,
                    file_name=f"{extraction.vendor_metadata.vendor_name.replace(' ', '_')}_audit.json",
                    mime="application/json",
                    use_container_width=True,
                )

        # ====================================================================
        # Section 4: Formatted Structured Clauses
        # ====================================================================
        st.markdown("---")
        st.markdown("### 4. Extracted Commercial Clauses (gemini-3.5-flash-lite)")

        clause_tab1, clause_tab2 = st.tabs(["📑 Formatted Clause Cards", "💻 Raw JSON Schema"])

        with clause_tab1:
            cl1, cl2 = st.columns(2)

            with cl1:
                st.markdown(
                    f"""
                    <div class="clause-card">
                        <div class="clause-label">🏢 Commercial Metadata</div>
                        <div class="clause-value">{extraction.vendor_metadata.vendor_name}</div>
                        <div style="font-size:0.88rem; color:#64748b; margin-top:4px;">
                            <strong>Contract Value:</strong> ${extraction.vendor_metadata.total_contract_value_usd or 0:,.2f} {extraction.vendor_metadata.currency}<br>
                            <strong>Effective Date:</strong> {extraction.vendor_metadata.effective_date or 'N/A'}<br>
                            <strong>Expiration Date:</strong> {extraction.vendor_metadata.expiration_date or 'N/A'}<br>
                            <strong>Governing Law:</strong> {extraction.vendor_metadata.governing_law or 'N/A'} ({extraction.vendor_metadata.jurisdiction or 'N/A'})
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="clause-card">
                        <div class="clause-label">💳 Payment Terms & Penalties</div>
                        <div class="clause-value">Net {extraction.payment_terms.payment_due_days or 'N/A'} Days</div>
                        <div style="font-size:0.88rem; color:#64748b; margin-top:4px;">
                            <strong>Late Payment Interest:</strong> {extraction.payment_terms.late_payment_interest_percent_monthly or 0}% per month<br>
                            <strong>Disputed Invoice Withholding:</strong> {'Allowed ✅' if extraction.payment_terms.disputed_invoice_withholding_allowed else 'Prohibited ❌'}<br>
                            <strong>Price Increase Cap:</strong> {extraction.payment_terms.price_increase_cap_percent or 'None'}% per year
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="clause-card">
                        <div class="clause-label">🛡️ Liability & Indemnification</div>
                        <div class="clause-value">{extraction.liability.liability_cap_multiple_of_fees or 0}x ACV Cap</div>
                        <div style="font-size:0.88rem; color:#64748b; margin-top:4px;">
                            <strong>Direct Dollar Cap:</strong> ${extraction.liability.direct_dollar_cap_usd or 0:,.2f} USD<br>
                            <strong>Mutual Indemnity:</strong> {'Yes ✅' if extraction.liability.mutual_indemnification else 'Unilateral ❌'}<br>
                            <strong>IP Carveout:</strong> {'Uncapped ✅' if extraction.liability.uncapped_ip_indemnity else 'Capped ❌'}<br>
                            <strong>Data Breach Carveout:</strong> {'Included ✅' if extraction.liability.data_breach_carveout else 'Missing ❌'}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with cl2:
                st.markdown(
                    f"""
                    <div class="clause-card">
                        <div class="clause-label">⏱️ SLA & Service Availability</div>
                        <div class="clause-value">{extraction.sla.uptime_guarantee_percent or 0}% Uptime</div>
                        <div style="font-size:0.88rem; color:#64748b; margin-top:4px;">
                            <strong>Service Credits:</strong> {'Provided ✅' if extraction.sla.service_credits_provided else 'None ❌'}<br>
                            <strong>Max Credit Cap:</strong> {extraction.sla.max_service_credit_percent_of_monthly_fee or 0}% of monthly fee<br>
                            <strong>Sole Remedy Disclaimer:</strong> {'Sole & Exclusive ⚠️' if extraction.sla.sole_remedy_clause else 'Standard'}<br>
                            <strong>Maintenance Notice:</strong> {extraction.sla.scheduled_maintenance_notice_hours or 'N/A'} hours
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="clause-card">
                        <div class="clause-label">🔄 Term & Auto-Renewal</div>
                        <div class="clause-value">{extraction.termination_renewal.initial_term_months or 12} Months</div>
                        <div style="font-size:0.88rem; color:#64748b; margin-top:4px;">
                            <strong>Auto-Renewal:</strong> {'Yes' if extraction.termination_renewal.auto_renewal else 'No'}<br>
                            <strong>Non-Renewal Notice:</strong> {extraction.termination_renewal.non_renewal_notice_days or 0} days before renewal<br>
                            <strong>Termination for Convenience:</strong> {'Permitted ✅' if extraction.termination_renewal.termination_for_convenience else 'Prohibited ❌'}<br>
                            <strong>Transition Assistance:</strong> {extraction.termination_renewal.transition_assistance_days or 0} days
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="clause-card">
                        <div class="clause-label">🔒 Security & Data Privacy</div>
                        <div class="clause-value">Certifications: {', '.join(extraction.security_compliance.certifications) if extraction.security_compliance.certifications else 'None Stated'}</div>
                        <div style="font-size:0.88rem; color:#64748b; margin-top:4px;">
                            <strong>Security Audit Rights:</strong> {'Included ✅' if extraction.security_compliance.security_audit_rights else 'Denied ❌'}<br>
                            <strong>Breach Notification:</strong> {extraction.security_compliance.data_breach_notification_window_hours or 'N/A'} hours<br>
                            <strong>Data Encryption at Rest:</strong> {'Yes ✅' if extraction.security_compliance.data_encryption_at_rest else 'No ❌'}<br>
                            <strong>Subprocessor Prior Consent:</strong> {'Required ✅' if extraction.security_compliance.subprocessor_prior_consent else 'No Consent Required ❌'}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with clause_tab2:
            st.json(extraction.model_dump())

        # ====================================================================
        # Section 5: Stage Telemetry & Free-Tier Quota Tracking
        # ====================================================================
        st.markdown("---")
        st.markdown("### 5. Pipeline Telemetry & Free-Tier Quota Tracking")
        tel_col_1, tel_col_2, tel_col_3, tel_col_4 = st.columns(4)
        tel_col_1.metric("Stage 1 (Extraction)", f"{res.latencies.extraction_ms:.1f} ms", f"Model: {res.model_routing['stage_1_extraction']}")
        tel_col_2.metric("Stage 2 (Policy RAG)", f"{res.latencies.rag_retrieval_ms:.1f} ms", "Local FAISS IndexFlatIP")
        tel_col_3.metric("Stage 3 (CoT Audit)", f"{res.latencies.audit_ms:.1f} ms", f"Model: {res.model_routing['stage_3_cot_audit']}")
        tel_col_4.metric("Total Free Cost", "$0.00", "Google AI Studio Free Tier")


# ============================================================================
# Tab 2: LLMOps Benchmark & Eval Dashboard
# ============================================================================

with tab_benchmarks:
    st.markdown("### Rigorous LLMOps Benchmark: Naive Zero-Shot vs. Production Pipeline")
    st.markdown(
        "Evaluates **15 diverse enterprise contracts** across 4 difficulty tiers: "
        "`EASY`, `MODERATE`, `HARD_BURIED`, and `ADVERSARIAL`. "
        "The benchmark proves why multi-stage Few-Shot extraction + local FAISS Policy RAG + CoT reasoning "
        "dramatically outperforms standard zero-shot prompting while operating at **$0.00 free-tier expense**."
    )

    bench_run_col1, bench_run_col2 = st.columns([1, 3])
    with bench_run_col1:
        run_bench = st.button("▶️ Re-Run 15-Contract Benchmark", type="primary", use_container_width=True)

    if run_bench:
        with st.spinner("Executing throttled evaluation benchmark across all 15 contracts..."):
            from eval.run_eval import run_benchmark
            run_async(run_benchmark(throttle_delay_seconds=0.2))
            st.success("✅ 15-Contract Benchmark completed and saved to `eval/benchmark_results.json`!")

    # Load results from disk
    if settings.EVAL_RESULTS_PATH.exists():
        bench_data = json.loads(settings.EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
        summary = bench_data.get("summary", {})
        base = summary.get("naive_zero_shot", {})
        pipe = summary.get("production_pipeline", {})
        tiers = bench_data.get("difficulty_breakdown", {})
        scorecards = bench_data.get("contract_scorecards", [])

        st.markdown(f"#### 1. Executive Performance Metrics ({bench_data.get('total_contracts_evaluated', 15)} Contracts Evaluated)")

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Violation Detection F1", f"{pipe.get('f1', 0.0):.3f}", f"+{(pipe.get('f1', 0.0) - base.get('f1', 0.0))*100:.1f}% vs Zero-Shot")
        b2.metric("Violation Recall (Sensitivity)", f"{pipe.get('recall', 0.0):.3f}", f"+{(pipe.get('recall', 0.0) - base.get('recall', 0.0))*100:.1f}% vs Zero-Shot")
        b3.metric("Violation Precision", f"{pipe.get('precision', 0.0):.3f}", f"+{(pipe.get('precision', 0.0) - base.get('precision', 0.0))*100:.1f}% vs Zero-Shot")
        b4.metric("Quote Citation Acc.", f"{pipe.get('citation_accuracy', 0.0)*100:.1f}%", f"+{(pipe.get('citation_accuracy', 0.0) - base.get('citation_accuracy', 0.0))*100:.1f}% vs Zero-Shot")

        # Visual Benchmark Charts
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.plotly_chart(create_benchmark_bar_chart(base, pipe), use_container_width=True)
        with chart_col2:
            if tiers:
                st.plotly_chart(create_tier_breakdown_chart(tiers), use_container_width=True)

        # Tabular Metric Comparison
        st.markdown("#### 2. Comprehensive Metric Comparison Table")
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

        if scorecards:
            with st.expander("📋 View Individual Contract Scorecards (15 Contracts Evaluated)", expanded=False):
                st.dataframe(scorecards, use_container_width=True)
    else:
        st.info("No benchmark results found on disk. Click 'Re-Run 15-Contract Benchmark' above to generate the metrics.")


# ============================================================================
# Tab 3: Corporate Policy Library
# ============================================================================

with tab_policies:
    st.markdown(f"### Corporate Policy Compliance Rules ({len(policies)} Standards)")
    st.markdown("These enterprise procurement, SLA, liability, and security policies are embedded into the local FAISS index once and queried during contract analysis.")

    # Search and Category Filter
    pol_filter_col1, pol_filter_col2 = st.columns([2, 1])
    with pol_filter_col1:
        search_query = st.text_input("🔍 Search Policies (by title, ID, or keyword):", placeholder="e.g. Net 45, Uptime, Breach, Liability...")
    with pol_filter_col2:
        cat_filter = st.selectbox("Category Filter:", ["All Categories", "Procurement / Payment", "Liability / Legal", "SLA / Service Level", "Security / Infosec", "Data Privacy / Governance"])

    filtered_policies = []
    for p in policies:
        matches_query = (
            not search_query
            or search_query.lower() in p.title.lower()
            or search_query.lower() in p.policy_id.lower()
            or search_query.lower() in p.rule_summary.lower()
        )
        matches_cat = (
            cat_filter == "All Categories"
            or (cat_filter == "Procurement / Payment" and p.category.lower() == "payment")
            or (cat_filter == "Liability / Legal" and p.category.lower() in ["liability", "governing_law"])
            or (cat_filter == "SLA / Service Level" and p.category.lower() == "sla")
            or (cat_filter == "Security / Infosec" and p.category.lower() == "security")
            or (cat_filter == "Data Privacy / Governance" and p.category.lower() in ["data_protection", "data_residency"])
        )
        if matches_query and matches_cat:
            filtered_policies.append(p)

    st.markdown(f"Showing **{len(filtered_policies)} matching policies**:")

    for p in filtered_policies:
        with st.expander(f"📌 [{p.severity_if_violated}] {p.policy_id}: {p.title}", expanded=True if search_query else False):
            c_p1, c_p2 = st.columns([3, 1])
            with c_p1:
                st.markdown(f"**Rule Summary:** {p.rule_summary}")
                st.markdown(f"**Minimum Requirement:** `{p.minimum_requirement}`")
                st.markdown(f"**Standard Redline Guidance:** {p.redline_guidance}")
            with c_p2:
                st.markdown(f"**Category:** `{p.category}`")
                st.markdown(f"**Violation Severity:** `{p.severity_if_violated}`")


# ============================================================================
# Tab 4: Free-Tier Architecture Blueprint
# ============================================================================

with tab_architecture:
    st.markdown("### Production $0 Free-Tier Architecture & Model Routing")
    st.markdown(
        "This enterprise document intelligence architecture is specifically engineered to run **completely free** on "
        "Google AI Studio's free-tier API quotas, while retaining full enterprise compatibility with GCP Vertex AI."
    )

    st.markdown(
        """
        ```mermaid
        flowchart TD
            A[Vendor Contract Document<br/>PDF / TXT / MD] --> B[Stage 1: Few-Shot Extraction]
            B -->|Model: gemini-3.5-flash-lite| C[Strict Pydantic ContractExtractionResult]
            C --> D[Stage 2: Policy RAG Matching]
            E[(Local FAISS Index<br/>gemini-embedding-001<br/>data/faiss_index/)] -->|Zero Cloud DB Cost| D
            D -->|Top-K Matching Corporate Rules| F[Stage 3: Deep Chain-of-Thought Audit]
            F -->|Model: gemini-3.8-flash| G[Strict Pydantic ChainOfThoughtAudit]
            G --> H[Actionable Executive Deliverables]
            H --> H1[Risk Meter 0-100 & Risk Level]
            H --> H2[Verbatim Contract Quote Citations]
            H --> H3[Procurement Legal Redlines]
            H --> H4[Executive Decision: APPROVE / REJECT]
        ```
        """
    )

    st.markdown("#### Google AI Studio Free-Tier Quota Allocation")
    st.markdown(
        """
        | Pipeline Stage | Free-Tier Model | RPM Quota | TPM Quota | Daily RPD | Free-Tier Cost |
        | :--- | :--- | :---: | :---: | :---: | :---: |
        | **Stage 1 (Extraction)** | `gemini-3.5-flash-lite` | 15 RPM | 1,000,000 TPM | 1,500 RPD | **$0.00** |
        | **Stage 2 (Embeddings)** | `gemini-embedding-001` | 15 RPM | 1,000,000 TPM | 1,500 RPD | **$0.00** |
        | **Stage 3 (CoT Audit)** | `gemini-3.8-flash` | 15 RPM | 1,000,000 TPM | 1,500 RPD | **$0.00** |
        | **Vector Database** | Local FAISS (`faiss-cpu`) | Unlimited | Unlimited | Unlimited | **$0.00** |
        """
    )

    st.markdown("#### Cloud Deployment Options")
    st.markdown(
        """
        - **Hugging Face Spaces (Docker SDK):** Deploys FastAPI + Streamlit simultaneously on port `7860` with zero credit card required.
        - **Render Free Web Service:** Runs the multi-stage Docker container on Render's free tier.
        - **Google Cloud Run (Vertex AI Mode):** Switch `USE_VERTEX_AI=true` to deploy into an enterprise GCP VPC with IAM service account authentication.
        """
    )
