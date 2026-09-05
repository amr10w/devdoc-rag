"""
src/app/dashboard.py

Streamlit interactive interface and telemetry monitoring dashboard for DevDoc RAG.
CRITICAL ARCHITECTURAL DIRECTIVE (Rule 1):
This dashboard communicates EXCLUSIVELY via HTTP calls to the FastAPI backend (main.py).
It NEVER imports DevDocSearcher or accesses Qdrant storage / SQLite files directly.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="DevDoc RAG — Technical Assistant & Monitoring",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS for a polished, professional look
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Gradient header banner */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
    .main-header p  { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1f2e 0%, #252b3b 100%);
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1rem;
    }
    div[data-testid="stMetric"] label { color: #aaa !important; font-size: 0.8rem; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #00CC96 !important; font-weight: 700; }

    /* Chat message styling */
    .stChatMessage { border-radius: 12px; }

    /* Sidebar polish */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1A1F2E 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1 { font-size: 1.4rem; }

    /* Tab underline color */
    .stTabs [data-baseweb="tab-highlight"] { background-color: #00CC96; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }

    /* Retrieved chunk card */
    .chunk-card {
        background: #1a1f2e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .chunk-card .badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# API Communication Helpers (Rule 1 Compliance)
# ─────────────────────────────────────────────

@st.cache_data(ttl=10)
def fetch_health() -> dict[str, Any] | None:
    """Cached health check — refreshes every 10 seconds."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=4)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=15)
def fetch_metrics_summary() -> dict[str, Any] | None:
    """Cached metrics summary — refreshes every 15 seconds."""
    try:
        resp = requests.get(f"{API_BASE_URL}/metrics/summary", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=15)
def fetch_logs(limit: int = 100) -> list[dict[str, Any]]:
    """Cached log fetch — refreshes every 15 seconds."""
    try:
        resp = requests.get(f"{API_BASE_URL}/metrics/logs", params={"limit": limit}, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def submit_query(
    query: str,
    source_lib: str | None,
    retrieval_method: str,
    top_k: int,
    rewrite_query: bool,
) -> dict[str, Any] | None:
    """Sends a RAG query to the FastAPI backend."""
    payload = {
        "query": query,
        "source_lib": source_lib if source_lib and source_lib != "All Libraries" else None,
        "retrieval_method": retrieval_method,
        "top_k": top_k,
        "rewrite_query": rewrite_query,
    }
    try:
        resp = requests.post(f"{API_BASE_URL}/ask", json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API Error ({resp.status_code}): {resp.text}")
    except requests.exceptions.Timeout:
        st.error("Request timed out. The LLM may be under heavy load — please try again.")
    except Exception as e:
        st.error(f"Failed to communicate with FastAPI backend: {e}")
    return None


def send_feedback(log_id: str, rating: int, comment: str = "") -> bool:
    """Sends user feedback to the backend."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/feedback",
            json={"log_id": log_id, "rating": rating, "comment": comment},
            timeout=5,
        )
        return resp.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────────
# Sidebar: System Status & Search Configuration
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("# ⚡ DevDoc RAG")
    st.caption("Technical Documentation Assistant")

    # Health status
    health = fetch_health()
    if health and health.get("status") == "healthy":
        st.success("● Backend Online")
        qdrant_points = health.get("qdrant", {}).get("points_count", 0)
        st.caption(f"**Qdrant:** {qdrant_points:,} points indexed")
    elif health and health.get("status") == "degraded":
        st.warning("● Backend Degraded")
    else:
        st.error("● Backend Offline")

    st.markdown("---")

    # Search configuration (moved from main area)
    st.markdown("### ⚙️ Search Settings")

    library_filter = st.selectbox(
        "Library Filter",
        ["All Libraries", "fastapi", "docker", "pytorch", "pydantic", "transformers", "qdrant", "postgresql"],
        index=0,
        help="Restrict search to a specific documentation library.",
    )
    search_method = st.selectbox(
        "Retrieval Strategy",
        ["hybrid_rrf", "vector", "text"],
        index=0,
        help="Hybrid RRF fuses dense vector and keyword search.",
    )
    num_chunks = st.slider("Top Chunks (k)", min_value=2, max_value=8, value=4)
    rewrite_toggle = st.checkbox(
        "🔄 Query Rewriting",
        value=False,
        help="Reformulates your question into an optimized search query.",
    )

    st.markdown("---")
    st.caption("LLM Zoomcamp Capstone Project")
    st.caption("Hybrid RRF · Qdrant · Ollama")


# ─────────────────────────────────────────────
# Initialize session state
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = set()


# ─────────────────────────────────────────────
# Main Content: Tabbed Navigation
# ─────────────────────────────────────────────

tab_chat, tab_monitor, tab_logs = st.tabs([
    "💬 Assistant",
    "📊 Monitoring Dashboard",
    "📋 Telemetry Logs",
])


# ═══════════════════════════════════════════
# TAB 1: Interactive Chat Assistant
# ═══════════════════════════════════════════

with tab_chat:
    # Header banner
    st.markdown(
        """
        <div class="main-header">
            <h1>💬 Technical Documentation Assistant</h1>
            <p>Ask questions across FastAPI, Docker, PyTorch, Pydantic, Qdrant, PostgreSQL & Transformers</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show metadata for assistant messages
            if msg["role"] == "assistant" and "metadata" in msg:
                meta = msg["metadata"]

                # Rewritten query info
                if meta.get("rewritten_query"):
                    st.info(f"🔄 **Query rewritten to:** `{meta['rewritten_query']}`")

                # Performance metrics row
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Latency", f"{meta.get('latency_ms', 0):.0f} ms")
                mc2.metric("Strategy", meta.get("retrieval_method", "hybrid_rrf").upper())
                mc3.metric("Chunks", len(meta.get("retrieved_chunks", [])))

                # Retrieved chunks expander
                chunks = meta.get("retrieved_chunks", [])
                if chunks:
                    with st.expander(f"🔍 View {len(chunks)} Retrieved Chunks", expanded=False):
                        for i, chunk in enumerate(chunks, 1):
                            score = chunk.get("score", 0)
                            lib = chunk.get("source_lib", "unknown")
                            section = chunk.get("section_title", "Document")
                            st.markdown(
                                f"**Chunk #{i}** · "
                                f"`{lib}` · `{section}` · "
                                f"Score: `{score:.4f}`"
                            )
                            st.caption(f"File: `{chunk.get('file_path', '')}` | ID: `{chunk.get('chunk_id', '')}`")
                            st.markdown(chunk.get("content", ""))
                            if i < len(chunks):
                                st.divider()

                # Feedback buttons
                log_id = meta.get("log_id", "")
                if log_id and log_id not in st.session_state.feedback_given:
                    st.markdown("**Was this helpful?**")
                    fb1, fb2, _ = st.columns([1, 1, 4])
                    with fb1:
                        if st.button("👍 Yes", key=f"up_{log_id}"):
                            if send_feedback(log_id, 1, "Helpful answer"):
                                st.session_state.feedback_given.add(log_id)
                                st.success("Thanks! Feedback recorded.")
                                st.rerun()
                    with fb2:
                        if st.button("👎 No", key=f"dn_{log_id}"):
                            if send_feedback(log_id, -1, "Not helpful"):
                                st.session_state.feedback_given.add(log_id)
                                st.warning("Feedback recorded. We'll improve!")
                                st.rerun()
                elif log_id:
                    st.caption("✅ Feedback recorded for this response.")

    # Chat input (handles Enter key natively)
    user_query = st.chat_input(
        "Ask a technical question (e.g., How do I create a FastAPI dependency?)"
    )

    if user_query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documentation & generating answer..."):
                result = submit_query(
                    query=user_query,
                    source_lib=library_filter,
                    retrieval_method=search_method,
                    top_k=num_chunks,
                    rewrite_query=rewrite_toggle,
                )

            if result:
                response_text = result.get("response", "No response generated.")
                st.markdown(response_text)

                # Store assistant message with full metadata
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "metadata": {
                        "log_id": result.get("log_id", ""),
                        "rewritten_query": result.get("rewritten_query"),
                        "latency_ms": result.get("latency_ms", 0),
                        "retrieval_method": result.get("retrieval_method", "hybrid_rrf"),
                        "retrieved_chunks": result.get("retrieved_chunks", []),
                    },
                })
                st.rerun()
            else:
                st.error("Failed to get a response. Please check backend connectivity.")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⚠️ Failed to get a response. Please check backend connectivity.",
                })


# ═══════════════════════════════════════════
# TAB 2: Monitoring Dashboard (5 Figures)
# ═══════════════════════════════════════════

with tab_monitor:
    st.markdown(
        """
        <div class="main-header">
            <h1>📊 Telemetry & Monitoring Dashboard</h1>
            <p>Real-time analytics from SQLite telemetry logs — 5 analytical figures</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary = fetch_metrics_summary()
    raw_logs = fetch_logs(limit=200)

    if not summary or summary.get("total_queries", 0) == 0:
        st.warning("📭 No telemetry data found yet. Submit queries via the Assistant tab first!")
        st.info("💡 The FastAPI backend auto-seeds 35 mock records on first startup. "
                "If you see this message, the backend may not have started yet.")
    else:
        # ── KPI Cards ──
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Queries", f"{summary.get('total_queries', 0):,}")
        kpi2.metric("Feedback Count", f"{summary.get('total_feedback', 0):,}")
        kpi3.metric("Satisfaction Rate", f"{summary.get('satisfaction_rate', 0):.1f}%")
        kpi4.metric("Avg Latency", f"{summary.get('avg_latency_ms', 0):.0f} ms")

        st.divider()

        # Build dataframe from logs
        df = pd.DataFrame(raw_logs) if raw_logs else pd.DataFrame()
        if not df.empty and "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"])
            df["date"] = df["datetime"].dt.date
            df["hour"] = df["datetime"].dt.hour

        # ── FIGURE 1: Queries Over Time ──
        st.subheader("1️⃣ Queries Over Time")
        if not df.empty and "date" in df.columns:
            daily_counts = df.groupby("date").size().reset_index(name="query_count")
            fig1 = px.area(
                daily_counts,
                x="date",
                y="query_count",
                title="Daily RAG Query Volume",
                labels={"date": "Date", "query_count": "Queries"},
                color_discrete_sequence=["#00CC96"],
            )
            fig1.update_layout(
                xaxis_title="Date", yaxis_title="Queries",
                hovermode="x unified",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Not enough data to plot query trends.")

        # ── FIGURE 2 & 3 side by side ──
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("2️⃣ Feedback Breakdown")
            pos = summary.get("positive_feedback", 0)
            neg = summary.get("negative_feedback", 0)
            no_fb = max(0, summary.get("total_queries", 0) - (pos + neg))

            fig2 = go.Figure(data=[go.Pie(
                labels=["Positive 👍", "Negative 👎", "Unrated"],
                values=[pos, neg, no_fb],
                hole=0.45,
                marker=dict(colors=["#2CA02C", "#D62728", "#555"]),
                textinfo="percent+label",
            )])
            fig2.update_layout(
                title="User Sentiment Distribution",
                showlegend=True,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            st.subheader("3️⃣ Latency Distribution")
            if not df.empty and "latency_ms" in df.columns:
                p50 = summary.get("p50_latency_ms", 0)
                p90 = summary.get("p90_latency_ms", 0)
                p99 = summary.get("p99_latency_ms", 0)
                fig3 = px.histogram(
                    df,
                    x="latency_ms",
                    nbins=20,
                    title=f"Latency (P50: {p50:.0f}ms · P90: {p90:.0f}ms · P99: {p99:.0f}ms)",
                    labels={"latency_ms": "Latency (ms)"},
                    color_discrete_sequence=["#636EFA"],
                )
                fig3.add_vline(x=p50, line_dash="dash", line_color="green", annotation_text="P50")
                fig3.add_vline(x=p90, line_dash="dash", line_color="orange", annotation_text="P90")
                fig3.add_vline(x=p99, line_dash="dash", line_color="red", annotation_text="P99")
                fig3.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Not enough data for latency distribution.")

        # ── FIGURE 4 & 5 side by side ──
        col_c, col_d = st.columns(2)

        with col_c:
            st.subheader("4️⃣ Top Queried Libraries")
            lib_data = summary.get("lib_counts", {})
            if lib_data:
                lib_df = pd.DataFrame(
                    list(lib_data.items()), columns=["Library", "Count"]
                ).sort_values("Count", ascending=True)
                fig4 = px.bar(
                    lib_df,
                    x="Count",
                    y="Library",
                    orientation="h",
                    title="Query Frequency by Library",
                    color="Count",
                    color_continuous_scale="Viridis",
                )
                fig4.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("No library data available.")

        with col_d:
            st.subheader("5️⃣ Response Length Distribution")
            if not df.empty and "char_count" in df.columns:
                avg_c = summary.get("avg_chars", 0)
                avg_w = summary.get("avg_words", 0)
                fig5 = px.box(
                    df,
                    y="char_count",
                    points="all",
                    title=f"Response Length (Avg: {avg_c:.0f} chars · ~{avg_w:.0f} words)",
                    labels={"char_count": "Characters"},
                    color_discrete_sequence=["#EF553B"],
                )
                fig5.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.info("Not enough data for response length analysis.")


# ═══════════════════════════════════════════
# TAB 3: Telemetry Logs Explorer
# ═══════════════════════════════════════════

with tab_logs:
    st.markdown(
        """
        <div class="main-header">
            <h1>📋 Telemetry Logs Explorer</h1>
            <p>Live query inspection table from FastAPI /metrics/logs</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    logs = fetch_logs(limit=50)
    if logs:
        df_logs = pd.DataFrame(logs)
        display_cols = [
            "timestamp",
            "query",
            "source_lib",
            "retrieval_method",
            "latency_ms",
            "feedback_rating",
            "feedback_comment",
        ]
        available_cols = [c for c in display_cols if c in df_logs.columns]
        st.dataframe(
            df_logs[available_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm"),
                "query": st.column_config.TextColumn("Query", width="large"),
                "source_lib": st.column_config.TextColumn("Library"),
                "retrieval_method": st.column_config.TextColumn("Strategy"),
                "latency_ms": st.column_config.NumberColumn("Latency (ms)", format="%.0f"),
                "feedback_rating": st.column_config.NumberColumn("Rating"),
                "feedback_comment": st.column_config.TextColumn("Comment"),
            },
        )
    else:
        st.info("No logs available yet. Submit queries through the Assistant tab to see telemetry data here.")


# ─────────────────────────────────────────────
# CLI Entrypoint
# ─────────────────────────────────────────────

def start() -> None:
    """CLI entrypoint for running the Streamlit dashboard."""
    from streamlit import runtime

    if runtime.exists():
        return

    import sys
    import streamlit.web.cli as stcli

    sys.argv = ["streamlit", "run", __file__, "--server.port=8501", "--server.address=0.0.0.0"]
    stcli.main()


if __name__ == "__main__":
    from streamlit import runtime

    if not runtime.exists():
        start()
