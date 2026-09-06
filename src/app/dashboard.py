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

# Fallback library list, used only if the backend doesn't expose /libraries
# (or the call fails). Kept in sync manually otherwise.
DEFAULT_LIBRARIES = ["fastapi", "docker", "pytorch", "pydantic", "transformers", "qdrant", "postgresql"]

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
    /* Relative score bar */
    .score-bar-track {
        background: #333;
        border-radius: 4px;
        height: 6px;
        width: 100%;
        margin: 4px 0 8px 0;
    }
    .score-bar-fill {
        background: #00CC96;
        border-radius: 4px;
        height: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────

def _num(value: Any, default: float = 0.0) -> float:
    """Coalesce None (and non-numeric junk) to a default.

    dict.get(key, default) only applies the default when the key is
    MISSING — if the API returns an explicit `null`, .get() happily
    returns None and any :.2f-style format spec on it raises. This
    normalizes both cases.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_list(value: Any) -> list:
    """Same idea as _num, but for list-typed fields (retrieved_chunks, etc.)."""
    return value if isinstance(value, list) else []


@st.cache_data(ttl=60)
def fetch_libraries() -> list[str]:
    """Fetch the live list of indexed libraries from the backend.

    Falls back to the static DEFAULT_LIBRARIES list if the endpoint is
    missing or unreachable, so this never breaks the sidebar.
    """
    try:
        resp = requests.get(f"{API_BASE_URL}/libraries", timeout=4)
        if resp.status_code == 200:
            data = resp.json()
            libs = data.get("libraries") if isinstance(data, dict) else data
            if isinstance(libs, list) and libs:
                return sorted({str(lib) for lib in libs})
    except Exception:
        pass
    return DEFAULT_LIBRARIES


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
            data = resp.json()
            return data if isinstance(data, list) else []
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
    except requests.exceptions.ConnectionError:
        st.error(f"Couldn't reach the backend at {API_BASE_URL}. Is it running?")
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
        qdrant_info = health.get("qdrant") or {}
        qdrant_points = qdrant_info.get("points_count", 0) if isinstance(qdrant_info, dict) else 0
        st.caption(f"**Qdrant:** {qdrant_points:,} points indexed")
    elif health and health.get("status") == "degraded":
        st.warning("● Backend Degraded")
    else:
        st.error("● Backend Offline")

    st.markdown("---")

    # Search configuration
    st.markdown("### ⚙️ Search Settings")

    library_options = ["All Libraries"] + fetch_libraries()
    library_filter = st.selectbox(
        "Library Filter",
        library_options,
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

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.feedback_given = set()
        st.rerun()

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
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show metadata for assistant messages
            if msg["role"] == "assistant" and "metadata" in msg:
                meta = msg["metadata"]

                # Rewritten query info
                if meta.get("rewritten_query"):
                    st.info(f"🔄 **Query rewritten to:** `{meta['rewritten_query']}`")

                # Performance metrics row
                chunks = _safe_list(meta.get("retrieved_chunks"))
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Latency", f"{_num(meta.get('latency_ms')):.0f} ms")
                mc2.metric("Strategy", str(meta.get("retrieval_method", "hybrid_rrf")).upper())
                mc3.metric("Chunks", len(chunks))

                # Retrieved chunks expander
                if chunks:
                    with st.expander(f"🔍 View {len(chunks)} Retrieved Chunks", expanded=False):
                        scores = [_num(c.get("score")) for c in chunks]
                        max_score = max(scores) if scores else 0.0

                        for i, chunk in enumerate(chunks, 1):
                            score = _num(chunk.get("score"))
                            lib = chunk.get("source_lib") or "unknown"
                            section = chunk.get("section_title") or "Document"
                            st.markdown(
                                f"**Chunk #{i}** · "
                                f"`{lib}` · `{section}` · "
                                f"Score: `{score:.4f}`"
                            )
                            # Relative score bar — helps compare chunks when
                            # the raw score scale varies by retrieval method
                            # (BM25 scores aren't 0-1 like cosine similarity is).
                            fill_pct = (score / max_score * 100) if max_score > 0 else 0
                            st.markdown(
                                f'<div class="score-bar-track">'
                                f'<div class="score-bar-fill" style="width:{fill_pct:.0f}%"></div>'
                                f"</div>",
                                unsafe_allow_html=True,
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
                        if st.button("👍 Yes", key=f"up_{log_id}_{msg_idx}"):
                            if send_feedback(log_id, 1, "Helpful answer"):
                                st.session_state.feedback_given.add(log_id)
                                st.success("Thanks! Feedback recorded.")
                                st.rerun()
                            else:
                                st.error("Couldn't record feedback — backend unreachable.")
                    with fb2:
                        if st.button("👎 No", key=f"dn_{log_id}_{msg_idx}"):
                            if send_feedback(log_id, -1, "Not helpful"):
                                st.session_state.feedback_given.add(log_id)
                                st.warning("Feedback recorded. We'll improve!")
                                st.rerun()
                            else:
                                st.error("Couldn't record feedback — backend unreachable.")
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
                response_text = result.get("response") or "No response generated."
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
                        "retrieved_chunks": _safe_list(result.get("retrieved_chunks")),
                    },
                })
                st.rerun()
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⚠️ Failed to get a response. Please check backend connectivity.",
                })
                st.rerun()


# ═══════════════════════════════════════════
# TAB 2: Monitoring Dashboard (5 Figures)
# ═══════════════════════════════════════════

with tab_monitor:
    header_col, refresh_col = st.columns([5, 1])
    with header_col:
        st.markdown(
            """
            <div class="main-header">
                <h1>📊 Telemetry & Monitoring Dashboard</h1>
                <p>Real-time analytics from SQLite telemetry logs — 5 analytical figures</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with refresh_col:
        st.write("")
        if st.button("🔄 Refresh now", use_container_width=True):
            fetch_metrics_summary.clear()
            fetch_logs.clear()
            st.rerun()

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
        kpi3.metric("Satisfaction Rate", f"{_num(summary.get('satisfaction_rate')):.1f}%")
        kpi4.metric("Avg Latency", f"{_num(summary.get('avg_latency_ms')):.0f} ms")

        st.divider()

        # Build dataframe from logs
        df = pd.DataFrame(raw_logs) if raw_logs else pd.DataFrame()
        if not df.empty and "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["datetime"])
            if not df.empty:
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
            pos = int(_num(summary.get("positive_feedback")))
            neg = int(_num(summary.get("negative_feedback")))
            no_fb = max(0, int(_num(summary.get("total_queries"))) - (pos + neg))

            if pos + neg + no_fb > 0:
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
            else:
                st.info("No feedback data available.")

        with col_b:
            st.subheader("3️⃣ Latency Distribution")
            if not df.empty and "latency_ms" in df.columns:
                p50 = _num(summary.get("p50_latency_ms"))
                p90 = _num(summary.get("p90_latency_ms"))
                p99 = _num(summary.get("p99_latency_ms"))
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
            lib_data = summary.get("lib_counts") or {}
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
                avg_c = _num(summary.get("avg_chars"))
                avg_w = _num(summary.get("avg_words"))
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

        # Free-text filter over the query column — small UX add, no behavior change otherwise.
        search_term = st.text_input("🔎 Filter logs by query text", "")
        if search_term and "query" in df_logs.columns:
            df_logs = df_logs[df_logs["query"].astype(str).str.contains(search_term, case=False, na=False)]

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

        if "timestamp" in df_logs.columns:
            df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"], errors="coerce")

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
        st.caption(f"Showing {len(df_logs)} of {len(logs)} log entries.")
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