"""Multi-Agent Clinical Trial Analytics App — Streamlit entry point.

Renders a Subject Disposition dashboard and mounts a multi-agent chat sidebar.
"""
from __future__ import annotations

import streamlit as st

from agents.supervisor import SupervisorAgent
from modules import db
from modules.chat_ui import render_sidebar
from modules.config import PAGE_TITLE, USE_MULTI_AGENT
from modules.observability import enable_tracing

st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# Turn on MLflow tracing once per app process. Safe no-op if MLflow or
# Databricks tracking is unavailable (see modules/observability.py).
enable_tracing()


@st.cache_resource
def get_supervisor() -> SupervisorAgent:
    return SupervisorAgent()


def handle_question(q: str) -> dict:
    if USE_MULTI_AGENT:
        resp = get_supervisor().handle(q)
        return {"answer": resp.answer, "source": resp.source,
                "confidence": resp.confidence}
    # Legacy Genie-only path could go here.
    from modules import genie_mock
    out = genie_mock.answer(q)
    return {"answer": out["answer"], "source": "genie",
            "confidence": out.get("confidence", 0.5)}


def main() -> None:
    st.title(PAGE_TITLE)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Overall population in Treatment")
        st.metric("Subject count in Treatment", db.get_overall_kpi())

    with col_right:
        st.subheader("Indicated population in Treatment (Biomarker Positive)")
        st.metric("Subject count in Treatment", db.get_indicated_kpi())

    st.markdown("---")
    population = st.selectbox("Selected Population",
                              ["Overall Population", "Indicated Population"])
    indicated = population == "Indicated Population"
    st.markdown(f"### {population} — Treatment by Region")
    st.dataframe(db.get_region_table(indicated=indicated),
                 use_container_width=True, hide_index=True)

    render_sidebar(handle_question)


if __name__ == "__main__":
    main()
