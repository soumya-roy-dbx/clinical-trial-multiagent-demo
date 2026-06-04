"""Chat sidebar component for the multi-agent system.

Renders FAQs, the input box, conversation history, and source badges. Keeps the
sidebar focused — orchestration lives in `agents.supervisor`.
"""
from __future__ import annotations

from typing import Callable

import streamlit as st

FAQ_GROUPS: dict[str, list[str]] = {
    "Event Incidence": [
        "What is the incidence of Event A in the Treatment arm?",
        "What is the incidence of Event B by sub-type?",
        "How does Event A symptomatic rate compare between Treatment and Control?",
    ],
    "Population & Demographics": [
        "How many subjects are in the Treatment group?",
        "What is the subject count by region?",
        "What is the biomarker status distribution?",
    ],
    "Protocol & CRF": [
        "What are the inclusion criteria?",
        "What are the primary and secondary endpoints?",
        "How is Event A defined in the protocol?",
    ],
    "Hybrid (Multi-Agent)": [
        "What is the incidence of Event A in the Treatment arm and how is Event A defined in the protocol?",
    ],
}


def _badge(source: str) -> str:
    color = {
        "genie": "#2DA66E",
        "vector": "#3B82F6",
        "hybrid": "#A855F7",
        "mock": "#F59E0B",
    }.get(source, "#6B7280")
    return f"<span style='background:{color};color:white;padding:2px 6px;border-radius:6px;font-size:11px;margin-right:4px'>{source}</span>"


def render_sidebar(handler: Callable[[str], dict]) -> None:
    """Render the chat sidebar. `handler(question)` should return a dict with
    keys: answer, source, confidence (optional), debug (optional)."""
    with st.sidebar:
        st.markdown("### Clinical Trial Assistant")
        st.caption("Genie: Mock · Docs: Mock")

        with st.expander("Suggested Questions", expanded=True):
            for group, questions in FAQ_GROUPS.items():
                st.markdown(f"**{group}**")
                for q in questions:
                    if st.button(q, key=f"faq_{q}", use_container_width=True):
                        st.session_state["pending_q"] = q

        st.markdown("---")
        user_q = st.text_area("Ask a question", value="", height=70,
                              placeholder="e.g., What is the Event A incidence in Treatment?")
        col1, col2 = st.columns(2)
        send = col1.button("Send", use_container_width=True)
        clear = col2.button("Clear", use_container_width=True)

        if clear:
            st.session_state["history"] = []

        if "history" not in st.session_state:
            st.session_state["history"] = []

        q = None
        if send and user_q.strip():
            q = user_q.strip()
        elif "pending_q" in st.session_state:
            q = st.session_state.pop("pending_q")

        if q:
            result = handler(q)
            st.session_state["history"].append({"q": q, "result": result})

        for item in reversed(st.session_state["history"][-5:]):
            st.markdown(f"**Q:** {item['q']}")
            r = item["result"]
            st.markdown(_badge(r.get("source", "mock")), unsafe_allow_html=True)
            st.markdown(r.get("answer", ""))
            st.markdown("---")
