"""Minimal chat panel for the dashboard.

A fixed-height, scrollable transcript (only this box scrolls, not the page) with plain question and answer
lines, sources tucked into a small collapsible line, and the input pinned right below. No avatars, no boxes.
The panel injects its own CSS, written with ``light-dark()`` like the rest of the theme.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src import theme
from src.assistant.agent import LLMConfig, answer_question

CHAT_HEIGHT = 420
HISTORY_TURNS = 3
EXAMPLES = [
    "Which Ferrari driver degrades the hard tyre more in 2026?",
    "How did soft-tyre degradation change from 2025 to 2026?",
    "What was the fastest clean lap at Monza in 2025?",
]

_CSS = """
<style>
.pw-q {
  font-size: 0.8rem; font-weight: 700; letter-spacing: 0.03em;
  color: light-dark(#5E5E6B, #9B9BA8); margin: 0 0 0.3rem 0; padding-top: 0.2rem;
}
.pw-q::before { content: "›"; color: __RED__; margin-right: 0.5rem; font-weight: 800; }
.pw-q.pw-sep {
  margin-top: 0.9rem; padding-top: 0.9rem;
  border-top: 1px solid light-dark(rgba(0,0,0,0.09), rgba(255,255,255,0.10));
}
.pw-meta { font-size: 0.72rem; color: light-dark(#5E5E6B, #9B9BA8); margin-top: 0.15rem; line-height: 1.5; }
.pw-meta summary { cursor: pointer; list-style: none; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.85; }
.pw-meta summary::-webkit-details-marker { display: none; }
.pw-meta summary::after { content: " +"; }
.pw-meta[open] summary::after { content: " –"; }
.pw-meta .pw-tools { opacity: 0.7; }
.pw-warn { font-size: 0.78rem; color: light-dark(#8A6A00, #FFF200); margin-top: 0.2rem; }
</style>
"""

_SCROLL = """
<script>
  const end = window.parent.document.getElementById("pw-chat-end");
  if (end) { end.scrollIntoView({block: "nearest"}); }
</script>
"""


def _question(text: str, first: bool) -> None:
    css = "pw-q" if first else "pw-q pw-sep"
    st.markdown(f'<div class="{css}">{html.escape(text)}</div>', unsafe_allow_html=True)


def _answer(turn: dict) -> None:
    st.markdown(turn["answer"])
    if turn.get("ungrounded"):
        figures = ", ".join(f"{value:g}" for value in turn["ungrounded"])
        st.markdown(f'<div class="pw-warn">Not found in the tool results: {html.escape(figures)}</div>', unsafe_allow_html=True)
    if turn.get("sources"):
        sources = "<br>".join(html.escape(s) for s in turn["sources"])
        tools = "<br>".join(html.escape(t) for t in turn.get("tools", []))
        st.markdown(
            f'<details class="pw-meta"><summary>Sources</summary>{sources}'
            f'<div class="pw-tools">{tools}</div></details>',
            unsafe_allow_html=True,
        )


def _tool_label(call: dict) -> str:
    arguments = ", ".join(f"{key}={value}" for key, value in call["arguments"].items())
    return f"{call['name']}({arguments})"


def _eval_panel(report: dict | None) -> None:
    if not report:
        return
    with st.expander("Assistant evaluation"):
        cards = st.columns(4)
        cards[0].metric("Fully correct", f"{report['overall_pass_rate']:.0%}")
        cards[1].metric("Right tool", f"{report['tool_accuracy']:.0%}" if report.get("tool_accuracy") is not None else "—")
        cards[2].metric("Numbers grounded", f"{report['grounded_rate']:.0%}" if report.get("grounded_rate") is not None else "—")
        cards[3].metric("Needed a fix", f"{report['repair_rate']:.0%}" if report.get("repair_rate") is not None else "—")
        st.caption(f"{report['questions']} questions scored automatically against values computed from the data, model: {report['model']}.")
        st.dataframe(pd.DataFrame(
            [{"Category": k, "Fully correct": v} for k, v in report["by_category"].items()]), hide_index=True)


def render(load_data, load_client, report: dict | None = None) -> None:
    """Draw the whole "Ask Pitwall" tab. ``load_data`` and ``load_client`` are the app's cached loaders."""
    st.markdown(_CSS.replace("__RED__", theme.RED), unsafe_allow_html=True)
    st.subheader("Ask Pitwall")
    theme.note(
        "Ask about tyre degradation, driver comparisons, race summaries or the pace model. Answers use only numbers "
        "returned by the analysis tools, and any figure that is not in them is flagged."
    )
    config = LLMConfig.from_env()
    if not config.configured:
        theme.callout(
            "<b>No language model configured.</b> Create a <code>.env</code> file in the project root and set "
            "<code>PITWALL_LLM_API_KEY</code> for a hosted provider, or set <code>PITWALL_LLM_BASE_URL</code> and "
            "<code>PITWALL_LLM_MODEL</code> for local Ollama. Restart the app after changing the file."
        )
        _eval_panel(report)
        return

    turns: list[dict] = st.session_state.setdefault("chat_turns", [])
    box = st.container(height=CHAT_HEIGHT, border=False)
    typed = st.chat_input("Ask about tyre degradation or race pace")
    question = typed or st.session_state.pop("chat_pending", None)

    with box:
        if not turns and not question:
            st.markdown('<div class="pw-meta">Try one of these</div>', unsafe_allow_html=True)
            for index, example in enumerate(EXAMPLES):
                if st.button(example, key=f"chat_example_{index}"):
                    st.session_state["chat_pending"] = example
                    st.rerun()
        for index, turn in enumerate(turns):
            _question(turn["question"], first=index == 0)
            _answer(turn)
        if question:
            _question(question, first=not turns)
            history = [m for t in turns[-HISTORY_TURNS:]
                       for m in ({"role": "user", "content": t["question"]}, {"role": "assistant", "content": t["answer"]})]
            try:
                with st.spinner("Checking the data..."):
                    result = answer_question(
                        question, load_data(), load_client(), config.model, config.temperature, history)
                turn = {
                    "question": question, "answer": result.answer, "sources": result.sources,
                    "tools": [_tool_label(call) for call in result.tool_calls],
                    "ungrounded": result.grounding.get("ungrounded", []),
                }
                turns.append(turn)
                _answer(turn)
            except Exception as exc:
                st.error(f"The language model call failed: {exc}")
        st.markdown('<div id="pw-chat-end"></div>', unsafe_allow_html=True)

    if turns:
        components.html(_SCROLL, height=0)
        if st.button("Clear conversation", key="chat_clear"):
            st.session_state["chat_turns"] = []
            st.rerun()
    _eval_panel(report)
