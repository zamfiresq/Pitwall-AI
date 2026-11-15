"""LLM agent loop: the model chooses tools, we run them, and the answer is checked for grounding.

Works with any OpenAI-compatible chat endpoint that supports tool calling, for example OpenAI itself or a
local server such as Ollama running a Granite model. Configure it with environment variables or a ``.env`` file:

    PITWALL_LLM_API_KEY      (or OPENAI_API_KEY)
    PITWALL_LLM_BASE_URL     optional, e.g. http://localhost:11434/v1 for Ollama
    PITWALL_LLM_MODEL        model name, default gpt-4o-mini
    PITWALL_LLM_TEMPERATURE  default 0; set to "none" for models that reject the parameter
    PITWALL_LLM_TIMEOUT      request timeout in seconds, default 120

Accuracy measures built into the loop:
  * every tool result carries a ready-made ``summary`` and a list of ``caveats`` computed from the numbers,
    and the prompt tells the model to use only those, so it has nothing to invent;
  * if the answer contains a figure that is not in the tool results, the model gets one chance to fix it;
  * the "Sources" line is added by code from the tool results, never trusted to the model.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from src.assistant.grounding import check_grounding
from src.assistant.tools import TOOL_SPECS, PitwallData, call_tool

SYSTEM_PROMPT = """You are Pitwall AI, an assistant for an analysis of Scuderia Ferrari tyre degradation and race pace \
(Charles Leclerc and Lewis Hamilton, 2025 and 2026 seasons).

How to answer:
1. Call the tools to get facts. Never use memory or outside knowledge for any figure.
2. Build the answer from each tool result's `summary`, and mention only the warnings listed in its `caveats`. Do not add \
reasons, explanations or interpretations that are not in the tool result (for example, never explain why an interval is wide).
3. If the user does not give a season, call the tool for both 2025 and 2026 and answer for both.
4. Quote numbers exactly as the tool result gives them. Do not round differently and do not compute new figures.
5. If a tool returns an error, or the question is outside the data (other drivers or teams, other seasons, future results, \
opinions), say so in one sentence and mention what you can answer. Do not guess.
6. Say "95% interval" (it is a bootstrap interval, not a classical confidence interval). Slopes are estimates of \
association, not exact tyre wear.
7. Be concise: at most 80 words, no headings. Do not write a sources line; sources are added automatically.

Answer shapes (placeholders, not data):
- Single estimate: "In <season> the <compound> compound lost <slope> s per tyre lap (95% interval <low> to <high>), from <n> stints."
- Comparison: "<Driver> has the higher median slope (<a> vs <b> s per tyre lap), <caveat from the tool, if any>."
- Out of scope: "I can't answer that from this data. I can tell you about <what the tools cover>."
"""

SOURCES_LINE = re.compile(r"\n?[ \t]*[*_]{0,2}sources?[*_]{0,2}[ \t]*:.*$", re.I | re.S)


@dataclass
class LLMConfig:
    api_key: str | None
    base_url: str | None
    model: str
    temperature: float | None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        base_url = os.getenv("PITWALL_LLM_BASE_URL") or None
        key = os.getenv("PITWALL_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if base_url and not key:
            key = "not-needed"  # local servers ignore the key but the SDK requires one
        raw_temperature = os.getenv("PITWALL_LLM_TEMPERATURE", "0").strip().lower()
        return cls(
            api_key=key, base_url=base_url, model=os.getenv("PITWALL_LLM_MODEL", "gpt-4o-mini"),
            temperature=None if raw_temperature in {"none", ""} else float(raw_temperature))

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


def make_client(config: LLMConfig):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The assistant needs the OpenAI SDK: pip install openai") from exc
    # Local models can take longer to load and reason than hosted endpoints.
    # Keep requests bounded, but let users tune the limit for their hardware.
    timeout = float(os.getenv("PITWALL_LLM_TIMEOUT", "120"))
    return OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=timeout, max_retries=0)


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    grounding: dict = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    cited: bool = False
    repaired: bool = False
    steps: int = 0


def _successful(calls: list[dict]) -> list[dict]:
    return [c["result"] for c in calls if "error" not in c["result"]]


def _sources(results: list[dict]) -> list[str]:
    found: list[str] = []
    for result in results:
        source = result.get("source")
        if source and source not in found:
            found.append(source)
    return found


def _repair_message(calls: list[dict], ungrounded: list[float]) -> str:
    if not calls:
        return "You answered without using the tools. Call the relevant tool first and answer only from its result."
    figures = ", ".join(f"{value:g}" for value in ungrounded)
    return (f"These figures are not in the tool results: {figures}. Rewrite the answer using only values and warnings "
            "from the tool results, and remove anything else.")


def answer_question(question: str, data: PitwallData, client, model: str, temperature: float | None = 0.0,
                    history: list[dict] | None = None, max_steps: int = 3, repair: bool = True) -> AgentResult:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *(history or []), {"role": "user", "content": question}]
    calls: list[dict] = []
    text = ""
    repaired = finished = False
    for _ in range(max_steps):
        options = {"model": model, "messages": messages, "tools": TOOL_SPECS}
        if temperature is not None:
            options["temperature"] = temperature
        message = client.chat.completions.create(**options).choices[0].message
        requested = getattr(message, "tool_calls", None) or []
        if not requested:
            text = SOURCES_LINE.sub("", (message.content or "")).strip()
            check = check_grounding(text, _successful(calls))
            if repair and not repaired and not check["grounded"]:
                repaired = True
                messages.append({"role": "assistant", "content": message.content or ""})
                messages.append({"role": "user", "content": _repair_message(calls, check["ungrounded"])})
                continue
            finished = True
            break
        messages.append({
            "role": "assistant", "content": message.content or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in requested],
        })
        for call in requested:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            result = call_tool(data, call.function.name, arguments)
            calls.append({"name": call.function.name, "arguments": arguments, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
    if not finished and not text:
        text = "I could not finish answering within the allowed number of tool calls."

    results = _successful(calls)
    sources = _sources(results)
    return AgentResult(
        answer=text, tool_calls=calls, grounding=check_grounding(text, results), sources=sources,
        cited=bool(sources), repaired=repaired, steps=len(calls))
