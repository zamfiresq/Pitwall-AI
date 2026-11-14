"""LLM agent loop: the model chooses tools, we run them, and the answer is checked for grounding.

Works with any OpenAI-compatible chat endpoint that supports tool calling, for example OpenAI itself or a
local server such as Ollama running a Granite model. Configure it with environment variables or a ``.env`` file:

    PITWALL_LLM_API_KEY      (or OPENAI_API_KEY)
    PITWALL_LLM_BASE_URL     optional, e.g. http://localhost:11434/v1 for Ollama
    PITWALL_LLM_MODEL        model name, default gpt-4o-mini
    PITWALL_LLM_TEMPERATURE  default 0; set to "none" for models that reject the parameter
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from src.assistant.grounding import check_grounding, cites_source
from src.assistant.tools import TOOL_SPECS, PitwallData, call_tool

SYSTEM_PROMPT = """You are Pitwall AI, an assistant for an analysis of Scuderia Ferrari tyre degradation and race pace \
(Charles Leclerc and Lewis Hamilton, 2025 and 2026 seasons).

Rules:
- Answer ONLY from tool results. Call tools to get numbers; never use outside knowledge or memory for figures.
- If the tools cannot answer (other drivers or teams, other seasons, future results, opinions about people), say so \
briefly and mention what you can answer instead. Do not guess.
- Report degradation in seconds per tyre lap with 3 decimals, with the 95% interval when available. Say when only a \
few stints back a number, because the interval is then wide. Describe results as estimates of association, not exact tyre wear.
- Keep the answer under 120 words. Finish with one line: "Sources:" followed by the exact `source` values from the tool results you used."""


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
    # Bound local-model waits and disable automatic retries so a stalled Ollama
    # request cannot leave the Streamlit interaction spinning indefinitely.
    return OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=30.0, max_retries=0)


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    grounding: dict = field(default_factory=dict)
    cited: bool = False
    steps: int = 0


def answer_question(question: str, data: PitwallData, client, model: str, temperature: float | None = 0.0,
                    history: list[dict] | None = None, max_steps: int = 3) -> AgentResult:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *(history or []), {"role": "user", "content": question}]
    calls: list[dict] = []
    text = ""
    for step in range(1, max_steps + 1):
        options = {"model": model, "messages": messages, "tools": TOOL_SPECS}
        if temperature is not None:
            options["temperature"] = temperature
        message = client.chat.completions.create(**options).choices[0].message
        requested = getattr(message, "tool_calls", None) or []
        if not requested:
            text = (message.content or "").strip()
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
    else:
        text = "I could not finish answering within the allowed number of tool calls."

    results = [c["result"] for c in calls if "error" not in c["result"]]
    return AgentResult(
        answer=text, tool_calls=calls, grounding=check_grounding(text, results),
        cited=cites_source(text, results) if results else False, steps=len(calls))
