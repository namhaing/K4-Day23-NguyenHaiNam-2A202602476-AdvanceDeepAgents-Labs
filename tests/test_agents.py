"""agents.py: subagent specs, limits and prompts. No LLM call: a fake chat model builds the graph."""
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import agents
import tools


class FakeModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_subagents_tools_and_limits():
    specs = {spec["name"]: spec for spec in agents.build_subagents()}
    assert set(specs) == {"researcher", "citation-checker"}
    assert specs["researcher"]["tools"] == tools.SOURCE_TOOLS
    assert specs["citation-checker"]["tools"] == [tools.web_fetch]
    for spec in specs.values():
        assert spec["description"] and spec["system_prompt"]
        assert {type(m).__name__ for m in spec["middleware"]} == {"ModelCallLimitMiddleware", "ToolCallLimitMiddleware"}
    assert specs["researcher"]["middleware"][0] is not specs["citation-checker"]["middleware"][0]   # own counters


def test_lead_prompt_carries_the_contract():
    for needle in ("write_todos", "ONE message", agents.NOTES_DIR, agents.SOURCES_PATH, agents.REPORT_PATH,
                   agents.FINALIZER_PATH, agents.VALIDATOR_PATH, "citation-checker", "## References",
                   "## TL;DR", "## Trends and open problems", agents.NOTE_FORMAT):
        assert needle in agents.LEAD_PROMPT, needle


def test_researcher_prompt_rules():
    for needle in ("UNTRUSTED", "from memory", "arxiv_search -> arxiv", "hf_search_papers -> hf-search",
                   agents.NOTE_FORMAT):
        assert needle in agents.RESEARCHER_PROMPT, needle
    assert "UNTRUSTED" in agents.CHECKER_PROMPT


def test_lead_agent_builds():
    graph = agents.build_lead_agent(None, FakeModel(messages=iter([AIMessage(content="done")])))
    assert type(graph).__name__ == "CompiledStateGraph"
