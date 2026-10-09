"""GUIDE part 6: shared token budget, per-run token cap, long-term memory, compaction tool. No LLM, no network."""
import json
import threading

from deepagents.backends import StateBackend
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

import agents
import research


class FakeModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def answer(input_tokens, output_tokens):
    return AIMessage(content="", usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens,
                                                 "total_tokens": input_tokens + output_tokens})


def test_budget_is_shared_warns_and_stops(capsys):
    budget = agents.TokenBudget(100)
    lead, sub = agents.TokenBudgetMiddleware(budget), agents.TokenBudgetMiddleware(budget)
    state = {"messages": [answer(30, 15)]}
    lead.after_model(state, None)
    assert sub.before_model(state, None) is None
    sub.after_model(state, None)                                   # 90 of 100: warning, still running
    assert budget.used == {"input": 60, "output": 30} and "warning" in capsys.readouterr().err
    assert lead.before_model(state, None) is None
    sub.after_model(state, None)                                   # 135 of 100: everybody stops
    for middleware in (lead, sub):
        stop = middleware.before_model(state, None)
        assert stop["jump_to"] == "end" and "shared token budget exhausted" in stop["messages"][0].content


def test_budget_counts_correctly_from_parallel_threads():
    budget = agents.TokenBudget(10**9)
    middleware = agents.TokenBudgetMiddleware(budget)
    state = {"messages": [answer(1, 1)]}

    def work():
        for _ in range(1000):
            middleware.after_model(state, None)

    threads = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert budget.used == {"input": 8000, "output": 8000}


def test_per_run_cap_reads_the_run_state():
    middleware = agents.TokenBudgetMiddleware(agents.TokenBudget(10**9), run_limit=100)
    assert middleware.before_model({"messages": [HumanMessage("go"), answer(40, 10)]}, None) is None
    stop = middleware.before_model({"messages": [HumanMessage("go"), answer(40, 10), answer(45, 10)]}, None)
    assert stop["jump_to"] == "end" and "token cap of this run" in stop["messages"][0].content
    assert middleware.before_model({"messages": [HumanMessage("new run")]}, None) is None   # next run starts at 0


def test_subagents_get_budget_cap_and_no_memory_writes():
    budget = agents.TokenBudget(10)
    for spec in agents.build_subagents(budget):
        caps = [m for m in spec["middleware"] if isinstance(m, agents.TokenBudgetMiddleware)]
        assert len(caps) == 1 and caps[0].budget is budget and caps[0].run_limit == agents.SUB_RUN_TOKENS
        (rule,) = spec["permissions"]
        assert rule.mode == "deny" and rule.operations == ["write"] and rule.paths == ["/memory/**"]


def test_lead_has_compaction_tool_and_budget(monkeypatch):
    monkeypatch.delenv("LAB_MEMORY", raising=False)
    graph = agents.build_lead_agent(StateBackend(), FakeModel(messages=iter([AIMessage(content="ok")])))
    tool_names = set(graph.get_graph().nodes) | {t for node in graph.nodes.values()
                                                for t in getattr(getattr(node, "bound", None), "tools_by_name", {})}
    assert "compact_conversation" in tool_names
    assert "Long-term memory" not in agents.LEAD_PROMPT


def test_memory_is_opt_in_and_routed_to_the_host(monkeypatch, tmp_path):
    monkeypatch.setattr(agents, "MEMORY_DIR", tmp_path / "memory")
    monkeypatch.delenv("LAB_MEMORY", raising=False)
    agents.build_lead_agent(StateBackend(), FakeModel(messages=iter([AIMessage(content="ok")])))
    assert not (tmp_path / "memory").exists()
    monkeypatch.setenv("LAB_MEMORY", "1")
    agents.build_lead_agent(StateBackend(), FakeModel(messages=iter([AIMessage(content="ok")])))
    assert (tmp_path / "memory").is_dir()


def test_meta_records_tokens_of_all_agents(tmp_path):
    from tests.test_research import FakeBackend
    sources = [{"n": 1, "url": "https://arxiv.org/abs/1", "source": "arxiv"}]
    backend = FakeBackend({agents.REPORT_PATH: b"# r [1]", agents.SOURCES_PATH: json.dumps(sources).encode()})
    research.save_outputs(backend, "t", [], 1, "m", tmp_path, extra_meta={"tokens_all_agents": {"input": 5, "output": 2}})
    meta = json.loads((tmp_path / "t.meta.json").read_text(encoding="utf-8"))
    assert meta["tokens_all_agents"] == {"input": 5, "output": 2} and meta["n_sources"] == 1
