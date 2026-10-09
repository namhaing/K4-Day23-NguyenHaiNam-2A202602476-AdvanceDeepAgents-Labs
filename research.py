"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import (FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent,
                    make_budget)
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator
RECURSION_LIMIT = 1000   # LangGraph steps of the LEAD graph (~2 per model -> tool turn); subagents have their own limits
MODEL_TIMEOUT = 180      # seconds per LLM request (a long report-writing answer stays well below it)


def log(message):
    print(f"[research] {message}", file=sys.stderr, flush=True)


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(topic).lower()).strip("-")[:60].strip("-")
    return slug or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent (the workflow itself lives in the lead's system prompt)."""
    return (f"Research topic: {topic}\n"
            "Write the survey report following your workflow: plan, delegate in parallel, check the notes, merge "
            "sources.json, write the body, finalize, validate until OK, spot-check.")


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    Lead messages only: subagent tokens are not included, so this undercounts the real cost.
    """
    tool_calls, tokens = Counter(), {"input": 0, "output": 0}
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            tool_calls[call["name"]] += 1
        usage = getattr(message, "usage_metadata", None) or {}
        tokens["input"] += usage.get("input_tokens", 0) or 0
        tokens["output"] += usage.get("output_tokens", 0) or 0
    return {"model": model_name, "elapsed_s": round(elapsed, 1), "subagent_calls": tool_calls.get("task", 0),
            "tool_calls": dict(tool_calls), "tokens": tokens}


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS, extra_meta=None):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    A missing/empty report or a missing/invalid sources.json raises RuntimeError and writes NOTHING: a failed run must
    never leave an empty or half-written report behind.
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report, raw_sources = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not report or not report.strip():
        raise RuntimeError("the agent produced no report")
    try:
        sources = json.loads(raw_sources or b"")
    except ValueError as exc:
        raise RuntimeError(f"sources.json is missing or invalid: {exc}") from exc
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("sources.json is not a non-empty JSON list")

    slug = slugify(topic)
    meta = {"topic": topic, **summarize(messages, elapsed, model_name), **(extra_meta or {}), "n_sources": len(sources),
            "source_families": sorted({s["source"] for s in sources if isinstance(s, dict) and s.get("source")})}
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{slug}.sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
    (reports_dir / f"{slug}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    path = reports_dir / f"{slug}.md"
    path.write_bytes(report)                     # byte-for-byte the file downloaded from the sandbox
    return path


def model_name_of(model):
    return getattr(model, "model_name", None) or getattr(model, "model", None) or type(model).__name__


def with_request_timeout(model, seconds=MODEL_TIMEOUT, retries=2):
    """Give every LLM request a timeout. model.py (provided, not editable) builds OpenAI models with timeout=None,
    which means NO timeout: one stalled connection then blocks the whole run forever (seen in a real run)."""
    if getattr(model, "root_client", None) is not None:
        model.root_client = model.root_client.with_options(timeout=seconds, max_retries=retries)
        model.client = model.root_client.chat.completions
    if getattr(model, "root_async_client", None) is not None:
        model.root_async_client = model.root_async_client.with_options(timeout=seconds, max_retries=retries)
        model.async_client = model.root_async_client.chat.completions
    return model


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic = (topic or "").strip()
    if not topic:
        print('usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    model = with_request_timeout(make_model())
    budget = make_budget()                       # one token counter for the lead and every subagent
    start = time.monotonic()
    with open_sandbox() as backend:              # the sandbox is always stopped and removed, even on errors
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
        log(f"sandbox ready; lead agent ({model_name_of(model)}) working on: {topic}")
        agent = build_lead_agent(backend, model, budget)
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                  config={"recursion_limit": RECURSION_LIMIT})
        except Exception as exc:                 # e.g. GraphRecursionError: a half-finished run saves nothing
            print(f"FAILED: agent stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        try:
            path = save_outputs(backend, topic, result["messages"], time.monotonic() - start, model_name_of(model),
                                extra_meta={"tokens_all_agents": dict(budget.used)})
        except RuntimeError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
    print(f"report saved: {path.relative_to(ROOT)}  ({time.monotonic() - start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
