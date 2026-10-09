"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET  # arXiv answers with Atom XML

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()   # EXA_API_KEY also works when this file runs on its own (python tools.py)

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

ATOM = "{http://www.w3.org/2005/Atom}"
RETRY_STATUS = {429, 500, 502, 503, 504}           # transient: worth waiting and trying again
TIMEOUT = httpx.Timeout(30.0)
MAX_WAIT = 600                                      # a Retry-After longer than this (s) is a quota, not a hiccup
SUMMARY_CHARS = 600


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again (at most `attempts` calls in total).

    The wait is the server's Retry-After when given, else exponential backoff base * 2**attempt with random jitter;
    both are capped at `cap` seconds. The last failure is re-raised without sleeping; any other exception propagates.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = min(float(exc.retry_after), cap)
            else:
                delay = min(base * 2 ** attempt, cap)
                delay *= 0.5 + random.random() / 2         # jitter: 50-100 % of the backoff
            time.sleep(delay)


def _retry_after(response):
    """Retry-After in seconds, or None (the header may be missing or an HTTP date)."""
    try:
        return float(response.headers["Retry-After"])
    except (KeyError, ValueError):
        return None


def _check_status(response):
    """429/5xx -> RetryableError; other 4xx -> httpx.HTTPStatusError (retrying a bad request is useless).

    A Retry-After longer than MAX_WAIT means a quota is exhausted (Exa answers ~19 h): waiting cannot help, so fail
    at once and let the agent switch to another source."""
    if response.status_code in RETRY_STATUS:
        retry_after = _retry_after(response)
        if retry_after is not None and retry_after > MAX_WAIT:
            raise RuntimeError(f"HTTP {response.status_code}: rate limited for {retry_after / 3600:.1f} h "
                               f"(quota exhausted, use another source)")
        raise RetryableError(f"HTTP {response.status_code}", retry_after)
    response.raise_for_status()


def _get(url, params):
    """One GET. Network errors and 429/5xx raise RetryableError; other 4xx raise httpx.HTTPStatusError (no retry)."""
    try:
        response = httpx.get(url, params=params, timeout=TIMEOUT, follow_redirects=True)
    except httpx.TransportError as exc:
        raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
    _check_status(response)
    return response


def _clean(text, limit=None):
    """Collapse whitespace and newlines; cut to `limit` characters."""
    text = " ".join(str(text or "").split())
    return text[:limit] if limit else text


def _error(exc):
    return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 2: arXiv ----
_last_arxiv_call = 0.0                              # monotonic time of the last arXiv request
_arxiv_lock = threading.Lock()                      # researchers run in parallel threads: one arXiv request at a time
_arxiv_blocked_until = 0.0                          # circuit breaker: skip arXiv until then after a 429 give-up
ARXIV_COOLDOWN = 300                                # seconds


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Use 2-5 plain keywords (no quotes or operators).
    Returns a JSON list of {id, url, published, title, summary}."""
    global _arxiv_blocked_until
    terms = [t for t in re.findall(r"[\w-]+", query or "") if t.upper() not in {"AND", "OR", "NOT"}]
    if not terms:
        return "NO RESULTS"
    blocked_for = _arxiv_blocked_until - time.monotonic()
    if blocked_for > 0:
        return (f"ERROR: arXiv is rate limiting this machine (HTTP 429); arxiv_search is paused for {blocked_for:.0f} s. "
                "Use hf_search_papers or web_search instead.")
    params = {"search_query": " AND ".join(f"all:{t}" for t in terms), "sortBy": "submittedDate",
              "sortOrder": "descending", "max_results": max(1, min(int(max_results), 30))}

    def call():
        global _last_arxiv_call
        with _arxiv_lock:                                  # arXiv etiquette: >= 3 s between two calls, across threads
            wait = 3.0 - (time.monotonic() - _last_arxiv_call)
            if wait > 0:
                time.sleep(wait)
            _last_arxiv_call = time.monotonic()
            return _get(ARXIV_URL, params)

    try:
        root = ET.fromstring(with_retry(call, attempts=6, cap=60.0).text)   # arXiv limits per IP: be patient
        records = []
        for entry in root.iter(f"{ATOM}entry"):
            arxiv_id = re.sub(r"v\d+$", "", entry.findtext(f"{ATOM}id", "").split("/abs/")[-1])
            records.append({"id": arxiv_id, "url": f"https://arxiv.org/abs/{arxiv_id}",
                            "published": entry.findtext(f"{ATOM}published", "")[:10],
                            "title": _clean(entry.findtext(f"{ATOM}title")),
                            "summary": _clean(entry.findtext(f"{ATOM}summary"), SUMMARY_CHARS)})
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except RetryableError as exc:                          # still failing after every retry
        if "429" in str(exc):                              # blocked: stop hammering arXiv, fail fast for a while
            _arxiv_blocked_until = time.monotonic() + ARXIV_COOLDOWN
        return _error(exc)
    except Exception as exc:
        return _error(exc)


# ---- TODO 3: Hugging Face ----
def _hf_record(item):
    """One item of the daily/search endpoints -> compact record, or None when it has no paper id."""
    paper = item.get("paper") or {}
    if not paper.get("id"):
        return None
    return {"id": paper["id"], "url": f"https://huggingface.co/papers/{paper['id']}",
            "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
            "title": _clean(paper.get("title") or item.get("title")),
            "summary": _clean(paper.get("ai_summary") or paper.get("summary") or item.get("summary"), SUMMARY_CHARS),
            "upvotes": paper.get("upvotes", 0), "github": paper.get("githubRepo"), "stars": paper.get("githubStars")}


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    params = {"limit": max(1, min(int(limit), 100))}
    if date:
        params["date"] = date
    try:
        records = [r for r in map(_hf_record, with_retry(lambda: _get(HF_DAILY_URL, params)).json()) if r]
        if keyword:
            records = [r for r in records if keyword.lower() in f"{r['title']} {r['summary']}".lower()]
        records.sort(key=lambda r: r["upvotes"] or 0, reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    if not (query or "").strip():
        return "NO RESULTS"
    params = {"q": query, "limit": max(1, min(int(limit), 50))}
    try:
        records = [r for r in map(_hf_record, with_retry(lambda: _get(HF_SEARCH_URL, params)).json()) if r]
        records = records[: params["limit"]]
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _redact(text):
    """Hide EXA_API_KEY: it travels in the URL, so httpx error messages contain it."""
    key = (os.getenv("EXA_API_KEY") or "").strip()
    return text.replace(key, "***") if key else text


def _exa_call(name, arguments):
    """Call one Exa MCP tool with a plain JSON-RPC "tools/call" over HTTP and return its text content."""
    key = (os.getenv("EXA_API_KEY") or "").strip()
    url = f"{EXA_URL}?exaApiKey={key}" if key else EXA_URL
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

    def call():
        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=httpx.Timeout(60.0))
        except httpx.TransportError as exc:
            raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
        _check_status(response)
        message = None
        for line in response.text.splitlines():           # server-sent events: "data: {...}"
            if line.startswith("data:"):
                message = json.loads(line[5:])
        if message is None:                               # plain JSON answer
            message = response.json()
        if "error" in message:
            raise RuntimeError(f"MCP error: {message['error']}")
        result = message.get("result") or {}
        text = "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
        meta = json.dumps(result.get("_meta") or {}).lower()
        # the free tier signals a rate limit with HTTP 200 + a flag in result._meta (and a short notice as text)
        if ("rate" in meta and "limit" in meta) or ("rate limit" in text[:500].lower() and len(text) < 2000):
            raise RetryableError("Exa rate limited")
        if result.get("isError"):
            raise RuntimeError(text[:300])
        return text

    return with_retry(call, attempts=6, cap=60.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        text = _exa_call("web_search_exa", {"query": query, "objective": objective or f"Find pages about: {query}",
                                            "numResults": max(1, min(int(num_results), 10))})
        return text.strip() or "NO RESULTS"
    except Exception as exc:
        return _redact(_error(exc))


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        text = _exa_call("web_fetch_exa", {"urls": [url]})
        return text.strip()[:12000] or "NO RESULTS"
    except Exception as exc:
        return _redact(_error(exc))


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
