"""tools.py: retry, the five source tools, Exa and arXiv edge cases. Offline (HTTP is mocked)."""
import json

import httpx
import pytest

import tools


@pytest.fixture
def sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.time, "sleep", calls.append)
    return calls


def test_retry_gives_up_without_last_sleep(sleeps):
    calls = []
    def always():
        calls.append(1)
        raise tools.RetryableError("429")
    with pytest.raises(tools.RetryableError):
        tools.with_retry(always, attempts=4, base=1, cap=3)
    assert len(calls) == 4 and len(sleeps) == 3 and all(0.5 <= s <= 3 for s in sleeps)


def test_backoff_grows_with_jitter(sleeps):
    with pytest.raises(tools.RetryableError):
        tools.with_retry(lambda: (_ for _ in ()).throw(tools.RetryableError("x")), attempts=5, base=1, cap=100)
    assert [1 * 0.5 <= s <= 1 for s in sleeps[:1]] == [True]
    assert all(2 ** i * 0.5 <= s <= 2 ** i for i, s in enumerate(sleeps))


def test_retry_after_and_cap(sleeps):
    state = {"n": 0}
    def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise tools.RetryableError("429", retry_after=7 if state["n"] == 1 else 999)
        return "ok"
    assert tools.with_retry(flaky, cap=30) == "ok" and sleeps == [7, 30]


def test_other_errors_not_retried(sleeps):
    with pytest.raises(ValueError):
        tools.with_retry(lambda: (_ for _ in ()).throw(ValueError("bug")))
    assert sleeps == []


def test_get_status_mapping(monkeypatch):
    def resp(code, headers=None):
        return lambda url, **k: httpx.Response(code, headers=headers or {}, request=httpx.Request("GET", url))
    monkeypatch.setattr(tools.httpx, "get", resp(429, {"Retry-After": "5"}))
    with pytest.raises(tools.RetryableError) as e:
        tools._get("https://x", {})
    assert e.value.retry_after == 5.0
    monkeypatch.setattr(tools.httpx, "get", resp(404))
    with pytest.raises(httpx.HTTPStatusError):
        tools._get("https://x", {})


def test_arxiv_sanitize_no_network(monkeypatch):
    monkeypatch.setattr(tools.httpx, "get", lambda *a, **k: pytest.fail("network"))
    assert tools.arxiv_search.invoke({"query": "\"::\" AND OR"}) == "NO RESULTS"
    assert tools.hf_search_papers.invoke({"query": "  "}) == "NO RESULTS"


def test_arxiv_parse_and_spacing(monkeypatch, sleeps):
    xml = """<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2501.00001v3</id>
    <published>2025-01-02T00:00:00Z</published><title>A  World
    Model</title><summary>  lots   of
    text </summary></entry></feed>"""
    seen = []
    def fake_get(url, params=None, **k):
        seen.append(params)
        return httpx.Response(200, text=xml, request=httpx.Request("GET", url))
    monkeypatch.setattr(tools.httpx, "get", fake_get)
    monkeypatch.setattr(tools, "_last_arxiv_call", 0.0)
    rec = json.loads(tools.arxiv_search.invoke({"query": "world: 'model'", "max_results": 99}))[0]
    assert rec == {"id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "published": "2025-01-02",
                   "title": "A World Model", "summary": "lots of text"}
    assert seen[0]["search_query"] == "all:world AND all:model" and seen[0]["max_results"] == 30
    tools.arxiv_search.invoke({"query": "x"})
    assert sleeps and sleeps[-1] > 2.5          # second call waited ~3 s


def test_hf_record_shape_and_keyword(monkeypatch):
    items = [{"paper": {"id": "1", "title": "World model", "summary": "long", "ai_summary": "short", "upvotes": 3,
                        "githubRepo": "g", "githubStars": 9, "publishedAt": "2025-05-05T1"}},
             {"paper": {"id": "2", "title": "Other", "summary": "x", "upvotes": 50}}, {"paper": {}}]
    monkeypatch.setattr(tools.httpx, "get", lambda url, **k: httpx.Response(200, json=items, request=httpx.Request("GET", url)))
    allr = json.loads(tools.hf_daily_papers.invoke({}))
    assert [r["id"] for r in allr] == ["2", "1"]
    one = json.loads(tools.hf_daily_papers.invoke({"keyword": "WORLD"}))
    assert one == [{"id": "1", "url": "https://huggingface.co/papers/1", "published": "2025-05-05", "title": "World model",
                    "summary": "short", "upvotes": 3, "github": "g", "stars": 9}]
    assert tools.hf_daily_papers.invoke({"keyword": "zzz"}) == "NO RESULTS"


def test_tools_never_raise(monkeypatch, sleeps):
    def broken(*a, **k):
        raise httpx.ConnectError("down")
    monkeypatch.setattr(tools.httpx, "get", broken)
    monkeypatch.setattr(tools.httpx, "post", broken)
    for t, a in [(tools.arxiv_search, {"query": "x"}), (tools.hf_daily_papers, {}), (tools.hf_search_papers, {"query": "x"}),
                 (tools.web_search, {"query": "x"}), (tools.web_fetch, {"url": "https://x"})]:
        assert t.invoke(a).startswith("ERROR:")


def test_exa_key_redacted(monkeypatch, sleeps):
    monkeypatch.setenv("EXA_API_KEY", "secret-key-123")
    def broken(url, **k):
        raise httpx.ConnectError(f"cannot reach {url}")
    monkeypatch.setattr(tools.httpx, "post", broken)
    out = tools.web_fetch.invoke({"url": "https://x"})
    assert out.startswith("ERROR:") and "secret-key-123" not in out and "***" in out


def test_exa_rate_limit_flag_retried(monkeypatch, sleeps):
    answers = [{"result": {"content": [{"type": "text", "text": "slow down"}], "_meta": {"rateLimited": True}}},
               {"result": {"content": [{"type": "text", "text": "Title: p\nURL: https://p"}]}}]
    sent = []
    def fake_post(url, json=None, **k):
        sent.append(json)
        return httpx.Response(200, text="event: message\ndata: " + __import__("json").dumps(answers.pop(0)),
                              request=httpx.Request("POST", url))
    monkeypatch.setattr(tools.httpx, "post", fake_post)
    assert "https://p" in tools.web_search.invoke({"query": "x"}) and len(sleeps) == 1
    args = sent[0]["params"]["arguments"]
    assert sent[0]["method"] == "tools/call" and args["objective"] and args["numResults"] == 5


def test_exa_jsonrpc_error(monkeypatch, sleeps):
    monkeypatch.setattr(tools.httpx, "post", lambda url, **k: httpx.Response(
        200, text='data: {"jsonrpc":"2.0","id":1,"error":{"code":-32602,"message":"bad"}}', request=httpx.Request("POST", url)))
    assert tools.web_fetch.invoke({"url": "https://x"}).startswith("ERROR: RuntimeError: MCP error")


def test_exhausted_quota_fails_fast(monkeypatch, sleeps):
    monkeypatch.setattr(tools.httpx, "post", lambda url, **k: httpx.Response(
        429, headers={"retry-after": "68705"}, text="{}", request=httpx.Request("POST", url)))
    out = tools.web_search.invoke({"query": "x"})
    assert out.startswith("ERROR: RuntimeError: HTTP 429: rate limited for 19.1 h") and sleeps == []


@pytest.fixture(autouse=True)
def reset_arxiv(monkeypatch):
    monkeypatch.setattr(tools, "_arxiv_blocked_until", 0.0)
    monkeypatch.setattr(tools, "_last_arxiv_call", 0.0)


def test_arxiv_circuit_breaker(monkeypatch, sleeps):
    calls = []
    def blocked(url, **k):
        calls.append(1)
        return httpx.Response(429, request=httpx.Request("GET", url))
    monkeypatch.setattr(tools.httpx, "get", blocked)
    first = tools.arxiv_search.invoke({"query": "world model"})
    assert first == "ERROR: RetryableError: HTTP 429" and len(calls) == 6
    second = tools.arxiv_search.invoke({"query": "other"})
    assert second.startswith("ERROR: arXiv is rate limiting") and "hf_search_papers" in second and len(calls) == 6


def test_arxiv_spacing_across_threads(monkeypatch):
    import threading, time as real_time
    stamps = []
    def fake_get(url, **k):
        stamps.append(real_time.monotonic())
        return httpx.Response(200, text='<feed xmlns="http://www.w3.org/2005/Atom"></feed>', request=httpx.Request("GET", url))
    monkeypatch.setattr(tools.httpx, "get", fake_get)
    monkeypatch.setattr(tools, "_last_arxiv_call", real_time.monotonic() - 3)
    threads = [threading.Thread(target=tools.arxiv_search.invoke, args=({"query": f"q{i}"},)) for i in range(3)]
    [t.start() for t in threads]; [t.join() for t in threads]
    gaps = [b - a for a, b in zip(sorted(stamps), sorted(stamps)[1:])]
    assert len(stamps) == 3 and all(g >= 2.95 for g in gaps), gaps
