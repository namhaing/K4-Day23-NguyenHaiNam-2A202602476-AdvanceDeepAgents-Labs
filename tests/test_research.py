"""research.py: slugify, summarize, save_outputs, main. Offline (the sandbox is faked)."""
import json

import pytest
from deepagents.backends.protocol import FileDownloadResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import research
from agents import REPORT_PATH, SOURCES_PATH


def test_slugify():
    s = research.slugify
    assert s("survey about world model") == "survey-about-world-model"
    assert s("../../x") == "x" and s(".." + chr(92) + ".." + chr(92) + "x") == "x" and s("C:/Windows") == "c-windows"
    assert s("") == "topic" and s("!!!") == "topic" and s("Tiếng Việt") == "ti-ng-vi-t"
    assert len(s("a" * 200)) == 60 and not s("a" * 59 + " b").endswith("-")


def test_summarize():
    msgs = [HumanMessage("hi"),
            AIMessage(content="", tool_calls=[{"name": "task", "args": {}, "id": "1"}, {"name": "task", "args": {}, "id": "2"},
                                              {"name": "task", "args": {}, "id": "3"}],
                      usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}),
            ToolMessage("r", tool_call_id="1"),
            AIMessage(content="", tool_calls=[{"name": "execute", "args": {}, "id": "4"}],
                      usage_metadata={"input_tokens": 5, "output_tokens": 1, "total_tokens": 6})]
    m = research.summarize(msgs, 12.34, "gpt")
    assert m == {"model": "gpt", "elapsed_s": 12.3, "subagent_calls": 3, "tool_calls": {"task": 3, "execute": 1},
                 "tokens": {"input": 15, "output": 4}}


class FakeBackend:
    def __init__(self, files): self.files = files
    def download_files(self, paths):
        return [FileDownloadResponse(path=p, content=self.files.get(p), error=None) for p in paths]


@pytest.mark.parametrize("files", [{}, {REPORT_PATH: b"  \n"}, {REPORT_PATH: b"# r"},
                                   {REPORT_PATH: b"# r", SOURCES_PATH: b"{bad"}, {REPORT_PATH: b"# r", SOURCES_PATH: b"[]"},
                                   {REPORT_PATH: b"# r", SOURCES_PATH: b'{"n": 1}'}])
def test_save_outputs_writes_nothing_on_failure(tmp_path, files):
    with pytest.raises(RuntimeError):
        research.save_outputs(FakeBackend(files), "t", [], 1, "m", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_save_outputs_ok(tmp_path):
    src = [{"n": 1, "url": "https://arxiv.org/abs/1", "source": "web"}, {"n": 2, "url": "https://h/p", "source": "hf-search"},
           {"n": 3, "url": "https://h/q", "source": "hf-search"}]
    report = "# r [1][2][3]\n\nTiếng Việt ✓\n".encode()
    p = research.save_outputs(FakeBackend({REPORT_PATH: report, SOURCES_PATH: json.dumps(src).encode()}),
                              "My Topic", [], 1, "m", tmp_path)
    meta = json.loads((tmp_path / "my-topic.meta.json").read_text(encoding="utf-8"))
    assert p.read_bytes() == report
    assert json.loads((tmp_path / "my-topic.sources.json").read_text(encoding="utf-8")) == src
    assert meta["topic"] == "My Topic" and meta["n_sources"] == 3 and meta["source_families"] == ["hf-search", "web"]
    assert {"model", "elapsed_s", "subagent_calls", "tool_calls", "tokens"} <= meta.keys()


def test_main_without_topic(capsys):
    assert research.main("   ") == 2 and "usage" in capsys.readouterr().err
