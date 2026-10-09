"""check_citations.py: the six citation rules of GUIDE part 4, plus family/url labels and the report structure."""
from check_citations import check
from finalize_citations import finalize

SOURCES = [
    {"n": 1, "id": "a", "url": "https://arxiv.org/abs/2501.00001", "title": "A", "date": "2025-01-01", "source": "arxiv"},
    {"n": 2, "id": "b", "url": "https://huggingface.co/papers/2502.00002", "title": "B", "date": "2025-02-02",
     "source": "hf-search"},
    {"n": 3, "id": "c", "url": "https://example.org/c", "title": "C", "date": "2025-03-03", "source": "web"},
]
BODY = ("# T\n\n## TL;DR\n- x [3]\n- y [1, 2]\n\n## Background\nz [2-3]. `[9]` is code, [4](https://link) a link.\n\n"
        "## Theme A\na [1]\n\n## Theme B\nb [2]\n\n## Theme C\nc [3]\n\n## Trends and open problems\nt [1]\n")


def finalized(body=BODY, sources=SOURCES):
    report, new_sources, problems = finalize(body, sources)
    assert problems == []
    return report, new_sources


def test_finalized_report_passes():
    assert check(*finalized()) == []


def test_empty_sources_and_missing_heading():
    assert check("x", []) == ["no sources in sources.json"]
    assert "missing '## References' heading" in check(BODY, SOURCES)


def test_source_entries():
    bad = [SOURCES[0], dict(SOURCES[0], n=2), {"n": "3", "url": "https://x.org", "source": "web"},
           {"n": 4, "url": "ftp://x", "source": "web"}, {"n": True, "url": "https://y.org", "source": "web"}]
    problems = check("[1][2][4]\n## References\n", bad)
    assert "source [2]: url duplicates source [1]" in problems
    assert "source '3': n is not an integer" in problems and "source True: n is not an integer" in problems
    assert any("source [4]: url 'ftp://x' is not http(s)" in p for p in problems)


def test_cited_but_missing_and_never_cited():
    report, sources = finalized()
    problems = check(report.replace("- x [1]", "- x [1] [7]"), sources)
    assert problems == ["[7] cited but missing from sources.json"]
    problems = check(report, sources + [{"n": 4, "url": "https://d.org", "source": "web"}])
    assert "source [4] never cited" in problems and "source [4] has no reference line" in problems


def test_reference_lines():
    report, sources = finalized()
    first_url = sources[0]["url"]
    assert check(report.replace(f"{first_url} ", f"{first_url} https://other.org "), sources) == [
        "reference [1] has 2 URLs (need exactly 1)"]
    assert any("!= sources.json url" in p for p in check(report.replace(first_url, "https://wrong.org/x"), sources))
    assert check(report + f"[1] again. {first_url} (2025)\n", sources) == ["reference [1] appears more than once"]
    assert check(report + "[9] ghost. https://ghost.org (2025)\n", sources) == [
        "reference [9] is not a source in sources.json"]
    without_line_2 = "\n".join(line for line in report.splitlines() if not line.startswith("[2]"))
    assert check(without_line_2, sources) == ["source [2] has no reference line"]


def test_family_must_match_url():
    relabeled = [dict(SOURCES[0], source="hf-daily"), dict(SOURCES[1], url="https://arxiv.org/html/2502.00002"),
                 dict(SOURCES[2], source="blog")]
    problems = check(*finalized(sources=relabeled))
    assert any("a hf-daily source needs a https://huggingface.co/papers/" in p for p in problems)
    assert any("a hf-search source needs a https://huggingface.co/papers/" in p for p in problems)
    assert any("source 'blog' is not one of" in p for p in problems)
    assert check(*finalized(sources=[*SOURCES[:2], dict(SOURCES[2], url="https://arxiv.org/pdf/1.pdf")])) == []


def test_report_structure():
    two_themes = BODY.replace("## Theme C\nc [3]\n\n", "")
    assert check(*finalized(two_themes)) == [
        "2 theme sections between Background and Trends (REPORT_TEMPLATE needs 3 to 6)"]
    glued = BODY.replace("a [1]\n\n## Theme B", "a [1] ## Theme B")
    assert any("heading glued to the end of a paragraph" in p for p in check(*finalized(glued)))
    no_tldr = BODY.replace("## TL;DR\n", "")
    assert "missing section '## TL;DR'" in check(*finalized(no_tldr))
    code_heading = BODY + "\n```\nfoo ## Not a heading\n```\n"
    assert check(*finalized(code_heading)) == []
