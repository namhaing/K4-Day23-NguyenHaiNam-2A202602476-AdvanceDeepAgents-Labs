"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

# Citations are read exactly like finalize_citations.py reads them, so the two scripts never disagree.
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)      # code spans/blocks: [n] inside them is not a citation
_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")  # [3] [1, 2] [1-3], but not a Markdown link [3](url)
_REF_LINE = re.compile(r"^\[(\d+)\]")
_URL = re.compile(r"https?://[^\s)>\]]+")
# the grader checks that a family label matches its url (RUBRIC 2.2); "web" may hold any url
FAMILY_URL = {"arxiv": "https://arxiv.org/abs/", "hf-daily": "https://huggingface.co/papers/",
              "hf-search": "https://huggingface.co/papers/", "web": "http"}


def _numbers(group):
    """'1, 3-5' -> [1, 3, 4, 5]"""
    out = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            out.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            out.append(int(part))
    return out


def _structure_problems(body):
    """REPORT_TEMPLATE.md: TL;DR, Background, 3 to 6 theme sections, Trends and open problems; headings on their own
    lines. (The template is graded by a person; checking it here makes the agent fix it before the validator says OK.)"""
    problems = [f"heading glued to the end of a paragraph: {m.group(0).strip()[:60]!r} (put it on its own line)"
                for m in re.finditer(r"(?m)^(?!#).*\S[ \t]+##[ \t]+\S.*$", body)]
    titles = [t.strip() for t in re.findall(r"(?m)^##[ \t]+(.+)$", body)]
    for required in ("TL;DR", "Background", "Trends and open problems"):
        if required not in titles:
            problems.append(f"missing section '## {required}'")
    themes = [t for t in titles if t not in ("TL;DR", "Background", "Trends and open problems")]
    if not 3 <= len(themes) <= 6:
        problems.append(f"{len(themes)} theme sections between Background and Trends (REPORT_TEMPLATE needs 3 to 6)")
    return problems


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    Rules: sources non-empty, each with an int `n`, an http(s) `url`, no duplicate url, and a `source` family whose url
    matches it (arxiv -> https://arxiv.org/abs/..., hf-* -> https://huggingface.co/papers/...); a `## References`
    heading and the sections of REPORT_TEMPLATE.md; every [n] of the body is a source and every source is cited;
    References holds exactly one line per source, with exactly one URL equal to that source's url. Grouped citations
    [1, 2] / [1-3] are expanded.
    """
    problems = []
    if not sources:
        return ["no sources in sources.json"]

    # 1. sources.json itself
    by_n, seen_urls = {}, {}
    for entry in sources:
        n, url = entry.get("n"), entry.get("url")
        if not isinstance(n, int) or isinstance(n, bool):
            problems.append(f"source {n!r}: n is not an integer")
            continue
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"source [{n}]: url {url!r} is not http(s)")
        elif url in seen_urls:
            problems.append(f"source [{n}]: url duplicates source [{seen_urls[url]}]")
        else:
            seen_urls[url] = n
        family = entry.get("source")
        if family not in FAMILY_URL:
            problems.append(f"source [{n}]: source {family!r} is not one of {sorted(FAMILY_URL)}")
        elif isinstance(url, str) and not url.startswith(FAMILY_URL[family]):
            problems.append(f"source [{n}]: a {family} source needs a {FAMILY_URL[family]}<id> url, got {url} "
                            f"(fix the url or relabel the source as web)")
        by_n[n] = entry

    # 2. split the report into body and reference list
    headings = list(_REF_HEADING.finditer(report_text))
    if not headings:
        return problems + ["missing '## References' heading"]
    body, refs = report_text[:headings[-1].start()], report_text[headings[-1].end():]
    problems += _structure_problems(_CODE.sub("", body))

    # 3. citations of the body <-> sources
    cited = set()
    for match in _GROUP.finditer(_CODE.sub("", body)):
        cited.update(_numbers(match.group(1)))
    problems += [f"[{n}] cited but missing from sources.json" for n in sorted(cited - by_n.keys())]
    problems += [f"source [{n}] never cited" for n in sorted(by_n.keys() - cited)]

    # 4. reference lines: one per source, exactly one URL, equal to the source's url
    ref_lines = {}
    for line in refs.splitlines():
        match = _REF_LINE.match(line.strip())
        if not match:
            continue
        n = int(match.group(1))
        if n in ref_lines:
            problems.append(f"reference [{n}] appears more than once")
            continue
        ref_lines[n] = line
        if n not in by_n:
            problems.append(f"reference [{n}] is not a source in sources.json")
            continue
        urls = _URL.findall(line)
        if len(urls) != 1:
            problems.append(f"reference [{n}] has {len(urls)} URLs (need exactly 1)")
        elif urls[0].rstrip(".,;") != by_n[n].get("url"):
            problems.append(f"reference [{n}] url {urls[0]} != sources.json url {by_n[n].get('url')}")
    problems += [f"source [{n}] has no reference line" for n in sorted(by_n.keys() - ref_lines.keys())]
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
