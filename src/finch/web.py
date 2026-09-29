"""A small, local-only browser interface for Finch.

Run with: python -m finch.web
Then visit: http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import parse_qs

from .local_model import LocalModelError
from .rag import Answer, ask
from .storage import DEFAULT_DB, Library, database_path
from .web_sources import ingest_url


def _field(values: dict[str, list[str]], name: str) -> str:
    return values.get(name, [""])[0].strip()


_MATH_SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "theta": "θ", "lambda": "λ", "mu": "μ", "pi": "π", "rho": "ρ",
    "sigma": "σ", "infty": "∞", "sum": "∑", "prod": "∏", "cdot": "·",
    "times": "×", "div": "÷", "pm": "±", "mp": "∓", "cdots": "…",
    "ldots": "…", "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥",
    "neq": "≠", "approx": "≈", "to": "→", "rightarrow": "→", "leftarrow": "←",
}
_MATH_WRAPPERS = {"text", "mathrm", "mathit", "mathbf", "operatorname"}


def _take_group(value: str, start: int) -> tuple[str, int]:
    """Return one balanced `{...}` group and the index after it."""
    if start >= len(value) or value[start] != "{":
        return "", start
    depth = 1
    index = start + 1
    while index < len(value) and depth:
        if value[index] == "{":
            depth += 1
        elif value[index] == "}":
            depth -= 1
        index += 1
    if depth:
        return value[start + 1 :], len(value)
    return value[start + 1 : index - 1], index


def _math_unit(value: str, start: int) -> tuple[str, int]:
    if start >= len(value):
        return "", start
    if value[start] == "{":
        group, end = _take_group(value, start)
        return _render_math(group), end
    return escape(value[start]), start + 1


def _render_math(value: str) -> str:
    """Render the small, common LaTex subset used in introductory finance formulas.

    Keeping this renderer local avoids loading a large remote math library just
    to present a few course formulas. Unknown input remains safely escaped.
    """
    output: list[str] = []
    index = 0
    while index < len(value):
        char = value[index]
        if char == "\\":
            command_match = re.match(r"[A-Za-z]+", value[index + 1 :])
            if not command_match:
                if index + 1 < len(value) and value[index + 1] in {",", ";", "!", " "}:
                    index += 2
                    continue
                output.append(escape(value[index + 1] if index + 1 < len(value) else "\\"))
                index += 2
                continue
            command = command_match.group(0)
            index += len(command) + 1
            if command == "frac":
                numerator, next_index = _take_group(value, index)
                denominator, end = _take_group(value, next_index)
                if next_index == index or end == next_index:
                    output.append("frac")
                else:
                    output.append(
                        "<span class='fraction'><span class='numerator'>"
                        + _render_math(numerator)
                        + "</span><span class='denominator'>"
                        + _render_math(denominator)
                        + "</span></span>"
                    )
                    index = end
            elif command == "sqrt":
                radicand, end = _take_group(value, index)
                if end != index:
                    output.append("√<span class='radicand'>" + _render_math(radicand) + "</span>")
                    index = end
                else:
                    output.append("√")
            elif command in _MATH_WRAPPERS:
                group, end = _take_group(value, index)
                if end != index:
                    output.append(_render_math(group))
                    index = end
                else:
                    output.append(escape(command))
            elif command in {"left", "right"}:
                continue
            else:
                output.append(_MATH_SYMBOLS.get(command, escape(command)))
        elif char in {"^", "_"}:
            unit, end = _math_unit(value, index + 1)
            tag = "sup" if char == "^" else "sub"
            output.append(f"<{tag}>{unit}</{tag}>")
            index = end
        elif char in "{}":
            # Braces only group LaTex input; they are not part of the formula.
            index += 1
        elif char == "~":
            output.append("&nbsp;")
            index += 1
        else:
            output.append(escape(char))
            index += 1
    return "".join(output)


def _render_plain(text: str) -> str:
    """Safely render the small Markdown subset encouraged by Finch's prompt."""
    rendered = escape(text)
    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", rendered)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", rendered)


def _render_inline(text: str) -> str:
    pieces = re.split(r"(\$[^$\n]+\$)", text)
    return "".join(
        "<span class='inline-math'>" + _render_math(piece[1:-1].strip()) + "</span>"
        if piece.startswith("$") and piece.endswith("$")
        else _render_plain(piece)
        for piece in pieces
    )


def _display_math_at(lines: list[str], start: int) -> tuple[str, int] | None:
    """Find a single-dollar or double-dollar formula that occupies its own block."""
    line = lines[start].strip()
    single = re.fullmatch(r"\$(?!\$)(.+?)(?<!\$)\$", line)
    if single:
        return single.group(1), start + 1
    if not line.startswith("$$"):
        return None
    content = line[2:]
    if content.endswith("$$") and len(content) > 2:
        return content[:-2], start + 1
    following = start + 1
    while following < len(lines):
        if lines[following].strip().endswith("$$"):
            content += " " + lines[following].strip()[:-2]
            return content, following + 1
        content += " " + lines[following]
        following += 1
    return None


def _render_answer_text(answer: str) -> str:
    """Turn a model response into readable study-note HTML without external assets."""
    lines = answer.replace("\r\n", "\n").split("\n")
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        if formula := _display_math_at(lines, index):
            math, index = formula
            blocks.append("<div class='formula' aria-label='Formula'><span class='math'>" + _render_math(math.strip()) + "</span></div>")
            continue
        if heading := re.match(r"^\s{0,3}(#{1,3})\s+(.+?)\s*$", lines[index]):
            level = min(len(heading.group(1)) + 1, 4)
            blocks.append(f"<h{level}>" + _render_inline(heading.group(2)) + f"</h{level}>")
            index += 1
            continue
        if re.match(r"^\s*[-*]\s+", lines[index]):
            items: list[str] = []
            while index < len(lines) and (item := re.match(r"^\s*[-*]\s+(.+)$", lines[index])):
                items.append("<li>" + _render_inline(item.group(1)) + "</li>")
                index += 1
            blocks.append("<ul>" + "".join(items) + "</ul>")
            continue
        if re.match(r"^\s*\d+[.)]\s+", lines[index]):
            items = []
            while index < len(lines) and (item := re.match(r"^\s*\d+[.)]\s+(.+)$", lines[index])):
                items.append("<li>" + _render_inline(item.group(1)) + "</li>")
                index += 1
            blocks.append("<ol>" + "".join(items) + "</ol>")
            continue
        paragraph: list[str] = []
        while index < len(lines) and lines[index].strip():
            if paragraph and (
                _display_math_at(lines, index)
                or re.match(r"^\s{0,3}#{1,3}\s+", lines[index])
                or re.match(r"^\s*(?:[-*]|\d+[.)])\s+", lines[index])
            ):
                break
            paragraph.append(lines[index].strip())
            index += 1
        blocks.append("<p>" + _render_inline(" ".join(paragraph)) + "</p>")
    return "".join(blocks)


def _render_answer(result: Answer | None) -> str:
    if not result:
        return ""
    sources = result.response.sources
    source_html = "<p class='muted'>No local sources supported this answer.</p>"
    if sources:
        items = []
        for number, source in enumerate(sources, start=1):
            path = str(source["path"])
            label = f"[S{number}] {source['title']} (chunk {source['chunk']})"
            if path.startswith(("https://", "http://")):
                items.append(f"<li><a href='{escape(path, quote=True)}' target='_blank' rel='noreferrer'>{escape(label)}</a></li>")
            else:
                items.append(f"<li>{escape(label)}</li>")
        source_html = "<ul>" + "".join(items) + "</ul>"
    return f"""
    <section class='answer card'>
      <div class='eyebrow'>RESPONSE {result.response.id}</div>
      <div class='answer-text'>{_render_answer_text(result.response.answer)}</div>
      <h3>Retrieved sources</h3>{source_html}
      <form method='post' class='feedback'>
        <input type='hidden' name='action' value='feedback'>
        <input type='hidden' name='response_id' value='{result.response.id}'>
        <button name='rating' value='up'>Helpful</button>
        <button name='rating' value='down' class='secondary'>Needs correction</button>
        <input name='correction' aria-label='Correction' placeholder='Optional correction to save locally'>
      </form>
    </section>"""


def _page(counts: dict[str, int], message: str = "", result: Answer | None = None) -> bytes:
    notice = f"<div class='notice'>{escape(message)}</div>" if message else ""
    answer = _render_answer(result)
    return f"""<!doctype html>
<html lang='en'>
<head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Finch RAG</title><style>
* {{ box-sizing: border-box; }} body {{ margin: 0; font-family: "Google Sans", "Segoe UI", system-ui, sans-serif; color: #1f1f1f; background: #fff; }}
main {{ max-width: 1050px; margin: auto; padding: 46px 24px 78px; }} h1 {{ font-size: clamp(2.5rem, 6vw, 4.6rem); letter-spacing: -.065em; margin: 0; font-weight: 650; }}
h2 {{ margin: 0 0 8px; font-size: 1.35rem; letter-spacing: -.02em; }} h3 {{ margin: 26px 0 7px; font-size: .95rem; }} .tag {{ color: #146c4f; font-weight: 750; letter-spacing: .11em; font-size: .74rem; }}
.intro {{ max-width: 720px; font-size: 1.13rem; line-height: 1.65; color: #444746; }} .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; margin: 28px 0; }}
.card {{ background: #fff; padding: 24px; border: 1px solid #e0e3e7; border-radius: 18px; box-shadow: 0 1px 2px #1f1f1f0a; }} label {{ display:block; font-weight: 600; margin: 15px 0 6px; }}
input, textarea {{ width: 100%; border: 1px solid #c4c7c5; border-radius: 10px; padding: 11px; font: inherit; background: #fff; }} textarea {{ min-height: 128px; resize: vertical; }}
button {{ margin-top: 14px; background: #146c4f; color: white; border: 0; border-radius: 999px; padding: 10px 17px; font: inherit; font-weight: 650; cursor: pointer; }} button.secondary {{ background: #5f6368; }}
.notice {{ background:#e6f4ea; border:1px solid #b7dfc5; padding:13px 16px; border-radius:12px; margin:22px 0; }} .muted, small {{ color:#5f6368; line-height:1.5; }}
.answer {{ margin-top: 28px; padding: clamp(26px, 4vw, 44px); max-width: 900px; border-color: #dadce0; }} .answer-text {{ max-width: 790px; font-size: 1.08rem; line-height: 1.72; }} .answer-text p {{ margin: 0 0 22px; }} .answer-text h2, .answer-text h3, .answer-text h4 {{ margin: 38px 0 13px; color: #202124; line-height: 1.3; }} .answer-text h2 {{ font-size: 1.55rem; }} .answer-text h3 {{ font-size: 1.22rem; }} .answer-text ul, .answer-text ol {{ margin: 0 0 22px; padding-left: 25px; }} .answer-text li {{ padding-left: 5px; margin: 9px 0; }} .answer-text code {{ background: #f1f3f4; padding: 2px 5px; border-radius: 4px; }}
.formula {{ margin: 27px 0; padding: 25px 16px; overflow-x: auto; text-align: center; border: 1px solid #e0e3e7; border-radius: 14px; background: #f8fafd; }} .math, .inline-math {{ font-family: "Cambria Math", Cambria, "Times New Roman", serif; }} .math {{ display: inline-block; min-width: max-content; font-size: clamp(1.25rem, 3vw, 1.65rem); letter-spacing: .015em; }} .inline-math {{ white-space: nowrap; }} .fraction {{ display: inline-flex; flex-direction: column; vertical-align: middle; text-align: center; line-height: 1.12; margin: 0 .08em; }} .numerator {{ padding: 0 .16em .08em; border-bottom: 1.5px solid currentColor; }} .denominator {{ padding: .08em .16em 0; }} .math sup, .math sub, .inline-math sup, .inline-math sub {{ font-size: .68em; line-height: 0; }} .radicand {{ border-top: 1px solid currentColor; padding-left: .09em; }} .eyebrow {{ color:#146c4f; font-weight:700; font-size:.75rem; letter-spacing:.1em; margin-bottom:16px; }}
.feedback {{ display:grid; grid-template-columns:auto auto minmax(180px,1fr); gap:8px; align-items:end; }} .feedback button {{ margin:0; }} .feedback input {{ min-width:0; }}
@media (max-width:600px) {{ .feedback {{ grid-template-columns: 1fr 1fr; }} .feedback input {{ grid-column: 1 / -1; }} }}
</style></head><body><main>
<div class='tag'>LOCAL-FIRST STUDY ASSISTANT</div><h1>Finch</h1>
<p class='intro'>Add a course webpage or a pasted note, then ask a source-grounded finance question. Sources stay in your local library. The small model only receives the few chunks needed for an answer.</p>
{notice}
<div class='grid'>
 <section class='card'><h2>Add a webpage</h2><p class='muted'>Finch fetches only the URL you submit; it does not crawl the site. HTML and plain-text pages only in this version.</p>
  <form method='post'><input type='hidden' name='action' value='url'>
   <label for='url'>Webpage URL</label><input id='url' name='url' type='url' placeholder='https://example.edu/finance/npv' required>
   <label for='url-title'>Citation title (optional)</label><input id='url-title' name='title' placeholder='Corporate finance course page'>
   <button>Save webpage locally</button></form></section>
 <section class='card'><h2>Paste a note</h2><p class='muted'>Use this for a short excerpt, instructor note, or a summary you wrote. Avoid pasting confidential or graded material.</p>
  <form method='post'><input type='hidden' name='action' value='note'>
   <label for='note-title'>Note title</label><input id='note-title' name='title' placeholder='Week 3: NPV' required>
   <label for='note'>Note text</label><textarea id='note' name='note' required></textarea><button>Save note locally</button></form></section>
</div>
<section class='card'><h2>Ask Finch</h2>
 <form method='post'><input type='hidden' name='action' value='ask'>
 <label for='question'>Question</label><textarea id='question' name='question' placeholder='Explain NPV and show the decision rule.' required></textarea>
 <label><input type='checkbox' name='sources_only' value='yes' style='width:auto'> Show sources only—do not run the language model</label>
 <button>Ask from local sources</button></form></section>
{answer}
<p class='muted'><strong>Local library:</strong> {counts['documents']} sources · {counts['chunks']} chunks · {counts['feedback']} feedback items. Feedback is stored locally and never retrains the model automatically.</p>
</main></body></html>""".encode("utf-8")


class FinchHandler(BaseHTTPRequestHandler):
    db_path: Path = DEFAULT_DB

    def log_message(self, format: str, *args: object) -> None:
        # Keep routine browser requests out of the beginner-facing terminal.
        return

    def _send_page(self, message: str = "", result: Answer | None = None) -> None:
        with Library(self.db_path) as library:
            body = _page(library.counts(), message, result)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - HTTP handler name
        self._send_page()

    def do_POST(self) -> None:  # noqa: N802 - HTTP handler name
        length = int(self.headers.get("Content-Length", "0"))
        if length > 600_000:
            self._send_page("That submission is too large. Add a smaller note or a webpage URL.")
            return
        values = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
        action = _field(values, "action")
        try:
            with Library(self.db_path) as library:
                if action == "url":
                    count = ingest_url(library, _field(values, "url"), _field(values, "title") or None)
                    message = f"Saved {count} chunks from the webpage." if count else "That webpage is already up to date."
                    result = None
                elif action == "note":
                    title = _field(values, "title")
                    source_id = "note:" + " ".join(title.lower().split())
                    count = library.ingest_text(source_id, _field(values, "note"), title)
                    message = f"Saved {count} chunks from your note." if count else "That note is already up to date."
                    result = None
                elif action == "ask":
                    question = _field(values, "question")
                    result = ask(library, question, sources_only=_field(values, "sources_only") == "yes")
                    message = "Answer generated from your local library."
                elif action == "feedback":
                    library.add_feedback(int(_field(values, "response_id")), _field(values, "rating"), _field(values, "correction"))
                    result = None
                    message = "Feedback saved locally. It will not retrain the model automatically."
                else:
                    raise ValueError("Unknown form action.")
        except (ValueError, LocalModelError) as error:
            self._send_page(f"Error: {error}")
            return
        self._send_page(message, result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local Finch web interface.")
    parser.add_argument("--db", help="SQLite database path (default: .finch/finch.sqlite3)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    FinchHandler.db_path = database_path(args.db)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), FinchHandler)
    print(f"Finch is running at http://127.0.0.1:{args.port}")
    print("Press Ctrl+C to stop it.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nFinch stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
