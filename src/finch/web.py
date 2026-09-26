"""A small, local-only browser interface for Finch.

Run with: python -m finch.web
Then visit: http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from .local_model import LocalModelError
from .rag import Answer, ask
from .storage import DEFAULT_DB, Library, database_path
from .web_sources import ingest_url


def _field(values: dict[str, list[str]], name: str) -> str:
    return values.get(name, [""])[0].strip()


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
      <div class='answer-text'>{escape(result.response.answer)}</div>
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
* {{ box-sizing: border-box; }} body {{ margin: 0; font-family: Inter, system-ui, sans-serif; color: #172121; background: #f5f6f1; }}
main {{ max-width: 1000px; margin: auto; padding: 42px 24px 70px; }} h1 {{ font-size: clamp(2.2rem, 6vw, 4.5rem); letter-spacing: -.06em; margin: 0; }}
h2 {{ margin: 0 0 8px; }} h3 {{ margin: 26px 0 7px; font-size: .95rem; }} .tag {{ color: #176b50; font-weight: 750; letter-spacing: .11em; font-size: .74rem; }}
.intro {{ max-width: 700px; font-size: 1.13rem; line-height: 1.6; }} .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; margin: 26px 0; }}
.card {{ background: white; padding: 24px; border: 1px solid #dce3dc; border-radius: 16px; box-shadow: 0 7px 22px #1832240d; }} label {{ display:block; font-weight: 650; margin: 15px 0 6px; }}
input, textarea {{ width: 100%; border: 1px solid #bdc9bf; border-radius: 8px; padding: 10px; font: inherit; background: #fbfcfa; }} textarea {{ min-height: 128px; resize: vertical; }}
button {{ margin-top: 14px; background: #176b50; color: white; border: 0; border-radius: 8px; padding: 10px 14px; font: inherit; font-weight: 700; cursor: pointer; }} button.secondary {{ background: #52635a; }}
.notice {{ background:#ddf3e6; border:1px solid #9bd1ad; padding:13px 16px; border-radius:10px; margin:20px 0; }} .muted, small {{ color:#52635a; line-height:1.45; }}
.answer {{ margin-top: 25px; }} .answer-text {{ white-space: pre-wrap; line-height: 1.6; }} .eyebrow {{ color:#176b50; font-weight:700; font-size:.75rem; letter-spacing:.1em; margin-bottom:12px; }}
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
