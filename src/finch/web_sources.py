"""Explicit, small-scale web-page ingestion for a student's local source library."""

from __future__ import annotations

from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .storage import Library


MAX_DOWNLOAD_BYTES = 1_000_000


class _ReadableTextParser(HTMLParser):
    """A dependency-free, deliberately conservative extractor for basic web pages."""

    _IGNORED = {"script", "style", "svg", "noscript", "template"}
    _BLOCKS = {"article", "br", "div", "h1", "h2", "h3", "h4", "li", "p", "section", "td"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._IGNORED:
            self._ignored_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._IGNORED and self._ignored_depth:
            self._ignored_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        if not self._ignored_depth:
            self.parts.append(data)

    def readable_text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self.parts).splitlines())
        return "\n\n".join(line for line in lines if line)


def _validated_url(value: str) -> str:
    url = value.strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a complete http:// or https:// webpage URL.")
    return url


def fetch_page(url: str) -> tuple[str, str, str]:
    """Fetch one user-requested HTML/text page. This never crawls linked pages."""
    url = _validated_url(url)
    request = Request(url, headers={"User-Agent": "FinchRAG/0.2 (local student library)"})
    try:
        with urlopen(request, timeout=20) as response:  # noqa: S310 - URL was explicitly supplied by user
            final_url = _validated_url(response.geturl())
            content_type = response.headers.get_content_type()
            if content_type not in {"text/html", "text/plain"}:
                raise ValueError("This URL is not an HTML or plain-text page. Download PDFs and ingest them after PDF support is added.")
            raw = response.read(MAX_DOWNLOAD_BYTES + 1)
            if len(raw) > MAX_DOWNLOAD_BYTES:
                raise ValueError("This page exceeds Finch's 1 MB source limit.")
            charset = response.headers.get_content_charset() or "utf-8"
    except HTTPError as error:
        raise ValueError(f"The webpage returned HTTP {error.code}.") from error
    except URLError as error:
        raise ValueError(f"Could not retrieve that webpage: {error.reason}") from error

    try:
        text = raw.decode(charset, errors="replace")
    except LookupError:
        text = raw.decode("utf-8", errors="replace")
    if content_type == "text/plain":
        return final_url, urlparse(final_url).netloc, text
    parser = _ReadableTextParser()
    parser.feed(text)
    parser.close()
    title = " ".join(parser.title.split()) or urlparse(final_url).netloc
    return final_url, title, parser.readable_text()


def ingest_url(library: Library, url: str, title: str | None = None) -> int:
    final_url, detected_title, text = fetch_page(url)
    return library.ingest_text(final_url, text, title.strip() if title and title.strip() else detected_title)
