"""Local SQLite storage and lightweight lexical retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Iterable


DEFAULT_DB = Path(".finch") / "finch.sqlite3"


@dataclass(frozen=True)
class SourceChunk:
    id: int
    document_id: int
    title: str
    path: str
    chunk_index: int
    content: str


@dataclass(frozen=True)
class StoredResponse:
    id: int
    question: str
    answer: str
    sources: list[dict[str, object]]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def database_path(value: str | Path | None) -> Path:
    return Path(value) if value else DEFAULT_DB


def read_text_file(path: Path) -> str:
    """Read the deliberately small first-pass set of source file formats."""
    if path.suffix.lower() not in {".txt", ".md", ".csv"}:
        raise ValueError("Only .txt, .md, and .csv files are supported in v0.1.")
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path.name} as text.")


def chunk_text(text: str, size: int = 900, overlap: int = 160) -> list[str]:
    """Create overlapping, paragraph-aware chunks while retaining local context."""
    if size <= overlap:
        raise ValueError("Chunk size must be larger than overlap.")
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + paragraph
        else:
            current = paragraph
        while len(current) > size:
            split_at = current.rfind(" ", 0, size)
            split_at = split_at if split_at > size // 2 else size
            chunks.append(current[:split_at].strip())
            current = current[max(0, split_at - overlap) :].strip()
    if current:
        chunks.append(current)
    return chunks


class Library:
    """A portable source library. All student data remains in one local file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = database_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self._has_fts = False
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Library":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                indexed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                UNIQUE(document_id, chunk_index)
            );
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY,
                response_id INTEGER NOT NULL REFERENCES responses(id) ON DELETE CASCADE,
                rating TEXT NOT NULL CHECK(rating IN ('up', 'down')),
                correction TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            """
        )
        try:
            self.connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
                "USING fts5(content, chunk_id UNINDEXED)"
            )
            self._has_fts = True
        except sqlite3.OperationalError:
            # Python builds without FTS5 still get a small, local LIKE fallback.
            self._has_fts = False
        self.connection.commit()

    def ingest(self, source_path: str | Path, title: str | None = None) -> int:
        path = Path(source_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Source file not found: {path}")
        content = read_text_file(path)
        source_title = title.strip() if title else path.stem.replace("_", " ")
        return self.ingest_text(str(path), content, source_title)

    def ingest_text(self, source_id: str, content: str, title: str) -> int:
        """Index supplied text under a stable local path, URL, or note identifier."""
        source_id = source_id.strip()
        source_title = title.strip()
        if not source_id:
            raise ValueError("Every source needs a stable identifier.")
        if not source_title:
            raise ValueError("Every source needs a title.")
        pieces = chunk_text(content)
        if not pieces:
            raise ValueError(f"{source_title} has no readable text to index.")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        previous = self.connection.execute(
            "SELECT id, content_hash FROM documents WHERE path = ?", (source_id,)
        ).fetchone()
        if previous and previous["content_hash"] == digest:
            return 0

        with self.connection:
            if previous:
                old_ids = [
                    row["id"]
                    for row in self.connection.execute(
                        "SELECT id FROM chunks WHERE document_id = ?", (previous["id"],)
                    )
                ]
                if self._has_fts:
                    self.connection.executemany(
                        "DELETE FROM chunks_fts WHERE rowid = ?", ((item,) for item in old_ids)
                    )
                self.connection.execute("DELETE FROM chunks WHERE document_id = ?", (previous["id"],))
                self.connection.execute(
                    "UPDATE documents SET title = ?, content_hash = ?, indexed_at = ? WHERE id = ?",
                    (source_title, digest, utc_now(), previous["id"]),
                )
                document_id = int(previous["id"])
            else:
                cursor = self.connection.execute(
                    "INSERT INTO documents(path, title, content_hash, indexed_at) VALUES (?, ?, ?, ?)",
                    (source_id, source_title, digest, utc_now()),
                )
                document_id = int(cursor.lastrowid)

            for index, piece in enumerate(pieces, start=1):
                cursor = self.connection.execute(
                    "INSERT INTO chunks(document_id, chunk_index, content) VALUES (?, ?, ?)",
                    (document_id, index, piece),
                )
                if self._has_fts:
                    self.connection.execute(
                        "INSERT INTO chunks_fts(rowid, content, chunk_id) VALUES (?, ?, ?)",
                        (cursor.lastrowid, piece, cursor.lastrowid),
                    )
        return len(pieces)

    @staticmethod
    def _fts_query(question: str) -> str:
        terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9._%-]*", question.lower())[:12]
        return " OR ".join(f'"{term}"' for term in terms)

    def search(self, question: str, limit: int = 4) -> list[SourceChunk]:
        if limit < 1:
            raise ValueError("Search limit must be at least one.")
        if not question.strip():
            return []
        if self._has_fts and (match := self._fts_query(question)):
            rows = self.connection.execute(
                """
                SELECT chunks.id, chunks.document_id, documents.title, documents.path,
                       chunks.chunk_index, chunks.content
                FROM chunks_fts
                JOIN chunks ON chunks_fts.rowid = chunks.id
                JOIN documents ON documents.id = chunks.document_id
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts)
                LIMIT ?
                """,
                (match, limit),
            ).fetchall()
        else:
            terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9._%-]*", question.lower())[:8]
            if not terms:
                return []
            where = " OR ".join("lower(chunks.content) LIKE ?" for _ in terms)
            rows = self.connection.execute(
                f"""
                SELECT chunks.id, chunks.document_id, documents.title, documents.path,
                       chunks.chunk_index, chunks.content
                FROM chunks JOIN documents ON documents.id = chunks.document_id
                WHERE {where}
                LIMIT ?
                """,
                tuple(f"%{term}%" for term in terms) + (limit,),
            ).fetchall()
        return [SourceChunk(**dict(row)) for row in rows]

    def save_response(
        self, question: str, answer: str, sources: Iterable[SourceChunk]
    ) -> StoredResponse:
        source_list = [
            {
                "id": item.id,
                "title": item.title,
                "path": item.path,
                "chunk": item.chunk_index,
            }
            for item in sources
        ]
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO responses(question, answer, sources_json, created_at) VALUES (?, ?, ?, ?)",
                (question, answer, json.dumps(source_list), utc_now()),
            )
        return StoredResponse(int(cursor.lastrowid), question, answer, source_list)

    def add_feedback(self, response_id: int, rating: str, correction: str = "") -> None:
        if rating not in {"up", "down"}:
            raise ValueError("Rating must be 'up' or 'down'.")
        exists = self.connection.execute("SELECT 1 FROM responses WHERE id = ?", (response_id,)).fetchone()
        if not exists:
            raise ValueError(f"No saved response has ID {response_id}.")
        with self.connection:
            self.connection.execute(
                "INSERT INTO feedback(response_id, rating, correction, created_at) VALUES (?, ?, ?, ?)",
                (response_id, rating, correction.strip(), utc_now()),
            )

    def training_examples(self, rating: str | None = None) -> list[dict[str, object]]:
        clauses = ["trim(feedback.correction) <> ''"]
        values: list[object] = []
        if rating:
            clauses.append("feedback.rating = ?")
            values.append(rating)
        rows = self.connection.execute(
            """
            SELECT responses.question, responses.answer, feedback.correction
            FROM feedback JOIN responses ON responses.id = feedback.response_id
            WHERE """ + " AND ".join(clauses) + " ORDER BY feedback.id",
            values,
        ).fetchall()
        return [
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a careful finance study assistant. State assumptions and do not give personalized investment advice.",
                    },
                    {"role": "user", "content": row["question"]},
                    {
                        "role": "assistant",
                        "content": "Original answer:\n"
                        + row["answer"]
                        + "\n\nStudent-reviewed correction:\n"
                        + row["correction"],
                    },
                ]
            }
            for row in rows
        ]

    def counts(self) -> dict[str, int]:
        tables = ("documents", "chunks", "responses", "feedback")
        return {
            table: int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }
