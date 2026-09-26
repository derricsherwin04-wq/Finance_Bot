"""Command-line interface for Finch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .local_model import LocalModelError
from .rag import ask
from .storage import Library, database_path
from .web_sources import ingest_url


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finch", description="Local-first, source-grounded study help for finance students."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create the local source library.")
    init.add_argument("--db", help="SQLite database path (default: .finch/finch.sqlite3)")

    ingest = sub.add_parser("ingest", help="Add or refresh a .txt, .md, or .csv source.")
    ingest.add_argument("path")
    ingest.add_argument("--title", help="Readable title to show in citations")
    ingest.add_argument("--db", help="SQLite database path")

    url = sub.add_parser("ingest-url", help="Add one user-provided HTML or text webpage.")
    url.add_argument("url")
    url.add_argument("--title", help="Readable title to show in citations")
    url.add_argument("--db", help="SQLite database path")

    question = sub.add_parser("ask", help="Retrieve sources and ask the local model.")
    question.add_argument("question")
    question.add_argument("--top-k", type=int, default=4, help="Number of source chunks (default: 4)")
    question.add_argument("--sources-only", action="store_true", help="Retrieve sources without model generation")
    question.add_argument("--db", help="SQLite database path")

    feedback = sub.add_parser("feedback", help="Save a rating and optional reviewed correction.")
    feedback.add_argument("response_id", type=int)
    feedback.add_argument("--rating", choices=("up", "down"), required=True)
    feedback.add_argument("--correction", default="", help="A specific, student-reviewed correction")
    feedback.add_argument("--db", help="SQLite database path")

    export = sub.add_parser("export-training", help="Export reviewed corrections as candidate JSONL.")
    export.add_argument("output", type=Path)
    export.add_argument("--rating", choices=("up", "down"), help="Only export one feedback rating")
    export.add_argument("--db", help="SQLite database path")

    status = sub.add_parser("status", help="Show local library counts.")
    status.add_argument("--db", help="SQLite database path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = database_path(getattr(args, "db", None))
    try:
        with Library(db_path) as library:
            if args.command == "init":
                print(f"Local Finch library is ready: {library.path}")
                return 0
            if args.command == "ingest":
                count = library.ingest(args.path, args.title)
                if count:
                    print(f"Indexed {count} chunks from {Path(args.path).name}.")
                else:
                    print("That file is unchanged; the existing index was kept.")
                return 0
            if args.command == "ingest-url":
                count = ingest_url(library, args.url, args.title)
                if count:
                    print(f"Indexed {count} chunks from the webpage.")
                else:
                    print("That webpage is unchanged; the existing index was kept.")
                return 0
            if args.command == "ask":
                result = ask(library, args.question, args.top_k, args.sources_only)
                print(result.response.answer)
                print(f"\nResponse ID: {result.response.id}")
                print("Sources:")
                if result.response.sources:
                    for number, source in enumerate(result.response.sources, start=1):
                        print(f"  [S{number}] {source['title']} (chunk {source['chunk']})")
                else:
                    print("  None")
                return 0
            if args.command == "feedback":
                library.add_feedback(args.response_id, args.rating, args.correction)
                print("Feedback saved locally. It will not train a model automatically.")
                return 0
            if args.command == "export-training":
                records = library.training_examples(args.rating)
                if not records:
                    print("No reviewed corrections match that filter; nothing was exported.")
                    return 0
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("w", encoding="utf-8", newline="\n") as handle:
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"Exported {len(records)} candidate reviewed examples to {args.output}.")
                return 0
            if args.command == "status":
                counts = library.counts()
                print(f"Library: {library.path}")
                for name, count in counts.items():
                    print(f"{name.capitalize()}: {count}")
                return 0
    except (FileNotFoundError, ValueError, LocalModelError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 1
