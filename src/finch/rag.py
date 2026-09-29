"""Grounded answer construction with a compact local context."""

from __future__ import annotations

from dataclasses import dataclass

from .local_model import LocalModel
from .storage import Library, SourceChunk, StoredResponse


SYSTEM_PROMPT = """You are Finch, a careful study assistant for finance students.
Use only the supplied course sources for factual claims. Cite source markers in
square brackets, such as [S1], next to the relevant claim. If the sources do not
support an answer, say that plainly and identify what is missing. Explain the
reasoning and formulas, with assumptions, but do not give personalized investment,
tax, legal, or accounting advice. Never invent a source, statistic, or citation.

Present the response as a clear study note:
- Start with a direct answer, then use short Markdown headings and lists only
  when they make the explanation easier to scan. Do not add a redundant title.
- Keep paragraphs short and explain finance terms in ordinary language.
- Put each formula on its own line using standard LaTeX between dollar signs,
  for example `$NPV = \\frac{CF_1}{(1+r)^1} - C_0$`. Do not put a formula in the
  middle of a sentence, and define its variables in normal words immediately
  below it. Finch will render the formula for the student.
- Do not put raw code, backslashes, dollar signs, or Markdown symbols in the
  explanatory prose."""


@dataclass(frozen=True)
class Answer:
    response: StoredResponse
    source_chunks: list[SourceChunk]


def format_sources(chunks: list[SourceChunk]) -> str:
    if not chunks:
        return "No matching local source chunks were found."
    return "\n".join(
        f"[S{number}] {chunk.title}, chunk {chunk.chunk_index}\n{chunk.content[:1400]}"
        for number, chunk in enumerate(chunks, start=1)
    )


def sources_only_answer(question: str, chunks: list[SourceChunk]) -> str:
    if not chunks:
        return "No relevant local sources were found. Try ingesting course material or using different terms."
    lines = ["Retrieved local sources (no model generation was used):"]
    for number, chunk in enumerate(chunks, start=1):
        preview = " ".join(chunk.content.split())[:420]
        lines.append(f"[S{number}] {chunk.title}, chunk {chunk.chunk_index}: {preview}")
    return "\n\n".join(lines)


def ask(
    library: Library,
    question: str,
    top_k: int = 4,
    sources_only: bool = False,
    model: LocalModel | None = None,
) -> Answer:
    chunks = library.search(question, limit=top_k)
    if sources_only:
        answer = sources_only_answer(question, chunks)
    elif not chunks:
        answer = (
            "I could not find supporting material in the local library, so I should not "
            "answer this as a source-grounded study response. Ingest the relevant notes or reading first."
        )
    else:
        user_prompt = f"Question: {question}\n\nCourse sources:\n{format_sources(chunks)}"
        answer = (model or LocalModel()).chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
        )
    return Answer(library.save_response(question, answer, chunks), chunks)
