# Finch RAG

Finch is a local-first, low-footprint retrieval assistant for finance students.
It is deliberately **not** a model trained from scratch. A small local language
model writes answers; the useful, course-specific knowledge stays in a local
SQLite library of the student's lecture notes, readings, and corrections.

That distinction is important: retrieval makes the assistant more useful for a
specific course without repeatedly training a neural network. Training after
every conversation is expensive, difficult to audit, and can make a model less
reliable. Finch stores feedback locally and can export only student-reviewed
examples for an occasional, opt-in LoRA fine-tuning run.

## What this prototype does

- Ingests `.txt`, `.md`, and `.csv` course material into a local SQLite file.
- Uses a no-download lexical retrieval baseline (SQLite FTS5) so it works on a
  modest laptop. A future embedding plug-in can improve semantic retrieval.
- Calls a local [llama.cpp](https://github.com/ggml-org/llama.cpp) server through
  its OpenAI-compatible chat API. No API key or cloud endpoint is required.
- Cites the local source chunks used for each answer.
- Saves feedback and exports *only reviewed corrections* as JSONL training data.
- Offers a `--sources-only` mode that does no language-model generation at all.

It is an educational support tool, not investment, tax, accounting, or legal
advice. It should show its sources and calculations, and a student should check
anything consequential against the assigned material or an instructor.

## Why this is a lower-impact design

| Choice | Effect |
| --- | --- |
| Local, quantized model | Avoids a remote inference request and lets the user choose a model that fits their hardware. |
| Retrieval before training | New notes become available immediately; most updates do not require model training. |
| FTS5 baseline | No separate embedding model, vector database, or always-running service. |
| Small context budget | Sends only the most relevant chunks instead of entire documents. |
| Opt-in adaptation export | Fine-tuning happens only when the student has enough reviewed examples to justify it. |

Lower footprint does not mean zero impact: local inference still uses electricity,
and model download/training has a cost. The most meaningful settings are keeping
the model small, the context short, and declining to generate when retrieved
sources alone answer the question.

## Quick start (Windows / PowerShell)

1. Install Python 3.11 or later and install a local model runtime. For the
   leanest setup, `llama.cpp` can run quantized GGUF models on CPU or GPU and
   exposes a local OpenAI-compatible API.
2. Launch the server with a small instruction model. For example, following the
   current llama.cpp command style:

   ```powershell
   llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF --port 8080
   ```

   A roughly 1B-parameter model is a good starting point for short definitions,
   study guides, and source-grounded explanations. If the laptop has 8–16 GB of
   spare memory, try a 3–4B model for more dependable reasoning. Quantization
   size and context length matter at least as much as the parameter count.
3. From this folder, make and populate a local library:

   ```powershell
   $env:PYTHONPATH = "src"
   python -m finch init
   python -m finch ingest "C:\Users\you\Documents\Finance 301\notes.md"
   python -m finch ask "Explain net present value and give a small example."
   ```

   The library defaults to `.finch/finch.sqlite3` in the current directory.
   This folder contains the student's material and feedback; do not commit it.

## Commands

```text
python -m finch init [--db PATH]
python -m finch ingest PATH [--title TITLE] [--db PATH]
python -m finch ask QUESTION [--sources-only] [--top-k 4] [--db PATH]
python -m finch feedback RESPONSE_ID --rating up|down --correction TEXT [--db PATH]
python -m finch export-training OUTPUT.jsonl [--min-rating up] [--db PATH]
python -m finch status [--db PATH]
```

`ask` prints a response ID. Use it when saving feedback, for example:

```powershell
python -m finch feedback 42 --rating down --correction "Use annual cash flow, not monthly cash flow."
```

The exported JSONL is a candidate dataset, not training data to use blindly.
Open it, remove private material and bad examples, then fine-tune a LoRA adapter
only when there is a coherent, sufficiently large set of reviewed corrections.

## Responsible finance behavior

Finch's prompt requires it to distinguish course explanation from real-world
recommendation, name assumptions, show formulas, cite retrieved material, and
say when the source library does not support an answer. Add your professor's
approved sources rather than scraping finance sites of uncertain provenance.

## Development

This initial prototype depends only on Python's standard library. Run its tests
with:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

The architecture and build roadmap are in [`docs/architecture.md`](docs/architecture.md).
