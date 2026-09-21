# Architecture and roadmap

## Initial architecture

```text
Student files (.txt/.md/.csv)
             |
             v
      chunking + metadata
             |
             v
SQLite + FTS5 <----> feedback / reviewed corrections
             |
       relevant chunks
             |
             v
compact prompt --> local llama.cpp server --> cited answer
```

Finch chooses lexical retrieval first because it has a very small operating
cost and no model download. This is often surprisingly effective for a student's
own materials, where exact terms such as `WACC`, `CAPM`, `EBITDA`, and
`discount rate` are meaningful.

## Next milestones

1. **Validate retrieval quality.** Add 30–50 representative questions from one
   course, and check whether a supporting source appears in the first four
   chunks. Improve document titles, chunk boundaries, and source quality before
   changing models.
2. **Add local embeddings only if needed.** Use a small local embedding model,
   persist vectors, and blend semantic similarity with the FTS ranking. Keep the
   lexical ranker as a fallback and do not use a hosted embedding API.
3. **Add calculators, not model guesses.** Implement transparent present value,
   NPV, bond price, and portfolio-return tools whose inputs/outputs are shown in
   the answer. The model should request a calculation; ordinary deterministic
   code should perform it.
4. **Evaluate before adaptation.** Retain a small held-out question set. Train
   an optional LoRA only from reviewed corrections, then compare it with the
   unadapted model on that held-out set. Keep the adapter only if it helps.
5. **Measure locally.** Record response time, estimated tokens, source-support
   rate, and user correction rate. Do not claim a carbon amount unless measuring
   the specific device's energy use and electricity mix.

## Model tiers

| Device capability | Suggested class | Intended use |
| --- | --- | --- |
| CPU laptop / limited memory | quantized 0.8–1.5B instruction model | Definitions, flashcards, extraction, source-grounded summaries |
| 8–16 GB spare RAM or modest GPU | quantized 3–4B instruction model | Better explanations and multi-step classroom reasoning |
| Larger local GPU | quantized 7–8B instruction model | More reliable synthesis; only use if evaluation justifies the additional energy |

Start with the smallest tier that passes a course-specific evaluation. RAG
quality and concise prompts will usually improve an assignment assistant more
than scaling the generator.

## Privacy and data boundaries

- Keep `.finch/` local; it contains source text, questions, answers, and feedback.
- Do not ingest graded assessments or confidential employer data unless policy
  explicitly permits it.
- Do not automatically send feedback to a trainer or a cloud endpoint.
- Make data deletion a deliberate future feature: delete source, chunks,
  conversations, and derived training examples together.
