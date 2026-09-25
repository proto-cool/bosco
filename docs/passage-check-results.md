# Passage check results

Rules: `docs/PASSAGE-CHECK.md`. Balanced accuracy of a logistic model on the antenna channels.

| condition | validation (≈1,500) | Jev bench (50) |
|---|---|---|
| single | 0.636 | 0.58 |
| sniff | 0.616 | 0.52 |
| aware | 0.687 | 0.76 |
| llm-ref | 0.682 | 0.64 |

Jev on the same 50 bench items: 0.94.

## By the rule

- **The A4 nose becomes aware**: 0.687 vs single 0.636 (+0.051).

## Reading

- By the rule, the question-aware nose (multilingual-e5-large-instruct) becomes
  the nose for the **full** A4: +0.051 over the current nose, a hair over the
  bar. The A4 pilot keeps the nose it was pre-registered with (e5-large-v2).
- **The passage gap mostly remains:** 0.687 against Jev's 0.94. Even an
  LLM-backbone embedder gets only 0.682 this way. A fingerprint, however
  question-aware, does not read. Passage questions stay a known limit.
- Sentence-by-sentence sniffing (mean and max over sentences) did not help:
  0.616.
- The report step first crashed on float16 embeddings; it was fixed with a
  cast and rerun. llm-ref failed in the main environment ("rope_theta") and
  ran in the isolated one, as the runner provides.
