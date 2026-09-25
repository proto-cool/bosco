# Passage check: can his nose read passages? (pre-registered 2026-09-24)

Jev scores 94% on passage yes/no (BoolQ, its public benchmark). Bosco's
one-fingerprint nose topped out at 0.656 through the antenna (A3 phase 1).
This measures the **ceiling of his senses** under two fixes that keep the
brain doing the work, and one reference. **No brain is trained.**

## Data

- BoolQ train: 4,000 passages sampled (seed 13), **excluding** the 20 bench
  passages that appear in it. BoolQ validation: 1,500 sampled, minus the bench
  passages.
- Jev's 50 bench yes/no items, reported separately.
- The question is "According to the passage, is the answer to this question
  yes? <question>", as the bench phrases it.

## Conditions (every one through the 46-channel antenna, fit label-free on training rows)

| condition | what his nose gets | a candidate? |
|---|---|---|
| **single** (baseline) | e5-large-v2: one fingerprint of question + passage | the current nose |
| **sniff** | e5-large-v2: the question with **each sentence** (split on . ! ?), one fingerprint per sentence; pooled as the per-channel mean and max over sentences (a brain sniffing sentence by sentence and keeping the strongest response) | yes |
| **aware** | multilingual-e5-large-instruct: one fingerprint of the passage **conditioned on the question** by its instruction ("Given the question …, represent the passage for answering yes or no") | yes |
| **llm-ref** | gte-Qwen2-1.5B-instruct (an LLM backbone): the same conditioned fingerprint | **no, reference only**: it shows how much reading an LLM-sized translator would take over |

Each is scored by a logistic model (balanced accuracy) trained on the
training passages: the most any brain could get from that nose.

## Rule (fixed now)

- The A4 nose becomes **sniff** or **aware**, whichever is higher on
  validation, **if** it beats **single** by ≥ 0.05. Otherwise the nose stays
  as is, and passage questions are declared out of reach for Bosco.
- llm-ref never becomes the nose, whatever it scores.
- A model that fails to load is reported as such; nothing else stands in for it.

Runner: `scripts/passage_check.py`. Results: `docs/passage-check-results.md`.
