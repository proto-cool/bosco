# A2 phase 1 results

Rule: `docs/A2-PHASE1.md`. Balanced accuracy, logistic model, held-out.

| nose | sweet raw / antenna | dangerous raw / antenna | junk raw / antenna | pictures raw / antenna | **mean antenna (text)** |
|---|---|---|---|---|---|
| e5-large | 0.936 / 0.917 | 0.840 / 0.807 | 0.958 / 0.963 | — | **0.896** |
| mxbai-large | 0.905 / 0.891 | 0.842 / 0.832 | 0.949 / 0.957 | — | **0.893** |
| bge-large | 0.915 / 0.887 | 0.847 / 0.823 | 0.954 / 0.950 | — | **0.887** |
| jina-clip-v2 | 0.891 / 0.884 | 0.819 / 0.778 | 0.964 / 0.954 | 0.943 / 0.930 | **0.872** |
| mpnet | 0.885 / 0.889 | 0.782 / 0.726 | 0.952 / 0.938 | — | **0.851** |
| clip-b32 | 0.763 / 0.741 | 0.794 / 0.753 | 0.926 / 0.944 | 0.960 / 0.923 | **0.813** |

## Choice by the rule

- **e5-large** for text (0.896); best picture nose jina-clip-v2 (0.872 on text) drives the visual Kenyon cells.
- antenna cost > 0.05: mpnet on dangerous (0.782 → 0.726)

## Notes

- **nomic-v1.5 was not measured.** Its own model code fails to load with the
  installed `transformers`, and again with 4.46.3 / sentence-transformers 3.3.1 /
  huggingface_hub 0.26.5 (`NomicBertConfig` cannot build a tokenizer). It
  would change the choice only by reaching 0.876 on text as one nose.
- jina-clip-v2 was measured in an isolated environment (transformers 4.46.3,
  sentence-transformers 3.3.1). Its embeddings are cached like the rest.
- **Result, by the rule:** e5-large-v2 is his nose for text, through the
  52 glomeruli. jina-clip-v2 is his eyes, through the visual Kenyon cells.
  Ceilings through the antenna: sweet/bitter 0.917, safe/dangerous 0.807,
  junk 0.963, pictures 0.930 (jina). Safe/dangerous is capped near 0.83 by
  every nose; its labels are the noisiest.
