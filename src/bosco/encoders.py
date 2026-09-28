"""The encoders, his translators (docs/DECISIONS-2026-09-25.md, decision 14: auditable first, clean second).

Pinned by commit: `trust_remote_code` runs whatever is at a revision, so a floating HEAD would change
both the code and the smell. Both load only with transformers 4.x, so embedding runs in its own env:

    uv run --with "transformers==4.46.3" --with "sentence-transformers==3.3.1" --with einops python ...

Provenance (docs/audit-2026-09-25/provenance.md, section 5): Apache 2.0 for code, weights and training
data list. The training data is published but includes scraped sources (Reddit, Amazon reviews, and MS
MARCO in the text model's fine-tuning; DataComp CommonPool images in the vision model). "Auditable,
not clean": to be replaced by our own encoders if they pass the encoder gate (PLAN-V1, track C).
"""

TEXT = {
    "id": "nomic-ai/nomic-embed-text-v1.5",
    "revision": "e9b6763023c676ca8431644204f50c2b100d9aab",
    "licence": "apache-2.0",
    "prefix": "classification: ",  # nomic's task prefix for classification-style embeddings
    "dim": 768,
}
VISION = {
    "id": "nomic-ai/nomic-embed-vision-v1.5",
    "revision": "e3a725bce72db07ca4adb1d83da08903f3ee02f8",
    "licence": "apache-2.0",
    "dim": 768,  # shares nomic-embed-text-v1.5's space
}
ENV = ["transformers==4.46.3", "sentence-transformers==3.3.1", "einops"]
