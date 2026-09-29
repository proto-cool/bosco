"""The encoders, his translators (docs/DECISIONS-2026-09-25.md, decision 14: auditable first, clean second).

Pinned by commit (weights and remote code): `trust_remote_code` runs whatever is at a revision, so a floating HEAD would change
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

# The nomic-bert-2048 remote code, pinned by commit (audit F9: `trust_remote_code` otherwise fetches its HEAD). Checked
# 2026-09-28 (offline, CPU): reproduces the cached v1-harm-dev embeddings to 1.5e-7 (cosine 1.000000).
TEXT["code_revision"] = "7710840340a098cfb869c4f65e87cf2b1b70caca"
TEXT_THREADS = 4  # the encoder is bit-exact at a fixed thread count only (1 vs 4 equal; 10 differs by ~1e-7)


def load_text(device: str = "cpu"):
    """The pinned text encoder. Needs the ENV above: `uv run --with ... python`, ideally with HF_HUB_OFFLINE=1."""
    from sentence_transformers import SentenceTransformer

    rev = TEXT["code_revision"]
    return SentenceTransformer(
        TEXT["id"],
        revision=TEXT["revision"],
        trust_remote_code=True,
        device=device,
        model_kwargs={"code_revision": rev},
        config_kwargs={"code_revision": rev},
    )


def embed_text(enc, texts):
    """The served smell input: each text alone (a batch pads, which moves the last bits), at TEXT_THREADS threads,
    with nomic's classification prefix and the first 200 words, normalised. Published numbers use this path."""
    import numpy as np
    import torch

    saved = torch.get_num_threads()
    torch.set_num_threads(TEXT_THREADS)
    try:
        return np.stack(
            [enc.encode([TEXT["prefix"] + " ".join(str(t).split()[:200])], normalize_embeddings=True)[0] for t in texts]
        ).astype(np.float32)
    finally:
        torch.set_num_threads(saved)
