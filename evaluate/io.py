"""Reading embeddings from ``embeddings/<model>/`` (or any h5ad) for the figures."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.layout import ExperimentLayout, canonical_modality

EMB_KEY = "X_emb"  # written by embed/embed_splits.py


def load_embedding(layout: ExperimentLayout, model: str, modality: str, part: str):
    """``embeddings/<model>/<modality>_<part>.h5ad`` as an AnnData with ``obsm['X_emb']``."""
    import anndata as ad

    path = layout.embedding_file(model, modality, part)
    if not path.exists():
        raise FileNotFoundError(
            f"No {canonical_modality(modality)} {part} embeddings for '{model}': {path}. "
            "Run embed/embed_splits.py first."
        )
    adata = ad.read_h5ad(path)
    if EMB_KEY not in adata.obsm:
        raise KeyError(f"{path} has no obsm['{EMB_KEY}']")
    return adata


def load_external(path: Path, obsm_key: str):
    """Any h5ad: obs plus one obsm matrix, with X left on disk."""
    import anndata as ad

    backed = ad.read_h5ad(path, backed="r")
    try:
        if obsm_key not in backed.obsm:
            raise KeyError(f"{path} has no obsm['{obsm_key}']; available: {list(backed.obsm.keys())}")
        out = ad.AnnData(obs=backed.obs.copy(), obsm={EMB_KEY: backed.obsm[obsm_key][:]})
    finally:
        backed.file.close()
    return out


def load_embed_card(layout: ExperimentLayout, model: str) -> dict:
    path = layout.embed_card(model)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def panel_hashes(layout: ExperimentLayout, model: str, files: list[Path]) -> dict[str, str | None]:
    """``{file name: panel hash}`` from ``embed.json``; None when it was not recorded."""
    card_files = load_embed_card(layout, model).get("files", {})
    return {f.name: (card_files.get(f.name) or {}).get("panel_hash") for f in files}
