"""Modality labels and display names used by the evaluation figures.

The labels are the ones ``obs["modality"]`` holds and the layout uses
(``experiments.layout.MODALITY_DIRS``). Pairing is not a modality: paired rows keep
their own modality and carry a pair id in a separate obs column (``paired``,
0 = unpaired), unlike the old ``paired_sc`` / ``paired_pb`` / ``paired_bulk`` labels.
"""

from __future__ import annotations

from experiments.layout import MODALITY_DIRS, canonical_modality

MOD_SC = "sc"
MOD_BULK = "bulk"
MOD_PB = "pseudobulk"
MODALITIES: tuple[str, ...] = (MOD_SC, MOD_PB, MOD_BULK)  # drawing order: dense SC underneath
assert set(MODALITIES) == set(MODALITY_DIRS)
# Not a modality: the mean embedding of the single cells sharing a link key (a
# pseudobulk or a pair), drawn by plot_alignment --cell-means on top of everything.
MOD_CELL_MEAN = "cell_mean"
DRAW_ORDER: tuple[str, ...] = MODALITIES + (MOD_CELL_MEAN,)

DISPLAY_NAMES: dict[str, str] = {
    MOD_SC: "Single-cell",
    MOD_PB: "Pseudobulk",
    MOD_BULK: "Bulk",
    MOD_CELL_MEAN: "Mean of cells",
}

PAIRED_COL = "paired"

__all__ = ["MOD_SC", "MOD_BULK", "MOD_PB", "MOD_CELL_MEAN", "MODALITIES", "DRAW_ORDER", "DISPLAY_NAMES", "PAIRED_COL", "canonical_modality"]
