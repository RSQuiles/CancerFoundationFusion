"""Generate cell embeddings from a CancerFoundation checkpoint and attach them to an h5ad file.

Usage
-----
    # Data is raw counts (CP10K + log1p normalization applied automatically):
    python embed.py --checkpoint path/to/model.ckpt --input data.h5ad

    # Data is already CP10K + log1p normalized:
    python embed.py --checkpoint path/to/model.ckpt --input data.h5ad --normalized

    # Data is CP10K-normalized but not log1p-transformed:
    python embed.py --checkpoint path/to/model.ckpt --input data.h5ad --log1p-only

    # Custom output path and obsm key:
    python embed.py --checkpoint path/to/model.ckpt --input data.h5ad \\
        --output data_embedded.h5ad --obsm-key X_cf

    # PCA embeddings only (no model required):
    python embed.py --input data.h5ad --pca --n-pcs 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scanpy as sc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed cells in an h5ad file using a CancerFoundation checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default=None, help="Path to .ckpt model checkpoint.")
    parser.add_argument("--input", required=True, help="Path to input .h5ad file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output .h5ad path. Defaults to <input_stem>_embedded.h5ad next to the input.",
    )
    parser.add_argument(
        "--obsm-key",
        default="X_cf",
        help="adata.obsm key under which embeddings are stored.",
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        default=False,
        help=(
            "Pass if the data is already CP10K + log1p normalized. "
            "No normalization is applied before embedding."
        ),
    )
    parser.add_argument(
        "--log1p-only",
        action="store_true",
        default=False,
        help=(
            "Pass if the data is already CP10K-normalized but not log1p-transformed. "
            "Only log1p is applied before embedding. "
            "Mutually exclusive with --normalized."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Force CPU inference even when a GPU is available.",
    )
    # PCA options
    parser.add_argument(
        "--pca",
        action="store_true",
        default=False,
        help=(
            "Compute PCA embeddings and store in adata.obsm['X_pca']. "
            "Mutually exclusive with model embedding. No checkpoint required."
        ),
    )
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=50,
        help="Number of principal components to compute (default: 50). Only used with --pca.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.normalized and args.log1p_only:
        print("Error: --normalized and --log1p-only are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    if not args.pca and args.checkpoint is None:
        print("Error: --checkpoint is required unless --pca is set.", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(input_path.stem + "_embedded.h5ad")
    )

    import scanpy as sc

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print(f"Loading data: {input_path}")
    adata = sc.read_h5ad(str(input_path))
    print(f"  {adata.n_obs} cells × {adata.n_vars} genes")

    # -------------------------------------------------------------------------
    # Model embedding branch
    # -------------------------------------------------------------------------
    if args.checkpoint is not None:
        import torch
        sys.path.insert(0, "../")
        from cancerfoundation.model.model import CancerFoundation

        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: checkpoint not found: {checkpoint_path}", file=sys.stderr)
            sys.exit(1)

        print(f"Loading checkpoint: {checkpoint_path}")
        model = CancerFoundation.load_from_checkpoint(str(checkpoint_path), strict=False)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        use_gpu = not args.cpu and torch.cuda.is_available()
        if use_gpu:
            model = model.cuda()
            print("Inference device: GPU")
        else:
            print("Inference device: CPU")

        print("Generating embeddings...")
        result = model.embed(
            adata,
            batch_size=args.batch_size,
            normalized=args.normalized,
            log1p_only=args.log1p_only,
        )
        emb_df = result[0] if isinstance(result, tuple) else result

        adata.obsm[args.obsm_key] = emb_df.to_numpy().astype(np.float32)
        print(
            f"Stored embeddings in adata.obsm['{args.obsm_key}'] "
            f"(shape: {adata.obsm[args.obsm_key].shape})"
        )

    # -------------------------------------------------------------------------
    # PCA branch
    # -------------------------------------------------------------------------
    if args.pca:
        print(f"Computing PCA with {args.n_pcs} components...")
        
        # Normalize to CP10K + log1p if not already normalized
        print("Normalizing input data!")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        sc.pp.pca(adata, n_comps=args.n_pcs)
        print(
            f"Stored PCA embeddings in adata.obsm['X_pca'] "
            f"(shape: {adata.obsm['X_pca'].shape})"
        )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    print(f"Saving to: {output_path}")
    adata.write_h5ad(str(output_path))
    print("Done.")


if __name__ == "__main__":
    main()