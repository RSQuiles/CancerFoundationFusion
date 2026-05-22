python embed.py \
    --checkpoint /cluster/work/boeva/rquiles/outputs/save_CFF/ablation_base_comparison/aggregation/step_step=700000_epoch_epoch=01.ckpt \
    --obsm-key X_agg \
    --input  /cluster/work/boeva/rquiles/data/embeddings/tcga_embedded.h5ad \
    --output /cluster/work/boeva/rquiles/data/embeddings/tcga_embedded.h5ad \