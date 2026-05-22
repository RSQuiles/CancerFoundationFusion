    python canc_type_pred.py \
        --tcga-data /cluster/work/boeva/rquiles/data/embeddings/tcga_embedded.h5ad \
        --obsm-key X_pca \
        --cohorts BRCA BLCA COAD GBM HNSC KIRC KIRP LAML LGG LIHC LUAD LUSC MESO OV PAAD PRAD READ SARC SKCM STAD TGCT THCA UCEC UCS \
        --epochs 100 \
        --no-merge-gbm-lgg