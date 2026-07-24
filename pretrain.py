import json
import os
import sys
from typing import Optional
from pathlib import Path

sys.path.insert(0, "../")
from utils import get_args, MyProgressBar
from cancerfoundation.model.model import CancerFoundation
from cancerfoundation.data.data_module import BulkSCDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def train_model(
    model: CancerFoundation,
    datamodule: pl.LightningDataModule,
    max_epochs: int,
    save_dir: str,
    precision: str,
    num_nodes: int,
    gpus: int,
    wandb_project: Optional[str],
    wandb_entity: Optional[str],
    wandb_name: Optional[str],
    resume_from_checkpoint: Optional[str],
    strategy: str,
    gradient_clip_val: float,
    accumulate_grad_batches: int,
    val_check_interval: float,
    log_interval: int,
    save_every: bool,
    ckpt_every_n_steps: Optional[int] = None,
    verbose: bool = False,
):
    """
    Train the model using PyTorch Lightning Trainer

    Args:
        lightning_module: The LightningModule to train
        max_epochs: Maximum number of epochs
        save_dir: Directory to save checkpoints
        wandb_project: Wandb project name for logging
        resume_from_checkpoint: Path to checkpoint to resume from
        accelerator: Accelerator type ('cpu', 'gpu', 'tpu', 'auto')
        devices: Number of devices to use ('auto', int, or list)
        strategy: Training strategy ('auto', 'ddp', 'deepspeed', etc.)
        precision: Precision ('16-mixed', '32', 'bf16-mixed')
        accumulate_grad_batches: Number of batches to accumulate gradients
        gradient_clip_val: Gradient clipping value
        val_check_interval: Validation frequency
        pretrained_model_path: Path to pretrained model weights
    """
    # Setup callbacks
    callbacks = []
    callbacks.append(MyProgressBar(refresh_rate=log_interval))
    # Epoch-level checkpoint (always active)
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="epoch_{epoch:02d}",
        every_n_epochs=1,
        save_top_k=-1 if save_every else 1,
    )
    callbacks.append(checkpoint_callback)

    # Step-level checkpoint (optional)
    if ckpt_every_n_steps is not None:
        step_checkpoint_callback = ModelCheckpoint(
            dirpath=save_dir,
            filename="step_{step:06d}_epoch_{epoch:02d}",
            every_n_train_steps=ckpt_every_n_steps,
            save_top_k=-1 if save_every else 1,
        )
        callbacks.append(step_checkpoint_callback)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    logger = None

    # logger = None
    global_rank = int(os.environ.get("GLOBAL_RANK", "0"))
    if wandb_project and global_rank == 0:
        if verbose:
            print("\n\nSetting up WANDB logger...\n\n")
        logger = WandbLogger(
            entity=wandb_entity,
            project=wandb_project,
            name=wandb_name,
            save_dir=save_dir,
        )

    # Create trainer
    if gpus == 0:
        accelerator = "cpu"
        devices = "auto"
    else:
        accelerator = "gpu"
        devices = gpus

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        val_check_interval=val_check_interval,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=log_interval,
        enable_progress_bar=True,
        enable_model_summary=True,
        use_distributed_sampler=False,
    )

    # Start training
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_from_checkpoint)

    return trainer


def _get_last_checkpoint_path_from_trainer(trainer: pl.Trainer, save_dir: str) -> Optional[str]:
    """Return the most recently saved checkpoint path, if any."""
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            # If configured, Lightning tracks the latest file explicitly.
            if callback.last_model_path:
                return callback.last_model_path
            # Fallback for setups that do not use save_last.
            if callback.best_model_path:
                return callback.best_model_path

    checkpoint_dir = Path(save_dir)
    candidates = list(checkpoint_dir.glob("*.ckpt"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)

def _get_last_checkpoint_path(save_dir: str) -> Optional[str]:
    """Return the most recently saved checkpoint path in save_dir, if any."""
    checkpoint_dir = Path(save_dir)
    if not checkpoint_dir.exists():
        return None
    candidates = list(checkpoint_dir.glob("*.ckpt"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


def main(input_args=None):
    if input_args is None:
        args = get_args()
    else:
        args = input_args

    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    if args.contrastive_training and not args.unified:
        raise ValueError("Contrastive training is only supported with unified_fm=True.")
    if args.mmd and not args.unified:
        raise ValueError("MMD alignment loss is only supported with unified_fm=True.")
    if args.agg_consistency and not args.unified:
        raise ValueError(
            "Aggregation consistency losses are only supported with unified_fm=True."
        )
    if args.precomputed_pb and not args.unified:
        raise ValueError("--precomputed-pb is only supported with --unified.")
    if args.precomputed_pb and not args.pb_label:
        raise ValueError(
            "--precomputed-pb requires --pb-label (the modality label identifying the "
            "precomputed pseudobulk rows in the memory-mapped dataset)."
        )
    if args.precomputed_pb and args.agg_consistency:
        raise ValueError(
            "--precomputed-pb is incompatible with --agg-consistency: precomputed "
            "pseudobulk batches contain no constituent single cells to enforce "
            "aggregation consistency against."
        )
    if args.paired_sampling and not args.unified:
        raise ValueError(
            "--paired-sampling is only supported with --unified: the paired alignment "
            "loss matches rows by their modality, which is injected into the "
            "conditions only in unified mode."
        )
    if args.paired_sampling and not args.pb_label:
        raise ValueError(
            "--paired-sampling requires --pb-label: paired batches match a precomputed "
            "pseudobulk row to a bulk row by shared pair id, so the dataset must contain "
            "precomputed pseudobulk rows identified by --pb-label. Without them no paired "
            "batch can be formed."
        )
    if args.agg_consistency and not args.agg_fn:
        # Fallback to sum
        args.agg_fn = "sum"
    if args.agg_fn not in [None, "sum", "mean"]:
        raise ValueError(
            "Invalid aggregation function. Supported values are None, 'sum', and 'mean'."
        )
    if args.esm_emb and args.esm_emb_path is None:
        raise ValueError(
            "--esm-emb requires --esm-emb-path to point to the pretrained gene embedding parquet file."
        )
    if args.cdd and not args.unified:
        raise ValueError("The CDD loss (--cdd) is only supported with unified_fm=True.")
    if args.cdd and args.cdd_class_column not in args.conditions:
        raise ValueError(
            f"--cdd-class-column '{args.cdd_class_column}' must be one of --conditions {args.conditions}."
        )
    if args.condition_encoders is not None:
        # Effective condition set includes the auto-added "modality" under --unified.
        effective_conditions = (
            (args.conditions or []) + ["modality"] if args.unified else (args.conditions or [])
        )
        unknown_enc = set(args.condition_encoders) - set(effective_conditions)
        if unknown_enc:
            raise ValueError(
                f"--condition-encoders {sorted(unknown_enc)} must be a subset of the "
                f"conditions {effective_conditions}."
            )
    if args.cdd_infer_labels and not args.cdd:
        raise ValueError("--cdd-infer-labels requires --cdd.")
    if args.cdd_class_aware and not args.pb_group_column:
        raise ValueError("--cdd-class-aware requires --pb-group-column (the tissue column).")
    if not 0.0 <= args.cdd_bulk_class_frac <= 1.0:
        raise ValueError(
            f"--cdd-bulk-class-frac must be in [0, 1], got {args.cdd_bulk_class_frac}."
        )
    # --cdd-class-aware composes with --paired-sampling: the sampler still yields a
    # paired batch every paired_every_n steps and a class-aware batch otherwise.
    # The CDD loss is computed only on the (class-aware) non-paired batches; paired
    # batches use the paired alignment loss instead.

    # CDD's warmup exists so the marginal MMD can pull the bulk and pseudobulk clouds
    # together before the first clustering runs; without an aligner the warmup does
    # nothing and the source-seeded centroids have no chance of competing. Enable MMD
    # for that window if the user did not. It decays to --cdd-mmd-final-weight (0 by
    # default) once the ramp completes, so an auto-enabled MMD turns itself back off.
    cdd_source_fallback = not args.cdd_no_source_fallback
    if args.cdd and args.cdd_cluster_warmup_steps > 0 and not args.mmd:
        args.mmd = True
        print(
            f"[CDD] --mmd auto-enabled for the {args.cdd_cluster_warmup_steps}-step warmup "
            f"(weight {args.loss_weight_mmd}), decaying to {args.cdd_mmd_final_weight} "
            f"over the following {args.cdd_ramp_steps} steps as CDD ramps in."
        )

    datamodule = BulkSCDataModule(
        data_path=args.train_path,
        zero_percentages=args.zero_percentages,
        batch_size=args.batch_size,
        epoch_size=args.epoch_size,
        conditions=args.conditions + ["modality"] if args.unified else args.conditions,
        balance_primary=args.balance_primary,
        balance_secondary=args.balance_secondary,
        bulk_ratio=args.bulk_ratio,
        pb_ratio=args.pb_ratio,
        n_sc_per_pseudobulk=args.n_sc_per_pseudobulk,
        max_seq_len=args.max_seq_len,
        input_style=args.input_style,
        input_data=args.input_data,
        mask_ratio=args.mask_ratio,
        TRUNC_BY_SAMPLE=args.trunc_by_sample,
        training_tasks=args.training_tasks,
        n_bins=args.n_bins,
        normalise_bins=args.normalise_bins,
        condition_token=args.where_condition == "begin",
        num_workers=args.num_workers,
        unified_fm=args.unified,
        balance=args.balanced_sampler,
        balance_labels=args.balanced_labels,
        pb_group_column=args.pb_group_column,
        agg_consistency=args.agg_consistency,
        pb_label=args.pb_label,
        precomputed_pb=args.precomputed_pb,
        paired_sampling=args.paired_sampling,
        paired_every_n=args.paired_every_n,
        paired_column=args.paired_column,
        verbose=args.verbose,
        class_aware_cdd=args.cdd_class_aware,
        cdd_exclude_labels=args.cdd_exclude_labels,
        cdd_min_class_count=args.cdd_min_class_count,
        cdd_bulk_class_frac=args.cdd_bulk_class_frac,
    )
    datamodule.setup(stage="fit")

    # Define training start point
    if args.resume_from_checkpoint == "last":
        resume_from_checkpoint = _get_last_checkpoint_path(args.save_dir)
    else:
        resume_from_checkpoint = args.resume_from_checkpoint

    if resume_from_checkpoint is not None:
        print(f"Resuming training from {resume_from_checkpoint}")
        # Disabled considering a config file will be used to reinitiate training
        """
        model = CancerFoundation.load_from_checkpoint(
            resume_from_checkpoint, vocab=datamodule.vocab
        )
    else:
        """
    model = CancerFoundation(
        n_bins=args.n_bins,
        vocab=datamodule.vocab,
        input_emb_style=args.input_emb_style,
        max_seq_len=args.max_seq_len,
        input_style=args.input_style,
        mask_ratio=args.mask_ratio,
        TRUNC_BY_SAMPLE=args.trunc_by_sample,
        training_tasks=args.training_tasks,
        embsize=args.embsize,
        nheads=args.nheads,
        d_hid=args.d_hid,
        nlayers=args.nlayers,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        warmup_ratio_or_step=args.warmup_ratio_or_step,
        scheduler_interval=args.scheduler_interval,
        scheduler_factor=args.scheduler_factor,
        loss_type=args.loss,
        do_dat=args.do_dat,
        dat_columns=list(args.dat_columns),
        no_invert_dat=args.no_invert_dat,
        conditions=args.conditions + ["modality"]
        if args.unified
        else args.conditions,
        conditions_nums=datamodule.conditions_nums if args.conditions else None,
        encoded_conditions=args.condition_encoders,
        mvc_decoder_style=args.mvc_decoder_style,
        scale_zero_expression=args.scale_zero_expression,
        data_path=args.train_path,
        do_mvc=args.do_mvc,
        zero_percentages=args.zero_percentages,
        balance_primary=args.balance_primary,
        balance_secondary=args.balance_secondary,
        compile_model=args.compile,
        activation=args.activation,
        norm_scheme=args.norm_scheme,
        norm_type=args.norm_type,
        cell_emb_style=args.cell_emb_style,
        batchnorm=args.batchnorm,
        explicit_zero_prob=args.explicit_zero_prob,
        dat_scale=args.dat_scale,
        dat_start_step=args.dat_start_step,
        dat_interval_steps=args.dat_interval_steps,
        normalise_bins=args.normalise_bins,
        where_condition=args.where_condition,
        gen_method=args.gen_method,
        their_init_weights=args.their_init_weights,
        # Unified FM parameters
        contrastive=args.contrastive_training,
        mmd=args.mmd,
        aggregation=args.agg_consistency,
        agg_fn=args.agg_fn,
        paired_alignment=args.paired_sampling,
        noise=args.noise,
        esm_emb=args.esm_emb,
        esm_emb_path=args.esm_emb_path,
        esm_emb_finetune=args.esm_finetune,
        verbose=args.verbose,
        weight_mvc=args.loss_weight_mvc,
        weight_contrastive=args.loss_weight_contrastive,
        weight_mmd=args.loss_weight_mmd,
        weight_paired=args.loss_weight_paired,
        weight_agg=args.loss_weight_agg,
        weight_dat=args.loss_weight_dat,
        weight_reconstruction=args.loss_weight_reconstruction,
        n_sc_per_pseudobulk=args.n_sc_per_pseudobulk,
        # Contrastive Domain Discrepancy (CAN)
        cdd=args.cdd,
        weight_cdd=args.loss_weight_cdd,
        cdd_class_column=args.cdd_class_column,
        cdd_min_class_count=args.cdd_min_class_count,
        cdd_exclude_labels=args.cdd_exclude_labels,
        cdd_class_aware=args.cdd_class_aware,
        cdd_infer_labels=args.cdd_infer_labels,
        cdd_cluster_warmup_steps=args.cdd_cluster_warmup_steps,
        cdd_cluster_interval=args.cdd_cluster_interval,
        cdd_cluster_iters=args.cdd_cluster_iters,
        cdd_cluster_ambiguity=args.cdd_cluster_ambiguity,
        cdd_cluster_min_size=args.cdd_cluster_min_size,
        cdd_relabel_known=args.cdd_relabel_known,
        cdd_cluster_source_fallback=cdd_source_fallback,
        cdd_cluster_source_pb=args.cdd_cluster_source_pb,
        cdd_ramp_steps=args.cdd_ramp_steps,
        cdd_mmd_final_weight=args.cdd_mmd_final_weight,
    )

    if args.pretrained:
        print(f"Loading pretrained weights from {args.pretrained}.")
        vocab_pretrained = json.load(open(args.pretrained / "vocab.json", "r"))
        gene_mapping = {}
        for key, value in datamodule.vocab.items():
            if key in vocab_pretrained:
                gene_mapping[value] = vocab_pretrained[key]
        model.load_pretrained_weights(
            args.pretrained / "best_model.pt", gene_mapping=gene_mapping
        )

    trainer = train_model(
        model=model,
        datamodule=datamodule,
        max_epochs=args.epochs,
        num_nodes=args.num_nodes,
        gpus=args.gpus,
        save_dir=args.save_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        val_check_interval=args.val_check_interval,
        wandb_project=args.wandb,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        accumulate_grad_batches=args.grad_accu_steps,
        strategy=args.strategy,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        log_interval=args.log_interval,
        save_every=args.save_every,
        ckpt_every_n_steps=args.ckpt_every_n_steps,
        verbose=args.verbose,
    )

    # Return latest model checkpoint
    return _get_last_checkpoint_path_from_trainer(trainer, args.save_dir)


if __name__ == "__main__":
    main()
