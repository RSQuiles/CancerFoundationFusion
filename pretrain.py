import json
import os
import sys
from typing import Optional

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
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="epoch_{epoch:02d}",
        every_n_epochs=1,
        save_top_k=-1 if save_every else 1,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    logger = None

    # logger = None
    global_rank = int(os.environ.get("GLOBAL_RANK", "0"))
    if wandb_project and global_rank == 0:
        print("\n\nSetting up WANDB logger...\n\n")
        logger = WandbLogger(
            entity=wandb_entity,
            project=wandb_project,
            name=wandb_name,
            save_dir=save_dir,
        )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=gpus,
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


def _get_last_checkpoint_path(trainer: pl.Trainer, save_dir: str) -> Optional[str]:
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


def main(input_args=None):
    if input_args is None:
        args = get_args()
    else:
        args = input_args

    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    if args.contrastive_training and not args.unified:
        raise ValueError("Contrastive training is only supported with unified_fm=True.")
    if args.agg_consistency and not args.unified:
        raise ValueError(
            "Aggregation consistency losses are only supported with unified_fm=True."
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
    )
    datamodule.setup(stage="fit")

    if args.resume_from_checkpoint:
        model = CancerFoundation.load_from_checkpoint(
            args.resume_from_checkpoint, vocab=datamodule.vocab
        )
    else:
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
            no_invert_dat=args.no_invert_dat,
            conditions=args.conditions + ["modality"]
            if args.unified
            else args.conditions,
            conditions_nums=datamodule.conditions_nums if args.conditions else None,
            mvc_decoder_style=args.mvc_decoder_style,
            scale_zero_expression=args.scale_zero_expression,
            data_path=args.train_path,
            do_mvc=args.do_mvc,
            zero_percentages=args.zero_percentages,
            balance_primary=args.balance_primary,
            balance_secondary=args.balance_secondary,
            compile_model=args.compile,
            activation=args.activation,
            norm_first=args.norm_first,
            cell_emb_style=args.cell_emb_style,
            batchnorm=args.batchnorm,
            explicit_zero_prob=args.explicit_zero_prob,
            dat_scale=args.dat_scale,
            normalise_bins=args.normalise_bins,
            where_condition=args.where_condition,
            gen_method=args.gen_method,
            their_init_weights=args.their_init_weights,
            # Unified FM parameters
            contrastive=args.contrastive_training,
            aggregation=args.agg_consistency,
            agg_fn=args.agg_fn,
            noise=args.noise,
            esm_emb=args.esm_emb,
            esm_emb_path=args.esm_emb_path,
            esm_emb_finetune=args.esm_finetune,
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
        resume_from_checkpoint=args.resume_from_checkpoint,
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
    )

    # Return latest model checkpoint
    return _get_last_checkpoint_path(trainer, args.save_dir)


if __name__ == "__main__":
    main()
