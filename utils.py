import argparse
import sys

sys.path.insert(0, "../")
from cancerfoundation.loss import LossType
from pathlib import Path
from pytorch_lightning.callbacks import TQDMProgressBar
from enum import Enum


class Precision(str, Enum):
    MIXED = "16-mixed"
    FULL_32 = "32"
    BFLOAT = "bf16-mixed"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--save-dir",
        type=str,
        help="The directory to save the trained model and the results.",
    )

    parser.add_argument(
        "--precision",
        type=str,
        choices=["32", "16-mixed", "bf16-mixed"],
        default="32",
        help="The precision to use for training. Default is 32.",
    )

    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="The directory to load the checkpoint.",
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=None,
        help="Path to the pretrained model weights.",
    )

    # settings for data
    parser.add_argument(
        "--n-hvg",
        type=int,
        default=None,
        help="The number of highly variable genes. If set to 0, will use all genes. "
        "Default is None, which will determine the n_hvg automatically.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="The number of gradient accumulation steps. Default is 1.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--grad-accu-steps",
        type=int,
        default=1,
        help="The number of gradient accumulation steps. Default is 1.",
    )

    parser.add_argument(
        "--loss",
        type=LossType,
        default=LossType.MSE,
        help="The loss function used for the gene expression prediction. Default is ordinal_cross_entropy (Ordinal Cross Entropy).",
    )
    parser.add_argument(
        "--input-style",
        type=str,
        choices=["normed_raw", "log1p", "binned"],
        default="binned",
        help="The style of the input data. Default is binned.",
    )
    parser.add_argument(
        "--input-emb-style",
        type=str,
        choices=["mine", "theirs"],
        default="continuous",
        help="The style of the input embedding. Default is continuous.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        help="The number of bins to use for the binned input style. Recommended is 51.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1200,
        help="The maximum length of the sequence. Default is 1200. The actual used "
        "max length would be the minimum of this value and the length of the longest "
        "sequence in the data.",
    )
    parser.add_argument(
        "--training-tasks",  # choices of "pcpt", "gen", "both"
        type=str,
        default="both",
        choices=["pcpt", "gen", "both"],
        help="The tasks to use for training. pcpt: perception training with maked token "
        "learning. gen: generation. Default is both.",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.40,
        help="The ratio of masked values in the training data. Default is 0.40. This"
        "value will be ignored if --training-tasks is set to gen or both.",
    )
    parser.add_argument(
        "--trunc-by-sample",
        action="store_true",
        help="Whether to truncate the input by sampling rather than cutting off if "
        "sequence length > max_seq_length. Default is False.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="The batch size for training. Default is 32.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="The number of epochs for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="The learning rate for training. Default is 1e-3.",
    )
    parser.add_argument(
        "--scheduler-interval",
        type=int,
        default=100,
        help="The interval iterations for updating the learning rate. Default is 100. "
        "This will only be used when warmup-ratio is 0.",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.99,
        help="The factor for updating the learning rate. Default is 0.99. "
        "This will only be used when warmup-ratio is 0.",
    )
    parser.add_argument(
        "--warmup-ratio-or-step",
        type=float,
        default=0.1,
        help="The ratio of warmup steps out of the total training steps. Default is 0.1. "
        "If warmup-ratio is above 0, will use a cosine scheduler with warmup. If "
        "the value is above 1, will use it as the number of warmup steps.",
    )
    parser.add_argument(
        "--no-cls",
        action="store_true",
        help="Whether to deactivate the classification loss. Default is False.",
    )
    # settings for model
    parser.add_argument(
        "--nlayers",
        type=int,
        default=4,
        help="The number of layers for the transformer. Default is 4.",
    )
    parser.add_argument(
        "--nheads",
        type=int,
        default=4,
        help="The number of heads for the transformer. Default is 4.",
    )
    parser.add_argument(
        "--embsize",
        type=int,
        default=64,
        help="The embedding size for the transformer. Default is 64.",
    )
    parser.add_argument(
        "--d-hid",
        type=int,
        default=64,
        help="dimension of the feedforward network model in the transformer. "
        "Default is 64.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="The dropout rate. Default is 0.2.",
    )
    parser.add_argument(
        "--wandb",
        type=str,
        default=None,
        help="The project name for wandb. If set to None, no logging will occur.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=None,
        help="The conditions (obs keys) the model should be invariant to.",
    )
    parser.add_argument(
        "--scale-zero-expression",
        type=float,
        default=None,
        help="How much weight should be placed on predicting if gene expression is above 0. If None, equal weighting is used.",
    )
    parser.add_argument(
        "--mvc-decoder-style",
        type=str,
        default="inner product",
        choices=["concat query", "inner product"],
    )
    parser.add_argument(
        "--train-path", type=str, default=None, help="Path to the training data."
    )
    parser.add_argument(
        "--do-dat",
        action="store_true",
        help="Whether or not to do domain adversarial training on the conditions.",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        help="Path to the list of genes.",
        default=None,
    )
    parser.add_argument(
        "--balance-primary",
        type=str,
        help="According to which metadata (primary) one should oversample to make the data more balanced.",
        default=None,
    )
    parser.add_argument(
        "--balance-secondary",
        type=str,
        help="According to which metadata (secondary) one should oversample to make the data more balanced.",
        default=None,
    )
    parser.add_argument(
        "--zero-percentages",
        nargs="+",
        type=float,
        default=None,
        help="The percentage of zero-expressed genes sampled.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of epochs to train for in this run.",
    )
    parser.add_argument(
        "--val-check-interval",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If not set, will use a random seed.",
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Whether to compile the model with torch.compile. Default is False.",
    )

    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Run name to use for Weights and Biases",
    )

    parser.add_argument(
        "--norm-first",
        action="store_true",
        help="If set, use Pre-LayerNorm transformer architecture. Default is Post-LayerNorm.",
    )

    parser.add_argument(
        "--activation",
        type=str,
        choices=["relu", "gelu"],
        default="relu",
        help="The activation function to use in the transformer model. Default is gelu.",
    )

    parser.add_argument(
        "--do-mvc",
        action="store_true",
        help="Whether to predict all gene expression levels from just the class embedding during pre-training.",
    )

    parser.add_argument(
        "--gradient-clip-val",
        type=float,
        default=1.0,
        help="The value at which gradients will be clipped. Default is 1.0.",
    )

    parser.add_argument(
        "--cell-emb-style",
        type=str,
        default="cls",
        help="The style of the cell embedding. Default is 'cls'.",
    )

    parser.add_argument(
        "--batchnorm",
        action="store_true",
        help="Whether to use batch normalization on the input embeddings. Default is False.",
    )

    parser.add_argument(
        "--explicit-zero-prob",
        action="store_true",
        help="Whether to explicitly model the probability of a gene being zero-inflated. Default is False.",
    )

    parser.add_argument(
        "--log-interval", type=int, default=5, help="Log every n steps."
    )

    parser.add_argument(
        "--normalise-bins",
        action="store_true",
    )

    parser.add_argument(
        "--dat-scale",
        type=float,
        default=1.0,
        help="Scale for the gradient reversal layer in DAT",
    )

    parser.add_argument(
        "--where-condition",
        type=str,
        choices=["begin", "end"],
    )

    parser.add_argument(
        "--no-invert-dat",
        action="store_true",
        help="If doing domain adversarial training, whether to invert the gradients.",
    )

    parser.add_argument(
        "--gen-method",
        type=str,
        choices=["theirs", "mine", "orig", "quick"],
        help="Which method to use for generative training.",
    )

    parser.add_argument(
        "--their-init-weights",
        action="store_true",
        help="Whether to use the weight initialisation from before.",
    )

    parser.add_argument(
        "--save-every",
        action="store_true",
        help="Whether to save every epoch.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        help="How many workers to use for the DataLoader.",
    )

    # Added by Rafa: Unified Foundation Model
    parser.add_argument(
        "--unified",
        action="store_true",
        help="Whether to use the unified foundation model that takes both single-cell and bulk data as input. Default is False, which means using the original model that only takes single-cell data as input.",
    )

    parser.add_argument(
        "--bulk-ratio",
        type=float,
        default=0.3,
        help="The ratio of bulk samples in each batch. Default is 0.3.",
    )

    parser.add_argument(
        "--pb-ratio",
        type=float,
        default=0.3,
        help="The ratio of pseudobulk samples in each batch (relative to the total batch size). Default is 0.3.",
    )

    parser.add_argument(
        "--n-sc-per-pseudobulk",
        type=int,
        default=10,
        help="Number of single-cell samples to aggregate into each pseudobulk. These are drawn from the same pool as the single-cell samples in the batch, but are guaranteed to be different samples. Default is 10.",
    )

    parser.add_argument(
        "--contrastive-training",
        action="store_true",
        help="Whether to include a contrastive loss that brings the pseudobulk and real bulk samples closer together in the embedding space. Default is False.",
    )

    parser.add_argument(
        "--agg-consistency",
        action="store_true",
        help="Whether to include an aggregation consistency loss that encourages the pseudobulk embeddings to be similar to the aggregated embeddings of their constituent single-cell samples. Default is False.",
    )

    parser.add_argument(
        "--agg-fn",
        type=str,
        choices=["mean", "sum"],
        default="sum",
        help="The function to use for aggregating single-cell embeddings into pseudobulk embeddings. Default is 'mean'.",
    )

    parser.add_argument(
        "--balanced-sampler",
        choices=[None, "joint", "separate"],
        default=None,
        help="Whether to use a balanced sampler that oversamples underrepresented classes in the training data. It can treat single-cell and bulk samples separately. Default is None.",
    )

    parser.add_argument(
        "--balanced-labels",
        type=str,
        nargs="+",
        default=None,
        help="The column(s) to use for balanced sampling labels",
    )
    parser.add_argument(
        "--esm-emb",
        action="store_true",
        help="Whether to use esm-derived embeddings"
    )
    parser.add_argument(
        "--esm-emb-path",
        type=str,
        default=None,
        help="Path to a parquet file containing pretrained ESM gene embeddings.",
    )
    parser.add_argument(
        "--esm-finetune",
        action="store_true",
        help="Whether to fine-tune the pretrained ESM gene embeddings instead of freezing them.",
    )
    parser.add_argument(
        "--noise",
        type=int,
        nargs="+",
        help="Whether to perform denoising task. Indicates the noise levels to iterate through"
    )

    return parser.parse_args()


class MyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar
