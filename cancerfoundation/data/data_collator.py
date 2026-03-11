from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import torch
import numpy as np
from cancerfoundation.data.preprocess import binning


@dataclass
class AnnDataCollator:
    """
    Data collator for the mask value learning task. It pads the sequences to
    the maximum length in the batch and masks the gene expression values.

    Args:
        do_padding (:obj:`bool`): whether to pad the sequences to the max length.
        pad_token_id (:obj:`int`, optional): the token id to use for padding.
            This is required if do_padding is True.
        pad_value (:obj:`int`): the value to use for padding the expression
            values to the max length.
        do_mlm (:obj:`bool`): whether to do masking with MLM.
        do_binning (:obj:`bool`): whether to bin the expression values.
        mask_ratio (:obj:`float`): the probability of masking with MLM.
        mask_value (:obj:`int`): the value to fill at the expression postions
            that are masked.
        max_length (:obj:`int`, optional): the maximum length of the sequences.
            This is required if do_padding is True.
        sampling (:obj:`bool`): whether to do sampling instead of truncation if
            length > max_length.
        reserve_keys (:obj:`List[str]`, optional): a list of keys in the examples
            to reserve in the output dictionary. Default to []. These fields
            will be kept unchanged in the output.
        keep_first_n_tokens (:obj:`int`): the number of tokens in the beginning
            of the sequence to keep unchanged from sampling. This is useful when
            special tokens have been added to the beginning of the sequence.
            Default to 1.
        data_style (:obj:`str`): the style of the data. If "pcpt", the data is
            masked and padded for perception training. If "gen", only the gene
            tokens are provided, but not the expression values, for pure generative
            training setting. If "both", the output will contain both fields above.
            Choices: "pcpt", "gen", "both". Default to "pcpt".
    """

    normalise_bins: bool
    condition_token: bool
    do_padding: bool = True
    gene_key: str = "var_gene_token"
    pad_token_id: Optional[int] = None
    pad_value: int = 0
    do_mlm: bool = True
    do_binning: bool = True

    probabilistic_augmentation: bool = False
    mask_ratio: float = 0.15
    mask_value: int = -1
    max_length: Optional[int] = None
    sampling: bool = True
    reserve_keys: List[str] = field(default_factory=lambda: [])
    keep_first_n_tokens: int = 1
    data_style: str = "pcpt"
    n_bins: int = None
    conditions: List[str] = None
    cls_predictions: List[str] = None
    zero_percentages: Optional[List[float]] = None

    def __post_init__(self):
        if self.do_padding:
            if self.pad_token_id is None:
                raise ValueError("`pad_token_id` is required if `do_padding`.")
            if self.max_length is None:
                raise ValueError("`max_length` is required if `do_padding`.")

        if isinstance(self.mask_ratio, float):
            if self.mask_ratio <= 0 or self.mask_ratio >= 1:
                raise ValueError("`mask_ratio` must be between 0 and 1.")
        elif isinstance(self.mask_ratio, (list, tuple)):
            if min(self.mask_ratio) <= 0 or max(self.mask_ratio) >= 1:
                raise ValueError("`mask_ratio` must be between 0 and 1.")
        else:
            raise ValueError("`mask_ratio` must be a float or iterable of floats.")

        if isinstance(self.reserve_keys, str):
            self.reserve_keys = [self.reserve_keys]

        if self.keep_first_n_tokens < 0 or self.keep_first_n_tokens > self.max_length:
            raise ValueError(
                "`keep_first_n_tokens` must be between 0 and `max_length` "
                f"({self.max_length})."
            )

        if self.data_style not in ["pcpt", "gen", "both"]:
            raise ValueError("`data_style` must be one of 'pcpt', 'gen', 'both'.")

        if self.do_binning != (self.n_bins is not None):
            raise ValueError("`do_binning` and `n_bins` have to be set in tandem.")

    def __call__(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            examples (:obj:`List[Dict[str, torch.Tensor]]`): a list of data dicts.
                Each dict is for one cell. It contains multiple 1 dimensional tensors
                like the following exmaple:
                    'X': tensor([36572, 17868, ..., 17072]),
                    '_store_idx': tensor([ 0.,  2., ..., 18.])}

        Returns:
            :obj:`Dict[str, torch.Tensor]`: a dict of tensors.
        """

        if len(self.reserve_keys) > 0:
            assert all(key in examples[0] for key in self.reserve_keys), (
                f"reserve_keys must be a subset of the keys in the examples. "
                f"Got {self.reserve_keys} but expected keys in {list(examples[0].keys())}."
            )

        if self.data_style == "pcpt":
            data_dict = self._call_pcpt(examples)
        elif self.data_style == "gen":
            data_dict = self._call_gen(examples)
        elif self.data_style == "both":
            data_dict = self._call_both(examples)

        # add reserved keys
        device = examples[0]["genes"].device
        for key in self.reserve_keys:
            data_ = [example[key] for example in examples]
            data_dict[key] = torch.stack(data_, dim=0).to(device)

        if self.conditions:
            data_dict["conditions"] = {}
            for condition in self.conditions:
                data_dict["conditions"][condition] = torch.LongTensor(
                    [d[condition] for d in examples]
                ).to(device)

        return data_dict

    def _call_pcpt(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Each example is like:
            {'id': tensor(184117),
            'genes': tensor([36572, 17868, ..., 17072]),
            'expressions': tensor([ 0.,  2., ..., 18.])}

        Args:
            examples (:obj:`List[Dict[str, torch.Tensor]]`): a list of examples.
                Each example is a dictionary of tensors.
        Returns:
            :obj:`Dict[str, torch.Tensor]`: a dictionary of tensors.
        """
        if not isinstance(examples[0], Mapping):
            return NotImplementedError

        device = examples[0]["genes"].device

        max_ori_len = max(len(example["genes"]) for example in examples)
        _max_length = self.max_length if max_ori_len >= self.max_length else max_ori_len

        # pad and truncate
        padded_genes = []
        padded_expressions = []
        for i in range(len(examples)):
            genes = examples[i]["genes"]
            expressions = examples[i]["expressions"]
            if self.do_binning:
                if self.normalise_bins:
                    expressions[self.keep_first_n_tokens :] = (
                        binning(
                            row=expressions[self.keep_first_n_tokens :],
                            n_bins=self.n_bins,
                        )
                        / self.n_bins
                    )  # scale to 0-1
                else:
                    expressions[self.keep_first_n_tokens :] = binning(
                        row=expressions[self.keep_first_n_tokens :],
                        n_bins=self.n_bins,
                    )

            zero_percentage = self._sample_zero_percentage()

            genes, expressions = self._sample_or_truncate_plus_pad(
                genes,
                expressions,
                _max_length,
                zero_percentage=zero_percentage,
                pcpt=True,
            )  # torch tensors of length _max_length

            padded_genes.append(genes)
            padded_expressions.append(expressions)

        padded_genes = torch.stack(padded_genes, dim=0).to(device)
        padded_expressions = torch.stack(padded_expressions, dim=0).to(device)

        # Create padding mask: True indicates a padded position
        gene_key_padding_mask = padded_genes.eq(self.pad_token_id)

        data_dict = {
            "gene": padded_genes,
            "expr": padded_expressions,
            "gene_key_padding_mask": gene_key_padding_mask,
        }

        # mask
        if self.do_mlm:
            masked_expressions = self._mask(
                padded_expressions, self.keep_first_n_tokens
            )
        else:
            masked_expressions = padded_expressions
        data_dict["masked_expr"] = masked_expressions
        return data_dict

    def _call_gen(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        This method will simply return the gene ids, with needed padding. There is
        no masking for pure generative training, and no input of expr values.

        Each example is like:
            {'id': tensor(184117),
            'genes': tensor([36572, 17868, ..., 17072])}

        Returns:
            Dict[str, torch.Tensor]: a dict of tensors.
            Example:
                {'pcpt_gene': tensor([[36572, 17868, ..., 17072],
                                        [36572, 17868, ..., 17072],
                                        ...,
                                        [36572, 17868, ..., 17072]]),
                'pcpt_expr': tensor([[ 0.,  2., ..., 18.],
                                        [ 0.,  2., ..., 18.],
                                        ...,
                                        [ 0.,  2., ..., 18.]])}
        """

        if not isinstance(examples[0], Mapping):
            return NotImplementedError

        device = examples[0]["genes"].device

        max_ori_len = max(len(example["genes"]) for example in examples)
        _max_length = self.max_length if max_ori_len >= self.max_length else max_ori_len

        # pad and truncate
        padded_pcpt_genes = []
        padded_pcpt_expressions = []
        for i in range(len(examples)):
            genes = examples[i]["genes"]
            expressions = examples[i]["expressions"]
            if self.do_binning:
                expressions[self.keep_first_n_tokens :] = binning(
                    row=expressions[self.keep_first_n_tokens :],
                    n_bins=self.n_bins,
                )
            genes, expressions = self._sample_or_truncate_plus_pad(
                genes, expressions, _max_length
            )
            padded_pcpt_genes.append(genes)
            padded_pcpt_expressions.append(expressions)

        padded_pcpt_genes = torch.stack(padded_pcpt_genes, dim=0).to(device)
        padded_pcpt_expressions = torch.stack(padded_pcpt_expressions, dim=0).to(device)

        # Create padding mask: True indicates a padded position
        pcpt_key_padding_mask = padded_pcpt_genes.eq(self.pad_token_id)

        data_dict = {
            "pcpt_gene": padded_pcpt_genes,
            "pcpt_expr": padded_pcpt_expressions,
            "pcpt_key_padding_mask": pcpt_key_padding_mask,
        }
        return data_dict

    def _call_both(
        self,
        examples: List[Dict[str, torch.Tensor]],
        gen_prob: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        This method will split the input into the peception part and the generation
        part. The perception part will be processed into gene ids and expr values,
        and the generation part will be processed into gene ids only.

        By default, the mask_ratio will be used to select the genes assigned to
        the generation part.

        Each example is like:
            {'id': tensor(184117),
            'genes': tensor([36572, 17868, ..., 17072]),
            'expressions': tensor([ 0.,  2., ..., 18.])}

        Args:
            gen_prob (float, optional): the probability of a gene being assigned to
                the generation part. If not provided, the mask_ratio will be used.

        Returns:
            Dict[str, torch.Tensor]: a dict of tensors.
            Example:
                {'pcpt_gene': tensor([[36572, 17868, ..., 17072],
                                        [36572, 17868, ..., 17072],
                                        ...,
                                        [36572, 17868, ..., 17072]]),
                'pcpt_expr': tensor([[ 0.,  2., ..., 18.],
                                        [ 0.,  2., ..., 18.],
                                        ...,
                                        [ 0.,  2., ..., 18.]]),
                'gen_gene': tensor([[36573, 17869, ..., 17073],
                                        [36573, 17869, ..., 17073],
                                        ...,
                                        [36573, 17869, ..., 17073]]),
                'gen_expr_target': tensor([[ 1.,  3., ..., 19.],
                                        [ 1.,  3., ..., 19.],
                                        ...,
                                        [ 1.,  3., ..., 19.]])}
        """
        if not isinstance(examples[0], Mapping):
            return NotImplementedError

        if not self.do_mlm:
            # if not doing mlm, then the perceptrual part is the whole input
            return self._call_gen(examples)

        if gen_prob is None:
            gen_prob = self.get_mask_ratio()

        max_ori_len = max(len(example["genes"]) for example in examples)
        _max_length = self.max_length if max_ori_len >= self.max_length else max_ori_len

        gen_length = int((_max_length - self.keep_first_n_tokens) * gen_prob)
        pcpt_length = _max_length - gen_length  # perception part length

        # pad and truncate
        padded_pcpt_genes = []
        padded_pcpt_expressions = []
        padded_gen_genes = []
        padded_gen_expressions = []
        for i in range(len(examples)):
            genes = examples[i]["genes"]
            expressions = examples[i]["expressions"]

            if self.probabilistic_augmentation:
                expressions[self.keep_first_n_tokens :] = self._probabilistic_augment(
                    X=expressions[self.keep_first_n_tokens :]
                )

            if self.do_binning:
                if self.normalise_bins:
                    expressions[self.keep_first_n_tokens :] = (
                        binning(
                            row=expressions[self.keep_first_n_tokens :],
                            n_bins=self.n_bins,
                        )
                        / self.n_bins
                    )  # scale to 0-1
                else:
                    expressions[self.keep_first_n_tokens :] = binning(
                        row=expressions[self.keep_first_n_tokens :],
                        n_bins=self.n_bins,
                    )

            (
                gen_genes,
                gen_expressions,
                pcpt_genes,
                pcpt_expressions,
            ) = self._random_split(
                genes[self.keep_first_n_tokens :],
                expressions[self.keep_first_n_tokens :],
                ratio=gen_prob,
            )

            pcpt_genes = torch.cat(
                (genes[: self.keep_first_n_tokens], pcpt_genes), dim=0
            )
            pcpt_expressions = torch.cat(
                (expressions[: self.keep_first_n_tokens], pcpt_expressions), dim=0
            )

            zero_percentage = self._sample_zero_percentage()

            pcpt_genes, pcpt_expressions = self._sample_or_truncate_plus_pad(
                pcpt_genes,
                pcpt_expressions,
                pcpt_length,
                zero_percentage=zero_percentage,
                pcpt=True,
            )  # torch tensors of length pcpt_length

            padded_pcpt_genes.append(pcpt_genes)
            padded_pcpt_expressions.append(pcpt_expressions)

            gen_genes, gen_expressions = self._sample_or_truncate_plus_pad(
                gen_genes,
                gen_expressions,
                gen_length,
                zero_percentage=zero_percentage,
                pcpt=False,
            )  # torch tensors of length gen_length

            padded_gen_genes.append(gen_genes)
            padded_gen_expressions.append(gen_expressions)

        padded_pcpt_genes = torch.stack(padded_pcpt_genes, dim=0)
        padded_pcpt_expressions = torch.stack(padded_pcpt_expressions, dim=0)
        padded_gen_genes = torch.stack(padded_gen_genes, dim=0)
        padded_gen_expressions = torch.stack(padded_gen_expressions, dim=0)

        # Create padding masks: True indicates a padded position
        pcpt_key_padding_mask = padded_pcpt_genes.eq(self.pad_token_id)
        gen_key_padding_mask = padded_gen_genes.eq(self.pad_token_id)

        # Create attention mask to prevent gen tokens from attending to other gen tokens
        L_pcpt = padded_pcpt_genes.shape[1]
        L_gen = padded_gen_genes.shape[1]
        attn_mask = torch.zeros((L_pcpt + L_gen, L_pcpt + L_gen), dtype=torch.bool)
        attn_mask[L_pcpt:, L_pcpt:] = True
        attn_mask[L_pcpt:, L_pcpt:].fill_diagonal_(False)

        data_dict = {
            "pcpt_gene": padded_pcpt_genes,
            "pcpt_expr": padded_pcpt_expressions,
            "pcpt_key_padding_mask": pcpt_key_padding_mask,
            "gen_gene": padded_gen_genes,
            "gen_expr_target": padded_gen_expressions,
            "gen_key_padding_mask": gen_key_padding_mask,
            "attn_mask": attn_mask,
        }

        return data_dict

    def _probabilistic_augment(
        self,
        X: torch.Tensor,
        add_noise: bool = False,
        noise_std: int = 0.01,
    ) -> torch.Tensor:
        """
        Augments data using a sampling-based approach with PyTorch.

        Parameters:
        - X: torch.Tensor
            Log1p-transformed count data (cells x genes).
        - add_noise: bool, optional
            Whether to add Gaussian noise to the augmented data.
        - noise_std: float, optional
            Standard deviation of the Gaussian noise (if add_noise=True).

        Returns:
        - augmented_data: torch.Tensor
            Augmented log1p-transformed data (cells x genes).
        """
        # Back-transform log1p-transformed data to counts
        cell_counts = torch.expm1(X)

        # Initialize the augmented data tensor
        augmented_data = torch.zeros_like(X)

        # Normalize counts to create a probability distribution
        total_count = cell_counts.sum()
        if total_count == 0:
            return X

        prob_dist = cell_counts / total_count

        # Determine the number of samples to draw
        n_samples = int(total_count.item())

        # Sample gene indices based on the probability distribution
        sampled_indices = torch.multinomial(prob_dist, n_samples, replacement=True)

        # Aggregate sampled indices into a new count vector
        sampled_counts = torch.bincount(
            sampled_indices, minlength=cell_counts.numel()
        ).float()

        # Reapply log1p transformation to the sampled data
        augmented_data = torch.log1p(sampled_counts).view_as(X)

        # Optionally add noise
        if add_noise:
            noise = torch.normal(
                mean=0.0,
                std=noise_std,
                size=augmented_data.shape,
                device=augmented_data.device,
            )
            augmented_data += noise

        return augmented_data

    def _sample_zero_percentage(self) -> Optional[float]:
        return (
            self.zero_percentages[torch.randperm(len(self.zero_percentages))[0]]
            if self.zero_percentages is not None
            else None
        )

    def _random_split(
        self,
        *arrays: torch.Tensor,
        ratio: float,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Randomly split the arrays into two parts. The first part will have the
        length of `ratio * length`, and the second part will have the length of
        `(1 - ratio) * length`. When multiple arrays are provided, they are supposed
        to have the same length.

        This method reflects the behavior of `sklearn.model_selection.train_test_split`

        Args:
            *arrays (torch.Tensor): the arrays to be split.
            ratio (float): the ratio of the first part.

        Returns:
            Tuple[torch.Tensor, ...]: the split arrays.
        """
        assert len(arrays) > 0
        assert 0 < ratio < 1
        if len(arrays) > 1:
            assert all(
                array.shape[0] == arrays[0].shape[0] for array in arrays
            ), "The arrays must have the same length."

        length = arrays[0].shape[0]
        split_index = int(length * ratio)

        indices = torch.randperm(length, device=arrays[0].device)
        first_part_indices = indices[:split_index]
        second_part_indices = indices[split_index:]

        first_parts = tuple(array[first_part_indices] for array in arrays)
        second_parts = tuple(array[second_part_indices] for array in arrays)

        return first_parts + second_parts

    def get_mask_ratio(self) -> float:
        """
        Get the mlm probability for the current step.
        """
        if isinstance(self.mask_ratio, float):
            return self.mask_ratio
        elif isinstance(self.mask_ratio, list):
            # random choose a probability
            return np.random.choice(self.mask_ratio)
        else:
            raise ValueError(
                "mask_ratio must be a float or a list of floats, "
                f"but got {self.mask_ratio}."
            )

    def _mask(
        self, expressions: torch.Tensor, keep_first_n_tokens: int = 0
    ) -> torch.Tensor:
        """
        Mask the expression values with MLM.
        """
        if keep_first_n_tokens > 0:
            result_ = self._mask(
                expressions[:, keep_first_n_tokens:],
                keep_first_n_tokens=0,
            )
            return torch.cat([expressions[:, :keep_first_n_tokens], result_], dim=1)

        device = expressions.device
        shape = expressions.shape

        probability_matrix = torch.full(shape, self.get_mask_ratio())
        # set padded postion probability to 0
        probability_matrix[expressions.eq(self.pad_value)] = 0
        if self.keep_first_n_tokens > 0:
            probability_matrix[:, : self.keep_first_n_tokens] = 0

        mask = torch.bernoulli(probability_matrix).bool()
        mask = mask.to(device)

        masked_expressions = expressions.masked_fill(mask, self.mask_value)
        return masked_expressions

    def _sample_or_truncate_plus_pad(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        max_length: int,
        pcpt: bool,
        zero_percentage: Optional[int] = None,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        assert len(genes) == len(expressions)
        if len(genes) == max_length:
            return genes, expressions

        if pcpt:
            max_length += self.keep_first_n_tokens

        if len(genes) > max_length:  # sample or truncate
            if self.sampling:
                genes, expressions = self._sample(
                    genes, expressions, max_length, zero_percentage
                )
                if (
                    genes.shape[0] == max_length
                ):  # If the sequence is not long enough, we have to pad
                    return genes, expressions
            else:
                return genes[:max_length], expressions[:max_length]
        return self._pad(genes, expressions, max_length)  # PAD

    def _sample(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        max_length: int,
        zero_percentage: Optional[float] = None,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        # NOTE: the fastest way to sample in torch has been benchmarked here
        # https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/19
        # it shows the randperm on gpu is the fastest.
        # NOTE: also, the current implementation permute the orders of the genes
        # and expressions, although it is probably a nice argmentation.
        device = genes.device
        if zero_percentage is not None:
            zero = torch.nonzero(expressions == 0).flatten()
            non_zero = torch.nonzero(expressions).flatten()
            max_length_ = max_length
            if self.keep_first_n_tokens > 0:
                non_zero = non_zero[self.keep_first_n_tokens :]
                max_length_ = max_length - self.keep_first_n_tokens

            if zero.shape[0] > 0:
                percentage_zero_relative = (zero_percentage * max_length_) / zero.shape[
                    0
                ]
                mask = torch.rand(zero.size()) <= percentage_zero_relative
                zero_sel_indices = zero[mask]
            else:
                zero_sel_indices = zero
            non_zero_max_length = min(
                non_zero.shape[0], max_length_ - zero_sel_indices.shape[0]
            )
            non_zero_sel_indices = non_zero[
                torch.randperm(len(non_zero), device=device)[:non_zero_max_length]
            ]

            indices = torch.cat((zero_sel_indices, non_zero_sel_indices), dim=0)
            if self.keep_first_n_tokens > 0:
                indices = torch.cat((torch.arange(self.keep_first_n_tokens), indices))

            return genes[indices], expressions[indices]
        else:
            if self.keep_first_n_tokens == 0:
                indices = torch.randperm(len(genes), device=device)[:max_length]
                return genes[indices], expressions[indices]

            # keep the first n tokens unchanged
            _n = self.keep_first_n_tokens
            indices = torch.randperm(len(genes) - _n, device=device)[: max_length - _n]
            indices = torch.cat([torch.arange(_n), indices + _n], dim=0)
            return genes[indices], expressions[indices]

    def _pad(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        max_length: int,
    ):
        device = genes.device
        genes = torch.cat(
            [
                genes,
                torch.full(
                    (max_length - len(genes),),
                    self.pad_token_id,
                    dtype=genes.dtype,
                    device=device,
                ),
            ]
        )
        expressions = torch.cat(
            [
                expressions,
                torch.full(
                    (max_length - len(expressions),),
                    self.pad_value,
                    dtype=expressions.dtype,
                    device=device,
                ),
            ]
        )
        return genes, expressions
