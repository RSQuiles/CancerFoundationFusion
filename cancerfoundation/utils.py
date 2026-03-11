from typing import Mapping, Optional

import torch
import re


def map_pretrained_keys_to_new_format(pretrained_key):
    """
    Maps keys from pretrained model format to new model format.

    Args:
        pretrained_key (str): Key from pretrained model

    Returns:
        str: Corresponding key in new model format
    """
    # Handle self_attn Wqkv -> in_proj mapping
    if ".self_attn.Wqkv.weight" in pretrained_key:
        return pretrained_key.replace(
            ".self_attn.Wqkv.weight", ".self_attn.self_attn.in_proj_weight"
        )
    elif ".self_attn.Wqkv.bias" in pretrained_key:
        return pretrained_key.replace(
            ".self_attn.Wqkv.bias", ".self_attn.self_attn.in_proj_bias"
        )

    # Handle self_attn out_proj mapping (add extra self_attn layer)
    elif ".self_attn.out_proj.weight" in pretrained_key:
        return pretrained_key.replace(
            ".self_attn.out_proj.weight", ".self_attn.self_attn.out_proj.weight"
        )
    elif ".self_attn.out_proj.bias" in pretrained_key:
        return pretrained_key.replace(
            ".self_attn.out_proj.bias", ".self_attn.self_attn.out_proj.bias"
        )

    # For all other keys (linear1, linear2, norm1, norm2), return unchanged
    else:
        return pretrained_key


def load_pretrained(
    model: torch.nn.Module,
    pretrained_params: Mapping[str, torch.Tensor],
    gene_mapping: Optional[dict] = None,
    verbose: bool = True,
) -> torch.nn.Module:
    """
    Load pretrained weights to the model.

    Args:
        model (torch.nn.Module): The model to load weights to.
        pretrained_params (Mapping[str, torch.Tensor]): The pretrained parameters.
        strict (bool): Whether to strictly enforce that the keys in :attr:`pretrained_params`
            match the keys returned by this module's :meth:`Module.state_dict`. Default to False.
        prefix (List[str]): The list of prefix strings to match with the keys in
            :attr:`pretrained_params`. The matched keys will be loaded. Default to None.

    Returns:
        torch.nn.Module: The model with pretrained weights.
    """

    model_new_params = model.state_dict()

    not_identical = []
    updated = []
    for key in pretrained_params.keys():
        if key == "encoder.embedding.weight" and gene_mapping:
            model_new_params[key][list(gene_mapping.keys())] = pretrained_params[key][
                list(gene_mapping.values())
            ]
            if verbose:
                print(
                    f"Updated {model_new_params[key].shape} dimensional gene embedding with {len(gene_mapping)} genes from pretrained weights."
                )

            continue

        model_key = map_pretrained_keys_to_new_format(key)
        if (
            model_key in model_new_params.keys()
            and model_new_params[model_key].shape == pretrained_params[key].shape
        ):
            model_new_params[model_key] = pretrained_params[key]
            updated.append(key)
        else:
            if key in model_new_params.keys():
                if verbose:
                    print(
                        f"Same key but not same shape: {model_new_params[key].shape} : {pretrained_params[key].shape}"
                    )
            not_identical.append(key)

    def modify_string(input_string):
        # Pattern to check if the string has the required format
        check_pattern = r"^transformer_encoder\.layers\.\d+\.self_attn\.self_attn\..*"

        # Check if the input string matches the pattern
        if re.match(check_pattern, input_string):
            # Pattern to perform the replacement
            modify_pattern = (
                r"(transformer_encoder\.layers\.\d+\.self_attn\.)self_attn\.(.*)"
            )

            # Replace the matched pattern with the required modification
            modified_string = re.sub(modify_pattern, r"\1\2", input_string)
            return modified_string
        else:
            return input_string

    for key in not_identical[:]:
        key_new = modify_string(key)
        if (
            key_new in model_new_params.keys()
            and model_new_params[key_new].shape == pretrained_params[key].shape
        ):
            model_new_params[key_new] = pretrained_params[key]
            updated.append(key_new)
            not_identical.remove(key)

    if verbose and len(not_identical) > 0:
        print("Couldn't load following keys: ", not_identical)
    if verbose:
        print(f"Updated: {updated}")

    model.load_state_dict(model_new_params)
    return model
