from collections import Counter
from typing import Optional

import numpy as np
from torch.utils.data import WeightedRandomSampler

from torch.utils.data import Subset


def scale_proportions_balanced(proportions, scale_factor, max_scale=5):
    if scale_factor == 1:
        return proportions
    # Calculate the original sum and the target sum
    max_scale = max(scale_factor, max_scale)

    original_sum = sum(proportions)
    target_sum = scale_factor * original_sum

    # Initialize the final scaled values (we will adjust these)
    final_values = [0.0] * len(proportions)

    # Calculate the target balanced value if all numbers were to be equal
    target_value = target_sum / len(proportions)

    # First pass: assign target_value to all numbers, respecting the scaling constraints
    remaining_sum = target_sum
    remaining_proportions = len(proportions)

    for i in range(len(proportions)):
        # Determine what scaling factor would bring the current proportion to the target value
        desired_value = target_value
        max_value = (
            proportions[i] * max_scale
        )  # Max possible value given the scaling limit
        min_value = proportions[i]  # Min possible value (scaling factor of 1)

        # If the target value exceeds the max possible value, cap it at max_value
        if desired_value > max_value:
            final_values[i] = max_value
            remaining_sum -= max_value
        # If the target value is below the min value, cap it at min_value
        elif desired_value < min_value:
            final_values[i] = min_value
            remaining_sum -= min_value
        else:
            final_values[i] = desired_value
            remaining_sum -= desired_value
        remaining_proportions -= 1

    # Second pass: distribute any remaining sum across numbers that haven't hit limits
    while remaining_sum > 0 and remaining_proportions > 0:
        distribute_value = remaining_sum / remaining_proportions
        for i in range(len(final_values)):
            if (
                final_values[i] == proportions[i] * max_scale
                or final_values[i] == proportions[i]
            ):
                continue  # Skip numbers that have hit their limit

            # Try to increase the value by distribute_value, respecting max limit
            potential_value = final_values[i] + distribute_value
            max_value = proportions[i] * max_scale

            if potential_value > max_value:
                final_values[i] = max_value
                remaining_sum -= max_value - final_values[i]
            else:
                final_values[i] = potential_value
                remaining_sum -= distribute_value

        remaining_proportions = sum(
            1
            for val in final_values
            if val != proportions[i] * max_scale and val != proportions[i]
        )

    return final_values


def get_balanced_sampler(
    dataset: Subset,
    primary_condition: str,
    secondary_condition: Optional[str] = None,
    oversample: bool = True,
):
    primary = dataset.dataset.get_metadata(primary_condition)[dataset.indices]

    if secondary_condition is not None:
        secondary = dataset.dataset.get_metadata(secondary_condition)[dataset.indices]

    class_counts = Counter(primary)

    max_class = max(class_counts.values())

    class_weights = {}
    total_count = 0

    for class_label, count in class_counts.items():
        if count < 0.4 * max_class:
            target_count = 0.4 * max_class
        elif 0.4 * max_class <= count < 0.7 * max_class:
            target_count = 0.7 * max_class
        else:
            target_count = max_class

        total_count += target_count

        if secondary_condition is None:
            class_weights[class_label] = target_count / count
        else:
            data_2 = []
            for i in range(len(dataset)):
                if primary[i] == class_label:
                    data_2.append(secondary[i])

            class_counts_2 = list(Counter(data_2).items())
            class_counts_2_balanced = scale_proportions_balanced(
                [class_counts_2_i[1] for class_counts_2_i in class_counts_2],
                scale_factor=target_count / count,
            )

            class_weights[class_label] = {}

            for i in range(len(class_counts_2_balanced)):
                class_weights[class_label][class_counts_2[i][0]] = (
                    class_counts_2_balanced[i] / class_counts_2[i][1]
                )

    if secondary_condition is not None:
        sample_weights = []
        for i in range(len(dataset)):
            sample_weights.append(class_weights[primary[i]][secondary[i]])
    else:
        sample_weights = np.vectorize(class_weights.get)(primary)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=round(total_count) if oversample else len(sample_weights),
        replacement=True,
    )
