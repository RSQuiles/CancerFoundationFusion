import numpy as np
import pandas as pd
from typing import List


def concat_categorical_codes(series_list: List[pd.Categorical]) -> pd.Categorical:
    """
    Efficiently combine multiple categorical arrays into a single encoding.

    Creates a combined categorical where each unique combination of input
    categories gets a unique code. Only combinations that exist in the data
    are assigned codes (sparse encoding).

    Args:
        series_list (List[pd.Categorical]): List of categorical arrays to combine.
            All arrays must have the same length.

    Returns:
        pd.Categorical: Combined categorical with compressed codes representing
            unique combinations present in the data.

    Example:
        >>> cat1 = pd.Categorical(["a", "a", "b", "b"])
        >>> cat2 = pd.Categorical(["x", "y", "x", "y"])
        >>> combined = concat_categorical_codes([cat1, cat2])
        >>> # Results in 4 unique codes for (a,x), (a,y), (b,x), (b,y)
    """
    # Get the codes for each categorical
    codes_list = [s.codes.astype(np.int32) for s in series_list]
    n_cats = [len(s.categories) for s in series_list]

    # Calculate combined codes
    combined_codes = codes_list[0]
    multiplier = n_cats[0]
    for codes, n_cat in zip(codes_list[1:], n_cats[1:]):
        combined_codes = (combined_codes * n_cat) + codes
        multiplier *= n_cat

    # Remap to dense 0..K-1 with a single numpy pass — avoids a 100M-element Python loop.
    _, combined_codes = np.unique(combined_codes, return_inverse=True)
    return pd.Categorical.from_codes(
        codes=combined_codes,
        categories=np.arange(int(combined_codes.max()) + 1),
        ordered=False,
    )
