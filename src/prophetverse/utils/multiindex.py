import pandas as pd

__all__ = ["reindex_time_series"]


def reindex_time_series(df: pd.DataFrame, new_time_index: pd.Index) -> pd.DataFrame:
    """Reindexes the time index level (-1) of a multi-index DataFrame with a new time index.

    Args:
        df (pd.DataFrame): The original DataFrame with a multi-level index.
        new_time_index (pd.Index or similar): The new time index to apply to the DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with the updated time index.
    """
    if df.index.nlevels == 1:
        return df.reindex(new_time_index)

    # Extract the current indices except for the time index
    levels = df.index.levels[:-1]
    names = df.index.names[:-1]

    # Create a new multi-index combining the existing levels with the new time index
    new_index = pd.MultiIndex.from_product(levels + [new_time_index], names=names + [df.index.names[-1]])

    # Reindex the DataFrame using the new index
    return df.reindex(new_index)
