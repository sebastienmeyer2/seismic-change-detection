"""Dataframes utils."""


from typing import List, Optional

import numpy as np
import pandas as pd

from tqdm import tqdm


def replace_df(
    df: Optional[pd.DataFrame] = None,
    min_value: float = -np.inf,
    max_value: float = np.inf
) -> Optional[pd.DataFrame]:
    """Replace values below or above thresholds with the median.

    parameters
    ----------
    df : optional pd.DataFrame, default=None
        Dataframe.

    min_value : float, default=-np.inf
        Values below minimal value are set to the median value.

    max_value : float, default=np.inf
        Values above maximal value are set to the median value.

    Returns
    -------
    df : optional pd.DataFrame
        Dataframe after processing.
    """
    if df is None:
        return df

    cols = df.columns.to_numpy()

    for col in tqdm(cols, desc="Replacing values"):

        mask = (df[col] <= min_value) | (df[col] >= max_value)

        df[col] = np.where(mask, df[col].median(), df[col])

    return df


def resample_df(
    df: Optional[pd.DataFrame] = None,
    between_begin: Optional[str] = None,
    between_end: Optional[str] = None,
    freq: str = "1D"
) -> Optional[pd.DataFrame]:
    """Resample a dataframe based on the datetime index.

    parameters
    ----------
    df : optional pd.DataFrame, default=None
        Dataframe.

    between_begin : optional str, default=None
        The beginning of the data range to keep values, e.g. "7:00".

    between_end : optional str, default=None
        The end of the data range to keep values, e.g. "19:00".

    freq : str, default="1D"
        Frequency to keep values in *pandas* format.

    Returns
    -------
    df : optional pd.DataFrame
        Dataframe after processing.
    """
    if df is None:
        return df

    if (
        between_begin is not None and between_end is None
    ) or (
        between_begin is None and between_end is not None
    ):

        err_msg = "Either give both between begin and end or none."
        raise ValueError(err_msg)

    if between_begin is not None and between_end is not None:

        df = df.between_time(between_begin, between_end)

    df = df.groupby(pd.Grouper(freq=freq)).median()

    return df


def get_dates_indices(
    df: pd.DataFrame,
    datelist: List[str]
) -> np.ndarray:
    """Get indices corresponding to datetimes.

    parameters
    ----------
    df : pd.DataFrame
        Dataframe.

    datelist : list of str
        List of datetimes.

    Returns
    -------
    indices : np.ndarray
        List of indices corresponding to datetimes.
    """
    indices = df.index.get_indexer(pd.to_datetime(datelist), method="nearest")

    return indices
