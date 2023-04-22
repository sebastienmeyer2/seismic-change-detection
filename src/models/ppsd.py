"""Prepare power spectral density values for change point detection."""


from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from routine.ppsd import _load_ppsd_data, _ppsd_to_dataframe
from utils.data import get_dates_indices


def load_ppsd(
    data_dir: str = "data",
    start_date: str = "2020-03-01",
    end_date: str = "2022-01-01",
    allow_pickle: bool = False
) -> Dict[str, pd.DataFrame]:
    """Load PPSD data and convert it to dataframes.

    This function is a wrapper of protected routines to allow for only loading PPSD data.

    Parameters
    ----------
    data_dir : str, default="data"
        Name of the directory where data is stored.

    start_date : str, default="2020-03-01"
        Start date.

    end_date : str, default="2022-01-01"
        End date

    allow_pickle : bool, default=False
        Use it to allow pickle when loading files.

    Returns
    -------
    ppsd_dfs : dict of {str: pd.DataFrame}
        Dictionary containing as keys the mseedid of sensors and as values the corresponding
        power spectral density dataframes.
    """
    # Load PPSD files
    ppsd_data = _load_ppsd_data(
        data_dir=data_dir, start_date=start_date, end_date=end_date, allow_pickle=allow_pickle
    )

    # Convert PPSD data to dataframe
    ppsd_dfs = _ppsd_to_dataframe(ppsd_data=ppsd_data)

    return ppsd_dfs


def extract_ppsd_signal(
    ppsd_df: pd.DataFrame,
    start_date_signal: str = "2020-03-01",
    end_date_signal: str = "2022-01-01",
    min_freq_signal: float = 4.0,
    max_freq_signal: float = 14.0,
    important_dates: List[str] = ["2020-03-15", "2020-05-11"]
) -> Tuple[pd.DatetimeIndex, np.ndarray, List[int]]:
    """Convert a power spectrum to a signal for change point detection.

    Parameters
    ----------
    ppsd_df : pd.DataFrame
        Dataframe containing power spectral density values.

    start_date_signal : str, default="2020-03-01"
        Start date of the extracted signal.

    end_date_signal : str, default="2022-01-01"
        End date of the extracted signal.

    min_freq_signal : float, default=4.0
        Minimal frequency of the extracted signal.

    max_freq_signal : float, default=14.0
        Maximal frequency of the extracted signal.

    important_dates : list of str, default=["2020-03-15", "2020-05-11"]
        List of important dates occurring within the extracted signal.

    Returns
    -------
    datelist : pd.DatetimeIndex
        List of datetimes of the signal.

    signal : np.ndarray
        Extracted signal for change point detection.

    true_bkps : list of int
        List of true indices for the breakpoints in the signal.
    """
    # Consider only a subset of the time range
    start_date_idx, end_date_idx = get_dates_indices(ppsd_df, [start_date_signal, end_date_signal])

    ppsd_df = ppsd_df[start_date_idx:end_date_idx]

    datelist = ppsd_df.index

    # Find corresponding breakpoints indices (by convention, last breakpoint is the length)
    true_bkps = list(get_dates_indices(ppsd_df, important_dates)) + [len(ppsd_df)]

    # Extract signal
    freqs = ppsd_df.columns.to_numpy()
    bin_mask = (freqs >= min_freq_signal) & (freqs <= max_freq_signal)
    freqs_in_bin = freqs[bin_mask]

    signal = ppsd_df[freqs_in_bin].to_numpy()

    return datelist, signal, true_bkps
