"""Prepare RMS data for change point detection."""


from typing import Dict, List, Tuple

from glob import glob

import numpy as np
import pandas as pd


from utils.data import get_dates_indices
from utils.misc import raise_if_no_dir


def load_rms(
    data_dir: str = "data",
    start_date: str = "2020-03-01",
    end_date: str = "2022-01-01",
    output: str = "displacement",
    sep: str = ","
) -> Dict[str, pd.DataFrame]:
    """Load RMS data.

    Parameters
    ----------
    data_dir : str, default="data"
        Name of the directory where data is stored.

    start_date : str, default="2020-03-01"
        Start date.

    end_date : str, default="2022-01-01"
        End date.

    output : {"acceleration", "velocity", "displacement"}, default="displacement
        RMS type of data.

    sep : str, default=","
        Separator in file.

    Returns
    -------
    rms_dfs : dict of {str: pd.DataFrame}
        Dictionary containing as keys the mseedid of sensors and as values the corresponding
        RMS values.
    """
    # Check that the dataset directory exists
    raise_if_no_dir(data_dir)

    # Load RMS data
    rms_dfs = {}

    filename_pattern = f"{data_dir}/{output}_from_{start_date}_to_{end_date}_*.csv"

    for filename in glob(filename_pattern):

        mseedid = filename.replace(".csv", "").split("_")[-1]

        rms_dfs[mseedid] = pd.read_csv(filename, index_col=0, parse_dates=True, sep=sep)

    return rms_dfs


def extract_rms_signal(
    rms_df: pd.DataFrame,
    start_date_signal: str = "2020-03-01",
    end_date_signal: str = "2022-01-01",
    band: str = "[4.00, 14.00] Hz",
    important_dates: List[str] = ["2020-03-15", "2020-05-11"]
) -> Tuple[pd.DatetimeIndex, np.ndarray, List[int]]:
    """Convert RMS to a signal for change point detection.

    Parameters
    ----------
    rms_df : pd.DataFrame
        Dataframe containing RMS values.

    start_date_signal : str, default="2020-03-01"
        Start date of the extracted signal.

    end_date_signal : str, default="2022-01-01"
        End date of the extracted signal.

    band : str, default="[4.00, 14.00] Hz"
        When RMS is computed, bands are saved under the convention "[{min:.2f}, {max:.2f}] Hz".

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
    start_date_idx, end_date_idx = get_dates_indices(rms_df, [start_date_signal, end_date_signal])

    rms_df = rms_df[start_date_idx:end_date_idx]

    datelist = rms_df.index

    # Find corresponding breakpoints indices (by convention, last breakpoint is the length)
    true_bkps = list(get_dates_indices(rms_df, important_dates)) + [len(rms_df)]

    # Extract signal
    signal = rms_df[band].to_numpy()

    return datelist, signal, true_bkps
