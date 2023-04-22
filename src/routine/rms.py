"""Compute and save RMS data."""


from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from obspy.signal import PPSD

from tqdm import tqdm


from utils.misc import create_if_no_dir


def _compute_rms(
    ppsd_dfs: Optional[Dict[str, PPSD]] = None,
    freqs_pairs: List[Tuple[float, float]] = [(0.1, 1.0), (1.0, 20.0), (4.0, 14.0), (4.0, 20.0)],
    output: str = "displacement"
) -> Dict[str, pd.DataFrame]:
    """Compute RMS values.

    Parameters
    ----------
    ppsd_dfs : optional dict of {str: pd.DataFrame}, default=None
        Dictionary containing as keys the mseedid of sensors and as values the corresponding
        power spectral density dataframes.

    freqs_pairs : list of tuple, default=[(0.1, 1.0), (1.0, 20.0), (4.0, 14.0), (4.0, 20.0)]
        All pairs of frequencies to compute RMS values.

    output : {"acceleration", "velocity", "displacement"}, default="displacement
        RMS type of data.

    returns
    -------
    rms_dfs : dict of {str: pd.DataFrame}
        Dictionary containing as keys the mseedid of sensors and as values the corresponding
        RMS dataframes.
    """
    rms_dfs = {}

    if ppsd_dfs is None or len(ppsd_dfs) <= 0:
        return rms_dfs

    pbar = tqdm(ppsd_dfs.items())

    for mseedid, ppsd_df in pbar:

        freqs = ppsd_df.columns.to_numpy()

        rms_df = pd.DataFrame(index=ppsd_df.index)

        pbar.set_description(f"Computing RMS {output} data ({mseedid})")

        for fmin, fmax in freqs_pairs:

            bin_mask = (freqs >= fmin) & (freqs <= fmax)
            freqs_in_bin = freqs[bin_mask]
            df_bin = ppsd_df[freqs_in_bin]

            w = 2.0 * np.pi * freqs_in_bin

            # Acceleration amplitude in m**2 x s**-4 x Hz**-1 (obspy computes 10*log10(amp))
            aamp = 10.0 ** (df_bin / 10.)

            # Velocity amplitude (acceleration / w**2)
            vamp = aamp / w**2

            # Displacement amplitude (velocity / w**2)
            damp = vamp / w**2

            if output == "acceleration":
                amp = aamp
            elif output == "velocity":
                amp = vamp
            elif output == "displacement":
                amp = damp
            else:
                err_msg = f"Unknown amplitude parameter {output}. Please choose one of:"
                err_msg += """ "acceleration", "velocity" and "displacement"."""
                raise ValueError(err_msg)

            # Compute the RMS in time domain as the sqrt of the integral of the power spectrum
            rms_df[f"[{fmin:.2f}, {fmax:.2f}] Hz"] = np.sqrt(np.trapz(amp, x=freqs_in_bin))

        rms_dfs[mseedid] = rms_df

    return rms_dfs


def _save_rms(
    rms_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    data_dir: str = "data",
    start_date: str = "2020-03-01",
    end_date: str = "2022-01-01",
    output: str = "displacement"
):
    """Save RMS values into a csv file.

    Parameters
    ----------
    rms_dfs : dict of {str: pd.DataFrame}
        Dictionary containing as keys the mseedid of sensors and as values the corresponding
        RMS dataframes.

    data_dir : str, default="data"
        Name of the directory where data is stored.

    start_date : str, default="2020-03-01"
        Start date.

    end_date : str, default="2022-01-01"
        End date.

    output : {"acceleration", "velocity", "displacement"}, default="displacement
        RMS type of data.
    """
    if rms_dfs is None or len(rms_dfs) <= 0:
        return

    # Create the dataset directory where data will be stored
    create_if_no_dir(data_dir)

    # Save the displacement data
    pbar = tqdm(rms_dfs.items())

    for mseedid, rms_df in pbar:

        pbar.set_description(f"Saving RMS {output} data ({mseedid})")

        rms_df = rms_df.sort_index()

        filename_csv = f"{data_dir}/{output}_from_{start_date}_to_{end_date}_{mseedid}.csv"

        rms_df.to_csv(filename_csv)
