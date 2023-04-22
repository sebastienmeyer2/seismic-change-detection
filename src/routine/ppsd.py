"""Routines for downloading raw data using the `obspy` package and extracting PPSD data.

Adapted from https://github.com/ThomasLecocq/SeismoRMS.
"""


import os

from typing import Dict, Optional, Tuple

import warnings

from glob import glob

import pandas as pd

from obspy.core import UTCDateTime
from obspy.core.stream import read
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.client import FDSNNoDataException
from obspy.signal import PPSD

from tqdm import tqdm


from routine import HALF_HOUR_IN_SECONDS, ONE_DAY_IN_SECONDS
from utils.misc import create_if_no_dir, raise_if_no_dir


def _download_mseed_data(
    data_dir: str = "data",
    provider_name: str = "RESIF",
    network: str = "FR",
    station: str = "STR",
    location: str = "00",
    channel: str = "BHZ",
    start_date: str = "2020-03-01",
    end_date: str = "2022-01-01"
):
    """Download data using *obspy*.

    Parameters
    ----------
    data_dir : str, default="data"
        Name of the directory where data is stored.

    provider_name : str, default="RESIF"
        Name of the data provider.

    network : str, default="FR"
        Name of the network.

    station : str, default="STR"
        Name of the station.

    location : str, default="00"
        Location of the desired sensor.

    channel : str, default="BHZ"
        Name of the channel.

    start_date : str, default="2020-03-01"
        Start date.

    end_date : str, default="2022-01-01"
        End date.
    """
    # Create the dataset directory where data will be stored
    create_if_no_dir(data_dir)

    mseed_dir = f"{data_dir}/mseed"
    create_if_no_dir(mseed_dir)

    # Create the miniSEED id
    mseedid = f"{network}.{station}.{location}.{channel}"
    # Make sure that wildcard characters are not in mseedid
    mseedid = mseedid.replace("*", "").replace("?", "")

    # Create the list of date times
    starttime = UTCDateTime(start_date)
    endtime = min(UTCDateTime(end_date), UTCDateTime())

    nb_days = int((endtime - starttime) / ONE_DAY_IN_SECONDS) + 1

    datelist = [starttime + i * ONE_DAY_IN_SECONDS for i in range(nb_days)]
    pbar = tqdm(datelist)

    # Connect the client to the data provider
    client = Client(provider_name)

    for utcdate in pbar:

        date_str = utcdate.strftime("%Y-%m-%d")
        pbar.set_description(f"Downloading data ({date_str})")

        filename_mseed = f"{mseed_dir}/{date_str}_{mseedid}.mseed"

        # Do not force download if file exists
        if os.path.isfile(filename_mseed):
            continue

        # Download the mseed data
        starttime = utcdate - HALF_HOUR_IN_SECONDS - 1
        endtime = utcdate + ONE_DAY_IN_SECONDS + HALF_HOUR_IN_SECONDS + 1

        try:

            bulk = [(network, station, location, channel, starttime, endtime)]

            st = client.get_waveforms_bulk(bulk, attach_response=True)

            # The result of `client.get_waveforms_bulk()` contains a list of traces which
            # correspond to uninterrupted sequence of dates for which the channel has available
            # data, thus:
            # - length 0 means we have no data at all for the day
            # - length 1 means we have data for the whole day without interruption
            # - length greater than 1 means we have data with (possibly multiple) interruption(s)
            if len(st) > 0:
                st.write(filename_mseed)

        except FDSNNoDataException:

            pbar.set_description(f"No data on FDSN server for {date_str}")
            continue


def _compute_ppsd_data(
    data_dir: str = "data",
    provider_name: str = "RESIF",
    network: str = "FR",
    station: str = "STR",
    location: str = "00",
    channel: str = "BHZ",
    start_date: str = "2020-03-01",
    end_date: str = "2022-01-01",
    db_bins: Tuple[float, float, float] = (-200, 20, 0.25),
    ppsd_length: int = 1800,
    overlap: float = 0.5,
    period_smoothing_width_octaves: float = 0.025,
    period_step_octaves: float = 0.0125,
    period_limits: Tuple[float, float] = (0.008, 50)
):
    """Compute and save power spectral density values.

    Parameters
    ----------
    data_dir : str, default="data"
        Name of the directory where data is stored.

    provider_name : str, default="RESIF"
        Name of the data provider.

    network : str, default="FR"
        Name of the network.

    station : str, default="STR"
        Name of the station.

    location : str, default="00"
        Location of the desired sensor.

    channel : str, default="BHZ"
        Name of the channel.

    start_date : str, default="2020-03-01"
        Start date.

    end_date : str, default="2022-01-01"
        End date.

    db_bins : tuple of float, default=(-200, 20, 0.25)
        Specify the lower and upper boundary and the width of the db bins.

    ppsd_length : int, default=1800
        Length of data segments passed to psd in seconds.

    overlap : float, default=0.5
        Overlap of segments passed to psd. Overlap may take values between 0 and 1.

    period_smoothing_width_octaves : float, default=0.025
        Determines over what period/frequency range the psd is smoothed around every central
        period/frequency.

    period_step_octaves : float, default=0.0125
        Step length on frequency axis in fraction of octaves.

    period_limits : tuple of float, default=(0.008, 50)
        Set custom lower and upper end of period range.
    """
    # Check that the dataset directory exists
    raise_if_no_dir(data_dir)

    mseed_dir = f"{data_dir}/mseed"
    raise_if_no_dir(mseed_dir)

    ppsd_dir = f"{data_dir}/ppsd"
    create_if_no_dir(ppsd_dir)

    # Create the miniSEED id
    mseedid = f"{network}.{station}.{location}.{channel}"
    # Make sure that wildcard characters are not in mseedid
    mseedid = mseedid.replace("*", "").replace("?", "")

    # Create the list of date times
    starttime = UTCDateTime(start_date)
    endtime = min(UTCDateTime(end_date), UTCDateTime())

    nb_days = int((endtime - starttime) / ONE_DAY_IN_SECONDS) + 1

    datelist = [starttime + i * ONE_DAY_IN_SECONDS for i in range(nb_days)]
    pbar = tqdm(datelist)

    # Connect the client to the data provider
    client = Client(provider_name)

    response = client.get_stations(
        starttime=UTCDateTime(endtime), network=network, station=station, location=location,
        channel=channel, level="response"
    )

    for utcdate in pbar:

        date_str = utcdate.strftime("%Y-%m-%d")
        pbar.set_description(f"Computing PPSD ({date_str})")

        filename_mseed = f"{mseed_dir}/{date_str}_{mseedid}.mseed"

        # Missing data for current day
        if not os.path.isfile(filename_mseed):
            continue

        # Read the mseed data
        st = read(filename_mseed, headonly=True)

        for mseedid in list({tr.id for tr in st}):

            filename_npz = f"{ppsd_dir}/{date_str}_{mseedid}"

            if os.path.isfile(f"{filename_npz}.npz"):
                continue

            st = read(filename_mseed, sourcename=mseedid)
            st.attach_response(response)

            ppsd = PPSD(
                st[0].stats, response, db_bins=db_bins, ppsd_length=ppsd_length, overlap=overlap,
                period_smoothing_width_octaves=period_smoothing_width_octaves,
                period_step_octaves=period_step_octaves, period_limits=period_limits
            )

            with warnings.catch_warnings():

                warnings.simplefilter("ignore")
                ppsd.add(st)

            ppsd.save_npz(filename_npz)

            del st, ppsd  # disk space saving


def _load_ppsd_data(
    data_dir: str = "data",
    start_date: str = "2020-03-01",
    end_date: str = "2022-01-01",
    allow_pickle: bool = False
) -> Dict[str, PPSD]:
    """Load PPSD data.

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
    ppsd_data : dict of {str: PPSD}
        Dictionary containing as keys the mseedid of sensors and as values the corresponding
        power spectral density values.
    """
    # Check that the dataset directory exists
    raise_if_no_dir(data_dir)

    ppsd_dir = f"{data_dir}/ppsd"
    raise_if_no_dir(ppsd_dir)

    # Create the list of date times
    starttime = UTCDateTime(start_date)
    endtime = min(UTCDateTime(end_date), UTCDateTime())

    nb_days = int((endtime - starttime) / ONE_DAY_IN_SECONDS) + 1

    datelist = [starttime + i * ONE_DAY_IN_SECONDS for i in range(nb_days)]
    pbar = tqdm(datelist)

    # Load data
    ppsd_data = {}

    for utcdate in pbar:

        date_str = utcdate.strftime("%Y-%m-%d")
        pbar.set_description(f"Loading PPSD ({date_str})")

        filename_pattern = f"{ppsd_dir}/{date_str}_*.npz"

        # Load PPSD data
        for filename in glob(filename_pattern):

            mseedid = filename.replace(".npz", "").split("_")[-1]

            if mseedid not in ppsd_data:

                ppsd_data[mseedid] = PPSD.load_npz(filename, allow_pickle=allow_pickle)

            else:

                # Since we treated each day independently by adding a few minutes before and after,
                # the datasets might overlap, which results in a warning
                with warnings.catch_warnings():

                    warnings.simplefilter("ignore")
                    ppsd_data[mseedid].add_npz(filename, allow_pickle=allow_pickle)

    return ppsd_data


def _ppsd_to_dataframe(
    ppsd_data: Optional[Dict[str, PPSD]] = None
) -> Dict[str, pd.DataFrame]:
    """Convert `PPSD` objects to dataframes.

    Parameters
    ----------
    ppsd_data : optional dict of {str: PPSD}, default=None
        Dictionary containing as keys the mseedid of sensors and as values the corresponding
        power spectral density values.

    Returns
    -------
    ppsd_dfs : dict of {str: pd.DataFrame}
        Dictionary containing as keys the mseedid of sensors and as values the corresponding
        power spectral density dataframes.
    """
    ppsd_dfs = {}

    if ppsd_data is None:
        return ppsd_dfs

    pbar = tqdm(ppsd_data.items())

    for mseedid, ppsd in pbar:

        ppsd = ppsd_data[mseedid]

        pbar.set_description(f"Converting PPSD to dataframe ({mseedid})")

        # PPSD values are associated to periods (f = 1/T)
        datelist = pd.DatetimeIndex([d.datetime for d in ppsd.current_times_used])
        freqs = 1.0 / ppsd.period_bin_centers

        ppsd_df = pd.DataFrame(ppsd.psd_values, index=datelist, columns=freqs)

        # Sort by frequency
        ppsd_df.sort_index(axis=1, inplace=True)

        # For some frequencies, we might have no data
        ppsd_df.dropna(axis=1, how="all", inplace=True)

        # Round dates for message clarity
        ppsd_df.index = ppsd_df.index.round("min")

        ppsd_dfs[mseedid] = ppsd_df

    return ppsd_dfs
