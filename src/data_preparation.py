"""Download seismic data, then compute power spectral density and RMS."""


from typing import Dict, List, Optional, Tuple

import argparse

import pandas as pd


from routine.ppsd import (
    _download_mseed_data, _compute_ppsd_data, _load_ppsd_data, _ppsd_to_dataframe
)
from routine.rms import _compute_rms, _save_rms
from utils.misc import str2bool, float_zero_one


def prepare_ppsd(
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
    period_limits: Tuple[float, float] = (0.008, 50),
    allow_pickle: bool = False
) -> Dict[str, pd.DataFrame]:
    """Download mseed data then compute power spectral density values.

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

    allow_pickle : bool, default=False
        Use it to allow pickle when loading files.

    Returns
    -------
    ppsd_dfs : dict of {str: pd.DataFrame}
        Dictionary containing as keys the mseedid of sensors and as values the corresponding
        power spectral density dataframes.
    """
    # Download raw data (no force download if files already exist)
    _download_mseed_data(
        data_dir=data_dir,
        provider_name=provider_name,
        network=network,
        station=station,
        location=location,
        channel=channel,
        start_date=start_date,
        end_date=end_date
    )

    # Compute PPSD data and save them on disk (no force computation if files already exist)
    _compute_ppsd_data(
        data_dir=data_dir,
        provider_name=provider_name,
        network=network,
        station=station,
        location=location,
        channel=channel,
        start_date=start_date,
        end_date=end_date,
        db_bins=db_bins,
        ppsd_length=ppsd_length,
        overlap=overlap,
        period_smoothing_width_octaves=period_smoothing_width_octaves,
        period_step_octaves=period_step_octaves,
        period_limits=period_limits
    )

    # Retrieve the PPSD data from the disk
    ppsd_data = _load_ppsd_data(
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date,
        allow_pickle=allow_pickle
    )

    # Convert PPSD objects to dataframes
    ppsd_dfs = _ppsd_to_dataframe(
        ppsd_data=ppsd_data
    )

    return ppsd_dfs


def prepare_rms(
    ppsd_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    data_dir: str = "data",
    start_date: str = "2020-03-01",
    end_date: str = "2022-01-01",
    freqs_pairs: List[Tuple[float, float]] = [(0.1, 1.0), (1.0, 20.0), (4.0, 14.0), (4.0, 20.0)],
    output: str = "displacement",
    save: bool = True
) -> Dict[str, pd.DataFrame]:
    """Load power spectral density values and compute RMS displacement.

    Parameters
    ----------
    ppsd_dfs : optional dict of {str: pd.DataFrame}, default=None
        Dictionary containing as keys the mseedid of sensors and as values the corresponding
        power spectral density dataframes.

    data_dir : str, default="data"
        Name of the directory where data is stored.

    start_date : str, default="2020-03-01"
        Start date.

    end_date : str, default="2022-01-01"
        End date.

    freqs_pairs : list of tuple, default=[(0.1, 1.0), (1.0, 20.0), (4.0, 14.0), (4.0, 20.0)]
        All pairs of frequencies to compute RMS values.

    output : {"acceleration", "velocity", "displacement"}, default="displacement
        RMS type of data.

    save : bool, default=True
        If True, will save the RMS data.

    Returns
    -------
    rms_dfs : dict of {str: pd.DataFrame}
        Dictionary containing as keys the mseedid of sensors and as values the corresponding
        RMS dataframes.
    """
    # Compute RMS data
    rms_dfs = _compute_rms(
        ppsd_dfs=ppsd_dfs,
        freqs_pairs=freqs_pairs,
        output=output
    )

    # Save RMS data
    if save:

        _save_rms(
            rms_dfs=rms_dfs,
            data_dir=data_dir,
            start_date=start_date,
            end_date=end_date,
            output=output
        )

    return rms_dfs


if __name__ == "__main__":

    # Command lines
    PARSER_DESC = "Main file to prepare seismic data."
    PARSER = argparse.ArgumentParser(description=PARSER_DESC)

    # Data
    PARSER.add_argument(
        "--data-dir",
        default="data",
        type=str,
        help="""
             Name of the directory where data is stored. Default: "data".
             """
    )

    PARSER.add_argument(
        "--save",
        default="True",
        type=str2bool,
        help="""
             If True, will save the RMS data. Default: True.
             """
    )

    # Seismic data
    PARSER.add_argument(
        "--provider-name",
        default="RESIF",
        type=str,
        help="""
             Name of the data provider. The names of all data providers supported by `obspy` are
             defined in `obspy.clients.fdsn.header.URL_MAPPINGS`. Default: "RESIF".
             """
    )

    PARSER.add_argument(
        "--network",
        default="FR",
        type=str,
        help="""
             Name of the network. Default: "FR".
             """
    )

    PARSER.add_argument(
        "--station",
        default="STR",
        type=str,
        help="""
             Name of the station. Default: "STR".
             """
    )

    PARSER.add_argument(
        "--location",
        default="00",
        type=str,
        help="""
             In some cases, a station might include multiple sensors. Location of the desired
             sensor. Default: "00".
             """
    )

    PARSER.add_argument(
        "--channel",
        default="BHZ",
        type=str,
        help="""
             Name of the channel. Names can differ between networks. A detailed description of
             channels is available at http://www.fdsn.org/pdf/SEEDManual_V2.4_Appendix-A.pdf.
             Default: "BHZ".
             """
    )

    # Dates
    PARSER.add_argument(
        "--start-date",
        default="2020-03-01",
        type=str,
        help="""
             Start date to query data in the format "%Y-%m-%d". Default: "2020-03-01".
             """
    )

    PARSER.add_argument(
        "--end-date",
        default="2022-01-01",
        type=str,
        help="""
             End date to query data in the format "%Y-%m-%d". Default: "2022-01-01".
             """
    )

    # PPSD
    PARSER.add_argument(
        "--db-bins",
        default=(-200, 20, 0.25),
        type=lambda x: tuple(float(spl) for spl in x.split("_")),
        help="""
             Specify the lower and upper boundary and the width of the db bins in the format
             lower_upper_width. Default: "-200_20_0.25".
             """
    )

    PARSER.add_argument(
        "--ppsd-length",
        default=1800,
        type=int,
        help="""
             Length of data segments passed to psd in seconds. Default: 1800.
             """
    )

    PARSER.add_argument(
        "--overlap",
        default=0.5,
        type=float_zero_one,
        help="""
             Overlap of segments passed to psd. Overlap may take values between 0 and 1.
             Default: 0.5.
             """
    )

    PARSER.add_argument(
        "--period-smoothing-width-octaves",
        default=0.025,
        type=float,
        help="""
             Determines over what period/frequency range the psd is smoothed around every central
             period/frequency. Default: 0.025.
             """
    )

    PARSER.add_argument(
        "--period-step-octaves",
        default=0.0125,
        type=float,
        help="""
             Step length on frequency axis in fraction of octaves. Default: 0.0125.
             """
    )

    PARSER.add_argument(
        "--period-limits",
        default=(0.008, 50),
        type=lambda x: tuple(float(spl) for spl in x.split("_")),
        help="""
             Set custom lower and upper end of period range in the format "lower_upper".
             Default: "0.008_50".
             """
    )

    # RMS
    PARSER.add_argument(
        "--freqs-pairs",
        default=[(0.1, 1.0), (1.0, 20.0), (4.0, 14.0), (4.0, 20.0)],
        nargs="*",
        type=lambda x: [(float(spl[0]), float(spl[1])) for spl in x.split(" ")],
        help="""
             All pairs of frequencies to compute RMS values in the format "low1-up1 low2-up2 ...".
             Default: "0.1-1.0 1.0-20.0 4.0-14.0 14.0-20.0".
             """
    )

    PARSER.add_argument(
        "--output",
        default="displacement",
        type=str,
        choices=["acceleration", "velocity", "displacement"],
        help="""
             Choose which RMS data to compute. Default: "displacement".
             """
    )

    # End of command lines
    ARGS = PARSER.parse_args()

    # Prepare power spectral density data
    PPSD_DFS = prepare_ppsd(
        data_dir=ARGS.data_dir,
        provider_name=ARGS.provider_name,
        network=ARGS.network,
        station=ARGS.station,
        location=ARGS.location,
        channel=ARGS.channel,
        start_date=ARGS.start_date,
        end_date=ARGS.end_date,
        db_bins=ARGS.db_bins,
        ppsd_length=ARGS.ppsd_length,
        overlap=ARGS.overlap,
        period_smoothing_width_octaves=ARGS.period_smoothing_width_octaves,
        period_step_octaves=ARGS.period_step_octaves,
        period_limits=ARGS.period_limits
    )

    # Prepare RMS data
    RMS_DFS = prepare_rms(
        ppsd_dfs=PPSD_DFS,
        data_dir=ARGS.data_dir,
        start_date=ARGS.start_date,
        end_date=ARGS.end_date,
        freqs_pairs=ARGS.freqs_pairs,
        output=ARGS.output,
        save=ARGS.save
    )
