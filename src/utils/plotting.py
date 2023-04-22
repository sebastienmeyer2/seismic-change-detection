"""Plotting utils."""


import os

from typing import List, Optional, Union

from itertools import cycle, tee

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


plt.rcParams.update({
    "text.usetex": True,
    "font.size": 12
})


def save_or_plot(
    fig: Figure,
    savefig: bool = True,
    results_dir: str = "results",
    filename: str = "out.png"
):
    """Save or plot given figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to plot or save.

    savefig : bool, default=True
        If True, figure will be saved on disk.

    results_dir : str, default="results"
        If **savefig** is True, name of the directory where results are stored.

    filename : str, default="out.png"
        If **savefig** is True, filename to save the figure.
    """
    if savefig:

        results_dir_split = results_dir.split("/")

        results_dir = ""

        for spl in results_dir_split:

            if len(results_dir) == 0:
                results_dir = spl
            else:
                results_dir = f"{results_dir}/{spl}"

            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

        # Create the results directory where data will be stored
        fig.savefig(f"{results_dir}/{filename}", facecolor="white")


def plot_ppsd(
    ppsd_df: Optional[pd.DataFrame] = None,
    start_date_plot: str = "2020-03-13",
    end_date_plot: str = "2020-03-23",
    min_freq_plot: str = 4.0,
    max_freq_plot: str = 14.0,
    savefig: bool = True,
    results_dir: str = "results"
):
    """Plot PPSD data.

    Parameters
    ----------
    ppsd_df : optional pd.DataFrame, default=None
        Dataframe containing power spectral density values.

    start_date_plot : str, default="2020-03-13"
        Start date of the plot.

    end_date_signal : str, default="2020-03-23"
        End date of the plot.

    min_freq_signal : float, default=4.0
        Minimal frequency of the plot.

    max_freq_signal : float, default=14.0
        Maximal frequency of the plot.

    savefig : bool, default=True
        If True, figure will be saved on disk.

    results_dir : str, default="results"
        If **savefig** is True, name of the directory where results are stored.
    """
    if ppsd_df is None:
        return

    freqs = ppsd_df.columns.to_numpy()
    bin_mask = (freqs >= min_freq_plot) & (freqs <= max_freq_plot)
    freqs_in_bin = freqs[bin_mask]

    ppsd_df_plot = ppsd_df[start_date_plot:end_date_plot][freqs_in_bin]

    # Plot PPSD in color map
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    datelist_plot = ppsd_df_plot.index
    freqs_plot = ppsd_df_plot.columns
    psd_plot = ppsd_df_plot.values.T

    color_map = ax.pcolor(datelist_plot, freqs_plot, psd_plot)

    # Legend for x axis
    nb_ticks = 7  # do not show too much labels
    nb_dates = len(np.unique(datelist_plot.strftime("%Y-%m-%d 00:00")))
    interval = nb_dates // nb_ticks
    new_xticklabels = np.unique(datelist_plot.strftime("%Y-%m-%d 00:00")).tolist()[::interval]
    new_xticks = [ax.get_xticks()[0] + i * interval for i in range(len(new_xticklabels))]
    ax.set_xticks(
        new_xticks, labels=new_xticklabels, rotation=45, rotation_mode="anchor", ha="right"
    )

    # Legend for y axis
    ax.set_yscale("log")
    ax.set_ylabel(r"Frequency ($Hz$)")

    # Color bar
    fig.colorbar(color_map, shrink=0.8, label=r"Amplitude ($m^2.s^{-4}.Hz$) ($dB$)")

    save_or_plot(fig, savefig=savefig, results_dir=results_dir, filename="psd.png")


def plot_rms(
    rms_df: Optional[pd.DataFrame] = None,
    start_date_plot: str = "2020-01-01",
    end_date_plot: str = "2020-01-03",
    bands_plot: Union[str, List[str]] = "[4.00, 14.00] Hz",
    log_scale: bool = True,
    savefig: bool = True,
    results_dir: str = "results"
):
    """Plot RMS data.

    Parameters
    ----------
    rms_df : optional pd.DataFrame, default=None
        Dataframe containing RMS values.

    start_date_plot : str, default="2020-03-13"
        Start date of the plot.

    end_date_signal : str, default="2020-03-23"
        End date of the plot.

    bands_plot : str or list of str, default="[4.00, 14.00] Hz"
        Name(s) of the band to plot under the convention "[{min:.2f}, {max:.2f}] Hz".

    log_scale : bool, default=True
        If True, will use log scale for y axis.

    savefig : bool, default=True
        If True, figure will be saved on disk.

    results_dir : str, default="results"
        If **savefig** is True, name of the directory where results are stored.
    """
    if rms_df is None:
        return

    if isinstance(bands_plot, str):
        bands_plot = [bands_plot]

    bands = rms_df.columns.to_numpy()
    bands_mask = list(set(bands_plot).intersection(set(bands)))

    rms_df_plot = rms_df[start_date_plot:end_date_plot][bands_mask]

    # Plot RMS bands
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    datelist_plot = rms_df_plot.index

    for band in bands_mask:

        rms_plot = rms_df_plot[band]

        label = band.replace(" ", r" \ ")
        ax.plot(datelist_plot, rms_plot, label=fr"${label}$")

    # Legend for x axis
    nb_ticks = 7  # do not show too much labels
    nb_dates = len(np.unique(datelist_plot.strftime("%Y-%m-%d 00:00")))
    interval = nb_dates // nb_ticks
    new_xticklabels = np.unique(datelist_plot.strftime("%Y-%m-%d 00:00")).tolist()[::interval]
    new_xticks = [ax.get_xticks()[0] + i * interval for i in range(len(new_xticklabels))]
    ax.set_xticks(
        new_xticks, labels=new_xticklabels, rotation=45, rotation_mode="anchor", ha="right"
    )

    # Legend for y axis
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel(r"Displacement ($nm$)")

    # Legend for labels
    ax.legend(loc="upper right")

    save_or_plot(fig, savefig=savefig, results_dir=results_dir, filename="rms.png")


def plot_bkps(
    signal: Optional[np.ndarray] = None,
    true_bkps: Optional[List[int]] = None,
    pred_bkps: Optional[List[int]] = None,
    datelist: Optional[pd.DatetimeIndex] = None,
    savefig: bool = True,
    results_dir: str = "results"
):
    """Plot true and predicted breakpoints over signal.

    Parameters
    ----------
    signal : optional np.ndarray, default=None
        Extracted signal for change point detection.

    true_bkps : optional list of int, default=None
        List of true indices for the breakpoints in the signal.

    pred_bkps : optional list of int, default=None
        List of predicted indices for the breakpoints in the signal.

    datelist : optional pd.DatetimeIndex, default=None
        List of datetimes of the signal.

    savefig : bool, default=True
        If True, figure will be saved on disk.

    results_dir : str, default="results"
        If **savefig** is True, name of the directory where results are stored.
    """
    if signal is None:
        return

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    n_samples = len(signal)

    # Plot the signal first
    ax.plot(range(n_samples), signal)

    # Color each (true) regime
    if true_bkps is not None:

        color_cycle = cycle(["#4286f4", "#f44174"])

        bkps = [0] + sorted(true_bkps)

        a, b = tee(bkps)
        next(b, None)
        pairwise_bkps = zip(a, b)

        for (start, end), color in zip(pairwise_bkps, color_cycle):
            ax.axvspan(max(0, start - 0.5), end - 0.5, facecolor=color, alpha=0.2)

    # Vertical lines to mark the predicted breakpoints
    if pred_bkps is not None:

        for bkp in pred_bkps:

            if bkp != 0 and bkp < n_samples:

                ax.axvline(
                    x=bkp - 0.5,
                    color="k",
                    linewidth=3,
                    linestyle="--",
                    alpha=1.0,
                )

    # Legend for x axis
    if datelist is not None:

        nb_ticks = 7  # do not show too much labels
        nb_dates = len(np.unique(datelist.strftime("%Y-%m-%d %H:%M")))
        interval = nb_dates // nb_ticks
        new_xticklabels = np.unique(datelist.strftime("%Y-%m-%d %H:%M")).tolist()[::interval]
        new_xticks = [i for i in range(len(datelist))][::interval]
        ax.set_xticks(
            new_xticks, labels=new_xticklabels, rotation=45, rotation_mode="anchor", ha="right"
        )

    # Legend for y axis
    if len(signal.shape) <= 1 or signal.shape[1] == 1:
        ax.set_ylabel(r"Displacement ($nm$)")
    else:
        ax.set_ylabel(r"Amplitude ($m^2.s^{-4}.Hz$) ($dB$)")

    save_or_plot(fig, savefig=savefig, results_dir=results_dir, filename="bkps.png")


def plot_elbow(
    array_of_n_bkps: np.ndarray,
    array_of_costs: np.ndarray,
    savefig: bool = True,
    results_dir: str = "results"
):
    """Plot true and predicted breakpoints over signal.

    Adapted from https://centre-borelli.github.io/ruptures-docs/examples/music-segmentation/.

    Parameters
    ----------
    array_of_n_bkps : np.ndarray
        List of number of breakpoints from 1 to **n_bkps_max**.

    array_of_costs : np.ndarray
        List of corresponding sum of costs when predicting using `Dynp` algorithm.

    savefig : bool, default=True
        If True, figure will be saved on disk.

    results_dir : str, default="results"
        If **savefig** is True, name of the directory where results are stored.
    """
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    # Plot the sum of costs
    ax.plot(array_of_n_bkps, array_of_costs, "-*", alpha=0.5)

    # Legend for x axis
    ax.set_xticks(array_of_n_bkps)
    ax.set_xlabel("Number of change points")
    ax.grid(axis="x")

    # Legend for y axis
    ax.set_ylabel("Sum of costs")
    ax.set_xlim(0, array_of_n_bkps[-1] + 1)

    save_or_plot(fig, savefig=savefig, results_dir=results_dir, filename="elbow.png")


def plot_dp(
    true_bkps: List[int],
    list_of_bkps: List[List[int]],
    datelist: Optional[pd.DatetimeIndex] = None,
    savefig: bool = True,
    results_dir: str = "results"
):
    """Plot variable number of predicted breakpoints wrt Dynp output.

    Parameters
    ----------
    true_bkps : optional list of int, default=None
        List of true indices for the breakpoints in the signal.

    list_of_bkps : list of list of int
        List of corresponding breakpoints indices.

    datelist : optional pd.DatetimeIndex, default=None
        List of datetimes of the signal.

    savefig : bool, default=True
        If True, figure will be saved on disk.

    results_dir : str, default="results"
        If **savefig** is True, name of the directory where results are stored.
    """
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    # Plot the true breakpoints
    ax.axvline(true_bkps[0], color="k", linestyle="dashed", label="Theoretical breakpoints")

    for j in range(len(true_bkps)-2):

        ax.axvline(true_bkps[j+1], color="k", linestyle="dashed")

    # Plot the predicted breakpoints
    for i, pred_bkps in enumerate(list_of_bkps):

        ax.scatter(pred_bkps[:-1], (i+1)*[i+1], marker="X")

    # Legend for x axis
    if datelist is not None:

        nb_ticks = 7  # do not show too much labels
        nb_dates = len(np.unique(datelist.strftime("%Y-%m-%d %H:%M")))
        interval = nb_dates // nb_ticks
        new_xticklabels = np.unique(datelist.strftime("%Y-%m-%d %H:%M")).tolist()[::interval]
        new_xticks = [i for i in range(len(datelist))][::interval]
        ax.set_xticks(
            new_xticks, labels=new_xticklabels, rotation=45, rotation_mode="anchor", ha="right"
        )

    # Legend for y axis
    ax.set_ylabel("Number of breakpoints")
    ax.grid(axis="y")
    ax.set_yticks(np.arange(1, len(list_of_bkps)+1), np.arange(1, len(list_of_bkps)+1, dtype=int))

    # Legend
    ax.legend(loc="best")

    save_or_plot(fig, savefig=savefig, results_dir=results_dir, filename="dp.png")


def plot_p(
    true_bkps: List[int],
    list_of_bkps: List[List[int]],
    datelist: Optional[pd.DatetimeIndex] = None,
    array_of_pens: Optional[np.ndarray] = None,
    savefig: bool = True,
    results_dir: str = "results"
):
    """Plot variable number of predicted breakpoints wrt penalty for Pelt.

    Parameters
    ----------
    true_bkps : optional list of int, default=None
        List of true indices for the breakpoints in the signal.

    list_of_bkps : list of list of int
        List of corresponding breakpoints indices.

    datelist : optional pd.DatetimeIndex, default=None
        List of datetimes of the signal.

    array_of_pens : np.ndarray
        Array of penalities used to compute **list_of_bkps**.

    savefig : bool, default=True
        If True, figure will be saved on disk.

    results_dir : str, default="results"
        If **savefig** is True, name of the directory where results are stored.
    """
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    # Plot the true breakpoints
    ax.axvline(true_bkps[0], color="k", linestyle="dashed", label="Theoretical breakpoints")

    for j in range(len(true_bkps)-2):

        ax.axvline(true_bkps[j+1], color="k", linestyle="dashed")

    # Plot the predicted breakpoints
    for i, pred_bkps in enumerate(list_of_bkps):

        ax.scatter(pred_bkps[:-1], (len(pred_bkps)-1)*[i+1], marker="X")

    # Legend for x axis
    if datelist is not None:

        nb_ticks = 7  # do not show too much labels
        nb_dates = len(np.unique(datelist.strftime("%Y-%m-%d %H:%M")))
        interval = nb_dates // nb_ticks
        new_xticklabels = np.unique(datelist.strftime("%Y-%m-%d %H:%M")).tolist()[::interval]
        new_xticks = [i for i in range(len(datelist))][::interval]
        ax.set_xticks(
            new_xticks, labels=new_xticklabels, rotation=45, rotation_mode="anchor", ha="right"
        )

    # Legend for y axis
    if array_of_pens is not None:

        new_yticklabels = [np.format_float_scientific(pen, precision=2) for pen in array_of_pens]
        new_yticks = np.arange(1, len(list_of_bkps)+1)
        ax.set_yticks(new_yticks, labels=new_yticklabels)

    ax.set_ylabel("Penality")
    ax.grid(axis="y")

    # Legend
    ax.legend(loc="best")

    save_or_plot(fig, savefig=savefig, results_dir=results_dir, filename="p.png")


def plot_nb(
    true_bkps: List[int],
    lists_of_bkps: List[List[List[int]]],
    list_of_labels: List[str],
    array_of_pens: Optional[np.ndarray] = None,
    savefig: bool = True,
    results_dir: str = "results"
):
    """Plot number of predicted breakpoints.

    Parameters
    ----------
    true_bkps : optional list of int, default=None
        List of true indices for the breakpoints in the signal.

    lists_of_bkps : list of list of list of int
        List of corresponding breakpoints indices.

    list_of_labels : list of str
        List of the labels for the different signals.

    array_of_pens : np.ndarray
        Array of penalities used to compute **list_of_bkps**.

    savefig : bool, default=True
        If True, figure will be saved on disk.

    results_dir : str, default="results"
        If **savefig** is True, name of the directory where results are stored.
    """
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    # Plot the number of predicted breakpoints
    for label, list_of_bkps in zip(list_of_labels, lists_of_bkps):

        nb_bkps = [len(bkps) for bkps in list_of_bkps]
        plt.plot(array_of_pens, nb_bkps, label=label)

    # Plot the theoretical number of breakpoints
    plt.plot(
        array_of_pens, len(array_of_pens)*[len(true_bkps)], label="Theoretical amount",
        linestyle="dashed", color="k"
    )

    # Legend
    ax.legend(loc="best")
    ax.set_ylabel("Number of breakpoints")
    ax.set_xlabel("Penality")

    save_or_plot(fig, savefig=savefig, results_dir=results_dir, filename="nb.png")
