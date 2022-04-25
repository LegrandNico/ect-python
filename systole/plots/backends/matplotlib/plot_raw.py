# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from pandas.core.indexes.datetimes import DatetimeIndex

from systole.plots import plot_rr


def plot_raw(
    time: DatetimeIndex,
    signal: np.ndarray,
    peaks: np.ndarray,
    modality: str = "ppg",
    show_heart_rate: bool = True,
    show_artefacts: bool = False,
    decim: int = 10,
    ax: Optional[Union[List, Axes]] = None,
    slider: bool = True,
    figsize: int = 300,
    events_params: Optional[Dict] = None,
) -> Axes:
    """Visualization of PPG or ECG signal with systolic peaks/R wave detection.

    The instantaneous heart rate can be derived in a second row.

    Parameters
    ----------
    time : :py:class:`pandas.core.indexes.datetimes.DatetimeIndex`
        The time index.
    signal : :py:class:`numpy.ndarray`
        The physiological signal (1d numpy array).
    peaks : :py:class:`numpy.ndarray`
        The peaks or R wave detection (1d boolean array).
    modality : str
        The recording modality. Can be `"ppg"` or `"ecg"`.
    show_heart_rate : bool
        If `True`, create a second row and plot the instantanesou heart rate
        derived from the physiological signal
        (calls :py:func:`systole.plots.plot_rr` internally). Defaults to
        `False`.
    show_artefacts : bool
        If `True`, the function will call
        py:func:`systole.detection.rr_artefacts` to detect outliers intervalin the time
        serie and outline them using different colors.
    decim : int
        Factor by which to subsample the raw signal. Selects every Nth sample (where N
        is the value passed to decim). Default set to `10` (considering that the imput
        signal has a sampling frequency of 1000 Hz) to save memory.
    ax : :class:`matplotlib.axes.Axes` list or None
        Where to draw the plot. Default is *None* (create a new figure). Only
        applies when `backend="matplotlib"`. If `show_heart_rate is True`, a
        list of axes can be provided to plot the signal and instantaneous heart
        rate separately.
    slider : bool
        If `True`, add a slider to zoom in/out in the signal (only working with
        bokeh backend).
    figsize : int
        Figure heights. Default is `300`.
    events_params : dict | None
        (Optional) Additional parameters that will be passed to
        py:func:`systole.plots.plot_events` and plot the events timing in the backgound.

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes` | tuple
        The matplotlib axes containing the plot.

    """

    if modality == "ppg":
        title = "PPG recording"
        ylabel = "PPG level (a.u.)"
        peaks_label = "Systolic peaks"
    elif modality == "ecg":
        title = "ECG recording"
        ylabel = "ECG (mV)"
        peaks_label = "R wave"

    #############
    # Upper panel
    #############
    if ax is None:
        if show_heart_rate is True:
            _, axs = plt.subplots(ncols=1, nrows=2, figsize=figsize, sharex=True)
            signal_ax, hr_ax = axs
        else:
            _, signal_ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)

    elif isinstance(ax, list):
        signal_ax, hr_ax = ax
    else:
        signal_ax = ax

    # Signal
    signal_ax.plot(
        time[::decim], signal[::decim], label="PPG signal", linewidth=1, color="#c44e52"
    )

    # Peaks
    signal_ax.scatter(
        x=time[peaks],
        y=signal[peaks],
        marker="o",
        label=peaks_label,
        s=30,
        color="white",
        edgecolors="DarkSlateGrey",
    )
    if modality == "ppg":
        signal_ax.set_title(title)
        signal_ax.set_ylabel(ylabel)
    elif modality == "ecg":
        signal_ax.set_title(title)
        signal_ax.set_ylabel(ylabel)
    signal_ax.grid(True)

    #############
    # Lower panel
    #############

    if show_heart_rate is True:

        # Instantaneous Heart Rate - Peaks
        plot_rr(
            peaks,
            input_type="peaks",
            backend="matplotlib",
            figsize=figsize,
            show_artefacts=show_artefacts,
            ax=hr_ax,
            events_params=events_params,
        )

        plt.tight_layout()

        return signal_ax, hr_ax

    else:
        return signal_ax
