# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from bokeh.plotting.figure import Figure
from matplotlib.axes import Axes

from systole.detection import ecg_peaks, ppg_peaks, rsp_peaks
from systole.plots.utils import get_plotting_function


def plot_raw(
    signal: Union[pd.DataFrame, np.ndarray, List],
    peaks: Optional[np.ndarray] = None,
    sfreq: int = 1000,
    modality: str = "ppg",
    ecg_method: str = "pan-tompkins",
    show_heart_rate: bool = False,
    show_artefacts: bool = False,
    bad_segments: Optional[Union[np.ndarray, List[int]]] = None,
    slider: bool = True,
    decim: Optional[int] = 10,
    ax: Optional[Axes] = None,
    figsize: Optional[Union[int, List[int], Tuple[int, int]]] = None,
    backend: str = "matplotlib",
    events_params: Optional[Dict] = None,
    **kwargs
) -> Union[Axes, Figure]:
    """Visualization of PPG or ECG signal with systolic peaks or R wave detection.

    The instantaneous heart rate can be derived in a second row, as well as the events
    temporal distribution.

    Parameters
    ----------
    signal : :py:class:`pandas.DataFrame` | :py:class:`numpy.ndarray` | list
        Dataframe of PPG or ECG signal in the long format. If a data frame is provided,
        it should contain at least one `'time'` and one colum for signal (either `'ppg'`
        or `'ecg'`). If an array is provided, it will automatically create a DataFrame
        using the array as signal and `sfreq` as sampling frequency.
    peaks : np.ndarray | None
        (Optional) A boolean vetor of peaks detection (should have same length than
        `signal`). If `peaks` is provided, the peaks detection part is skipped and this
        vector is used instead.
    sfreq : int
        Signal sampling frequency. Default is set to 1000 Hz.
    modality : str
        The type of signal provided. Can be `'ppg'` (pulse oximeter), `'ecg'`
        (electrocardiography) or `'resp'`. This parameter will control the type of
        peak detection algorithm to use. Only relevant if `peaks` is not provided.
    ecg_method : str
        Peak detection algorithm used by the
        :py:func:`systole.detection.ecg_peaks` function. Can be one of the following:
        `'hamilton'`, `'christov'`, `'engelse-zeelenberg'`, `'pan-tompkins'`,
        `'wavelet-transform'`, `'moving-average'`. The default is `'pan-tompkins'`.
    show_heart_rate : bool
        If `True`, show the instnataneous heart rate below the raw signal. Defaults to
        `False`.
    show_artefacts : bool
        If `True`, the function will call py:func:`systole.detection.rr_artefacts` to
        detect outliers intervalin the time serie and outline them using different
        colors.
    bad_segments : np.ndarray | list | None
        Mark some portion of the recording as bad. Grey areas are displayed on the top
        of the signal to help visualization (this is not correcting or transforming the
        post-processed signals). If a np.ndarray is provided, it should be a boolean
        of same length than `signal` where `False` indicates a bad segment. If a list
        is provided, it should be a list of tuples shuch as (start_idx, end_idx) for
        each bad segment.
    slider : bool
        If `True`, will add a slider to select the time window to plot (requires bokeh
        backend).
    decim : int
        Factor by which to subsample the raw signal. Selects every Nth sample (where N
        is the value passed to decim). Default set to `10` (considering that the imput
        signal has a sampling frequency of 1000 Hz) to save memory.
    ax : :class:`matplotlib.axes.Axes` | None
        Where to draw the plot. Default is *None* (create a new figure). Only
        applies when `backend="matplotlib"`.
    figsize : tuple, int or None
        Figure size. Default is `(13, 5)` for matplotlib backend, and the height is
        `300` when using bokeh backend.
    backend: str
        Select plotting backend {"matplotlib", "bokeh"}. Defaults to
        "matplotlib".
    events_params : dict | None
        (Optional) Additional parameters that will be passed to
        py:func:`systole.plots.plot_events` and plot the events timing in the backgound.
    **kwargs : keyword arguments
        Additional arguments will be passed to
        `:py:func:systole.detection.ppg_peaks()` or
        `:py:func:systole.detection.ecg_peaks()`, depending on the type
        of data.

    Returns
    -------
    plot : :class:`matplotlib.axes.Axes` | :class:`bokeh.plotting.figure.Figure`
        The matplotlib axes, or the bokeh figure containing the plot.

    See also
    --------
    plot_events, plot_rr

    Examples
    --------

    Plotting raw ECG recording.

    .. jupyter-execute::

       from systole import import_dataset1
       from systole.plots import plot_raw

       # Import PPG recording as pandas data frame
       ecg = import_dataset1(modalities=['ECG'])

       # Only use the first 60 seconds for demonstration
       ecg = ecg[ecg.time.between(60, 90)]
       plot_raw(ecg, modality='ecg', sfreq=1000, ecg_method='pan-tompkins')

    Plotting raw PPG recording.

    .. jupyter-execute::

       from systole import import_ppg

       # Import PPG recording as pandas data frame
       ppg = import_ppg()

       # Only use the first 60 seconds for demonstration
       plot_raw(ppg[ppg.time<60], sfreq=75)

    Highlighting a bad segment in the recording.

    .. jupyter-execute::

       from systole import import_ppg
       from systole.plots import plot_raw

       # Import PPG recording as pandas data frame
       ppg = import_ppg()

       # Only use the first 60 seconds for demonstration
       # The bad segments are annotated using a tuple (start, end) in miliseconds
       plot_raw(ppg[ppg.time<60], sfreq=75, bad_segments=[(15000, 17000)])

    Using Bokeh backend, with instantaneous heart rate and artefacts.

    .. jupyter-execute::

       from bokeh.io import output_notebook
       from bokeh.plotting import show
       output_notebook()

       show(
           plot_raw(ppg, backend="bokeh", show_heart_rate=True, show_artefacts=True)
        )

    """
    if figsize is None:
        if backend == "matplotlib":
            figsize = (13, 5)
        elif backend == "bokeh":
            figsize = 300

    if peaks is None:

        if isinstance(signal, pd.DataFrame):

            # Find peaks - Remove learning phase
            if modality == "ppg":
                signal, peaks = ppg_peaks(
                    signal=signal.ppg, moving_average=False, sfreq=sfreq, **kwargs
                )
            elif modality == "resp":
                signal, (peaks, troughs) = rsp_peaks(
                    signal=signal.resp, sfreq=sfreq, **kwargs
                )
            elif modality == "ecg":
                signal, peaks = ecg_peaks(
                    signal=signal.ecg,
                    method=ecg_method,
                    find_local=True,
                    sfreq=sfreq,
                    **kwargs
                )
            else:
                raise ValueError(
                    "Invalid modality parameter. Should be 'ecg', 'ppg' or 'resp'."
                )
        else:
            if modality == "ppg":
                signal, peaks = ppg_peaks(
                    signal=signal, moving_average=False, sfreq=sfreq, **kwargs
                )
            elif modality == "resp":
                signal, (peaks, troughs) = rsp_peaks(
                    signal=signal, sfreq=sfreq, **kwargs
                )
            elif modality == "ecg":
                signal, peaks = ecg_peaks(
                    signal=signal,
                    method=ecg_method,
                    sfreq=sfreq,
                    find_local=True,
                    **kwargs
                )
            else:
                raise ValueError(
                    "Invalid modality parameter. Should be 'ecg', 'ppg' or 'resp'."
                )

    if bad_segments is not None:
        if isinstance(bad_segments, np.ndarray):
            assert len(bad_segments) == len(signal)

            # Find the start and end of each bad segments
            bad_segments = [
                idx
                for idx in range(len(bad_segments))
                if (bad_segments[idx] == 1) & (bad_segments[idx - 1] == 0)
                | (bad_segments[idx] == 0) & (bad_segments[idx - 1] == 1)
                | (bad_segments[idx] == 0) & (idx == 0)
                | (bad_segments[idx] == 0) & (idx == len(bad_segments) - 1)
            ]

            # Make it a list of tuples (start, end)
            bad_segments = [
                (bad_segments[i], bad_segments[i + 1])
                for i in range(0, len(bad_segments), 2)
            ]

    time = pd.to_datetime(np.arange(0, len(signal)), unit="ms", origin="unix")

    plot_raw_args = {
        "time": time,
        "signal": signal,
        "peaks": peaks,
        "modality": modality,
        "show_heart_rate": show_heart_rate,
        "show_artefacts": show_artefacts,
        "bad_segments": bad_segments,
        "ax": ax,
        "figsize": figsize,
        "slider": slider,
        "decim": decim,
        "events_params": events_params,
    }

    plotting_function = get_plotting_function("plot_raw", "plot_raw", backend)
    plot = plotting_function(**plot_raw_args)

    return plot
