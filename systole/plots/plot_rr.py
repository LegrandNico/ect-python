# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib.axes import Axes

from systole.detection import rr_artefacts
from systole.plots.utils import get_plotting_function


def plot_rr(
    rr: Union[List, np.ndarray],
    unit: str = "rr",
    kind: str = "cubic",
    line: bool = True,
    points: bool = True,
    input_type: str = "peaks",
    show_artefacts: bool = False,
    ax: Optional[Axes] = None,
    figsize: Optional[Union[Tuple[float, float], int]] = None,
    backend: str = "matplotlib",
) -> Axes:
    """Plot continuous and/or discontinuous RR intervals time series.

    Parameters
    ----------
    rr : np.ndarray or list
        Boolean vector of peaks detection, peaks indexs or RR intervals.
    unit : str
        The heart rate unit in use. Can be `'rr'` (R-R intervals, in ms)
        or `'bpm'` (beats per minutes). Default is `'rr'`.
    kind : str
        The method to use (parameter of `scipy.interpolate.interp1d`). The
        possible relevant methods for instantaneous heart rate are `'cubic'`
        (defalut), `'linear'`, `'previous'` and `'next'`.
    line : bool
        If `True`, plot the interpolated instantaneous heart rate.
    points : bool
        If `True`, plot each peaks (R wave or systolic peaks) as separated
        points
    input_type : str
        The type of input vector. Default is `"peaks"` (a boolean vector where
        `1` represents the occurrence of R waves or systolic peaks).
        Can also be `"peaks_idx"`, the idexs of samples where a peaks is detected,
        `"rr_s"` or `"rr_ms"` for vectors of RR intervals, or interbeat intervals
        (IBI), expressed in seconds or milliseconds (respectively).
    show_artefacts : bool
        If `True`, the function will call
        py:func:`systole.detection.rr_artefacts` to detect outliers interval
        in the time serie and outline them using different colors.
    ax : :class:`matplotlib.axes.Axes` or None
        Where to draw the plot. Default is *None* (create a new figure).
    figsize : tuple, int or None
        Figure size. Default is `(13, 5)`.
    backend: str
        Select plotting backend {"matplotlib", "bokeh"}. Defaults to
        "matplotlib".

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        The matplotlib axes containing the plot.

    See also
    --------
    plot_events, plot_subspaces, plot_events, plot_psd, plot_oximeter, plot_raw

    Examples
    --------

    .. plot::

        >>> from systole import import_rr
        >>> from systole.plots import plot_rr
        >>> rr = import_rr().rr.values
        >>> plot_rr(rr=rr, input_type="rr_ms", unit="bpm",)
    """

    if (points is False) & (line is False):
        raise ValueError("Either points or line should be True")

    if figsize is None:
        if backend == "matplotlib":
            figsize = (13, 5)
        elif backend == "bokeh":
            figsize = 200

    if input_type not in ["peaks", "rr_ms", "rr_s", "peaks_idx"]:
        raise ValueError("Invalid input type")

    # Detect artefacts in the rr time series if required
    artefacts: Optional[Dict[str, np.ndarray]] = None
    if show_artefacts is True:
        if points is False:
            raise Warning("show_artefacts is True but points is set to False")
        artefacts = rr_artefacts(rr, input_type=input_type)

    plot_rr_args = {
        "rr": rr,
        "unit": unit,
        "kind": kind,
        "line": line,
        "points": points,
        "artefacts": artefacts,
        "input_type": input_type,
        "ax": ax,
        "figsize": figsize,
    }

    plotting_function = get_plotting_function("plot_rr", "plot_rr", backend)
    plot = plotting_function(**plot_rr_args)

    return plot
