# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from sleepecg import detect_heartbeats

from systole.detectors import (
    christov,
    engelse_zeelenberg,
    hamilton,
    moving_average,
    pan_tompkins,
)
from systole.utils import find_clipping, input_conversion, nan_cleaning, to_neighbour


def ppg_peaks(
    signal: Union[List, np.ndarray, pd.Series],
    sfreq: int,
    win: float = 0.75,
    new_sfreq: int = 1000,
    clipping: bool = True,
    clipping_thresholds: Union[Tuple, List, str] = "auto",
    moving_average: bool = True,
    moving_average_length: float = 0.05,
    peak_enhancement: bool = True,
    distance: float = 0.3,
    clean_extra: bool = False,
    clean_nan: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """A simple systolic peak finder for PPG signals.

    This method uses a rolling average + standard deviation approach to update a
    detection threshold. All the peaks found above this threshold are potential
    systolic peaks.

    Parameters
    ----------
    signal : np.ndarray | list | pd.Series
        The raw signal recorded from the pulse oximeter time series.
    sfreq : int
        The sampling frequency (Hz).
    win : int
        Window size (in seconds) used to compute the threshold (i.e. rolling mean +
        standard deviation).
    new_sfreq : int
        If resample is `True`, the new sampling frequency (Hz). Defaults to `1000`.
    clipping : boolean
        If `True`, will apply the clipping artefact correction described in [1]_.
        Defaults to `True`.
    clipping_thresholds : tuple | list | str
        The values of the minumum and maximum clipping thresholds. Can be a float or
        `None`. If `None`, no correction is applied. If "auto" is provided, will use
        py:func:`systole.utils.find_clipping` to find the values. Defaults to `"auto"`.
        This parameter is only relevant if `cliping` is `True`.
    moving_average : bool
        Apply mooving average to remove high frequency noise before peaks detection. The
        length of the time windows can be controlled with `moving_average_length`.
    moving_average_length : float
        The length of the window used for moveing average (seconds). Default to `0.05`.
    resample : boolean
        If `True` (default), will resample the signal at *new_sfreq*. Default
        value is 1000 Hz.
    peak_enhancement : boolean
        If `True` (default), the ppg signal is squared before peaks detection.
    distance : float
        The minimum interval between two peaks (seconds).
    clean_extra : bool
        If `True`, use `:py:func:systole.detection.rr_artefacts()` to find and
        remove extra peaks. Default is `False`.
    clean_nan : bool
        If `True`, will interpolate NaNs values if any before any other operation.
        Defaults to `False`.
    verbose : bool
        Control function verbosity. Defaults to `False` (do not print processing steps).

    Returns
    -------
    resampled_signal : np.ndarray
        Signal resampled to the `new_sfreq` frequency.
    peaks : np.ndarray
        Boolean array of systolic peaks detection.

    Raises
    ------
    ValueError
        If `clipping_thresholds` is not a tuple, a list or `"auto"`.

    Notes
    -----
    This algorithm use a simple rolling average to detect peaks. The signal is
    first resampled and a rolling average is applyed to correct high frequency
    noise and clipping, using method detailled in [1]_. The signal is then
    squared and detection of peaks is performed using threshold corresponding
    to the moving averagte + stadard deviation.

    .. warning :: This function will resample the signal to 1000 Hz by default.

    Examples
    --------
    >>> from systole import import_ppg
    >>> from systole.detection import ppg_peaks
    >>> df = import_ppg()  # Import PPG recording
    >>> signal, peaks = ppg_peaks(signal=df.ppg.to_numpy())
    >>> print(f'{sum(peaks)} peaks detected.')
    378 peaks detected.

    References
    ----------
    .. [1] van Gent, P., Farah, H., van Nes, N. and van Arem, B., 2019.
       Analysing Noisy Driver Physiology Real-Time Using Off-the-Shelf Sensors:
       Heart Rate Analysis Software from the Taking the Fast Lane Project. Journal
       of Open Research Software, 7(1), p.32. DOI: http://doi.org/10.5334/jors.241

    """

    x = np.asarray(signal)

    # Interpolate NaNs values if any and if requested
    if clean_nan is True:
        if np.isnan(x).any():
            x = nan_cleaning(signal=x, verbose=verbose)

    # Resample signal to the new frequnecy if required
    if sfreq != new_sfreq:
        time = np.arange(0, len(x) / sfreq, 1 / sfreq)
        new_time = np.arange(0, len(x) / sfreq, 1 / new_sfreq)
        x = np.interp(new_time, time, x)

    # Copy resampled signal for output
    resampled_signal = np.copy(x)

    # Remove clipping artefacts with cubic interpolation
    if clipping is True:
        if clipping_thresholds == "auto":
            min_threshold, max_threshold = find_clipping(signal=x)
        elif isinstance(clipping_thresholds, list) | isinstance(
            clipping_thresholds, tuple
        ):
            min_threshold, max_threshold = clipping_thresholds  # type: ignore
        else:
            raise ValueError(
                (
                    "The variable clipping_thresholds should be a list"
                    "or a tuple with length 2 or 'auto'."
                )
            )
        x = interpolate_clipping(
            signal=x, min_threshold=min_threshold, max_threshold=max_threshold
        )

    if moving_average is True:
        # Moving average (high frequency noise)
        rollingNoise = max(int(new_sfreq * moving_average_length), 1)  # 0.05 second
        x = (
            pd.DataFrame({"signal": x})
            .rolling(rollingNoise, center=True)
            .mean()
            .signal.to_numpy()
        )
    if peak_enhancement is True:
        # Square signal (peak enhancement)
        x = (np.asarray(x) ** 2) * np.sign(x)

    # Compute moving average and standard deviation
    signal = pd.DataFrame({"signal": x})
    mean_signal = (
        signal.rolling(int(new_sfreq * win), center=True).mean().signal.to_numpy()
    )
    std_signal = (
        signal.rolling(int(new_sfreq * win), center=True).std().signal.to_numpy()
    )

    # Substract moving average + standard deviation
    x -= mean_signal + std_signal

    # Find positive peaks
    peaks_idx = find_peaks(x, height=0, distance=int(new_sfreq * distance))[0]

    # Create boolean vector
    peaks = np.zeros(len(x), dtype=bool)
    peaks[peaks_idx] = 1

    # Remove extra peaks
    if clean_extra:

        # Search artefacts
        rr = np.diff(np.where(peaks)[0])  # Convert to RR time series
        artefacts = rr_artefacts(rr)

        # Clean peak vector
        peaks[peaks_idx[1:][artefacts["extra"]]] = 0

    return resampled_signal, peaks


def ecg_peaks(
    signal: Union[List, np.ndarray, pd.Series],
    sfreq: int = 1000,
    new_sfreq: int = 1000,
    method: str = "sleepecg",
    find_local: bool = False,
    win_size: float = 0.1,
    clean_nan: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """A simple wrapper for many popular R peaks detectors algorithms.

    This function calls methods from `py-ecg-detectors` [1]_.

    Parameters
    ----------
    signal : np.ndarray | list | pd.Series
        The raw ECG signal.
    sfreq : int
        The sampling frequency. Default is set to `1000` Hz.
    new_sfreq : int
        The new sampling frequency. Defaults to `1000` Hz.
    method : str
        The method used. Can be one of the following: `'sleepecg'`, `'hamilton'`,
        `'christov'`, `'engelse-zeelenberg'`, `'pan-tompkins'`, `'moving-average'`.
    find_local : bool
        If *True*, will use peaks indexs to search for local peaks given the
        window size (win_size).
    win_size : int
        Size of the time window used by :py:func:`systole.utils.to_neighbour()`
        expressed in seconds. Defaut set to `0.1`.
    clean_nan : bool
        If `True`, will interpolate NaNs values if any before any other operation.
        Defaults to `False`.
    verbose : bool
        Control function verbosity. Defaults to `False` (do not print processing steps).

    Returns
    -------
    resampled_signal : np.ndarray
        Signal resampled to the `new_sfreq` frequency.
    peaks : np.ndarray
        Boolean array corresponding to the R peaks detection.

    Raises
    ------
    ValueError
        If `method` is not one of the following: `'hamilton'`, `'christov'`,
            `'engelse-zeelenberg'`, `'pan-tompkins'`, `'moving-average'`

    Notes
    -----
    .. warning :: This function will resample the signal to 1000 Hz.

    Examples
    --------
    >>> from systole import import_dataset
    >>> from systole.detection import ecg_peaks
    >>> signal_df = import_dataset()[:20*2000]
    >>> signal, peaks = ecg_peaks(signal_df.ecg.to_numpy(), method='hamilton',
    >>>                           sfreq=2000, find_local=True)
    >>> print(f'{sum(peaks)} peaks detected.')
    24 peaks detected.

    References
    ----------
    .. [1] Howell, L., Porr, B. Popular ECG R peak detectors written in
       python. DOI: 10.5281/zenodo.3353396

    """

    x = np.asarray(signal)

    # Interpolate NaNs values if any and if requested
    if clean_nan is True:
        if np.isnan(x).any():
            x = nan_cleaning(signal=x, verbose=verbose)

    # Resample signal to the new frequnecy if required
    if sfreq != new_sfreq:
        time = np.arange(0, len(x) / sfreq, 1 / sfreq)
        new_time = np.arange(0, len(x) / sfreq, 1 / new_sfreq)
        x = np.interp(new_time, time, x)

    # Copy resampled signal for output
    resampled_signal = np.copy(x)

    if method == "sleepecg":
        peaks_idx = detect_heartbeats(resampled_signal, fs=new_sfreq)
    elif method == "hamilton":
        peaks_idx = hamilton(resampled_signal, sfreq=new_sfreq)
    elif method == "christov":
        peaks_idx = christov(resampled_signal, sfreq=new_sfreq)
    elif method == "engelse-zeelenberg":
        peaks_idx = engelse_zeelenberg(resampled_signal, sfreq=new_sfreq)
    elif method == "pan-tompkins":
        peaks_idx = pan_tompkins(resampled_signal, sfreq=new_sfreq)
    elif method == "moving-average":
        peaks_idx = moving_average(resampled_signal, sfreq=new_sfreq)
    else:
        raise ValueError(
            "Invalid method provided, should be: sleepecg, hamilton, "
            "christov, engelse-zeelenberg, pan-tompkins, wavelet-transform, "
            "moving-average"
        )
    peaks = np.zeros(len(resampled_signal), dtype=bool)
    peaks[peaks_idx] = True

    if find_local is True:
        peaks = to_neighbour(resampled_signal, peaks, size=int(win_size * new_sfreq))

    return resampled_signal, peaks


def rsp_peaks(
    signal: Union[List, np.ndarray, pd.Series],
    sfreq: int,
    new_sfreq: int = 1000,
    win: float = 0.025,
    kind: str = "peaks-troughs",
    clean_nan: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """Identify peaks and/or troughs in respiratory signal.

    Parameters
    ----------
    signal : np.ndarray | list | pd.Series
        The respiratory signal. Peaks are considered to represent end of inspiration,
        trough represent end of expiration.
    sfreq : int
        The sampling frequency.
    new_sfreq : int
        If resample is `True`, the new sampling frequency. Defaults to `1000` Hz.
    win : int
        Window size (in seconds). Default is set to 25ms, following recommandation
        from [1]_.
    kind : str
        What kind of detection to perform. Peak detection (`"peaks"`), trough detection
        (`"troughs"`) or both (`"peaks-troughs"`, default).
    clean_nan : bool
        If `True`, will interpolate NaNs values if any before any other operation.
        Defaults to `False`.
    verbose : bool
        Control function verbosity. Defaults to `False` (do not print processing steps).

    Returns
    -------
    resampled_signal : np.ndarray
        Signal resampled to the `new_sfreq` frequency.
    peaks | trough | (peaks, trough) : np.ndarray | np.ndarray | (np.ndarray, np.ndarray)
        Boolean arrays of peaks and / or troughs in the respiratory signal.

    Raises
    ------
    ValueError
        If `kind` is not one of the following: `"peaks"`, `"troughs"` or
        `"peaks-troughs"`.

    Examples
    --------

    Notes
    -----
    The processing steps are largely inspired by the method described in [1]_.

    References
    ----------
    .. [1] Torben Noto, Guangyu Zhou, Stephan Schuele, Jessica Templer, Christina
       Zelano,Automated analysis of breathing waveforms using BreathMetrics: a
       respiratory signal processing toolbox, Chemical Senses, Volume 43, Issue 8,
       October 2018, Pages 583-597, https://doi.org/10.1093/chemse/bjy045

    """
    if kind not in ["peaks", "troughs", "peaks-troughs"]:
        raise ValueError(
            "Invalid kind parameter. Should be 'peaks', 'troughs' or 'peaks-troughs'"
        )

    x = np.asarray(signal)

    # Interpolate NaNs values if any and if requested
    if clean_nan is True:
        if np.isnan(x).any():
            x = nan_cleaning(signal=x, verbose=verbose)

    # Resample signal to the new frequnecy if required
    if sfreq != new_sfreq:
        time = np.arange(0, len(x) / sfreq, 1 / sfreq)
        new_time = np.arange(0, len(x) / sfreq, 1 / new_sfreq)
        x = np.interp(new_time, time, x)

    # Copy resampled signal for output
    resampled_signal = np.copy(x)

    # Soothing using rolling mean
    x = (
        pd.DataFrame({"signal": x})
        .rolling(int(sfreq * win), center=True)
        .mean()
        .fillna(method="bfill")
        .fillna(method="ffill")
        .signal.to_numpy()
    )

    # Normalize (z-score) the respiration signal
    x = (x - x.mean()) / x.std()  # type: ignore

    # Peak enhancement
    x = x ** 3

    # Find peaks and trough in preprocessed signal
    if "peaks" in kind:
        peaks_idx = find_peaks(x, height=0, distance=int(2 * sfreq))[0]
        peaks = np.zeros(len(resampled_signal), dtype=bool)
        peaks[peaks_idx] = True
    if "troughs" in kind:
        troughs_idx = find_peaks(-x, height=0, distance=int(2 * sfreq))[0]
        troughs = np.zeros(len(resampled_signal), dtype=bool)
        troughs[troughs_idx] = True

    if kind == "peaks":
        return resampled_signal, peaks
    elif kind == "trough":
        return resampled_signal, troughs
    else:
        return resampled_signal, (peaks, troughs)


def rr_artefacts(
    rr: Union[List, np.ndarray],
    c1: float = 0.13,
    c2: float = 0.17,
    alpha: float = 5.2,
    input_type: str = "rr_ms",
) -> Dict[str, np.ndarray]:
    """Artefacts detection from RR time series using the subspaces approach
    proposed by Lipponen & Tarvainen (2019).

    Parameters
    ----------
    rr : np.ndarray or list
        1d numpy array of RR intervals (in seconds or miliseconds) or peaks
        vector (boolean array).
    c1 : float
        Fixed variable controling the slope of the threshold lines. Default is
        `0.13`.
    c2 : float
        Fixed variable controling the intersect of the threshold lines. Default
        is `0.17`.
    alpha : float
        Scaling factor used to normalize the RR intervals first deviation.
    input_type : str
        The type of input vector. Defaults to `"rr_ms"` for vectors of RR
        intervals, or  interbeat intervals (IBI), expressed in milliseconds.
        Can also be a boolean vector where `1` represents the occurrence of
        R waves or systolic peakspeaks vector `"rr_s"` or IBI expressed in
        seconds.

    Returns
    -------
    artefacts : dict
        Dictionary storing the parameters of RR artefacts rejection. All the vectors
        outputed have the same length as the provided RR time serie:

        * subspace1 : np.ndarray
            The first dimension. First derivative of R-R interval time serie.
        * subspace2 : np.ndarray
            The second dimension (1st plot).
        * subspace3 : np.ndarray
            The third dimension (2nd plot).
        * mRR : np.ndarray
            The mRR time serie.
        * ectopic : np.ndarray
            Boolean array indexing probable ectopic beats.
        * long : np.ndarray
            Boolean array indexing long RR intervals.
        * short : np.ndarray
            Boolean array indexing short RR intervals.
        * missed : np.ndarray
            Boolean array indexing missed RR intervals.
        * extra : np.ndarray
            Boolean array indexing extra RR intervals.
        * threshold1 : np.ndarray
            Threshold 1.
        * threshold2 : np.ndarray
            Threshold 2.

    Notes
    -----
    This function will use the method proposed by [1]_ to detect ectopic beats, long,
    shorts, missed and extra RR intervals.

    Examples
    --------
    >>> from systole import simulate_rr
    >>> from systole.detection import rr_artefacts
    >>> rr = simulate_rr()  # Simulate RR time series
    >>> artefacts = rr_artefacts(rr)
    >>> print(artefacts.keys())
    dict_keys(['subspace1', 'subspace2', 'subspace3', 'mRR', 'ectopic', 'long',
    'short', 'missed', 'extra', 'threshold1', 'threshold2'])

    References
    ----------
    .. [1] Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for
        heart rate variability time series artefact correction using novel
        beat classification. Journal of Medical Engineering & Technology,
        43(3), 173-181. https://doi.org/10.1080/03091902.2019.1640306

    """
    rr = np.asarray(rr)

    if input_type != "rr_ms":
        rr = input_conversion(rr, input_type, output_type="rr_ms")

    ###########
    # Detection
    ###########

    # Subspace 1 (dRRs time serie)
    dRR = np.diff(rr, prepend=0)
    dRR[0] = dRR[1:].mean()  # Set first item to a realistic value

    dRR_df = pd.DataFrame({"signal": np.abs(dRR)})
    q1 = dRR_df.rolling(91, center=True, min_periods=1).quantile(0.25).signal.to_numpy()
    q3 = dRR_df.rolling(91, center=True, min_periods=1).quantile(0.75).signal.to_numpy()

    th1 = alpha * ((q3 - q1) / 2)
    dRR = dRR / th1
    s11 = dRR

    # mRRs time serie
    medRR = (
        pd.DataFrame({"signal": rr})
        .rolling(11, center=True, min_periods=1)
        .median()
        .signal.to_numpy()
    )
    mRR = rr - medRR
    mRR[mRR < 0] = 2 * mRR[mRR < 0]

    mRR_df = pd.DataFrame({"signal": np.abs(mRR)})
    q1 = mRR_df.rolling(91, center=True, min_periods=1).quantile(0.25).signal.to_numpy()
    q3 = mRR_df.rolling(91, center=True, min_periods=1).quantile(0.75).signal.to_numpy()

    th2 = alpha * ((q3 - q1) / 2)
    mRR /= th2

    # Subspace 2
    ma = np.hstack(
        [0, [np.max([dRR[i - 1], dRR[i + 1]]) for i in range(1, len(dRR) - 1)], 0]
    )
    mi = np.hstack(
        [0, [np.min([dRR[i - 1], dRR[i + 1]]) for i in range(1, len(dRR) - 1)], 0]
    )
    s12 = ma
    s12[dRR < 0] = mi[dRR < 0]

    # Subspace 3
    ma = np.hstack(
        [[np.max([dRR[i + 1], dRR[i + 2]]) for i in range(0, len(dRR) - 2)], 0, 0]
    )
    mi = np.hstack(
        [[np.min([dRR[i + 1], dRR[i + 2]]) for i in range(0, len(dRR) - 2)], 0, 0]
    )
    s22 = ma
    s22[dRR >= 0] = mi[dRR >= 0]

    ##########
    # Decision
    ##########

    # Find ectobeats
    cond1 = (s11 > 1) & (s12 < (-c1 * s11 - c2))
    cond2 = (s11 < -1) & (s12 > (-c1 * s11 + c2))
    ectopic = cond1 | cond2
    # No ectopic detection and correction at time serie edges
    ectopic[-2:] = False
    ectopic[:2] = False

    # Find long or shorts
    longBeats = ((s11 > 1) & (s22 < -1)) | ((np.abs(mRR) > 3) & (rr > np.median(rr)))
    shortBeats = ((s11 < -1) & (s22 > 1)) | ((np.abs(mRR) > 3) & (rr <= np.median(rr)))

    # Test if next interval is also outlier
    for cond in [longBeats, shortBeats]:
        for i in range(len(cond) - 2):
            if cond[i] is True:
                if np.abs(s11[i + 1]) < np.abs(s11[i + 2]):
                    cond[i + 1] = True

    # Ectopic beats are not considered as short or long
    shortBeats[ectopic] = False
    longBeats[ectopic] = False

    # Missed vector
    missed = np.abs((rr / 2) - medRR) < th2
    missed = missed & longBeats
    longBeats[missed] = False  # Missed beats are not considered as long

    # Etra vector
    extra = np.abs(rr + np.append(rr[1:], 0) - medRR) < th2
    extra = extra & shortBeats
    shortBeats[extra] = False  # Extra beats are not considered as short

    # No short or long intervals at time serie edges
    shortBeats[0], shortBeats[-1] = False, False
    longBeats[0], longBeats[-1] = False, False

    artefacts = {
        "subspace1": s11,
        "subspace2": s12,
        "subspace3": s22,
        "mRR": mRR,
        "ectopic": ectopic,
        "long": longBeats,
        "short": shortBeats,
        "missed": missed,
        "extra": extra,
        "threshold1": th1,
        "threshold2": th2,
    }

    return artefacts


def interpolate_clipping(
    signal: Union[List, np.ndarray],
    min_threshold: Optional[float] = 0.0,
    max_threshold: Optional[float] = 255.0,
    kind: str = "cubic",
) -> np.ndarray:
    """Interoplate clipping artefacts.

    This function removes all data points equalling the provided threshold
    and re-creates the missing segments using cubic spline interpolation.

    Parameters
    ----------
    signal : np.ndarray or list
        The PPG signal.
    min_threshold, max_threshold : float | None
        Minimum and maximum thresholds for clipping artefacts. If `None`, no correction
        os provided for the given threshold. Defaults to `min_threshold=0.0` and
        `max_threshold=255.0`, which corresponds to the expected values when reading
        data from the Nonin 3012LP Xpod USB pulse oximeter together with Nonin 8000SM
        'soft-clip' fingertip sensors.
    kind : str
        Specifies the kind of interpolation to perform(see
        py:func:`scipy.interpolate.interp1d`).

    Returns
    -------
    clean_signal : np.ndarray
        Interpolated signal.

    Examples
    --------
    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> from systole import import_ppg
        >>> from systole.detection import interpolate_clipping
        >>> df = import_ppg()
        >>> # Create lower and upper clipping artefacts
        >>> df.ppg.loc[df.ppg<=50] = 50
        >>> df.ppg.loc[df.ppg>=230] = 230
        >>> # Correct clipping artefacts
        >>> clean_signal = interpolate_clipping(df.ppg.to_numpy(), min_threshold=50, max_threshold=230)
        >>> # Plot
        >>> plt.plot(df.time, clean_signal, color='#F15854', label="Corrected signal")
        >>> plt.plot(df.time, df.ppg, color='#5DA5DA', label="Clipping artefacts")
        >>> plt.axhline(y=50, linestyle='--', color='k')
        >>> plt.axhline(y=230, linestyle='--', color='k')
        >>> plt.xlabel('Time (s)')
        >>> plt.ylabel('PPG level (a.u)')
        >>> plt.xlim(10, 40)
        >>> plt.legend()

    Notes
    -----
    Correct signal segment reaching recording threshold using a cubic spline
    interpolation. Adapted from [1]_.

    .. Warning:: If clipping artefact is found at the edge of the signal, this
        function will decrement/increment the first/last value to allow interpolation.

    References
    ----------
    .. [1] https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/

    """
    clean_signal = np.asarray(signal)
    time = np.arange(0, len(signal))

    if max_threshold is not None:

        # Security check for clipping at signal edge
        if clean_signal[0] == max_threshold:
            clean_signal[0] = max_threshold - 1
        if clean_signal[-1] == max_threshold:
            clean_signal[-1] = max_threshold - 1

        # Interpolate
        f = interp1d(
            time[np.where(clean_signal < max_threshold)[0]],
            clean_signal[np.where(clean_signal < max_threshold)[0]],
            kind=kind,
        )

        # Use the peaks vector as time input
        clean_signal = f(time)

    if min_threshold is not None:

        # Security check for clipping at signal edge
        if clean_signal[0] == min_threshold:
            clean_signal[0] = min_threshold + 1
        if clean_signal[-1] == min_threshold:
            clean_signal[-1] = min_threshold + 1

        # Interpolate
        f = interp1d(
            time[np.where(clean_signal > min_threshold)[0]],
            clean_signal[np.where(clean_signal > min_threshold)[0]],
            kind=kind,
        )

        # Use the peaks vector as time input
        clean_signal = f(time)

    return clean_signal
