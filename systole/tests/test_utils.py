# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import numpy as np
import pytest

from systole import import_ppg, import_rr
from systole.detection import ppg_peaks
from systole.utils import (
    heart_rate,
    input_conversion,
    norm_triggers,
    simulate_rr,
    time_shift,
    to_angles,
    to_epochs,
)


class TestUtils(TestCase):
    def test_norm_triggers(self):
        ppg = import_ppg().ppg.to_numpy()  # Import PPG recording
        signal, peaks = ppg_peaks(ppg)
        peaks[np.where(peaks)[0] + 1] = 1
        peaks[np.where(peaks)[0] + 2] = 1
        peaks[-1:] = 1
        y = norm_triggers(peaks)
        assert sum(y) == 379
        peaks = -peaks.astype(int)
        y = norm_triggers(peaks, threshold=-1, direction="lower")
        assert sum(y) == 379
        with pytest.raises(ValueError):
            norm_triggers(None)
        with pytest.raises(ValueError):
            norm_triggers(peaks, direction="invalid")

    def test_heart_rate(self):
        """Test heart_rate function"""
        ppg = import_ppg().ppg.to_numpy()  # Import PPG recording
        _, peaks = ppg_peaks(ppg)
        heartrate, time = heart_rate(peaks)
        assert len(heartrate) == len(time)
        assert np.nanmean(heartrate) == 884.92526408453
        heartrate, time = heart_rate(list(peaks))
        assert len(heartrate) == len(time)
        assert np.nanmean(heartrate) == 884.92526408453
        heartrate, time = heart_rate(peaks, unit="bpm", kind="cubic", sfreq=500)
        assert len(heartrate) == len(time)
        assert np.nanmean(heartrate) == 34.34558271737578
        with pytest.raises(ValueError):
            heartrate, time = heart_rate([1, 2, 3])
        heartrate, time = heart_rate(
            np.diff(np.where(peaks)), kind="cubic", input_type="rr_ms"
        )
        assert np.nanmean(heartrate) == 884.9253824912565
        heartrate, time = heart_rate(
            np.diff(np.where(peaks)) / 1000, kind="cubic", input_type="rr_s"
        )
        assert np.nanmean(heartrate) == 884.92526408453

    def test_time_shift(self):
        """Test time_shift function"""
        lag = time_shift([40, 50, 60], [45, 52])
        assert np.all(lag == [5, 2])

    def test_to_angle(self):
        """Test to_angles function"""
        rr = import_rr().rr.values
        # Create event vector
        events = rr + np.random.normal(500, 100, len(rr))
        ang = to_angles(list(np.cumsum(rr)), list(np.cumsum(events)))
        assert ~np.any(np.asarray(ang) < 0)
        assert ~np.any(np.asarray(ang) > np.pi * 2)
        ppg = import_ppg().ppg.to_numpy()  # Import PPG recording
        signal, peaks = ppg_peaks(ppg)
        ang = to_angles(peaks, peaks)

    def test_to_epochs(self):
        """Test ppg_peaks function"""
        ppg = import_ppg().ppg.to_numpy()  # Import PPG recording
        events = import_ppg().ppg.to_numpy()  # Import events
        events[2] = 1
        epochs = to_epochs(ppg, events, sfreq=75, verbose=True, apply_baseline=(-1, 0))
        assert epochs.ndim == 2
        epochs = to_epochs(list(ppg), list(events), sfreq=75, apply_baseline=None)
        reject = np.arange(0, len(ppg))
        reject[50:55] = 1
        epochs = to_epochs(
            ppg, events, sfreq=75, apply_baseline=-1, reject=reject, verbose=True
        )
        with pytest.raises(ValueError):
            epochs = to_epochs(ppg[1:], events, sfreq=75)

    def test_simulate_rr(self):
        """Test ppg_peaks function"""
        rr = simulate_rr(artefacts=True)
        assert isinstance(rr, np.ndarray)
        assert len(rr) == 350

    def test_input_conversion(self):

        # Load example PPG signal
        ppg = import_ppg().ppg.to_numpy()
        _, peaks = ppg_peaks(ppg)

        # input_type = "peaks"
        rr_ms = input_conversion(peaks, input_type="peaks", output_type="rr_ms")
        rr_s = input_conversion(peaks, input_type="peaks", output_type="rr_s")
        peaks_idx = input_conversion(peaks, input_type="peaks", output_type="peaks_idx")
        assert rr_ms.mean() == rr_s.mean() * 1000
        assert rr_ms.mean() == np.diff(peaks_idx).mean()

        # input_type = "peaks_idx"
        pks_idx = np.where(peaks)[0]
        rr_ms = input_conversion(pks_idx, input_type="peaks_idx", output_type="rr_ms")
        rr_s = input_conversion(pks_idx, input_type="peaks_idx", output_type="rr_s")
        pks = input_conversion(pks_idx, input_type="peaks_idx", output_type="peaks")
        assert rr_ms.mean() == rr_s.mean() * 1000
        assert rr_ms.mean() == np.diff(np.where(pks)[0]).mean()

        # input_type = "rr_ms"
        rr_ms = np.diff(np.where(peaks)[0])
        pks = input_conversion(rr_ms, input_type="rr_ms", output_type="peaks")
        rr_s = input_conversion(rr_ms, input_type="rr_ms", output_type="rr_s")
        peaks_idx = input_conversion(rr_ms, input_type="rr_ms", output_type="peaks_idx")
        assert np.diff(np.where(pks)[0]).mean() == rr_s.mean() * 1000
        assert rr_s.mean() * 1000 == np.diff(peaks_idx).mean()

        # input_type = "rr_s"
        rr_s = np.diff(np.where(peaks)[0]) / 1000
        pks = input_conversion(rr_s, input_type="rr_s", output_type="peaks")
        rr_ms = input_conversion(rr_s, input_type="rr_s", output_type="rr_ms")
        peaks_idx = input_conversion(rr_s, input_type="rr_s", output_type="peaks_idx")
        assert np.diff(np.where(pks)[0]).mean() == rr_ms.mean()
        assert rr_ms.mean() == np.diff(peaks_idx).mean()


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
