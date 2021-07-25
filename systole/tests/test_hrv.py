# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

from systole import import_rr
from systole.hrv import frequency_domain, nnX, nonlinear, pnnX, psd, rmssd, time_domain

rr = import_rr().rr.values


class TestHrv(TestCase):
    def test_nnX(self):
        """Test nnX function"""
        nn = nnX(list(rr), input_type="rr_ms")
        assert nn == 64
        nn = nnX(rr / 1000, input_type="rr_s")
        assert nn == 64
        with pytest.raises(ValueError):
            nnX(np.array([[1, 1], [1, 1]]))

    def test_pnnX(self):
        """Test pnnX function"""
        pnn = pnnX(list(rr), input_type="rr_ms")
        assert round(pnn, 2) == 26.23
        pnn = pnnX(rr / 1000, input_type="rr_s")
        assert round(pnn, 2) == 26.23
        with pytest.raises(ValueError):
            pnnX(np.array([[1, 1], [1, 1]]))

    def test_rmssd(self):
        """Test rmssd function"""
        rms = rmssd(list(rr))
        assert round(rms, 2) == 45.55
        rms = rmssd(rr / 1000, input_type="rr_s")
        assert round(rms, 2) == 45.55
        with pytest.raises(ValueError):
            rmssd(np.array([[1, 1], [1, 1]]))

    def test_time_domain(self):
        """Test time_domain function"""
        stats = time_domain(list(rr))
        assert isinstance(stats, pd.DataFrame)
        assert stats.size == 24
        with pytest.raises(ValueError):
            time_domain(np.array([[1, 1], [1, 1]]))
        stats = time_domain(rr / 1000, input_type="rr_s")

    def test_psd(self):
        """Test frequency_domain function"""
        freq, pwr = psd(rr=list(rr))
        freq2, pwr2 = psd(rr=rr / 1000, input_type="rr_s")
        assert (freq - freq2).sum() == 0.0
        assert (pwr - pwr2).sum() < 1e-10

    def test_frequency_domain(self):
        """Test frequency_domain function"""
        stats = frequency_domain(rr=list(rr))
        assert isinstance(stats, pd.DataFrame)
        assert stats.size == 22
        stats = frequency_domain(rr=rr / 1000, input_type="rr_s")

    def test_nonlinear(self):
        """Test nonlinear_domain function"""
        stats = nonlinear(list(rr))
        assert isinstance(stats, pd.DataFrame)
        self.assertEqual(stats.size, 4)
        stats = nonlinear(rr / 1000, input_type="rr_s")


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
