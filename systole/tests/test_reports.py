# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import matplotlib.pyplot as plt
from bokeh.models import Column

from systole import import_rr
from systole.hrv import frequency_domain, nonlinear_domain, time_domain
from systole.reports import frequency_table, nonlinear_table, time_table


class TestReports(TestCase):
    def test_time_table(self):
        """Test the time_table function"""
        rr = import_rr().rr
        time_df = time_domain(rr, input_type="rr_ms")

        # With a df as input
        table_df = time_table(time_df=time_df, backend="tabulate")
        assert isinstance(table_df, str)

        table = time_table(time_df=time_df, backend="bokeh")
        assert isinstance(table, Column)

        # With RR intervals as inputs
        table_rr = time_table(rr=rr, backend="tabulate")
        assert isinstance(table_rr, str)

        table = time_table(rr=rr, backend="bokeh")
        assert isinstance(table, Column)

        # Check for consistency between methods
        assert table_rr == table_df

        plt.close("all")

    def test_frequency_table(self):
        """Test plot_subspaces function"""
        rr = import_rr().rr
        frequency_df = frequency_domain(rr, input_type="rr_ms")

        # With a df as input
        table_df = frequency_table(frequency_df=frequency_df, backend="tabulate")
        assert isinstance(table_df, str)

        table = frequency_table(frequency_df=frequency_df, backend="bokeh")
        assert isinstance(table, Column)

        # With RR intervals as inputs
        table_rr = frequency_table(rr=rr, backend="tabulate")
        assert isinstance(table_rr, str)

        table = frequency_table(rr=rr, backend="bokeh")
        assert isinstance(table, Column)

        # Check for consistency between methods
        assert table_rr == table_df

        plt.close("all")

    def test_nonlinear_table(self):
        """Test plot_subspaces function"""
        rr = import_rr().rr
        nonlinear_df = nonlinear_domain(rr, input_type="rr_ms")

        # With a df as input
        table_df = nonlinear_table(nonlinear_df=nonlinear_df, backend="tabulate")
        assert isinstance(table_df, str)

        table = nonlinear_table(nonlinear_df=nonlinear_df, backend="bokeh")
        assert isinstance(table, Column)

        # With RR intervals as inputs
        table_rr = nonlinear_table(rr=rr, backend="tabulate")
        assert isinstance(table_rr, str)

        table = nonlinear_table(rr=rr, backend="bokeh")
        assert isinstance(table, Column)

        # Check for consistency between methods
        assert table_rr == table_df

        plt.close("all")


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
