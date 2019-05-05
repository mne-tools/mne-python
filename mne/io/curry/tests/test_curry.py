# -*- coding: UTF-8 -*-
#
# Authors: Dirk Gütlin <dirk.guetlin@stud.sbg.ac.at>
#
#
# License: BSD (3-clause)

import mne
import inspect
import os
import os.path as op
from mne.datasets import testing
from mne.io.curry import read_raw_curry
from mne.io.ctf import read_raw_ctf
from mne.io.bti import read_raw_bti
from numpy.testing import assert_allclose, assert_equal, assert_array_equal

FILE = inspect.getfile(inspect.currentframe())
curry_data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')

ext_data_dir = testing.data_path(download=False)

bdf_file = op.join(ext_data_dir, 'BDF/test_bdf_stim_channel.bdf')

ctf_alp_file = op.join(ext_data_dir, 'CTF/catch-alp-good-f.ds')

bti_rfDC_file = op.join(ext_data_dir, 'BTi/erm_HFH/c,rfDC')

curry7_rfDC_file = op.join(curry_data_dir, "c,rfDC Curry 7.dat")
curry8_rfDC_file = op.join(curry_data_dir, "c,rfDC Curry 8.cdt")

curry7_alp_file = op.join(curry_data_dir, "catch-alp-good-f Curry 7.dat")
curry8_alp_file = op.join(curry_data_dir, "catch-alp-good-f Curry 8.cdt")

curry7_bdf_file = op.join(curry_data_dir, "test_bdf_stim_channel Curry 7.dat")
curry7_bdf_ascii_file = op.join(curry_data_dir,
                                "test_bdf_stim_channel Curry 7 ASCII.dat")

curry8_bdf_file = op.join(curry_data_dir, "test_bdf_stim_channel Curry 8.cdt")
curry8_bdf_ascii_file = op.join(curry_data_dir,
                                "test_bdf_stim_channel Curry 8 ASCII.cdt")


def test_io_curry():
    """Test reading CURRY files."""

    # make sure all test files can be passed to read_raw_curry
    for filename in os.listdir(curry_data_dir):
        read_raw_curry(op.join(curry_data_dir, filename))

    ctf_alp = read_raw_ctf(ctf_alp_file)
    curry7_alp = read_raw_curry(curry7_alp_file)
    curry8_alp = read_raw_curry(curry8_alp_file)

    assert_equal(curry7_alp.n_times, ctf_alp.n_times)
    assert_equal(curry8_alp.n_times, ctf_alp.n_times)
    assert_equal(curry7_alp.info["sfreq"], ctf_alp.info["sfreq"])
    assert_equal(curry8_alp.info["sfreq"], ctf_alp.info["sfreq"])

    assert_array_equal(curry7_alp._data, ctf_alp.get_data())
    assert_array_equal(curry8_alp._data, ctf_alp.get_data())

    bti_rfDC = read_raw_bti(bti_rfDC_file)  # where is the headfile for this?
    curry7_rfDC = read_raw_curry(curry7_rfDC_file)
    curry8_rfDC = read_raw_curry(curry8_rfDC_file)

    assert_array_equal(curry7_rfDC._data, bti_rfDC.get_data())
    assert_array_equal(curry8_rfDC._data, bti_rfDC.get_data())

    bdf = mne.io.read_raw_bdf(bdf_file)
    curry7_bdf = read_raw_curry(curry7_bdf_file)
    curry7_bdf_ascii = read_raw_curry(curry7_bdf_ascii_file)
    curry8_bdf = read_raw_curry(curry8_bdf_file)
    curry8_bdf_ascii = read_raw_curry(curry7_bdf_ascii_file)

    assert_allclose([curry7_bdf.n_times, curry7_bdf_ascii.n_times,
                     curry8_bdf.n_times, curry8_bdf_ascii.n_times],
                    bdf.n_times)
    assert_allclose([curry7_bdf.info["sfreq"], curry7_bdf_ascii.info["sfreq"],
                     curry8_bdf.info["sfreq"], curry8_bdf_ascii.info["sfreq"]],
                    bdf.info["sfreq"])

    assert_array_equal(curry7_bdf._data, bdf.get_data())
    assert_array_equal(curry7_bdf_ascii._data, bdf.get_data())
    assert_array_equal(curry8_bdf._data, bdf.get_data())
    assert_array_equal(curry8_bdf_ascii._data, bdf.get_data())
