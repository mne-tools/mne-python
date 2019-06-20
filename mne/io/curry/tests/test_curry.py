# -*- coding: UTF-8 -*-
#
# Authors: Dirk Gütlin <dirk.guetlin@stud.sbg.ac.at>
#
#
# License: BSD (3-clause)

import os
import os.path as op
from numpy.testing import assert_allclose, assert_array_equal
import mne
from mne.datasets import testing
from mne.io.curry import read_raw_curry
from mne.io.bti import read_raw_bti
from mne.io.constants import FIFF

data_dir = testing.data_path(download=False)
curry_dir = op.join(data_dir, "curry")

bdf_file = op.join(data_dir, 'BDF/test_bdf_stim_channel.bdf')

bti_rfDC_file = op.join(data_dir, 'BTi/erm_HFH/c,rfDC')

curry7_rfDC_file = op.join(curry_dir, "c,rfDC Curry 7.dat")
curry8_rfDC_file = op.join(curry_dir, "c,rfDC Curry 8.cdt")

curry7_bdf_file = op.join(curry_dir, "test_bdf_stim_channel Curry 7.dat")
curry7_bdf_ascii_file = op.join(curry_dir,
                                "test_bdf_stim_channel Curry 7 ASCII.dat")

curry8_bdf_file = op.join(curry_dir, "test_bdf_stim_channel Curry 8.cdt")
curry8_bdf_ascii_file = op.join(curry_dir,
                                "test_bdf_stim_channel Curry 8 ASCII.cdt")


def test_io_curry():
    """Test reading CURRY files."""

    # make sure all test files can be passed to read_raw_curry
    for filename in os.listdir(curry_dir):
        read_raw_curry(op.join(curry_dir, filename))


    bti_rfDC = read_raw_bti(pdf_fname=bti_rfDC_file, head_shape_fname=None)
    curry7_rfDC = read_raw_curry(curry7_rfDC_file)
    curry8_rfDC = read_raw_curry(curry8_rfDC_file)

    # test on the eeg chans, since these were not renamed by curry
    eeg_names = [ch["ch_name"] for ch in curry7_rfDC.info["chs"] \
                 if ch["kind"] == FIFF.FIFFV_EEG_CH]

    assert_allclose(curry7_rfDC.get_data(eeg_names),
                    bti_rfDC.get_data(eeg_names), rtol=1e-6)
    assert_allclose(curry8_rfDC.get_data(eeg_names),
                    bti_rfDC.get_data(eeg_names), rtol=1e-3)

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

    # we only use chans [0:3] here since the stim channel was already extracted
    for curry_file in [curry7_bdf, curry7_bdf_ascii, curry8_bdf, curry8_bdf_ascii]:
        assert_allclose([ch["kind"] for ch in curry_file.info["chs"]],
                        [ch["kind"] for ch in bdf.info["chs"][0:3]])
        assert_array_equal([ch["ch_name"] for ch in curry_file.info["chs"]],
                           [ch["ch_name"] for ch in bdf.info["chs"][0:3]])
        assert_allclose(curry_file.get_data(), bdf.get_data()[0:3], atol=1e-6)
