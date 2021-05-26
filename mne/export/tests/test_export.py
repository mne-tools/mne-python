# -*- coding: utf-8 -*-
"""Test exporting functions."""
# Authors: MNE Developers
#
# License: BSD (3-clause)

from pathlib import Path
import os.path as op

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mne import read_epochs_eeglab, Epochs
from mne.tests.test_epochs import _get_data
from mne.io import read_raw_fif, read_raw_eeglab
from mne.utils import _check_eeglabio_installed


@pytest.mark.skipif(not _check_eeglabio_installed(strict=False),
                    reason='eeglabio not installed')
def test_export_raw_eeglab(tmpdir):
    """Test saving a Raw instance to EEGLAB's set format."""
    fname = (Path(__file__).parent.parent.parent /
             "io" / "tests" / "data" / "test_raw.fif")
    raw = read_raw_fif(fname)
    raw.load_data()
    temp_fname = op.join(str(tmpdir), 'test.set')
    raw.export(temp_fname)
    raw.drop_channels([ch for ch in ['epoc']
                       if ch in raw.ch_names])
    raw_read = read_raw_eeglab(temp_fname, preload=True)
    assert raw.ch_names == raw_read.ch_names
    cart_coords = np.array([d['loc'][:3] for d in raw.info['chs']])  # just xyz
    cart_coords_read = np.array([d['loc'][:3] for d in raw_read.info['chs']])
    assert_allclose(cart_coords, cart_coords_read)
    assert_allclose(raw.times, raw_read.times)
    assert_allclose(raw.get_data(), raw_read.get_data())


@pytest.mark.skipif(not _check_eeglabio_installed(strict=False),
                    reason='eeglabio not installed')
@pytest.mark.parametrize('preload', (True, False))
def test_export_epochs_eeglab(tmpdir, preload):
    """Test saving an Epochs instance to EEGLAB's set format."""
    raw, events = _get_data()[:2]
    raw.load_data()
    epochs = Epochs(raw, events, preload=preload)
    temp_fname = op.join(str(tmpdir), 'test.set')
    epochs.export(temp_fname)
    epochs.drop_channels([ch for ch in ['epoc', 'STI 014']
                          if ch in epochs.ch_names])
    epochs_read = read_epochs_eeglab(temp_fname)
    assert epochs.ch_names == epochs_read.ch_names
    cart_coords = np.array([d['loc'][:3]
                           for d in epochs.info['chs']])  # just xyz
    cart_coords_read = np.array([d['loc'][:3]
                                for d in epochs_read.info['chs']])
    assert_allclose(cart_coords, cart_coords_read)
    assert_array_equal(epochs.events[:, 0],
                       epochs_read.events[:, 0])  # latency
    assert epochs.event_id.keys() == epochs_read.event_id.keys()  # just keys
    assert_allclose(epochs.times, epochs_read.times)
    assert_allclose(epochs.get_data(), epochs_read.get_data())
