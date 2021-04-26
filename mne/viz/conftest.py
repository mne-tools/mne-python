# Authors: Robert Luke <mail@robertluke.net>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import inspect
from textwrap import dedent

import pytest
import numpy as np
import os.path as op

from mne import create_info, EvokedArray, events_from_annotations, Epochs
from mne.channels import make_standard_montage
from mne.datasets.testing import data_path, _pytest_param
from mne.preprocessing.nirs import optical_density, beer_lambert_law
from mne.io import read_raw_nirx
from mne.utils import Bunch


@pytest.fixture()
def fnirs_evoked():
    """Create an fnirs evoked structure."""
    montage = make_standard_montage('biosemi16')
    ch_names = montage.ch_names
    ch_types = ['eeg'] * 16
    info = create_info(ch_names=ch_names, sfreq=20, ch_types=ch_types)
    evoked_data = np.random.randn(16, 30)
    evoked = EvokedArray(evoked_data, info=info, tmin=-0.2, nave=4)
    evoked.set_montage(montage)
    evoked.set_channel_types({'Fp1': 'hbo', 'Fp2': 'hbo', 'F4': 'hbo',
                             'Fz': 'hbo'}, verbose='error')
    return evoked


@pytest.fixture(params=[_pytest_param()])
def fnirs_epochs():
    """Create an fnirs epoch structure."""
    fname = op.join(data_path(download=False),
                    'NIRx', 'nirscout', 'nirx_15_2_recording_w_overlap')
    raw_intensity = read_raw_nirx(fname, preload=False)
    raw_od = optical_density(raw_intensity)
    raw_haemo = beer_lambert_law(raw_od)
    evts, _ = events_from_annotations(raw_haemo, event_id={'1.0': 1})
    evts_dct = {'A': 1}
    tn, tx = -1, 2
    epochs = Epochs(raw_haemo, evts, event_id=evts_dct, tmin=tn, tmax=tx)
    return epochs


# Create one nbclient and reuse it
@pytest.fixture(scope='session')
def _nbclient():
    try:
        import nbformat
        from jupyter_client import AsyncKernelManager
        from nbclient import NotebookClient
        from ipywidgets import Button  # noqa
        import ipyvtk_simple  # noqa
    except Exception as exc:
        return pytest.skip(f'Skipping Notebook test: {exc}')
    km = AsyncKernelManager(config=None)
    nb = nbformat.reads("""
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata":{},
   "outputs": [],
   "source":[]
  }
 ],
 "metadata": {
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version":3},
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}""", as_version=4)
    client = NotebookClient(nb, km=km)
    yield client
    client._cleanup_kernel()


@pytest.fixture(scope='function')
def nbexec(_nbclient):
    """Execute Python code in a notebook."""
    # Adapted/simplified from nbclient/client.py (BSD 3-clause)
    _nbclient._cleanup_kernel()

    def execute(code, reset=False):
        _nbclient.reset_execution_trackers()
        with _nbclient.setup_kernel():
            assert _nbclient.kc is not None
            cell = Bunch(cell_type='code', metadata={}, source=dedent(code))
            _nbclient.execute_cell(cell, 0, execution_count=0)
            _nbclient.set_widgets_metadata()

    yield execute


def pytest_runtest_call(item):
    """Run notebook code written in Python."""
    if 'nbexec' in getattr(item, 'fixturenames', ()):
        nbexec = item.funcargs['nbexec']
        code = inspect.getsource(getattr(item.module, item.name.split('[')[0]))
        code = code.splitlines()
        ci = 0
        for ci, c in enumerate(code):
            if c.startswith('    '):  # actual content
                break
        code = '\n'.join(code[ci:])

        def run(nbexec=nbexec, code=code):
            nbexec(code)

        item.runtest = run
    return
