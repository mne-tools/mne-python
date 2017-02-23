from nose.tools import assert_true
from unittest import TestCase

import os.path as op
from mne import Epochs, pick_types, read_events
from mne.preprocessing import ICA, read_ica
from mne.io import read_raw_fif
from mne.datasets import sample
from mne.utils import _TempDir, run_tests_if_main

import warnings
warnings.simplefilter('always')


class TestLoadICA(TestCase):
    def setUp(self):
        data_path = sample.data_path()
        raw_fname = op.join(data_path,
                            'MEG/sample/sample_audvis_filt-0-40_raw.fif')

        event_fname = op.join(data_path,
                              'MEG/sample/sample_audvis_filt-0-40_raw-eve.fif')

        tmin, tmax = -0.2, 0.5
        event_id = dict(aud_l=1, vis_l=3)

        raw = read_raw_fif(raw_fname, preload=True)
        events = read_events(event_fname)

        picks = pick_types(raw.info, eeg=True, meg=False)

        self.epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True,
                             baseline=None, preload=True, picks=picks)

        self._output_path = _TempDir()
        self.output_fname = op.join(self._output_path, 'test_ica-ica.fif')
        self.random_state = 12345

    def test_n_components_none(self):
        ica = ICA(max_pca_components=10, n_components=None,
                  random_state=self.random_state)
        ica.fit(self.epochs)
        ica.save(self.output_fname)

        ica = read_ica(self.output_fname)

        assert_true(ica.max_pca_components == 10)
        assert_true(ica.n_components is None)

    def test_max_pca_components_none(self):
        ica = ICA(max_pca_components=None, n_components=10,
                  random_state=self.random_state)
        ica.fit(self.epochs)
        ica.save(self.output_fname)

        ica = read_ica(self.output_fname)

        # ICA.fit() replaced max_pca_components, which was previously None,
        # with the appropriate integer value.
        assert_true(ica.max_pca_components == self.epochs.info['nchan'])
        assert_true(ica.n_components == 10)

    def test_n_components_and_max_pca_components_none(self):
        ica = ICA(max_pca_components=None, n_components=None,
                  random_state=self.random_state)
        ica.fit(self.epochs)
        ica.save(self.output_fname)

        ica = read_ica(self.output_fname)

        # ICA.fit() replaced max_pca_components, which was previously None,
        # with the appropriate integer value.
        assert_true(ica.max_pca_components == self.epochs.info['nchan'])
        assert_true(ica.n_components is None)


run_tests_if_main()
