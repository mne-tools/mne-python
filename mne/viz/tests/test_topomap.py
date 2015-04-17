# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import os.path as op
import warnings

import numpy as np
from numpy.testing import assert_raises

from nose.tools import assert_true, assert_equal


from mne import io, read_evokeds, read_proj
from mne.io.constants import FIFF
from mne.channels import read_layout, make_eeg_layout
from mne.datasets import testing
from mne.time_frequency.tfr import AverageTFR
from mne.utils import slow_test

from mne.viz import plot_evoked_topomap, plot_projs_topomap
from mne.viz.topomap import _check_outlines

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server
import matplotlib.pyplot as plt  # noqa
from matplotlib.patches import Circle  # noqa

warnings.simplefilter('always')  # enable b/c these tests throw warnings


data_dir = testing.data_path(download=False)
subjects_dir = op.join(data_dir, 'subjects')
ecg_fname = op.join(data_dir, 'MEG', 'sample', 'sample_audvis_ecg_proj.fif')

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
evoked_fname = op.join(base_dir, 'test-ave.fif')
fname = op.join(base_dir, 'test-ave.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
layout = read_layout('Vectorview-all')


def _get_raw():
    return io.Raw(raw_fname, preload=False)


@slow_test
@testing.requires_testing_data
def test_plot_topomap():
    """Test topomap plotting
    """
    # evoked
    warnings.simplefilter('always', UserWarning)
    res = 16
    with warnings.catch_warnings(record=True):
        evoked = read_evokeds(evoked_fname, 'Left Auditory',
                              baseline=(None, 0))
        evoked.plot_topomap(0.1, 'mag', layout=layout)
        mask = np.zeros_like(evoked.data, dtype=bool)
        mask[[1, 5], :] = True
        evoked.plot_topomap(None, ch_type='mag', outlines=None)
        times = [0.1]
        evoked.plot_topomap(times, ch_type='eeg', res=res)
        evoked.plot_topomap(times, ch_type='grad', mask=mask, res=res)
        evoked.plot_topomap(times, ch_type='planar1', res=res)
        evoked.plot_topomap(times, ch_type='planar2', res=res)
        evoked.plot_topomap(times, ch_type='grad', mask=mask, res=res,
                            show_names=True, mask_params={'marker': 'x'})
        assert_raises(ValueError, evoked.plot_topomap, times, ch_type='eeg',
                      res=res, average=-1000)
        assert_raises(ValueError, evoked.plot_topomap, times, ch_type='eeg',
                      res=res, average='hahahahah')

        p = evoked.plot_topomap(times, ch_type='grad', res=res,
                                show_names=lambda x: x.replace('MEG', ''),
                                image_interp='bilinear')
        subplot = [x for x in p.get_children() if
                   isinstance(x, matplotlib.axes.Subplot)][0]
        assert_true(all('MEG' not in x.get_text()
                        for x in subplot.get_children()
                        if isinstance(x, matplotlib.text.Text)))

        # Test title
        def get_texts(p):
            return [x.get_text() for x in p.get_children() if
                    isinstance(x, matplotlib.text.Text)]

        p = evoked.plot_topomap(times, ch_type='eeg', res=res, average=0.01)
        assert_equal(len(get_texts(p)), 0)
        p = evoked.plot_topomap(times, ch_type='eeg', title='Custom', res=res)
        texts = get_texts(p)
        assert_equal(len(texts), 1)
        assert_equal(texts[0], 'Custom')

        # delaunay triangulation warning
        with warnings.catch_warnings(record=True):
            evoked.plot_topomap(times, ch_type='mag', layout=None, res=res)
        assert_raises(RuntimeError, plot_evoked_topomap, evoked, 0.1, 'mag',
                      proj='interactive')  # projs have already been applied

        # change to no-proj mode
        evoked = read_evokeds(evoked_fname, 'Left Auditory',
                              baseline=(None, 0), proj=False)
        evoked.plot_topomap(0.1, 'mag', proj='interactive', res=res)
        assert_raises(RuntimeError, plot_evoked_topomap, evoked,
                      np.repeat(.1, 50))
        assert_raises(ValueError, plot_evoked_topomap, evoked, [-3e12, 15e6])

        projs = read_proj(ecg_fname)
        projs = [pp for pp in projs if pp['desc'].lower().find('eeg') < 0]
        plot_projs_topomap(projs, res=res)
        plt.close('all')
        for ch in evoked.info['chs']:
            if ch['coil_type'] == FIFF.FIFFV_COIL_EEG:
                if ch['eeg_loc'] is not None:
                    ch['eeg_loc'].fill(0)
                ch['loc'].fill(0)

        # Remove extra digitization point, so EEG digitization points
        # correspond with the EEG electrodes
        del evoked.info['dig'][85]

        pos = make_eeg_layout(evoked.info).pos
        pos, outlines = _check_outlines(pos, outlines='head')

        def patch():
            return Circle((0.5, 0.4687), radius=.46,
                          clip_on=True, transform=plt.gca().transAxes)
        import pdb;pdb.set_trace()
        plot_evoked_topomap(evoked, times, ch_type='eeg', outlines='head')
        # Remove digitization points. Now topomap should fail
        evoked.info['dig'] = None
        assert_raises(RuntimeError, plot_evoked_topomap, evoked,
                      times, ch_type='eeg')


def test_plot_tfr_topomap():
    """Test plotting of TFR data
    """
    raw = _get_raw()
    times = np.linspace(-0.1, 0.1, 200)
    n_freqs = 3
    nave = 1
    rng = np.random.RandomState(42)
    data = rng.randn(len(raw.ch_names), n_freqs, len(times))
    tfr = AverageTFR(raw.info, data, times, np.arange(n_freqs), nave)
    tfr.plot_topomap(ch_type='mag', tmin=0.05, tmax=0.150, fmin=0, fmax=10,
                     res=16)


def test_prepare_topo_plot():
    """Test obtaining 2D coordinates from 3D sensor locations"""
