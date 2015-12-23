# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import os.path as op
import warnings

import numpy as np
from numpy.testing import assert_raises, assert_array_equal

from nose.tools import assert_true, assert_equal


from mne import io, read_evokeds, read_proj
from mne.io.constants import FIFF
from mne.channels import read_layout, make_eeg_layout
from mne.datasets import testing
from mne.time_frequency.tfr import AverageTFR
from mne.utils import slow_test

from mne.viz import plot_evoked_topomap, plot_projs_topomap
from mne.viz.topomap import (_check_outlines, _onselect, plot_topomap,
                             _find_peaks)

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

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
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    # evoked
    warnings.simplefilter('always')
    res = 16
    evoked = read_evokeds(evoked_fname, 'Left Auditory',
                          baseline=(None, 0))
    ev_bad = evoked.pick_types(meg=False, eeg=True, copy=True)
    ev_bad.pick_channels(ev_bad.ch_names[:2])
    ev_bad.plot_topomap(times=ev_bad.times[:2] - 1e-6)  # auto, should plot EEG
    assert_raises(ValueError, ev_bad.plot_topomap, ch_type='mag')
    assert_raises(TypeError, ev_bad.plot_topomap, head_pos='foo')
    assert_raises(KeyError, ev_bad.plot_topomap, head_pos=dict(foo='bar'))
    assert_raises(ValueError, ev_bad.plot_topomap, head_pos=dict(center=0))
    assert_raises(ValueError, ev_bad.plot_topomap, times=[-100])  # bad time
    assert_raises(ValueError, ev_bad.plot_topomap, times=[[0]])  # bad time
    assert_raises(ValueError, ev_bad.plot_topomap, times=[[0]])  # bad time

    evoked.plot_topomap(0.1, layout=layout, scale=dict(mag=0.1))
    plt.close('all')
    axes = [plt.subplot(221), plt.subplot(222)]
    evoked.plot_topomap(axes=axes, colorbar=False)
    plt.close('all')
    evoked.plot_topomap(times=[-0.1, 0.2])
    plt.close('all')
    mask = np.zeros_like(evoked.data, dtype=bool)
    mask[[1, 5], :] = True
    evoked.plot_topomap(ch_type='mag', outlines=None)
    times = [0.1]
    evoked.plot_topomap(times, ch_type='eeg', res=res, scale=1)
    evoked.plot_topomap(times, ch_type='grad', mask=mask, res=res)
    evoked.plot_topomap(times, ch_type='planar1', res=res)
    evoked.plot_topomap(times, ch_type='planar2', res=res)
    evoked.plot_topomap(times, ch_type='grad', mask=mask, res=res,
                        show_names=True, mask_params={'marker': 'x'})
    plt.close('all')
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
    plt.close('all')

    # delaunay triangulation warning
    with warnings.catch_warnings(record=True):  # can't show
        warnings.simplefilter('always')
        evoked.plot_topomap(times, ch_type='mag', layout=None, res=res)
    assert_raises(RuntimeError, plot_evoked_topomap, evoked, 0.1, 'mag',
                  proj='interactive')  # projs have already been applied

    # change to no-proj mode
    evoked = read_evokeds(evoked_fname, 'Left Auditory',
                          baseline=(None, 0), proj=False)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        evoked.plot_topomap(0.1, 'mag', proj='interactive', res=res)
    assert_raises(RuntimeError, plot_evoked_topomap, evoked,
                  np.repeat(.1, 50))
    assert_raises(ValueError, plot_evoked_topomap, evoked, [-3e12, 15e6])

    with warnings.catch_warnings(record=True):  # file conventions
        warnings.simplefilter('always')
        projs = read_proj(ecg_fname)
    projs = [pp for pp in projs if pp['desc'].lower().find('eeg') < 0]
    plot_projs_topomap(projs, res=res)
    plt.close('all')
    ax = plt.subplot(111)
    plot_projs_topomap([projs[0]], res=res, axes=ax)  # test axes param
    plt.close('all')
    for ch in evoked.info['chs']:
        if ch['coil_type'] == FIFF.FIFFV_COIL_EEG:
            ch['loc'].fill(0)

    # Remove extra digitization point, so EEG digitization points
    # correspond with the EEG electrodes
    del evoked.info['dig'][85]

    pos = make_eeg_layout(evoked.info).pos[:, :2]
    pos, outlines = _check_outlines(pos, 'head')
    assert_true('head' in outlines.keys())
    assert_true('nose' in outlines.keys())
    assert_true('ear_left' in outlines.keys())
    assert_true('ear_right' in outlines.keys())
    assert_true('autoshrink' in outlines.keys())
    assert_true(outlines['autoshrink'])
    assert_true('clip_radius' in outlines.keys())
    assert_array_equal(outlines['clip_radius'], 0.5)

    pos, outlines = _check_outlines(pos, 'skirt')
    assert_true('head' in outlines.keys())
    assert_true('nose' in outlines.keys())
    assert_true('ear_left' in outlines.keys())
    assert_true('ear_right' in outlines.keys())
    assert_true('autoshrink' in outlines.keys())
    assert_true(not outlines['autoshrink'])
    assert_true('clip_radius' in outlines.keys())
    assert_array_equal(outlines['clip_radius'], 0.625)

    pos, outlines = _check_outlines(pos, 'skirt',
                                    head_pos={'scale': [1.2, 1.2]})
    assert_array_equal(outlines['clip_radius'], 0.75)

    # Plot skirt
    evoked.plot_topomap(times, ch_type='eeg', outlines='skirt')

    # Pass custom outlines without patch
    evoked.plot_topomap(times, ch_type='eeg', outlines=outlines)
    plt.close('all')

    # Pass custom outlines with patch callable
    def patch():
        return Circle((0.5, 0.4687), radius=.46,
                      clip_on=True, transform=plt.gca().transAxes)
    outlines['patch'] = patch
    plot_evoked_topomap(evoked, times, ch_type='eeg', outlines=outlines)

    # Remove digitization points. Now topomap should fail
    evoked.info['dig'] = None
    assert_raises(RuntimeError, plot_evoked_topomap, evoked,
                  times, ch_type='eeg')
    plt.close('all')

    # Error for missing names
    n_channels = len(pos)
    data = np.ones(n_channels)
    assert_raises(ValueError, plot_topomap, data, pos, show_names=True)

    # Test error messages for invalid pos parameter
    pos_1d = np.zeros(n_channels)
    pos_3d = np.zeros((n_channels, 2, 2))
    assert_raises(ValueError, plot_topomap, data, pos_1d)
    assert_raises(ValueError, plot_topomap, data, pos_3d)
    assert_raises(ValueError, plot_topomap, data, pos[:3, :])

    pos_x = pos[:, :1]
    pos_xyz = np.c_[pos, np.zeros(n_channels)[:, np.newaxis]]
    assert_raises(ValueError, plot_topomap, data, pos_x)
    assert_raises(ValueError, plot_topomap, data, pos_xyz)

    # An #channels x 4 matrix should work though. In this case (x, y, width,
    # height) is assumed.
    pos_xywh = np.c_[pos, np.zeros((n_channels, 2))]
    plot_topomap(data, pos_xywh)
    plt.close('all')

    # Test peak finder
    axes = [plt.subplot(131), plt.subplot(132)]
    evoked.plot_topomap(times='peaks', axes=axes)
    plt.close('all')
    evoked.data = np.zeros(evoked.data.shape)
    evoked.data[50][1] = 1
    assert_array_equal(_find_peaks(evoked, 10), evoked.times[1])
    evoked.data[80][100] = 1
    assert_array_equal(_find_peaks(evoked, 10), evoked.times[[1, 100]])
    evoked.data[2][95] = 2
    assert_array_equal(_find_peaks(evoked, 10), evoked.times[[1, 95]])
    assert_array_equal(_find_peaks(evoked, 1), evoked.times[95])


def test_plot_tfr_topomap():
    """Test plotting of TFR data
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    raw = _get_raw()
    times = np.linspace(-0.1, 0.1, 200)
    n_freqs = 3
    nave = 1
    rng = np.random.RandomState(42)
    data = rng.randn(len(raw.ch_names), n_freqs, len(times))
    tfr = AverageTFR(raw.info, data, times, np.arange(n_freqs), nave)
    tfr.plot_topomap(ch_type='mag', tmin=0.05, tmax=0.150, fmin=0, fmax=10,
                     res=16)

    eclick = mpl.backend_bases.MouseEvent('button_press_event',
                                          plt.gcf().canvas, 0, 0, 1)
    eclick.xdata = 0.1
    eclick.ydata = 0.1
    eclick.inaxes = plt.gca()
    erelease = mpl.backend_bases.MouseEvent('button_release_event',
                                            plt.gcf().canvas, 0.9, 0.9, 1)
    erelease.xdata = 0.3
    erelease.ydata = 0.2
    pos = [[0.11, 0.11], [0.25, 0.5], [0.0, 0.2], [0.2, 0.39]]
    _onselect(eclick, erelease, tfr, pos, 'mag', 1, 3, 1, 3, 'RdBu_r', list())
    tfr._onselect(eclick, erelease, None, 'mean', None)
    plt.close('all')
