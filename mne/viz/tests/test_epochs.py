# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import os.path as op
import warnings

import numpy as np

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server
import matplotlib.pyplot as plt

from mne import io, read_events, Epochs
from mne import pick_types
from mne.layouts import read_layout
from mne.datasets import sample

from mne.viz import plot_drop_log, plot_image_epochs


warnings.simplefilter('always')  # enable b/c these tests throw warnings


data_dir = sample.data_path(download=False)
subjects_dir = op.join(data_dir, 'subjects')
ecg_fname = op.join(data_dir, 'MEG', 'sample', 'sample_audvis_ecg_proj.fif')

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
evoked_fname = op.join(base_dir, 'test-ave.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
cov_fname = op.join(base_dir, 'test-cov.fif')
event_name = op.join(base_dir, 'test-eve.fif')
event_id, tmin, tmax = 1, -0.1, 0.1
n_chan = 15
layout = read_layout('Vectorview-all')


def _get_raw():
    return io.Raw(raw_fname, preload=False)


def _get_events():
    return read_events(event_name)


def _get_picks(raw):
    return pick_types(raw.info, meg=True, eeg=False, stim=False,
                      ecg=False, eog=False, exclude='bads')


def _get_epochs():
    raw = _get_raw()
    events = _get_events()
    picks = _get_picks(raw)
    # Use a subset of channels for plotting speed
    picks = np.round(np.linspace(0, len(picks) + 1, n_chan)).astype(int)
    epochs = Epochs(raw, events[:5], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0))
    return epochs


def _get_epochs_delayed_ssp():
    raw = _get_raw()
    events = _get_events()
    picks = _get_picks(raw)
    reject = dict(mag=4e-12)
    epochs_delayed_ssp = Epochs(raw, events[:10], event_id, tmin, tmax,
                                picks=picks, baseline=(None, 0),
                                proj='delayed', reject=reject)
    return epochs_delayed_ssp


def test_plot_epochs():
    """ Test plotting epochs
    """
    epochs = _get_epochs()
    epochs.plot([0, 1], picks=[0, 2, 3], scalings=None, title_str='%s')
    epochs[0].plot(picks=[0, 2, 3], scalings=None, title_str='%s')
    # test clicking: should increase coverage on
    # 3200-3226, 3235, 3237, 3239-3242, 3245-3255, 3260-3280
    fig = plt.gcf()
    fig.canvas.button_press_event(10, 10, 'left')
    # now let's add a bad channel
    epochs.info['bads'] = [epochs.ch_names[0]]  # include a bad one
    epochs.plot([0, 1], picks=[0, 2, 3], scalings=None, title_str='%s')
    epochs[0].plot(picks=[0, 2, 3], scalings=None, title_str='%s')
    plt.close('all')


def test_plot_image_epochs():
    """Test plotting of epochs image
    """
    epochs = _get_epochs()
    plot_image_epochs(epochs, picks=[1, 2])
    plt.close('all')


def test_plot_drop_log():
    """Test plotting a drop log
    """
    epochs = _get_epochs()
    epochs.drop_bad_epochs()

    warnings.simplefilter('always', UserWarning)
    with warnings.catch_warnings(record=True):
        epochs.plot_drop_log()

        plot_drop_log([['One'], [], []])
        plot_drop_log([['One'], ['Two'], []])
        plot_drop_log([['One'], ['One', 'Two'], []])
    plt.close('all')

