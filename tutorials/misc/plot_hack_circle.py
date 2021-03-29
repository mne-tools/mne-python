"""
=========
Debugging
=========

Abuse CircleCI's 'rerun with SSH' capabilities for debugging a thorny MPL
failure.
"""

import os.path as op

import numpy as np
import pytest

from mne import read_events, Epochs, pick_types
from mne.channels import read_layout
from mne.io import read_raw_fif
from mne.utils import _click_ch_name, _close_event
from mne.viz.utils import _fake_click
from mne.datasets import testing


base_dir = op.join('..', '..', 'mne', 'io', 'tests', 'data')
evoked_fname = op.join(base_dir, 'test-ave.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
cov_fname = op.join(base_dir, 'test-cov.fif')
event_name = op.join(base_dir, 'test-eve.fif')
event_id, tmin, tmax = 1, -0.1, 1.0
layout = read_layout('Vectorview-all')
test_base_dir = testing.data_path(download=False)
ctf_fname = op.join(test_base_dir, 'CTF', 'testdata_ctf.ds')


def _get_epochs(stop=5, meg=True, eeg=False, n_chan=20):
    """Get epochs."""
    raw = read_raw_fif(raw_fname)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=meg, eeg=eeg, stim=False,
                       ecg=False, eog=False, exclude='bads')
    # Use a subset of channels for plotting speed
    picks = np.round(np.linspace(0, len(picks) + 1, n_chan)).astype(int)
    with pytest.warns(RuntimeWarning, match='projection'):
        epochs = Epochs(raw, events[:stop], event_id, tmin, tmax, picks=picks,
                        proj=False, preload=False)
    epochs.info.normalize_proj()  # avoid warnings
    return epochs


def test_plot_epochs_clicks():
    """Test plot_epochs mouse interaction."""
    epochs = _get_epochs().load_data()
    fig = epochs.plot(events=epochs.events)
    data_ax = fig.mne.ax_main
    x = fig.mne.traces[0].get_xdata()[3]
    y = fig.mne.traces[0].get_ydata()[3]
    n_epochs = len(epochs)
    epoch_num = fig.mne.inst.selection[0]
    # test (un)marking bad epochs
    _fake_click(fig, data_ax, [x, y], xform='data')  # mark a bad epoch
    assert epoch_num in fig.mne.bad_epochs
    _fake_click(fig, data_ax, [x, y], xform='data')  # unmark it
    assert epoch_num not in fig.mne.bad_epochs
    _fake_click(fig, data_ax, [x, y], xform='data')  # mark it bad again
    assert epoch_num in fig.mne.bad_epochs
    # test vline
    fig.canvas.key_press_event('escape')  # close and drop epochs
    _close_event(fig)  # XXX workaround, MPL Agg doesn't trigger close event
    assert(n_epochs - 1 == len(epochs))
    # test marking bad channels
    epochs = _get_epochs(None).load_data()  # need more than 1 epoch this time
    fig = epochs.plot(n_epochs=3)
    data_ax = fig.mne.ax_main
    first_ch = data_ax.get_yticklabels()[0].get_text()
    assert first_ch not in fig.mne.info['bads']
    _click_ch_name(fig, ch_index=0, button=1)  # click ch name to mark bad
    assert first_ch in fig.mne.info['bads']
    # test clicking scrollbars
    _fake_click(fig, fig.mne.ax_vscroll, [0.5, 0.5])
    _fake_click(fig, fig.mne.ax_hscroll, [0.5, 0.5])
    # test moving bad epoch offscreen
    fig.canvas.key_press_event('right')  # move right
    x = fig.mne.traces[0].get_xdata()[-3]
    y = fig.mne.traces[0].get_ydata()[-3]
    _fake_click(fig, data_ax, [x, y], xform='data')  # mark a bad epoch
    fig.canvas.key_press_event('left')  # move back
    # out, err = capsys.readouterr()
    # assert 'out of bounds' not in out
    # assert 'out of bounds' not in err
    fig.canvas.key_press_event('escape')
    _close_event(fig)  # XXX workaround, MPL Agg doesn't trigger close event
    assert len(epochs) == 6
    # test rightclick → image plot
    fig = epochs.plot()
    _click_ch_name(fig, ch_index=0, button=3)  # show image plot
    assert len(fig.mne.child_figs) == 1
    # test scroll wheel
    fig.canvas.scroll_event(0.5, 0.5, -0.5)  # scroll down
    fig.canvas.scroll_event(0.5, 0.5, 0.5)  # scroll up


test_plot_epochs_clicks()
