# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import numpy as np
import pytest

from mne import read_evokeds, pick_types, compute_proj_evoked
from mne.datasets import testing
from mne.viz import plot_projs_joint

data_dir = testing.data_path(download=False)
sample_dir = data_dir / 'MEG' / 'sample'
evoked_fname = sample_dir / 'sample_audvis-ave.fif'


@testing.requires_testing_data
def test_plot_projs_joint():
    """Test plot_projs_joint."""
    evoked = read_evokeds(evoked_fname)[0].apply_baseline((None, 0))
    evoked.info['bads'] = []
    n_mag, n_grad, n_eeg = 9, 10, 11
    n_mag_proj, n_grad_proj, n_eeg_proj = 2, 2, 1
    # We pick in this weird order to ensure our plotting order follows it
    picks = np.concatenate([
        pick_types(evoked.info, meg='grad')[:n_grad],
        pick_types(evoked.info, meg=False, eeg=True)[:n_eeg],
        pick_types(evoked.info, meg='mag')[:n_mag],
    ])
    evoked.pick(picks)
    assert len(evoked.ch_names) == n_mag + n_grad + n_eeg
    assert evoked.get_channel_types(unique=True) == ['grad', 'eeg', 'mag']
    projs = compute_proj_evoked(
        evoked, n_mag=n_mag_proj, n_grad=n_grad_proj, n_eeg=n_eeg_proj)
    assert len(projs) == 5
    with pytest.warns(RuntimeWarning, match='aliasing'):
        evoked.crop(-0.1, 0.1).decimate(10)
    topomap_kwargs = dict(res=8, contours=0, sensors=False)
    fig = plot_projs_joint(
        projs, evoked, topomap_kwargs=topomap_kwargs, verbose='error')
    ylab = fig.axes[0].get_ylabel()
    assert ylab.startswith('Grad'), ylab
    ylab = fig.axes[4].get_ylabel()
    assert ylab.startswith('EEG'), ylab
    ylab = fig.axes[7].get_ylabel()
    assert ylab.startswith('Mag'), ylab
    mag_trace_ax_idx = 10
    mag_trace_ax = fig.axes[mag_trace_ax_idx]
    assert mag_trace_ax.get_ylabel() == ''
    assert len(mag_trace_ax.lines) == n_mag + 2 * n_mag_proj
    old_len = len(mag_trace_ax.lines)
    assert len(fig.axes) == 11  # 3x4
    fig = plot_projs_joint(
        projs, evoked, picks_trace='MEG 0111', topomap_kwargs=topomap_kwargs,
        verbose='error')
    assert len(fig.axes[mag_trace_ax_idx].lines) == old_len + 1
