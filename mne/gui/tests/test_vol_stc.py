# -*- coding: utf-8 -*-
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-clause

import sys
import numpy as np
from numpy.testing import assert_allclose

import pytest

import mne
from mne.datasets import testing
from mne.io.constants import FIFF
from mne.viz.utils import _fake_click

data_path = testing.data_path(download=False)
subject = 'sample'
subjects_dir = data_path / 'subjects'
fname_raw = data_path / 'MEG' / 'sample' / 'sample_audvis_trunc_raw.fif'
fname_fwd_vol = data_path / 'MEG' / 'sample' / \
    'sample_audvis_trunc-meg-vol-7-fwd.fif'
fname_fwd = \
    data_path / 'MEG' / 'sample' / 'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif'

# TO DO: remove when Azure fixed, causes
# 'Windows fatal exception: access violation'
# but fails to replicate locally on a Windows machine
if sys.platform == 'win32':
    pytest.skip('Azure CI problem on Windows', allow_module_level=True)


def _fake_stc(src_type='vol'):
    """Fake a 5D source time estimate."""
    rng = np.random.default_rng(11)
    n_epochs = 3
    info = mne.io.read_info(fname_raw)
    info = mne.pick_info(info, mne.pick_types(info, meg='grad'))
    if src_type == 'vol':
        src = mne.setup_volume_source_space(
            subject='sample', subjects_dir=subjects_dir,
            mri='aseg.mgz', volume_label='Left-Cerebellum-Cortex',
            pos=20, add_interpolator=False)
    else:
        assert src_type == 'surf'
        forward = mne.read_forward_solution(fname_fwd)
        src = forward['src']
    for this_src in src:
        this_src['coord_frame'] = FIFF.FIFFV_COORD_MRI
        this_src['subject_his_id'] = 'sample'
    freqs = np.arange(8, 10)
    times = np.arange(0.1, 0.11, 1 / info['sfreq'])
    data = rng.integers(-1000, 1000, size=(n_epochs, len(info.ch_names),
                                           freqs.size, times.size)) + \
        1j * rng.integers(-1000, 1000, size=(n_epochs, len(info.ch_names),
                                             freqs.size, times.size))
    epochs_tfr = mne.time_frequency.EpochsTFR(
        info, data, times=times, freqs=freqs)
    nuse = sum([this_src['nuse'] for this_src in src])
    stc_data = rng.integers(-1000, 1000, size=(n_epochs, nuse, 3,
                                               freqs.size, times.size)) + \
        1j * rng.integers(-1000, 1000, size=(n_epochs, nuse, 3,
                                             freqs.size, times.size))
    return stc_data, src, epochs_tfr


@pytest.mark.allow_unclosed_pyside2
def test_stc_viewer_io(renderer_interactive_pyvistaqt):
    """Test the input/output of the stc viewer GUI."""
    pytest.importorskip('nibabel')
    pytest.importorskip('dipy')
    from mne.gui._vol_stc import VolSourceEstimateViewer

    stc_data, src, epochs_tfr = _fake_stc()
    with pytest.raises(NotImplementedError,
                       match='surface source estimate '
                             'viewing is not yet supported'):
        VolSourceEstimateViewer(stc_data, inst=epochs_tfr)
    with pytest.raises(NotImplementedError, match='source estimate object'):
        VolSourceEstimateViewer(stc_data, src=src)
    with pytest.raises(ValueError, match='`data` must be an array'):
        VolSourceEstimateViewer('foo', subject='sample',
                                subjects_dir=subjects_dir,
                                src=src, inst=epochs_tfr)
    with pytest.raises(ValueError,
                       match='Number of epochs in `inst` does not match'):
        VolSourceEstimateViewer(stc_data[1:], src=src, inst=epochs_tfr)
    with pytest.raises(RuntimeError,
                       match='ource vertices in `data` do not match '):
        VolSourceEstimateViewer(stc_data[:, :1], subject='sample',
                                subjects_dir=subjects_dir,
                                src=src, inst=epochs_tfr)
    src[0]['coord_frame'] = FIFF.FIFFV_COORD_HEAD
    with pytest.raises(RuntimeError, match='must be in the `mri`'):
        VolSourceEstimateViewer(stc_data, subject='sample',
                                subjects_dir=subjects_dir,
                                src=src, inst=epochs_tfr)
    src[0]['coord_frame'] = FIFF.FIFFV_COORD_MRI

    src[0]['subject_his_id'] = 'foo'
    with pytest.raises(RuntimeError, match='Source space subject'):
        with pytest.warns(RuntimeWarning, match='`pial` surface not found'):
            VolSourceEstimateViewer(stc_data, subject='sample',
                                    subjects_dir=subjects_dir,
                                    src=src, inst=epochs_tfr)

    with pytest.raises(ValueError,
                       match='Frequencies in `inst` do not match'):
        VolSourceEstimateViewer(
            stc_data[:, :, :, 1:], src=src, inst=epochs_tfr)

    with pytest.raises(ValueError, match='Complex data is required'):
        VolSourceEstimateViewer(stc_data.real, src=src, inst=epochs_tfr)

    with pytest.raises(ValueError,
                       match='Times in `inst` do not match'):
        VolSourceEstimateViewer(
            stc_data[:, :, :, :, 1:], src=src, inst=epochs_tfr)


@pytest.mark.allow_unclosed_pyside2
@testing.requires_testing_data
def test_stc_viewer_display(renderer_interactive_pyvistaqt):
    """Test that the stc viewer GUI displays properly."""
    pytest.importorskip('nibabel')
    pytest.importorskip('dipy')
    from mne.gui._vol_stc import VolSourceEstimateViewer

    stc_data, src, epochs_tfr = _fake_stc()
    with pytest.warns(RuntimeWarning, match='`pial` surface not found'):
        viewer = VolSourceEstimateViewer(
            stc_data, subject='sample', subjects_dir=subjects_dir,
            src=src, inst=epochs_tfr)
    # test go to max
    viewer._go_to_extreme_button.click()
    assert_allclose(viewer._ras, [-20, -60, -20], atol=0.01)

    src_coord = viewer._get_src_coord()
    stc_idx = viewer._src_lut[src_coord]

    viewer._epoch_selector.setCurrentText('Epoch 0')
    assert viewer._epoch_idx == 'Epoch 0'

    viewer._freq_slider.setValue(1)
    assert viewer._f_idx == 1

    viewer._time_slider.setValue(2)
    assert viewer._t_idx == 2

    plot_data = np.linalg.norm((stc_data[0] * stc_data[0].conj()).real,
                               axis=1)[stc_idx]
    assert_allclose(plot_data, viewer._stc_plot.get_array())

    # test clicking on stc plot
    _fake_click(viewer._fig, viewer._fig.axes[0],
                (0, 0), xform='data', kind='release')
    assert viewer._t_idx == 0
    assert viewer._f_idx == 0

    # test baseline
    for mode in ('zscore', 'ratio'):
        viewer.set_baseline((0.1, None), mode)

    # done with time-frequency, close
    viewer.close()

    # test time only, not frequencies
    epochs = mne.EpochsArray(epochs_tfr.data[:, :, 0].real, epochs_tfr.info,
                             tmin=epochs_tfr.tmin)
    stc_time_data = stc_data[:, :, :, 0:1].real
    with pytest.warns(RuntimeWarning, match='`pial` surface not found'):
        viewer = VolSourceEstimateViewer(
            stc_time_data, subject='sample',
            subjects_dir=subjects_dir, src=src, inst=epochs)

    # test go to max
    viewer._go_to_extreme_button.click()
    assert_allclose(viewer._ras, [-20, -60, -20], atol=0.01)

    src_coord = viewer._get_src_coord()
    stc_idx = viewer._src_lut[src_coord]

    viewer._epoch_selector.setCurrentText('Epoch 0')
    assert viewer._epoch_idx == 'Epoch 0'

    with pytest.raises(ValueError, match='Source estimate does '
                                         'not contain frequencies'):
        viewer.set_freq(10)

    viewer._time_slider.setValue(2)
    assert viewer._t_idx == 2

    assert_allclose(np.linalg.norm(stc_time_data[0], axis=1)[stc_idx][0],
                    viewer._stc_plot.get_data()[1])
    viewer.close()


@testing.requires_testing_data
def test_stc_viewer_surface(renderer_interactive_pyvistaqt):
    """Test the stc viewer with a surface source space."""
    pytest.importorskip('nibabel')
    pytest.importorskip('dipy')
    from mne.gui._vol_stc import VolSourceEstimateViewer
    stc_data, src, epochs_tfr = _fake_stc(src_type='surf')
    with pytest.raises(RuntimeError, match='not implemented yet'):
        VolSourceEstimateViewer(
            stc_data, subject='sample',
            subjects_dir=subjects_dir, src=src, inst=epochs_tfr)
