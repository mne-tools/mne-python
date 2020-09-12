# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#          Mark Wronkiewicz <wronk.mark@gmail.com>
#
# License: Simplified BSD

import os.path as op

import numpy as np
import pytest

from mne import (read_forward_solution, VolSourceEstimate, SourceEstimate,
                 VolVectorSourceEstimate, compute_source_morph)
from mne.datasets import testing
from mne.utils import (requires_dipy, requires_nibabel, requires_version,
                       catch_logging)
from mne.viz import plot_volume_source_estimates
from mne.viz.utils import _fake_click

data_dir = testing.data_path(download=False)
subjects_dir = op.join(data_dir, 'subjects')
fwd_fname = op.join(data_dir, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-vol-7-fwd.fif')


@pytest.mark.slowtest  # can be slow on OSX
@testing.requires_testing_data
@requires_dipy()
@requires_nibabel()
@requires_version('nilearn', '0.4')
@pytest.mark.parametrize(
    'mode, stype, init_t, want_t, init_p, want_p, bg_img', [
        ('glass_brain', 's', None, 2, None, (-30.9, 18.4, 56.7), None),
        ('stat_map', 'vec', 1, 1, None, (15.7, 16.0, -6.3), None),
        ('glass_brain', 'vec', None, 1, (10, -10, 20), (6.6, -9., 19.9), None),
        ('stat_map', 's', 1, 1, (-10, 5, 10), (-12.3, 2.0, 7.7), 'brain.mgz')])
def test_plot_volume_source_estimates(mode, stype, init_t, want_t,
                                      init_p, want_p, bg_img):
    """Test interactive plotting of volume source estimates."""
    forward = read_forward_solution(fwd_fname)
    sample_src = forward['src']
    if init_p is not None:
        init_p = np.array(init_p) / 1000.

    vertices = [s['vertno'] for s in sample_src]
    n_verts = sum(len(v) for v in vertices)
    n_time = 2
    data = np.random.RandomState(0).rand(n_verts, n_time)

    if stype == 'vec':
        stc = VolVectorSourceEstimate(
            np.tile(data[:, np.newaxis], (1, 3, 1)), vertices, 1, 1)
    else:
        assert stype == 's'
        stc = VolSourceEstimate(data, vertices, 1, 1)
    with pytest.warns(None):  # sometimes get scalars/index warning
        with catch_logging() as log:
            fig = stc.plot(
                sample_src, subject='sample', subjects_dir=subjects_dir,
                mode=mode, initial_time=init_t, initial_pos=init_p,
                bg_img=bg_img, verbose=True)
    log = log.getvalue()
    want_str = 't = %0.3f s' % want_t
    assert want_str in log, (want_str, init_t)
    want_str = '(%0.1f, %0.1f, %0.1f) mm' % want_p
    assert want_str in log, (want_str, init_p)
    for ax_idx in [0, 2, 3, 4]:
        _fake_click(fig, fig.axes[ax_idx], (0.3, 0.5))
    fig.canvas.key_press_event('left')
    fig.canvas.key_press_event('shift+right')
    if bg_img is not None:
        with pytest.raises(FileNotFoundError, match='MRI file .* not found'):
            stc.plot(sample_src, subject='sample', subjects_dir=subjects_dir,
                     mode='stat_map', bg_img='junk.mgz')


@pytest.mark.slowtest  # can be slow on OSX
@testing.requires_testing_data
@requires_dipy()
@requires_nibabel()
@requires_version('nilearn', '0.4')
def test_plot_volume_source_estimates_morph():
    """Test interactive plotting of volume source estimates with morph."""
    forward = read_forward_solution(fwd_fname)
    sample_src = forward['src']
    vertices = [s['vertno'] for s in sample_src]
    n_verts = sum(len(v) for v in vertices)
    n_time = 2
    data = np.random.RandomState(0).rand(n_verts, n_time)
    stc = VolSourceEstimate(data, vertices, 1, 1)
    sample_src[0]['subject_his_id'] = 'sample'  # old src
    morph = compute_source_morph(sample_src, 'sample', 'fsaverage', zooms=5,
                                 subjects_dir=subjects_dir)
    initial_pos = (-0.05, -0.01, -0.006)
    with pytest.warns(None):  # sometimes get scalars/index warning
        with catch_logging() as log:
            stc.plot(morph, subjects_dir=subjects_dir, mode='glass_brain',
                     initial_pos=initial_pos, verbose=True)
    log = log.getvalue()
    assert 't = 1.000 s' in log
    assert '(-52.0, -8.0, -7.0) mm' in log

    with pytest.raises(ValueError, match='Allowed values are'):
        stc.plot(sample_src, 'sample', subjects_dir, mode='abcd')
    vertices.append([])
    surface_stc = SourceEstimate(data, vertices, 1, 1)
    with pytest.raises(TypeError, match='an instance of VolSourceEstimate'):
        plot_volume_source_estimates(surface_stc, sample_src, 'sample',
                                     subjects_dir)
    with pytest.raises(ValueError, match='Negative colormap limits'):
        stc.plot(sample_src, 'sample', subjects_dir,
                 clim=dict(lims=[-1, 2, 3], kind='value'))
