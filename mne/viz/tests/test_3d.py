# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#          Mark Wronkiewicz <wronk.mark@gmail.com>
#
# License: Simplified BSD

import os.path as op
import warnings

import numpy as np
from numpy.testing import assert_raises, assert_equal

from mne import (make_field_map, pick_channels_evoked, read_evokeds,
                 read_trans, read_dipole, SourceEstimate)
from mne.viz import (plot_sparse_source_estimates, plot_source_estimates,
                     plot_trans)
from mne.utils import requires_mayavi, requires_pysurfer, run_tests_if_main
from mne.datasets import testing
from mne.source_space import read_source_spaces


# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

data_dir = testing.data_path(download=False)
subjects_dir = op.join(data_dir, 'subjects')
trans_fname = op.join(data_dir, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')
src_fname = op.join(data_dir, 'subjects', 'sample', 'bem',
                    'sample-oct-6-src.fif')
dip_fname = op.join(data_dir, 'MEG', 'sample', 'sample_audvis_trunc_set1.dip')

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
evoked_fname = op.join(base_dir, 'test-ave.fif')

warnings.simplefilter('always')  # enable b/c these tests throw warnings


@testing.requires_testing_data
@requires_pysurfer
@requires_mayavi
def test_plot_sparse_source_estimates():
    """Test plotting of (sparse) source estimates
    """
    sample_src = read_source_spaces(src_fname)

    # dense version
    vertices = [s['vertno'] for s in sample_src]
    n_time = 5
    n_verts = sum(len(v) for v in vertices)
    stc_data = np.zeros((n_verts * n_time))
    stc_size = stc_data.size
    stc_data[(np.random.rand(stc_size / 20) * stc_size).astype(int)] = \
        np.random.RandomState(0).rand(stc_data.size / 20)
    stc_data.shape = (n_verts, n_time)
    stc = SourceEstimate(stc_data, vertices, 1, 1)
    colormap = 'mne_analyze'
    plot_source_estimates(stc, 'sample', colormap=colormap,
                          config_opts={'background': (1, 1, 0)},
                          subjects_dir=subjects_dir, colorbar=True,
                          clim='auto')
    assert_raises(TypeError, plot_source_estimates, stc, 'sample',
                  figure='foo', hemi='both', clim='auto')

    # now do sparse version
    vertices = sample_src[0]['vertno']
    inds = [111, 333]
    stc_data = np.zeros((len(inds), n_time))
    stc_data[0, 1] = 1.
    stc_data[1, 4] = 2.
    vertices = [vertices[inds], np.empty(0, dtype=np.int)]
    stc = SourceEstimate(stc_data, vertices, 1, 1)
    plot_sparse_source_estimates(sample_src, stc, bgcolor=(1, 1, 1),
                                 opacity=0.5, high_resolution=False)


@testing.requires_testing_data
@requires_mayavi
def test_plot_evoked_field():
    """Test plotting evoked field
    """
    evoked = read_evokeds(evoked_fname, condition='Left Auditory',
                          baseline=(-0.2, 0.0))
    evoked = pick_channels_evoked(evoked, evoked.ch_names[::10])  # speed
    for t in ['meg', None]:
        maps = make_field_map(evoked, trans_fname, subject='sample',
                              subjects_dir=subjects_dir, n_jobs=1, ch_type=t)

        evoked.plot_field(maps, time=0.1)


@testing.requires_testing_data
@requires_mayavi
def test_plot_trans():
    """Test plotting of -trans.fif files
    """
    evoked = read_evokeds(evoked_fname, condition='Left Auditory',
                          baseline=(-0.2, 0.0))
    plot_trans(evoked.info, trans_fname, subject='sample',
               subjects_dir=subjects_dir)
    assert_raises(ValueError, plot_trans, evoked.info, trans_fname,
                  subject='sample', subjects_dir=subjects_dir,
                  ch_type='bad-chtype')


@testing.requires_testing_data
@requires_pysurfer
@requires_mayavi
def test_limits_to_control_points():
    """Test functionality for determing control points
    """
    sample_src = read_source_spaces(src_fname)

    vertices = [s['vertno'] for s in sample_src]
    n_time = 5
    n_verts = sum(len(v) for v in vertices)
    stc_data = np.random.RandomState(0).rand((n_verts * n_time))
    stc_data.shape = (n_verts, n_time)
    stc = SourceEstimate(stc_data, vertices, 1, 1, 'sample')

    # Test for simple use cases
    from mayavi import mlab
    stc.plot(subjects_dir=subjects_dir)
    stc.plot(clim=dict(pos_lims=(10, 50, 90)), subjects_dir=subjects_dir)
    stc.plot(clim=dict(kind='value', lims=(10, 50, 90)), figure=99,
             subjects_dir=subjects_dir)
    stc.plot(colormap='hot', clim='auto', subjects_dir=subjects_dir)
    stc.plot(colormap='mne', clim='auto', subjects_dir=subjects_dir)
    figs = [mlab.figure(), mlab.figure()]
    assert_raises(RuntimeError, stc.plot, clim='auto', figure=figs,
                  subjects_dir=subjects_dir)

    # Test both types of incorrect limits key (lims/pos_lims)
    assert_raises(KeyError, plot_source_estimates, stc, colormap='mne',
                  clim=dict(kind='value', lims=(5, 10, 15)),
                  subjects_dir=subjects_dir)
    assert_raises(KeyError, plot_source_estimates, stc, colormap='hot',
                  clim=dict(kind='value', pos_lims=(5, 10, 15)),
                  subjects_dir=subjects_dir)

    # Test for correct clim values
    assert_raises(ValueError, stc.plot,
                  clim=dict(kind='value', pos_lims=[0, 1, 0]),
                  subjects_dir=subjects_dir)
    assert_raises(ValueError, stc.plot, colormap='mne',
                  clim=dict(pos_lims=(5, 10, 15, 20)),
                  subjects_dir=subjects_dir)
    assert_raises(ValueError, stc.plot,
                  clim=dict(pos_lims=(5, 10, 15), kind='foo'),
                  subjects_dir=subjects_dir)
    assert_raises(ValueError, stc.plot, colormap='mne', clim='foo',
                  subjects_dir=subjects_dir)
    assert_raises(ValueError, stc.plot, clim=(5, 10, 15),
                  subjects_dir=subjects_dir)
    assert_raises(ValueError, plot_source_estimates, 'foo', clim='auto',
                  subjects_dir=subjects_dir)
    assert_raises(ValueError, stc.plot, hemi='foo', clim='auto',
                  subjects_dir=subjects_dir)

    # Test handling of degenerate data
    stc.plot(clim=dict(kind='value', lims=[0, 0, 1]),
             subjects_dir=subjects_dir)  # ok
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # thresholded maps
        stc._data.fill(1.)
        plot_source_estimates(stc, subjects_dir=subjects_dir)
        assert_equal(len(w), 0)
        stc._data[0].fill(0.)
        plot_source_estimates(stc, subjects_dir=subjects_dir)
        assert_equal(len(w), 0)
        stc._data.fill(0.)
        plot_source_estimates(stc, subjects_dir=subjects_dir)
        assert_equal(len(w), 1)
    mlab.close()


@testing.requires_testing_data
@requires_mayavi
def test_plot_dipole_locations():
    """Test plotting dipole locations
    """
    dipoles = read_dipole(dip_fname)
    trans = read_trans(trans_fname)
    dipoles.plot_locations(trans, 'sample', subjects_dir, fig_name='foo')
    assert_raises(ValueError, dipoles.plot_locations, trans, 'sample',
                  subjects_dir, mode='foo')


run_tests_if_main()
