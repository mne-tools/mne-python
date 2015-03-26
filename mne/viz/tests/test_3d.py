# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#
# License: Simplified BSD

import os.path as op
import warnings

import numpy as np
from numpy.testing import assert_raises

from mne import SourceEstimate
from mne import make_field_map, pick_channels_evoked, read_evokeds
from mne.viz import (plot_sparse_source_estimates, plot_source_estimates,
                     plot_trans, mne_analyze_colormap)
from mne.utils import requires_mayavi, requires_pysurfer
from mne.datasets import testing
from mne.source_space import read_source_spaces

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

data_dir = testing.data_path(download=False)
subjects_dir = op.join(data_dir, 'subjects')
trans_fname = op.join(data_dir, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')
src_fname = op.join(data_dir, 'subjects', 'sample',
                    'bem', 'sample-oct-6-src.fif')

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
    stc_data[(np.random.rand(20) * n_verts * n_time).astype(int)] = 1
    stc_data.shape = (n_verts, n_time)
    stc = SourceEstimate(stc_data, vertices, 1, 1)
    colormap = mne_analyze_colormap(format='matplotlib')
    # don't really need to test matplotlib method since it's not used now...
    colormap = mne_analyze_colormap()
    plot_source_estimates(stc, 'sample', colormap=colormap,
                          config_opts={'background': (1, 1, 0)},
                          subjects_dir=subjects_dir, colorbar=True)
    assert_raises(TypeError, plot_source_estimates, stc, 'sample',
                  figure='foo', hemi='both')

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
