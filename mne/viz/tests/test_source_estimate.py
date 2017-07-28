# License: Simplified BSD

import os.path as op
import numpy as np
import matplotlib

from mne.utils import run_tests_if_main, requires_nibabel, requires_nilearn
from mne.datasets import testing
from mne import SourceEstimate, read_source_spaces, read_source_estimate

# Set our plotters to test mode
matplotlib.use('Agg')  # for testing don't use X server


@requires_nilearn
@testing.requires_testing_data
@requires_nibabel()
def test_plot_glass_brain():
    """Test plotting stc with nilearn."""
    import matplotlib.pyplot as plt
    data_dir = testing.data_path()
    subjects_dir = op.join(data_dir, 'subjects')
    src_fname = op.join(data_dir, 'subjects', 'sample', 'bem',
                        'sample-oct-6-src.fif')
    fname_vol = op.join(data_dir, 'MEG', 'sample',
                        'sample_audvis_trunc-grad-vol-7-fwd-sensmap-vol.w')
    sample_src = read_source_spaces(src_fname)
    vertices = [s['vertno'] for s in sample_src]
    n_times = 5
    n_verts = sum(len(v) for v in vertices)
    stc_data = np.ones((n_verts * n_times))
    stc_data.shape = (n_verts, n_times)
    stc = SourceEstimate(stc_data, vertices, 1, 1, 'sample')
    # spm dataset has the ribbon.mgz.
    stc.plot_glass_brain('sample', subjects_dir, src=None, initial_time=0.)

    stc = read_source_estimate(fname_vol)  # Volume stc.
    src = read_source_spaces(src_fname)
    stc.plot_glass_brain('sample', subjects_dir, src=src)
    plt.close('all')


run_tests_if_main()
