# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: Simplified BSD

import os.path as op
import warnings

from mne.viz.utils import compare_fiff


warnings.simplefilter('always')  # enable b/c these tests throw warnings

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server
import matplotlib.pyplot as plt


base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
cov_fname = op.join(base_dir, 'test-cov.fif')


def test_compare_fiff():
    """Test comparing fiff files
    """
    compare_fiff(raw_fname, cov_fname, read_limit=0, show=False)
    plt.close('all')
