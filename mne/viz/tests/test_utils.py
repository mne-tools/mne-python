# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: Simplified BSD

import os.path as op
import warnings
import numpy as np

from mne.viz.utils import compare_fiff, ClickableImage
from numpy.testing import assert_allclose
from nose.tools import assert_true
from mne.viz.utils import _fake_click

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

warnings.simplefilter('always')  # enable b/c these tests throw warnings

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
cov_fname = op.join(base_dir, 'test-cov.fif')


def test_compare_fiff():
    import matplotlib.pyplot as plt
    compare_fiff(raw_fname, cov_fname, read_limit=0, show=False)
    plt.close('all')


def test_clickable_image():
    """Test the ClickableImage class."""
    # Gen data and create clickable image
    im = np.random.randn(100, 100)
    clk = ClickableImage(im)
    clicks = [(12, 8), (46, 48), (10, 24)]

    # Generate clicks
    for click in clicks:
        _fake_click(clk.fig, clk.ax, click, xform='data')
    assert_allclose(np.array(clicks), np.array(clk.coords))
    assert_true(len(clicks) == len(clk.coords))

    # Exporting to layout
    lt = clk.to_layout()
    assert_true(lt.pos.shape[0] == len(clicks))
    assert_allclose(lt.pos[1, 0] / lt.pos[2, 0],
                    clicks[1][0] / float(clicks[2][0]))
