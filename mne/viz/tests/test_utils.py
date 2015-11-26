# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: Simplified BSD

import os.path as op
import warnings
import numpy as np
from nose.tools import assert_true, assert_raises
from numpy.testing import assert_allclose

from mne.viz.utils import compare_fiff, _fake_click
from mne.viz import ClickableImage, add_background_image, mne_analyze_colormap
from mne.utils import run_tests_if_main

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

warnings.simplefilter('always')  # enable b/c these tests throw warnings

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
cov_fname = op.join(base_dir, 'test-cov.fif')


def test_mne_analyze_colormap():
    """Test mne_analyze_colormap
    """
    assert_raises(ValueError, mne_analyze_colormap, [0])
    assert_raises(ValueError, mne_analyze_colormap, [-1, 1, 2])
    assert_raises(ValueError, mne_analyze_colormap, [0, 2, 1])


def test_compare_fiff():
    import matplotlib.pyplot as plt
    compare_fiff(raw_fname, cov_fname, read_limit=0, show=False)
    plt.close('all')


def test_clickable_image():
    """Test the ClickableImage class."""
    # Gen data and create clickable image
    import matplotlib.pyplot as plt
    im = np.random.RandomState(0).randn(100, 100)
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
    clk.plot_clicks()
    plt.close('all')


def test_add_background_image():
    """Test adding background image to a figure."""
    import matplotlib.pyplot as plt
    rng = np.random.RandomState(0)
    f, axs = plt.subplots(1, 2)
    x, y = rng.randn(2, 10)
    im = rng.randn(10, 10)
    axs[0].scatter(x, y)
    axs[1].scatter(y, x)
    for ax in axs:
        ax.set_aspect(1)

    # Background without changing aspect
    ax_im = add_background_image(f, im)
    assert_true(ax_im.get_aspect() == 'auto')
    for ax in axs:
        assert_true(ax.get_aspect() == 1)

    # Background with changing aspect
    ax_im_asp = add_background_image(f, im, set_ratios='auto')
    assert_true(ax_im_asp.get_aspect() == 'auto')
    for ax in axs:
        assert_true(ax.get_aspect() == 'auto')


run_tests_if_main()
