# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: Simplified BSD

import os.path as op
import warnings
import numpy as np
from nose.tools import assert_true, assert_raises
from numpy.testing import assert_allclose

from mne.viz.utils import (compare_fiff, _fake_click, _compute_scalings,
                           _validate_if_list_of_axes)
from mne.viz import ClickableImage, add_background_image, mne_analyze_colormap
from mne.utils import run_tests_if_main
from mne.io import read_raw_fif
from mne.event import read_events
from mne.epochs import Epochs

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

warnings.simplefilter('always')  # enable b/c these tests throw warnings

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
cov_fname = op.join(base_dir, 'test-cov.fif')
ev_fname = op.join(base_dir, 'test_raw-eve.fif')


def test_mne_analyze_colormap():
    """Test mne_analyze_colormap."""
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

    # Make sure passing None as image returns None
    assert_true(add_background_image(f, None) is None)


def test_auto_scale():
    """Test auto-scaling of channels for quick plotting."""
    raw = read_raw_fif(raw_fname)
    epochs = Epochs(raw, read_events(ev_fname))
    rand_data = np.random.randn(10, 100)

    for inst in [raw, epochs]:
        scale_grad = 1e10
        scalings_def = dict([('eeg', 'auto'), ('grad', scale_grad),
                             ('stim', 'auto')])

        # Test for wrong inputs
        assert_raises(ValueError, inst.plot, scalings='foo')
        assert_raises(ValueError, _compute_scalings, 'foo', inst)

        # Make sure compute_scalings doesn't change anything not auto
        scalings_new = _compute_scalings(scalings_def, inst)
        assert_true(scale_grad == scalings_new['grad'])
        assert_true(scalings_new['eeg'] != 'auto')

    assert_raises(ValueError, _compute_scalings, scalings_def, rand_data)
    epochs = epochs[0].load_data()
    epochs.pick_types(eeg=True, meg=False)
    assert_raises(ValueError, _compute_scalings,
                  dict(grad='auto'), epochs)


def test_validate_if_list_of_axes():
    """Test validation of axes."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2)
    assert_raises(ValueError, _validate_if_list_of_axes, ax)
    ax_flat = ax.ravel()
    ax = ax.ravel().tolist()
    _validate_if_list_of_axes(ax_flat)
    _validate_if_list_of_axes(ax_flat, 4)
    assert_raises(ValueError, _validate_if_list_of_axes, ax_flat, 5)
    assert_raises(ValueError, _validate_if_list_of_axes, ax, 3)
    assert_raises(ValueError, _validate_if_list_of_axes, 'error')
    assert_raises(ValueError, _validate_if_list_of_axes, ['error'] * 2)
    assert_raises(ValueError, _validate_if_list_of_axes, ax[0])
    assert_raises(ValueError, _validate_if_list_of_axes, ax, 3)
    ax_flat[2] = 23
    assert_raises(ValueError, _validate_if_list_of_axes, ax_flat)
    _validate_if_list_of_axes(ax, 4)


run_tests_if_main()
