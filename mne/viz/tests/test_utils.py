# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: Simplified BSD

import os.path as op

import numpy as np
from numpy.testing import assert_allclose
import pytest
import matplotlib.pyplot as plt

from mne.viz.utils import (compare_fiff, _fake_click, _compute_scalings,
                           _validate_if_list_of_axes, _get_color_list,
                           _setup_vmin_vmax, centers_to_edges,
                           _make_event_color_dict)
from mne.viz import ClickableImage, add_background_image, mne_analyze_colormap
from mne.io import read_raw_fif
from mne.event import read_events
from mne.epochs import Epochs

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
cov_fname = op.join(base_dir, 'test-cov.fif')
ev_fname = op.join(base_dir, 'test_raw-eve.fif')


def test_setup_vmin_vmax_warns():
    """Test that _setup_vmin_vmax warns properly."""
    expected_msg = r'\(min=0.0, max=1\) range.*minimum of data is -1'
    with pytest.warns(UserWarning, match=expected_msg):
        _setup_vmin_vmax(data=[-1, 0], vmin=None, vmax=None, norm=True)


def test_get_color_list():
    """Test getting a colormap from rcParams."""
    colors = _get_color_list()
    assert isinstance(colors, list)
    colors_no_red = _get_color_list(annotations=True)
    assert '#ff0000' not in colors_no_red


def test_mne_analyze_colormap():
    """Test mne_analyze_colormap."""
    pytest.raises(ValueError, mne_analyze_colormap, [0])
    pytest.raises(ValueError, mne_analyze_colormap, [-1, 1, 2])
    pytest.raises(ValueError, mne_analyze_colormap, [0, 2, 1])


def test_compare_fiff():
    """Test compare_fiff."""
    compare_fiff(raw_fname, cov_fname, read_limit=0, show=False)
    plt.close('all')


def test_clickable_image():
    """Test the ClickableImage class."""
    # Gen data and create clickable image
    im = np.random.RandomState(0).randn(100, 100)
    clk = ClickableImage(im)
    clicks = [(12, 8), (46, 48), (10, 24)]

    # Generate clicks
    for click in clicks:
        _fake_click(clk.fig, clk.ax, click, xform='data')
    assert_allclose(np.array(clicks), np.array(clk.coords))
    assert (len(clicks) == len(clk.coords))

    # Exporting to layout
    lt = clk.to_layout()
    assert (lt.pos.shape[0] == len(clicks))
    assert_allclose(lt.pos[1, 0] / lt.pos[2, 0],
                    clicks[1][0] / float(clicks[2][0]))
    clk.plot_clicks()
    plt.close('all')


def test_add_background_image():
    """Test adding background image to a figure."""
    rng = np.random.RandomState(0)
    for ii in range(2):
        f, axs = plt.subplots(1, 2)
        x, y = rng.randn(2, 10)
        im = rng.randn(10, 10)
        axs[0].scatter(x, y)
        axs[1].scatter(y, x)
        for ax in axs:
            ax.set_aspect(1)

        # Background without changing aspect
        if ii == 0:
            ax_im = add_background_image(f, im)
            return
            assert (ax_im.get_aspect() == 'auto')
            for ax in axs:
                assert (ax.get_aspect() == 1)
        else:
            # Background with changing aspect
            ax_im_asp = add_background_image(f, im, set_ratios='auto')
            assert (ax_im_asp.get_aspect() == 'auto')
            for ax in axs:
                assert (ax.get_aspect() == 'auto')
        plt.close('all')

    # Make sure passing None as image returns None
    f, axs = plt.subplots(1, 2)
    assert (add_background_image(f, None) is None)
    plt.close('all')


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
        with pytest.raises(ValueError, match=r".*scalings.*'foo'.*"):
            inst.plot(scalings='foo')

        # Make sure compute_scalings doesn't change anything not auto
        scalings_new = _compute_scalings(scalings_def, inst)
        assert (scale_grad == scalings_new['grad'])
        assert (scalings_new['eeg'] != 'auto')

    with pytest.raises(ValueError, match='Must supply either Raw or Epochs'):
        _compute_scalings(scalings_def, rand_data)
    epochs = epochs[0].load_data()
    epochs.pick_types(eeg=True, meg=False)


def test_validate_if_list_of_axes():
    """Test validation of axes."""
    fig, ax = plt.subplots(2, 2)
    pytest.raises(ValueError, _validate_if_list_of_axes, ax)
    ax_flat = ax.ravel()
    ax = ax.ravel().tolist()
    _validate_if_list_of_axes(ax_flat)
    _validate_if_list_of_axes(ax_flat, 4)
    pytest.raises(ValueError, _validate_if_list_of_axes, ax_flat, 5)
    pytest.raises(ValueError, _validate_if_list_of_axes, ax, 3)
    pytest.raises(ValueError, _validate_if_list_of_axes, 'error')
    pytest.raises(ValueError, _validate_if_list_of_axes, ['error'] * 2)
    pytest.raises(ValueError, _validate_if_list_of_axes, ax[0])
    pytest.raises(ValueError, _validate_if_list_of_axes, ax, 3)
    ax_flat[2] = 23
    pytest.raises(ValueError, _validate_if_list_of_axes, ax_flat)
    _validate_if_list_of_axes(ax, 4)
    plt.close('all')


def test_centers_to_edges():
    """Test centers_to_edges."""
    assert_allclose(centers_to_edges([0, 1, 2])[0], [-0.5, 0.5, 1.5, 2.5])
    assert_allclose(centers_to_edges([0])[0], [-0.001, 0.001])
    assert_allclose(centers_to_edges([1])[0], [0.999, 1.001])
    assert_allclose(centers_to_edges([1000])[0], [999., 1001.])


def test_event_color_dict():
    """Test handling of event_color."""
    one = _make_event_color_dict('k')
    two = _make_event_color_dict((0, 0, 0))
    three = _make_event_color_dict('#000')
    assert one == two
    assert one == three
    # test dict with integer keys / event name keys
    event_id = dict(foo=1, bar=2)
    one = _make_event_color_dict({1: 'r', 2: 'b'}, event_id=event_id)
    two = _make_event_color_dict(dict(foo='r', bar='b'), event_id=event_id)
    assert one == two
    # test default value
    one = _make_event_color_dict({1: 'r', -1: 'b'}, event_id=event_id)
    two = _make_event_color_dict({1: 'r', 2: 'b'}, event_id=event_id)
    assert one[2] == two[2]
    # test error
    with pytest.raises(KeyError, match='must be strictly positive, or -1'):
        _ = _make_event_color_dict({-2: 'r', -1: 'b'})
