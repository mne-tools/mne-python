# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from cycler import cycler
from matplotlib import rc_context
from numpy.testing import assert_allclose

from mne import read_evokeds
from mne.epochs import Epochs
from mne.event import read_events
from mne.io import read_raw_fif
from mne.viz import ClickableImage, add_background_image, mne_analyze_colormap
from mne.viz.ui_events import ColormapRange, link, subscribe
from mne.viz.utils import (
    SelectFromCollection,
    _compute_scalings,
    _fake_click,
    _fake_keypress,
    _fake_scroll,
    _get_color_list,
    _make_event_color_dict,
    _setup_vmin_vmax,
    _validate_if_list_of_axes,
    centers_to_edges,
    compare_fiff,
    concatenate_images,
)

base_dir = Path(__file__).parents[2] / "io" / "tests" / "data"
raw_fname = base_dir / "test_raw.fif"
cov_fname = base_dir / "test-cov.fif"
ev_fname = base_dir / "test_raw-eve.fif"
ave_fname = base_dir / "test-ave.fif"


def test_setup_vmin_vmax_warns():
    """Test that _setup_vmin_vmax warns properly."""
    expected_msg = r"\(min=0.0, max=1\) range.*minimum of data is -1"
    with pytest.warns(UserWarning, match=expected_msg):
        _setup_vmin_vmax(data=[-1, 0], vmin=None, vmax=None, norm=True)


def test_get_color_list():
    """Test getting a colormap from rcParams."""
    with rc_context({"axes.prop_cycle": cycler(color=["#ff0000", "#00ff00"])}):
        colors = _get_color_list()
        assert isinstance(colors, list)
        assert len(colors) == 2
        assert "#ff0000" in colors
        colors_no_red = _get_color_list(remove=("#ff0000",))
        assert "#ff0000" not in colors_no_red
        assert len(colors_no_red) == 1


def test_mne_analyze_colormap():
    """Test mne_analyze_colormap."""
    pytest.raises(ValueError, mne_analyze_colormap, [0])
    pytest.raises(ValueError, mne_analyze_colormap, [-1, 1, 2])
    pytest.raises(ValueError, mne_analyze_colormap, [0, 2, 1])


def test_compare_fiff():
    """Test compare_fiff."""
    compare_fiff(raw_fname, cov_fname, read_limit=0, show=False)
    plt.close("all")


def test_clickable_image():
    """Test the ClickableImage class."""
    # Gen data and create clickable image
    im = np.random.RandomState(0).randn(100, 100)
    clk = ClickableImage(im)
    clicks = [(12, 8), (46, 48), (10, 24)]

    # Generate clicks
    for click in clicks:
        _fake_click(clk.fig, clk.ax, click, xform="data")
    assert_allclose(np.array(clicks), np.array(clk.coords))
    assert len(clicks) == len(clk.coords)

    # Exporting to layout
    lt = clk.to_layout()
    assert lt.pos.shape[0] == len(clicks)
    assert_allclose(lt.pos[1, 0] / lt.pos[2, 0], clicks[1][0] / float(clicks[2][0]))
    clk.plot_clicks()
    plt.close("all")


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
            assert ax_im.get_aspect() == "auto"
            for ax in axs:
                assert ax.get_aspect() == 1
        else:
            # Background with changing aspect
            ax_im_asp = add_background_image(f, im, set_ratios="auto")
            assert ax_im_asp.get_aspect() == "auto"
            for ax in axs:
                assert ax.get_aspect() == "auto"
        plt.close("all")

    # Make sure passing None as image returns None
    f, axs = plt.subplots(1, 2)
    assert add_background_image(f, None) is None
    plt.close("all")


def test_auto_scale():
    """Test auto-scaling of channels for quick plotting."""
    raw = read_raw_fif(raw_fname)
    epochs = Epochs(raw, read_events(ev_fname))
    rand_data = np.random.randn(10, 100)

    for inst in [raw, epochs]:
        scale_grad = 1e10
        scalings_def = dict([("eeg", "auto"), ("grad", scale_grad), ("stim", "auto")])

        # Test for wrong inputs
        with pytest.raises(ValueError, match=r".*scalings.*'foo'.*"):
            inst.plot(scalings="foo")

        # Make sure compute_scalings doesn't change anything not auto
        scalings_new = _compute_scalings(scalings_def, inst)
        assert scale_grad == scalings_new["grad"]
        assert scalings_new["eeg"] != "auto"

    with pytest.raises(ValueError, match="Must supply either Raw or Epochs"):
        _compute_scalings(scalings_def, rand_data)
    epochs = epochs[0].load_data()
    epochs.pick(picks="eeg")


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
    pytest.raises(TypeError, _validate_if_list_of_axes, "error")
    pytest.raises(TypeError, _validate_if_list_of_axes, ["error"] * 2)
    pytest.raises(TypeError, _validate_if_list_of_axes, ax[0])
    pytest.raises(ValueError, _validate_if_list_of_axes, ax, 3)
    ax_flat[2] = 23
    pytest.raises(TypeError, _validate_if_list_of_axes, ax_flat)
    _validate_if_list_of_axes(ax, 4)
    plt.close("all")


def test_centers_to_edges():
    """Test centers_to_edges."""
    assert_allclose(centers_to_edges([0, 1, 2])[0], [-0.5, 0.5, 1.5, 2.5])
    assert_allclose(centers_to_edges([0])[0], [-0.001, 0.001])
    assert_allclose(centers_to_edges([1])[0], [0.999, 1.001])
    assert_allclose(centers_to_edges([1000])[0], [999.0, 1001.0])


def test_event_color_dict():
    """Test handling of event_color."""
    one = _make_event_color_dict("k")
    two = _make_event_color_dict((0, 0, 0))
    three = _make_event_color_dict("#000")
    assert one == two
    assert one == three
    # test dict with integer keys / event name keys
    event_id = dict(foo=1, bar=2)
    one = _make_event_color_dict({1: "r", 2: "b"}, event_id=event_id)
    two = _make_event_color_dict(dict(foo="r", bar="b"), event_id=event_id)
    assert one == two
    # test default value
    one = _make_event_color_dict({1: "r", -1: "b"}, event_id=event_id)
    two = _make_event_color_dict({1: "r", 2: "b"}, event_id=event_id)
    assert one[2] == two[2]
    # test error
    with pytest.raises(KeyError, match="must be strictly positive, or -1"):
        _ = _make_event_color_dict({-2: "r", -1: "b"})


@pytest.mark.parametrize("axis", (0, 1))
@pytest.mark.parametrize("b_h", (2, 4))
@pytest.mark.parametrize("b_w", (3, 5))
@pytest.mark.parametrize("a_h", (2, 4))
@pytest.mark.parametrize("a_w", (3, 5))
def test_concatenate_images(a_w, a_h, b_w, b_h, axis):
    """Test that concat with arbitrary sizes works."""
    a = np.zeros((a_h, a_w, 3))
    b = np.zeros((b_h, b_w, 3))
    img = concatenate_images([a, b], axis=axis)
    if axis == 0:
        want_shape = (a_h + b_h, max(a_w, b_w), 3)
    else:
        want_shape = (max(a_h, b_h), a_w + b_w, 3)
    assert img.shape == want_shape


def test_draggable_colorbar():
    """Test that DraggableColorbar publishes correct UI Events."""
    evokeds = read_evokeds(ave_fname)
    left_auditory = evokeds[0]
    right_auditory = evokeds[1]
    vmin, vmax = -400, 400
    fig = left_auditory.plot_topomap("interactive", vlim=(vmin, vmax))
    fig2 = right_auditory.plot_topomap("interactive", vlim=(vmin, vmax))
    link(fig, fig2)
    callback_calls = []

    def callback(event):
        callback_calls.append(event)

    subscribe(fig, "colormap_range", callback)

    # Test that correct event is published
    _fake_keypress(fig, "down")
    _fake_keypress(fig, "up")
    assert len(callback_calls) == 2
    event = callback_calls.pop()
    assert type(event) is ColormapRange
    # Test that scrolling changes color limits
    _fake_scroll(fig, 10, 10, 1)
    event = callback_calls.pop()
    assert abs(event.fmin) < abs(vmin)
    assert abs(event.fmax) < abs(vmax)
    fmin, fmax = event.fmin, event.fmax
    _fake_scroll(fig, 10, 10, -1)
    event = callback_calls.pop()
    assert abs(event.fmin) > abs(fmin)
    assert abs(event.fmax) > abs(fmax)
    fmin, fmax = event.fmin, event.fmax
    # Test that plus and minus change color limits
    _fake_keypress(fig, "+")
    event = callback_calls.pop()
    assert abs(event.fmin) < abs(fmin)
    assert abs(event.fmax) < abs(fmax)
    fmin, fmax = event.fmin, event.fmax
    _fake_keypress(fig, "-")
    event = callback_calls.pop()
    assert abs(event.fmin) > abs(fmin)
    assert abs(event.fmax) > abs(fmax)
    fmin, fmax = event.fmin, event.fmax
    # Test that page up and page down change color limits
    _fake_keypress(fig, "pageup")
    event = callback_calls.pop()
    assert event.fmin < fmin
    assert event.fmax < fmax
    fmin, fmax = event.fmin, event.fmax
    _fake_keypress(fig, "pagedown")
    event = callback_calls.pop()
    assert event.fmin > fmin
    assert event.fmax > fmax
    # Test that space key resets color limits
    _fake_keypress(fig, " ")
    event = callback_calls.pop()
    assert event.fmax == vmax
    assert event.fmin == vmin
    # Test that colormap change in one figure changes that of another one
    cmap_want = fig.axes[0].CB.cycle[fig.axes[0].CB.index + 1]
    cmap_old = fig.axes[0].CB.mappable.get_cmap().name
    _fake_keypress(fig, "down")
    cmap_new1 = fig.axes[0].CB.mappable.get_cmap().name
    cmap_new2 = fig2.axes[0].CB.mappable.get_cmap().name
    assert cmap_new1 == cmap_new2 == cmap_want != cmap_old


def test_select_from_collection():
    """Test the lasso selector for matplotlib figures."""
    fig, ax = plt.subplots()
    collection = ax.scatter([1, 2, 2, 1], [1, 1, 0, 0], color="black", edgecolor="red")
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 2)
    lasso = SelectFromCollection(ax, collection, names=["A", "B", "C", "D"])
    assert lasso.selection == []

    # Make a selection with no patches inside of it.
    _fake_click(fig, ax, (0, 0), xform="data")
    _fake_click(fig, ax, (0.5, 0), xform="data", kind="motion")
    _fake_click(fig, ax, (0.5, 1), xform="data", kind="motion")
    _fake_click(fig, ax, (0.5, 1), xform="data", kind="release")
    assert lasso.selection == []

    # Doing a single click on a patch should not select it.
    _fake_click(fig, ax, (1, 1), xform="data")
    assert lasso.selection == []

    # Make a selection with two patches in it.
    _fake_click(fig, ax, (0, 0.5), xform="data")
    _fake_click(fig, ax, (3, 0.5), xform="data", kind="motion")
    _fake_click(fig, ax, (3, 1.5), xform="data", kind="motion")
    _fake_click(fig, ax, (0, 1.5), xform="data", kind="motion")
    _fake_click(fig, ax, (0, 0.5), xform="data", kind="motion")
    _fake_click(fig, ax, (0, 0.5), xform="data", kind="release")
    assert lasso.selection == ["A", "B"]

    # Use Control key to lasso an additional patch.
    _fake_keypress(fig, "control")
    _fake_click(fig, ax, (0.5, -0.5), xform="data")
    _fake_click(fig, ax, (1.5, -0.5), xform="data", kind="motion")
    _fake_click(fig, ax, (1.5, 0.5), xform="data", kind="motion")
    _fake_click(fig, ax, (0.5, 0.5), xform="data", kind="motion")
    _fake_click(fig, ax, (0.5, 0.5), xform="data", kind="release")
    _fake_keypress(fig, "control", kind="release")
    assert lasso.selection == ["A", "B", "D"]

    # Use CTRL+SHIFT to remove a patch.
    _fake_keypress(fig, "ctrl+shift")
    _fake_click(fig, ax, (0.5, 0.5), xform="data")
    _fake_click(fig, ax, (1.5, 0.5), xform="data", kind="motion")
    _fake_click(fig, ax, (1.5, 1.5), xform="data", kind="motion")
    _fake_click(fig, ax, (0.5, 1.5), xform="data", kind="motion")
    _fake_click(fig, ax, (0.5, 1.5), xform="data", kind="release")
    _fake_keypress(fig, "ctrl+shift", kind="release")
    assert lasso.selection == ["B", "D"]

    # Check that the two selected patches have a different appearance.
    fc = lasso.collection.get_facecolors()
    ec = lasso.collection.get_edgecolors()
    assert (fc[:, -1] == [0.5, 1.0, 0.5, 1.0]).all()
    assert (ec[:, -1] == [0.25, 1.0, 0.25, 1.0]).all()

    # Test adding and removing single channels.
    lasso.select_one(2)  # should not do anything without modifier keys
    assert lasso.selection == ["B", "D"]
    _fake_keypress(fig, "control")
    lasso.select_one(2)  # add to selection
    _fake_keypress(fig, "control", kind="release")
    assert lasso.selection == ["B", "C", "D"]
    _fake_keypress(fig, "ctrl+shift")
    lasso.select_one(1)  #  remove from selection
    assert lasso.selection == ["C", "D"]
    _fake_keypress(fig, "ctrl+shift", kind="release")
