# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)

from mne import pick_info, pick_types
from mne._fiff.constants import FIFF
from mne._fiff.meas_info import _empty_info
from mne.channels import (
    Layout,
    find_layout,
    make_eeg_layout,
    make_grid_layout,
    read_layout,
)
from mne.channels.layout import _box_size, _find_topomap_coords, generate_2d_layout
from mne.defaults import HEAD_SIZE_DEFAULT
from mne.io import read_info, read_raw_kit

io_dir = Path(__file__).parents[2] / "io"
fif_fname = io_dir / "tests" / "data" / "test_raw.fif"
lout_path = io_dir / "tests" / "data"
bti_dir = io_dir / "bti" / "tests" / "data"
fname_ctf_raw = io_dir / "tests" / "data" / "test_ctf_comp_raw.fif"
fname_kit_157 = io_dir / "kit" / "tests" / "data" / "test.sqd"
fname_kit_umd = io_dir / "kit" / "tests" / "data" / "test_umd-raw.sqd"


def _get_test_info():
    """Make test info."""
    test_info = _empty_info(1000)
    loc = np.array(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )
    test_info["chs"] = [
        {
            "cal": 1,
            "ch_name": "ICA 001",
            "coil_type": 0,
            "coord_frame": 0,
            "kind": 502,
            "loc": loc.copy(),
            "logno": 1,
            "range": 1.0,
            "scanno": 1,
            "unit": -1,
            "unit_mul": 0,
        },
        {
            "cal": 1,
            "ch_name": "ICA 002",
            "coil_type": 0,
            "coord_frame": 0,
            "kind": 502,
            "loc": loc.copy(),
            "logno": 2,
            "range": 1.0,
            "scanno": 2,
            "unit": -1,
            "unit_mul": 0,
        },
        {
            "cal": 0.002142000012099743,
            "ch_name": "EOG 061",
            "coil_type": 1,
            "coord_frame": 0,
            "kind": 202,
            "loc": loc.copy(),
            "logno": 61,
            "range": 1.0,
            "scanno": 376,
            "unit": 107,
            "unit_mul": 0,
        },
    ]
    test_info._unlocked = False
    test_info._update_redundant()
    test_info._check_consistency()
    return test_info


@pytest.fixture(scope="module")
def layout():
    """Get a layout."""
    return Layout(
        (0.1, 0.2, 0.1, 1.2),
        pos=np.array([[0, 0, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1], [0.4, 0.4, 0.1, 0.1]]),
        names=["0", "1", "2"],
        ids=[0, 1, 2],
        kind="test",
    )


def test_io_layout_lout(tmp_path):
    """Test IO with .lout files."""
    layout = read_layout(fname="Vectorview-all", scale=False)
    layout.save(tmp_path / "foobar.lout", overwrite=True)
    layout_read = read_layout(
        fname=tmp_path / "foobar.lout",
        scale=False,
    )
    assert_array_almost_equal(layout.pos, layout_read.pos, decimal=2)
    assert layout.names == layout_read.names
    assert "<Layout |" in layout.__repr__()


def test_io_layout_lay(tmp_path):
    """Test IO with .lay files."""
    layout = read_layout(fname="CTF151", scale=False)
    layout.save(str(tmp_path / "foobar.lay"))
    layout_read = read_layout(fname=tmp_path / "foobar.lay", scale=False)
    assert_array_almost_equal(layout.pos, layout_read.pos, decimal=2)
    assert layout.names == layout_read.names


def test_find_topomap_coords():
    """Test mapping of coordinates in 3D space to 2D."""
    info = read_info(fif_fname)
    picks = pick_types(info, meg=False, eeg=True, eog=False, stim=False)

    # Remove extra digitization point, so EEG digitization points match up
    # with the EEG channels
    del info["dig"][85]

    # Use channel locations
    kwargs = dict(ignore_overlap=False, to_sphere=True, sphere=HEAD_SIZE_DEFAULT)
    l0 = _find_topomap_coords(info, picks, **kwargs)

    # Remove electrode position information, use digitization points from now
    # on.
    for ch in info["chs"]:
        ch["loc"].fill(np.nan)

    l1 = _find_topomap_coords(info, picks, **kwargs)
    assert_allclose(l1, l0, atol=1e-3)

    for z_pt in ((HEAD_SIZE_DEFAULT, 0.0, 0.0), (0.0, HEAD_SIZE_DEFAULT, 0.0)):
        info["dig"][-1]["r"] = np.array(z_pt)
        l1 = _find_topomap_coords(info, picks, **kwargs)
        assert_allclose(l1[-1], z_pt[:2], err_msg="Z=0 point moved", atol=1e-6)

    # Test plotting mag topomap without channel locations: it should fail
    mag_picks = pick_types(info, meg="mag")
    with pytest.raises(ValueError, match="Cannot determine location"):
        _find_topomap_coords(info, mag_picks, **kwargs)

    # Test function with too many EEG digitization points: it should fail
    info["dig"].append({"r": [1, 2, 3], "kind": FIFF.FIFFV_POINT_EEG})
    with pytest.raises(ValueError, match="Number of EEG digitization points"):
        _find_topomap_coords(info, picks, **kwargs)

    # Test function with too little EEG digitization points: it should fail
    info._unlocked = True
    info["dig"] = info["dig"][:-2]
    with pytest.raises(ValueError, match="Number of EEG digitization points"):
        _find_topomap_coords(info, picks, **kwargs)

    # Electrode positions must be unique
    info["dig"].append(info["dig"][-1])
    with pytest.raises(ValueError, match="overlapping positions"):
        _find_topomap_coords(info, picks, **kwargs)

    # Test function without EEG digitization points: it should fail
    info["dig"] = [d for d in info["dig"] if d["kind"] != FIFF.FIFFV_POINT_EEG]
    with pytest.raises(RuntimeError, match="Did not find any digitization"):
        _find_topomap_coords(info, picks, **kwargs)

    # Test function without any digitization points, it should fail
    info["dig"] = None
    with pytest.raises(RuntimeError, match="No digitization points found"):
        _find_topomap_coords(info, picks, **kwargs)
    info["dig"] = []
    with pytest.raises(RuntimeError, match="No digitization points found"):
        _find_topomap_coords(info, picks, **kwargs)


def test_make_eeg_layout(tmp_path):
    """Test creation of EEG layout."""
    lout_orig = read_layout(fname=lout_path / "test_raw.lout")
    info = read_info(fif_fname)
    info["bads"].append(info["ch_names"][360])
    layout = make_eeg_layout(info, exclude=[])
    assert_array_equal(
        len(layout.names),
        len([ch for ch in info["ch_names"] if ch.startswith("EE")]),
    )
    layout.save(str(tmp_path / "foo.lout"))
    lout_new = read_layout(fname=tmp_path / "foo.lout", scale=False)
    assert_array_equal(lout_new.kind, "foo")
    assert_allclose(layout.pos, lout_new.pos, atol=0.1)
    assert_array_equal(lout_orig.names, lout_new.names)

    # Test input validation
    pytest.raises(ValueError, make_eeg_layout, info, radius=-0.1)
    pytest.raises(ValueError, make_eeg_layout, info, radius=0.6)
    pytest.raises(ValueError, make_eeg_layout, info, width=-0.1)
    pytest.raises(ValueError, make_eeg_layout, info, width=1.1)
    pytest.raises(ValueError, make_eeg_layout, info, height=-0.1)
    pytest.raises(ValueError, make_eeg_layout, info, height=1.1)


def test_make_grid_layout(tmp_path):
    """Test creation of grid layout."""
    lout_orig = read_layout(fname=lout_path / "test_ica.lout")
    layout = make_grid_layout(_get_test_info())
    layout.save(str(tmp_path / "bar.lout"))
    lout_new = read_layout(fname=tmp_path / "bar.lout")
    assert_array_equal(lout_new.kind, "bar")
    assert_array_equal(lout_orig.pos, lout_new.pos)
    assert_array_equal(lout_orig.names, lout_new.names)

    # Test creating grid layout with specified number of columns
    layout = make_grid_layout(_get_test_info(), n_col=2)
    # Vertical positions should be equal
    assert layout.pos[0, 1] == layout.pos[1, 1]
    # Horizontal positions should be unequal
    assert layout.pos[0, 0] != layout.pos[1, 0]
    # Box sizes should be equal
    assert_array_equal(layout.pos[0, 3:], layout.pos[1, 3:])


def test_find_layout():
    """Test finding layout."""
    with pytest.raises(ValueError, match="Invalid value for the 'ch_type'"):
        find_layout(_get_test_info(), ch_type="meep")

    sample_info = read_info(fif_fname)
    sample_info2 = pick_info(sample_info, pick_types(sample_info, meg="grad"))
    sample_info3 = pick_info(sample_info, pick_types(sample_info, meg="mag"))
    sample_info4 = copy.deepcopy(sample_info)
    for ii, name in enumerate(sample_info4["ch_names"]):  # mock new convention
        new = name.replace(" ", "")
        sample_info4["chs"][ii]["ch_name"] = new
    sample_info5 = pick_info(sample_info, pick_types(sample_info, meg=False, eeg=True))

    lout = find_layout(sample_info, ch_type=None)
    assert lout.kind == "Vectorview-all"
    assert all(" " in k for k in lout.names)

    lout = find_layout(sample_info2, ch_type="meg")
    assert_equal(lout.kind, "Vectorview-all")

    # test new vector-view
    lout = find_layout(sample_info4, ch_type=None)
    assert_equal(lout.kind, "Vectorview-all")
    assert all(" " not in k for k in lout.names)

    lout = find_layout(sample_info, ch_type="grad")
    assert_equal(lout.kind, "Vectorview-grad")
    lout = find_layout(sample_info2)
    assert_equal(lout.kind, "Vectorview-grad")
    lout = find_layout(sample_info2, ch_type="grad")
    assert_equal(lout.kind, "Vectorview-grad")
    lout = find_layout(sample_info2, ch_type="meg")
    assert_equal(lout.kind, "Vectorview-all")

    lout = find_layout(sample_info, ch_type="mag")
    assert_equal(lout.kind, "Vectorview-mag")
    lout = find_layout(sample_info3)
    assert_equal(lout.kind, "Vectorview-mag")
    lout = find_layout(sample_info3, ch_type="mag")
    assert_equal(lout.kind, "Vectorview-mag")
    lout = find_layout(sample_info3, ch_type="meg")
    assert_equal(lout.kind, "Vectorview-all")

    lout = find_layout(sample_info, ch_type="eeg")
    assert_equal(lout.kind, "EEG")
    lout = find_layout(sample_info5)
    assert_equal(lout.kind, "EEG")
    lout = find_layout(sample_info5, ch_type="eeg")
    assert_equal(lout.kind, "EEG")
    # no common layout, 'meg' option not supported

    lout = find_layout(read_info(fname_ctf_raw))
    assert_equal(lout.kind, "CTF-275")

    fname_bti_raw = bti_dir / "exported4D_linux_raw.fif"
    lout = find_layout(read_info(fname_bti_raw))
    assert_equal(lout.kind, "magnesWH3600")

    raw_kit = read_raw_kit(fname_kit_157)
    lout = find_layout(raw_kit.info)
    assert_equal(lout.kind, "KIT-157")

    raw_kit.info["bads"] = ["MEG 013", "MEG 014", "MEG 015", "MEG 016"]
    raw_kit.info._check_consistency()
    lout = find_layout(raw_kit.info)
    assert_equal(lout.kind, "KIT-157")
    # fallback for missing IDs
    for val in (35, 52, 54, 1001):
        with raw_kit.info._unlock():
            raw_kit.info["kit_system_id"] = val
        lout = find_layout(raw_kit.info)
        assert lout.kind == "custom"

    raw_umd = read_raw_kit(fname_kit_umd)
    lout = find_layout(raw_umd.info)
    assert_equal(lout.kind, "KIT-UMD-3")

    # Test plotting
    lout.plot()
    lout.plot(picks=np.arange(10))
    plt.close("all")


def test_box_size():
    """Test calculation of box sizes."""
    # No points. Box size should be 1,1.
    assert_allclose(_box_size([]), (1.0, 1.0))

    # Create one point. Box size should be 1,1.
    point = [(0, 0)]
    assert_allclose(_box_size(point), (1.0, 1.0))

    # Create two points. Box size should be 0.5,1.
    points = [(0.25, 0.5), (0.75, 0.5)]
    assert_allclose(_box_size(points), (0.5, 1.0))

    # Create three points. Box size should be (0.5, 0.5).
    points = [(0.25, 0.25), (0.75, 0.25), (0.5, 0.75)]
    assert_allclose(_box_size(points), (0.5, 0.5))

    # Create a grid of points. Box size should be (0.1, 0.1).
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, 11), np.linspace(-0.5, 0.5, 11))
    x, y = x.ravel(), y.ravel()
    assert_allclose(_box_size(np.c_[x, y]), (0.1, 0.1))

    # Create a random set of points. This should never break the function.
    rng = np.random.RandomState(42)
    points = rng.rand(100, 2)
    width, height = _box_size(points)
    assert width is not None
    assert height is not None

    # Test specifying an existing width.
    points = [(0.25, 0.25), (0.75, 0.25), (0.5, 0.75)]
    assert_allclose(_box_size(points, width=0.4), (0.4, 0.5))

    # Test specifying an existing width that has influence on the calculated
    # height.
    points = [(0.25, 0.25), (0.75, 0.25), (0.5, 0.75)]
    assert_allclose(_box_size(points, width=0.2), (0.2, 1.0))

    # Test specifying an existing height.
    points = [(0.25, 0.25), (0.75, 0.25), (0.5, 0.75)]
    assert_allclose(_box_size(points, height=0.4), (0.5, 0.4))

    # Test specifying an existing height that has influence on the calculated
    # width.
    points = [(0.25, 0.25), (0.75, 0.45), (0.5, 0.75)]
    assert_allclose(_box_size(points, height=0.1), (1.0, 0.1))

    # Test specifying both width and height. The function should simply return
    # these.
    points = [(0.25, 0.25), (0.75, 0.45), (0.5, 0.75)]
    assert_array_equal(_box_size(points, width=0.1, height=0.1), (0.1, 0.1))

    # Test specifying a width that will cause unfixable horizontal overlap and
    # essentially breaks the function (height will be 0).
    points = [(0.25, 0.25), (0.75, 0.25), (0.5, 0.75)]
    assert_array_equal(_box_size(points, width=1), (1, 0))

    # Test adding some padding.
    # Create three points. Box size should be a little less than (0.5, 0.5).
    points = [(0.25, 0.25), (0.75, 0.25), (0.5, 0.75)]
    assert_allclose(_box_size(points, padding=0.1), (0.9 * 0.5, 0.9 * 0.5))


def test_generate_2d_layout():
    """Test creation of a layout from 2d points."""
    snobg = 10
    sbg = 15
    side = range(snobg)
    bg_image = np.random.RandomState(42).randn(sbg, sbg)
    w, h = [0.2, 0.5]

    # Generate fake data
    xy = np.array([(i, j) for i in side for j in side])
    lt = generate_2d_layout(xy, w=w, h=h)

    # Correct points ordering / minmaxing
    comp_1, comp_2 = [(5, 0), (7, 0)]
    assert lt.pos[:, :2].max() == 1
    assert lt.pos[:, :2].min() == 0
    with np.errstate(invalid="ignore"):  # divide by zero
        assert_allclose(
            xy[comp_2] / float(xy[comp_1]), lt.pos[comp_2] / float(lt.pos[comp_1])
        )
    assert_allclose(lt.pos[0, [2, 3]], [w, h])

    # Correct number elements
    assert lt.pos.shape[1] == 4
    assert len(lt.box) == 4

    # Make sure background image normalizing is correct
    lt_bg = generate_2d_layout(xy, bg_image=bg_image)
    assert_allclose(lt_bg.pos[:, :2].max(), xy.max() / float(sbg))


def test_layout_copy(layout):
    """Test copying a layout."""
    layout2 = layout.copy()
    assert_allclose(layout.pos, layout2.pos)
    assert layout.names == layout2.names
    layout2.names[0] = "foo"
    layout2.pos[0, 0] = 0.8
    assert layout.names != layout2.names
    assert layout.pos[0, 0] != layout2.pos[0, 0]


@pytest.mark.parametrize(
    "picks, exclude",
    [
        ([0, 1], ()),
        (["0", 1], ()),
        (None, ["2"]),
        (None, "2"),
        (None, [2]),
        (None, 2),
        ("all", 2),
        ("all", "2"),
        (slice(0, 2), ()),
        (("0", "1"), ("0", "1")),
        (("0", 1), ("0", "1")),
        (("0", 1), (0, "1")),
        (set(["0", 1]), ()),
        (set([0, 1]), set()),
        (None, set([2])),
        (np.array([0, 1]), ()),
        (None, np.array([2])),
        (np.array(["0", "1"]), ()),
    ],
)
def test_layout_pick(layout, picks, exclude):
    """Test selection of channels in a layout."""
    layout2 = layout.copy()
    layout2.pick(picks, exclude)
    assert layout2.names == layout.names[:2]
    assert_allclose(layout2.pos, layout.pos[:2, :])


def test_layout_pick_more(layout):
    """Test more channel selection in a layout."""
    layout2 = layout.copy()
    layout2.pick(0)
    assert len(layout2.names) == 1
    assert layout2.names[0] == layout.names[0]
    assert_allclose(layout2.pos, layout.pos[:1, :])

    layout2 = layout.copy()
    layout2.pick("all", exclude=("0", "1"))
    assert len(layout2.names) == 1
    assert layout2.names[0] == layout.names[2]
    assert_allclose(layout2.pos, layout.pos[2:, :])

    layout2 = layout.copy()
    layout2.pick("all", exclude=("0", 1))
    assert len(layout2.names) == 1
    assert layout2.names[0] == layout.names[2]
    assert_allclose(layout2.pos, layout.pos[2:, :])


def test_layout_pick_errors(layout):
    """Test validation of layout.pick."""
    with pytest.raises(TypeError, match="must be a list, tuple, set or ndarray"):
        layout.pick(lambda x: x)
    with pytest.raises(TypeError, match="must be a list, tuple, set or ndarray"):
        layout.pick(None, lambda x: x)
    with pytest.raises(TypeError, match="must be integers or strings"):
        layout.pick([0, lambda x: x])
    with pytest.raises(TypeError, match="must be integers or strings"):
        layout.pick(None, [0, lambda x: x])
    with pytest.raises(ValueError, match="does not match any channels"):
        layout.pick("foo")
    with pytest.raises(ValueError, match="does not match any channels"):
        layout.pick(None, "foo")
    with pytest.raises(ValueError, match="does not match any channels"):
        layout.pick(101)
    with pytest.raises(ValueError, match="does not match any channels"):
        layout.pick(None, 101)
    with pytest.warns(RuntimeWarning, match="has duplicates which will be ignored"):
        layout.copy().pick(["0", "0"])
    with pytest.warns(RuntimeWarning, match="has duplicates which will be ignored"):
        layout.copy().pick(["0", 0])
    with pytest.warns(RuntimeWarning, match="has duplicates which will be ignored"):
        layout.copy().pick(None, ["0", "0"])
    with pytest.warns(RuntimeWarning, match="has duplicates which will be ignored"):
        layout.copy().pick(None, ["0", 0])
    with pytest.raises(RuntimeError, match="selection yielded no remaining channels"):
        layout.copy().pick(None, ["0", "1", "2"])
    with pytest.raises(ValueError, match="must be a 1D array-like"):
        layout.copy().pick(None, np.array([[0, 1]]))
    with pytest.raises(TypeError, match="slice of integers"):
        layout.copy().pick(slice("2342342342", 0, 3), ())
