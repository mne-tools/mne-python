# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-clause

import numpy as np
from numpy.testing import assert_allclose

import pytest

import mne
from mne.datasets import testing
from mne.transforms import apply_trans
from mne.utils import requires_version, use_log_level
from mne.viz.utils import _fake_click

data_path = testing.data_path(download=False)
subject = "sample"
subjects_dir = data_path / "subjects"
sample_dir = data_path / "MEG" / subject
raw_path = sample_dir / "sample_audvis_trunc_raw.fif"
fname_trans = sample_dir / "sample_audvis_trunc-trans.fif"

# Module-level ignore
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*locate_ieeg.*deprecated.*:FutureWarning"
)


@pytest.fixture
def _fake_CT_coords(skull_size=5, contact_size=2):
    """Make somewhat realistic CT data with contacts."""
    nib = pytest.importorskip("nibabel")
    brain = nib.load(subjects_dir / subject / "mri" / "brain.mgz")
    verts = mne.read_surface(subjects_dir / subject / "bem" / "outer_skull.surf")[0]
    verts = apply_trans(np.linalg.inv(brain.header.get_vox2ras_tkr()), verts)
    x, y, z = np.array(brain.shape).astype(int) // 2
    coords = [
        (x, y - 14, z),
        (x - 10, y - 15, z),
        (x - 20, y - 16, z + 1),
        (x - 30, y - 16, z + 1),
    ]
    center = np.array(brain.shape) / 2
    # make image
    np.random.seed(99)
    ct_data = np.random.random(brain.shape).astype(np.float32) * 100
    # make skull
    for vert in verts:
        x, y, z = np.round(vert).astype(int)
        ct_data[
            slice(x - skull_size, x + skull_size + 1),
            slice(y - skull_size, y + skull_size + 1),
            slice(z - skull_size, z + skull_size + 1),
        ] = 1000
    # add electrode with contacts
    for x, y, z in coords:
        # make sure not in skull
        assert np.linalg.norm(center - np.array((x, y, z))) < 50
        ct_data[
            slice(x - contact_size, x + contact_size + 1),
            slice(y - contact_size, y + contact_size + 1),
            slice(z - contact_size, z + contact_size + 1),
        ] = 1000 - np.linalg.norm(
            np.array(np.meshgrid(*[range(-contact_size, contact_size + 1)] * 3)), axis=0
        )
    ct = nib.MGHImage(ct_data, brain.affine)
    coords = apply_trans(ct.header.get_vox2ras_tkr(), np.array(coords))
    return ct, coords


def test_ieeg_elec_locate_io(renderer_interactive_pyvistaqt):
    """Test the input/output of the intracranial location GUI."""
    nib = pytest.importorskip("nibabel")
    import mne.gui

    info = mne.create_info([], 1000)

    # fake as T1 so that aligned
    aligned_ct = nib.load(subjects_dir / subject / "mri" / "brain.mgz")

    trans = mne.transforms.Transform("head", "mri")
    with pytest.raises(ValueError, match="No channels found in `info` to locate"):
        mne.gui.locate_ieeg(info, trans, aligned_ct, subject, subjects_dir)

    info = mne.create_info(["test"], 1000, "seeg")
    montage = mne.channels.make_dig_montage({"test": [0, 0, 0]}, coord_frame="mri")
    with pytest.warns(RuntimeWarning, match="nasion not found"):
        info.set_montage(montage)
    with pytest.raises(RuntimeError, match='must be in the "head" coordinate frame'):
        with pytest.warns(RuntimeWarning, match="`pial` surface not found"):
            mne.gui.locate_ieeg(info, trans, aligned_ct, subject, subjects_dir)


@requires_version("sphinx_gallery")
@testing.requires_testing_data
def test_locate_scraper(renderer_interactive_pyvistaqt, _fake_CT_coords, tmp_path):
    """Test sphinx-gallery scraping of the GUI."""
    import mne.gui

    raw = mne.io.read_raw_fif(raw_path)
    raw.pick_types(eeg=True)
    ch_dict = {
        "EEG 001": "LAMY 1",
        "EEG 002": "LAMY 2",
        "EEG 003": "LSTN 1",
        "EEG 004": "LSTN 2",
    }
    raw.pick_channels(list(ch_dict.keys()))
    raw.rename_channels(ch_dict)
    raw.set_montage(None)
    aligned_ct, _ = _fake_CT_coords
    trans = mne.read_trans(fname_trans)
    with pytest.warns(RuntimeWarning, match="`pial` surface not found"):
        gui = mne.gui.locate_ieeg(
            raw.info, trans, aligned_ct, subject=subject, subjects_dir=subjects_dir
        )
    (tmp_path / "_images").mkdir()
    image_path = tmp_path / "_images" / "temp.png"
    gallery_conf = dict(builder_name="html", src_dir=tmp_path)
    block_vars = dict(
        example_globals=dict(gui=gui), image_path_iterator=iter([str(image_path)])
    )
    assert not image_path.is_file()
    assert not getattr(gui, "_scraped", False)
    mne.gui._GUIScraper()(None, block_vars, gallery_conf)
    assert image_path.is_file()
    assert gui._scraped
    # no need to call .close


@testing.requires_testing_data
def test_ieeg_elec_locate_display(renderer_interactive_pyvistaqt, _fake_CT_coords):
    """Test that the intracranial location GUI displays properly."""
    raw = mne.io.read_raw_fif(raw_path, preload=True)
    raw.pick_types(eeg=True)
    ch_dict = {
        "EEG 001": "LAMY 1",
        "EEG 002": "LAMY 2",
        "EEG 003": "LSTN 1",
        "EEG 004": "LSTN 2",
    }
    raw.pick_channels(list(ch_dict.keys()))
    raw.rename_channels(ch_dict)
    raw.set_eeg_reference("average")
    raw.set_channel_types({name: "seeg" for name in raw.ch_names})
    raw.set_montage(None)
    aligned_ct, coords = _fake_CT_coords
    trans = mne.read_trans(fname_trans)

    with pytest.warns(RuntimeWarning, match="`pial` surface not found"):
        gui = mne.gui.locate_ieeg(
            raw.info,
            trans,
            aligned_ct,
            subject=subject,
            subjects_dir=subjects_dir,
            verbose=True,
        )

    with pytest.raises(ValueError, match="read-only"):
        gui._ras[:] = coords[0]  # start in the right position
    gui.set_RAS(coords[0])
    gui.mark_channel()

    with pytest.raises(ValueError, match="not found"):
        gui.mark_channel("foo")

    assert not gui._lines and not gui._lines_2D  # no lines for one contact
    for ci, coord in enumerate(coords[1:], 1):
        coord_vox = apply_trans(gui._ras_vox_t, coord)
        with use_log_level("debug"):
            _fake_click(
                gui._figs[2],
                gui._figs[2].axes[0],
                coord_vox[:-1],
                xform="data",
                kind="release",
            )
        assert_allclose(coord[:2], gui._ras[:2], atol=0.1, err_msg=f"coords[{ci}][:2]")
        assert_allclose(coord[2], gui._ras[2], atol=2, err_msg=f"coords[{ci}][2]")
        gui.mark_channel()

    # ensure a 3D line was made for each group
    assert len(gui._lines) == 2

    # test snap to center
    gui._ch_index = 0
    gui.set_RAS(coords[0])  # move to first position
    gui.mark_channel()
    assert_allclose(coords[0], gui._chs["LAMY 1"], atol=0.2)
    gui._snap_button.click()
    assert gui._snap_button.text() == "Off"
    # now make sure no snap happens
    gui._ch_index = 0
    gui.set_RAS(coords[1] + 1)
    gui.mark_channel()
    assert_allclose(coords[1] + 1, gui._chs["LAMY 1"], atol=0.01)
    # check that it turns back on
    gui._snap_button.click()
    assert gui._snap_button.text() == "On"

    # test remove
    gui.remove_channel("LAMY 2")
    assert np.isnan(gui._chs["LAMY 2"]).all()

    with pytest.raises(ValueError, match="not found"):
        gui.remove_channel("foo")

    # check that raw object saved
    assert not np.isnan(raw.info["chs"][0]["loc"][:3]).any()  # LAMY 1
    assert np.isnan(raw.info["chs"][1]["loc"][:3]).all()  # LAMY 2 (removed)

    # move sliders
    gui._alpha_slider.setValue(75)
    assert gui._ch_alpha == 0.75
    gui._radius_slider.setValue(5)
    assert gui._radius == 5
    ct_sum_before = np.nansum(gui._images["ct"][0].get_array().data)
    gui._ct_min_slider.setValue(500)
    assert np.nansum(gui._images["ct"][0].get_array().data) < ct_sum_before

    # test buttons
    gui._toggle_show_brain()
    assert "mri" in gui._images
    assert "local_max" not in gui._images
    gui._toggle_show_max()
    assert "local_max" in gui._images
    assert "mip" not in gui._images
    gui._toggle_show_mip()
    assert "mip" in gui._images
    assert "mip_chs" in gui._images
    assert len(gui._lines_2D) == 1  # LAMY only has one contact

    # check montage
    montage = raw.get_montage()
    assert montage is not None
    assert_allclose(
        montage.get_positions()["ch_pos"]["LAMY 1"],
        [0.00726235, 0.01713514, 0.04167233],
        atol=0.01,
    )
    gui.close()
