# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-clause

import os.path as op
import numpy as np
from numpy.testing import assert_allclose

import pytest

import mne
from mne.datasets import testing
from mne.utils import requires_nibabel, requires_version
from mne.viz.utils import _fake_click

data_path = testing.data_path(download=False)
subject = 'sample'
subjects_dir = op.join(data_path, 'subjects')
sample_dir = op.join(data_path, 'MEG', subject)
raw_path = op.join(sample_dir, 'sample_audvis_trunc_raw.fif')
fname_trans = op.join(sample_dir, 'sample_audvis_trunc-trans.fif')


@requires_nibabel()
@pytest.fixture
def _fake_CT_coords(skull_size=5, contact_size=2):
    """Make somewhat realistic CT data with contacts."""
    import nibabel as nib
    brain = nib.load(
        op.join(subjects_dir, subject, 'mri', 'brain.mgz'))
    verts = mne.read_surface(
        op.join(subjects_dir, subject, 'bem', 'outer_skull.surf'))[0]
    verts = mne.transforms.apply_trans(
        np.linalg.inv(brain.header.get_vox2ras_tkr()), verts)
    x, y, z = np.array(brain.shape).astype(int) // 2
    coords = [(x, y - 14, z), (x - 10, y - 15, z),
              (x - 20, y - 16, z + 1), (x - 30, y - 16, z + 1)]
    center = np.array(brain.shape) / 2
    # make image
    np.random.seed(99)
    ct_data = np.random.random(brain.shape).astype(np.float32) * 100
    # make skull
    for vert in verts:
        x, y, z = np.round(vert).astype(int)
        ct_data[slice(x - skull_size, x + skull_size + 1),
                slice(y - skull_size, y + skull_size + 1),
                slice(z - skull_size, z + skull_size + 1)] = 1000
    # add electrode with contacts
    for (x, y, z) in coords:
        # make sure not in skull
        assert np.linalg.norm(center - np.array((x, y, z))) < 50
        ct_data[slice(x - contact_size, x + contact_size + 1),
                slice(y - contact_size, y + contact_size + 1),
                slice(z - contact_size, z + contact_size + 1)] = \
            1000 - np.linalg.norm(np.array(np.meshgrid(
                *[range(-contact_size, contact_size + 1)] * 3)), axis=0)
    ct = nib.MGHImage(ct_data, brain.affine)
    coords = mne.transforms.apply_trans(
        ct.header.get_vox2ras_tkr(), np.array(coords))
    return ct, coords


@requires_nibabel()
@pytest.fixture
def _locate_ieeg(renderer_interactive_pyvistaqt):
    # Use a fixture to create these classes so we can ensure that they
    # are closed at the end of the test
    guis = list()

    def fun(*args, **kwargs):
        guis.append(mne.gui.locate_ieeg(*args, **kwargs))
        return guis[-1]

    yield fun

    for gui in guis:
        try:
            gui.close()
        except Exception:
            pass


def test_ieeg_elec_locate_gui_io(_locate_ieeg):
    """Test the input/output of the intracranial location GUI."""
    import nibabel as nib
    info = mne.create_info([], 1000)
    aligned_ct = nib.MGHImage(np.zeros((256, 256, 256), dtype=np.float32),
                              np.eye(4))
    trans = mne.transforms.Transform('head', 'mri')
    with pytest.raises(ValueError,
                       match='No channels found in `info` to locate'):
        _locate_ieeg(info, aligned_ct, subject, subjects_dir)
    info = mne.create_info(['test'], 1000, ['seeg'])
    with pytest.raises(ValueError, match='CT is not aligned to MRI'):
        _locate_ieeg(info, trans, aligned_ct, subject=subject,
                     subjects_dir=subjects_dir)


@requires_version('sphinx_gallery')
@testing.requires_testing_data
def test_locate_scraper(_locate_ieeg, _fake_CT_coords, tmp_path):
    """Test sphinx-gallery scraping of the GUI."""
    raw = mne.io.read_raw_fif(raw_path)
    raw.pick_types(eeg=True)
    ch_dict = {'EEG 001': 'LAMY 1', 'EEG 002': 'LAMY 2',
               'EEG 003': 'LSTN 1', 'EEG 004': 'LSTN 2'}
    raw.pick_channels(list(ch_dict.keys()))
    raw.rename_channels(ch_dict)
    raw.set_montage(None)
    aligned_ct, _ = _fake_CT_coords
    trans = mne.read_trans(fname_trans)
    with pytest.warns(RuntimeWarning, match='`pial` surface not found'):
        gui = _locate_ieeg(raw.info, trans, aligned_ct,
                           subject=subject, subjects_dir=subjects_dir)
    (tmp_path / '_images').mkdir()
    image_path = str(tmp_path / '_images' / 'temp.png')
    gallery_conf = dict(builder_name='html', src_dir=str(tmp_path))
    block_vars = dict(
        example_globals=dict(gui=gui),
        image_path_iterator=iter([image_path]))
    assert not op.isfile(image_path)
    assert not getattr(gui, '_scraped', False)
    mne.gui._LocateScraper()(None, block_vars, gallery_conf)
    assert op.isfile(image_path)
    assert gui._scraped


@testing.requires_testing_data
def test_ieeg_elec_locate_gui_display(_locate_ieeg, _fake_CT_coords):
    """Test that the intracranial location GUI displays properly."""
    raw = mne.io.read_raw_fif(raw_path)
    raw.pick_types(eeg=True)
    ch_dict = {'EEG 001': 'LAMY 1', 'EEG 002': 'LAMY 2',
               'EEG 003': 'LSTN 1', 'EEG 004': 'LSTN 2'}
    raw.pick_channels(list(ch_dict.keys()))
    raw.rename_channels(ch_dict)
    raw.set_montage(None)
    aligned_ct, coords = _fake_CT_coords
    trans = mne.read_trans(fname_trans)
    with pytest.warns(RuntimeWarning, match='`pial` surface not found'):
        gui = _locate_ieeg(raw.info, trans, aligned_ct,
                           subject=subject, subjects_dir=subjects_dir)

    gui._ras[:] = coords[0]  # start in the right position
    gui._move_cursors_to_pos()
    for coord in coords:
        coord_vox = mne.transforms.apply_trans(gui._ras_vox_t, coord)
        _fake_click(gui._figs[2], gui._figs[2].axes[0],
                    coord_vox[:-1], xform='data', kind='release')
        assert_allclose(coord, gui._ras, atol=3)  # clicks are a bit off

    # test snap to center
    gui._ras[:] = coords[0]  # move to first position
    gui._move_cursors_to_pos()
    gui._mark_ch()
    assert_allclose(coords[0], gui._chs['LAMY 1'], atol=0.2)
    gui._snap_button.click()
    assert gui._snap_button.text() == 'Off'
    # now make sure no snap happens
    gui._ras[:] = coords[1] + 1
    gui._mark_ch()
    assert_allclose(coords[1] + 1, gui._chs['LAMY 2'], atol=0.01)
    # check that it turns back on
    gui._snap_button.click()
    assert gui._snap_button.text() == 'On'

    # test remove
    gui._ch_index = 1
    gui._update_ch_selection()
    gui._remove_ch()
    assert np.isnan(gui._chs['LAMY 2']).all()

    # check that raw object saved
    assert not np.isnan(raw.info['chs'][0]['loc'][:3]).any()  # LAMY 1
    assert np.isnan(raw.info['chs'][1]['loc'][:3]).all()  # LAMY 2 (removed)

    # move sliders
    gui._alpha_slider.setValue(75)
    assert gui._ch_alpha == 0.75
    gui._radius_slider.setValue(5)
    assert gui._radius == 5
    ct_sum_before = np.nansum(gui._images['ct'][0].get_array().data)
    gui._ct_min_slider.setValue(500)
    assert np.nansum(gui._images['ct'][0].get_array().data) < ct_sum_before
