import os.path as op
import numpy as np

import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mne
from mne import (vertex_to_mni, head_to_mni,
                 read_talxfm, read_freesurfer_lut,
                 get_volume_labels_from_aseg)
from mne.datasets import testing
from mne._freesurfer import _get_mgz_header, _check_subject_dir
from mne.transforms import apply_trans, _get_trans
from mne.utils import requires_nibabel

data_path = testing.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')
fname_mri = op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz')
aseg_fname = op.join(data_path, 'subjects', 'sample', 'mri', 'aseg.mgz')
trans_fname = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')
rng = np.random.RandomState(0)


@testing.requires_testing_data
def test_check_subject_dir():
    """Test checking for a Freesurfer recon-all subject directory."""
    _check_subject_dir('sample', subjects_dir)
    with pytest.raises(ValueError, match='subject folder is incorrect'):
        _check_subject_dir('foo', data_path)


@testing.requires_testing_data
@requires_nibabel()
def test_mgz_header():
    """Test MGZ header reading."""
    import nibabel
    header = _get_mgz_header(fname_mri)
    mri_hdr = nibabel.load(fname_mri).header
    assert_allclose(mri_hdr.get_data_shape(), header['dims'])
    assert_allclose(mri_hdr.get_vox2ras_tkr(), header['vox2ras_tkr'])
    assert_allclose(mri_hdr.get_ras2vox(), np.linalg.inv(header['vox2ras']))


@testing.requires_testing_data
def test_vertex_to_mni():
    """Test conversion of vertices to MNI coordinates."""
    # obtained using "tksurfer (sample) (l/r)h white"
    vertices = [100960, 7620, 150549, 96761]
    coords = np.array([[-60.86, -11.18, -3.19], [-36.46, -93.18, -2.36],
                       [-38.00, 50.08, -10.61], [47.14, 8.01, 46.93]])
    hemis = [0, 0, 0, 1]
    coords_2 = vertex_to_mni(vertices, hemis, 'sample', subjects_dir)
    # less than 1mm error
    assert_allclose(coords, coords_2, atol=1.0)


@testing.requires_testing_data
def test_head_to_mni():
    """Test conversion of aseg vertices to MNI coordinates."""
    # obtained using freeview
    coords = np.array([[22.52, 11.24, 17.72], [22.52, 5.46, 21.58],
                       [16.10, 5.46, 22.23], [21.24, 8.36, 22.23]]) / 1000.

    xfm = read_talxfm('sample', subjects_dir)
    coords_MNI = apply_trans(xfm['trans'], coords) * 1000.

    mri_head_t, _ = _get_trans(trans_fname, 'mri', 'head', allow_none=False)

    # obtained from sample_audvis-meg-oct-6-mixed-fwd.fif
    coo_right_amygdala = np.array([[0.01745682, 0.02665809, 0.03281873],
                                   [0.01014125, 0.02496262, 0.04233755],
                                   [0.01713642, 0.02505193, 0.04258181],
                                   [0.01720631, 0.03073877, 0.03850075]])
    coords_MNI_2 = head_to_mni(coo_right_amygdala, 'sample', mri_head_t,
                               subjects_dir)
    # less than 1mm error
    assert_allclose(coords_MNI, coords_MNI_2, atol=10.0)


@requires_nibabel()
@testing.requires_testing_data
def test_vertex_to_mni_fs_nibabel(monkeypatch):
    """Test equivalence of vert_to_mni for nibabel and freesurfer."""
    n_check = 1000
    subject = 'sample'
    vertices = rng.randint(0, 100000, n_check)
    hemis = rng.randint(0, 1, n_check)
    coords = vertex_to_mni(vertices, hemis, subject, subjects_dir)
    read_mri = mne._freesurfer._read_mri_info
    monkeypatch.setattr(
        mne._freesurfer, '_read_mri_info',
        lambda *args, **kwargs: read_mri(*args, use_nibabel=True, **kwargs))
    coords_2 = vertex_to_mni(vertices, hemis, subject, subjects_dir)
    # less than 0.1 mm error
    assert_allclose(coords, coords_2, atol=0.1)


@testing.requires_testing_data
@requires_nibabel()
@pytest.mark.parametrize('fname', [
    None,
    op.join(op.dirname(mne.__file__), 'data', 'FreeSurferColorLUT.txt'),
])
def test_read_freesurfer_lut(fname, tmp_path):
    """Test reading volume label names."""
    atlas_ids, colors = read_freesurfer_lut(fname)
    assert list(atlas_ids).count('Brain-Stem') == 1
    assert len(colors) == len(atlas_ids) == 1266
    label_names, label_colors = get_volume_labels_from_aseg(
        aseg_fname, return_colors=True)
    assert isinstance(label_names, list)
    assert isinstance(label_colors, list)
    assert label_names.count('Brain-Stem') == 1
    for c in label_colors:
        assert isinstance(c, np.ndarray)
        assert c.shape == (4,)
    assert len(label_names) == len(label_colors) == 46
    with pytest.raises(ValueError, match='must be False'):
        get_volume_labels_from_aseg(
            aseg_fname, return_colors=True, atlas_ids=atlas_ids)
    label_names_2 = get_volume_labels_from_aseg(
        aseg_fname, atlas_ids=atlas_ids)
    assert label_names == label_names_2
    # long name (only test on one run)
    if fname is not None:
        return
    fname = tmp_path / 'long.txt'
    names = ['Anterior_Cingulate_and_Medial_Prefrontal_Cortex-' + hemi
             for hemi in ('lh', 'rh')]
    ids = np.arange(1, len(names) + 1)
    colors = [(id_,) * 4 for id_ in ids]
    with open(fname, 'w') as fid:
        for name, id_, color in zip(names, ids, colors):
            out_color = ' '.join('%3d' % x for x in color)
            line = '%d    %s %s\n' % (id_, name, out_color)
            fid.write(line)
    lut, got_colors = read_freesurfer_lut(fname)
    assert len(lut) == len(got_colors) == len(names) == len(ids)
    for name, id_, color in zip(names, ids, colors):
        assert name in lut
        assert name in got_colors
        assert_array_equal(got_colors[name][:3], color[:3])
        assert lut[name] == id_
    with open(fname, 'w') as fid:
        for name, id_, color in zip(names, ids, colors):
            out_color = ' '.join('%3d' % x for x in color[:3])  # wrong length!
            line = '%d    %s %s\n' % (id_, name, out_color)
            fid.write(line)
    with pytest.raises(RuntimeError, match='formatted'):
        read_freesurfer_lut(fname)
