# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

from contextlib import nullcontext
import os.path as op
import os

import pytest
from numpy.testing import assert_allclose
import numpy as np

import mne
from mne.datasets import testing
from mne.io import read_info
from mne.io.kit.tests import data_dir as kit_data_dir
from mne.io.constants import FIFF
from mne.utils import get_config, catch_logging
from mne.channels import DigMontage
from mne.coreg import Coregistration
from mne.viz import _3d


data_path = testing.data_path(download=False)
raw_path = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
fname_trans = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')
kit_raw_path = op.join(kit_data_dir, 'test_bin_raw.fif')
subjects_dir = op.join(data_path, 'subjects')
fid_fname = op.join(subjects_dir, 'sample', 'bem', 'sample-fiducials.fif')
ctf_raw_path = op.join(data_path, 'CTF', 'catch-alp-good-f.ds')
nirx_15_0_raw_path = op.join(data_path, 'NIRx', 'nirscout',
                             'nirx_15_0_recording', 'NIRS-2019-10-27_003.hdr')
nirsport2_raw_path = op.join(data_path, 'NIRx', 'nirsport_v2', 'aurora_2021_9',
                             '2021-10-01_002_config.hdr')
snirf_nirsport2_raw_path = op.join(data_path, 'SNIRF', 'NIRx', 'NIRSport2',
                                   '1.0.3', '2021-05-05_001.snirf')


class TstVTKPicker(object):
    """Class to test cell picking."""

    def __init__(self, mesh, cell_id, event_pos):
        self.mesh = mesh
        self.cell_id = cell_id
        self.point_id = None
        self.event_pos = event_pos

    def GetCellId(self):
        """Return the picked cell."""
        return self.cell_id

    def GetDataSet(self):
        """Return the picked mesh."""
        return self.mesh

    def GetPickPosition(self):
        """Return the picked position."""
        vtk_cell = self.mesh.GetCell(self.cell_id)
        cell = [vtk_cell.GetPointId(point_id) for point_id
                in range(vtk_cell.GetNumberOfPoints())]
        self.point_id = cell[0]
        return self.mesh.points[self.point_id]

    def GetEventPosition(self):
        """Return event position."""
        return self.event_pos


@pytest.mark.slowtest
@testing.requires_testing_data
@pytest.mark.parametrize(
    'inst_path', (raw_path, 'gen_montage', ctf_raw_path, nirx_15_0_raw_path,
                  nirsport2_raw_path, snirf_nirsport2_raw_path))
def test_coreg_gui_pyvista_file_support(inst_path, tmp_path,
                                        renderer_interactive_pyvistaqt):
    """Test reading supported files."""
    from mne.gui import coregistration

    if inst_path == 'gen_montage':
        # generate a montage fig to use as inst.
        tmp_info = read_info(raw_path)
        eeg_chans = []
        for pt in tmp_info['dig']:
            if pt['kind'] == FIFF.FIFFV_POINT_EEG:
                eeg_chans.append(f"EEG {pt['ident']:03d}")

        dig = DigMontage(dig=tmp_info['dig'],
                         ch_names=eeg_chans)
        inst_path = tmp_path / 'tmp-dig.fif'
        dig.save(inst_path)

    if inst_path == ctf_raw_path:
        ctx = pytest.warns(RuntimeWarning, match='MEG ref channel RMSP')
    elif inst_path == snirf_nirsport2_raw_path:  # TODO: This is maybe a bug?
        ctx = pytest.warns(RuntimeWarning, match='assuming "head"')
    else:
        ctx = nullcontext()
    with ctx:
        coregistration(inst=inst_path, subject='sample',
                       subjects_dir=subjects_dir)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_coreg_gui_pyvista_basic(tmp_path, renderer_interactive_pyvistaqt,
                                 monkeypatch):
    """Test that using CoregistrationUI matches mne coreg."""
    from mne.gui import coregistration
    config = get_config()
    # the sample subject in testing has MRI fids
    assert op.isfile(op.join(
        subjects_dir, 'sample', 'bem', 'sample-fiducials.fif'))

    coreg = coregistration(subject='sample', subjects_dir=subjects_dir,
                           trans=fname_trans)
    assert coreg._lock_fids
    coreg._reset_fiducials()
    coreg.close()

    # make it always log the distances
    monkeypatch.setattr(_3d.logger, 'info', _3d.logger.warning)
    with catch_logging() as log:
        coreg = coregistration(inst=raw_path, subject='sample',
                               head_high_res=False,  # for speed
                               subjects_dir=subjects_dir, verbose='debug')
    log = log.getvalue()
    assert 'Total 16/78 points inside the surface' in log
    coreg._set_fiducials_file(fid_fname)
    assert coreg._fiducials_file == fid_fname

    # fitting (with scaling)
    assert not coreg._mri_scale_modified
    coreg._reset()
    coreg._reset_fitting_parameters()
    coreg._set_scale_mode("uniform")
    coreg._fits_fiducials()
    assert_allclose(coreg.coreg._scale,
                    np.array([97.46, 97.46, 97.46]) * 1e-2,
                    atol=1e-3)
    shown_scale = [coreg._widgets[f's{x}'].get_value() for x in 'XYZ']
    assert_allclose(shown_scale, coreg.coreg._scale * 100, atol=1e-2)
    coreg._set_icp_fid_match("nearest")
    coreg._set_scale_mode("3-axis")
    coreg._fits_icp()
    assert_allclose(coreg.coreg._scale,
                    np.array([104.43, 101.47, 125.78]) * 1e-2,
                    atol=1e-3)
    shown_scale = [coreg._widgets[f's{x}'].get_value() for x in 'XYZ']
    assert_allclose(shown_scale, coreg.coreg._scale * 100, atol=1e-2)
    coreg._set_scale_mode("None")
    coreg._set_icp_fid_match("matched")
    assert coreg._mri_scale_modified

    # unlock fiducials
    assert coreg._lock_fids
    coreg._set_lock_fids(False)
    assert not coreg._lock_fids

    # picking
    assert not coreg._mri_fids_modified
    vtk_picker = TstVTKPicker(coreg._surfaces['head'], 0, (0, 0))
    coreg._on_mouse_move(vtk_picker, None)
    coreg._on_button_press(vtk_picker, None)
    coreg._on_pick(vtk_picker, None)
    coreg._on_button_release(vtk_picker, None)
    coreg._on_pick(vtk_picker, None)  # also pick when locked
    assert coreg._mri_fids_modified

    # lock fiducials
    coreg._set_lock_fids(True)
    assert coreg._lock_fids

    # fitting (no scaling)
    assert coreg._nasion_weight == 10.
    coreg._set_point_weight(11., 'nasion')
    assert coreg._nasion_weight == 11.
    coreg._fit_fiducials()
    with catch_logging() as log:
        coreg._redraw()  # actually emit the log
    log = log.getvalue()
    assert 'Total 6/78 points inside the surface' in log
    with catch_logging() as log:
        coreg._fit_icp()
        coreg._redraw()
    log = log.getvalue()
    assert 'Total 38/78 points inside the surface' in log
    assert coreg.coreg._extra_points_filter is None
    coreg._omit_hsp()
    with catch_logging() as log:
        coreg._redraw()
    log = log.getvalue()
    assert 'Total 29/53 points inside the surface' in log
    assert coreg.coreg._extra_points_filter is not None
    coreg._reset_omit_hsp_filter()
    with catch_logging() as log:
        coreg._redraw()
    log = log.getvalue()
    assert 'Total 38/78 points inside the surface' in log
    assert coreg.coreg._extra_points_filter is None

    assert coreg._grow_hair == 0
    coreg._fit_fiducials()  # go back to few inside to start
    with catch_logging() as log:
        coreg._redraw()
    log = log.getvalue()
    assert 'Total 6/78 points inside the surface' in log
    norm = np.linalg.norm(coreg._head_geo['rr'])  # what's used for inside
    assert_allclose(norm, 5.949288, atol=1e-3)
    coreg._set_grow_hair(20.0)
    with catch_logging() as log:
        coreg._redraw()
    assert coreg._grow_hair == 20.0
    norm = np.linalg.norm(coreg._head_geo['rr'])
    assert_allclose(norm, 6.555220, atol=1e-3)  # outward
    log = log.getvalue()
    assert 'Total 8/78 points inside the surface' in log  # more outside now

    # visualization
    assert not coreg._helmet
    assert coreg._actors['helmet'] is None
    coreg._set_helmet(True)
    assert coreg._helmet
    with catch_logging() as log:
        coreg._redraw(verbose='debug')
    log = log.getvalue()
    assert 'Drawing helmet' in log
    coreg._set_point_weight(1., 'nasion')
    coreg._fit_fiducials()
    with catch_logging() as log:
        coreg._redraw(verbose='debug')
    log = log.getvalue()
    assert 'Drawing helmet' in log
    assert coreg._orient_glyphs
    assert coreg._scale_by_distance
    assert coreg._mark_inside
    assert_allclose(
        coreg._head_opacity,
        float(config.get('MNE_COREG_HEAD_OPACITY', '0.8')))
    assert coreg._hpi_coils
    assert coreg._eeg_channels
    assert coreg._head_shape_points
    assert coreg._scale_mode == 'None'
    assert coreg._icp_fid_match == 'matched'
    assert coreg._head_resolution is False

    assert coreg._trans_modified
    tmp_trans = tmp_path / 'tmp-trans.fif'
    coreg._save_trans(tmp_trans)
    assert not coreg._trans_modified
    assert op.isfile(tmp_trans)

    # first, disable auto cleanup
    coreg._renderer._window_close_disconnect(after=True)
    # test _close_callback()
    coreg.close()
    coreg._widgets['close_dialog'].trigger('Discard')  # do not save
    coreg._clean()  # finally, cleanup internal structures

    # Coregistration instance should survive
    assert isinstance(coreg.coreg, Coregistration)

    # Fullscreen mode
    coreg = coregistration(
        subject='sample', subjects_dir=subjects_dir, fullscreen=True
    )


@pytest.mark.slowtest
@testing.requires_testing_data
def test_coreg_gui_scraper(tmp_path, renderer_interactive_pyvistaqt):
    """Test the scrapper for the coregistration GUI."""
    from mne.gui import coregistration
    coreg = coregistration(subject='sample', subjects_dir=subjects_dir,
                           trans=fname_trans)
    (tmp_path / '_images').mkdir()
    image_path = str(tmp_path / '_images' / 'temp.png')
    gallery_conf = dict(builder_name='html', src_dir=str(tmp_path))
    block_vars = dict(
        example_globals=dict(gui=coreg),
        image_path_iterator=iter([image_path]))
    assert not op.isfile(image_path)
    assert not getattr(coreg, '_scraped', False)
    mne.gui._GUIScraper()(None, block_vars, gallery_conf)
    assert op.isfile(image_path)
    assert coreg._scraped


@pytest.mark.slowtest
@testing.requires_testing_data
def test_coreg_gui_notebook(renderer_notebook, nbexec):
    """Test the coregistration UI in a notebook."""
    import os
    import pytest
    import mne
    from mne.datasets import testing
    from mne.gui import coregistration
    mne.viz.set_3d_backend('notebook')  # set the 3d backend
    with pytest.MonkeyPatch().context() as mp:
        mp.delenv('_MNE_FAKE_HOME_DIR')
        data_path = testing.data_path(download=False)
    subjects_dir = os.path.join(data_path, 'subjects')
    coregistration(subject='sample', subjects_dir=subjects_dir)


@pytest.mark.slowtest
def test_no_sparse_head(subjects_dir_tmp, renderer_interactive_pyvistaqt,
                        monkeypatch):
    """Test mne.gui.coregistration with no sparse head."""
    from mne.gui import coregistration
    subject = 'sample'
    out_rr, out_tris = mne.read_surface(
        op.join(subjects_dir_tmp, subject, 'bem', 'outer_skin.surf'))
    for head in ('sample-head.fif', 'outer_skin.surf'):
        os.remove(op.join(subjects_dir_tmp, subject, 'bem', head))
    # Avoid actually doing the decimation (it's slow)
    monkeypatch.setattr(
        mne.coreg, 'decimate_surface',
        lambda rr, tris, n_triangles: (out_rr, out_tris))
    with pytest.warns(RuntimeWarning, match='No low-resolution head found'):
        coreg = coregistration(
            inst=raw_path, subject=subject, subjects_dir=subjects_dir_tmp)
    coreg.close()
