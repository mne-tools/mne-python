# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

import os
import os.path as op

import pytest
import warnings
from numpy.testing import assert_allclose
import numpy as np

from mne.datasets import testing
from mne.io import read_info
from mne.io.kit.tests import data_dir as kit_data_dir
from mne.io.constants import FIFF
from mne.utils import get_config
from mne.channels import DigMontage

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

    # Suppressing warnings here is not ideal.
    # However ctf_raw_path (catch-alp-good-f.ds) is poorly formed and causes
    # mne.io.read_raw to issue warning.
    # XXX consider replacing ctf_raw_path and removing warning ignore filter.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coregistration(inst=inst_path, subject='sample',
                       subjects_dir=subjects_dir)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_coreg_gui_pyvista(tmp_path, renderer_interactive_pyvistaqt):
    """Test that using CoregistrationUI matches mne coreg."""
    from mne.gui import coregistration
    from mne.gui._coreg import CoregistrationUI
    with pytest.warns(DeprecationWarning, match='standalone is deprecated'):
        CoregistrationUI(info_file=None, subject='sample',
                         subjects_dir=subjects_dir, standalone=False)

    config = get_config(home_dir=os.environ.get('_MNE_FAKE_HOME_DIR'))
    tmp_trans = tmp_path / 'tmp-trans.fif'
    # the sample subject in testing has MRI fids
    assert op.isfile(op.join(
        subjects_dir, 'sample', 'bem', 'sample-fiducials.fif'))
    coreg = coregistration(subject='sample', subjects_dir=subjects_dir,
                           trans=fname_trans)
    assert coreg._lock_fids
    coreg._reset_fiducials()
    coreg.close()

    coreg = coregistration(inst=raw_path, subject='sample',
                           subjects_dir=subjects_dir)
    coreg._set_fiducials_file(fid_fname)
    assert coreg._fiducials_file == fid_fname

    # fitting (with scaling)
    coreg._reset()
    coreg._reset_fitting_parameters()
    coreg._set_scale_mode("uniform")
    coreg._fits_fiducials()
    assert_allclose(coreg.coreg._scale,
                    np.array([97.46, 97.46, 97.46]) * 1e-2,
                    atol=1e-3)
    coreg._set_icp_fid_match("nearest")
    coreg._set_scale_mode("3-axis")
    coreg._fits_icp()
    assert_allclose(coreg.coreg._scale,
                    np.array([104.43, 101.47, 125.78]) * 1e-2,
                    atol=1e-3)
    coreg._set_scale_mode("None")
    coreg._set_icp_fid_match("matched")

    # unlock fiducials
    assert coreg._lock_fids
    coreg._set_lock_fids(False)
    assert not coreg._lock_fids

    # picking
    vtk_picker = TstVTKPicker(coreg._surfaces['head'], 0, (0, 0))
    coreg._on_mouse_move(vtk_picker, None)
    coreg._on_button_press(vtk_picker, None)
    coreg._on_pick(vtk_picker, None)
    coreg._on_button_release(vtk_picker, None)
    coreg._on_pick(vtk_picker, None)  # also pick when locked

    # lock fiducials
    assert not coreg._head_transparency
    coreg._set_lock_fids(True)
    assert coreg._lock_fids
    assert coreg._head_transparency

    # fitting (no scaling)
    assert coreg._nasion_weight == 10.
    coreg._set_point_weight(11., 'nasion')
    assert coreg._nasion_weight == 11.
    coreg._fit_fiducials()
    coreg._fit_icp()
    assert coreg.coreg._extra_points_filter is None
    coreg._omit_hsp()
    assert coreg.coreg._extra_points_filter is not None
    coreg._reset_omit_hsp_filter()
    assert coreg.coreg._extra_points_filter is None

    assert coreg._grow_hair == 0
    coreg._set_grow_hair(0.1)
    assert coreg._grow_hair == 0.1

    # visualization
    assert not coreg._helmet
    coreg._set_helmet(True)
    assert coreg._helmet
    assert coreg._orient_glyphs
    assert coreg._scale_by_distance
    assert coreg._mark_inside
    assert coreg._project_eeg == \
        (config.get('MNE_COREG_PROJECT_EEG', '') == 'true')
    assert coreg._hpi_coils
    assert coreg._eeg_channels
    assert coreg._head_shape_points
    assert coreg._scale_mode == 'None'
    assert coreg._icp_fid_match == 'matched'
    assert coreg._head_resolution == \
        (config.get('MNE_COREG_HEAD_HIGH_RES', 'true') == 'true')

    coreg._save_trans(tmp_trans)
    assert op.isfile(tmp_trans)

    coreg.close()
