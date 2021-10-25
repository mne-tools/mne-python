import pytest
import os
import os.path as op
from mne.datasets import testing

data_path = testing.data_path(download=False)
subjects_dir = os.path.join(data_path, 'subjects')
fid_fname = op.join(subjects_dir, 'sample', 'bem', 'sample-fiducials.fif')
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
trans_fname = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')


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
def test_coregistration_ui(tmpdir):
    """Test that using CoregistrationUI matches mne coreg."""
    from mne.gui._coreg import CoregistrationUI
    tempdir = str(tmpdir)
    tmp_trans = op.join(tempdir, 'tmp-trans.fif')
    coreg = CoregistrationUI(info_file=None, subject='sample',
                             subjects_dir=subjects_dir, trans=trans_fname,
                             show=False)
    coreg.close()
    coreg = CoregistrationUI(info_file=raw_fname, subject='sample',
                             subjects_dir=subjects_dir, show=False)
    coreg._set_fiducials_file(fid_fname)
    assert coreg._fiducials_file == fid_fname
    # picking
    vtk_picker = TstVTKPicker(coreg._surfaces['head'], 0, (0, 0))
    coreg._on_mouse_move(vtk_picker, None)
    coreg._on_button_press(vtk_picker, None)
    coreg._on_pick(vtk_picker, None)
    coreg._on_button_release(vtk_picker, None)
    coreg._set_lock_fids(True)
    assert coreg._lock_fids
    coreg._on_pick(vtk_picker, None)  # also pick when locked
    coreg._set_lock_fids(False)
    assert not coreg._lock_fids
    coreg._set_lock_fids(True)
    assert coreg._lock_fids
    assert coreg._nasion_weight == 10.
    coreg._set_point_weight(11., 'nasion')
    assert coreg._nasion_weight == 11.
    coreg._fit_fiducials()
    coreg._fit_icp()
    coreg._omit_hsp()
    assert coreg._coreg._extra_points_filter is not None
    coreg._reset_omit_hsp_filter()
    assert coreg._coreg._extra_points_filter is None
    coreg._set_orient_glyphs(True)
    assert coreg._orient_glyphs
    coreg._set_head_resolution(True)
    assert coreg._head_resolution
    coreg._set_head_transparency(True)
    assert coreg._head_transparency
    coreg._save_trans(tmp_trans)
    assert op.isfile(tmp_trans)
    coreg.close()
