import os.path as op
import numpy as np
from ..io import read_info, read_raw
from ..io.meas_info import _empty_info
from ..io.open import fiff_open, dir_tree_find
from ..channels import read_dig_fif
from ..io.constants import FIFF
from ..viz._3d import _fiducial_coords
from ..coreg import _append_fiducials


class _DigSource(object):
    def __init__(self):
        self.file = None
        self.inst_fname = None
        self.inst_dir = None
        self._info = None

        self.points_filter = None
        self.n_omitted = None

        # head shape
        self._hsp_points = None
        self.points = None

        # fiducials
        self.lpa = None
        self.nasion = None
        self.rpa = None

        # EEG
        self.eeg_points = None
        self.hpi_points = None

    def _get_n_omitted(self):
        if self.points_filter is None:
            return 0
        else:
            return np.sum(self.points_filter == False)  # noqa: E712

    def _get__info(self):
        if not self.file:
            return
        elif self.file.endswith(('.fif', '.fif.gz')):
            info = None
            fid, tree, _ = fiff_open(self.file)
            fid.close()
            if len(dir_tree_find(tree, FIFF.FIFFB_MEAS_INFO)) > 0:
                info = read_info(self.file, verbose=False)
            elif len(dir_tree_find(tree, FIFF.FIFFB_ISOTRAK)) > 0:
                info = _empty_info(1)
                info['dig'] = read_dig_fif(fname=self.file).dig
        else:
            info = read_raw(self.file).info

        # check that digitizer info is present
        if info is None or info['dig'] is None:
            raise ValueError("The selected file does not contain digitization "
                             "information. Please select a different file.",
                             "Error Reading Digitization File")
            self.reset_traits(['file'])
            return

        # check that all fiducial points are present
        point_kinds = {d['kind'] for d in info['dig']}
        missing = [key for key in ('LPA', 'Nasion', 'RPA') if
                   getattr(FIFF, f'FIFFV_POINT_{key.upper()}') not in
                   point_kinds]
        if missing:
            points = _fiducial_coords(info['dig'])
            if len(points == 3):
                _append_fiducials(info['dig'], *points.T)
            else:
                raise ValueError("The selected digitization file does not"
                                 "contain all cardinal points "
                                 f"(missing: {', '.join(missing)}). "
                                 "Please select a different file.",
                                 "Error Reading Digitization File")
                self.reset_traits(['file'])
                return
        return info

    def _get_inst_dir(self):
        return op.dirname(self.file)

    def _get_inst_fname(self):
        if self.file:
            return op.basename(self.file)
        else:
            return '-'

    def _get__hsp_points(self):
        if not self._info or not self._info['dig']:
            return np.empty((0, 3))

        points = np.array([d['r'] for d in self._info['dig']
                           if d['kind'] == FIFF.FIFFV_POINT_EXTRA])
        points = np.empty((0, 3)) if len(points) == 0 else points
        return points

    def _get_points(self):
        if self.points_filter is None:
            return self._hsp_points
        else:
            return self._hsp_points[self.points_filter]

    def _cardinal_point(self, ident):
        """Coordinates for a cardinal point."""
        if not self._info or not self._info['dig']:
            return np.zeros((1, 3))

        for d in self._info['dig']:
            if d['kind'] == FIFF.FIFFV_POINT_CARDINAL and d['ident'] == ident:
                return d['r'][None, :]
        return np.zeros((1, 3))

    def _get_nasion(self):
        return self._cardinal_point(FIFF.FIFFV_POINT_NASION)

    def _get_lpa(self):
        return self._cardinal_point(FIFF.FIFFV_POINT_LPA)

    def _get_rpa(self):
        return self._cardinal_point(FIFF.FIFFV_POINT_RPA)

    def _get_eeg_points(self):
        if not self._info or not self._info['dig']:
            return np.empty((0, 3))

        out = [d['r'] for d in self._info['dig'] if
               d['kind'] == FIFF.FIFFV_POINT_EEG and
               d['coord_frame'] == FIFF.FIFFV_COORD_HEAD]
        out = np.empty((0, 3)) if len(out) == 0 else np.array(out)
        return out

    def _get_hpi_points(self):
        if not self._info or not self._info['dig']:
            return np.zeros((0, 3))

        out = [d['r'] for d in self._info['dig'] if
               d['kind'] == FIFF.FIFFV_POINT_HPI and
               d['coord_frame'] == FIFF.FIFFV_COORD_HEAD]
        out = np.empty((0, 3)) if len(out) == 0 else np.array(out)
        return out

    def _file_changed(self):
        self.reset_traits(('points_filter',))
