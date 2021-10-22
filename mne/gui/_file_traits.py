# -*- coding: utf-8 -*-
"""File data sources for traits GUIs."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

import os
import os.path as op

import numpy as np

from traits.api import (Any, HasTraits, HasPrivateTraits, cached_property,
                        on_trait_change, Array, Bool, Button, DelegatesTo,
                        Directory, Enum, Event, File, Instance, Int, List,
                        Property, Str, ArrayOrNone, BaseFile)
from traitsui.api import View, Item, VGroup
from pyface.api import DirectoryDialog, OK, ProgressDialog, error, information

from ._viewer import _DIG_SOURCE_WIDTH

from ..bem import read_bem_surfaces
from ..io.constants import FIFF
from ..io import read_info, read_fiducials, read_raw
from ..io._read_raw import supported
from ..io.meas_info import _empty_info
from ..io.open import fiff_open, dir_tree_find
from ..surface import read_surface, complete_surface_info
from ..coreg import (_is_mri_subject, _mri_subject_has_bem,
                     create_default_subject)
from ..utils import get_config, set_config
from ..viz._3d import _fiducial_coords
from ..channels import read_dig_fif


fid_wildcard = "*.fif"
trans_wildcard = "*.fif"
# for wx backend:
# fid_wildcard = "Fiducials FIFF file (*.fif)|*.fif"
# trans_wildcard = "Trans File (*.fif)|*.fif"


def _expand_path(p):
    return op.abspath(op.expandvars(op.expanduser(p)))


def get_fs_home():
    """Get the FREESURFER_HOME directory.

    Returns
    -------
    fs_home : None | str
        The FREESURFER_HOME path or None if the user cancels.

    Notes
    -----
    If FREESURFER_HOME can't be found, the user is prompted with a file dialog.
    If specified successfully, the resulting path is stored with
    mne.set_config().
    """
    return _get_root_home('FREESURFER_HOME', 'freesurfer', _fs_home_problem)


def _get_root_home(cfg, name, check_fun):
    root = get_config(cfg)
    problem = check_fun(root)
    while problem:
        info = ("Please select the %s directory. This is the root "
                "directory of the %s installation." % (cfg, name))
        msg = '\n\n'.join((problem, info))
        information(None, msg, "Select the %s Directory" % cfg)
        msg = "Please select the %s Directory" % cfg
        dlg = DirectoryDialog(message=msg, new_directory=False)
        if dlg.open() == OK:
            root = dlg.path
            problem = check_fun(root)
            if problem is None:
                set_config(cfg, root, set_env=False)
        else:
            return None
    return root


def set_fs_home():
    """Set the FREESURFER_HOME environment variable.

    Returns
    -------
    success : bool
        True if the environment variable could be set, False if FREESURFER_HOME
        could not be found.

    Notes
    -----
    If FREESURFER_HOME can't be found, the user is prompted with a file dialog.
    If specified successfully, the resulting path is stored with
    mne.set_config().
    """
    fs_home = get_fs_home()
    if fs_home is None:
        return False
    else:
        os.environ['FREESURFER_HOME'] = fs_home
        return True


def _fs_home_problem(fs_home):
    """Check FREESURFER_HOME path.

    Return str describing problem or None if the path is okay.
    """
    if fs_home is None:
        return "FREESURFER_HOME is not set."
    elif not op.exists(fs_home):
        return "FREESURFER_HOME (%s) does not exist." % fs_home
    else:
        test_dir = op.join(fs_home, 'subjects', 'fsaverage')
        if not op.exists(test_dir):
            return ("FREESURFER_HOME (%s) does not contain the fsaverage "
                    "subject." % fs_home)


def _mne_root_problem(mne_root):
    """Check MNE_ROOT path.

    Return str describing problem or None if the path is okay.
    """
    if mne_root is None:
        return "MNE_ROOT is not set."
    elif not op.exists(mne_root):
        return "MNE_ROOT (%s) does not exist." % mne_root
    else:
        test_dir = op.join(mne_root, 'share', 'mne', 'mne_analyze')
        if not op.exists(test_dir):
            return ("MNE_ROOT (%s) is missing files. If this is your MNE "
                    "installation, consider reinstalling." % mne_root)


class FileOrDir(File):
    """Subclass File because *.mff files are actually directories."""

    def validate(self, object, name, value):
        """Validate that a specified value is valid for this trait."""
        value = os.fspath(value)
        validated_value = super(BaseFile, self).validate(object, name, value)
        if not self.exists:
            return validated_value
        elif op.exists(value):
            return validated_value

        self.error(object, name, value)


class Surf(HasTraits):
    """Expose a surface similar to the ones used elsewhere in MNE."""

    rr = Array(shape=(None, 3), value=np.empty((0, 3)))
    nn = Array(shape=(None, 3), value=np.empty((0, 3)))
    tris = Array(shape=(None, 3), value=np.empty((0, 3)))


class SurfaceSource(HasTraits):
    """Expose points and tris of a file storing a surface.

    Parameters
    ----------
    file : File
        Path to a *-bem.fif file or a surface containing a Freesurfer surface.

    Attributes
    ----------
    pts : Array, shape = (n_pts, 3)
        Point coordinates.
    tris : Array, shape = (n_tri, 3)
        Triangles.

    Notes
    -----
    tri is always updated after pts, so in case downstream objects depend on
    both, they should sync to a change in tris.
    """

    file = File(exists=True, filter=['*.fif', '*.*'])
    surf = Instance(Surf)

    @on_trait_change('file')
    def read_file(self):
        """Read the file."""
        if op.exists(self.file):
            if self.file.endswith('.fif'):
                bem = read_bem_surfaces(
                    self.file, on_defects='warn', verbose=False
                )[0]
            else:
                try:
                    bem = read_surface(self.file, return_dict=True)[2]
                    bem['rr'] *= 1e-3
                    complete_surface_info(bem, copy=False)
                except Exception:
                    error(parent=None,
                          message="Error loading surface from %s (see "
                                  "Terminal for details)." % self.file,
                          title="Error Loading Surface")
                    self.reset_traits(['file'])
                    raise
            self.surf = Surf(rr=bem['rr'], tris=bem['tris'], nn=bem['nn'])
        else:
            self.surf = self._default_surf()

    def _surf_default(self):
        return Surf(rr=np.empty((0, 3)),
                    tris=np.empty((0, 3), int), nn=np.empty((0, 3)))


class FiducialsSource(HasTraits):
    """Expose points of a given fiducials fif file.

    Parameters
    ----------
    file : File
        Path to a fif file with fiducials (*.fif).

    Attributes
    ----------
    points : Array, shape = (n_points, 3)
        Fiducials file points.
    """

    file = File(filter=[fid_wildcard])
    fname = Property(depends_on='file')
    points = Property(ArrayOrNone, depends_on='file')
    mni_points = ArrayOrNone(float, shape=(3, 3))

    def _get_fname(self):
        return op.basename(self.file)

    @cached_property
    def _get_points(self):
        if not op.exists(self.file):
            return self.mni_points  # can be None
        try:
            return _fiducial_coords(*read_fiducials(self.file))
        except Exception as err:
            error(None, "Error reading fiducials from %s: %s (See terminal "
                  "for more information)" % (self.fname, str(err)),
                  "Error Reading Fiducials")
            self.reset_traits(['file'])
            raise


class DigSource(HasPrivateTraits):
    """Expose digitization information from a file.

    Parameters
    ----------
    file : File
        Path to the BEM file (*.fif).

    Attributes
    ----------
    fid : Array, shape = (3, 3)
        Each row contains the coordinates for one fiducial point, in the order
        Nasion, RAP, LAP. If no file is set all values are 0.
    """

    file = FileOrDir(exists=True,
                     filter=[' '.join([f'*{ext}' for ext in supported])])

    inst_fname = Property(Str, depends_on='file')
    inst_dir = Property(depends_on='file')
    _info = Property(depends_on='file')

    points_filter = Any(desc="Index to select a subset of the head shape "
                             "points")
    n_omitted = Property(Int, depends_on=['points_filter'])

    # head shape
    _hsp_points = Property(depends_on='_info',
                           desc="Head shape points in the file (n x 3 array)")
    points = Property(depends_on=['_hsp_points', 'points_filter'],
                      desc="Head shape points selected by the filter (n x 3 "
                           "array)")

    # fiducials
    lpa = Property(depends_on='_info',
                   desc="LPA coordinates (1 x 3 array)")
    nasion = Property(depends_on='_info',
                      desc="Nasion coordinates (1 x 3 array)")
    rpa = Property(depends_on='_info',
                   desc="RPA coordinates (1 x 3 array)")

    # EEG
    eeg_points = Property(depends_on='_info',
                          desc="EEG sensor coordinates (N x 3 array)")
    hpi_points = Property(depends_on='_info',
                          desc='HPI coil coordinates (N x 3 array)')

    view = View(Item('file', width=_DIG_SOURCE_WIDTH, tooltip='FIF file '
                     '(Raw, Epochs, Evoked, or DigMontage)', show_label=False))

    @cached_property
    def _get_n_omitted(self):
        if self.points_filter is None:
            return 0
        else:
            return np.sum(self.points_filter == False)  # noqa: E712

    @cached_property
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
                info._unlocked = False
        else:
            info = read_raw(self.file).info

        # check that digitizer info is present
        if info is None or info['dig'] is None:
            error(None, "The selected file does not contain digitization "
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
                error(None, "The selected digitization file does not contain "
                      f"all cardinal points (missing: {', '.join(missing)}). "
                      "Please select a different file.",
                      "Error Reading Digitization File")
                self.reset_traits(['file'])
                return
        return info

    @cached_property
    def _get_inst_dir(self):
        return op.dirname(self.file)

    @cached_property
    def _get_inst_fname(self):
        if self.file:
            return op.basename(self.file)
        else:
            return '-'

    @cached_property
    def _get__hsp_points(self):
        if not self._info or not self._info['dig']:
            return np.empty((0, 3))

        points = np.array([d['r'] for d in self._info['dig']
                           if d['kind'] == FIFF.FIFFV_POINT_EXTRA])
        points = np.empty((0, 3)) if len(points) == 0 else points
        return points

    @cached_property
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

    @cached_property
    def _get_nasion(self):
        return self._cardinal_point(FIFF.FIFFV_POINT_NASION)

    @cached_property
    def _get_lpa(self):
        return self._cardinal_point(FIFF.FIFFV_POINT_LPA)

    @cached_property
    def _get_rpa(self):
        return self._cardinal_point(FIFF.FIFFV_POINT_RPA)

    @cached_property
    def _get_eeg_points(self):
        if not self._info or not self._info['dig']:
            return np.empty((0, 3))

        out = [d['r'] for d in self._info['dig'] if
               d['kind'] == FIFF.FIFFV_POINT_EEG and
               d['coord_frame'] == FIFF.FIFFV_COORD_HEAD]
        out = np.empty((0, 3)) if len(out) == 0 else np.array(out)
        return out

    @cached_property
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


def _append_fiducials(dig, lpa, nasion, rpa):
    dig.append({'coord_frame': FIFF.FIFFV_COORD_HEAD,
                'ident': FIFF.FIFFV_POINT_LPA,
                'kind': FIFF.FIFFV_POINT_CARDINAL,
                'r': lpa})
    dig.append({'coord_frame': FIFF.FIFFV_COORD_HEAD,
                'ident': FIFF.FIFFV_POINT_NASION,
                'kind': FIFF.FIFFV_POINT_CARDINAL,
                'r': nasion})
    dig.append({'coord_frame': FIFF.FIFFV_COORD_HEAD,
                'ident': FIFF.FIFFV_POINT_RPA,
                'kind': FIFF.FIFFV_POINT_CARDINAL,
                'r': rpa})


class MRISubjectSource(HasPrivateTraits):
    """Find subjects in SUBJECTS_DIR and select one.

    Parameters
    ----------
    subjects_dir : directory
        SUBJECTS_DIR.
    subject : str
        Subject, corresponding to a folder in SUBJECTS_DIR.
    """

    refresh = Event(desc="Refresh the subject list based on the directory "
                    "structure of subjects_dir.")

    # settings
    subjects_dir = Directory(exists=True)
    subjects = Property(List(Str), depends_on=['subjects_dir', 'refresh'])
    subject = Enum(values='subjects')

    # info
    can_create_fsaverage = Property(Bool, depends_on=['subjects_dir',
                                                      'subjects'])
    subject_has_bem = Property(Bool, depends_on=['subjects_dir', 'subject'],
                               desc="whether the subject has a file matching "
                               "the bem file name pattern")
    bem_pattern = Property(depends_on='mri_dir')

    @cached_property
    def _get_can_create_fsaverage(self):
        if not op.exists(self.subjects_dir) or 'fsaverage' in self.subjects:
            return False
        return True

    @cached_property
    def _get_mri_dir(self):
        if not self.subject:
            return
        elif not self.subjects_dir:
            return
        else:
            return op.join(self.subjects_dir, self.subject)

    @cached_property
    def _get_subjects(self):
        sdir = self.subjects_dir
        is_dir = sdir and op.isdir(sdir)
        if is_dir:
            dir_content = os.listdir(sdir)
            subjects = [s for s in dir_content if _is_mri_subject(s, sdir)]
            if len(subjects) == 0:
                subjects.append('')
        else:
            subjects = ['']

        return sorted(subjects)

    @cached_property
    def _get_subject_has_bem(self):
        if not self.subject:
            return False
        return _mri_subject_has_bem(self.subject, self.subjects_dir)

    def create_fsaverage(self):  # noqa: D102
        if not self.subjects_dir:
            raise RuntimeError(
                "No subjects directory is selected. Please specify "
                "subjects_dir first.")

        fs_home = get_fs_home()
        if fs_home is None:
            raise RuntimeError(
                "FreeSurfer contains files that are needed for copying the "
                "fsaverage brain. Please install FreeSurfer and try again.")

        create_default_subject(fs_home=fs_home, update=True,
                               subjects_dir=self.subjects_dir)
        self.refresh = True
        self.subject = 'fsaverage'

    @on_trait_change('subjects_dir')
    def _emit_subject(self):
        # This silliness is the only way I could figure out to get the
        # on_trait_change('subject_panel.subject') in CoregFrame to work!
        self.subject = self.subject


class SubjectSelectorPanel(HasPrivateTraits):
    """Subject selector panel."""

    model = Instance(MRISubjectSource)

    can_create_fsaverage = DelegatesTo('model')
    subjects_dir = DelegatesTo('model')
    subject = DelegatesTo('model')
    subjects = DelegatesTo('model')

    create_fsaverage = Button(
        u"fsaverage⇨SUBJECTS_DIR",
        desc="whether to copy the files for the fsaverage subject to the "
             "subjects directory. This button is disabled if "
             "fsaverage already exists in the selected subjects directory.")

    view = View(VGroup(Item('subjects_dir', width=_DIG_SOURCE_WIDTH,
                            tooltip='Subject MRI structurals (SUBJECTS_DIR)'),
                       Item('subject', width=_DIG_SOURCE_WIDTH,
                            tooltip='Subject to use within SUBJECTS_DIR'),
                       Item('create_fsaverage',
                            enabled_when='can_create_fsaverage',
                            width=_DIG_SOURCE_WIDTH),
                       show_labels=False))

    def _create_fsaverage_fired(self):
        # progress dialog with indefinite progress bar
        title = "Creating FsAverage ..."
        message = "Copying fsaverage files ..."
        prog = ProgressDialog(title=title, message=message)
        prog.open()
        prog.update(0)

        try:
            self.model.create_fsaverage()
        except Exception as err:
            error(None, str(err), "Error Creating FsAverage")
            raise
        finally:
            prog.close()

    def _subjects_dir_changed(self, old, new):
        if new and self.subjects == ['']:
            information(None, "The directory selected as subjects-directory "
                        "(%s) does not contain any valid MRI subjects. If "
                        "this is not expected make sure all MRI subjects have "
                        "head surface model files which "
                        "can be created by running:\n\n    $ mne "
                        "make_scalp_surfaces" % self.subjects_dir,
                        "No Subjects Found")
