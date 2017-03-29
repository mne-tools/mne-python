"""File data sources for traits GUIs."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os
import os.path as op

import numpy as np

from traits.api import (Any, HasTraits, HasPrivateTraits, cached_property,
                        on_trait_change, Array, Bool, Button, DelegatesTo,
                        Directory, Enum, Event, File, Instance, Int, List,
                        Property, Str)
from traitsui.api import View, Item, VGroup, HGroup, Label
from pyface.api import DirectoryDialog, OK, ProgressDialog, error, information

from ..bem import read_bem_surfaces
from ..io.constants import FIFF
from ..io import read_info, read_fiducials
from ..surface import read_surface
from ..coreg import (_is_mri_subject, _mri_subject_has_bem,
                     create_default_subject)
from ..utils import get_config, set_config
from ..viz._3d import _fiducial_coords


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
    points = Array(shape=(None, 3), value=np.empty((0, 3)))
    norms = Array
    tris = Array(shape=(None, 3), value=np.empty((0, 3)))

    @on_trait_change('file')
    def read_file(self):
        if op.exists(self.file):
            if self.file.endswith('.fif'):
                bem = read_bem_surfaces(self.file, verbose=False)[0]
                self.points = bem['rr']
                self.norms = bem['nn']
                self.tris = bem['tris']
            else:
                try:
                    points, tris = read_surface(self.file)
                    points /= 1e3
                    self.points = points
                    self.norms = []
                    self.tris = tris
                except Exception:
                    error(message="Error loading surface from %s (see "
                                  "Terminal for details).",
                          title="Error Loading Surface")
                    self.reset_traits(['file'])
                    raise
        else:
            self.points = np.empty((0, 3))
            self.norms = np.empty((0, 3))
            self.tris = np.empty((0, 3))


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
    points = Property(depends_on='file')

    @cached_property
    def _get_fname(self):
        fname = op.basename(self.file)
        return fname

    @cached_property
    def _get_points(self):
        if not op.exists(self.file):
            return None

        try:
            fids, coord_frame = read_fiducials(self.file)
            points = _fiducial_coords(fids, coord_frame)
            assert points.shape == (3, 3)
            return points
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

    file = File(exists=True, filter=['*.fif'])

    inst_fname = Property(Str, depends_on='file')
    inst_dir = Property(depends_on='file')
    inst = Property(depends_on='file')

    points_filter = Any(desc="Index to select a subset of the head shape "
                        "points")
    n_omitted = Property(Int, depends_on=['points_filter'])

    # head shape
    inst_points = Property(depends_on='inst', desc="Head shape points in the "
                           "inst file(n x 3 array)")
    points = Property(depends_on=['inst_points', 'points_filter'], desc="Head "
                      "shape points selected by the filter (n x 3 array)")

    # fiducials
    fid_dig = Property(depends_on='inst', desc="Fiducial points "
                       "(list of dict)")
    fid_points = Property(depends_on='fid_dig', desc="Fiducial points {ident: "
                          "point} dict}")
    lpa = Property(depends_on='fid_points', desc="LPA coordinates (1 x 3 "
                   "array)")
    nasion = Property(depends_on='fid_points', desc="Nasion coordinates (1 x "
                      "3 array)")
    rpa = Property(depends_on='fid_points', desc="RPA coordinates (1 x 3 "
                   "array)")

    # EEG
    eeg_dig = Property(depends_on='inst', desc="EEG points (list of dict)")
    eeg_points = Property(depends_on='eeg_dig', desc="EEG coordinates (N x 3 "
                          "array)")

    view = View(VGroup(Item('file'),
                       Item('inst_fname', show_label=False, style='readonly')))

    @cached_property
    def _get_n_omitted(self):
        if self.points_filter is None:
            return 0
        else:
            return np.sum(self.points_filter == False)  # noqa: E712

    @cached_property
    def _get_inst(self):
        if self.file:
            info = read_info(self.file, verbose=False)
            if info['dig'] is None:
                error(None, "The selected FIFF file does not contain "
                      "digitizer information. Please select a different "
                      "file.", "Error Reading FIFF File")
                self.reset_traits(['file'])
            else:
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
    def _get_inst_points(self):
        if not self.inst:
            return np.zeros((1, 3))

        points = np.array([d['r'] for d in self.inst['dig']
                           if d['kind'] == FIFF.FIFFV_POINT_EXTRA])
        return points

    @cached_property
    def _get_points(self):
        if self.points_filter is None:
            return self.inst_points
        else:
            return self.inst_points[self.points_filter]

    @cached_property
    def _get_fid_dig(self):
        """Get fiducials from info['dig']."""
        if not self.inst:
            return []
        dig = [d for d in self.inst['dig']
               if d['kind'] == FIFF.FIFFV_POINT_CARDINAL]
        return dig

    @cached_property
    def _get_eeg_dig(self):
        """Get EEG from info['dig']."""
        if not self.inst:
            return []
        dig = [d for d in self.inst['dig']
               if d['kind'] == FIFF.FIFFV_POINT_EEG]
        return dig

    @cached_property
    def _get_fid_points(self):
        if not self.inst:
            return {}
        return dict((d['ident'], d) for d in self.fid_dig)

    @cached_property
    def _get_nasion(self):
        if self.fid_points:
            return self.fid_points[FIFF.FIFFV_POINT_NASION]['r'][None, :]
        else:
            return np.zeros((1, 3))

    @cached_property
    def _get_lpa(self):
        if self.fid_points:
            return self.fid_points[FIFF.FIFFV_POINT_LPA]['r'][None, :]
        else:
            return np.zeros((1, 3))

    @cached_property
    def _get_rpa(self):
        if self.fid_points:
            return self.fid_points[FIFF.FIFFV_POINT_RPA]['r'][None, :]
        else:
            return np.zeros((1, 3))

    @cached_property
    def _get_eeg_points(self):
        if not self.inst or not self.eeg_dig:
            return np.empty((0, 3))
        dig = np.array([d['r'] for d in self.eeg_dig])
        return dig

    def _file_changed(self):
        self.reset_traits(('points_filter',))


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
    use_high_res_head = Bool(True)

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
            err = ("No subjects directory is selected. Please specify "
                   "subjects_dir first.")
            raise RuntimeError(err)

        fs_home = get_fs_home()
        if fs_home is None:
            err = ("FreeSurfer contains files that are needed for copying the "
                   "fsaverage brain. Please install FreeSurfer and try again.")
            raise RuntimeError(err)

        create_default_subject(fs_home=fs_home, subjects_dir=self.subjects_dir)
        self.refresh = True
        self.use_high_res_head = False
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
    use_high_res_head = DelegatesTo('model')

    create_fsaverage = Button(
        "Copy 'fsaverage' to subjects directory",
        desc="Copy the files for the fsaverage subject to the subjects "
             "directory. This button is disabled if a subject called "
             "fsaverage already exists in the selected subjects-directory.")

    view = View(VGroup(Label('Subjects directory and subject:',
                             show_label=True),
                       HGroup('subjects_dir', show_labels=False),
                       HGroup('subject', show_labels=False),
                       HGroup(Item('use_high_res_head',
                                   label='High Resolution Head',
                                   show_label=True)),
                       Item('create_fsaverage',
                            enabled_when='can_create_fsaverage'),
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
