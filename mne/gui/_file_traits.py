"""File data sources for traits GUIs"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os

import numpy as np
from ..externals.six.moves import map

# allow import without traits
try:
    from traits.api import (Any, HasTraits, HasPrivateTraits, cached_property,
                            on_trait_change, Array, Bool, Button, DelegatesTo,
                            Directory, Enum, Event, File, Instance, Int, List,
                            Property, Str)
    from traitsui.api import View, Item, VGroup
    from pyface.api import (DirectoryDialog, OK, ProgressDialog, error,
                            information)
except Exception:
    from ..utils import trait_wraith
    HasTraits = HasPrivateTraits = object
    cached_property = on_trait_change = Any = Array = Bool = Button = \
        DelegatesTo = Directory = Enum = Event = File = Instance = \
        Int = List = Property = Str = View = Item = VGroup = trait_wraith

from ..io.constants import FIFF
from ..io import read_info, read_fiducials
from ..surface import read_bem_surfaces
from ..coreg import (_is_mri_subject, _mri_subject_has_bem,
                     create_default_subject)
from ..utils import get_config, set_config


fid_wildcard = "*.fif"
trans_wildcard = "*.fif"
# for wx backend:
# fid_wildcard = "Fiducials FIFF file (*.fif)|*.fif"
# trans_wildcard = "Trans File (*.fif)|*.fif"


def _expand_path(p):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(p)))


def get_fs_home():
    """Get the FREESURFER_HOME directory

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


def get_mne_root():
    """Get the MNE_ROOT directory

    Returns
    -------
    mne_root : None | str
        The MNE_ROOT path or None if the user cancels.

    Notes
    -----
    If MNE_ROOT can't be found, the user is prompted with a file dialog.
    If specified successfully, the resulting path is stored with
    mne.set_config().
    """
    return _get_root_home('MNE_ROOT', 'MNE', _mne_root_problem)


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
                set_config(cfg, root)
        else:
            return None
    return root


def set_fs_home():
    """Set the FREESURFER_HOME environment variable

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
    """Check FREESURFER_HOME path

    Return str describing problem or None if the path is okay.
    """
    if fs_home is None:
        return "FREESURFER_HOME is not set."
    elif not os.path.exists(fs_home):
        return "FREESURFER_HOME (%s) does not exist." % fs_home
    else:
        test_dir = os.path.join(fs_home, 'subjects', 'fsaverage')
        if not os.path.exists(test_dir):
            return ("FREESURFER_HOME (%s) does not contain the fsaverage "
                    "subject." % fs_home)


def set_mne_root(set_mne_bin=False):
    """Set the MNE_ROOT environment variable

    Parameters
    ----------
    set_mne_bin : bool
        Also add the MNE binary directory to the PATH (default: False).

    Returns
    -------
    success : bool
        True if the environment variable could be set, False if MNE_ROOT
        could not be found.

    Notes
    -----
    If MNE_ROOT can't be found, the user is prompted with a file dialog.
    If specified successfully, the resulting path is stored with
    mne.set_config().
    """
    mne_root = get_mne_root()
    if mne_root is None:
        return False
    else:
        os.environ['MNE_ROOT'] = mne_root
        if set_mne_bin:
            mne_bin = os.path.realpath(os.path.join(mne_root, 'bin'))
            if mne_bin not in map(_expand_path, os.environ['PATH'].split(':')):
                os.environ['PATH'] += ':' + mne_bin
        return True


def _mne_root_problem(mne_root):
    """Check MNE_ROOT path

    Return str describing problem or None if the path is okay.
    """
    if mne_root is None:
        return "MNE_ROOT is not set."
    elif not os.path.exists(mne_root):
        return "MNE_ROOT (%s) does not exist." % mne_root
    else:
        test_dir = os.path.join(mne_root, 'share', 'mne', 'mne_analyze')
        if not os.path.exists(test_dir):
            return ("MNE_ROOT (%s) is missing files. If this is your MNE "
                    "installation, consider reinstalling." % mne_root)


class BemSource(HasTraits):
    """Expose points and tris of a given BEM file

    Parameters
    ----------
    file : File
        Path to the BEM file (*.fif).

    Attributes
    ----------
    pts : Array, shape = (n_pts, 3)
        BEM file points.
    tri : Array, shape = (n_tri, 3)
        BEM file triangles.

    Notes
    -----
    tri is always updated after pts, so in case downstream objects depend on
    both, they should sync to a change in tri.
    """
    file = File(exists=True, filter=['*.fif'])
    points = Array(shape=(None, 3), value=np.empty((0, 3)))
    norms = Array
    tris = Array(shape=(None, 3), value=np.empty((0, 3)))

    @on_trait_change('file')
    def read_file(self):
        if os.path.exists(self.file):
            bem = read_bem_surfaces(self.file)[0]
            self.points = bem['rr']
            self.norms = bem['nn']
            self.tris = bem['tris']
        else:
            self.points = np.empty((0, 3))
            self.norms = np.empty((0, 3))
            self.tris = np.empty((0, 3))


class FiducialsSource(HasTraits):
    """Expose points of a given fiducials fif file

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
        fname = os.path.basename(self.file)
        return fname

    @cached_property
    def _get_points(self):
        if not os.path.exists(self.file):
            return None

        points = np.zeros((3, 3))
        fids, _ = read_fiducials(self.file)
        for fid in fids:
            ident = fid['ident']
            if ident == FIFF.FIFFV_POINT_LPA:
                points[0] = fid['r']
            elif ident == FIFF.FIFFV_POINT_NASION:
                points[1] = fid['r']
            elif ident == FIFF.FIFFV_POINT_RPA:
                points[2] = fid['r']
        return points


class InstSource(HasPrivateTraits):
    """Expose measurement information from a inst file

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

    view = View(VGroup(Item('file'),
                       Item('inst_fname', show_label=False, style='readonly')))

    @cached_property
    def _get_n_omitted(self):
        if self.points_filter is None:
            return 0
        else:
            return np.sum(self.points_filter == False)  # noqa

    @cached_property
    def _get_inst(self):
        if self.file:
            return read_info(self.file)

    @cached_property
    def _get_inst_dir(self):
        return os.path.dirname(self.file)

    @cached_property
    def _get_inst_fname(self):
        if self.file:
            return os.path.basename(self.file)
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
        """Fiducials for info['dig']"""
        if not self.inst:
            return []
        dig = self.inst['dig']
        dig = [d for d in dig if d['kind'] == FIFF.FIFFV_POINT_CARDINAL]
        return dig

    @cached_property
    def _get_fid_points(self):
        if not self.inst:
            return {}
        digs = dict((d['ident'], d) for d in self.fid_dig)
        return digs

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

    def _file_changed(self):
        self.reset_traits(('points_filter',))


class MRISubjectSource(HasPrivateTraits):
    """Find subjects in SUBJECTS_DIR and select one

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
        if not os.path.exists(self.subjects_dir):
            return False
        if 'fsaverage' in self.subjects:
            return False
        return True

    @cached_property
    def _get_mri_dir(self):
        if not self.subject:
            return
        elif not self.subjects_dir:
            return
        else:
            return os.path.join(self.subjects_dir, self.subject)

    @cached_property
    def _get_subjects(self):
        sdir = self.subjects_dir
        is_dir = sdir and os.path.isdir(sdir)
        if is_dir:
            dir_content = os.listdir(sdir)
            subjects = [s for s in dir_content if _is_mri_subject(s, sdir)]
            if len(subjects) == 0:
                subjects.append('')
        else:
            subjects = ['']

        return subjects

    @cached_property
    def _get_subject_has_bem(self):
        if not self.subject:
            return False
        return _mri_subject_has_bem(self.subject, self.subjects_dir)

    def create_fsaverage(self):
        if not self.subjects_dir:
            err = ("No subjects directory is selected. Please specify "
                   "subjects_dir first.")
            raise RuntimeError(err)

        mne_root = get_mne_root()
        if mne_root is None:
            err = ("MNE contains files that are needed for copying the "
                   "fsaverage brain. Please install MNE and try again.")
            raise RuntimeError(err)
        fs_home = get_fs_home()
        if fs_home is None:
            err = ("FreeSurfer contains files that are needed for copying the "
                   "fsaverage brain. Please install FreeSurfer and try again.")
            raise RuntimeError(err)

        create_default_subject(mne_root, fs_home,
                               subjects_dir=self.subjects_dir)
        self.refresh = True
        self.subject = 'fsaverage'


class SubjectSelectorPanel(HasPrivateTraits):
    model = Instance(MRISubjectSource)

    can_create_fsaverage = DelegatesTo('model')
    subjects_dir = DelegatesTo('model')
    subject = DelegatesTo('model')
    subjects = DelegatesTo('model')

    create_fsaverage = Button("Copy FsAverage to Subjects Folder",
                              desc="Copy the files for the fsaverage subject "
                              "to the subjects directory.")

    view = View(VGroup(Item('subjects_dir', label='subjects_dir'),
                       'subject',
                       Item('create_fsaverage', show_label=False,
                            enabled_when='can_create_fsaverage')))

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
            msg = str(err)
            error(None, msg, "Error Creating FsAverage")
            raise
        finally:
            prog.close()
