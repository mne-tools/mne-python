"""File data sources for traits GUIs"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os

import numpy as np

# allow import without traits
try:
    from traits.api import (Any, HasTraits, HasPrivateTraits, cached_property,
                            on_trait_change, Array, Bool, Button, DelegatesTo,
                            Directory, Enum, Event, File, Instance, Int, List,
                            Property, Str)
    from traitsui.api import View, Item, VGroup
    from pyface.api import (DirectoryDialog, OK, ProgressDialog, error,
                            information)
except:
    from ..utils import trait_wraith
    HasTraits = object
    HasPrivateTraits = object
    cached_property = trait_wraith
    on_trait_change = trait_wraith
    Any = trait_wraith
    Array = trait_wraith
    Bool = trait_wraith
    Button = trait_wraith
    DelegatesTo = trait_wraith
    Directory = trait_wraith
    Enum = trait_wraith
    Event = trait_wraith
    File = trait_wraith
    Instance = trait_wraith
    Int = trait_wraith
    List = trait_wraith
    Property = trait_wraith
    Str = trait_wraith
    View = trait_wraith
    Item = trait_wraith
    VGroup = trait_wraith

from ..fiff import FIFF, Raw, read_fiducials
from ..surface import read_bem_surfaces
from ..coreg import _is_mri_subject, create_default_subject
from ..utils import get_config


fid_wildcard = "*.fif"
trans_wildcard = "*.fif"
# for wx backend:
# fid_wildcard = "Fiducials FIFF file (*.fif)|*.fif"
# trans_wildcard = "Trans File (*.fif)|*.fif"


def _expand_path(p):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(p)))


def assert_env_set(mne_root=True, fs_home=False):
    """Make sure that environment variables are correctly set

    Parameters
    ----------
    mne_root : bool
        Make sure the MNE_ROOT environment variable is set correctly, and the
        mne bin directory is in the PATH.
    fs_home : bool
        Make sure the FREESURFER_HOME environment variable is set correctly.

    Returns
    -------
    success : bool
        Whether the requested environment variables are successfully set or
        not.

    Notes
    -----
    Environment variables are added to ``os.environ`` to make sure that bash
    tools can find them.
    """
    if fs_home:
        fs_home = os.environ.get('FREESURFER_HOME', None)
        test_dir = os.path.join('%s', 'subjects', 'fsaverage')
        while (fs_home is None) or not os.path.exists(test_dir % fs_home):
            msg = ("Please select the FREESURFER_HOME directory. This is the "
                   "root directory of the freesurfer installation. In order "
                   "to avoid this prompt in the future, set the "
                   "FREESURFER_HOME environment variable. "
                   "In Python, this can be done with:\n"
                   ">>> os.environ['FREESURFER_HOME'] = path")
            information(None, msg, "Select FREESURFER_HOME Directory")
            msg = "Please select the FREESURFER_HOME Directory"
            dlg = DirectoryDialog(message=msg, new_directory=False)
            if dlg.open() == OK:
                fs_home = dlg.path
            else:
                return False
        os.environ['FREESURFER_HOME'] = fs_home

    if mne_root:
        mne_root = get_config('MNE_ROOT')
        test_dir = os.path.join('%s', 'share', 'mne', 'mne_analyze')
        while (mne_root is None) or not os.path.exists(test_dir % mne_root):
            msg = ("Please select the MNE_ROOT directory. This is the root "
                   "directory of the MNE installation. In order to "
                   "avoid this prompt in the future, set the MNE_ROOT "
                   "environment variable. "
                   "In Python, this can be done with:\n"
                   ">>> os.environ['MNE_ROOT'] = path")
            information(None, msg, "Select MNE_ROOT Directory")
            msg = "Please select the MNE_ROOT Directory"
            dlg = DirectoryDialog(message=msg, new_directory=False)
            if dlg.open() == OK:
                mne_root = dlg.path
            else:
                return False
        os.environ['MNE_ROOT'] = mne_root

        # add mne bin directory to PATH
        mne_bin = os.path.realpath(os.path.join(mne_root, 'bin'))
        if mne_bin not in map(_expand_path, os.environ['PATH'].split(':')):
            os.environ['PATH'] += ':' + mne_bin

    return True


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
    tris = Array(shape=(None, 3), value=np.empty((0, 3)))

    @on_trait_change('file')
    def read_file(self):
        if os.path.exists(self.file):
            bem = read_bem_surfaces(self.file)[0]
            self.points = bem['rr']
            self.tris = bem['tris']
        else:
            self.points = np.empty((0, 3))
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


class RawSource(HasPrivateTraits):
    """Expose measurement information from a raw file

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

    raw_fname = Property(Str, depends_on='file')
    raw_dir = Property(depends_on='file')
    raw = Property(depends_on='file')

    points_filter = Any(desc="Index to select a subset of the head shape "
                        "points")
    n_omitted = Property(Int, depends_on=['points_filter'])

    # head shape
    raw_points = Property(depends_on='raw', desc="Head shape points in the "
                          "raw file(n x 3 array)")
    points = Property(depends_on=['raw_points', 'points_filter'], desc="Head "
                      "shape points selected by the filter (n x 3 array)")

    # fiducials
    fid_dig = Property(depends_on='raw', desc="Fiducial points (list of dict)")
    fid_points = Property(depends_on='fid_dig', desc="Fiducial points {ident: "
                          "point} dict}")
    lpa = Property(depends_on='fid_points', desc="LPA coordinates (1 x 3 "
                   "array)")
    nasion = Property(depends_on='fid_points', desc="Nasion coordinates (1 x "
                      "3 array)")
    rpa = Property(depends_on='fid_points', desc="RPA coordinates (1 x 3 "
                   "array)")

    view = View(VGroup(Item('file'),
                       Item('raw_fname', show_label=False, style='readonly')))

    @cached_property
    def _get_n_omitted(self):
        if self.points_filter is None:
            return 0
        else:
            return np.sum(self.points_filter == False)

    @cached_property
    def _get_raw(self):
        if self.file:
            return Raw(self.file)

    @cached_property
    def _get_raw_dir(self):
        return os.path.dirname(self.file)

    @cached_property
    def _get_raw_fname(self):
        if self.file:
            return os.path.basename(self.file)
        else:
            return '-'

    @cached_property
    def _get_raw_points(self):
        if not self.raw:
            return np.zeros((1, 3))

        points = np.array([d['r'] for d in self.raw.info['dig']
                           if d['kind'] == FIFF.FIFFV_POINT_EXTRA])
        return points

    @cached_property
    def _get_points(self):
        if self.points_filter is None:
            return self.raw_points
        else:
            return self.raw_points[self.points_filter]

    @cached_property
    def _get_fid_dig(self):
        """Fiducials for info['dig']"""
        if not self.raw:
            return []
        dig = self.raw.info['dig']
        dig = [d for d in dig if d['kind'] == FIFF.FIFFV_POINT_CARDINAL]
        return dig

    @cached_property
    def _get_fid_points(self):
        if not self.raw:
            return {}
        digs = {d['ident']: d for d in self.fid_dig}
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

    def create_fsaverage(self):
        if not self.subjects_dir:
            err = ("No subjects diretory is selected. Please specify "
                   "subjects_dir first.")
            raise RuntimeError(err)

        if not assert_env_set(mne_root=True, fs_home=True):
            err = ("Not all files required for creating the fsaverage brain "
                   "were found. Both mne and freesurfer are required.")
            raise RuntimeError(err)

        create_default_subject(subjects_dir=self.subjects_dir)
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
