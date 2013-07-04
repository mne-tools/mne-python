"""File data sources for trait GUIs"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os

import numpy as np
from traits.api import HasTraits, HasPrivateTraits, cached_property, \
                       on_trait_change, Array, Bool, Button, Directory, Enum, \
                       Event, File, List, Property, Str
from traitsui.api import View, Item, VGroup
from pyface.api import DirectoryDialog, OK, ProgressDialog, error, information

from ..fiff import Raw, read_fiducials
from ..surface import read_bem_surfaces
from ..transforms.coreg import is_mri_subject, create_default_subject
from ..utils import get_config


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
    """Dynamically updates pts and tri when file changes

    Notes
    -----
    tri is always updated after pts, so in case downstream objects depend on
    both, they should sync to a change in tri.
    """
    file = File(exists=True, filter=['*.fif'])
    pts = Array(shape=(None, 3))
    tri = Array(shape=(None, 3))

    @on_trait_change('file')
    def _get_geom(self):
        if os.path.exists(self.file):
            bem = read_bem_surfaces(self.file)[0]
            self.pts = bem['rr']
            self.tri = bem['tris']
            return bem
        else:
            self.pts = np.empty((0, 3))
            self.tri = np.empty((0, 3))


class FidSource(HasPrivateTraits):
    """Read fiducials from a fiff file"""
    file = File(exists=True, filter=['*.fif'])
    fid = Property(depends_on='file')

    @cached_property
    def _get_fid(self):
        if os.path.exists(self.file):
            dig, _ = read_fiducials(self.file)
            digs = {d['ident']: d for d in dig if d['kind'] == 1}
            nasion = digs[2]['r']
            rap = digs[1]['r']
            lap = digs[3]['r']
            return np.array([nasion, rap, lap])
        else:
            return np.zeros((3, 3))


class RawHspSource(HasPrivateTraits):
    """Extract head shape information from a raw file"""
    raw_file = File(exists=True, filter=['*.fif'])
    raw_fname = Property(Str, depends_on='raw_file')
    raw_dir = Property(depends_on='raw_file')
    raw = Property(depends_on='raw_file')
    pts = Property(depends_on='raw')
    fid = Property(depends_on='raw')
    fid_dig = Property(depends_on='raw')

    view = View(VGroup(Item('raw_file'),
                       Item('raw_fname', show_label=False, style='readonly')))

    @cached_property
    def _get_raw(self):
        if self.raw_file:
            return Raw(self.raw_file)

    @cached_property
    def _get_raw_dir(self):
        return os.path.dirname(self.raw_file)

    @cached_property
    def _get_raw_fname(self):
        if self.raw_file:
            return os.path.basename(self.raw_file)
        else:
            return '-'

    @cached_property
    def _get_fid(self):
        if not self.raw:
            return np.zeros((3, 3))

        dig = self.raw.info['dig']
        digs = {d['ident']: d for d in dig if d['kind'] == 1}
        nasion = digs[2]['r']
        rap = digs[1]['r']
        lap = digs[3]['r']

        return np.array([nasion, rap, lap])

    @cached_property
    def _get_pts(self):
        if not self.raw:
            return np.zeros((3, 3))

        pts = filter(lambda d: d['kind'] == 4, self.raw.info['dig'])
        pts = np.array([d['r'] for d in pts])
        return pts

    @cached_property
    def _get_fid_dig(self):
        """Fiducials for info['dig']"""
        if not self.raw:
            return []

        dig = self.raw.info['dig']
        dig = [d for d in dig if d['kind'] == 1]
        return dig


class SubjectSelector(HasPrivateTraits):
    """Select a subjects directory and a subject it contains"""
    refresh = Event(desc="Refresh the subject list based on the directory "
                    "structure of subjects_dir.")

    can_create_fsaverage = Bool(False)
    create_fsaverage = Button("Create FsAverage", desc="Create the fsaverage "
                              "brain in subjects_dir.")

    subjects_dir = Directory(exists=True)
    subjects = Property(List(Str), depends_on=['subjects_dir', 'refresh'])
    subject = Enum(values='subjects')
    mri_dir = Property(depends_on=['subjects_dir', 'subject'], desc="the "
                       "current subject's mri directory")
    bem_file = Property(depends_on='mri_dir')

    view = View(VGroup(Item('subjects_dir', label='subjects_dir'),
                       'subject',
                       Item('create_fsaverage', show_label=False,
                            enabled_when='can_create_fsaverage')))

    def _create_fsaverage_fired(self):
        if not self.subjects_dir:
            error(None, "No subjects diretory is selected. Please specify "
                  "subjects_dir first.", "No SUBJECTS_DIR")
            return

        if not assert_env_set(mne_root=True, fs_home=True):
            error(None, "Not all files required for creating the fsaverage brain\n"
                   "were found. Both mne and freesurfer are required.",
                   "Error Creating FsAverage")
            return

        # progress dialog with indefinite progress bar
        title = "Creating FsAverage ..."
        message = "Copying fsaverage files ..."
        prog = ProgressDialog(title=title, message=message)
        prog.open()
        prog.update(0)

        try:
            create_default_subject(subjects_dir=self.subjects_dir)
        except Exception as err:
            msg = str(err)
            error(None, msg, "Error Creating FsAverage")
        else:
            self.refresh = True
            self.subject = 'fsaverage'
        prog.close()

    @cached_property
    def _get_bem_file(self):
        if not self.mri_dir:
            return

        fname = os.path.join(self.mri_dir, 'bem', self.subject + '-%s.fif')
        return fname

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
            subjects = [s for s in dir_content if is_mri_subject(s, sdir)]
            if len(subjects) == 0:
                subjects.append('')
        else:
            subjects = ['']

        if is_dir and ('fsaverage' not in dir_content):
            self.can_create_fsaverage = True
        else:
            self.can_create_fsaverage = False

        return subjects
