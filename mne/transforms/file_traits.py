"""File data sources for trait GUIs"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os

import numpy as np
from traits.api import HasTraits, HasPrivateTraits, cached_property, on_trait_change, Instance, Property, \
                       Any, Array, Bool, Button, Color, Directory, Enum, File, Float, Int, List, \
                       Range, Str, Tuple
from traitsui.api import View, Item, Group, HGroup, VGroup, EnumEditor

from .coreg import is_mri_subject
from ..fiff import Raw, FIFF, read_fiducials, write_fiducials
from ..surface import read_bem_surfaces



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
    subjects_dir = Directory(exists=True)
    subjects = Property(List(Str), depends_on=['subjects_dir'])
    subject = Enum(values='subjects')
    mri_dir = Property(depends_on=['subjects_dir', 'subject'])  # path to the current subject's mri directory
    bem_file = Property(depends_on='mri_dir')

    view = View(VGroup(Item('subjects_dir', label='subjects_dir'),
                       'subject'))

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
        if sdir and os.path.isdir(sdir):
            return [s for s in os.listdir(sdir) if is_mri_subject(s, sdir)]
        else:
            return ()
