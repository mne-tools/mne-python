# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Oleh Kozynets <ok7mailbox@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#          jona-sassenhagen <jona.sassenhagen@gmail.com>
#
# License: Simplified BSD

from os import path as path

import numpy as np
from ...utils import (_check_option, get_subjects_dir, _check_fname,
                      _validate_type)
from ...surface import (complete_surface_info, read_surface, read_curvature,
                        _read_patch)


class _Surface(object):
    """Container for a brain surface.

    It is used for storing vertices, faces and morphometric data
    (curvature) of a hemisphere mesh.

    Parameters
    ----------
    subject_id : string
        Name of subject
    hemi : {'lh', 'rh'}
        Which hemisphere to load
    surf : string
        Name of the surface to load (eg. inflated, orig ...).
    subjects_dir : str | None
        If not None, this directory will be used as the subjects directory
        instead of the value set using the SUBJECTS_DIR environment variable.
    offset : float | None
        If 0.0, the surface will be offset such that the medial
        wall is aligned with the origin. If None, no offset will
        be applied. If != 0.0, an additional offset will be used.
    units : str
        Can be 'm' or 'mm' (default).
    x_dir : ndarray | None
        The x direction to use for offset alignment.

    Attributes
    ----------
    bin_curv : numpy.ndarray
        Curvature values stored as non-negative integers.
    coords : numpy.ndarray
        nvtx x 3 array of vertex (x, y, z) coordinates.
    curv : numpy.ndarray
        Vector representation of surface morpometry (curvature) values as
        loaded from a file.
    grey_curv : numpy.ndarray
        Normalized morphometry (curvature) data, used in order to get
        a gray cortex.
    faces : numpy.ndarray
        nfaces x 3 array of defining mesh triangles.
    hemi : {'lh', 'rh'}
        Which hemisphere to load.
    nn : numpy.ndarray
        Vertex normals for a triangulated surface.
    offset : float | None
        If float, align inside edge of each hemisphere to center + offset.
        If None, do not change coordinates (default).
    subject_id : string
        Name of subject.
    surf : string
        Name of the surface to load (eg. inflated, orig ...).
    units : str
        Can be 'm' or 'mm' (default).
    """

    def __init__(self, subject_id, hemi, surf, subjects_dir=None, offset=None,
                 units='mm', x_dir=None):

        x_dir = np.array([1., 0, 0]) if x_dir is None else x_dir
        assert isinstance(x_dir, np.ndarray)
        assert np.isclose(np.linalg.norm(x_dir), 1., atol=1e-6)
        assert hemi in ('lh', 'rh')
        _validate_type(offset, (None, 'numeric'), 'offset')

        self.units = _check_option('units', units, ('mm', 'm'))
        self.subject_id = subject_id
        self.hemi = hemi
        self.surf = surf
        self.offset = offset
        self.bin_curv = None
        self.coords = None
        self.curv = None
        self.faces = None
        self.grey_curv = None
        self.nn = None
        self.labels = dict()
        self.x_dir = x_dir

        subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
        self.data_path = path.join(subjects_dir, subject_id)
        if surf == 'seghead':
            raise ValueError('`surf` cannot be seghead, use '
                             '`mne.viz.Brain.add_head` to plot the seghead')

    def load_geometry(self):
        """Load geometry of the surface.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.surf == 'flat':  # special case
            fname = path.join(self.data_path, 'surf',
                              '%s.%s' % (self.hemi, 'cortex.patch.flat'))
            _check_fname(fname, overwrite='read', must_exist=True,
                         name='flatmap surface file')
            coords, faces, orig_faces = _read_patch(fname)
            # rotate 90 degrees to get to a more standard orientation
            # where X determines the distance between the hemis
            coords = coords[:, [1, 0, 2]]
            coords[:, 1] *= -1
        else:
            coords, faces = read_surface(
                path.join(self.data_path, 'surf',
                          '%s.%s' % (self.hemi, self.surf)))
            orig_faces = faces
        if self.units == 'm':
            coords /= 1000.
        if self.offset is not None:
            x_ = coords @ self.x_dir
            if self.hemi == 'lh':
                coords -= (np.max(x_) + self.offset) * self.x_dir
            else:
                coords -= (np.min(x_) + self.offset) * self.x_dir
        surf = dict(rr=coords, tris=faces)
        complete_surface_info(
            surf, copy=False, verbose=False, do_neighbor_tri=False)
        nn = surf['nn']
        self.coords = coords
        self.faces = faces
        self.orig_faces = orig_faces
        self.nn = nn

    def __len__(self):
        """Return number of vertices."""
        return len(self.coords)

    @property
    def x(self):
        return self.coords[:, 0]

    @property
    def y(self):
        return self.coords[:, 1]

    @property
    def z(self):
        return self.coords[:, 2]

    def load_curvature(self):
        """Load in curvature values from the ?h.curv file."""
        curv_path = path.join(self.data_path, 'surf', '%s.curv' % self.hemi)
        self.curv = read_curvature(curv_path, binary=False)
        self.bin_curv = np.array(self.curv > 0, np.int64)
        # morphometry (curvature) normalization in order to get gray cortex
        # TODO: delete self.grey_curv after cortex parameter
        # will be fully supported
        color = (self.curv > 0).astype(float)
        color = 0.5 - (color - 0.5) / 3
        color = color[:, np.newaxis] * [1, 1, 1]
        self.grey_curv = color
