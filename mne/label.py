# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

from collections import defaultdict
from colorsys import hsv_to_rgb, rgb_to_hsv
from os import path as op
import os
import copy as cp
import re

import numpy as np
from scipy import linalg, sparse

from .fixes import digitize, in1d
from .utils import (get_subjects_dir, _check_subject, logger, verbose,
                    deprecated)
from .source_estimate import (_read_stc, mesh_edges, mesh_dist, morph_data,
                              SourceEstimate, spatial_src_connectivity)
from .source_space import add_source_space_distances
from .surface import read_surface, fast_cross_3d
from .source_space import SourceSpaces
from .parallel import parallel_func, check_n_jobs
from .stats.cluster_level import _find_clusters
from .externals.six import b, string_types
from .externals.six.moves import zip, xrange


def _blend_colors(color_1, color_2):
    """Blend two colors in HSV space

    Parameters
    ----------
    color_1, color_2 : None | tuple
        RGBA tuples with values between 0 and 1. None if no color is available.
        If both colors are None, the output is None. If only one is None, the
        output is the other color.

    Returns
    -------
    color : None | tuple
        RGBA tuple of the combined color. Saturation, value and alpha are
        averaged, whereas the new hue is determined as angle half way between
        the two input colors' hues.
    """
    if color_1 is None and color_2 is None:
        return None
    elif color_1 is None:
        return color_2
    elif color_2 is None:
        return color_1

    r_1, g_1, b_1, a_1 = color_1
    h_1, s_1, v_1 = rgb_to_hsv(r_1, g_1, b_1)
    r_2, g_2, b_2, a_2 = color_2
    h_2, s_2, v_2 = rgb_to_hsv(r_2, g_2, b_2)
    hue_diff = abs(h_1 - h_2)
    if hue_diff < 0.5:
        h = min(h_1, h_2) + hue_diff / 2.
    else:
        h = max(h_1, h_2) + (1. - hue_diff) / 2.
        h %= 1.
    s = (s_1 + s_2) / 2.
    v = (v_1 + v_2) / 2.
    r, g, b = hsv_to_rgb(h, s, v)
    a = (a_1 + a_2) / 2.
    color = (r, g, b, a)
    return color


def _split_colors(color, n):
    """Create n colors in HSV space that occupy a gradient in value

    Parameters
    ----------
    color : tuple
        RGBA tuple with values between 0 and 1.
    n : int >= 2
        Number of colors on the gradient.

    Returns
    -------
    colors : tuple of tuples, len = n
        N RGBA tuples that occupy a gradient in value (low to high) but share
        saturation and hue with the input color.
    """
    r, g, b, a = color
    h, s, v = rgb_to_hsv(r, g, b)
    gradient_range = np.sqrt(n / 10.)
    if v > 0.5:
        v_max = min(0.95, v + gradient_range / 2)
        v_min = max(0.05, v_max - gradient_range)
    else:
        v_min = max(0.05, v - gradient_range / 2)
        v_max = min(0.95, v_min + gradient_range)

    hsv_colors = ((h, s, v_) for v_ in np.linspace(v_min, v_max, n))
    rgb_colors = (hsv_to_rgb(h_, s_, v_) for h_, s_, v_ in hsv_colors)
    rgba_colors = ((r_, g_, b_, a,) for r_, g_, b_ in rgb_colors)
    return tuple(rgba_colors)


def _n_colors(n, bytes_=False, cmap='hsv'):
    """Produce a list of n unique RGBA color tuples based on a colormap

    Parameters
    ----------
    n : int
        Number of colors.
    bytes : bool
        Return colors as integers values between 0 and 255 (instead of floats
        between 0 and 1).
    cmap : str
        Which colormap to use.

    Returns
    -------
    colors : array, shape (n, 4)
        RGBA color values.
    """
    n_max = 2 ** 10
    if n > n_max:
        raise NotImplementedError("Can't produce more than %i unique "
                                  "colors" % n_max)

    from matplotlib.cm import get_cmap
    cm = get_cmap(cmap, n_max)
    pos = np.linspace(0, 1, n, False)
    colors = cm(pos, bytes=bytes_)
    if bytes_:
        # make sure colors are unique
        for ii, c in enumerate(colors):
            if np.any(np.all(colors[:ii] == c, 1)):
                raise RuntimeError('Could not get %d unique colors from %s '
                                   'colormap. Try using a different colormap.'
                                   % (n, cmap))
    return colors


class Label(object):
    """A FreeSurfer/MNE label with vertices restricted to one hemisphere

    Labels can be combined with the ``+`` operator:

         - Duplicate vertices are removed.
         - If duplicate vertices have conflicting position values, an error
           is raised.
         - Values of duplicate vertices are summed.

    Parameters
    ----------
    vertices : array (length N)
        vertex indices (0 based).
    pos : array (N by 3) | None
        locations in meters. If None, then zeros are used.
    values : array (length N) | None
        values at the vertices. If None, then ones are used.
    hemi : 'lh' | 'rh'
        Hemisphere to which the label applies.
    comment : str
        Kept as information but not used by the object itself.
    name : str
        Kept as information but not used by the object itself.
    filename : str
        Kept as information but not used by the object itself.
    subject : str | None
        Name of the subject the label is from.
    color : None | matplotlib color
        Default label color and alpha (e.g., ``(1., 0., 0., 1.)`` for red).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Attributes
    ----------
    color : None | tuple
        Default label color, represented as RGBA tuple with values between 0
        and 1.
    comment : str
        Comment from the first line of the label file.
    hemi : 'lh' | 'rh'
        Hemisphere.
    name : None | str
        A name for the label. It is OK to change that attribute manually.
    pos : array, shape = (n_pos, 3)
        Locations in meters.
    subject : str | None
        Subject name. It is best practice to set this to the proper
        value on initialization, but it can also be set manually.
    values : array, len = n_pos
        Values at the vertices.
    verbose : bool, str, int, or None
        See above.
    vertices : array, len = n_pos
        Vertex indices (0 based)
    """
    @verbose
    def __init__(self, vertices, pos=None, values=None, hemi=None, comment="",
                 name=None, filename=None, subject=None, color=None,
                 verbose=None):
        # check parameters
        if not isinstance(hemi, string_types):
            raise ValueError('hemi must be a string, not %s' % type(hemi))
        vertices = np.asarray(vertices)
        if np.any(np.diff(vertices.astype(int)) <= 0):
            raise ValueError('Vertices must be ordered in increasing order.')

        if color is not None:
            from matplotlib.colors import colorConverter
            color = colorConverter.to_rgba(color)

        if values is None:
            values = np.ones(len(vertices))
        else:
            values = np.asarray(values)

        if pos is None:
            pos = np.zeros((len(vertices), 3))
        else:
            pos = np.asarray(pos)

        if not (len(vertices) == len(values) == len(pos)):
            raise ValueError("vertices, values and pos need to have same "
                             "length (number of vertices)")

        # name
        if name is None and filename is not None:
            name = op.basename(filename[:-6])

        self.vertices = vertices
        self.pos = pos
        self.values = values
        self.hemi = hemi
        self.comment = comment
        self.verbose = verbose
        self.subject = _check_subject(None, subject, False)
        self.color = color
        self.name = name
        self.filename = filename

    def __setstate__(self, state):
        self.vertices = state['vertices']
        self.pos = state['pos']
        self.values = state['values']
        self.hemi = state['hemi']
        self.comment = state['comment']
        self.verbose = state['verbose']
        self.subject = state.get('subject', None)
        self.color = state.get('color', None)
        self.name = state['name']
        self.filename = state['filename']

    def __getstate__(self):
        out = dict(vertices=self.vertices,
                   pos=self.pos,
                   values=self.values,
                   hemi=self.hemi,
                   comment=self.comment,
                   verbose=self.verbose,
                   subject=self.subject,
                   color=self.color,
                   name=self.name,
                   filename=self.filename)
        return out

    def __repr__(self):
        name = 'unknown, ' if self.subject is None else self.subject + ', '
        name += repr(self.name) if self.name is not None else "unnamed"
        n_vert = len(self)
        return "<Label  |  %s, %s : %i vertices>" % (name, self.hemi, n_vert)

    def __len__(self):
        return len(self.vertices)

    def __add__(self, other):
        if isinstance(other, BiHemiLabel):
            return other + self
        elif isinstance(other, Label):
            if self.subject != other.subject:
                raise ValueError('Label subject parameters must match, got '
                                 '"%s" and "%s". Consider setting the '
                                 'subject parameter on initialization, or '
                                 'setting label.subject manually before '
                                 'combining labels.' % (self.subject,
                                                        other.subject))
            if self.hemi != other.hemi:
                name = '%s + %s' % (self.name, other.name)
                if self.hemi == 'lh':
                    lh, rh = self.copy(), other.copy()
                else:
                    lh, rh = other.copy(), self.copy()
                color = _blend_colors(self.color, other.color)
                return BiHemiLabel(lh, rh, name, color)
        else:
            raise TypeError("Need: Label or BiHemiLabel. Got: %r" % other)

        # check for overlap
        duplicates = np.intersect1d(self.vertices, other.vertices)
        n_dup = len(duplicates)
        if n_dup:
            self_dup = [np.where(self.vertices == d)[0][0]
                        for d in duplicates]
            other_dup = [np.where(other.vertices == d)[0][0]
                         for d in duplicates]
            if not np.all(self.pos[self_dup] == other.pos[other_dup]):
                err = ("Labels %r and %r: vertices overlap but differ in "
                       "position values" % (self.name, other.name))
                raise ValueError(err)

            isnew = np.array([v not in duplicates for v in other.vertices])

            vertices = np.hstack((self.vertices, other.vertices[isnew]))
            pos = np.vstack((self.pos, other.pos[isnew]))

            # find position of other's vertices in new array
            tgt_idx = [np.where(vertices == v)[0][0] for v in other.vertices]
            n_self = len(self.values)
            n_other = len(other.values)
            new_len = n_self + n_other - n_dup
            values = np.zeros(new_len, dtype=self.values.dtype)
            values[:n_self] += self.values
            values[tgt_idx] += other.values
        else:
            vertices = np.hstack((self.vertices, other.vertices))
            pos = np.vstack((self.pos, other.pos))
            values = np.hstack((self.values, other.values))

        indcs = np.argsort(vertices)
        vertices, pos, values = vertices[indcs], pos[indcs, :], values[indcs]

        comment = "%s + %s" % (self.comment, other.comment)

        name0 = self.name if self.name else 'unnamed'
        name1 = other.name if other.name else 'unnamed'
        name = "%s + %s" % (name0, name1)

        color = _blend_colors(self.color, other.color)
        verbose = self.verbose or other.verbose

        label = Label(vertices, pos, values, self.hemi, comment, name, None,
                      self.subject, color, verbose)
        return label

    def __sub__(self, other):
        if isinstance(other, BiHemiLabel):
            if self.hemi == 'lh':
                return self - other.lh
            else:
                return self - other.rh
        elif isinstance(other, Label):
            if self.subject != other.subject:
                raise ValueError('Label subject parameters must match, got '
                                 '"%s" and "%s". Consider setting the '
                                 'subject parameter on initialization, or '
                                 'setting label.subject manually before '
                                 'combining labels.' % (self.subject,
                                                        other.subject))
        else:
            raise TypeError("Need: Label or BiHemiLabel. Got: %r" % other)

        if self.hemi == other.hemi:
            keep = in1d(self.vertices, other.vertices, True, invert=True)
        else:
            keep = np.arange(len(self.vertices))

        name = "%s - %s" % (self.name or 'unnamed', other.name or 'unnamed')
        return Label(self.vertices[keep], self.pos[keep], self.values[keep],
                     self.hemi, self.comment, name, None, self.subject,
                     self.color, self.verbose)

    def save(self, filename):
        """Write to disk as FreeSurfer \*.label file

        Parameters
        ----------
        filename : string
            Path to label file to produce.

        Notes
        -----
        Note that due to file specification limitations, the Label's subject
        and color attributes are not saved to disk.
        """
        write_label(filename, self)

    def copy(self):
        """Copy the label instance.

        Returns
        -------
        label : instance of Label
            The copied label.
        """
        return cp.deepcopy(self)

    def fill(self, src, name=None):
        """Fill the surface between sources for a label defined in source space

        Parameters
        ----------
        src : SourceSpaces
            Source space in which the label was defined. If a source space is
            provided, the label is expanded to fill in surface vertices that
            lie between the vertices included in the source space. For the
            added vertices, ``pos`` is filled in with positions from the
            source space, and ``values`` is filled in from the closest source
            space vertex.
        name : None | str
            Name for the new Label (default is self.name).

        Returns
        -------
        label : Label
            The label covering the same vertices in source space but also
            including intermediate surface vertices.
        """
        # find source space patch info
        if self.hemi == 'lh':
            hemi_src = src[0]
        elif self.hemi == 'rh':
            hemi_src = src[1]

        if not np.all(in1d(self.vertices, hemi_src['vertno'])):
            msg = "Source space does not contain all of the label's vertices"
            raise ValueError(msg)

        nearest = hemi_src['nearest']
        if nearest is None:
            logger.warn("Computing patch info for source space, this can take "
                        "a while. In order to avoid this in the future, run "
                        "mne.add_source_space_distances() on the source space "
                        "and save it.")
            add_source_space_distances(src)
            nearest = hemi_src['nearest']

        # find new vertices
        include = in1d(nearest, self.vertices, False)
        vertices = np.nonzero(include)[0]

        # values
        nearest_in_label = digitize(nearest[vertices], self.vertices, True)
        values = self.values[nearest_in_label]
        # pos
        pos = hemi_src['rr'][vertices]

        if name is None:
            name = self.name
        label = Label(vertices, pos, values, self.hemi, self.comment, name,
                      None, self.subject, self.color)
        return label

    @verbose
    def smooth(self, subject=None, smooth=2, grade=None,
               subjects_dir=None, n_jobs=1, copy=True, verbose=None):
        """Smooth the label

        Useful for filling in labels made in a
        decimated source space for display.

        Parameters
        ----------
        subject : str | None
            The name of the subject used. If None, the value will be
            taken from self.subject.
        smooth : int
            Number of iterations for the smoothing of the surface data.
            Cannot be None here since not all vertices are used. For a
            grade of 5 (e.g., fsaverage), a smoothing of 2 will fill a
            label.
        grade : int, list (of two arrays), array, or None
            Resolution of the icosahedral mesh (typically 5). If None, all
            vertices will be used (potentially filling the surface). If a list,
            values will be morphed to the set of vertices specified in grade[0]
            and grade[1], assuming that these are vertices for the left and
            right hemispheres. Note that specifying the vertices (e.g.,
            grade=[np.arange(10242), np.arange(10242)] for fsaverage on a
            standard grade 5 source space) can be substantially faster than
            computing vertex locations. If one array is used, it is assumed
            that all vertices belong to the hemisphere of the label. To create
            a label filling the surface, use None.
        subjects_dir : string, or None
            Path to SUBJECTS_DIR if it is not set in the environment.
        n_jobs : int
            Number of jobs to run in parallel
        copy : bool
            If False, smoothing is done in-place.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.

        Returns
        -------
        label : instance of Label
            The smoothed label.

        Notes
        -----
        This function will set label.pos to be all zeros. If the positions
        on the new surface are required, consider using mne.read_surface
        with label.vertices.
        """
        subject = _check_subject(self.subject, subject)
        return self.morph(subject, subject, smooth, grade, subjects_dir,
                          n_jobs, copy)

    @verbose
    def morph(self, subject_from=None, subject_to=None, smooth=5, grade=None,
              subjects_dir=None, n_jobs=1, copy=True, verbose=None):
        """Morph the label

        Useful for transforming a label from one subject to another.

        Parameters
        ----------
        subject_from : str | None
            The name of the subject of the current label. If None, the
            initial subject will be taken from self.subject.
        subject_to : str
            The name of the subject to morph the label to. This will
            be put in label.subject of the output label file.
        smooth : int
            Number of iterations for the smoothing of the surface data.
            Cannot be None here since not all vertices are used.
        grade : int, list (of two arrays), array, or None
            Resolution of the icosahedral mesh (typically 5). If None, all
            vertices will be used (potentially filling the surface). If a list,
            values will be morphed to the set of vertices specified in grade[0]
            and grade[1], assuming that these are vertices for the left and
            right hemispheres. Note that specifying the vertices (e.g.,
            ``grade=[np.arange(10242), np.arange(10242)]`` for fsaverage on a
            standard grade 5 source space) can be substantially faster than
            computing vertex locations. If one array is used, it is assumed
            that all vertices belong to the hemisphere of the label. To create
            a label filling the surface, use None.
        subjects_dir : string, or None
            Path to SUBJECTS_DIR if it is not set in the environment.
        n_jobs : int
            Number of jobs to run in parallel.
        copy : bool
            If False, the morphing is done in-place.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).

        Returns
        -------
        label : instance of Label
            The morphed label.

        Notes
        -----
        This function will set label.pos to be all zeros. If the positions
        on the new surface are required, consider using `mne.read_surface`
        with `label.vertices`.
        """
        subject_from = _check_subject(self.subject, subject_from)
        if not isinstance(subject_to, string_types):
            raise TypeError('"subject_to" must be entered as a string')
        if not isinstance(smooth, int):
            raise ValueError('smooth must be an integer')
        if np.all(self.values == 0):
            raise ValueError('Morphing label with all zero values will result '
                             'in the label having no vertices. Consider using '
                             'something like label.values.fill(1.0).')
        if(isinstance(grade, np.ndarray)):
            if self.hemi == 'lh':
                grade = [grade, np.array([], int)]
            else:
                grade = [np.array([], int), grade]
        if self.hemi == 'lh':
            vertices = [self.vertices, np.array([], int)]
        else:
            vertices = [np.array([], int), self.vertices]
        data = self.values[:, np.newaxis]
        stc = SourceEstimate(data, vertices, tmin=1, tstep=1,
                             subject=subject_from)
        stc = morph_data(subject_from, subject_to, stc, grade=grade,
                         smooth=smooth, subjects_dir=subjects_dir,
                         warn=False, n_jobs=n_jobs)
        inds = np.nonzero(stc.data)[0]
        if copy is True:
            label = self.copy()
        else:
            label = self
        label.values = stc.data[inds, :].ravel()
        label.pos = np.zeros((len(inds), 3))
        if label.hemi == 'lh':
            label.vertices = stc.vertices[0][inds]
        else:
            label.vertices = stc.vertices[1][inds]
        label.subject = subject_to
        return label

    def split(self, parts=2, subject=None, subjects_dir=None,
              freesurfer=False):
        """Split the Label into two or more parts

        Parameters
        ----------
        parts : int >= 2 | tuple of str
            A sequence of strings specifying label names for the new labels
            (from posterior to anterior), or the number of new labels to create
            (default is 2). If a number is specified, names of the new labels
            will be the input label's name with div1, div2 etc. appended.
        subject : None | str
            Subject which this label belongs to (needed to locate surface file;
            should only be specified if it is not specified in the label).
        subjects_dir : None | str
            Path to SUBJECTS_DIR if it is not set in the environment.
        freesurfer : bool
            By default (``False``) ``split_label`` uses an algorithm that is
            slightly optimized for performance and numerical precision. Set
            ``freesurfer`` to ``True`` in order to replicate label splits from
            FreeSurfer's ``mris_divide_parcellation``.

        Returns
        -------
        labels : list of Label (len = n_parts)
            The labels, starting from the lowest to the highest end of the
            projection axis.

        Notes
        -----
        Works by finding the label's principal eigen-axis on the spherical
        surface, projecting all label vertex coordinates onto this axis and
        dividing them at regular spatial intervals.
        """
        return split_label(self, parts, subject, subjects_dir, freesurfer)

    def get_vertices_used(self, vertices=None):
        """Get the source space's vertices inside the label

        Parameters
        ----------
        vertices : ndarray of int, shape (n_vertices,) | None
            The set of vertices to compare the label to. If None, equals to
            ```np.arange(10242)```. Defaults to None.

        Returns
        -------
        label_verts : ndarray of in, shape (n_label_vertices,)
            The vertices of the label corresponding used by the data.
        """
        if vertices is None:
            vertices = np.arange(10242)

        label_verts = vertices[in1d(vertices, self.vertices)]
        return label_verts

    def get_tris(self, tris, vertices=None):
        """Get the source space's triangles inside the label

        Parameters
        ----------
        tris : ndarray of int, shape (n_tris, 3)
            The set of triangles corresponding to the vertices in a
            source space.
        vertices : ndarray of int, shape (n_vertices,) | None
            The set of vertices to compare the label to. If None, equals to
            ```np.arange(10242)```. Defaults to None.

        Returns
        -------
        label_tris : ndarray of int, shape (n_tris, 3)
            The subset of tris used by the label
        """
        vertices_ = self.get_vertices_used(vertices)
        selection = np.all(in1d(tris, vertices_).reshape(tris.shape),
                           axis=1)
        label_tris = tris[selection]
        if len(np.unique(label_tris)) < len(vertices_):
            logger.info('Surprising label structure. Trying to repair '
                        'triangles.')
            dropped_vertices = np.setdiff1d(vertices_, label_tris)
            n_dropped = len(dropped_vertices)
            assert n_dropped == (len(vertices_) - len(np.unique(label_tris)))

            #  put missing vertices as extra zero-length triangles
            add_tris = (dropped_vertices +
                        np.zeros((len(dropped_vertices), 3), dtype=int).T)

            label_tris = np.r_[label_tris, add_tris.T]
            assert len(np.unique(label_tris)) == len(vertices_)

        return label_tris


class BiHemiLabel(object):
    """A freesurfer/MNE label with vertices in both hemispheres

    Parameters
    ----------
    lh : Label
        Label for the left hemisphere.
    rh : Label
        Label for the right hemisphere.
    name : None | str
        name for the label
    color : None | matplotlib color
        Label color and alpha (e.g., ``(1., 0., 0., 1.)`` for red).
        Note that due to file specification limitations, the color isn't saved
        to or loaded from files written to disk.

    Attributes
    ----------
    lh : Label
        Label for the left hemisphere.
    rh : Label
        Label for the right hemisphere.
    name : None | str
        A name for the label. It is OK to change that attribute manually.
    subject : str | None
        Subject the label is from.
    """
    hemi = 'both'

    def __init__(self, lh, rh, name=None, color=None):
        if lh.subject != rh.subject:
            raise ValueError('lh.subject (%s) and rh.subject (%s) must '
                             'agree' % (lh.subject, rh.subject))
        self.lh = lh
        self.rh = rh
        self.name = name
        self.subject = lh.subject
        self.color = color

    def __repr__(self):
        temp = "<BiHemiLabel  |  %s, lh : %i vertices,  rh : %i vertices>"
        name = 'unknown, ' if self.subject is None else self.subject + ', '
        name += repr(self.name) if self.name is not None else "unnamed"
        return temp % (name, len(self.lh), len(self.rh))

    def __len__(self):
        return len(self.lh) + len(self.rh)

    def __add__(self, other):
        if isinstance(other, Label):
            if other.hemi == 'lh':
                lh = self.lh + other
                rh = self.rh
            else:
                lh = self.lh
                rh = self.rh + other
        elif isinstance(other, BiHemiLabel):
            lh = self.lh + other.lh
            rh = self.rh + other.rh
        else:
            raise TypeError("Need: Label or BiHemiLabel. Got: %r" % other)

        name = '%s + %s' % (self.name, other.name)
        color = _blend_colors(self.color, other.color)
        return BiHemiLabel(lh, rh, name, color)

    def __sub__(self, other):
        if isinstance(other, Label):
            if other.hemi == 'lh':
                lh = self.lh - other
                rh = self.rh
            else:
                rh = self.rh - other
                lh = self.lh
        elif isinstance(other, BiHemiLabel):
            lh = self.lh - other.lh
            rh = self.rh - other.rh
        else:
            raise TypeError("Need: Label or BiHemiLabel. Got: %r" % other)

        if len(lh.vertices) == 0:
            return rh
        elif len(rh.vertices) == 0:
            return lh
        else:
            name = '%s - %s' % (self.name, other.name)
            return BiHemiLabel(lh, rh, name, self.color)


def read_label(filename, subject=None, color=None):
    """Read FreeSurfer Label file

    Parameters
    ----------
    filename : string
        Path to label file.
    subject : str | None
        Name of the subject the data are defined for.
        It is good practice to set this attribute to avoid combining
        incompatible labels and SourceEstimates (e.g., ones from other
        subjects). Note that due to file specification limitations, the
        subject name isn't saved to or loaded from files written to disk.
    color : None | matplotlib color
        Default label color and alpha (e.g., ``(1., 0., 0., 1.)`` for red).
        Note that due to file specification limitations, the color isn't saved
        to or loaded from files written to disk.

    Returns
    -------
    label : Label
        Instance of Label object with attributes:

            - ``comment``: comment from the first line of the label file
            - ``vertices``: vertex indices (0 based, column 1)
            - ``pos``: locations in meters (columns 2 - 4 divided by 1000)
            - ``values``: values at the vertices (column 5)

    See Also
    --------
    read_labels_from_annot
    """
    if subject is not None and not isinstance(subject, string_types):
        raise TypeError('subject must be a string')

    # find hemi
    basename = op.basename(filename)
    if basename.endswith('lh.label') or basename.startswith('lh.'):
        hemi = 'lh'
    elif basename.endswith('rh.label') or basename.startswith('rh.'):
        hemi = 'rh'
    else:
        raise ValueError('Cannot find which hemisphere it is. File should end'
                         ' with lh.label or rh.label')

    # find name
    if basename.startswith(('lh.', 'rh.')):
        basename_ = basename[3:]
        if basename.endswith('.label'):
            basename_ = basename[:-6]
    else:
        basename_ = basename[:-9]
    name = "%s-%s" % (basename_, hemi)

    # read the file
    with open(filename, 'r') as fid:
        comment = fid.readline().replace('\n', '')[1:]
        nv = int(fid.readline())
        data = np.empty((5, nv))
        for i, line in enumerate(fid):
            data[:, i] = line.split()

    # let's make sure everything is ordered correctly
    vertices = np.array(data[0], dtype=np.int32)
    pos = 1e-3 * data[1:4].T
    values = data[4]
    order = np.argsort(vertices)
    vertices = vertices[order]
    pos = pos[order]
    values = values[order]

    label = Label(vertices, pos, values, hemi, comment, name, filename,
                  subject, color)

    return label


@verbose
def write_label(filename, label, verbose=None):
    """Write a FreeSurfer label

    Parameters
    ----------
    filename : string
        Path to label file to produce.
    label : Label
        The label object to save.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Notes
    -----
    Note that due to file specification limitations, the Label's subject and
    color attributes are not saved to disk.

    See Also
    --------
    write_labels_to_annot
    """
    hemi = label.hemi
    path_head, name = op.split(filename)
    if name.endswith('.label'):
        name = name[:-6]
    if not (name.startswith(hemi) or name.endswith(hemi)):
        name += '-' + hemi
    filename = op.join(path_head, name) + '.label'

    logger.info('Saving label to : %s' % filename)

    with open(filename, 'wb') as fid:
        n_vertices = len(label.vertices)
        data = np.zeros((n_vertices, 5), dtype=np.float)
        data[:, 0] = label.vertices
        data[:, 1:4] = 1e3 * label.pos
        data[:, 4] = label.values
        fid.write(b("#%s\n" % label.comment))
        fid.write(b("%d\n" % n_vertices))
        for d in data:
            fid.write(b("%d %f %f %f %f\n" % tuple(d)))
    return label


def split_label(label, parts=2, subject=None, subjects_dir=None,
                freesurfer=False):
    """Split a Label into two or more parts

    Parameters
    ----------
    label : Label | str
        Label which is to be split (Label object or path to a label file).
    parts : int >= 2 | tuple of str
        A sequence of strings specifying label names for the new labels (from
        posterior to anterior), or the number of new labels to create (default
        is 2). If a number is specified, names of the new labels will be the
        input label's name with div1, div2 etc. appended.
    subject : None | str
        Subject which this label belongs to (needed to locate surface file;
        should only be specified if it is not specified in the label).
    subjects_dir : None | str
        Path to SUBJECTS_DIR if it is not set in the environment.
    freesurfer : bool
        By default (``False``) ``split_label`` uses an algorithm that is
        slightly optimized for performance and numerical precision. Set
        ``freesurfer`` to ``True`` in order to replicate label splits from
        FreeSurfer's ``mris_divide_parcellation``.

    Returns
    -------
    labels : list of Label (len = n_parts)
        The labels, starting from the lowest to the highest end of the
        projection axis.

    Notes
    -----
    Works by finding the label's principal eigen-axis on the spherical surface,
    projecting all label vertex coordinates onto this axis and dividing them at
    regular spatial intervals.
    """
    # find the label
    if isinstance(label, BiHemiLabel):
        raise TypeError("Can only split labels restricted to one hemisphere.")
    elif isinstance(label, string_types):
        label = read_label(label)

    # find the parts
    if np.isscalar(parts):
        n_parts = int(parts)
        if label.name.endswith(('lh', 'rh')):
            basename = label.name[:-3]
            name_ext = label.name[-3:]
        else:
            basename = label.name
            name_ext = ''
        name_pattern = "%s_div%%i%s" % (basename, name_ext)
        names = tuple(name_pattern % i for i in range(1, n_parts + 1))
    else:
        names = parts
        n_parts = len(names)

    if n_parts < 2:
        raise ValueError("Can't split label into %i parts" % n_parts)

    # find the subject
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    if label.subject is None and subject is None:
        raise ValueError("The subject needs to be specified.")
    elif subject is None:
        subject = label.subject
    elif label.subject is None:
        pass
    elif subject != label.subject:
        raise ValueError("The label specifies a different subject (%r) from "
                         "the subject parameter (%r)."
                         % label.subject, subject)

    # find the spherical surface
    surf_fname = '.'.join((label.hemi, 'sphere'))
    surf_path = os.path.join(subjects_dir, subject, "surf", surf_fname)
    surface_points, surface_tris = read_surface(surf_path)
    # find the label coordinates on the surface
    points = surface_points[label.vertices]
    center = np.mean(points, axis=0)
    centered_points = points - center

    # find the label's normal
    if freesurfer:
        # find the Freesurfer vertex closest to the center
        distance = np.sqrt(np.sum(centered_points ** 2, axis=1))
        i_closest = np.argmin(distance)
        closest_vertex = label.vertices[i_closest]
        # find the normal according to freesurfer convention
        idx = np.any(surface_tris == closest_vertex, axis=1)
        tris_for_normal = surface_tris[idx]
        r1 = surface_points[tris_for_normal[:, 0], :]
        r2 = surface_points[tris_for_normal[:, 1], :]
        r3 = surface_points[tris_for_normal[:, 2], :]
        tri_normals = fast_cross_3d((r2 - r1), (r3 - r1))
        normal = np.mean(tri_normals, axis=0)
        normal /= linalg.norm(normal)
    else:
        # Normal of the center
        normal = center / linalg.norm(center)

    # project all vertex coordinates on the tangential plane for this point
    q, _ = linalg.qr(normal[:, np.newaxis])
    tangent_u = q[:, 1:]
    m_obs = np.dot(centered_points, tangent_u)
    # find principal eigendirection
    m_cov = np.dot(m_obs.T, m_obs)
    w, vr = linalg.eig(m_cov)
    i = np.argmax(w)
    eigendir = vr[:, i]
    # project back into 3d space
    axis = np.dot(tangent_u, eigendir)
    # orient them from posterior to anterior
    if axis[1] < 0:
        axis *= -1

    # project the label on the axis
    proj = np.dot(points, axis)

    # assign mark (new label index)
    proj -= proj.min()
    proj /= (proj.max() / n_parts)
    mark = proj // 1
    mark[mark == n_parts] = n_parts - 1

    # colors
    if label.color is None:
        colors = (None,) * n_parts
    else:
        colors = _split_colors(label.color, n_parts)

    # construct new labels
    labels = []
    for i, name, color in zip(range(n_parts), names, colors):
        idx = (mark == i)
        vert = label.vertices[idx]
        pos = label.pos[idx]
        values = label.values[idx]
        hemi = label.hemi
        comment = label.comment
        lbl = Label(vert, pos, values, hemi, comment, name, None, subject,
                    color)
        labels.append(lbl)

    return labels


@deprecated("'label_time_courses' will be removed in version 0.10, please use "
            "'in_label' method of SourceEstimate instead")
def label_time_courses(labelfile, stcfile):
    """Extract the time courses corresponding to a label file from an stc file

    Parameters
    ----------
    labelfile : string
        Path to the label file.
    stcfile : string
        Path to the stc file. The name of the stc file (must be on the
        same subject and hemisphere as the stc file).

    Returns
    -------
    values : 2d array
        The time courses.
    times : 1d array
        The time points.
    vertices : array
        The indices of the vertices corresponding to the time points.
    """
    stc = _read_stc(stcfile)
    lab = read_label(labelfile)

    vertices = np.intersect1d(stc['vertices'], lab.vertices)
    idx = [k for k in range(len(stc['vertices']))
           if stc['vertices'][k] in vertices]

    if len(vertices) == 0:
        raise ValueError('No vertices match the label in the stc file')

    values = stc['data'][idx]
    times = stc['tmin'] + stc['tstep'] * np.arange(stc['data'].shape[1])

    return values, times, vertices


def label_sign_flip(label, src):
    """Compute sign for label averaging

    Parameters
    ----------
    label : Label
        A label.
    src : list of dict
        The source space over which the label is defined.

    Returns
    -------
    flip : array
        Sign flip vector (contains 1 or -1)
    """
    if len(src) != 2:
        raise ValueError('Only source spaces with 2 hemisphers are accepted')

    lh_vertno = src[0]['vertno']
    rh_vertno = src[1]['vertno']

    # get source orientations
    if label.hemi == 'lh':
        vertno_sel = np.intersect1d(lh_vertno, label.vertices)
        if len(vertno_sel) == 0:
            return np.array([], int)
        ori = src[0]['nn'][vertno_sel]
    elif label.hemi == 'rh':
        vertno_sel = np.intersect1d(rh_vertno, label.vertices)
        if len(vertno_sel) == 0:
            return np.array([], int)
        ori = src[1]['nn'][vertno_sel]
    else:
        raise Exception("Unknown hemisphere type")

    _, _, Vh = linalg.svd(ori, full_matrices=False)

    # Comparing to the direction of the first right singular vector
    flip = np.sign(np.dot(ori, Vh[:, 0] if len(vertno_sel) > 3 else Vh[0]))
    return flip


def stc_to_label(stc, src=None, smooth=True, connected=False,
                 subjects_dir=None):
    """Compute a label from the non-zero sources in an stc object.

    Parameters
    ----------
    stc : SourceEstimate
        The source estimates.
    src : SourceSpaces | str | None
        The source space over which the source estimates are defined.
        If it's a string it should the subject name (e.g. fsaverage).
        Can be None if stc.subject is not None.
    smooth : bool
        Fill in vertices on the cortical surface that are not in the source
        space based on the closest source space vertex (requires
        src to be a SourceSpace).
    connected : bool
        If True a list of connected labels will be returned in each
        hemisphere. The labels are ordered in decreasing order depending
        of the maximum value in the stc.
    subjects_dir : str | None
        Path to SUBJECTS_DIR if it is not set in the environment.

    Returns
    -------
    labels : list of Labels | list of list of Labels
        The generated labels. If connected is False, it returns
        a list of Labels (one per hemisphere). If no Label is available
        in a hemisphere, None is returned. If connected is True,
        it returns for each hemisphere a list of connected labels
        ordered in decreasing order depending of the maximum value in the stc.
        If no Label is available in an hemisphere, an empty list is returned.
    """
    if not isinstance(smooth, bool):
        raise ValueError('smooth should be True or False. Got %s.' % smooth)

    src = stc.subject if src is None else src
    if src is None:
        raise ValueError('src cannot be None if stc.subject is None')
    if isinstance(src, string_types):
        subject = src
    else:
        subject = stc.subject

    if not isinstance(stc, SourceEstimate):
        raise ValueError('SourceEstimate should be surface source estimates')

    if isinstance(src, string_types):
        if connected:
            raise ValueError('The option to return only connected labels is '
                             'only available if source spaces are provided.')
        if smooth:
            msg = ("stc_to_label with smooth=True requires src to be an "
                   "instance of SourceSpace")
            raise ValueError(msg)
        subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
        surf_path_from = op.join(subjects_dir, src, 'surf')
        rr_lh, tris_lh = read_surface(op.join(surf_path_from, 'lh.white'))
        rr_rh, tris_rh = read_surface(op.join(surf_path_from, 'rh.white'))
        rr = [rr_lh, rr_rh]
        tris = [tris_lh, tris_rh]
    else:
        if not isinstance(src, SourceSpaces):
            raise TypeError('src must be a string or a set of source spaces')
        if len(src) != 2:
            raise ValueError('source space should contain the 2 hemispheres')
        rr = [1e3 * src[0]['rr'], 1e3 * src[1]['rr']]
        tris = [src[0]['tris'], src[1]['tris']]
        src_conn = spatial_src_connectivity(src).tocsr()

    labels = []
    cnt = 0
    cnt_full = 0
    for hemi_idx, (hemi, this_vertno, this_tris, this_rr) in enumerate(
            zip(['lh', 'rh'], stc.vertices, tris, rr)):
        this_data = stc.data[cnt:cnt + len(this_vertno)]
        e = mesh_edges(this_tris)
        e.data[e.data == 2] = 1
        n_vertices = e.shape[0]
        e = e + sparse.eye(n_vertices, n_vertices)

        if connected:  # we know src *must* be a SourceSpaces now
            vertno = np.where(src[hemi_idx]['inuse'])[0]
            if not len(np.setdiff1d(this_vertno, vertno)) == 0:
                raise RuntimeError('stc contains vertices not present '
                                   'in source space, did you morph?')
            tmp = np.zeros((len(vertno), this_data.shape[1]))
            this_vertno_idx = np.searchsorted(vertno, this_vertno)
            tmp[this_vertno_idx] = this_data
            this_data = tmp
            offset = cnt_full + len(this_data)
            this_src_conn = src_conn[cnt_full:offset, cnt_full:offset].tocoo()
            this_data_abs_max = np.abs(this_data).max(axis=1)
            clusters, _ = _find_clusters(this_data_abs_max, 0.,
                                         connectivity=this_src_conn)
            cnt_full += len(this_data)
            # Then order clusters in descending order based on maximum value
            clusters_max = np.argsort([np.max(this_data_abs_max[c])
                                       for c in clusters])[::-1]
            clusters = [clusters[k] for k in clusters_max]
            clusters = [vertno[c] for c in clusters]
        else:
            clusters = [this_vertno[np.any(this_data, axis=1)]]

        cnt += len(this_vertno)

        clusters = [c for c in clusters if len(c) > 0]

        if len(clusters) == 0:
            if not connected:
                this_labels = None
            else:
                this_labels = []
        else:
            this_labels = []
            colors = _n_colors(len(clusters))
            for c, color in zip(clusters, colors):
                idx_use = c
                label = Label(idx_use, this_rr[idx_use], None, hemi,
                              'Label from stc', subject=subject,
                              color=color)
                if smooth:
                    label = label.fill(src)

                this_labels.append(label)

            if not connected:
                this_labels = this_labels[0]

        labels.append(this_labels)

    return labels


def _verts_within_dist(graph, sources, max_dist):
    """Find all vertices wihin a maximum geodesic distance from source

    Parameters
    ----------
    graph : scipy.sparse.csr_matrix
        Sparse matrix with distances between adjacent vertices.
    sources : list of int
        Source vertices.
    max_dist : float
        Maximum geodesic distance.

    Returns
    -------
    verts : array
        Vertices within max_dist.
    dist : array
        Distances from source vertex.
    """
    dist_map = {}
    verts_added_last = []
    for source in sources:
        dist_map[source] = 0
        verts_added_last.append(source)

    # add neighbors until no more neighbors within max_dist can be found
    while len(verts_added_last) > 0:
        verts_added = []
        for i in verts_added_last:
            v_dist = dist_map[i]
            row = graph[i, :]
            neighbor_vert = row.indices
            neighbor_dist = row.data
            for j, d in zip(neighbor_vert, neighbor_dist):
                n_dist = v_dist + d
                if j in dist_map:
                    if n_dist < dist_map[j]:
                        dist_map[j] = n_dist
                else:
                    if n_dist <= max_dist:
                        dist_map[j] = n_dist
                        # we found a new vertex within max_dist
                        verts_added.append(j)
        verts_added_last = verts_added

    verts = np.sort(np.array(list(dist_map.keys()), dtype=np.int))
    dist = np.array([dist_map[v] for v in verts])

    return verts, dist


def _grow_labels(seeds, extents, hemis, names, dist, vert, subject):
    """Helper for parallelization of grow_labels
    """
    labels = []
    for seed, extent, hemi, name in zip(seeds, extents, hemis, names):
        label_verts, label_dist = _verts_within_dist(dist[hemi], seed, extent)

        # create a label
        if len(seed) == 1:
            seed_repr = str(seed)
        else:
            seed_repr = ','.join(map(str, seed))
        comment = 'Circular label: seed=%s, extent=%0.1fmm' % (seed_repr,
                                                               extent)
        label = Label(vertices=label_verts,
                      pos=vert[hemi][label_verts],
                      values=label_dist,
                      hemi=hemi,
                      comment=comment,
                      name=str(name),
                      subject=subject)
        labels.append(label)
    return labels


def grow_labels(subject, seeds, extents, hemis, subjects_dir=None, n_jobs=1,
                overlap=True, names=None, surface='white'):
    """Generate circular labels in source space with region growing

    This function generates a number of labels in source space by growing
    regions starting from the vertices defined in "seeds". For each seed, a
    label is generated containing all vertices within a maximum geodesic
    distance on the white matter surface from the seed.

    Note: "extents" and "hemis" can either be arrays with the same length as
          seeds, which allows using a different extent and hemisphere for each
          label, or integers, in which case the same extent and hemisphere is
          used for each label.

    Parameters
    ----------
    subject : string
        Name of the subject as in SUBJECTS_DIR.
    seeds : int | list
        Seed, or list of seeds. Each seed can be either a vertex number or
        a list of vertex numbers.
    extents : array | float
        Extents (radius in mm) of the labels.
    hemis : array | int
        Hemispheres to use for the labels (0: left, 1: right).
    subjects_dir : string
        Path to SUBJECTS_DIR if not set in the environment.
    n_jobs : int
        Number of jobs to run in parallel. Likely only useful if tens
        or hundreds of labels are being expanded simultaneously. Does not
        apply with ``overlap=False``.
    overlap : bool
        Produce overlapping labels. If True (default), the resulting labels
        can be overlapping. If False, each label will be grown one step at a
        time, and occupied territory will not be invaded.
    names : None | list of str
        Assign names to the new labels (list needs to have the same length as
        seeds).
    surface : string
        The surface used to grow the labels, defaults to the white surface.

    Returns
    -------
    labels : list of Label
        The labels' ``comment`` attribute contains information on the seed
        vertex and extent; the ``values``  attribute contains distance from the
        seed in millimeters
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    n_jobs = check_n_jobs(n_jobs)

    # make sure the inputs are arrays
    if np.isscalar(seeds):
        seeds = [seeds]
    seeds = np.atleast_1d([np.atleast_1d(seed) for seed in seeds])
    extents = np.atleast_1d(extents)
    hemis = np.atleast_1d(hemis)
    n_seeds = len(seeds)

    if len(extents) != 1 and len(extents) != n_seeds:
        raise ValueError('The extents parameter has to be of length 1 or '
                         'len(seeds)')

    if len(hemis) != 1 and len(hemis) != n_seeds:
        raise ValueError('The hemis parameter has to be of length 1 or '
                         'len(seeds)')

    # make the arrays the same length as seeds
    if len(extents) == 1:
        extents = np.tile(extents, n_seeds)

    if len(hemis) == 1:
        hemis = np.tile(hemis, n_seeds)

    hemis = np.array(['lh' if h == 0 else 'rh' for h in hemis])

    # names
    if names is None:
        names = ["Label_%i-%s" % items for items in enumerate(hemis)]
    else:
        if np.isscalar(names):
            names = [names]
        if len(names) != n_seeds:
            raise ValueError('The names parameter has to be None or have '
                             'length len(seeds)')
        for i, hemi in enumerate(hemis):
            if not names[i].endswith(hemi):
                names[i] = '-'.join((names[i], hemi))
    names = np.array(names)

    # load the surfaces and create the distance graphs
    tris, vert, dist = {}, {}, {}
    for hemi in set(hemis):
        surf_fname = op.join(subjects_dir, subject, 'surf', hemi + '.' +
                             surface)
        vert[hemi], tris[hemi] = read_surface(surf_fname)
        dist[hemi] = mesh_dist(tris[hemi], vert[hemi])

    if overlap:
        # create the patches
        parallel, my_grow_labels, _ = parallel_func(_grow_labels, n_jobs)
        seeds = np.array_split(seeds, n_jobs)
        extents = np.array_split(extents, n_jobs)
        hemis = np.array_split(hemis, n_jobs)
        names = np.array_split(names, n_jobs)
        labels = sum(parallel(my_grow_labels(s, e, h, n, dist, vert, subject)
                              for s, e, h, n
                              in zip(seeds, extents, hemis, names)), [])
    else:
        # special procedure for non-overlapping labels
        labels = _grow_nonoverlapping_labels(subject, seeds, extents, hemis,
                                             vert, dist, names)

    # add a unique color to each label
    colors = _n_colors(len(labels))
    for label, color in zip(labels, colors):
        label.color = color

    return labels


def _grow_nonoverlapping_labels(subject, seeds_, extents_, hemis, vertices_,
                                graphs, names_):
    """Grow labels while ensuring that they don't overlap
    """
    labels = []
    for hemi in set(hemis):
        hemi_index = (hemis == hemi)
        seeds = seeds_[hemi_index]
        extents = extents_[hemi_index]
        names = names_[hemi_index]
        graph = graphs[hemi]  # distance graph
        n_vertices = len(vertices_[hemi])
        n_labels = len(seeds)

        # prepare parcellation
        parc = np.empty(n_vertices, dtype='int32')
        parc[:] = -1

        # initialize active sources
        sources = {}  # vert -> (label, dist_from_seed)
        edge = []  # queue of vertices to process
        for label, seed in enumerate(seeds):
            if np.any(parc[seed] >= 0):
                raise ValueError("Overlapping seeds")
            parc[seed] = label
            for s in np.atleast_1d(seed):
                sources[s] = (label, 0.)
                edge.append(s)

        # grow from sources
        while edge:
            vert_from = edge.pop(0)
            label, old_dist = sources[vert_from]

            # add neighbors within allowable distance
            row = graph[vert_from, :]
            for vert_to, dist in zip(row.indices, row.data):
                new_dist = old_dist + dist

                # abort if outside of extent
                if new_dist > extents[label]:
                    continue

                vert_to_label = parc[vert_to]
                if vert_to_label >= 0:
                    _, vert_to_dist = sources[vert_to]
                    # abort if the vertex is occupied by a closer seed
                    if new_dist > vert_to_dist:
                        continue
                    elif vert_to in edge:
                        edge.remove(vert_to)

                # assign label value
                parc[vert_to] = label
                sources[vert_to] = (label, new_dist)
                edge.append(vert_to)

        # convert parc to labels
        for i in xrange(n_labels):
            vertices = np.nonzero(parc == i)[0]
            name = str(names[i])
            label_ = Label(vertices, hemi=hemi, name=name, subject=subject)
            labels.append(label_)

    return labels


def _read_annot(fname):
    """Read a Freesurfer annotation from a .annot file.

    Note : Copied from PySurfer

    Parameters
    ----------
    fname : str
        Path to annotation file

    Returns
    -------
    annot : numpy array, shape=(n_verts)
        Annotation id at each vertex
    ctab : numpy array, shape=(n_entries, 5)
        RGBA + label id colortable array
    names : list of str
        List of region names as stored in the annot file

    """
    if not op.isfile(fname):
        dir_name = op.split(fname)[0]
        if not op.isdir(dir_name):
            raise IOError('Directory for annotation does not exist: %s',
                          fname)
        cands = os.listdir(dir_name)
        cands = [c for c in cands if '.annot' in c]
        if len(cands) == 0:
            raise IOError('No such file %s, no candidate parcellations '
                          'found in directory' % fname)
        else:
            raise IOError('No such file %s, candidate parcellations in '
                          'that directory: %s' % (fname, ', '.join(cands)))
    with open(fname, "rb") as fid:
        n_verts = np.fromfile(fid, '>i4', 1)[0]
        data = np.fromfile(fid, '>i4', n_verts * 2).reshape(n_verts, 2)
        annot = data[data[:, 0], 1]
        ctab_exists = np.fromfile(fid, '>i4', 1)[0]
        if not ctab_exists:
            raise Exception('Color table not found in annotation file')
        n_entries = np.fromfile(fid, '>i4', 1)[0]
        if n_entries > 0:
            length = np.fromfile(fid, '>i4', 1)[0]
            orig_tab = np.fromfile(fid, '>c', length)
            orig_tab = orig_tab[:-1]

            names = list()
            ctab = np.zeros((n_entries, 5), np.int)
            for i in range(n_entries):
                name_length = np.fromfile(fid, '>i4', 1)[0]
                name = np.fromfile(fid, "|S%d" % name_length, 1)[0]
                names.append(name)
                ctab[i, :4] = np.fromfile(fid, '>i4', 4)
                ctab[i, 4] = (ctab[i, 0] + ctab[i, 1] * (2 ** 8) +
                              ctab[i, 2] * (2 ** 16) +
                              ctab[i, 3] * (2 ** 24))
        else:
            ctab_version = -n_entries
            if ctab_version != 2:
                raise Exception('Color table version not supported')
            n_entries = np.fromfile(fid, '>i4', 1)[0]
            ctab = np.zeros((n_entries, 5), np.int)
            length = np.fromfile(fid, '>i4', 1)[0]
            np.fromfile(fid, "|S%d" % length, 1)[0]  # Orig table path
            entries_to_read = np.fromfile(fid, '>i4', 1)[0]
            names = list()
            for i in range(entries_to_read):
                np.fromfile(fid, '>i4', 1)[0]  # Structure
                name_length = np.fromfile(fid, '>i4', 1)[0]
                name = np.fromfile(fid, "|S%d" % name_length, 1)[0]
                names.append(name)
                ctab[i, :4] = np.fromfile(fid, '>i4', 4)
                ctab[i, 4] = (ctab[i, 0] + ctab[i, 1] * (2 ** 8) +
                              ctab[i, 2] * (2 ** 16))

        # convert to more common alpha value
        ctab[:, 3] = 255 - ctab[:, 3]

    return annot, ctab, names


def _get_annot_fname(annot_fname, subject, hemi, parc, subjects_dir):
    """Helper function to get the .annot filenames and hemispheres"""
    if annot_fname is not None:
        # we use use the .annot file specified by the user
        hemis = [op.basename(annot_fname)[:2]]
        if hemis[0] not in ['lh', 'rh']:
            raise ValueError('Could not determine hemisphere from filename, '
                             'filename has to start with "lh" or "rh".')
        annot_fname = [annot_fname]
    else:
        # construct .annot file names for requested subject, parc, hemi
        if hemi not in ['lh', 'rh', 'both']:
            raise ValueError('hemi has to be "lh", "rh", or "both"')
        if hemi == 'both':
            hemis = ['lh', 'rh']
        else:
            hemis = [hemi]

        annot_fname = list()
        for hemi in hemis:
            subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
            fname = op.join(subjects_dir, subject, 'label',
                            '%s.%s.annot' % (hemi, parc))
            annot_fname.append(fname)

    return annot_fname, hemis


@verbose
def read_labels_from_annot(subject, parc='aparc', hemi='both',
                           surf_name='white', annot_fname=None, regexp=None,
                           subjects_dir=None, verbose=None):
    """Read labels from a FreeSurfer annotation file

    Note: Only cortical labels will be returned.

    Parameters
    ----------
    subject : str
        The subject for which to read the parcellation for.
    parc : str
        The parcellation to use, e.g., 'aparc' or 'aparc.a2009s'.
    hemi : str
        The hemisphere to read the parcellation for, can be 'lh', 'rh',
        or 'both'.
    surf_name : str
        Surface used to obtain vertex locations, e.g., 'white', 'pial'
    annot_fname : str or None
        Filename of the .annot file. If not None, only this file is read
        and 'parc' and 'hemi' are ignored.
    regexp : str
        Regular expression or substring to select particular labels from the
        parcellation. E.g. 'superior' will return all labels in which this
        substring is contained.
    subjects_dir : string, or None
        Path to SUBJECTS_DIR if it is not set in the environment.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    labels : list of Label
        The labels, sorted by label name (ascending).
    """
    logger.info('Reading labels from parcellation..')

    subjects_dir = get_subjects_dir(subjects_dir)

    # get the .annot filenames and hemispheres
    annot_fname, hemis = _get_annot_fname(annot_fname, subject, hemi, parc,
                                          subjects_dir)

    if regexp is not None:
        # allow for convenient substring match
        r_ = (re.compile('.*%s.*' % regexp if regexp.replace('_', '').isalnum()
              else regexp))

    # now we are ready to create the labels
    n_read = 0
    labels = list()
    for fname, hemi in zip(annot_fname, hemis):
        # read annotation
        annot, ctab, label_names = _read_annot(fname)
        label_rgbas = ctab[:, :4]
        label_ids = ctab[:, -1]

        # load the vertex positions from surface
        fname_surf = op.join(subjects_dir, subject, 'surf',
                             '%s.%s' % (hemi, surf_name))
        vert_pos, _ = read_surface(fname_surf)
        vert_pos /= 1e3  # the positions in labels are in meters
        for label_id, label_name, label_rgba in\
                zip(label_ids, label_names, label_rgbas):
            vertices = np.where(annot == label_id)[0]
            if len(vertices) == 0:
                # label is not part of cortical surface
                continue
            name = label_name.decode() + '-' + hemi
            if (regexp is not None) and not r_.match(name):
                continue
            pos = vert_pos[vertices, :]
            values = np.zeros(len(vertices))
            label_rgba = tuple(label_rgba / 255.)
            label = Label(vertices, pos, values, hemi, name=name,
                          subject=subject, color=label_rgba)
            labels.append(label)

        n_read = len(labels) - n_read
        logger.info('   read %d labels from %s' % (n_read, fname))

    # sort the labels by label name
    labels = sorted(labels, key=lambda l: l.name)

    if len(labels) == 0:
        msg = 'No labels found.'
        if regexp is not None:
            msg += ' Maybe the regular expression %r did not match?' % regexp
        raise RuntimeError(msg)

    logger.info('[done]')
    return labels


def _write_annot(fname, annot, ctab, names):
    """Write a Freesurfer annotation to a .annot file.

    Parameters
    ----------
    fname : str
        Path to annotation file
    annot : numpy array, shape=(n_verts)
        Annotation id at each vertex. Note: IDs must be computed from
        RGBA colors, otherwise the mapping will be invalid.
    ctab : numpy array, shape=(n_entries, 4)
        RGBA colortable array.
    names : list of str
        List of region names to be stored in the annot file
    """

    with open(fname, 'wb') as fid:
        n_verts = len(annot)
        np.array(n_verts, dtype='>i4').tofile(fid)

        data = np.zeros((n_verts, 2), dtype='>i4')
        data[:, 0] = np.arange(n_verts)
        data[:, 1] = annot
        data.ravel().tofile(fid)

        # indicate that color table exists
        np.array(1, dtype='>i4').tofile(fid)

        # color table version 2
        np.array(-2, dtype='>i4').tofile(fid)

        # write color table
        n_entries = len(ctab)
        np.array(n_entries, dtype='>i4').tofile(fid)

        # write dummy color table name
        table_name = 'MNE-Python Colortable'
        np.array(len(table_name), dtype='>i4').tofile(fid)
        np.fromstring(table_name, dtype=np.uint8).tofile(fid)

        # number of entries to write
        np.array(n_entries, dtype='>i4').tofile(fid)

        # write entries
        for ii, (name, color) in enumerate(zip(names, ctab)):
            np.array(ii, dtype='>i4').tofile(fid)
            np.array(len(name), dtype='>i4').tofile(fid)
            np.fromstring(name, dtype=np.uint8).tofile(fid)
            np.array(color[:4], dtype='>i4').tofile(fid)


@verbose
def write_labels_to_annot(labels, subject=None, parc=None, overwrite=False,
                          subjects_dir=None, annot_fname=None,
                          colormap='hsv', hemi='both', verbose=None):
    """Create a FreeSurfer annotation from a list of labels

    Parameters
    ----------
    labels : list with instances of mne.Label
        The labels to create a parcellation from.
    subject : str | None
        The subject for which to write the parcellation for.
    parc : str | None
        The parcellation name to use.
    overwrite : bool
        Overwrite files if they already exist.
    subjects_dir : string, or None
        Path to SUBJECTS_DIR if it is not set in the environment.
    annot_fname : str | None
        Filename of the .annot file. If not None, only this file is written
        and 'parc' and 'subject' are ignored.
    colormap : str
        Colormap to use to generate label colors for labels that do not
        have a color specified.
    hemi : 'both' | 'lh' | 'rh'
        The hemisphere(s) for which to write \*.annot files (only applies if
        annot_fname is not specified; default is 'both').
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Notes
    -----
    Vertices that are not covered by any of the labels are assigned to a label
    named "unknown".
    """
    logger.info('Writing labels to parcellation..')

    subjects_dir = get_subjects_dir(subjects_dir)

    # get the .annot filenames and hemispheres
    annot_fname, hemis = _get_annot_fname(annot_fname, subject, hemi, parc,
                                          subjects_dir)

    if not overwrite:
        for fname in annot_fname:
            if op.exists(fname):
                raise ValueError('File %s exists. Use "overwrite=True" to '
                                 'overwrite it' % fname)

    # prepare container for data to save:
    to_save = []
    # keep track of issues found in the labels
    duplicate_colors = []
    invalid_colors = []
    overlap = []
    no_color = (-1, -1, -1, -1)
    no_color_rgb = (-1, -1, -1)
    for hemi, fname in zip(hemis, annot_fname):
        hemi_labels = [label for label in labels if label.hemi == hemi]
        n_hemi_labels = len(hemi_labels)

        if n_hemi_labels == 0:
            ctab = np.empty((0, 4), dtype=np.int32)
        else:
            hemi_labels.sort(key=lambda label: label.name)

            # convert colors to 0-255 RGBA tuples
            hemi_colors = [no_color if label.color is None else
                           tuple(int(round(255 * i)) for i in label.color)
                           for label in hemi_labels]
            ctab = np.array(hemi_colors, dtype=np.int32)
            ctab_rgb = ctab[:, :3]

            # make color dict (for annot ID, only R, G and B count)
            labels_by_color = defaultdict(list)
            for label, color in zip(hemi_labels, ctab_rgb):
                labels_by_color[tuple(color)].append(label.name)

            # check label colors
            for color, names in labels_by_color.items():
                if color == no_color_rgb:
                    continue

                if color == (0, 0, 0):
                    # we cannot have an all-zero color, otherw. e.g. tksurfer
                    # refuses to read the parcellation
                    msg = ('At least one label contains a color with, "r=0, '
                           'g=0, b=0" value. Some FreeSurfer tools may fail '
                           'to read the parcellation')
                    logger.warning(msg)

                if any(i > 255 for i in color):
                    msg = ("%s: %s (%s)" % (color, ', '.join(names), hemi))
                    invalid_colors.append(msg)

                if len(names) > 1:
                    msg = "%s: %s (%s)" % (color, ', '.join(names), hemi)
                    duplicate_colors.append(msg)

            # replace None values (labels with unspecified color)
            if labels_by_color[no_color_rgb]:
                default_colors = _n_colors(n_hemi_labels, bytes_=True,
                                           cmap=colormap)
                # keep track of colors known to be in hemi_colors :
                safe_color_i = 0
                for i in xrange(n_hemi_labels):
                    if ctab[i, 0] == -1:
                        color = default_colors[i]
                        # make sure to add no duplicate color
                        while np.any(np.all(color[:3] == ctab_rgb, 1)):
                            color = default_colors[safe_color_i]
                            safe_color_i += 1
                        # assign the color
                        ctab[i] = color

        # find number of vertices in surface
        if subject is not None and subjects_dir is not None:
            fpath = os.path.join(subjects_dir, subject, 'surf',
                                 '%s.white' % hemi)
            points, _ = read_surface(fpath)
            n_vertices = len(points)
        else:
            if len(hemi_labels) > 0:
                max_vert = max(np.max(label.vertices) for label in hemi_labels)
                n_vertices = max_vert + 1
            else:
                n_vertices = 1
            msg = ('    Number of vertices in the surface could not be '
                   'verified because the surface file could not be found; '
                   'specify subject and subjects_dir parameters.')
            logger.warning(msg)

        # Create annot and color table array to write
        annot = np.empty(n_vertices, dtype=np.int)
        annot[:] = -1
        # create the annotation ids from the colors
        annot_id_coding = np.array((1, 2 ** 8, 2 ** 16))
        annot_ids = list(np.sum(ctab_rgb * annot_id_coding, axis=1))
        for label, annot_id in zip(hemi_labels, annot_ids):
            # make sure the label is not overwriting another label
            if np.any(annot[label.vertices] != -1):
                other_ids = set(annot[label.vertices])
                other_ids.discard(-1)
                other_indices = (annot_ids.index(i) for i in other_ids)
                other_names = (hemi_labels[i].name for i in other_indices)
                other_repr = ', '.join(other_names)
                msg = "%s: %s overlaps %s" % (hemi, label.name, other_repr)
                overlap.append(msg)

            annot[label.vertices] = annot_id

        hemi_names = [label.name for label in hemi_labels]

        # Assign unlabeled vertices to an "unknown" label
        unlabeled = (annot == -1)
        if np.any(unlabeled):
            msg = ("Assigning %i unlabeled vertices to "
                   "'unknown-%s'" % (unlabeled.sum(), hemi))
            logger.info(msg)

            # find an unused color (try shades of gray first)
            for i in range(1, 257):
                if not np.any(np.all((i, i, i) == ctab_rgb, 1)):
                    break
            if i < 256:
                color = (i, i, i, 0)
            else:
                err = ("Need one free shade of gray for 'unknown' label. "
                       "Please modify your label colors, or assign the "
                       "unlabeled vertices to another label.")
                raise ValueError(err)

            # find the id
            annot_id = np.sum(annot_id_coding * color[:3])

            # update data to write
            annot[unlabeled] = annot_id
            ctab = np.vstack((ctab, color))
            hemi_names.append("unknown")

        # convert to FreeSurfer alpha values
        ctab[:, 3] = 255 - ctab[:, 3]

        # remove hemi ending in names
        hemi_names = [name[:-3] if name.endswith(hemi) else name
                      for name in hemi_names]

        to_save.append((fname, annot, ctab, hemi_names))

    issues = []
    if duplicate_colors:
        msg = ("Some labels have the same color values (all labels in one "
               "hemisphere must have a unique color):")
        duplicate_colors.insert(0, msg)
        issues.append(os.linesep.join(duplicate_colors))
    if invalid_colors:
        msg = ("Some labels have invalid color values (all colors should be "
               "RGBA tuples with values between 0 and 1)")
        invalid_colors.insert(0, msg)
        issues.append(os.linesep.join(invalid_colors))
    if overlap:
        msg = ("Some labels occupy vertices that are also occupied by one or "
               "more other labels. Each vertex can only be occupied by a "
               "single label in *.annot files.")
        overlap.insert(0, msg)
        issues.append(os.linesep.join(overlap))

    if issues:
        raise ValueError('\n\n'.join(issues))

    # write it
    for fname, annot, ctab, hemi_names in to_save:
        logger.info('   writing %d labels to %s' % (len(hemi_names), fname))
        _write_annot(fname, annot, ctab, hemi_names)

    logger.info('[done]')
