# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

from collections import defaultdict
from colorsys import hsv_to_rgb, rgb_to_hsv
import os
import os.path as op
import copy as cp
import re

import numpy as np
from scipy import linalg, sparse

from .parallel import parallel_func, check_n_jobs
from .source_estimate import (SourceEstimate, _center_of_mass,
                              spatial_src_connectivity)
from .source_space import add_source_space_distances, SourceSpaces
from .stats.cluster_level import _find_clusters, _get_components
from .surface import (read_surface, fast_cross_3d, mesh_edges, mesh_dist,
                      read_morph_map)
from .utils import (get_subjects_dir, _check_subject, logger, verbose, warn,
                    check_random_state, _validate_type, fill_doc,
                    _check_option, check_version)


def _blend_colors(color_1, color_2):
    """Blend two colors in HSV space.

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
    """Create n colors in HSV space that occupy a gradient in value.

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
    """Produce a list of n unique RGBA color tuples based on a colormap.

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


@fill_doc
class Label(object):
    """A FreeSurfer/MNE label with vertices restricted to one hemisphere.

    Labels can be combined with the ``+`` operator:

        * Duplicate vertices are removed.
        * If duplicate vertices have conflicting position values, an error
          is raised.
        * Values of duplicate vertices are summed.

    Parameters
    ----------
    vertices : array, shape (N,)
        Vertex indices (0 based).
    pos : array, shape (N, 3) | None
        Locations in meters. If None, then zeros are used.
    values : array, shape (N,) | None
        Values at the vertices. If None, then ones are used.
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
    %(verbose)s

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
    pos : array, shape (N, 3)
        Locations in meters.
    subject : str | None
        Subject name. It is best practice to set this to the proper
        value on initialization, but it can also be set manually.
    values : array, shape (N,)
        Values at the vertices.
    %(verbose)s
    vertices : array, shape (N,)
        Vertex indices (0 based)
    """

    @verbose
    def __init__(self, vertices=(), pos=None, values=None, hemi=None,
                 comment="", name=None, filename=None, subject=None,
                 color=None, verbose=None):  # noqa: D102
        # check parameters
        if not isinstance(hemi, str):
            raise ValueError('hemi must be a string, not %s' % type(hemi))
        vertices = np.asarray(vertices, int)
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

    def __setstate__(self, state):  # noqa: D105
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

    def __getstate__(self):  # noqa: D105
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

    def __repr__(self):  # noqa: D105
        name = 'unknown, ' if self.subject is None else self.subject + ', '
        name += repr(self.name) if self.name is not None else "unnamed"
        n_vert = len(self)
        return "<Label  |  %s, %s : %i vertices>" % (name, self.hemi, n_vert)

    def __len__(self):
        """Return the number of vertices."""
        return len(self.vertices)

    def __add__(self, other):
        """Add Labels."""
        _validate_type(other, (Label, BiHemiLabel), 'other')
        if isinstance(other, BiHemiLabel):
            return other + self
        else:  # isinstance(other, Label)
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
        """Subtract Labels."""
        _validate_type(other, (Label, BiHemiLabel), 'other')
        if isinstance(other, BiHemiLabel):
            if self.hemi == 'lh':
                return self - other.lh
            else:
                return self - other.rh
        else:  # isinstance(other, Label):
            if self.subject != other.subject:
                raise ValueError('Label subject parameters must match, got '
                                 '"%s" and "%s". Consider setting the '
                                 'subject parameter on initialization, or '
                                 'setting label.subject manually before '
                                 'combining labels.' % (self.subject,
                                                        other.subject))

        if self.hemi == other.hemi:
            keep = np.in1d(self.vertices, other.vertices, True, invert=True)
        else:
            keep = np.arange(len(self.vertices))

        name = "%s - %s" % (self.name or 'unnamed', other.name or 'unnamed')
        return Label(self.vertices[keep], self.pos[keep], self.values[keep],
                     self.hemi, self.comment, name, None, self.subject,
                     self.color, self.verbose)

    def save(self, filename):
        r"""Write to disk as FreeSurfer \*.label file.

        Parameters
        ----------
        filename : str
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
        """Fill the surface between sources for a source space label.

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

        See Also
        --------
        Label.restrict
        Label.smooth
        """
        # find source space patch info
        if len(self.vertices) == 0:
            return self.copy()
        hemi_src = _get_label_src(self, src)

        if not np.all(np.in1d(self.vertices, hemi_src['vertno'])):
            msg = "Source space does not contain all of the label's vertices"
            raise ValueError(msg)

        if hemi_src['nearest'] is None:
            warn("Source space is being modified in place because patch "
                 "information is needed. To avoid this in the future, run "
                 "mne.add_source_space_distances() on the source space "
                 "and save it to disk.")
            if check_version('scipy', '1.3'):
                dist_limit = 0
            else:
                warn('SciPy < 1.3 detected, adding source space patch '
                     'information will be slower. Consider upgrading SciPy.')
                dist_limit = np.inf
            add_source_space_distances(src, dist_limit=dist_limit)
        nearest = hemi_src['nearest']

        # find new vertices
        include = np.in1d(nearest, self.vertices, False)
        vertices = np.nonzero(include)[0]

        # values
        nearest_in_label = np.digitize(nearest[vertices], self.vertices, True)
        values = self.values[nearest_in_label]
        # pos
        pos = hemi_src['rr'][vertices]

        name = self.name if name is None else name
        label = Label(vertices, pos, values, self.hemi, self.comment, name,
                      None, self.subject, self.color)
        return label

    def restrict(self, src, name=None):
        """Restrict a label to a source space.

        Parameters
        ----------
        src : instance of SourceSpaces
            The source spaces to use to restrict the label.
        name : None | str
            Name for the new Label (default is self.name).

        Returns
        -------
        label : instance of Label
            The Label restricted to the set of source space vertices.

        See Also
        --------
        Label.fill

        Notes
        -----
        .. versionadded:: 0.20
        """
        if len(self.vertices) == 0:
            return self.copy()
        hemi_src = _get_label_src(self, src)
        mask = np.in1d(self.vertices, hemi_src['vertno'])
        name = self.name if name is None else name
        label = Label(self.vertices[mask], self.pos[mask], self.values[mask],
                      self.hemi, self.comment, name, None, self.subject,
                      self.color)
        return label

    @verbose
    def smooth(self, subject=None, smooth=2, grade=None,
               subjects_dir=None, n_jobs=1, verbose=None):
        """Smooth the label.

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
        grade : int, list of shape (2,), array, or None
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
        %(subjects_dir)s
        %(n_jobs)s
        %(verbose_meth)s

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
                          n_jobs, verbose)

    @verbose
    def morph(self, subject_from=None, subject_to=None, smooth=5, grade=None,
              subjects_dir=None, n_jobs=1, verbose=None):
        """Morph the label.

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
        grade : int, list of shape (2,), array, or None
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
        %(subjects_dir)s
        %(n_jobs)s
        %(verbose_meth)s

        Returns
        -------
        label : instance of Label
            The morphed label.

        See Also
        --------
        mne.morph_labels : Morph a set of labels.

        Notes
        -----
        This function will set label.pos to be all zeros. If the positions
        on the new surface are required, consider using `mne.read_surface`
        with `label.vertices`.
        """
        from .morph import compute_source_morph, grade_to_vertices
        subject_from = _check_subject(self.subject, subject_from)
        if not isinstance(subject_to, str):
            raise TypeError('"subject_to" must be entered as a string')
        if not isinstance(smooth, int):
            raise TypeError('smooth must be an integer')
        if np.all(self.values == 0):
            raise ValueError('Morphing label with all zero values will result '
                             'in the label having no vertices. Consider using '
                             'something like label.values.fill(1.0).')
        idx = 0 if self.hemi == 'lh' else 1
        if isinstance(grade, np.ndarray):
            grade_ = [np.array([], int)] * 2
            grade_[idx] = grade
            grade = grade_
            del grade_
        grade = grade_to_vertices(subject_to, grade, subjects_dir=subjects_dir)
        spacing = [np.array([], int)] * 2
        spacing[idx] = grade[idx]
        vertices = [np.array([], int)] * 2
        vertices[idx] = self.vertices
        data = self.values[:, np.newaxis]
        assert len(data) == sum(len(v) for v in vertices)
        stc = SourceEstimate(data, vertices, tmin=1, tstep=1,
                             subject=subject_from)
        stc = compute_source_morph(
            stc, subject_from, subject_to, spacing=spacing, smooth=smooth,
            subjects_dir=subjects_dir, warn=False).apply(stc)
        inds = np.nonzero(stc.data)[0]
        self.values = stc.data[inds, :].ravel()
        self.pos = np.zeros((len(inds), 3))
        self.vertices = stc.vertices[idx][inds]
        self.subject = subject_to
        return self

    @fill_doc
    def split(self, parts=2, subject=None, subjects_dir=None,
              freesurfer=False):
        """Split the Label into two or more parts.

        Parameters
        ----------
        parts : int >= 2 | tuple of str | str
            Number of labels to create (default is 2), or tuple of strings
            specifying label names for new labels (from posterior to anterior),
            or 'contiguous' to split the label into connected components.
            If a number or 'contiguous' is specified, names of the new labels
            will be the input label's name with div1, div2 etc. appended.
        subject : None | str
            Subject which this label belongs to (needed to locate surface file;
            should only be specified if it is not specified in the label).
        %(subjects_dir)s
        freesurfer : bool
            By default (``False``) ``split_label`` uses an algorithm that is
            slightly optimized for performance and numerical precision. Set
            ``freesurfer`` to ``True`` in order to replicate label splits from
            FreeSurfer's ``mris_divide_parcellation``.

        Returns
        -------
        labels : list of Label, shape (n_parts,)
            The labels, starting from the lowest to the highest end of the
            projection axis.

        Notes
        -----
        If using 'contiguous' split, you must ensure that the label being split
        uses the same triangular resolution as the surface mesh files in
        ``subjects_dir`` Also, some small fringe labels may be returned that
        are close (but not connected) to the large components.

        The spatial split finds the label's principal eigen-axis on the
        spherical surface, projects all label vertex coordinates onto this
        axis, and divides them at regular spatial intervals.
        """
        if isinstance(parts, str) and parts == 'contiguous':
            return _split_label_contig(self, subject, subjects_dir)
        elif isinstance(parts, (tuple, int)):
            return split_label(self, parts, subject, subjects_dir, freesurfer)
        else:
            raise ValueError("Need integer, tuple of strings, or string "
                             "('contiguous'). Got %s)" % type(parts))

    def get_vertices_used(self, vertices=None):
        """Get the source space's vertices inside the label.

        Parameters
        ----------
        vertices : ndarray of int, shape (n_vertices,) | None
            The set of vertices to compare the label to. If None, equals to
            ``np.arange(10242)``. Defaults to None.

        Returns
        -------
        label_verts : ndarray of in, shape (n_label_vertices,)
            The vertices of the label corresponding used by the data.
        """
        if vertices is None:
            vertices = np.arange(10242)

        label_verts = vertices[np.in1d(vertices, self.vertices)]
        return label_verts

    def get_tris(self, tris, vertices=None):
        """Get the source space's triangles inside the label.

        Parameters
        ----------
        tris : ndarray of int, shape (n_tris, 3)
            The set of triangles corresponding to the vertices in a
            source space.
        vertices : ndarray of int, shape (n_vertices,) | None
            The set of vertices to compare the label to. If None, equals to
            ``np.arange(10242)``. Defaults to None.

        Returns
        -------
        label_tris : ndarray of int, shape (n_tris, 3)
            The subset of tris used by the label.
        """
        vertices_ = self.get_vertices_used(vertices)
        selection = np.all(np.in1d(tris, vertices_).reshape(tris.shape),
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

    @fill_doc
    def center_of_mass(self, subject=None, restrict_vertices=False,
                       subjects_dir=None, surf='sphere'):
        """Compute the center of mass of the label.

        This function computes the spatial center of mass on the surface
        as in [1]_.

        Parameters
        ----------
        subject : str | None
            The subject the label is defined for.
        restrict_vertices : bool | array of int | instance of SourceSpaces
            If True, returned vertex will be one from the label. Otherwise,
            it could be any vertex from surf. If an array of int, the
            returned vertex will come from that array. If instance of
            SourceSpaces (as of 0.13), the returned vertex will be from
            the given source space. For most accuruate estimates, do not
            restrict vertices.
        %(subjects_dir)s
        surf : str
            The surface to use for Euclidean distance center of mass
            finding. The default here is "sphere", which finds the center
            of mass on the spherical surface to help avoid potential issues
            with cortical folding.

        Returns
        -------
        vertex : int
            Vertex of the spatial center of mass for the inferred hemisphere,
            with each vertex weighted by its label value.

        See Also
        --------
        SourceEstimate.center_of_mass
        vertex_to_mni

        Notes
        -----
        .. versionadded:: 0.13

        References
        ----------
        .. [1] Larson and Lee, "The cortical dynamics underlying effective
               switching of auditory spatial attention", NeuroImage 2012.
        """
        if not isinstance(surf, str):
            raise TypeError('surf must be a string, got %s' % (type(surf),))
        subject = _check_subject(self.subject, subject)
        if np.any(self.values < 0):
            raise ValueError('Cannot compute COM with negative values')
        if np.all(self.values == 0):
            raise ValueError('Cannot compute COM with all values == 0. For '
                             'structural labels, consider setting to ones via '
                             'label.values[:] = 1.')
        vertex = _center_of_mass(self.vertices, self.values, self.hemi, surf,
                                 subject, subjects_dir, restrict_vertices)
        return vertex


def _get_label_src(label, src):
    _validate_type(src, SourceSpaces, 'src')
    if src.kind != 'surface':
        raise RuntimeError('Cannot operate on SourceSpaces that are not '
                           'surface type, got %s' % (src.kind,))
    if label.hemi == 'lh':
        hemi_src = src[0]
    else:
        hemi_src = src[1]
    return hemi_src


class BiHemiLabel(object):
    """A freesurfer/MNE label with vertices in both hemispheres.

    Parameters
    ----------
    lh : Label
        Label for the left hemisphere.
    rh : Label
        Label for the right hemisphere.
    name : None | str
        Name for the label.
    color : None | color
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

    def __init__(self, lh, rh, name=None, color=None):  # noqa: D102
        if lh.subject != rh.subject:
            raise ValueError('lh.subject (%s) and rh.subject (%s) must '
                             'agree' % (lh.subject, rh.subject))
        self.lh = lh
        self.rh = rh
        self.name = name
        self.subject = lh.subject
        self.color = color
        self.hemi = 'both'

    def __repr__(self):  # noqa: D105
        temp = "<BiHemiLabel  |  %s, lh : %i vertices,  rh : %i vertices>"
        name = 'unknown, ' if self.subject is None else self.subject + ', '
        name += repr(self.name) if self.name is not None else "unnamed"
        return temp % (name, len(self.lh), len(self.rh))

    def __len__(self):
        """Return the number of vertices."""
        return len(self.lh) + len(self.rh)

    def __add__(self, other):
        """Add labels."""
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
        """Subtract labels."""
        _validate_type(other, (Label, BiHemiLabel), 'other')
        if isinstance(other, Label):
            if other.hemi == 'lh':
                lh = self.lh - other
                rh = self.rh
            else:
                rh = self.rh - other
                lh = self.lh
        else:  # isinstance(other, BiHemiLabel)
            lh = self.lh - other.lh
            rh = self.rh - other.rh

        if len(lh.vertices) == 0:
            return rh
        elif len(rh.vertices) == 0:
            return lh
        else:
            name = '%s - %s' % (self.name, other.name)
            return BiHemiLabel(lh, rh, name, self.color)


def read_label(filename, subject=None, color=None):
    """Read FreeSurfer Label file.

    Parameters
    ----------
    filename : str
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
    if subject is not None and not isinstance(subject, str):
        raise TypeError('subject must be a string')

    # find hemi
    basename = op.basename(filename)
    if basename.endswith('lh.label') or basename.startswith('lh.'):
        hemi = 'lh'
    elif basename.endswith('rh.label') or basename.startswith('rh.'):
        hemi = 'rh'
    else:
        raise ValueError('Cannot find which hemisphere it is. File should end'
                         ' with lh.label or rh.label: %s' % (basename,))

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
    """Write a FreeSurfer label.

    Parameters
    ----------
    filename : str
        Path to label file to produce.
    label : Label
        The label object to save.
    %(verbose)s

    See Also
    --------
    write_labels_to_annot

    Notes
    -----
    Note that due to file specification limitations, the Label's subject and
    color attributes are not saved to disk.
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
        fid.write(b'#%s\n' % label.comment.encode())
        fid.write(b'%d\n' % n_vertices)
        for d in data:
            fid.write(b'%d %f %f %f %f\n' % tuple(d))


def _prep_label_split(label, subject=None, subjects_dir=None):
    """Get label and subject information prior to label splitting."""
    # If necessary, find the label
    if isinstance(label, BiHemiLabel):
        raise TypeError("Can only split labels restricted to one hemisphere.")
    elif isinstance(label, str):
        label = read_label(label)

    # Find the subject
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

    return label, subject, subjects_dir


def _split_label_contig(label_to_split, subject=None, subjects_dir=None):
    """Split label into contiguous regions (i.e., connected components).

    Parameters
    ----------
    label_to_split : Label | str
        Label which is to be split (Label object or path to a label file).
    subject : None | str
        Subject which this label belongs to (needed to locate surface file;
        should only be specified if it is not specified in the label).
    %(subjects_dir)s

    Returns
    -------
    labels : list of Label
        The contiguous labels, in order of descending size.
    """
    # Convert to correct input if necessary
    label_to_split, subject, subjects_dir = _prep_label_split(label_to_split,
                                                              subject,
                                                              subjects_dir)

    # Find the spherical surface to get vertices and tris
    surf_fname = '.'.join((label_to_split.hemi, 'sphere'))
    surf_path = op.join(subjects_dir, subject, 'surf', surf_fname)
    surface_points, surface_tris = read_surface(surf_path)

    # Get vertices we want to keep and compute mesh edges
    verts_arr = label_to_split.vertices
    edges_all = mesh_edges(surface_tris)

    # Subselect rows and cols of vertices that belong to the label
    select_edges = edges_all[verts_arr][:, verts_arr].tocoo()

    # Compute connected components and store as lists of vertex numbers
    comp_labels = _get_components(verts_arr, select_edges)

    # Convert to indices in the original surface space
    label_divs = []
    for comp in comp_labels:
        label_divs.append(verts_arr[comp])

    # Construct label division names
    n_parts = len(label_divs)
    if label_to_split.name.endswith(('lh', 'rh')):
        basename = label_to_split.name[:-3]
        name_ext = label_to_split.name[-3:]
    else:
        basename = label_to_split.name
        name_ext = ''
    name_pattern = "%s_div%%i%s" % (basename, name_ext)
    names = tuple(name_pattern % i for i in range(1, n_parts + 1))

    # Colors
    if label_to_split.color is None:
        colors = (None,) * n_parts
    else:
        colors = _split_colors(label_to_split.color, n_parts)

    # Sort label divisions by their size (in vertices)
    label_divs.sort(key=lambda x: len(x), reverse=True)
    labels = []
    for div, name, color in zip(label_divs, names, colors):
        # Get indices of dipoles within this division of the label
        verts = np.array(sorted(list(div)), int)
        vert_indices = np.in1d(verts_arr, verts, assume_unique=True)

        # Set label attributes
        pos = label_to_split.pos[vert_indices]
        values = label_to_split.values[vert_indices]
        hemi = label_to_split.hemi
        comment = label_to_split.comment
        lbl = Label(verts, pos, values, hemi, comment, name, None, subject,
                    color)
        labels.append(lbl)

    return labels


@fill_doc
def split_label(label, parts=2, subject=None, subjects_dir=None,
                freesurfer=False):
    """Split a Label into two or more parts.

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
    %(subjects_dir)s
    freesurfer : bool
        By default (``False``) ``split_label`` uses an algorithm that is
        slightly optimized for performance and numerical precision. Set
        ``freesurfer`` to ``True`` in order to replicate label splits from
        FreeSurfer's ``mris_divide_parcellation``.

    Returns
    -------
    labels : list of Label, shape (n_parts,)
        The labels, starting from the lowest to the highest end of the
        projection axis.

    Notes
    -----
    Works by finding the label's principal eigen-axis on the spherical surface,
    projecting all label vertex coordinates onto this axis and dividing them at
    regular spatial intervals.
    """
    label, subject, subjects_dir = _prep_label_split(label, subject,
                                                     subjects_dir)

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

    # find the spherical surface
    surf_fname = '.'.join((label.hemi, 'sphere'))
    surf_path = op.join(subjects_dir, subject, "surf", surf_fname)
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


def label_sign_flip(label, src):
    """Compute sign for label averaging.

    Parameters
    ----------
    label : Label | BiHemiLabel
        A label.
    src : SourceSpaces
        The source space over which the label is defined.

    Returns
    -------
    flip : array
        Sign flip vector (contains 1 or -1).
    """
    if len(src) != 2:
        raise ValueError('Only source spaces with 2 hemisphers are accepted')

    lh_vertno = src[0]['vertno']
    rh_vertno = src[1]['vertno']

    # get source orientations
    ori = list()
    if label.hemi in ('lh', 'both'):
        vertices = label.vertices if label.hemi == 'lh' else label.lh.vertices
        vertno_sel = np.intersect1d(lh_vertno, vertices)
        ori.append(src[0]['nn'][vertno_sel])
    if label.hemi in ('rh', 'both'):
        vertices = label.vertices if label.hemi == 'rh' else label.rh.vertices
        vertno_sel = np.intersect1d(rh_vertno, vertices)
        ori.append(src[1]['nn'][vertno_sel])
    if len(ori) == 0:
        raise Exception('Unknown hemisphere type "%s"' % (label.hemi,))
    ori = np.concatenate(ori, axis=0)
    if len(ori) == 0:
        return np.array([], int)

    _, _, Vh = linalg.svd(ori, full_matrices=False)

    # The sign of Vh is ambiguous, so we should align to the max-positive
    # (outward) direction
    dots = np.dot(ori, Vh[0])
    if np.mean(dots) < 0:
        dots *= -1

    # Comparing to the direction of the first right singular vector
    flip = np.sign(dots)
    return flip


@verbose
def stc_to_label(stc, src=None, smooth=True, connected=False,
                 subjects_dir=None, verbose=None):
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
    %(subjects_dir)s
    %(verbose)s

    Returns
    -------
    labels : list of Label | list of list of Label
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
    if isinstance(src, str):
        subject = src
    else:
        subject = stc.subject

    if not isinstance(stc, SourceEstimate):
        raise ValueError('SourceEstimate should be surface source estimates')

    if isinstance(src, str):
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
    """Find all vertices wihin a maximum geodesic distance from source.

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

    verts = np.sort(np.array(list(dist_map.keys()), int))
    dist = np.array([dist_map[v] for v in verts], int)

    return verts, dist


def _grow_labels(seeds, extents, hemis, names, dist, vert, subject):
    """Parallelize grow_labels."""
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


@fill_doc
def grow_labels(subject, seeds, extents, hemis, subjects_dir=None, n_jobs=1,
                overlap=True, names=None, surface='white'):
    """Generate circular labels in source space with region growing.

    This function generates a number of labels in source space by growing
    regions starting from the vertices defined in "seeds". For each seed, a
    label is generated containing all vertices within a maximum geodesic
    distance on the white matter surface from the seed.

    Parameters
    ----------
    subject : str
        Name of the subject as in SUBJECTS_DIR.
    seeds : int | list
        Seed, or list of seeds. Each seed can be either a vertex number or
        a list of vertex numbers.
    extents : array | float
        Extents (radius in mm) of the labels.
    hemis : array | int
        Hemispheres to use for the labels (0: left, 1: right).
    %(subjects_dir)s
    %(n_jobs)s
        Likely only useful if tens or hundreds of labels are being expanded
        simultaneously. Does not apply with ``overlap=False``.
    overlap : bool
        Produce overlapping labels. If True (default), the resulting labels
        can be overlapping. If False, each label will be grown one step at a
        time, and occupied territory will not be invaded.
    names : None | list of str
        Assign names to the new labels (list needs to have the same length as
        seeds).
    surface : str
        The surface used to grow the labels, defaults to the white surface.

    Returns
    -------
    labels : list of Label
        The labels' ``comment`` attribute contains information on the seed
        vertex and extent; the ``values``  attribute contains distance from the
        seed in millimeters.

    Notes
    -----
    "extents" and "hemis" can either be arrays with the same length as
    seeds, which allows using a different extent and hemisphere for
    label, or integers, in which case the same extent and hemisphere is
    used for each label.
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    n_jobs = check_n_jobs(n_jobs)

    # make sure the inputs are arrays
    if np.isscalar(seeds):
        seeds = [seeds]
    # these can have different sizes so need to use object array
    seeds = np.asarray([np.atleast_1d(seed) for seed in seeds], dtype='O')
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
    """Grow labels while ensuring that they don't overlap."""
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
        for i in range(n_labels):
            vertices = np.nonzero(parc == i)[0]
            name = str(names[i])
            label_ = Label(vertices, hemi=hemi, name=name, subject=subject)
            labels.append(label_)

    return labels


@fill_doc
def random_parcellation(subject, n_parcel, hemi, subjects_dir=None,
                        surface='white', random_state=None):
    """Generate random cortex parcellation by growing labels.

    This function generates a number of labels which don't intersect and
    cover the whole surface. Regions are growing around randomly chosen
    seeds.

    Parameters
    ----------
    subject : str
        Name of the subject as in SUBJECTS_DIR.
    n_parcel : int
        Total number of cortical parcels.
    hemi : str
        Hemisphere id (ie 'lh', 'rh', 'both'). In the case
        of 'both', both hemispheres are processed with (n_parcel // 2)
        parcels per hemisphere.
    %(subjects_dir)s
    surface : str
        The surface used to grow the labels, defaults to the white surface.
    %(random_state)s

    Returns
    -------
    labels : list of Label
        Random cortex parcellation.
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    if hemi == 'both':
        hemi = ['lh', 'rh']
    hemis = np.atleast_1d(hemi)

    # load the surfaces and create the distance graphs
    tris, vert, dist = {}, {}, {}
    for hemi in set(hemis):
        surf_fname = op.join(subjects_dir, subject, 'surf', hemi + '.' +
                             surface)
        vert[hemi], tris[hemi] = read_surface(surf_fname)
        dist[hemi] = mesh_dist(tris[hemi], vert[hemi])

    # create the patches
    labels = _cortex_parcellation(subject, n_parcel, hemis, vert, dist,
                                  random_state)

    # add a unique color to each label
    colors = _n_colors(len(labels))
    for label, color in zip(labels, colors):
        label.color = color

    return labels


def _cortex_parcellation(subject, n_parcel, hemis, vertices_, graphs,
                         random_state=None):
    """Random cortex parcellation."""
    labels = []
    rng = check_random_state(random_state)
    for hemi in set(hemis):
        parcel_size = len(hemis) * len(vertices_[hemi]) // n_parcel
        graph = graphs[hemi]  # distance graph
        n_vertices = len(vertices_[hemi])

        # prepare parcellation
        parc = np.full(n_vertices, -1, dtype='int32')

        # initialize active sources
        s = rng.choice(range(n_vertices))
        label_idx = 0
        edge = [s]  # queue of vertices to process
        parc[s] = label_idx
        label_size = 1
        rest = len(parc) - 1
        # grow from sources
        while rest:
            # if there are not free neighbors, start new parcel
            if not edge:
                rest_idx = np.where(parc < 0)[0]
                s = rng.choice(rest_idx)
                edge = [s]
                label_idx += 1
                label_size = 1
                parc[s] = label_idx
                rest -= 1

            vert_from = edge.pop(0)

            # add neighbors within allowable distance
            # row = graph[vert_from, :]
            # row_indices, row_data = row.indices, row.data
            sl = slice(graph.indptr[vert_from], graph.indptr[vert_from + 1])
            row_indices, row_data = graph.indices[sl], graph.data[sl]
            for vert_to, dist in zip(row_indices, row_data):
                vert_to_label = parc[vert_to]

                # abort if the vertex is already occupied
                if vert_to_label >= 0:
                    continue

                # abort if outside of extent
                if label_size > parcel_size:
                    label_idx += 1
                    label_size = 1
                    edge = [vert_to]
                    parc[vert_to] = label_idx
                    rest -= 1
                    break

                # assign label value
                parc[vert_to] = label_idx
                label_size += 1
                edge.append(vert_to)
                rest -= 1

        # merging small labels
        # label connectivity matrix
        n_labels = label_idx + 1
        label_sizes = np.empty(n_labels, dtype=int)
        label_conn = np.zeros([n_labels, n_labels], dtype='bool')
        for i in range(n_labels):
            vertices = np.nonzero(parc == i)[0]
            label_sizes[i] = len(vertices)
            neighbor_vertices = graph[vertices, :].indices
            neighbor_labels = np.unique(np.array(parc[neighbor_vertices]))
            label_conn[i, neighbor_labels] = 1
        np.fill_diagonal(label_conn, 0)

        # merging
        label_id = range(n_labels)
        while n_labels > n_parcel // len(hemis):
            # smallest label and its smallest neighbor
            i = np.argmin(label_sizes)
            neighbors = np.nonzero(label_conn[i, :])[0]
            j = neighbors[np.argmin(label_sizes[neighbors])]

            # merging two labels
            label_conn[j, :] += label_conn[i, :]
            label_conn[:, j] += label_conn[:, i]
            label_conn = np.delete(label_conn, i, 0)
            label_conn = np.delete(label_conn, i, 1)
            label_conn[j, j] = 0
            label_sizes[j] += label_sizes[i]
            label_sizes = np.delete(label_sizes, i, 0)
            n_labels -= 1
            vertices = np.nonzero(parc == label_id[i])[0]
            parc[vertices] = label_id[j]
            label_id = np.delete(label_id, i, 0)

        # convert parc to labels
        for i in range(n_labels):
            vertices = np.nonzero(parc == label_id[i])[0]
            name = 'label_' + str(i)
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
        cands = sorted(set(c.lstrip('lh.').lstrip('rh.').rstrip('.annot')
                           for c in cands if '.annot' in c),
                       key=lambda x: x.lower())
        if len(cands) == 0:
            raise IOError('No such file %s, no candidate parcellations '
                          'found in directory' % fname)
        else:
            raise IOError('No such file %s, candidate parcellations in '
                          'that directory:\n%s' % (fname, '\n'.join(cands)))
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
            np.fromfile(fid, '>c', length)  # discard orig_tab

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
            np.fromfile(fid, "|S%d" % length, 1)  # Orig table path
            entries_to_read = np.fromfile(fid, '>i4', 1)[0]
            names = list()
            for i in range(entries_to_read):
                np.fromfile(fid, '>i4', 1)  # Structure
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
    """Get the .annot filenames and hemispheres."""
    if annot_fname is not None:
        # we use use the .annot file specified by the user
        hemis = [op.basename(annot_fname)[:2]]
        if hemis[0] not in ['lh', 'rh']:
            raise ValueError('Could not determine hemisphere from filename, '
                             'filename has to start with "lh" or "rh".')
        annot_fname = [annot_fname]
    else:
        # construct .annot file names for requested subject, parc, hemi
        _check_option('hemi', hemi, ['lh', 'rh', 'both'])
        if hemi == 'both':
            hemis = ['lh', 'rh']
        else:
            hemis = [hemi]

        subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
        dst = op.join(subjects_dir, subject, 'label', '%%s.%s.annot' % parc)
        annot_fname = [dst % hemi_ for hemi_ in hemis]

    return annot_fname, hemis


def _load_vert_pos(subject, subjects_dir, surf_name, hemi, n_expected,
                   extra=''):
    fname_surf = op.join(subjects_dir, subject, 'surf',
                         '%s.%s' % (hemi, surf_name))
    vert_pos, _ = read_surface(fname_surf)
    vert_pos /= 1e3  # the positions in labels are in meters
    if len(vert_pos) != n_expected:
        raise RuntimeError('Number of surface vertices (%s) for subject %s'
                           ' does not match the expected number of vertices'
                           '(%s)%s'
                           % (len(vert_pos), subject, n_expected, extra))
    return vert_pos


@verbose
def read_labels_from_annot(subject, parc='aparc', hemi='both',
                           surf_name='white', annot_fname=None, regexp=None,
                           subjects_dir=None, verbose=None):
    """Read labels from a FreeSurfer annotation file.

    Note: Only cortical labels will be returned.

    Parameters
    ----------
    subject : str
        The subject for which to read the parcellation.
    parc : str
        The parcellation to use, e.g., 'aparc' or 'aparc.a2009s'.
    hemi : str
        The hemisphere from which to read the parcellation, can be 'lh', 'rh',
        or 'both'.
    surf_name : str
        Surface used to obtain vertex locations, e.g., 'white', 'pial'.
    annot_fname : str or None
        Filename of the .annot file. If not None, only this file is read
        and 'parc' and 'hemi' are ignored.
    regexp : str
        Regular expression or substring to select particular labels from the
        parcellation. E.g. 'superior' will return all labels in which this
        substring is contained.
    %(subjects_dir)s
    %(verbose)s

    Returns
    -------
    labels : list of Label
        The labels, sorted by label name (ascending).
    """
    logger.info('Reading labels from parcellation...')

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
        label_rgbas = ctab[:, :4] / 255.
        label_ids = ctab[:, -1]

        # load the vertex positions from surface
        vert_pos = _load_vert_pos(
            subject, subjects_dir, surf_name, hemi, len(annot),
            extra='for annotation file %s' % fname)
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
            label = Label(vertices, pos, hemi=hemi, name=name,
                          subject=subject, color=tuple(label_rgba))
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

    return labels


def _check_labels_subject(labels, subject, name):
    _validate_type(labels, (list, tuple), 'labels')
    for label in labels:
        _validate_type(label, Label, 'each entry in labels')
        if subject is None:
            subject = label.subject
        if subject is not None:  # label.subject can be None, depending on init
            if subject != label.subject:
                raise ValueError('Got multiple values of %s: %s and %s'
                                 % (name, subject, label.subject))
    if subject is None:
        raise ValueError('if label.subject is None for all labels, '
                         '%s must be provided' % name)
    return subject


@verbose
def morph_labels(labels, subject_to, subject_from=None, subjects_dir=None,
                 surf_name='white', verbose=None):
    """Morph a set of labels.

    This is useful when morphing a set of non-overlapping labels (such as those
    obtained with :func:`read_labels_from_annot`) from one subject to
    another.

    Parameters
    ----------
    labels : list
        The labels to morph.
    subject_to : str
        The subject to morph labels to.
    subject_from : str | None
        The subject to morph labels from. Can be None if the labels
        have the ``.subject`` property defined.
    %(subjects_dir)s
    surf_name : str
        Surface used to obtain vertex locations, e.g., 'white', 'pial'.
    %(verbose)s

    Returns
    -------
    labels : list
        The morphed labels.

    See Also
    --------
    read_labels_from_annot
    mne.Label.morph

    Notes
    -----
    This does not use the same algorithm as Freesurfer, so the results
    morphing (e.g., from ``'fsaverage'`` to your subject) might not match
    what Freesurfer produces during ``recon-all``.

    .. versionadded:: 0.18
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    subject_from = _check_labels_subject(labels, subject_from, 'subject_from')
    mmaps = read_morph_map(subject_from, subject_to, subjects_dir)
    vert_poss = [_load_vert_pos(subject_to, subjects_dir, surf_name, hemi,
                                mmap.shape[0])
                 for hemi, mmap in zip(('lh', 'rh'), mmaps)]
    idxs = [mmap.argmax(axis=1) for mmap in mmaps]
    out_labels = list()
    values = filename = None
    for label in labels:
        li = dict(lh=0, rh=1)[label.hemi]
        vertices = np.where(np.in1d(idxs[li], label.vertices))[0]
        pos = vert_poss[li][vertices]
        out_labels.append(
            Label(vertices, pos, values, label.hemi, label.comment, label.name,
                  filename, subject_to, label.color, label.verbose))
    return out_labels


@verbose
def labels_to_stc(labels, values, tmin=0, tstep=1, subject=None, verbose=None):
    """Convert a set of labels and values to a STC.

    This function is meant to work like the opposite of
    `extract_label_time_course`.

    Parameters
    ----------
    labels : list of Label
        The labels. Must not overlap.
    values : ndarray, shape (n_labels, ...)
        The values in each label. Can be 1D or 2D.
    tmin : float
        The tmin to use for the STC.
    tstep : float
        The tstep to use for the STC.
    subject : str | None
        The subject for which to create the STC.
    %(verbose)s

    Returns
    -------
    stc : instance of SourceEstimate
        The values-in-labels converted to a STC.

    See Also
    --------
    extract_label_time_course

    Notes
    -----
    Vertices that appear in more than one label will be averaged.

    .. versionadded:: 0.18
    """
    subject = _check_labels_subject(labels, subject, 'subject')
    values = np.array(values, float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2:
        raise ValueError('values must have 1 or 2 dimensions, got %s'
                         % (values.ndim,))
    if len(labels) != len(values):
        raise ValueError('values.shape[0] (%s) must match len(labels) (%s)'
                         % (values.shape[0], len(labels)))
    vertices = dict(lh=[], rh=[])
    data = dict(lh=[], rh=[])
    for li, label in enumerate(labels):
        data[label.hemi].append(
            np.repeat(values[li][np.newaxis], len(label.vertices), axis=0))
        vertices[label.hemi].append(label.vertices)
    hemis = ('lh', 'rh')
    for hemi in hemis:
        vertices[hemi] = np.concatenate(vertices[hemi], axis=0)
        data[hemi] = np.concatenate(data[hemi], axis=0).astype(float)
        cols = np.arange(len(vertices[hemi]))
        vertices[hemi], rows = np.unique(vertices[hemi], return_inverse=True)
        mat = sparse.coo_matrix((np.ones(len(rows)), (rows, cols))).tocsr()
        mat = mat * sparse.diags(1. / np.asarray(mat.sum(axis=-1))[:, 0])
        data[hemi] = mat.dot(data[hemi])
    vertices = [vertices[hemi] for hemi in hemis]
    data = np.concatenate([data[hemi] for hemi in hemis], axis=0)
    stc = SourceEstimate(data, vertices, tmin, tstep, subject, verbose)
    return stc


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
        np.frombuffer(table_name.encode('ascii'), dtype=np.uint8).tofile(fid)

        # number of entries to write
        np.array(n_entries, dtype='>i4').tofile(fid)

        # write entries
        for ii, (name, color) in enumerate(zip(names, ctab)):
            np.array(ii, dtype='>i4').tofile(fid)
            np.array(len(name), dtype='>i4').tofile(fid)
            np.frombuffer(name.encode('ascii'), dtype=np.uint8).tofile(fid)
            np.array(color[:4], dtype='>i4').tofile(fid)


@verbose
def write_labels_to_annot(labels, subject=None, parc=None, overwrite=False,
                          subjects_dir=None, annot_fname=None,
                          colormap='hsv', hemi='both', verbose=None):
    r"""Create a FreeSurfer annotation from a list of labels.

    Parameters
    ----------
    labels : list with instances of mne.Label
        The labels to create a parcellation from.
    subject : str | None
        The subject for which to write the parcellation.
    parc : str | None
        The parcellation name to use.
    overwrite : bool
        Overwrite files if they already exist.
    %(subjects_dir)s
    annot_fname : str | None
        Filename of the .annot file. If not None, only this file is written
        and 'parc' and 'subject' are ignored.
    colormap : str
        Colormap to use to generate label colors for labels that do not
        have a color specified.
    hemi : 'both' | 'lh' | 'rh'
        The hemisphere(s) for which to write \*.annot files (only applies if
        annot_fname is not specified; default is 'both').
    %(verbose)s

    Notes
    -----
    Vertices that are not covered by any of the labels are assigned to a label
    named "unknown".
    """
    logger.info('Writing labels to parcellation...')

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
            ctab_rgb = ctab[:, :3]
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
                    warn('At least one label contains a color with, "r=0, '
                         'g=0, b=0" value. Some FreeSurfer tools may fail '
                         'to read the parcellation')

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
                for i in range(n_hemi_labels):
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
            fpath = op.join(subjects_dir, subject, 'surf', '%s.white' % hemi)
            points, _ = read_surface(fpath)
            n_vertices = len(points)
        else:
            if len(hemi_labels) > 0:
                max_vert = max(np.max(label.vertices) for label in hemi_labels)
                n_vertices = max_vert + 1
            else:
                n_vertices = 1
            warn('Number of vertices in the surface could not be '
                 'verified because the surface file could not be found; '
                 'specify subject and subjects_dir parameters.')

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

        if None in hemi_names:
            msg = ("Found %i labels with no name. Writing annotation file"
                   "requires all labels named" % (hemi_names.count(None)))
            # raise the error immediately rather than crash with an
            # uninformative error later (e.g. cannot join NoneType)
            raise ValueError(msg)

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
        issues.append('\n'.join(duplicate_colors))
    if invalid_colors:
        msg = ("Some labels have invalid color values (all colors should be "
               "RGBA tuples with values between 0 and 1)")
        invalid_colors.insert(0, msg)
        issues.append('\n'.join(invalid_colors))
    if overlap:
        msg = ("Some labels occupy vertices that are also occupied by one or "
               "more other labels. Each vertex can only be occupied by a "
               "single label in *.annot files.")
        overlap.insert(0, msg)
        issues.append('\n'.join(overlap))

    if issues:
        raise ValueError('\n\n'.join(issues))

    # write it
    for fname, annot, ctab, hemi_names in to_save:
        logger.info('   writing %d labels to %s' % (len(hemi_names), fname))
        _write_annot(fname, annot, ctab, hemi_names)


@fill_doc
def select_sources(subject, label, location='center', extent=0.,
                   grow_outside=True, subjects_dir=None, name=None,
                   random_state=None, surf='white'):
    """Select sources from a label.

    Parameters
    ----------
    subject : string
        Name of the subject as in SUBJECTS_DIR.
    label : instance of Label | str
        Define where the seed will be chosen. If str, can be 'lh' or 'rh',
        which correspond to left or right hemisphere, respectively.
    location : 'random' | 'center' | int
        Location to grow label from. If the location is an int, it represents
        the vertex number in the corresponding label. If it is a str, it can be
        either 'random' or 'center'.
    extent : float
        Extents (radius in mm) of the labels, i.e. maximum geodesic distance
        on the white matter surface from the seed. If 0, the resulting label
        will contain only one vertex.
    grow_outside : bool
        Let the region grow outside the original label where location was
        defined.
    %(subjects_dir)s
    name : None | str
        Assign name to the new label.
    %(random_state)s
    surf : string
        The surface used to simulated the label, defaults to the white surface.

    Returns
    -------
    label : instance of Label
        The label that contains the selected sources.

    Notes
    -----
    This function selects a region of interest on the cortical surface based
    on a label (or a hemisphere). The sources are selected by growing a region
    around a seed which is selected randomly, is the center of the label, or
    is a specific vertex. The selected vertices can extend beyond the initial
    provided label. This can be prevented by setting grow_outside to False.

    The selected sources are returned in the form of a new Label object. The
    values of the label contain the distance from the seed in millimeters.

    .. versionadded:: 0.18
    """
    # If label is a string, convert it to a label that contains the whole
    # hemisphere.
    if isinstance(label, str):
        _check_option('label', label, ['lh', 'rh'])
        surf_filename = op.join(subjects_dir, subject, 'surf',
                                label + '.white')
        vertices, _ = read_surface(surf_filename)
        indices = np.arange(len(vertices), dtype=int)
        label = Label(indices, vertices, hemi=label)

    # Choose the seed according to the selected strategy.
    if isinstance(location, str):
        _check_option('location', location, ['center', 'random'])

        if location == 'center':
            seed = label.center_of_mass(
                subject, restrict_vertices=True, subjects_dir=subjects_dir,
                surf=surf)
        else:
            rng = check_random_state(random_state)
            seed = rng.choice(label.vertices)
    else:
        seed = label.vertices[location]

    hemi = 0 if label.hemi == 'lh' else 1
    new_label = grow_labels(subject, seed, extent, hemi, subjects_dir)[0]

    # We override the name because grow_label automatically adds a -rh or -lh
    # to the given parameter.
    new_label.name = name

    # Restrict the new label to the vertices of the input label if needed.
    if not grow_outside:
        to_keep = np.array([v in label.vertices for v in new_label.vertices])
        new_label = Label(new_label.vertices[to_keep], new_label.pos[to_keep],
                          hemi=new_label.hemi, name=name, subject=subject)

    return new_label
