# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import os
import copy
from math import ceil
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
import warnings

from .parallel import parallel_func


def read_stc(filename):
    """Read an STC file and return as dict

    STC files contain activations or source reconstructions

    Parameters
    ----------
    filename: string
        The name of the STC file

    Returns
    -------
    data: dict
        The STC structure. It has the following keys:
           tmin           The first time point of the data in seconds
           tstep          Time between frames in seconds
           vertices       vertex indices (0 based)
           data           The data matrix (nvert * ntime)

    See Also
    --------
    read_source_estimate

    """
    fid = open(filename, 'rb')

    stc = dict()

    fid.seek(0, 2)  # go to end of file
    file_length = fid.tell()
    fid.seek(0, 0)  # go to beginning of file

    # read tmin in ms
    stc['tmin'] = float(np.fromfile(fid, dtype=">f4", count=1))
    stc['tmin'] /= 1000.0

    # read sampling rate in ms
    stc['tstep'] = float(np.fromfile(fid, dtype=">f4", count=1))
    stc['tstep'] /= 1000.0

    # read number of vertices/sources
    vertices_n = int(np.fromfile(fid, dtype=">u4", count=1))

    # read the source vector
    stc['vertices'] = np.fromfile(fid, dtype=">u4", count=vertices_n)

    # read the number of timepts
    data_n = int(np.fromfile(fid, dtype=">u4", count=1))

    if ((file_length / 4 - 4 - vertices_n) % (data_n * vertices_n)) != 0:
        raise ValueError('incorrect stc file size')

    # read the data matrix
    stc['data'] = np.fromfile(fid, dtype=">f4", count=vertices_n * data_n)
    stc['data'] = stc['data'].reshape([data_n, vertices_n]).T

    # close the file
    fid.close()
    return stc


def write_stc(filename, tmin, tstep, vertices, data):
    """Write an STC file

    Parameters
    ----------
    filename: string
        The name of the STC file
    tmin: float
        The first time point of the data in seconds
    tstep: float
        Time between frames in seconds
    vertices: array of integers
        Vertex indices (0 based)
    data: 2D array
        The data matrix (nvert * ntime)
    """
    fid = open(filename, 'wb')

    # write start time in ms
    fid.write(np.array(1000 * tmin, dtype='>f4').tostring())
    # write sampling rate in ms
    fid.write(np.array(1000 * tstep, dtype='>f4').tostring())
    # write number of vertices
    fid.write(np.array(vertices.shape[0], dtype='>u4').tostring())
    # write the vertex indices
    fid.write(np.array(vertices, dtype='>u4').tostring())

    # write the number of timepts
    fid.write(np.array(data.shape[1], dtype='>u4').tostring())
    #
    # write the data
    #
    fid.write(np.array(data.T, dtype='>f4').tostring())

    # close the file
    fid.close()


def _read_3(fid):
    """ Read 3 byte integer from file
    """
    data = np.fromfile(fid, dtype=np.uint8, count=3).astype(np.int32)

    out = np.left_shift(data[0], 16) + np.left_shift(data[1], 8) + data[2]

    return out


def read_w(filename):
    """Read a w file and return as dict

    w files contain activations or source reconstructions for a single time
    point

    Parameters
    ----------
    filename: string
        The name of the w file

    Returns
    -------
    data: dict
        The w structure. It has the following keys:
           vertices       vertex indices (0 based)
           data           The data matrix (nvert long)
    """

    fid = open(filename, 'rb')

    # skip first 2 bytes
    fid.read(2)

    # read number of vertices/sources (3 byte integer)
    vertices_n = int(_read_3(fid))

    vertices = np.zeros((vertices_n), dtype=np.int32)
    data = np.zeros((vertices_n), dtype=np.float32)

    # read the vertices and data
    for i in range(vertices_n):
        vertices[i] = _read_3(fid)
        data[i] = np.fromfile(fid, dtype='>f4', count=1)

    w = dict()
    w['vertices'] = vertices
    w['data'] = data

    # close the file
    fid.close()
    return w


def _write_3(fid, val):
    """ Write 3 byte integer to file
    """

    f_bytes = np.zeros((3), dtype=np.uint8)

    f_bytes[0] = (val >> 16) & 255
    f_bytes[1] = (val >> 8) & 255
    f_bytes[2] = val & 255

    fid.write(f_bytes.tostring())


def write_w(filename, vertices, data):
    """Read a w file

    w files contain activations or source reconstructions for a single time
    point

    Parameters
    ----------
    filename: string
        The name of the w file
    vertices: array of integers
        Vertex indices (0 based)
    data: 1D array
        The data array (nvert)


    See Also
    --------
    read_source_estimate

    """

    assert(len(vertices) == len(data))

    fid = open(filename, 'wb')

    # write 2 zero bytes
    fid.write(np.zeros((2), dtype=np.uint8).tostring())

    # write number of vertices/sources (3 byte integer)
    vertices_n = len(vertices)
    _write_3(fid, vertices_n)

    # write the vertices and data
    for i in range(vertices_n):
        _write_3(fid, vertices[i])
        #XXX: without float() endianness is wrong, not sure why
        fid.write(np.array(float(data[i]), dtype='>f4').tostring())

    # close the file
    fid.close()


def read_source_estimate(fname):
    """Returns a SourceEstimate object.

    Parameters
    ----------
    The single argument ``fname`` should provide the path to (a) source-estimate
    file(s) as string.

     - for volume source estimates, ``fname`` should provide the path to a
       single file named '*-vl.stc`
     - for surface source estimates, ``fname`` should either provide the
       path to the file corresponding to a single hemisphere ('*-lh.stc',
       '*-rh.stc') or only specify the asterisk part in these patterns. In any
       case, the function expects files for both hemisphere with names
       following this pattern.
     - for single time point .w files, ``fname`` should follow the same
       pattern as for surface estimates, except that files are named
       '*-lh.w' and '*-rh.w'.

    See Also
    --------
    read_stc, read_w

    """
    fname_arg = fname

    # make sure corresponding file(s) can be found
    ftype = None
    if os.path.exists(fname):
        if fname.endswith('-vl.stc'):
            ftype = 'volume'
        elif fname.endswith('.stc'):
            ftype = 'surface'
            if fname.endswith(('-lh.stc', '-rh.stc')):
                fname = fname[:-7]
            else:
                err = ("Invalid .stc filename: %r; needs to end with "
                       "hemisphere tag ('...-lh.stc' or '...-rh.stc')"
                       % fname)
                raise IOError(err)
        elif fname.endswith('.w'):
            ftype = 'w'
            if fname.endswith(('-lh.w', '-rh.w')):
                fname = fname[:-5]
            else:
                err = ("Invalid .w filename: %r; needs to end with "
                       "hemisphere tag ('...-lh.w' or '...-rh.w')"
                       % fname)
                raise IOError(err)

    if ftype is not 'volume':
        stc_exist = map(os.path.exists, (fname + '-rh.stc', fname + '-lh.stc'))
        w_exist = map(os.path.exists, (fname + '-rh.w', fname + '-lh.w'))
        if all(stc_exist) and (ftype is not 'w'):
            ftype = 'surface'
        elif all(w_exist):
            ftype = 'w'
        elif any(stc_exist) or any(w_exist):
            raise IOError("Hemisphere missing for %r" % fname_arg)
        else:
            raise IOError("SourceEstimate File(s) not found for: %r" % fname_arg)

    # read the files
    if ftype == 'volume':  # volume source space
        kwargs = read_stc(fname)
    elif ftype == 'surface': # stc file with surface source spaces
        lh = read_stc(fname + '-lh.stc')
        rh = read_stc(fname + '-rh.stc')
        assert lh['tmin'] == rh['tmin']
        assert lh['tstep'] == rh['tstep']
        kwargs = lh.copy()
        kwargs['data'] = np.r_[lh['data'], rh['data']]
        kwargs['vertices'] = [lh['vertices'], rh['vertices']]
    elif ftype == 'w': # w file with surface source spaces
        lh = read_w(fname + '-lh.w')
        rh = read_w(fname + '-rh.w')
        kwargs = lh.copy()
        kwargs['data'] = np.atleast_2d(np.r_[lh['data'], rh['data']]).T
        kwargs['vertices'] = [lh['vertices'], rh['vertices']]
        # w files only have a single time point
        kwargs['tmin'] = 0.0
        kwargs['tstep'] = 1.0

    return SourceEstimate(**kwargs)


class SourceEstimate(object):
    """SourceEstimate container

    Can be saved and loaded from .stc or .w files.

    Parameters
    ----------
    data : array of shape [n_dipoles x n_times]
        The data in source space

    vertices : array | list of two arrays
        vertex number corresponding to the data

    tmin : scalar
        time point of the first sample in data

    tstep : scalar
        time step between successive samples in data

    .. note::
        For backwards compatibility, the SourceEstimate can also be
        initialized with a single argument, which can be ``None`` (an
        attribute-less SourceEstimate object will be returned) or a path as
        string, in which case the corresponding file(s) will be loaded. This
        usage is deprecated and will be removed in v0.6.

    Attributes
    ----------
    data : array of shape [n_dipoles x n_times]
        The data in source space

    times : array of shape [n_times]
        The time vector

    vertno : list of array of shape [n_dipoles in each source space]
        The indices of the dipoles in the different source spaces

    """
    def __init__(self, data, vertices=None, tmin=None, tstep=None):
        if data is None:
            warnings.warn('Constructing a SourceEstimate object with no '
                          'attributes is deprecated and will stop working in '
                          'v0.6. Use the proper constructor.')
            return
        elif isinstance(data, basestring):
            warnings.warn('Constructing a SourceEstimate object with a '
                          'filename is deprecated and will stop working in '
                          'v0.6. Use read_source_estimate().')
            se = read_source_estimate(data)
            data = se.data
            tmin = se.tmin
            tstep = se.tstep
            vertices = se.vertno

        self.data = data
        self.tmin = tmin
        self.tstep = tstep
        self.times = tmin + (tstep * np.arange(data.shape[1]))
        self.vertno = vertices

    def save(self, fname, ftype='stc'):
        """Save the source estimates to a file

        Parameters
        ----------
        fname : string
            The stem of the file name. The file names used for surface source
            spaces are obtained by adding "-lh.stc" and "-rh.stc" (or "-lh.w"
            and "-rh.w") to the stem provided, for the left and the right
            hemisphere, respectively. For volume source spaces, the stem is
            extended with "-vl.stc".
        ftype : string
            File format to use. Allowed values are "stc" (default) and "w".
            The "stc" format can be for surface and volume source spaces,
            while the "w" format only supports surface source spaces with a
            single time point.
        """
        if self.is_surface():
            lh_data = self.data[:len(self.lh_vertno)]
            rh_data = self.data[-len(self.rh_vertno):]

            if ftype == 'stc':
                print 'Writing STC to disk...',
                write_stc(fname + '-lh.stc', tmin=self.tmin, tstep=self.tstep,
                          vertices=self.lh_vertno, data=lh_data)
                write_stc(fname + '-rh.stc', tmin=self.tmin, tstep=self.tstep,
                           vertices=self.rh_vertno, data=rh_data)
            elif ftype == 'w':
                if self.data.shape[1] != 1:
                    raise ValueError('w files can only contain a single time '
                                     'point')
                print 'Writing STC to disk (w format)...',
                write_w(fname + '-lh.w', vertices=self.lh_vertno,
                        data=lh_data[:, 0])
                write_w(fname + '-rh.w', vertices=self.rh_vertno,
                        data=rh_data[:, 0])
            else:
                raise ValueError('invalid file type')
        else:
            if ftype != 'stc':
                raise ValueError('ftype has to be \"stc\" volume source '
                                 'spaces')
            print 'Writing STC to disk...',
            if not fname.endswith('-vl.stc'):
                fname += '-vl.stc'
            write_stc(fname, tmin=self.tmin, tstep=self.tstep,
                           vertices=self.vertno[0], data=self.data)
        print '[done]'

    def __repr__(self):
        s = "%d vertices" % sum([len(v) for v in self.vertno])
        s += ", tmin : %s (ms)" % (1e3 * self.tmin)
        s += ", tmax : %s (ms)" % (1e3 * self.times[-1])
        s += ", tstep : %s (ms)" % (1e3 * self.tstep)
        s += ", data size : %s x %s" % self.data.shape
        return "SourceEstimate (%s)" % s

    def crop(self, tmin=None, tmax=None):
        """Restrict SourceEstimate to a time interval

        Parameters
        ----------
        tmin : float or None
            The first time point in seconds. If None the first present is used.
        tmax : float or None
            The last time point in seconds. If None the last present is used.
        """
        mask = np.ones(len(self.times), dtype=np.bool)
        if tmin is not None:
            mask = mask & (self.times >= tmin)
        if tmax is not None:
            mask = mask & (self.times <= tmax)
        self.times = self.times[mask]
        self.data = self.data[:, mask]
        self.tmin = self.times[0]

    @property
    def lh_data(self):
        return self.data[:len(self.lh_vertno)]

    @property
    def rh_data(self):
        return self.data[len(self.lh_vertno):]

    @property
    def lh_vertno(self):
        return self.vertno[0]

    @property
    def rh_vertno(self):
        return self.vertno[1]

    def is_surface(self):
        """Returns True if source estimate is defined over surfaces
        """
        if len(self.vertno) == 1:
            return False
        else:
            return True

    def __add__(self, a):
        stc = copy.deepcopy(self)
        stc += a
        return stc

    def __iadd__(self, a):
        if isinstance(a, SourceEstimate):
            self.data += a.data
        else:
            self.data += a
        return self

    def __sub__(self, a):
        stc = copy.deepcopy(self)
        stc -= a
        return stc

    def __isub__(self, a):
        if isinstance(a, SourceEstimate):
            self.data -= a.data
        else:
            self.data -= a
        return self

    def __div__(self, a):
        stc = copy.deepcopy(self)
        stc /= a
        return stc

    def __idiv__(self, a):
        if isinstance(a, SourceEstimate):
            self.data /= a.data
        else:
            self.data /= a
        return self

    def __mul__(self, a):
        stc = copy.deepcopy(self)
        stc *= a
        return stc

    def __imul__(self, a):
        if isinstance(a, SourceEstimate):
            self.data *= a.data
        else:
            self.data *= a
        return self

    def __pow__(self, a):
        stc = copy.deepcopy(self)
        stc **= a
        return stc

    def __ipow__(self, a):
        self.data **= a
        return self

    def __radd__(self, a):
        return self + a

    def __rsub__(self, a):
        return self - a

    def __rmul__(self, a):
        return self * a

    def __rdiv__(self, a):
        return self / a

    def __neg__(self):
        stc = copy.deepcopy(self)
        stc.data *= -1
        return stc

    def __pos__(self):
        return self

    def sqrt(self):
        return self ** (0.5)

    def bin(self, width, tstart=None, tstop=None, func=np.mean):
        """Returns a SourceEstimate object with data summarized over time bins

        Time bins of ``width`` seconds. This method is intended for visualization
        only. No filter is applied to the data before binning, making the
        method inappropriate as a tool for downsampling data.

        Parameters
        ----------
        width : scalar
            Width of the individual bins in seconds.

        func : callable
            Function that is applied to summarize the data. Needs to accept a
            numpy.array as first input and an ``axis`` keyword argument.

        tstart : scalar | None
            Time point where the first bin starts. The default is the first
            time point of the stc.

        tstop : scalar | None
            Last possible time point contained in a bin (if the last bin would
            be shorter than width it is dropped). The default is the last time
            point of the stc.
        """
        if tstart is None:
            tstart = self.tmin
        if tstop is None:
            tstop = self.times[-1]

        times = np.arange(tstart, tstop + self.tstep, width)
        nv, _ = self.data.shape
        nt = len(times) - 1
        data = np.empty((nv, nt), dtype=self.data.dtype)
        for i in xrange(nt):
            idx = (self.times >= times[i]) & (self.times < times[i + 1])
            data[:, i] = func(self.data[:, idx], axis=1)

        tmin = times[0] + width / 2.
        stc = SourceEstimate(data, vertices=self.vertno, tmin=tmin, tstep=width)
        return stc

    def _hemilabel_stc(self, label):
        is_surface = self.is_surface()

        # find applicable SourceEstimate vertices
        if is_surface:
            if label.hemi == 'lh':
                stc_vertices = self.vertno[0]
            else:
                stc_vertices = self.vertno[1]
        else:
            stc_vertices = self.vertno[0]

        # find index of the Label's vertices
        idx = np.nonzero(map(label.vertices.__contains__, stc_vertices))[0]

        # find output vertices
        vertices = stc_vertices[idx]

        # find data
        if is_surface and (label.hemi == 'rh'):
            values = self.data[idx + len(self.vertno[0])]
        else:
            values = self.data[idx]

        return vertices, values

    def label_stc(self, label):
        """Returns a SourceEstimate object restricted to a label

        SourceEstimate contains the time course of
        activation of all sources inside the label.

        Parameters
        ----------
        label : Label | BiHemiLabel
            The label (as created for example by mne.read_label). If the label
            does not match any sources in the SourceEstimate, a ValueError is
            raised.
        """
        if not self.is_surface():
            raise NotImplementedError

        if label.hemi == 'both':
            lh_vert, lh_val = self._hemilabel_stc(label.lh)
            rh_vert, rh_val = self._hemilabel_stc(label.rh)
            vertices = [lh_vert, rh_vert]
            values = np.vstack((lh_val, rh_val))
        elif label.hemi == 'lh':
            lh_vert, values = self._hemilabel_stc(label)
            vertices = [lh_vert, np.array([])]
        elif label.hemi == 'rh':
            rh_vert, values = self._hemilabel_stc(label)
            vertices = [np.array([]), rh_vert]
        else:
            raise TypeError("Expected  Label or BiHemiLabel; got %r" % label)

        if sum(map(len, vertices)) == 0:
            raise ValueError('No vertices match the label in the stc file')

        label_stc = SourceEstimate(values, vertices=vertices,
                                   tmin=self.tmin, tstep=self.tstep)
        return label_stc


###############################################################################
# Morphing

from .fiff.constants import FIFF
from .fiff.tag import find_tag
from .fiff.open import fiff_open
from .fiff.tree import dir_tree_find
from .surface import read_bem_surfaces, read_surface


def read_morph_map(subject_from, subject_to, subjects_dir=None):
    """Read morph map generated with mne_make_morph_maps

    Parameters
    ----------
    subject_from : string
        Name of the original subject as named in the SUBJECTS_DIR
    subject_to : string
        Name of the subject on which to morph as named in the SUBJECTS_DIR
    subjects_dir : string
        Path to SUBJECTS_DIR is not set in the environment

    Returns
    -------
    maps : dict
        The morph maps for the 2 hemisphere
    """

    if subjects_dir is None:
        if 'SUBJECTS_DIR' in os.environ:
            subjects_dir = os.environ['SUBJECTS_DIR']
        else:
            raise ValueError('SUBJECTS_DIR environment variable not set')

    # Does the file exist
    name = '%s/morph-maps/%s-%s-morph.fif' % (subjects_dir, subject_from,
                                              subject_to)
    if not os.path.exists(name):
        name = '%s/morph-maps/%s-%s-morph.fif' % (subjects_dir, subject_to,
                                                  subject_from)
        if not os.path.exists(name):
            raise ValueError('The requested morph map does not exist')

    fid, tree, _ = fiff_open(name)

    # Locate all maps
    maps = dir_tree_find(tree, FIFF.FIFFB_MNE_MORPH_MAP)
    if len(maps) == 0:
        fid.close()
        raise ValueError('Morphing map data not found')

    # Find the correct ones
    left_map = None
    right_map = None
    for m in maps:
        tag = find_tag(fid, m, FIFF.FIFF_MNE_MORPH_MAP_FROM)
        if tag.data == subject_from:
            tag = find_tag(fid, m, FIFF.FIFF_MNE_MORPH_MAP_TO)
            if tag.data == subject_to:
                #  Names match: which hemishere is this?
                tag = find_tag(fid, m, FIFF.FIFF_MNE_HEMI)
                if tag.data == FIFF.FIFFV_MNE_SURF_LEFT_HEMI:
                    tag = find_tag(fid, m, FIFF.FIFF_MNE_MORPH_MAP)
                    left_map = tag.data
                    print '    Left-hemisphere map read.'
                elif tag.data == FIFF.FIFFV_MNE_SURF_RIGHT_HEMI:
                    tag = find_tag(fid, m, FIFF.FIFF_MNE_MORPH_MAP)
                    right_map = tag.data
                    print '    Right-hemisphere map read.'

    fid.close()
    if left_map is None:
        raise ValueError('Left hemisphere map not found in %s' % name)

    if right_map is None:
        raise ValueError('Left hemisphere map not found in %s' % name)

    return left_map, right_map


def mesh_edges(tris):
    """Returns sparse matrix with edges as an adjacency matrix

    Parameters
    ----------
    tris : array of shape [n_triangles x 3]
        The triangles

    Returns
    -------
    edges : sparse matrix
        The adjacency matrix
    """
    npoints = np.max(tris) + 1
    ntris = len(tris)
    a, b, c = tris.T
    edges = sparse.coo_matrix((np.ones(ntris), (a, b)),
                                            shape=(npoints, npoints))
    edges = edges + sparse.coo_matrix((np.ones(ntris), (b, c)),
                                            shape=(npoints, npoints))
    edges = edges + sparse.coo_matrix((np.ones(ntris), (c, a)),
                                            shape=(npoints, npoints))
    edges = edges.tocsr()
    edges = edges + edges.T
    return edges


def mesh_dist(tris, vert):
    """Compute adjacency matrix weighted by distances

    It generates an adjacency matrix where the entries are the distances
    between neighboring vertices.

    Parameters
    ----------
    tris : array (n_tris x 3)
        Mesh triangulation
    vert : array (n_vert x 3)
        Vertex locations

    Returns
    -------
    dist_matrix : scipy.sparse.csr_matrix
        Sparse matrix with distances between adjacent vertices
    """
    edges = mesh_edges(tris).tocoo()

    # Euclidean distances between neighboring vertices
    dist = np.sqrt(np.sum((vert[edges.row, :] - vert[edges.col, :]) ** 2,
                          axis=1))

    dist_matrix = csr_matrix((dist, (edges.row, edges.col)), shape=edges.shape)

    return dist_matrix


def _morph_buffer(data, idx_use, e, smooth, n_vertices, nearest, maps):
    n_iter = 100  # max nb of smoothing iterations
    for k in range(n_iter):
        e_use = e[:, idx_use]
        data1 = e_use * np.ones(len(idx_use))
        data = e_use * data
        idx_use = np.where(data1)[0]
        if smooth is None:
            if ((k == (n_iter - 1)) or (len(idx_use) >= n_vertices)):
                # stop when source space in filled with non-zeros
                break
        elif k == (smooth - 1):
            break
        data = data[idx_use, :] / data1[idx_use][:, None]

    data[idx_use, :] /= data1[idx_use][:, None]
    print '    %d smooth iterations done.' % (k + 1)
    data_morphed = maps[nearest, :] * data
    return data_morphed


def _compute_nearest(xhs, rr):
    nearest = np.zeros(len(rr), dtype=np.int)
    dr = 32
    for k in range(0, len(rr), dr):
        dots = np.dot(rr[k:k + dr], xhs.T)
        nearest[k:k + dr] = np.argmax(dots, axis=1)
    return nearest


def morph_data(subject_from, subject_to, stc_from, grade=5, smooth=None,
               subjects_dir=None, buffer_size=64, n_jobs=1, verbose=0):
    """Morph a source estimate from one subject to another

    The functions requires to set MNE_ROOT and SUBJECTS_DIR variables.

    Parameters
    ----------
    subject_from : string
        Name of the original subject as named in the SUBJECTS_DIR
    subject_to : string
        Name of the subject on which to morph as named in the SUBJECTS_DIR
    stc_from : SourceEstimate
        Source estimates for subject "from" to morph
    grade : int
        Resolution of the icosahedral mesh (typically 5)
    smooth : int or None
        Number of iterations for the smoothing of the surface data.
        If None, smooth is automatically defined to fill the surface
        with non-zero values.
    subjects_dir : string
        Path to SUBJECTS_DIR is not set in the environment
    buffer_size : int
        Morph data in chunks of `buffer_size` time instants.
        Saves memory when morphing long time intervals.
    n_jobs: int
        Number of jobs to run in parallel
    verbose: int
        Verbosity level.

    Returns
    -------
    stc_to : SourceEstimate
        Source estimate for the destination subject.
    """
    from scipy import sparse

    if not stc_from.is_surface():
        raise ValueError('Morphing is only possible with surface source '
                         'estimates')

    if subjects_dir is None:
        if 'SUBJECTS_DIR' in os.environ:
            subjects_dir = os.environ['SUBJECTS_DIR']
        else:
            raise ValueError('SUBJECTS_DIR environment variable not set')

    tris = list()
    surf_path_from = os.path.join(subjects_dir, subject_from, 'surf')
    tris.append(read_surface(os.path.join(surf_path_from, 'lh.sphere.reg'))[1])
    tris.append(read_surface(os.path.join(surf_path_from, 'rh.sphere.reg'))[1])

    sphere = os.path.join(subjects_dir, subject_to, 'surf', 'lh.sphere.reg')
    lhs = read_surface(sphere)[0]
    sphere = os.path.join(subjects_dir, subject_to, 'surf', 'rh.sphere.reg')
    rhs = read_surface(sphere)[0]

    # find which vertices to use in "to mesh"
    ico_file_name = os.path.join(os.environ['MNE_ROOT'], 'share', 'mne',
                                 'icos.fif')

    surfaces = read_bem_surfaces(ico_file_name)

    for s in surfaces:
        if s['id'] == (9000 + grade):
            ico = s
            break

    lhs /= np.sqrt(np.sum(lhs ** 2, axis=1))[:, None]
    rhs /= np.sqrt(np.sum(rhs ** 2, axis=1))[:, None]

    # Compute nearest vertices in high dim mesh
    parallel, my_compute_nearest, _ = \
                        parallel_func(_compute_nearest, n_jobs, verbose)
    lhs, rhs, rr = [a.astype(np.float32) for a in [lhs, rhs, ico['rr']]]
    nearest = parallel(my_compute_nearest(xhs, rr) for xhs in [lhs, rhs])

    # morph the data
    maps = read_morph_map(subject_from, subject_to, subjects_dir)

    lh_data = stc_from.data[:len(stc_from.lh_vertno)]
    rh_data = stc_from.data[-len(stc_from.rh_vertno):]
    data = [lh_data, rh_data]
    data_morphed = [None, None]

    n_chunks = ceil(stc_from.data.shape[1] / float(buffer_size))

    parallel, my_morph_buffer, _ = \
                        parallel_func(_morph_buffer, n_jobs, verbose)

    for hemi in [0, 1]:
        e = mesh_edges(tris[hemi])
        e.data[e.data == 2] = 1
        n_vertices = e.shape[0]
        e = e + sparse.eye(n_vertices, n_vertices)
        idx_use = stc_from.vertno[hemi]
        if len(idx_use) == 0:
            continue
        data_morphed[hemi] = np.concatenate(
                    parallel(my_morph_buffer(data_buffer, idx_use, e, smooth,
                                   n_vertices, nearest[hemi], maps[hemi])
                     for data_buffer
                     in np.array_split(data[hemi], n_chunks, axis=1)), axis=1)

    stc_to = copy.deepcopy(stc_from)
    stc_to.vertno = [nearest[0], nearest[1]]
    stc_to.data = np.r_[data_morphed[0], data_morphed[1]]

    print '[done]'

    return stc_to


def spatio_temporal_src_connectivity(src, n_times):
    """Compute connectivity for a source space activation over time

    Parameters
    ----------
    src : source space
        The source space.

    n_times : int
        Number of time instants

    Returns
    -------
    connectivity : sparse COO matrix
        The connectivity matrix describing the spatio-temporal
        graph structure. If N is the number of vertices in the
        source space, the N first nodes in the graph are the
        vertices are time 1, the nodes from 2 to 2N are the vertices
        during time 2, etc.

    """
    if src[0]['use_tris'] is None:
        raise Exception("The source space does not appear to be an ico "
                        "surface. Connectivity cannot be extracted from "
                        "non-ico source spaces.")
    lh_tris = np.searchsorted(np.unique(src[0]['use_tris']),
                              src[0]['use_tris'])
    rh_tris = np.searchsorted(np.unique(src[1]['use_tris']),
                              src[1]['use_tris'])
    tris = np.concatenate((lh_tris, rh_tris + np.max(lh_tris) + 1))
    return spatio_temporal_tris_connectivity(tris, n_times)


def spatio_temporal_tris_connectivity(tris, n_times):
    """Compute connectivity from triangles and time instants"""
    edges = mesh_edges(tris).tocoo()
    n_vertices = edges.shape[0]
    print "-- number of connected vertices : %d" % n_vertices
    nnz = edges.col.size
    aux = n_vertices * np.arange(n_times)[:, None] * np.ones((1, nnz), np.int)
    col = (edges.col[None, :] + aux).ravel()
    row = (edges.row[None, :] + aux).ravel()
    if n_times > 1:  # add temporal edges
        o = (n_vertices * np.arange(n_times - 1)[:, None]
                                  + np.arange(n_vertices)[None, :]).ravel()
        d = (n_vertices * np.arange(1, n_times)[:, None]
                                  + np.arange(n_vertices)[None, :]).ravel()
        row = np.concatenate((row, o, d))
        col = np.concatenate((col, d, o))
    data = np.ones(edges.data.size * n_times + 2 * n_vertices * (n_times - 1),
                   dtype=np.int)
    connectivity = sparse.coo_matrix((data, (row, col)),
                                     shape=(n_times * n_vertices, ) * 2)
    return connectivity


def _get_ico_tris(grade):
    """Get triangles for ico surface."""
    mne_root = os.environ.get('MNE_ROOT')
    if mne_root is None:
        raise Exception('Please set MNE_ROOT environment variable.')
    ico_file_name = os.path.join(mne_root, 'share', 'mne', 'icos.fif')
    surfaces = read_bem_surfaces(ico_file_name)
    for s in surfaces:
        if s['id'] == (9000 + grade):
            ico = s
            break
    return ico['tris']


def save_stc_as_volume(fname, stc, src, dest='mri', mri_resolution=False):
    """Save a volume source estimate in a nifti file

    Parameters
    ----------
    fname: string
        The name of the generated nifti file.
    stc: instance of SourceEstimate
        The source estimate
    src: list
        The list of source spaces (should actually be of length 1)
    dest: 'mri' | 'surf'
        If 'mri' the volume is defined in the coordinate system of
        the original T1 image. If 'surf' the coordinate system
        of the FreeSurfer surface is used (Surface RAS).
    mri_resolution: bool
        It True the image is saved in MRI resolution.
        WARNING: if you have many time points the file produced can be
        huge.

    Returns
    -------
    img : instance Nifti1Image
        The image object.
    """
    if stc.is_surface():
        raise Exception('Only volume source estimates can be saved as '
                        'volumes')

    n_times = stc.data.shape[1]
    shape = src[0]['shape']
    shape3d = (shape[2], shape[1], shape[0])
    shape = (n_times, shape[2], shape[1], shape[0])
    vol = np.zeros(shape)
    mask3d = src[0]['inuse'].reshape(shape3d).astype(np.bool)

    if mri_resolution:
        mri_shape3d = (src[0]['mri_height'], src[0]['mri_depth'],
                       src[0]['mri_width'])
        mri_shape = (n_times, src[0]['mri_height'], src[0]['mri_depth'],
                     src[0]['mri_width'])
        mri_vol = np.zeros(mri_shape)
        interpolator = src[0]['interpolator']

    for k, v in enumerate(vol):
        v[mask3d] = stc.data[:, k]
        if mri_resolution:
            mri_vol[k] = (interpolator * v.ravel()).reshape(mri_shape3d)

    if mri_resolution:
        vol = mri_vol

    vol = vol.T

    if mri_resolution:
        affine = src[0]['vox_mri_t']['trans'].copy()
    else:
        affine = src[0]['src_mri_t']['trans'].copy()
    if dest == 'mri':
        affine = np.dot(src[0]['mri_ras_t']['trans'], affine)
    affine[:3] *= 1e3

    try:
        import nibabel as nib  # lazy import to avoid dependency
    except ImportError:
        raise ImportError("nibabel is required to save volume images.")

    header = nib.nifti1.Nifti1Header()
    header.set_xyzt_units('mm', 'msec')
    header['pixdim'][4] = 1e3 * stc.tstep
    img = nib.Nifti1Image(vol, affine, header=header)
    nib.save(img, fname)
    return img
