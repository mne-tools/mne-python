# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import os
import copy
import numpy as np
from scipy import sparse


def read_stc(filename):
    """Read an STC file

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
    vertices_n = int(np.fromfile(fid, dtype=">I4", count=1))

    # read the source vector
    stc['vertices'] = np.fromfile(fid, dtype=">I4", count=vertices_n)

    # read the number of timepts
    data_n = int(np.fromfile(fid, dtype=">I4", count=1))

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
    fid.write(np.array(vertices.shape[0], dtype='>I4').tostring())
    # write the vertex indices
    fid.write(np.array(vertices, dtype='>I4').tostring())

    # write the number of timepts
    fid.write(np.array(data.shape[1], dtype='>I4').tostring())
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
    """Read a w file

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


class SourceEstimate(object):
    """SourceEstimate container

    Can be saved and loaded from .stc or .w files.

    Attributes
    ----------
    data : array of shape [n_dipoles x n_times]
        The data in source space
    times : array of shape [n_times]
        The time vector
    vertno : list of array of shape [n_dipoles in each source space]
        The indices of the dipoles in the different source spaces
    """
    def __init__(self, fname):
        if fname is not None:
            if fname.endswith('-vl.stc'):  # volumne source space
                vl = read_stc(fname)
                self.data = vl['data']
                self.tmin = vl['tmin']
                self.tstep = vl['tstep']
                self.times = self.tmin + (self.tstep *
                                          np.arange(self.data.shape[1]))
                self.vertno = [vl['vertices']]
            elif (fname.endswith('.stc') or os.path.exists(fname + '-lh.stc')
                  or os.path.exists(fname + '-rh.stc')):
                # stc file with surface source spaces

                if fname.endswith('-lh.stc') or fname.endswith('-rh.stc'):
                    fname = fname[:-7]
                lh = read_stc(fname + '-lh.stc')
                rh = read_stc(fname + '-rh.stc')
                self.data = np.r_[lh['data'], rh['data']]
                assert lh['tmin'] == rh['tmin']
                assert lh['tstep'] == rh['tstep']
                self.tmin = lh['tmin']
                self.tstep = lh['tstep']
                self.times = self.tmin + (self.tstep *
                                          np.arange(self.data.shape[1]))
                self.vertno = [lh['vertices'], rh['vertices']]
            elif (fname.endswith('.w') or os.path.exists(fname + '-lh.w')
                  or os.path.exists(fname + '-rh.w')):
                # w file with surface source spaces

                if fname.endswith('-lh.w') or fname.endswith('-rh.w'):
                    fname = fname[:-5]
                lh = read_w(fname + '-lh.w')
                rh = read_w(fname + '-rh.w')
                self.data = np.atleast_2d(np.r_[lh['data'], rh['data']]).T

                # w files only have a single time point
                self.tmin = 0.0
                self.tstep = 1.0
                self.times = np.array([0.0])
                self.vertno = [lh['vertices'], rh['vertices']]
            else:
                raise ValueError('file type not supported')

    def _init_times(self):
        """create self.times"""
        self.times = self.tmin + self.tstep * np.arange(self.data.shape[1])

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
            The first time point in seconds. It None the first present is used.
        tmax : float or None
            The last time point in seconds. It None the last present is used.
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


def morph_data(subject_from, subject_to, stc_from, grade=5, smooth=None,
               subjects_dir=None):
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

    nearest = np.zeros((2, ico['np']), dtype=np.int)

    lhs /= np.sqrt(np.sum(lhs ** 2, axis=1))[:, None]
    rhs /= np.sqrt(np.sum(rhs ** 2, axis=1))[:, None]

    rr = ico['rr']
    dr = 16
    for k in range(0, len(rr), dr):
        dots = np.dot(rr[k:k + dr], lhs.T)
        nearest[0, k:k + dr] = np.argmax(dots, axis=1)
        dots = np.dot(rr[k:k + dr], rhs.T)
        nearest[1, k:k + dr] = np.argmax(dots, axis=1)

    # morph the data
    maps = read_morph_map(subject_from, subject_to, subjects_dir)

    lh_data = stc_from.data[:len(stc_from.lh_vertno)]
    rh_data = stc_from.data[-len(stc_from.rh_vertno):]
    data = [lh_data, rh_data]
    data_morphed = [None, None]

    for hemi in [0, 1]:
        e = mesh_edges(tris[hemi])
        e.data[e.data == 2] = 1
        n_vertices = e.shape[0]
        e = e + sparse.eye(n_vertices, n_vertices)
        idx_use = stc_from.vertno[hemi]
        if len(idx_use) == 0:
            continue
        n_iter = 100  # max nb of smoothing iterations
        for k in range(n_iter):
            e_use = e[:, idx_use]
            data1 = e_use * np.ones(len(idx_use))
            data[hemi] = e_use * data[hemi]
            idx_use = np.where(data1)[0]
            if smooth is None:
                if ((k == (n_iter - 1)) or (len(idx_use) >= n_vertices)):
                    # stop when source space in filled with non-zeros
                    break
            elif k == (smooth - 1):
                break
            data[hemi] = data[hemi][idx_use, :] / data1[idx_use][:, None]

        data[hemi][idx_use, :] /= data1[idx_use][:, None]

        print '    %d smooth iterations done.' % (k + 1)
        data_morphed[hemi] = maps[hemi][nearest[hemi], :] * data[hemi]

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
