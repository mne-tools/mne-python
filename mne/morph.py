# Author(s): Tommy Clausner <tommy.clausner@gmail.com>

# License: BSD (3-clause)


import os.path as op
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse import block_diag as sparse_block_diag

from .externals.h5io import read_hdf5, write_hdf5
from .source_estimate import (VolSourceEstimate, SourceSpaces, SourceEstimate,
                              VectorSourceEstimate)
from .surface import read_surface, read_morph_map, mesh_edges
from .utils import logger, verbose, get_subjects_dir, warn as warn_


class SourceMorph:
    """Container for source estimate morphs.
    Parameters
    ----------
    subject_from : string
        Name of the original subject as named in the SUBJECTS_DIR
    subject_to : string
        Name of the subject on which to morph as named in the SUBJECTS_DIR
    data_from : array | list of two arrays | str | None
        If morphing a surface source estimate, data_from is the
        Vertex numbers corresponding to the data.
        If morphing a volume source estimate data_from is the absolute path
        where subject_from/mri/brain.mgz is stored
        If None data_from will be found through subject_from if present
    data_to : array | list of two arrays | str | None
        If morphing a surface source estimate , data_to is the
        Vertex numbers corresponding to the data.
        If morphing a volume source estimate data_to is the absolute path
        where subject_to/mri/brain.mgz is stored
        If None data_to will be found through subject_to if present
    src : str | SourceSpaces
        Information about the respective source space.
        If morphing a surface source estimate, src='surf' (default)
        If morphing a volume source estimate, src=src with src being the
        corresponding list of SourceSpaces
    fname : str | None
        If not None, a previously saved morpher will be loaded. fname has to be
        an absolute path.
    subjects_dir : string, or None
        Path to SUBJECTS_DIR if it is not set in the environment.
    **options : **kwargs
        Keyword arguments for the respective morphing function. For surface
        source estimates see :func:`mne.compute_morph_matrix` and for volume
        source estimates :func:`mne.compute_morph_sdr`

    Attributes
    ----------
    type : str | None
        Type of source estimate. E.g. 'vol' or 'surf'
    subjects_dir : str
        The path to the FreeSurfer subjects reconstructions.
        It corresponds to FreeSurfer environment variable SUBJECTS_DIR.
    subject_from : string
        Name of the original subject as named in the SUBJECTS_DIR
    subject_to : string
        Name of the subject on which to morph as named in the SUBJECTS_DIR
    vol_info : dict
        Information about reference volumes for volumetric morph. Additional
        information required to transform volume source estimate into
        Nifti1Image using as_volume
    src_info : dict
        Information required to transform volume source estimate into
        Nifti1Image using as_volume
    surf_info : dict
        The shape of the data. A tuple of int (n_dipoles, n_times).
    morph_data : Trans | dict
        If morphing a surface source estimate , morph_data is an instance
        of Trans to morph the data
        If morphing a volume source estimate morph_data is a dict storing
        linear and non-linear mapping information.
    options : **kwargs
        Keyword arguments for the respective morphing function. For surface
        source estimates see :func:`mne.compute_morph_matrix` and for volume
        source estimates :func:`mne.compute_morph_sdr`

    Notes
    -----
    .. versionadded:: X.X.X

    See Also
    --------

    """

    # initialize and pre-compute morph
    def __init__(self, subject_from, subject_to, data_from=None, data_to=None,
                 src='surf',
                 fname=None,
                 subjects_dir=None,
                 **options):

        # Set attributes
        self.type = src
        if isinstance(src, SourceSpaces):
            self.type = src[0]['type']

        self.subjects_dir = subjects_dir
        self.subject_from = subject_from
        self.subject_to = subject_to
        self.vol_info = dict()
        self.src_info = dict()
        self.surf_info = dict()
        self.morph_data = dict()
        self.options = options

        if fname is not None:
            self.load(fname)
            return
        # VolSourceEstimate
        if self.type == 'vol':

            try:
                import nibabel as nib
            except ImportError:
                raise ImportError(
                    "nibabel is required to laod volume images.")

            self.vol_info = dict({'mri_from': dict(), 'mri_to': dict()})

            # use subject or manually defined volume
            if data_from is not None:
                mri_from = nib.load(data_from)
            else:
                mri_from = nib.load(
                    subjects_dir + '/' + subject_from + '/mri/brain.mgz')
            if data_to is not None:
                mri_to = nib.load(data_to)
            else:
                mri_to = nib.load(
                    subjects_dir + '/' + subject_to + '/mri/brain.mgz')

            # get MRI meta data from both reference volumes
            self.vol_info['mri_from']['shape'] = mri_from.shape
            self.vol_info['mri_from']['zooms'] = mri_from.header.get_zooms()
            self.vol_info['mri_from']['affine'] = mri_from.affine

            self.vol_info['mri_to']['shape'] = mri_to.shape
            self.vol_info['mri_to']['zooms'] = mri_to.header.get_zooms()
            self.vol_info['mri_to']['affine'] = mri_to.affine

            # get source space meta data
            shape = src[0]['shape']
            self.src_info['shape'] = (shape[2], shape[1], shape[0])
            self.src_info['zooms'] = tuple(np.asanyarray(
                self.vol_info['mri_from']['shape'][:3]).astype(
                'float') / np.asanyarray(self.src_info['shape'][:3]))
            self.src_info['zooms_mri'] = self.vol_info['mri_from']['zooms']
            self.src_info['affine'] = src[0]['vox_mri_t']['trans']
            self.src_info['mri_ras_t'] = src[0]['mri_ras_t']['trans']
            self.src_info['interpolator'] = src[0]['interpolator']
            self.src_info['inuse'] = src[0]['inuse']

            # pre-compute morph
            self.morph_data = compute_morph_sdr(mri_from, mri_to,
                                                **self.options)
        # SourceEstimate | MixedSourceEstimate
        if self.type == 'surf':
            # get surface data
            self.surf_info = dict()
            self.surf_info['vertices_from'] = data_from
            self.surf_info['vertices_to'] = data_to

            # pre-compute morph
            self.morph_data = compute_morph_matrix(
                self.subject_from,
                self.subject_to,
                data_from,
                data_to,
                subjects_dir=self.subjects_dir,
                **self.options)

    def __call__(self, data_from):
        """Morph data"""
        data_to = apply_morph(data_from, self, **self.options)
        if self.type == 'vol':
            update_src_vol_info(self, self.morph_data)
        data_to.subject = self.subject_to
        return data_to

    @verbose
    def load(self, fname, verbose=None):
        """Load the morph for source estimates from a file.

        Parameters
        ----------
        fname : string
            Full filename including path
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more). Defaults to self.verbose.
        """
        logger.info('loading morph...')
        morph = read_hdf5(fname)
        self.subject_from = morph['subject_from']
        self.subject_to = morph['subject_to']
        self.subjects_dir = morph['subjects_dir']
        self.type = morph['type']
        self.src_info = morph['src_info']
        self.morph_data = morph['morph_data']
        self.options = morph['options']
        self.vol_info = morph['vol_info']
        self.src_info = morph['src_info']
        self.surf_info = morph['surf_info']
        logger.info('[done]')

    @verbose
    def save(self, fname, verbose=None):
        """Save the morph for source estimates to a file.

        Parameters
        ----------
        fname : string
            The stem of the file name. '-morph.h5' will be added
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and
            :ref:`Logging documentation <tut_logging>`
            for more). Defaults to self.verbose.
        """
        logger.info('saving morph...')
        if not fname.endswith('.h5'):
            fname += '-morph.h5'

        write_hdf5(fname, self.__dict__,
                   overwrite=True)
        logger.info('[done]')

    def as_volume(self, stc, shape=None, affine=None, zooms=None,
                  new_zooms=None, fname=None, mri_resolution=False):
        """Return volume source space as Nifti1Image and or save

        Parameters
        ----------
        stc : VolSourceEstimate | SourceEstimate | VectorSourceEstimate
            Data to be transformed
        shape : tuple | None
            Shape of the volume the data projects to
        affine : ndarray | None
            MRI to world transformation
        zooms : tuple | None
            Current voxel size
        new_zooms : tuple | None
            Output volume voxel size.
        fname : str | None
            String to where to save the volume. If not None that volume will
            be saved at fname.
        mri_resolution : bool
            Whether to use MRI resolution

        Returns
        -------
        img : instance of Nifti1Image
            The image object.
        """
        return stc_as_volume(stc, self, shape=shape, affine=affine,
                             zooms=zooms, new_zooms=new_zooms, fname=fname,
                             mri_resolution=mri_resolution)


###############################################################################
# Make Nifti1Image
def update_src_vol_info(self, vol_info):
    """Update information needed for as_volume"""
    self.src_info['shape'] = vol_info['shape']
    self.src_info['affine'] = vol_info['affine']
    self.src_info['zooms'] = vol_info['zooms']
    self.src_info['zooms_mri'] = vol_info['zooms_mri']


def _interpolate_data(stc, morpher):
    n_times = stc.data.shape[1]
    shape = morpher.src_info['shape']
    shape3d = shape
    shape = (n_times, shape[0], shape[1], shape[2])
    vol = np.zeros(shape)

    mri_shape3d = morpher.vol_info['mri_from']['shape']
    mri_shape = (n_times, mri_shape3d[0], mri_shape3d[1],
                 mri_shape3d[2])
    mri_vol = np.zeros(mri_shape)
    interpolator = morpher.src_info['interpolator']

    mask3d = morpher.src_info['inuse'].reshape(shape3d).astype(np.bool)
    n_vertices = np.sum(mask3d)

    n_vertices_seen = 0
    for k, v in enumerate(vol):  # loop over time instants
        stc_slice = slice(n_vertices_seen, n_vertices_seen + n_vertices)
        v[mask3d] = stc.data[stc_slice, k]

    n_vertices_seen += n_vertices

    for k, v in enumerate(vol):
        mri_vol[k] = (interpolator * v.ravel()).reshape(mri_shape3d)
    vol = mri_vol

    vol = vol.T

    affine = morpher.src_info['affine']
    affine = np.dot(morpher.src_info['mri_ras_t'], affine)
    affine[:3] *= 1e3

    try:
        import nibabel as nib  # lazy import to avoid dependency
    except ImportError:
        raise ImportError("nibabel is required to save volume images.")

    header = nib.nifti1.Nifti1Header()
    header.set_xyzt_units('mm', 'msec')
    header['pixdim'][4] = 1e3 * stc.tstep

    with warnings.catch_warnings(record=True):  # nibabel<->numpy warning
        img = nib.Nifti1Image(vol, affine, header=header)

    return img


def stc_as_volume(stc, morph, shape=None, affine=None, zooms=None,
                  new_zooms=None, hdr=None, fname=None, mri_resolution=False):
    """Return volume source space as Nifti1Image and or save

    Parameters
    ----------
    stc : VolSourceEstimate | SourceEstimate | VectorSourceEstimate
        Data to be transformed
    shape : tuple | None
        Shape of the volume the data projects to
    affine : ndarray | None
        MRI to world transformation
    zooms : tuple | None
        Current voxel size
    new_zooms : tuple | None
        Output volume voxel size.
    fname : str | None
        String to where to save the volume. If not None that volume will
        be saved at fname.
    mri_resolution : bool
        Whether to use MRI resolution

    Returns
    -------
    img : instance of Nifti1Image
        The image object.
    """
    try:
        from dipy.align.reslice import reslice
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "dipy and nibabel is required to save volume images.")

    if shape is None:
        shape = tuple([int(i) for i in morph.src_info['shape']])
    if affine is None:
        affine = morph.src_info['affine']
    if zooms is None:
        zooms = morph.src_info['zooms'][:3]
    if hdr is None:
        hdr = nib.nifti1.Nifti1Header()
        hdr.set_xyzt_units('mm', 'msec')
        hdr['pixdim'][4] = 1e3 * stc.tstep
    if mri_resolution:
        new_zooms = morph.src_info['zooms_mri']

    img = np.zeros(shape).reshape(-1, stc.shape[1])
    img[stc.vertices, :] = stc.data

    img = img.reshape(shape + (-1,))

    img = nib.Nifti1Image(img, affine, header=hdr)

    if new_zooms is not None:
        new_zooms = new_zooms[:3]
        img, affine = reslice(img.get_data(),
                              img.affine,
                              zooms,
                              new_zooms)
        img = nib.Nifti1Image(img, affine)
        zooms = new_zooms
    img.header.set_zooms(zooms + (1,))
    if fname is not None:
        nib.save(img, fname)
    return img


###############################################################################
# Morph for VolSourceEstimate -
# compute AffineMap and DiffeomorphicMap from MRIs
@verbose
def compute_morph_sdr(mri_from, mri_to,
                      niter_affine=(100, 100, 10),
                      niter_sdr=(5, 5, 3),
                      grid_spacing=None,
                      verbose=None, **kwargs):
    """Get a matrix that morphs data from one subject to another.

    Parameters
    ----------
    mri_from : string | Nifti1Image
        Path to source subject's anatomical MRI or Nifti1Image
    mri_to : string | Nifti1Image
        Path to destination subject's anatomical MRI or Nifti1Image
    niter_affine : tuple of int
        Number of levels (``niter_affine.__len__()``) and number per level of
        iterations to perform the affine transform. Increasing index values
        for the tuple mean later levels and each int represents the number
        of iterations in that level.
    niter_sdr : tuple of int
        Number of levels (``niter_sdr.__len__()``) and number per level of
        iterations to perform the sdr transform. Increasing index values
        for the tuple mean later levels and each int represents the number
        of iterations in that level.
    grid_spacing : tuple
        Voxel size of volume for each spatial dimension.
        If grid_spacing is None, MRIs won't be resliced. Note that in this case
        both volumes must have the same number of slices in every
        spatial dimension.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    morph : list of AffineMap and DiffeomorphicMap
        Affine and Diffeomorphic registration

    Notes
    -----
    This function will be used to morph VolSourceEstimate based on an
    affine transformation and a nonlinear morph, estimated based on
    respective transformation from the subject's anatomical T1 (brain) to
    a destination subject's anatomical T1 (e.g. fsaverage). Afterwards the
    transformation can be applied. Affine transformations are computed
    based on the mutual information. This metric relates structural changes
    in image intensity values. Because different still brains expose high
    structural similarities this method works quite well to relate
    corresponding features [1]_.
    The nonlinear transformations will be performed as Symmetric
    Diffeomorphic Registration using the cross-correlation metric [2]_.

    References
    ----------
    .. [1] Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K., &
    Eubank, W. (2003). PET-CT image registration in the chest using
    free-form deformations. IEEE transactions on medical imaging, 22(1),
    120-128.

    .. [2] Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2009).
    Symmetric Diffeomorphic Image Registration with Cross- Correlation:
    Evaluating Automated Labeling of Elderly and Neurodegenerative Brain,
    12(1), 26-41. Asymmetry. Journal of Cognitive Neuroscience 25(9),
    1477-1492, 2013.
        """

    try:
        from dipy.align import imaffine, imwarp, metrics, transforms
        from dipy.align.reslice import reslice
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "dipy and nibabel is required compute nonlinear "
            "Symmetric Diffeomorphic Image Registration.")

    logger.info('Computing nonlinear Symmetric Diffeomorphic Registration...')

    morph = dict()
    morph['zooms_mri'] = mri_to.header.get_zooms()

    # use voxel size of mri_from
    if grid_spacing is None:
        grid_spacing = mri_from.header.get_zooms()[:3]

    # reslice mri_from
    mri_from_res, mri_from_res_affine = reslice(
        mri_from.get_data(),
        mri_from.affine,
        mri_from.header.get_zooms()[:3],
        grid_spacing)

    mri_from = nib.Nifti1Image(mri_from_res, mri_from_res_affine)

    # reslice mri_to
    mri_to_res, mri_to_res_affine = reslice(
        mri_to.get_data(),
        mri_to.affine,
        mri_to.header.get_zooms()[:3],
        grid_spacing)

    mri_to = nib.Nifti1Image(mri_to_res, mri_to_res_affine)

    # get mri_to to world transform
    mri_to_grid2world = mri_to.affine

    # output mri_to as ndarray
    mri_to = mri_to.dataobj[:, :, :]

    # normalize values
    mri_to = mri_to.astype('float') / mri_to.max()

    # get mri_from to world transform
    mri_from_grid2world = mri_from.affine

    # output mri_from as ndarray
    mri_from = mri_from.dataobj[:, :, :]

    # normalize values
    mri_from = mri_from.astype('float') / mri_from.max()

    # compute center of mass
    c_of_mass = imaffine.transform_centers_of_mass(mri_to, mri_to_grid2world,
                                                   mri_from,
                                                   mri_from_grid2world)

    nbins = 32

    # set up Affine Registration
    affreg = imaffine.AffineRegistration(
        metric=imaffine.MutualInformationMetric(nbins, None),
        level_iters=list(niter_affine),
        sigmas=[3.0, 1.0, 0.0],
        factors=[4, 2, 1])

    # translation
    translation = affreg.optimize(mri_to, mri_from,
                                  transforms.TranslationTransform3D(), None,
                                  mri_to_grid2world, mri_from_grid2world,
                                  starting_affine=c_of_mass.affine)

    # rigid body transform (translation + rotation)
    rigid = affreg.optimize(mri_to, mri_from,
                            transforms.RigidTransform3D(), None,
                            mri_to_grid2world, mri_from_grid2world,
                            starting_affine=translation.affine)

    # affine transform (translation + rotation + scaling)
    affine = affreg.optimize(mri_to, mri_from,
                             transforms.AffineTransform3D(), None,
                             mri_to_grid2world, mri_from_grid2world,
                             starting_affine=rigid.affine)

    # apply affine transformation
    mri_from_affine = affine.transform(mri_from)

    # set up Symmetric Diffeomorphic Registration (metric, iterations)
    sdr = imwarp.SymmetricDiffeomorphicRegistration(
        metrics.CCMetric(3), list(niter_sdr))

    # compute mapping
    mapping = sdr.optimize(mri_to, mri_from_affine)

    morph['shape'] = tuple(mapping.domain_shape.astype('float'))
    morph['zooms'] = grid_spacing
    morph['affine'] = mri_to_grid2world
    morph['AffineMap'] = affine.__dict__
    morph['DiffeomorphicMap'] = mapping.__dict__

    logger.info('done.')

    return morph


###############################################################################
# Morph for SourceEstimate | VectorSourceEstimate

@verbose
def _morph_buffer(data, idx_use, e, smooth, n_vertices, nearest, maps,
                  warn=True, verbose=None):
    """Morph data from one subject's source space to another.

    Parameters
    ----------
    data : array, or csr sparse matrix
        A n_vertices [x 3] x n_times (or other dimension) dataset to morph.
    idx_use : array of int
        Vertices from the original subject's data.
    e : sparse matrix
        The mesh edges of the "from" subject.
    smooth : int
        Number of smoothing iterations to perform. A hard limit of 100 is
        also imposed.
    n_vertices : int
        Number of vertices.
    nearest : array of int
        Vertices on the destination surface to use.
    maps : sparse matrix
        Morph map from one subject to the other.
    warn : bool
        If True, warn if not all vertices were used.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    data_morphed : array, or csr sparse matrix
        The morphed data (same type as input).
    """
    # When operating on vector data, morph each dimension separately
    if data.ndim == 3:
        data_morphed = np.zeros((len(nearest), 3, data.shape[2]),
                                dtype=data.dtype)
        for dim in range(3):
            data_morphed[:, dim, :] = _morph_buffer(
                data=data[:, dim, :], idx_use=idx_use, e=e, smooth=smooth,
                n_vertices=n_vertices, nearest=nearest, maps=maps, warn=warn,
                verbose=verbose
            )
        return data_morphed

    n_iter = 99  # max nb of smoothing iterations (minus one)
    if smooth is not None:
        if smooth <= 0:
            raise ValueError('The number of smoothing operations ("smooth") '
                             'has to be at least 1.')
        smooth -= 1
    # make sure we're in CSR format
    e = e.tocsr()
    if sparse.issparse(data):
        use_sparse = True
        if not isinstance(data, sparse.csr_matrix):
            data = data.tocsr()
    else:
        use_sparse = False

    done = False
    # do the smoothing
    for k in range(n_iter + 1):
        # get the row sum
        mult = np.zeros(e.shape[1])
        mult[idx_use] = 1
        idx_use_data = idx_use
        data_sum = e * mult

        # new indices are non-zero sums
        idx_use = np.where(data_sum)[0]

        # typically want to make the next iteration have these indices
        idx_out = idx_use

        # figure out if this is the last iteration
        if smooth is None:
            if k == n_iter or len(idx_use) >= n_vertices:
                # stop when vertices filled
                idx_out = None
                done = True
        elif k == smooth:
            idx_out = None
            done = True

        # do standard smoothing multiplication
        data = _morph_mult(data, e, use_sparse, idx_use_data, idx_out)

        if done is True:
            break

        # do standard normalization
        if use_sparse:
            data.data /= data_sum[idx_use].repeat(np.diff(data.indptr))
        else:
            data /= data_sum[idx_use][:, None]

    # do special normalization for last iteration
    if use_sparse:
        data_sum[data_sum == 0] = 1
        data.data /= data_sum.repeat(np.diff(data.indptr))
    else:
        data[idx_use, :] /= data_sum[idx_use][:, None]
    if len(idx_use) != len(data_sum) and warn:
        warn_('%s/%s vertices not included in smoothing, consider increasing '
              'the number of steps'
              % (len(data_sum) - len(idx_use), len(data_sum)))

    logger.info('    %d smooth iterations done.' % (k + 1))

    data_morphed = maps[nearest, :] * data
    return data_morphed


def _morph_mult(data, e, use_sparse, idx_use_data, idx_use_out=None):
    """Help morphing.

    Equivalent to "data = (e[:, idx_use_data] * data)[idx_use_out]"
    but faster.
    """
    if len(idx_use_data) < e.shape[1]:
        if use_sparse:
            data = e[:, idx_use_data] * data
        else:
            # constructing a new sparse matrix is faster than sub-indexing
            # e[:, idx_use_data]!
            col, row = np.meshgrid(np.arange(data.shape[1]), idx_use_data)
            d_sparse = sparse.csr_matrix((data.ravel(),
                                          (row.ravel(), col.ravel())),
                                         shape=(e.shape[1], data.shape[1]))
            data = e * d_sparse
            data = np.asarray(data.todense())
    else:
        data = e * data

    # trim data
    if idx_use_out is not None:
        data = data[idx_use_out]
    return data


def _get_subject_sphere_tris(subject, subjects_dir):
    spheres = [op.join(subjects_dir, subject, 'surf',
                       xh + '.sphere.reg') for xh in ['lh', 'rh']]
    tris = [read_surface(s)[1] for s in spheres]
    return tris


@verbose
def compute_morph_matrix(subject_from, subject_to, vertices_from, vertices_to,
                         smooth=None, subjects_dir=None, warn=True,
                         xhemi=False, verbose=None):
    """Get a matrix that morphs data from one subject to another.

    Parameters
    ----------
    subject_from : string
        Name of the original subject as named in the SUBJECTS_DIR.
    subject_to : string
        Name of the subject on which to morph as named in the SUBJECTS_DIR.
    vertices_from : list of arrays of int
        Vertices for each hemisphere (LH, RH) for subject_from.
    vertices_to : list of arrays of int
        Vertices for each hemisphere (LH, RH) for subject_to.
    smooth : int or None
        Number of iterations for the smoothing of the surface data.
        If None, smooth is automatically defined to fill the surface
        with non-zero values.
    subjects_dir : string
        Path to SUBJECTS_DIR is not set in the environment.
    warn : bool
        If True, warn if not all vertices were used.
    xhemi : bool
        Morph across hemisphere. Currently only implemented for
        ``subject_to == subject_from``. See notes below.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    morph_matrix : sparse matrix
        matrix that morphs data from ``subject_from`` to ``subject_to``.

    Notes
    -----
    This function can be used to morph data between hemispheres by setting
    ``xhemi=True``. The full cross-hemisphere morph matrix maps left to right
    and right to left. A matrix for cross-mapping only one hemisphere can be
    constructed by specifying the appropriate vertices, for example, to map the
    right hemisphere to the left:
    ``vertices_from=[[], vert_rh], vertices_to=[vert_lh, []]``.

    Cross-hemisphere mapping requires appropriate ``sphere.left_right``
    morph-maps in the subject's directory. These morph maps are included
    with the ``fsaverage_sym`` FreeSurfer subject, and can be created for other
    subjects with the ``mris_left_right_register`` FreeSurfer command. The
    ``fsaverage_sym`` subject is included with FreeSurfer > 5.1 and can be
    obtained as described `here
    <http://surfer.nmr.mgh.harvard.edu/fswiki/Xhemi>`_. For statistical
    comparisons between hemispheres, use of the symmetric ``fsaverage_sym``
    model is recommended to minimize bias [1]_.

    References
    ----------
    .. [1] Greve D. N., Van der Haegen L., Cai Q., Stufflebeam S., Sabuncu M.
           R., Fischl B., Brysbaert M.
           A Surface-based Analysis of Language Lateralization and Cortical
           Asymmetry. Journal of Cognitive Neuroscience 25(9), 1477-1492, 2013.
    """

    logger.info('Computing morph matrix...')
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)

    tris = _get_subject_sphere_tris(subject_from, subjects_dir)
    maps = read_morph_map(subject_from, subject_to, subjects_dir, xhemi)

    if xhemi:
        hemi_indexes = [(0, 1), (1, 0)]
    else:
        hemi_indexes = [(0, 0), (1, 1)]

    morpher = []
    for hemi_from, hemi_to in hemi_indexes:
        idx_use = vertices_from[hemi_from]
        if len(idx_use) == 0:
            continue
        e = mesh_edges(tris[hemi_from])
        e.data[e.data == 2] = 1
        n_vertices = e.shape[0]
        e = e + sparse.eye(n_vertices, n_vertices)
        m = sparse.eye(len(idx_use), len(idx_use), format='csr')
        mm = _morph_buffer(m, idx_use, e, smooth, n_vertices,
                           vertices_to[hemi_to], maps[hemi_from], warn=warn)
        morpher.append(mm)

    if len(morpher) == 0:
        raise ValueError("Empty morph-matrix")
    elif len(morpher) == 1:
        morpher = morpher[0]
    else:
        morpher = sparse_block_diag(morpher, format='csr')
    logger.info('[done]')
    return morpher


###############################################################################
# Apply pre-computed morph
def apply_morph(data_from, morpher, subject_from=None, subject_to=None,
                **kwargs):
    """Morph a source estimate from one subject to another.

    Parameters
    ----------
    data_from : VolSourceEstimate | VectorSourceEstimate | SourceEstimate
        Data to be morphed
    morpher : SourceMorph
        SourceMorph obtained from SourceMorph(subject_from, subject_to)
    subject_from : string
        Name of the subject on which to morph as named in the SUBJECTS_DIR
    **kwargs : **kwargs
        Keyword arguments for the respective type of morph
        (e.g. 'vol' or 'surf')

    Returns
    -------
    stc_to : VolSourceEstimate | VectorSourceEstimate | SourceEstimate
        Source estimate for the destination subject.
    """
    if subject_from is None:
        subject_from = morpher.subject_from
    if subject_to is None:
        subject_to = morpher.subject_to

    if morpher.type == 'vol':

        try:
            from dipy.align.imwarp import DiffeomorphicMap
            from dipy.align.imaffine import AffineMap
            from dipy.align.reslice import reslice
        except ImportError:
            raise ImportError(
                "dipy is required to morph volume data.")

        # prepare data to be morphed
        img_to = _interpolate_data(data_from, morpher)

        # setup morphers
        affine_morph = AffineMap(None)
        affine_morph.__dict__ = morpher.morph_data['AffineMap']

        sdr_morph = DiffeomorphicMap(None, [])
        sdr_morph.__dict__ = morpher.morph_data['DiffeomorphicMap']

        # reslice to match morph
        img_to, img_to_affine = reslice(
            img_to.get_data(),
            morpher.morph_data['affine'],
            morpher.vol_info['mri_from']['zooms'],
            morpher.morph_data['zooms'])

        # morph data
        for vol in range(img_to.shape[3]):
            img_to[:, :, :, vol] = sdr_morph.transform(
                affine_morph.transform(img_to[:, :, :, vol]))

        # reshape to nvoxel x nvol
        img_to = img_to.reshape(-1, img_to.shape[3])

        vertices = [i for i, d in enumerate(img_to.sum(axis=1) == 0) if
                    not d]

        # create new source estimate
        stc_to = VolSourceEstimate(img_to[vertices, :],
                                   vertices=np.asanyarray(vertices),
                                   tmin=data_from.tmin,
                                   tstep=data_from.tstep,
                                   verbose=data_from.verbose,
                                   subject=subject_to)

    elif morpher.type == 'surf':

        morph_mat = morpher.morph_data
        vertices_to = morpher.surf_info['vertices_to']

        if not sparse.issparse(morph_mat):
            raise ValueError('morph_mat must be a sparse matrix')

        if not isinstance(vertices_to, list) or not len(vertices_to) == 2:
            raise ValueError('vertices_to must be a list of length 2')

        if not sum(len(v) for v in vertices_to) == morph_mat.shape[0]:
            raise ValueError('number of vertices in vertices_to must match '
                             'morph_mat.shape[0]')

        if not data_from.data.shape[0] == morph_mat.shape[1]:
            raise ValueError('data_from.data.shape[0] must be the same as '
                             'morph_mat.shape[0]')

        if data_from.subject is not None and data_from.subject != subject_from:
            raise ValueError('data_from.subject and subject_from must match')

        if isinstance(data_from, VectorSourceEstimate):
            # Morph the locations of the dipoles, but not their orientation
            n_verts, _, n_samples = data_from.data.shape
            data = morph_mat * data_from.data.reshape(n_verts, 3 * n_samples)
            data = data.reshape(morph_mat.shape[0], 3, n_samples)
            stc_to = VectorSourceEstimate(data, vertices=vertices_to,
                                          tmin=data_from.tmin,
                                          tstep=data_from.tstep,
                                          verbose=data_from.verbose,
                                          subject=subject_to)
        else:
            data = morph_mat * data_from.data
            stc_to = SourceEstimate(data, vertices=vertices_to,
                                    tmin=data_from.tmin,
                                    tstep=data_from.tstep,
                                    verbose=data_from.verbose,
                                    subject=subject_to)
    return stc_to
