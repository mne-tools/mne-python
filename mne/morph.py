# Author(s): Tommy Clausner <tommy.clausner@gmail.com>

# License: BSD (3-clause)


import os.path as op
import warnings

import numpy as np

from . import read_forward_solution, compute_morph_matrix
from .externals.h5io import read_hdf5, write_hdf5
from .externals.six import string_types
from .source_estimate import (VolSourceEstimate, SourceEstimate,
                              VectorSourceEstimate)
from .source_space import SourceSpaces
from .utils import logger, verbose, check_version


class SourceMorph(object):
    """Container for source estimate morphs.

    Parameters
    ----------
    src : instance of SourceSpaces
        The list of SourceSpaces corresponding subject_from
    subject_from : str | None
        Name of the original subject as named in the SUBJECTS_DIR.
        If None src[0]['subject_his_id]' will be used. The default is None.
    subject_to : str | array | list of two arrays
        Name of the subject on which to morph as named in the SUBJECTS_DIR
        The default is 'fsaverage'. If morphing a surface source extimate,
        subject_to can also be an array of vertices or a list of two arrays of
        vertices to morph to. If morphing a volume source space, subject_to can
        be the path to a MRI volume.
    subjects_dir : str | None
        Path to SUBJECTS_DIR if it is not set in the environment. The default
        is None.
    verbose : bool | str | int | None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Surface / Vector Source Estimate parameters
    -------------------------------------------
    smooth : int | None
        Number of iterations for the smoothing of the surface data.
        If None, smooth is automatically defined to fill the surface
        with non-zero values.
    warn : bool
        If True, warn if not all vertices were used.
    xhemi : bool
        Morph across hemisphere. Currently only implemented for
        ``subject_to == subject_from``. See notes below.


    Volume Source Estimate parameters
    ---------------------------------
    niter_affine : tuple of int
        Number of levels (``len(niter_affine)``) and number per level of
        iterations to refine the affine registration. Increasing index values
        for the tuple mean later levels and each int represents the number
        of iterations in that level. Default is niter_affine=(100, 100, 10)
    niter_sdr : tuple of int
        Number of levels (``len(niter_sdr)``) and number per level of
        iterations to refine the Symmetric Diffeomorphic Registration (sdr).
        Increasing index values for the tuple mean later levels and
        each int represents the number of iterations in that level. Default is
        niter_sdr=(5, 5, 3)
    grid_spacing : tuple
        Voxel size of volume for each spatial dimension in mm.
        If grid_spacing is None, MRIs won't be resliced. Note that in this case
        both volumes must have the same number of slices in every
        spatial dimension. Default is grid_spacing=(5., 5., 5.)

    Attributes
    ----------
    kind : str | None
        Kind of source estimate. E.g. 'volume' or 'surface'
    subjects_dir : str | None
        The path to the FreeSurfer subjects reconstructions.
        It corresponds to FreeSurfer environment variable SUBJECTS_DIR.
    subject_from : str | None
        Name of the subject from which to morph as named in the SUBJECTS_DIR
    subject_to : str | array | list of two arrays
        Name of the subject on which to morph as named in the SUBJECTS_DIR
        The default is 'fsaverage'. If morphing a surface source extimate,
        subject_to can also be an array of vertices or a list of two arrays of
        vertices to morph to. If morphing a volume source space, subject_to can
        be the path to a MRI volume.
    niter_affine : tuple of int
        Number of levels (``len(niter_affine)``) and number per level of
        iterations to perform the affine transform. Increasing index values
        for the tuple mean later levels and each int represents the number
        of iterations in that level.
    niter_sdr : tuple of int
        Number of levels (``len(niter_sdr)``) and number per level of
        iterations to perform the sdr transform. Increasing index values
        for the tuple mean later levels and each int represents the number
        of iterations in that level.
    grid_spacing : tuple
        Voxel size of volume for each spatial dimension in mm.
        If grid_spacing is None, MRIs won't be resliced. Note that in this case
        both volumes must have the same number of slices in every
        spatial dimension.
    smooth : int | None
        Number of iterations for the smoothing of the surface data.
        If None, smooth is automatically defined to fill the surface
        with non-zero values.
    warn : bool
        If True, warn if not all vertices were used.
    xhemi : bool
        Morph across hemisphere.

    Notes
    -----
    .. versionadded:: X.X.X

    See Also
    --------
    X

    """

    def __init__(self, src, subject_from=None, subject_to='fsaverage',
                 subjects_dir=None, niter_affine=(100, 100, 10),
                 niter_sdr=(5, 5, 3), grid_spacing=(5., 5., 5.), smooth=None,
                 warn=True, xhemi=False, verbose=False):

        # it's impossible to use the class without passing this check, so it
        # only needs to be checked here
        if not check_version('nibabel', '') or not check_version('dipy', ''):
            raise ImportError(
                'NiBabel (Python) and DiPy (Python) must be correctly '
                'installed and accessible from Python')

        # Set attributes
        self.kind = None
        self.subject_from = subject_from
        self.subject_to = subject_to
        self.subjects_dir = subjects_dir
        self.niter_affine = niter_affine
        self.niter_sdr = niter_sdr
        self.grid_spacing = grid_spacing
        self.smooth = smooth
        self.warn = warn
        self.xhemi = xhemi
        self.morph_data = None

        if src is None:
            return

        if isinstance(src, string_types):
            src = read_forward_solution(src)['src']

        if isinstance(src, SourceSpaces):
            self.kind = src.kind
        else:
            raise ValueError('src must be an instance of SourceSpaces or a '
                             'path to a saved instance of SourceSpaces')

        if src[0]['subject_his_id'] is not None:
            self.subject_from = src[0]['subject_his_id']

        if self.subject_from is None:
            raise KeyError('subject id in src is None. Please specify'
                           'subject_from')

        self.morph_data = _compute_morph_data(self, src, verbose=verbose)

    # Forward verbose decorator to _apply_morph_data
    def __call__(self, stc_from, verbose=None):
        """Morph data.

        Parameters
        ----------
        stc_from : VolSourceEstimate | SourceEstimate | VectorSourceEstimate
            The source estimate to morph.

        verbose : bool | str | int | None
            If not None, override default verbose level (see :func:`mne.
            verbose` and :ref:`Logging documentation <tut_logging>` for more).

        Returns
        -------
        stc_to : VolSourceEstimate | SourceEstimate | VectorSourceEstimate
            The morphed source estimate.
        """
        if stc_from.subject != self.subject_from:
            raise ValueError('stc_from.subject and '
                             'morph.subject_from must match')

        # if VolSourceEstimate update mri properties for '.as_volume'
        if self.kind == 'volume':
            self.morph_data.update(
                {'mri_zooms': self.morph_data['mri_zooms_to'],
                 'mri_shape': self.morph_data['mri_shape_to']})

        return _apply_morph_data(self, stc_from, verbose=verbose)

    def __repr__(self):  # noqa: D105
        if self.kind is None:
            return 'None'

        s = "%s" % self.kind
        s += ", subject_from : %s" % self.subject_from
        s += ", subject_to : %s" % self.subject_to
        if self.kind == 'volume':
            s += ", niter_affine : {}".format(self.niter_affine)
            s += ", niter_sdr : {}".format(self.niter_sdr)
            s += ", grid_spacing : {}".format(self.grid_spacing)

        elif self.kind == 'surface' or self.kind == 'vector':
            s += ", smooth : %s" % self.smooth
            s += ", xhemi : %s" % self.xhemi

        return "<SourceMorph  |  %s>" % s

    @verbose
    def save(self, fname, verbose=None):
        """Save the morph for source estimates to a file.

        Parameters
        ----------
        fname : str
            The stem of the file name. '-morph.h5' will be added if fname does
            not end with '.h5'
        verbose : bool | str | int | None
            If not None, override default verbose level (see
            :func:`mne.verbose` and
            :ref:`Logging documentation <tut_logging>`
            for more). Defaults to self.verbose.
        """
        logger.info('saving morph...')
        if not fname.endswith('.h5'):
            fname = '%s-morph.h5' % fname

        write_hdf5(fname, self.__dict__,
                   overwrite=True)
        logger.info('[done]')

    def as_volume(self, stc, fname=None, mri_resolution=False):
        """Return volume source space as Nifti1Image and/or save to disk.

        Parameters
        ----------
        stc : VolSourceEstimate | SourceEstimate | VectorSourceEstimate
            Data to be transformed
        fname : str | None
            String to where to save the volume. If not None that volume will
            be saved at fname.
        mri_resolution : bool | tuple
            Whether to use MRI resolution. If False the morph's resolution
            will be used. If tuple the voxel size must be given in float values
            in mm. E.g. mri_resolution=(3., 3., 3.)

        Returns
        -------
        img : instance of Nifti1Image
            The image object.
        """
        import nibabel as nib
        from dipy.align.reslice import reslice

        shape = tuple([int(i) for i in self.morph_data['morph_shape']])
        affine = self.morph_data['morph_affine']
        zooms = self.morph_data['morph_zooms'][:3]

        hdr = nib.nifti1.Nifti1Header()
        hdr.set_xyzt_units('mm', 'msec')
        hdr['pixdim'][4] = 1e3 * stc.tstep

        new_zooms = None

        if isinstance(mri_resolution, bool) and mri_resolution:
            new_zooms = self.morph_data['mri_zooms']

        elif isinstance(mri_resolution, tuple):
            new_zooms = mri_resolution

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


@verbose
def read_source_morph(fname, verbose=None):
    """Load the morph for source estimates from a file.

    Parameters
    ----------
    fname : str
        Full filename including path
    verbose : bool | str | int | None
        If not None, override default verbose level (see
        :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
        for more). Defaults to self.verbose.

    Returns
    -------
    source_morph : instance of SourceMorph
        The loaded morph.
    """
    try:
        logger.info('loading morph...')
        morph_data = read_hdf5(fname)
    except IOError:
        raise IOError('cannot read file: %s' % fname)

    source_morph = SourceMorph(None)
    source_morph.__dict__ = morph_data
    logger.info('[done]')
    return source_morph


def _compute_morph_data(morph, src, verbose=None):
    """Compute source estimate specific morph."""

    morph_data = []

    # VolSourceEstimate
    if morph.kind == 'volume':

        logger.info('volume source space inferred...')

        import nibabel as nib

        # load moving mri
        mri_subpath = op.join('mri', 'brain.mgz')
        mri_path_from = op.join(morph.subjects_dir, morph.subject_from,
                                mri_subpath)

        logger.info('loading %s as moving volume' % mri_path_from)
        mri_from = nib.load(mri_path_from)

        # load static mri
        static_path = op.join(morph.subjects_dir, morph.subject_to)

        if not op.isdir(static_path):
            mri_path_to = static_path
        else:
            mri_path_to = op.join(static_path, mri_subpath)
        try:
            logger.info('loading %s as static volume' % mri_path_to)
            mri_to = nib.load(mri_path_to)
        except IOError:
            raise IOError('cannot read file: %s' % mri_path_to)

        # pre-compute non-linear morph
        morph_data = _compute_morph_sdr(
            mri_from,
            mri_to,
            niter_affine=morph.niter_affine,
            niter_sdr=morph.niter_sdr,
            grid_spacing=morph.grid_spacing,
            verbose=verbose)

        # get MRI and source space meta data
        morph_data['mri_shape'] = mri_from.shape
        morph_data['mri_zooms'] = mri_from.header.get_zooms()

        morph_data['mri_shape_to'] = mri_to.shape
        morph_data['mri_zooms_to'] = mri_to.header.get_zooms()

        morph_data['src_shape_full'] = mri_from.shape
        shape = src[0]['shape']
        morph_data['src_shape'] = (shape[2], shape[1], shape[0])
        morph_data['src_affine'] = np.dot(src[0]['mri_ras_t']['trans'],
                                          src[0]['vox_mri_t']['trans'])

        morph_data['interpolator'] = src[0]['interpolator']
        morph_data['inuse'] = src[0]['inuse']

    # SourceEstimate | VectorSourceEstimate
    elif morph.kind == 'surface':
        logger.info('surface source space inferred...')

        # get surface data
        data_from = []
        for s in src:
            data_from.append(s['vertno'])

        data_to = None
        if morph.subject_to == 'fsaverage':
            data_to = [np.arange(10242)] * len(data_from)

        # pre-compute morph matrix
        morph_data = compute_morph_matrix(
            subject_from=morph.subject_from,
            subject_to=morph.subject_to,
            vertices_from=data_from,
            vertices_to=data_to,
            subjects_dir=morph.subjects_dir,
            smooth=morph.smooth,
            warn=morph.warn,
            xhemi=morph.xhemi,
            verbose=verbose)
        morph_data = {'morph_mat': morph_data, 'vertno': data_to}
    return morph_data


def _interpolate_data(stc, morph):
    """Interpolate source estimate data to MRI."""
    import nibabel as nib

    n_times = stc.data.shape[1]
    shape = morph.morph_data['src_shape']
    shape3d = shape
    shape = (n_times, shape[0], shape[1], shape[2])
    vol = np.zeros(shape)

    mri_shape3d = morph.morph_data['src_shape_full']
    mri_shape = (n_times, mri_shape3d[0], mri_shape3d[1],
                 mri_shape3d[2])
    mri_vol = np.zeros(mri_shape)
    interpolator = morph.morph_data['interpolator']

    mask3d = morph.morph_data['inuse'].reshape(shape3d).astype(np.bool)
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

    affine = morph.morph_data['src_affine']
    affine[:3] *= 1e3

    header = nib.nifti1.Nifti1Header()
    header.set_xyzt_units('mm', 'msec')
    header['pixdim'][4] = 1e3 * stc.tstep

    with warnings.catch_warnings(record=True):  # nibabel<->numpy warning
        img = nib.Nifti1Image(vol, affine, header=header)

    return img


###############################################################################
# Morph for VolSourceEstimate -
# compute AffineMap and DiffeomorphicMap from MRIs
@verbose
def _compute_morph_sdr(mri_from, mri_to,
                       niter_affine=(100, 100, 10),
                       niter_sdr=(5, 5, 3),
                       grid_spacing=(5., 5., 5.),
                       verbose=None):
    """Get a matrix that morphs data from one subject to another.

    Parameters
    ----------
    mri_from : str | Nifti1Image
        Path to source subject's anatomical MRI or Nifti1Image
    mri_to : str | Nifti1Image
        Path to destination subject's anatomical MRI or Nifti1Image
    niter_affine : tuple of int
        Number of levels (``len(niter_affine)``) and number per level of
        iterations to perform the affine transform. Increasing index values
        for the tuple mean later levels and each int represents the number
        of iterations in that level.
    niter_sdr : tuple of int
        Number of levels (``len(niter_sdr)``) and number per level of
        iterations to perform the sdr transform. Increasing index values
        for the tuple mean later levels and each int represents the number
        of iterations in that level.
    grid_spacing : tuple
        Voxel size of volume for each spatial dimension.
        If grid_spacing is None, MRIs won't be resliced. Note that in this case
        both volumes must have the same number of slices in every
        spatial dimension.
    verbose : bool | str | int | None
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
    import nibabel as nib
    from dipy.align import imaffine, imwarp, metrics, transforms
    from dipy.align.reslice import reslice

    logger.info('Computing nonlinear Symmetric Diffeomorphic Registration...')

    morph = dict()

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

    morph['morph_shape'] = tuple(mapping.domain_shape.astype('float'))
    morph['morph_zooms'] = grid_spacing
    morph['morph_affine'] = mri_to_grid2world

    morph['AffineMap'] = affine.__dict__
    morph['DiffeomorphicMap'] = mapping.__dict__

    logger.info('done.')

    return morph


@verbose
def _apply_morph_data(morph, stc_from, verbose=None):
    """Morph a source estimate from one subject to another.

    Parameters
    ----------
    stc_from : VolSourceEstimate | VectorSourceEstimate | SourceEstimate
        Data to be morphed
    verbose : bool | str | int | None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc_to : VolSourceEstimate | VectorSourceEstimate | SourceEstimate
        Source estimate for the destination subject.
    """
    stc_to = None
    if morph.kind == 'volume':

        from dipy.align.imwarp import DiffeomorphicMap
        from dipy.align.imaffine import AffineMap
        from dipy.align.reslice import reslice

        # prepare data to be morphed
        img_to = _interpolate_data(stc_from, morph)

        # setup morphs
        affine_morph = AffineMap(None)
        affine_morph.__dict__ = morph.morph_data['AffineMap']

        sdr_morph = DiffeomorphicMap(None, [])
        sdr_morph.__dict__ = morph.morph_data['DiffeomorphicMap']

        # reslice to match morph
        img_to, img_to_affine = reslice(
            img_to.get_data(),
            morph.morph_data['morph_affine'],
            morph.morph_data['mri_zooms'],
            morph.morph_data['morph_zooms'])

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
                                   tmin=stc_from.tmin,
                                   tstep=stc_from.tstep,
                                   verbose=verbose,
                                   subject=morph.subject_to)

    elif morph.kind == 'surface':

        morph_mat = morph.morph_data['morph_mat']
        vertices_to = morph.morph_data['vertno']

        if isinstance(stc_from, VectorSourceEstimate):
            # Morph the locations of the dipoles, but not their orientation
            n_verts, _, n_samples = stc_from.data.shape
            data = morph_mat * stc_from.data.reshape(n_verts,
                                                     3 * n_samples)
            data = data.reshape(morph_mat.shape[0], 3, n_samples)
            stc_to = VectorSourceEstimate(data, vertices=vertices_to,
                                          tmin=stc_from.tmin,
                                          tstep=stc_from.tstep,
                                          verbose=verbose,
                                          subject=morph.subject_to)
        else:
            data = morph_mat * stc_from.data
            stc_to = SourceEstimate(data, vertices=vertices_to,
                                    tmin=stc_from.tmin,
                                    tstep=stc_from.tstep,
                                    verbose=verbose,
                                    subject=morph.subject_to)
    return stc_to
