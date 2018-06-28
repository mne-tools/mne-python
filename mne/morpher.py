from copy import deepcopy
from gzip import GzipFile

import numpy as np
from mne import (VolSourceEstimate, SourceEstimate, SourceSpaces,
                 VectorSourceEstimate, setup_volume_source_space)
from mne.externals.six import string_types
from mne.transforms import (invert_transform, apply_trans, _print_coord_trans,
                            combine_transforms, Transform)
from mne.utils import verbose, _check_subject, logger
from scipy import sparse, linalg


class Morpher:
    """class for morphing operations


        Parameters
        ----------


        Attributes
        ----------

        """

    def __init__(self, src, data_from, data_to,
                 subject_to=None, fname=None, **options):

        self.type = src

        if self.type is not None:
            self.type = src[0]['type']
        self.subject_from = None
        self.subject_to = subject_to
        self.morpher = None

        if self.type == 'vol':
            if data_from is not None and data_to is not None:
                self.morpher = dict()
                self.morpher['mri_to'] = data_to
                shape = copy(src[0]['shape'])
                shape = (shape[2], shape[1], shape[0])
                self.morpher['shape'] = shape

                mri_shape = (
                    src[0]['mri_height'], src[0]['mri_depth'],
                    src[0]['mri_width'])
                self.morpher['stc_voxel_size'] = tuple(np.asanyarray(
                    mri_shape) / np.asanyarray(shape).astype('float'))
                self.morpher.update(
                    compute_morph_sdr(data_from, data_to, **options).items())

        elif self.type == 'surf':
            self.morpher = compute_morph_matrix(
                self.subject_from, self.subject_to, data_from, data_to,
                **options)

        if fname is not None:
            _load_morpher(self, fname)

    def __call__(self, data_from, voxel_size=None):

        if self.type == 'vol':

            data_to, affine = morph_data_precomputed(data_from,
                                                     self,
                                                     voxel_size=voxel_size)

            return data_to
        else:
            pass

    def save(self, fname):
        pass

    def as_volume(self, stc, voxel_size=None):
        return _stc_as_volume(stc, self, voxel_size=voxel_size)


def copy(input):
    """Make a copy
    """
    copied_input = deepcopy(input)
    return copied_input


def _save_morpher(morpher, fname):
    # needs to be discussed
    pass


def _load_morpher(morpher, fname):
    # needs to be discussed
    return morpher


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


@verbose
def compute_morph_sdr(mri_from, mri_to,
                      niter_affine=(100, 100, 10),
                      niter_sdr=(5, 5, 3),
                      grid_spacing=None,
                      verbose=None):
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
        Voxel size of volume for each spatial dimension. If grid_spacing is None,
        MRIs won't be resliced. Note that in this case both volumes must have
        the same number of slices in every spatial dimension.
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
    transformation will be applied to. Affine transformations are computed
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
    from dipy.align import imaffine, imwarp, metrics, reslice, transforms
    import nibabel as nib
    logger.info('Computing nonlinear Symmetric Diffeomorphic Registration...')

    if isinstance(mri_from, string_types):
        mri_from = nib.load(mri_from)

    if isinstance(mri_to, string_types):
        mri_to = nib.load(mri_to)

    morph = dict()
    morph['mri_voxel_size'] = mri_from.header.get_zooms()[:3]
    morph['mri_affine'] = mri_from.affine

    if grid_spacing is None:
        grid_spacing = mri_from.header.get_zooms()[:3]

    # reslice Moving
    mri_from_res, mri_from_res_affine = reslice.reslice(
        mri_from.get_data(),
        mri_from.affine,
        mri_from.header.get_zooms()[:3],
        grid_spacing)

    mri_from = nib.Nifti1Image(mri_from_res, mri_from_res_affine)

    # reslice Static
    mri_to_res, mri_to_res_affine = reslice.reslice(
        mri_to.get_data(),
        mri_to.affine,
        mri_to.header.get_zooms()[:3],
        grid_spacing)

    mri_to = nib.Nifti1Image(mri_to_res, mri_to_res_affine)

    # get Static to world transform
    mri_to_grid2world = mri_to.affine

    # output Static as ndarray
    mri_to = mri_to.dataobj[:, :, :]

    # normalize values
    mri_to = mri_to.astype('float') / mri_to.max()

    # get Moving to world transform
    mri_from_grid2world = mri_from.affine

    # output Moving as ndarray
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

    morph['affine'] = affine
    morph['mapping'] = mapping
    morph['affine_reg'] = mri_to_grid2world
    morph['grid_spacing'] = grid_spacing

    logger.info('done.')

    return morph


def morph_data_precomputed(data_from,
                           morph, voxel_size=None, as_volume=True):
    """Morph source estimate between subjects using a precomputed matrix.

    Parameters
    ----------
    subject_from : string
        Name of the original subject as named in the SUBJECTS_DIR.
    subject_to : string
        Name of the subject on which to morph as named in the SUBJECTS_DIR.
    data_from : SourceEstimate | VectorSourceEstimate | VolSourceEstimate
        Source estimates for subject "from" to morph.
    space_to : list of array of int or src
        If "data_from" is SourceEstimate | VectorSourceEstimate "space_to"
        contains the vertices on the destination subject's brain.
        If "data_from" is a VolSourceEstimate, "space_to" is an instance of src
        e.g. forward['src_from'] to define the target volumetric space, but in
        this case it's the source target's brain.
    morph : sparse matrix or dict
        If "data_from" is SourceEstimate | VectorSourceEstimate "morph" is the
        morphing matrix, typically from compute_morph_matrix. If "data_from"
        is a VolSourceEstimate, morph is a dict containing an AffineMap
        (``morph['affine']``), a DiffeomorphicMap (``morph['mapping']``),
        4 x 4 ndarray as a volume to world registration matrix
        (``morph['affine_reg']``). Typically
        from compute_morph_sdr.
    as_volume : bool
        Whether the function returns a Nifti1Image or not.

    Returns
    -------
    data_to : SourceEstimate | VectorSourceEstimate | VolSourceEstimate |
        SourceSpaces | Nifti1Image
        Source estimate for the destination subject.
        if data_from is VolSourceEstimate and as_volume=True data_to becomes a
        Nifti1Image in the destination subject's space.
    """
    subject_from = morph.subject_from,
    subject_to = morph.subject_to,
    if isinstance(data_from, VolSourceEstimate):

        # Note that the order is important: first affine and later a
        # non-linear transform
        morph_components = ['affine', 'mapping']

        # check if the correct morpher is present
        if not isinstance(morph.morpher, dict):
            raise ValueError(
                'morph must be a dictionary, containing at least one of'
                'the following keys: ' + ', '.join(morph_components))

        # check if the correct morpher is present
        if not sum(comp in morph.morpher for comp in morph_components) > 0:
            raise ValueError(
                'morph must be a dictionary, containing at least one of'
                'the following keys: ' + ', '.join(morph_components))

        if data_from.subject is not None and data_from.subject != subject_from:
            raise ValueError('data_from.subject and subject_from must match')

        from dipy.align import reslice
        import nibabel as nib

        img_to = _stc_as_volume(data_from, morph)

        old_voxel_size = np.asanyarray(
            img_to.header.get_zooms()[:3]).astype('float')

        img_slice_ratio = img_to.shape[:3] / old_voxel_size

        new_voxel_size = img_slice_ratio / \
                         morph.morpher['mapping'].domain_shape.astype('float')[
                         :3]

        # reslice to match morph
        img_to, img_to_affine = reslice.reslice(
            img_to.get_data(),
            img_to.affine,
            tuple(old_voxel_size),
            tuple(new_voxel_size))

        # transform according to pre-defined morph order
        for comp in morph_components:
            if comp in morph.morpher:
                for vol in range(img_to.shape[3]):
                    img_to[:, :, :, vol] = morph.morpher[comp].transform(
                        img_to[:, :, :, vol])

        if voxel_size is not None:
            # reslice to match voxel_size
            img_to, = reslice.reslice(
                img_to,
                None,
                tuple(new_voxel_size),
                voxel_size)

        data_to = nib.Nifti1Image(img_to, img_to_affine)
        morph.morpher['shape'] = data_to.shape[:3]
        morph.morpher['mri_affine'] = data_to.affine
        morph.morpher['stc_voxel_size'] = data_to.header.get_zooms()[:3]
        if not as_volume:
            if img_to.ndim is 4:
                nvol = img_to.shape[3]
            else:
                nvol = 1
            data_to = data_to.get_data().reshape(-1, nvol)

            # inuse = np.swapaxes(inuse.reshape(img_to.shape[:3]), 0,
            #                    2).flatten()
            vertices = [i for i, d in enumerate(sum(data_to, axis=1) == 0) if
                        not d]
            data_to = VolSourceEstimate(data_to[vertices, :],
                                        vertices=np.asanyarray(vertices),
                                        tmin=data_from.tmin,
                                        tstep=data_from.tstep,
                                        verbose=data_from.verbose,
                                        subject=subject_to)

    else:

        if not sparse.issparse(morph):
            raise ValueError('morph must be a sparse matrix')

        if not isinstance(space_to, list) or not len(space_to) == 2:
            raise ValueError('space_to must be a list of length 2')

        if not sum(len(v) for v in space_to) == morph.shape[0]:
            raise ValueError('number of vertices in space_to must match '
                             'morph.shape[0]')

        if not data_from.data.shape[0] == morph.shape[1]:
            raise ValueError('data_from.data.shape[0] must be the same as '
                             'morph.shape[0]')

        if data_from.subject is not None and data_from.subject != subject_from:
            raise ValueError('data_from.subject and subject_from must match')

        if isinstance(data_from, VectorSourceEstimate):
            # Morph the locations of the dipoles, but not their orientation
            n_verts, _, n_samples = data_from.data.shape
            data = morph * data_from.data.reshape(n_verts, 3 * n_samples)
            data = data.reshape(morph.shape[0], 3, n_samples)
            data_to = VectorSourceEstimate(data, vertices=space_to,
                                           tmin=data_from.tmin,
                                           tstep=data_from.tstep,
                                           verbose=data_from.verbose,
                                           subject=subject_to)
        else:
            data = morph * data_from.data
            data_to = SourceEstimate(data, vertices=space_to,
                                     tmin=data_from.tmin,
                                     tstep=data_from.tstep,
                                     verbose=data_from.verbose,
                                     subject=subject_to)

    return data_to


def _stc_as_volume(stc, morph, mri_resolution=True, voxel_size=None):
    from dipy.align import reslice
    import nibabel as nib

    shape = morph.morpher['shape']
    img_data = np.zeros((np.prod(shape), stc.data.shape[1]))
    img_data[stc.vertices, :] = stc.data

    affine = morph.morpher['mri_affine']
    img_data = img_data.reshape(shape[0], shape[1], shape[2], -1)
    img = nib.Nifti1Image(img_data, affine)

    if mri_resolution:
        if voxel_size is None:
            voxel_size = morph.morpher['mri_voxel_size']
        img, affine = reslice.reslice(
            img.get_data(),
            img.affine,
            morph.morpher['stc_voxel_size'],
            voxel_size)

        img = nib.Nifti1Image(img, affine)
    return img
