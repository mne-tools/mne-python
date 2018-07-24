# Author(s): Tommy Clausner <tommy.clausner@gmail.com>

# License: BSD (3-clause)


import os
import warnings
import copy

import numpy as np
from scipy import sparse
from scipy.sparse import block_diag as sparse_block_diag

from .parallel import parallel_func
from .source_estimate import (VolSourceEstimate, SourceEstimate,
                              VectorSourceEstimate, _get_ico_tris)
from .source_space import SourceSpaces
from .surface import read_morph_map, mesh_edges, read_surface, _compute_nearest
from .utils import (logger, verbose, check_version, get_subjects_dir,
                    warn as warn_, deprecated)
from .externals.h5io import read_hdf5, write_hdf5


class SourceMorph(object):
    """Container for source estimate morphs.

    Parameters
    ----------
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
    src : instance of SourceSpaces
        The list of SourceSpaces corresponding subject_from
    niter_affine : tuple of int
        Number of levels (``len(niter_affine)``) and number of
        iterations per level - for each successive stage of iterative
        refinement - to perform the affine transform.
        Default is niter_affine=(100, 100, 10)
    niter_sdr : tuple of int
        Number of levels (``len(niter_sdr)``) and number of
        iterations per level - for each successive stage of iterative
        refinement - to perform the Symmetric Diffeomorphic Registration (sdr)
        transform. Default is niter_sdr=(5, 5, 3)
    spacing : tuple | int | float | list | None
        If morphing VolSourceEstimate, spacing is a tuple carrying the voxel
        size of volume for each spatial dimension in mm.
        If spacing is None, MRIs won't be resliced. Note that in this case
        both volumes (used to compute the morph) must have the same number of
        spatial dimensions.
        If morphing SourceEstimate or Vector SourceEstimate, spacing can be
        int, list (of two arrays), or None, defining the resolution of the
        icosahedral mesh (typically 5). If None, all vertices will be used
        (potentially filling the surface). If a list, then values will be
        morphed to the set of vertices specified in in spacing[0] and
        spacing[1].
        Note that specifying the vertices (e.g., spacing=[np.arange(10242)] * 2
        for fsaverage on a standard spacing 5 source space) can be
        substantially faster than computing vertex locations.
    smooth : int | None
        Number of iterations for the smoothing of the surface data.
        If None, smooth is automatically defined to fill the surface
        with non-zero values.
    warn : bool
        If True, warn if not all vertices were used.
    xhemi : bool
        Morph across hemisphere. Currently only implemented for
        ``subject_to == subject_from``. See notes below.
    precomputed : dict | None
        Precomputed morphing data. For (Vector)SourceEstimates it contains
        of 'morph_mat' and 'vertno' where morph_mat is the respective
        transformation and vertno the corresponding vertices. For
        VolSourceEstimates it should contain 'DiffeomorphicMap' and
        'AffineMap' data, as well as 'morph_shape', 'morph_zooms' and
        'morph_affine', referring to the respective volume parameters.
    verbose : bool | str | int | None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).


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
        Number of levels (``len(niter_affine)``) and number of
        iterations per level - for each successive stage of iterative
        refinement - to perform the affine transform.
        Default is niter_affine=(100, 100, 10)
    niter_sdr : tuple of int
        Number of levels (``len(niter_sdr)``) and number of
        iterations per level - for each successive stage of iterative
        refinement - to perform the Symmetric Diffeomorphic Registration (sdr)
        transform. Default is niter_sdr=(5, 5, 3)
    spacing : tuple | int | float | list | None
        If morphing VolSourceEstimate, spacing is a tuple carrying the voxel
        size of volume for each spatial dimension in mm.
        If spacing is None, MRIs won't be resliced. Note that in this case
        both volumes (used to compute the morph) must have the same number of
        spatial dimensions.
        If morphing SourceEstimate or Vector SourceEstimate, spacing can be
        int, list (of two arrays), or None, defining the resolution of the
        icosahedral mesh (typically 5). If None, all vertices will be used
        (potentially filling the surface). If a list, then values will be
        morphed to the set of vertices specified in in spacing[0] and
        spacing[1].
        Note that specifying the vertices (e.g., spacing=[np.arange(10242),
        np.arange(10242)] for fsaverage on a standard spacing 5 source space)
        can be substantially faster than computing vertex locations.
    smooth : int | None
        Number of iterations for the smoothing of the surface data.
        If None, smooth is automatically defined to fill the surface
        with non-zero values.
    warn : bool
        If True, warn if not all vertices were used.
    xhemi : bool
        Morph across hemisphere.
    params : dict
        The morph data. Contains all data relevant for morphing and / or volume
        output.

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

    .. versionadded:: 0.17.0

    References
    ----------
    .. [1] Greve D. N., Van der Haegen L., Cai Q., Stufflebeam S., Sabuncu M.
           R., Fischl B., Brysbaert M.
           A Surface-based Analysis of Language Lateralization and Cortical
           Asymmetry. Journal of Cognitive Neuroscience 25(9), 1477-1492, 2013.

    See Also
    --------
    .. :meth::`stc.morph <mne.SourceEstimate.morph>`
    .. :meth::`stc.as_volume <mne.VolSourceEstimate.as_volume>`
    .. :ref::`example <sphx_glr_auto_examples_plot_use_sourcemorph.py>`
    .. :ref::`tutorial <sphx_glr_auto_tutorials_plot_morph.py>`
    """

    def __init__(self, subject_from=None, subject_to='fsaverage',
                 subjects_dir=None, src=None, niter_affine=(100, 100, 10),
                 niter_sdr=(5, 5, 3), spacing=5, smooth=None, warn=True,
                 xhemi=False, precomputed=None, verbose=False):

        # it's impossible to use the class without passing this check, so it
        # only needs to be checked here
        if (not check_version('nibabel', '2.1.0') or
                not check_version('dipy', '0.10.1')):
            raise ImportError(
                'NiBabel 2.1.0 and DiPy 0.10.1 or higher must be correctly '
                'installed and accessible from Python')

        if src is not None and not isinstance(src, SourceSpaces):
            raise ValueError('src must be an instance of SourceSpaces or a '
                             'path to a saved instance of SourceSpaces')
        # Params
        self.kind = 'surface' if src is None else src.kind
        self.subject_from = _check_subject_from(subject_from, src)
        self.subject_to = subject_to
        self.subjects_dir = subjects_dir

        # Params for volumetric morphing
        self.niter_affine = niter_affine
        self.niter_sdr = niter_sdr

        self.spacing = spacing

        # Params for surface morphing
        self.smooth = smooth
        self.warn = warn
        self.xhemi = xhemi

        self.params = dict()

        # apply precomputed data and return
        if precomputed is not None:
            self._update_morph_data(precomputed)
            return

        # get data to perform morph and as_volume
        if src is not None:
            self._update_morph_data(_get_src_data(src))
            self._compute_morph_data(verbose=verbose)

    # Forward verbose decorator to _apply_morph_data
    def __call__(self, stc_from, as_volume=False, mri_resolution=False,
                 mri_space=False, apply_morph=True, verbose=None):
        """Morph data.

        Parameters
        ----------
        stc_from : VolSourceEstimate | SourceEstimate | VectorSourceEstimate
            The source estimate to morph.
        as_volume : bool
            Whether to output a NIfTI volume. stc_from has to be a
            VolSourceEstimate. Default is False.
        mri_resolution: bool | tuple | int | float
            If True the image is saved in MRI resolution. Default False.
            WARNING: if you have many time points the file produced can be
            huge.
        mri_space : bool
            Whether the image to world registration should be in mri space.
            Default is False.
        apply_morph : bool
            If as_volume=True and apply_morph=True, the input stc will be
            morphed and outputted as a volume.
        verbose : bool | str | int | None
            If not None, override default verbose level (see :func:`mne.
            verbose` and :ref:`Logging documentation <tut_logging>` for more).

        Returns
        -------
        stc_to : VolSourceEstimate | SourceEstimate | VectorSourceEstimate | Nifti1Image
            The morphed source estimate or a NIfTI image if as_volume=True.
        """  # noqa: E501
        stc = copy.deepcopy(stc_from)

        if as_volume:
            return self.as_volume(stc, fname=None,
                                  mri_resolution=mri_resolution,
                                  mri_space=mri_space,
                                  apply_morph=apply_morph)

        if stc.subject is None:
            stc.subject = self.subject_from

        if self.subject_from is None:
            self.subject_from = stc.subject

        if stc.subject != self.subject_from:
            raise ValueError('stc_from.subject and '
                             'morph.subject_from must match. (%s != %s)' %
                             (stc.subject, self.subject_from))

        # if not precomputed
        if 'morph_mat' not in self.params and self.kind == 'surface':
            self._update_morph_data({'0': stc.lh_vertno,
                                     '1': stc.rh_vertno,
                                     'hemis': [0, 1]},
                                    kind='surface')
            self._compute_morph_data(verbose=verbose)

        return _apply_morph_data(self, stc, verbose=verbose)

    def __repr__(self):  # noqa: D105
        s = "%s" % self.kind
        s += ", subject_from : %s" % self.subject_from
        s += ", subject_to : %s" % self.subject_to
        s += ", spacing : {}".format(self.spacing)
        if self.kind == 'volume':
            s += ", niter_affine : {}".format(self.niter_affine)
            s += ", niter_sdr : {}".format(self.niter_sdr)

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

        write_hdf5(fname, self.__dict__, overwrite=True)
        logger.info('[done]')

    def as_volume(self, stc, fname=None, mri_resolution=False, mri_space=True,
                  apply_morph=False, format='nifti1'):
        """Return volume source space as Nifti1Image and / or save to disk.

        Parameters
        ----------
        stc : VolSourceEstimate
            Data to be transformed
        fname : str | None
            String to where to save the volume. If not None that volume will
            be saved at fname.
        mri_resolution: bool | tuple | int | float
            Whether to use MRI resolution. If False the morph's resolution
            will be used. If tuple the voxel size must be given in float values
            in mm. E.g. mri_resolution=(3., 3., 3.)
            WARNING: if you have many time points the file produced can be
            huge.
        mri_space : bool
            Whether the image to world registration should be in MRI space.
        apply_morph : bool
            Whether to apply the precomputed morph to stc or not. Default is
            False.
        format : str
            Either 'nifti1' (default) or 'nifti2'

        Returns
        -------
        img : instance of Nifti1Image
            The image object.
        """
        if format != 'nifti1' and format != 'nifti2':
            raise ValueError("invalid format specifier %s. Must be 'nifti1' or"
                             " 'nifti2'" % format)
        if apply_morph:
            stc = self.__call__(stc)  # apply morph if desired
        return _stc_as_volume(self, stc, fname=fname,
                              mri_resolution=mri_resolution,
                              mri_space=mri_space,
                              format=format)

    def _update_morph_data(self, data=None, kind=None):
        """Update morph data and kind."""
        if data is not None:
            self.params.update(data)

        if kind is not None:
            self.kind = kind

    def _compute_morph_data(self, verbose=None):
        """Compute morph data."""
        self._update_morph_data(_compute_morph_data(self, verbose=verbose))


###############################################################################
# I/O
def _check_subject_from(subject_from, src_in):
    src = copy.deepcopy(src_in)
    if src is None:
        return subject_from

    if src[0]['subject_his_id'] is not None:
        subject_his_id = src[0]['subject_his_id']
        if subject_from is not None and subject_from != subject_his_id:
            raise ValueError('subject_from does not match source space subject'
                             ' (%s != %s)' % (subject_from, subject_his_id))
        subject_from = subject_his_id

    if src[0]['subject_his_id'] is not None and subject_from is None:
        subject_from = src[0]['subject_his_id']

    if subject_from is None and src.kind == 'volume':
        raise ValueError(
            'subject_from is None. Please specify subject_from when working '
            'with volume source space.')
    del src
    return subject_from


@verbose
def read_source_morph(fname, verbose=None):
    """Load the morph for source estimates from a file.

    Parameters
    ----------
    fname : str
        Full filename including path.
    verbose : bool | str | int | None
        If not None, override default verbose level (see
        :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
        for more). Defaults to self.verbose.

    Returns
    -------
    source_morph : instance of SourceMorph
        The loaded morph.
    """
    logger.info('loading morph...')
    data = read_hdf5(fname)

    source_morph = SourceMorph()
    source_morph.__dict__.update(data)
    logger.info('[done]')
    return source_morph


###############################################################################
# Helper functions for SourceMorph methods
def _check_hemi_data(data_in, data_ref):
    """Check and setup correct data for hemispheres."""
    return [np.array([], int) if
            len(data_ref[h]) == 0 or
            data_ref[h] is None else
            data_in[h]
            for h in range(len(data_ref))]


def _stc_as_volume(morph, stc, fname=None, mri_resolution=False,
                   mri_space=True, format='nifti1'):
    """Return volume source space as Nifti1Image and/or save to disk.

    Parameters
    ----------
    morph : instance of SourceMorph
        Instance of SourceMorph carrying the relevant volume transform
        information. Typically computed using SourceMorph().
    stc : VolSourceEstimate
        Data to be transformed.
    fname : str | None
        String to where to save the volume. If not None that volume will
        be saved at fname.
    mri_resolution: bool | tuple | int | float
        Whether to use MRI resolution. If False the morph's resolution
        will be used. If tuple the voxel size must be given in float values
        in mm. E.g. mri_resolution=(3., 3., 3.)
        WARNING: if you have many time points the file produced can be
        huge.
    mri_space : bool
        Whether the image to world registration should be in MRI space.
    format : str
        Either 'nifti1' (default) or 'nifti2'

    Returns
    -------
    img : instance of Nifti1Image
        The image object.
    """
    import nibabel as nib
    if not isinstance(stc, VolSourceEstimate):
        raise ValueError('Only volume source estimates can be converted to '
                         'volumes')

    # this is a special case when as_volume is called without having done a
    # morph beforehand (to assure compatibility to previous versions)
    if 'morph_shape' not in morph.params:
        img = _interpolate_data(stc, morph.params,
                                mri_resolution=mri_resolution,
                                mri_space=mri_space,
                                format=format)
        if fname is not None:
            nib.save(img, fname)
        return img

    if format == 'nifti1':
        from nibabel import (Nifti1Image as NiftiImage,
                             Nifti1Header as NiftiHeader)
    elif format == 'nifti2':
        from nibabel import (Nifti2Image as NiftiImage,
                             Nifti2Header as NiftiHeader)

    from dipy.align.reslice import reslice

    new_zooms = None

    # if full MRI resolution, compute zooms from shape and MRI zooms
    if isinstance(mri_resolution, bool) and mri_resolution:
        new_zooms = _get_zooms_orig(morph.params)

    # if MRI resolution is set manually as a single value, convert to tuple
    if isinstance(mri_resolution, (int, float)) and not isinstance(
            mri_resolution, bool):
        # use iso voxel size
        new_zooms = (float(mri_resolution),) * 3

    # if MRI resolution is set manually as a tuple, use it
    if isinstance(mri_resolution, tuple):
        new_zooms = mri_resolution

    # setup volume properties
    shape = tuple([int(i) for i in morph.params['morph_shape']])
    affine = morph.params['morph_affine']
    zooms = morph.params['morph_zooms'][:3]

    # create header
    hdr = NiftiHeader()
    hdr.set_xyzt_units('mm', 'msec')
    hdr['pixdim'][4] = 1e3 * stc.tstep

    # setup empty volume
    img = np.zeros(shape + (stc.shape[1],)).reshape(-1, stc.shape[1])
    img[stc.vertices, :] = stc.data

    img = img.reshape(shape + (-1,))

    # make nifti from data
    with warnings.catch_warnings(record=True):  # nibabel<->numpy warning
        img = NiftiImage(img, affine, header=hdr)

    # reslice in case of manually defined voxel size
    if new_zooms is not None:
        new_zooms = new_zooms[:3]
        img, affine = reslice(img.get_data(),
                              img.affine,  # MRI to world registration
                              zooms,  # old voxel size in mm
                              new_zooms)  # new voxel size in mm
        with warnings.catch_warnings(record=True):  # nibabel<->numpy warning
            img = NiftiImage(img, affine)
        zooms = new_zooms

    #  set zooms in header
    img.header.set_zooms(tuple(zooms) + (1,))

    # save if fname is provided
    if fname is not None:
        nib.save(img, fname)

    return img


def _get_src_data(src):
    """Obtain src data relevant for as _volume."""
    src_data = dict()

    # copy data to avoid conflicts
    src_t = copy.deepcopy(src)

    # extract all relevant data for volume operations
    if src.kind == 'volume':
        shape = src_t[0]['shape']
        src_data.update({'src_shape': (shape[2], shape[1], shape[0]),
                         'src_affine_vox': src_t[0]['vox_mri_t']['trans'],
                         'src_affine_src': src_t[0]['src_mri_t']['trans'],
                         'src_affine_ras': src_t[0]['mri_ras_t']['trans'],
                         'src_shape_full': (
                             src_t[0]['mri_height'], src_t[0]['mri_depth'],
                             src_t[0]['mri_width']),
                         'interpolator': src_t[0]['interpolator'],
                         'inuse': src_t[0]['inuse']})
    elif src_t.kind == 'surface':
        hemis = []
        # extract information for each hemisphere and save order (necessary for
        # xhemi)
        for n, s in enumerate(src_t):
            src_data.update({str(n): s['vertno']})
            hemis.append(n)
        src_data.update({'hemis': hemis})

    # delete copy
    del src_t
    return src_data


def _compute_morph_data(morph, verbose=None):
    """Compute source estimate specific morph."""
    data = dict()

    # get data currently present in morph
    data.update(morph.params)
    subjects_dir = get_subjects_dir(morph.subjects_dir,
                                    raise_error=True)

    # VolSourceEstimate
    if morph.kind == 'volume' and morph.subject_to is not None:

        logger.info('volume source space inferred...')
        import nibabel as nib

        # load moving MRI
        mri_subpath = os.path.join('mri', 'brain.mgz')
        mri_path_from = os.path.join(subjects_dir, morph.subject_from,
                                     mri_subpath)

        logger.info('loading %s as moving volume' % mri_path_from)
        mri_from = nib.load(mri_path_from)

        # load static MRI
        static_path = os.path.join(subjects_dir, morph.subject_to)

        if not os.path.isdir(static_path):
            mri_path_to = static_path
        else:
            mri_path_to = os.path.join(static_path, mri_subpath)

        if os.path.isfile(mri_path_to):
            logger.info('loading %s as static volume' % mri_path_to)
            mri_to = nib.load(mri_path_to)
        else:
            raise IOError('cannot read file: %s' % mri_path_to)

        # pre-compute non-linear morph
        data.update(_compute_morph_sdr(
            mri_from,
            mri_to,
            niter_affine=morph.niter_affine,
            niter_sdr=morph.niter_sdr,
            spacing=morph.spacing,
            verbose=verbose))

    # SourceEstimate | VectorSourceEstimate
    elif morph.kind == 'surface':
        logger.info('surface source space inferred...')

        # get surface data
        data_from = []
        hemis = morph.params['hemis']
        for h in hemis:
            data_from.append(morph.params[str(h)])

        data_to = None
        # default for fsaverage
        if morph.subject_to == 'fsaverage':
            data_to = [np.arange(10242)] * 2

        if (isinstance(morph.spacing, int) or
                isinstance(morph.spacing, list) or
                morph.subject_to != 'fsaverage'):
            data_to = grade_to_vertices(morph.subject_to, morph.spacing,
                                        subjects_dir, 1)
            data_from = _check_hemi_data(data_from, data_to)

        if data_to is None:
            raise ValueError('Please specify target to morph to.')

        # check emptiness of hemispheres
        data_to = _check_hemi_data(data_to, data_from)

        # pre-compute morph matrix
        morph_mat = _compute_morph_matrix(
            subject_from=morph.subject_from,
            subject_to=morph.subject_to,
            vertices_from=data_from,
            vertices_to=data_to,
            subjects_dir=subjects_dir,
            smooth=morph.smooth,
            warn=morph.warn,
            xhemi=morph.xhemi,
            verbose=verbose)
        data.update({'morph_mat': morph_mat, 'vertno': data_to})

    return data


def _interpolate_data(stc, morph_data, mri_resolution=True, mri_space=True,
                      format='nifti1'):
    """Interpolate source estimate data to MRI."""
    if format != 'nifti1' and format != 'nifti2':
        raise ValueError("invalid format specifier %s. Must be 'nifti1' or"
                         " 'nifti2'" % format)
    if format == 'nifti1':
        from nibabel import (Nifti1Image as NiftiImage,
                             Nifti1Header as NiftiHeader)
    elif format == 'nifti2':
        from nibabel import (Nifti2Image as NiftiImage,
                             Nifti2Header as NiftiHeader)
    from dipy.align.reslice import reslice

    # setup volume parameters
    n_times = stc.data.shape[1]
    shape3d = morph_data['src_shape']
    shape = (n_times,) + shape3d
    vol = np.zeros(shape)

    voxel_size_defined = False

    if isinstance(mri_resolution, (int, float)) and not isinstance(
            mri_resolution, bool):
        # use iso voxel size
        mri_resolution = (float(mri_resolution),) * 3

    if isinstance(mri_resolution, tuple):
        voxel_size = mri_resolution
        voxel_size_defined = True
        mri_resolution = True

    # if data wasn't morphed yet - necessary for call of
    # stc_unmorphed.as_volume. Since only the shape of src is known, it cannot
    # be resliced to a given voxel size without knowing the original.
    if 'morph_shape' not in morph_data and voxel_size_defined:
        raise ValueError(
            "Cannot infer original voxel size for reslicing... "
            "set mri_resolution to boolean value or apply morph first.")

    # use mri resolution as represented in src
    if mri_resolution:
        mri_shape3d = morph_data['src_shape_full']
        mri_shape = (n_times,) + mri_shape3d
        mri_vol = np.zeros(mri_shape)
        interpolator = morph_data['interpolator']

    mask3d = morph_data['inuse'].reshape(shape3d).astype(np.bool)
    n_vertices = np.sum(mask3d)

    n_vertices_seen = 0
    for k, v in enumerate(vol):  # loop over time instants
        stc_slice = slice(n_vertices_seen, n_vertices_seen + n_vertices)
        v[mask3d] = stc.data[stc_slice, k]

    n_vertices_seen += n_vertices

    if mri_resolution:
        for k, v in enumerate(vol):
            mri_vol[k] = (interpolator * v.ravel()).reshape(mri_shape3d)
        vol = mri_vol

    vol = vol.T

    # set correct space
    affine = morph_data['src_affine_vox']

    if not mri_resolution:
        affine = morph_data['src_affine_src']

    if mri_space:
        affine = np.dot(morph_data['src_affine_ras'], affine)

    affine[:3] *= 1e3

    # pre-define header
    header = NiftiHeader()
    header.set_xyzt_units('mm', 'msec')
    header['pixdim'][4] = 1e3 * stc.tstep

    with warnings.catch_warnings(record=True):  # nibabel<->numpy warning
        img = NiftiImage(vol, affine, header=header)

    # if a specific voxel size was targeted (only possible after morphing)
    if voxel_size_defined:
        # reslice mri
        img, img_affine = reslice(
            img.get_data(),
            img.affine,
            _get_zooms_orig(morph_data),
            voxel_size)
        with warnings.catch_warnings(record=True):  # nibabel<->numpy warning
            img = NiftiImage(img, img_affine, header=header)

    return img


###############################################################################
# Morph for VolSourceEstimate
@verbose
def _compute_morph_sdr(mri_from, mri_to,
                       niter_affine=(100, 100, 10),
                       niter_sdr=(5, 5, 3),
                       spacing=(5., 5., 5.),
                       verbose=None):
    """Get a matrix that morphs data from one subject to another.

    Parameters
    ----------
    mri_from : str | Nifti1Image
        Path to source subject's anatomical MRI or Nifti1Image
    mri_to : str | Nifti1Image
        Path to destination subject's anatomical MRI or Nifti1Image
    niter_affine : tuple of int
        Number of levels (``len(niter_affine)``) and number of
        iterations per level - for each successive stage of iterative
        refinement - to perform the affine transform.
        Default is niter_affine=(100, 100, 10)
    niter_sdr : tuple of int
        Number of levels (``len(niter_sdr)``) and number of
        iterations per level - for each successive stage of iterative
        refinement - to perform the Symmetric Diffeomorphic Registration (sdr)
        transform. Default is niter_sdr=(5, 5, 3)
    spacing : tuple | int | float | None
        Voxel size of volume for each spatial dimension separately (tuple) or
        isometric (int).
        If spacing is None, MRIs won't be resliced. Note that in this case
        both volumes must have the same number of slices in every
        spatial dimension.
    verbose : bool | str | int | None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    morph : dict
        Details about AffineMap, DiffeomorphicMap, morph_shape, morph_zooms,
        morph_affine for affine and diffeomorphic registration.

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

    morph_params = dict()

    # use voxel size of mri_from
    if spacing is None:
        spacing = mri_from.header.get_zooms()[:3]

    # use iso voxel size
    if isinstance(spacing, (int, float)):
        spacing = (float(spacing),) * 3

    # reslice mri_from
    mri_from_res, mri_from_res_affine = reslice(
        mri_from.get_data(),
        mri_from.affine,
        mri_from.header.get_zooms()[:3],
        spacing)

    with warnings.catch_warnings(record=True):  # nibabel<->numpy warning
        mri_from = nib.Nifti1Image(mri_from_res, mri_from_res_affine)

    # reslice mri_to
    mri_to_res, mri_to_res_affine = reslice(
        mri_to.get_data(),
        mri_to.affine,
        mri_to.header.get_zooms()[:3],
        spacing)

    with warnings.catch_warnings(record=True):  # nibabel<->numpy warning
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
    morph_params.update(
        {'morph_shape': tuple(mapping.domain_shape.astype('float')),
         'morph_zooms': spacing,
         'morph_affine': mri_to_grid2world,
         'AffineMap': affine.__dict__,
         'DiffeomorphicMap': mapping.__dict__})

    logger.info('done.')

    return morph_params


###############################################################################
# Morph for SourceEstimate |  VectorSourceEstimate
@deprecated("This function is deprecated and might be removed in a future "
            "release. Use morph = mne.SourceMorph and morph(stc). Access the "
            "morph matrix via morph.params['morph_mat']")
def compute_morph_matrix(subject_from, subject_to, vertices_from, vertices_to,
                         smooth=None, subjects_dir=None, warn=True,
                         xhemi=False, verbose=None):
    """Wrapper for _compute_morph_matrix to assure backwards compatibility."""
    return _compute_morph_matrix(subject_from, subject_to, vertices_from,
                                 vertices_to, smooth, subjects_dir, warn,
                                 xhemi, verbose)


@verbose
def _compute_morph_matrix(subject_from, subject_to, vertices_from, vertices_to,
                          smooth=None, subjects_dir=None, warn=True,
                          xhemi=False, verbose=None):
    """Get a matrix that morphs data from one subject to another.

    Parameters
    ----------
    subject_from : str
        Name of the original subject as named in the SUBJECTS_DIR.
    subject_to : str
        Name of the subject on which to morph as named in the SUBJECTS_DIR.
    vertices_from : list of arrays of int
        Vertices for each hemisphere (LH, RH) for subject_from.
    vertices_to : list of arrays of int
        Vertices for each hemisphere (LH, RH) for subject_to.
    smooth : int or None
        Number of iterations for the smoothing of the surface data.
        If None, smooth is automatically defined to fill the surface
        with non-zero values.
    subjects_dir : str
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

    # morph the data

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
def grade_to_vertices(subject, grade, subjects_dir=None, n_jobs=1,
                      verbose=None):
    """Convert a grade to source space vertices for a given subject.

    Parameters
    ----------
    subject : str
        Name of the subject
    grade : int | list
        Resolution of the icosahedral mesh (typically 5). If None, all
        vertices will be used (potentially filling the surface). If a list,
        then values will be morphed to the set of vertices specified in
        in grade[0] and grade[1]. Note that specifying the vertices (e.g.,
        grade=[np.arange(10242), np.arange(10242)] for fsaverage on a
        standard grade 5 source space) can be substantially faster than
        computing vertex locations. Note that if subject='fsaverage'
        and 'grade=5', this set of vertices will automatically be used
        (instead of computed) for speed, since this is a common morph.
    subjects_dir : str | None
        Path to SUBJECTS_DIR if it is not set in the environment
    n_jobs : int
        Number of jobs to run in parallel
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    vertices : list of arrays of int
        Vertex numbers for LH and RH
    """
    # add special case for fsaverage for speed
    if subject == 'fsaverage' and grade == 5:
        return [np.arange(10242)] * 2
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)

    spheres_to = [os.path.join(subjects_dir, subject, 'surf',
                               xh + '.sphere.reg') for xh in ['lh', 'rh']]
    lhs, rhs = [read_surface(s)[0] for s in spheres_to]

    if grade is not None:  # fill a subset of vertices
        if isinstance(grade, list):
            if not len(grade) == 2:
                raise ValueError('grade as a list must have two elements '
                                 '(arrays of output vertices)')
            vertices = grade
        else:
            # find which vertices to use in "to mesh"
            ico = _get_ico_tris(grade, return_surf=True)
            lhs /= np.sqrt(np.sum(lhs ** 2, axis=1))[:, None]
            rhs /= np.sqrt(np.sum(rhs ** 2, axis=1))[:, None]

            # Compute nearest vertices in high dim mesh
            parallel, my_compute_nearest, _ = \
                parallel_func(_compute_nearest, n_jobs)
            lhs, rhs, rr = [a.astype(np.float32)
                            for a in [lhs, rhs, ico['rr']]]
            vertices = parallel(my_compute_nearest(xhs, rr)
                                for xhs in [lhs, rhs])
            # Make sure the vertices are ordered
            vertices = [np.sort(verts) for verts in vertices]
            for verts in vertices:
                if (np.diff(verts) == 0).any():
                    raise ValueError(
                        'Cannot use icosahedral grade %s with subject %s, '
                        'mapping %s vertices onto the high-resolution mesh '
                        'yields repeated vertices, use a lower grade or a '
                        'list of vertices from an existing source space'
                        % (grade, subject, len(verts)))
    else:  # potentially fill the surface
        vertices = [np.arange(lhs.shape[0]), np.arange(rhs.shape[0])]

    return vertices


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


def _sparse_argmax_nnz_row(csr_mat):
    """Return index of the maximum non-zero index in each row."""
    n_rows = csr_mat.shape[0]
    idx = np.empty(n_rows, dtype=np.int)
    for k in range(n_rows):
        row = csr_mat[k].tocoo()
        idx[k] = row.col[np.argmax(row.data)]
    return idx


def _morph_sparse(stc, subject_from, subject_to, subjects_dir=None):
    """Morph sparse source estimates to an other subject.

    Parameters
    ----------
    stc : SourceEstimate | VectorSourceEstimate
        The sparse STC.
    subject_from : str
        The subject on which stc is defined.
    subject_to : str
        The target subject.
    subjects_dir : str
        Path to SUBJECTS_DIR if it is not set in the environment.

    Returns
    -------
    stc_morph : SourceEstimate | VectorSourceEstimate
        The morphed source estimates.
    """
    maps = read_morph_map(subject_to, subject_from, subjects_dir)
    stc_morph = stc.copy()
    stc_morph.subject = subject_to

    cnt = 0
    for h in [0, 1]:
        if len(stc.vertices[h]) > 0:
            map_hemi = maps[h]
            vertno_h = _sparse_argmax_nnz_row(map_hemi[stc.vertices[h]])
            order = np.argsort(vertno_h)
            n_active_hemi = len(vertno_h)
            data_hemi = stc_morph.data[cnt:cnt + n_active_hemi]
            stc_morph.data[cnt:cnt + n_active_hemi] = data_hemi[order]
            stc_morph.vertices[h] = vertno_h[order]
            cnt += n_active_hemi
        else:
            stc_morph.vertices[h] = np.array([], int)

    return stc_morph


def _get_subject_sphere_tris(subject, subjects_dir):
    spheres = [os.path.join(subjects_dir, subject, 'surf',
                            xh + '.sphere.reg') for xh in ['lh', 'rh']]
    tris = [read_surface(s)[1] for s in spheres]
    return tris


###############################################################################
# Apply morph to source estimate
def _get_zooms_orig(morph_data):
    """Compute src zooms from morph zooms, morph shape and src shape."""
    morph_zooms = morph_data['morph_zooms']
    morph_shape = morph_data['morph_shape']
    src_shape = morph_data['src_shape_full']

    # zooms_to = zooms_from / shape_to * shape_from for each spatial dimension
    return [mz / ss * ms for mz, ms, ss in
            zip(morph_zooms, morph_shape, src_shape)]


@verbose
def _apply_morph_data(morph, stc_from, verbose=None):
    """Morph a source estimate from one subject to another.

    Parameters
    ----------
    morph : SourceMorph
        Instance of SourceMorph used to compute the respective morphing data
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
    if morph.kind == 'volume':

        from dipy.align.imwarp import DiffeomorphicMap
        from dipy.align.imaffine import AffineMap
        from dipy.align.reslice import reslice

        # prepare data to be morphed
        img_to = _interpolate_data(stc_from, morph.params, mri_resolution=True,
                                   mri_space=True)

        # setup morphs to not carry those custom objects around
        # (issues in saving / loading)
        affine_morph = AffineMap(None)
        affine_morph.__dict__ = morph.params['AffineMap']

        sdr_morph = DiffeomorphicMap(None, [])
        sdr_morph.__dict__ = morph.params['DiffeomorphicMap']

        # reslice to match morph
        img_to, img_to_affine = reslice(
            img_to.get_data(),
            morph.params['morph_affine'],
            _get_zooms_orig(morph.params),
            morph.params['morph_zooms'])

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

        morph_mat = morph.params['morph_mat']
        vertices_to = morph.params['vertno']

        data = stc_from.data

        # apply morph and return new morphed instance of (Vector)SourceEstimate
        if isinstance(stc_from, VectorSourceEstimate):
            # Morph the locations of the dipoles, but not their orientation
            n_verts, _, n_samples = stc_from.data.shape
            data = morph_mat * data.reshape(n_verts, 3 * n_samples)
            data = data.reshape(morph_mat.shape[0], 3, n_samples)
            stc_to = VectorSourceEstimate(data, vertices=vertices_to,
                                          tmin=stc_from.tmin,
                                          tstep=stc_from.tstep,
                                          verbose=verbose,
                                          subject=morph.subject_to)
        else:
            data = morph_mat * data
            stc_to = SourceEstimate(data, vertices=vertices_to,
                                    tmin=stc_from.tmin,
                                    tstep=stc_from.tstep,
                                    verbose=verbose,
                                    subject=morph.subject_to)
    else:
        stc_to = None

    return stc_to
