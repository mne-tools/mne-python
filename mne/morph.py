# Author(s): Tommy Clausner <tommy.clausner@gmail.com>
#            Alexandre Gramfort <alexandre.gramfort@inria.fr>
#            Eric Larson <larson.eric.d@gmail.com>

# License: BSD (3-clause)

import os.path as op
import warnings
import copy
import numpy as np
from scipy import sparse

from .fixes import _get_img_fdata
from .parallel import parallel_func
from .source_estimate import (
    VolSourceEstimate, _BaseSurfaceSourceEstimate,
    _BaseVolSourceEstimate, _BaseSourceEstimate, _get_ico_tris)
from .source_space import SourceSpaces, _ensure_src
from .surface import read_morph_map, mesh_edges, read_surface, _compute_nearest
from .transforms import _angle_between_quats, rot_to_quat
from .utils import (logger, verbose, check_version, get_subjects_dir,
                    warn as warn_, fill_doc, _check_option, _validate_type,
                    BunchConst, wrapped_stdout, _check_fname, warn,
                    _ensure_int)
from .externals.h5io import read_hdf5, write_hdf5


@verbose
def compute_source_morph(src, subject_from=None, subject_to='fsaverage',
                         subjects_dir=None, zooms='auto',
                         niter_affine=(100, 100, 10), niter_sdr=(5, 5, 3),
                         spacing=5, smooth=None, warn=True, xhemi=False,
                         sparse=False, src_to=None, verbose=False):
    """Create a SourceMorph from one subject to another.

    Method is based on spherical morphing by FreeSurfer for surface
    cortical estimates :footcite:`GreveEtAl2013` and
    Symmetric Diffeomorphic Registration for volumic data
    :footcite:`AvantsEtAl2008`.

    Parameters
    ----------
    src : instance of SourceSpaces | instance of SourceEstimate
        The SourceSpaces of subject_from (can be a
        SourceEstimate if only using a surface source space).
    subject_from : str | None
        Name of the original subject as named in the SUBJECTS_DIR.
        If None (default), then ``src[0]['subject_his_id]'`` will be used.
    subject_to : str | None
        Name of the subject to which to morph as named in the SUBJECTS_DIR.
        Default is ``'fsaverage'``. If None, ``src_to[0]['subject_his_id']``
        will be used.

        .. versionchanged:: 0.20
           Support for subject_to=None.
    %(subjects_dir)s
    zooms : float | tuple | str | None
        The voxel size of volume for each spatial dimension in mm.
        If spacing is None, MRIs won't be resliced, and both volumes
        must have the same number of spatial dimensions.
        Can also be ``'auto'`` to use ``5.`` if ``src_to is None`` and
        the zooms from ``src_to`` otherwise.

        .. versionchanged:: 0.20
           Support for 'auto' mode.
    niter_affine : tuple of int
        Number of levels (``len(niter_affine)``) and number of
        iterations per level - for each successive stage of iterative
        refinement - to perform the affine transform.
        Default is niter_affine=(100, 100, 10).
    niter_sdr : tuple of int
        Number of levels (``len(niter_sdr)``) and number of
        iterations per level - for each successive stage of iterative
        refinement - to perform the Symmetric Diffeomorphic Registration (sdr)
        transform. Default is niter_sdr=(5, 5, 3).
    spacing : int | list | None
        The resolution of the icosahedral mesh (typically 5).
        If None, all vertices will be used (potentially filling the
        surface). If a list, then values will be morphed to the set of
        vertices specified in in ``spacing[0]`` and ``spacing[1]``.
        This will be ignored if ``src_to`` is supplied.

        .. versionchanged:: 0.21
           src_to, if provided, takes precedence.
    smooth : int | str | None
        Number of iterations for the smoothing of the surface data.
        If None, smooth is automatically defined to fill the surface
        with non-zero values. Can also be ``'nearest'`` to use the nearest
        vertices on the surface (requires SciPy >= 1.3).

        .. versionchanged:: 0.20
           Added support for 'nearest'.
    warn : bool
        If True, warn if not all vertices were used. The default is warn=True.
    xhemi : bool
        Morph across hemisphere. Currently only implemented for
        ``subject_to == subject_from``. See notes below.
        The default is xhemi=False.
    sparse : bool
        Morph as a sparse source estimate. Works only with (Vector)
        SourceEstimate. If True the only parameters used are subject_to and
        subject_from, and spacing has to be None. Default is sparse=False.
    src_to : instance of SourceSpaces | None
        The destination source space.

        - For surface-based morphing, this is the preferred over ``spacing``
          for providing the vertices.
        - For volumetric morphing, this should be passed so that 1) the
          resultingmorph volume is properly constrained to the brain volume,
          and 2) STCs from multiple subjects morphed to the same destination
          subject/source space have the vertices.
        - For mixed (surface + volume) morphing, this is required.

        .. versionadded:: 0.20
    %(verbose)s

    Returns
    -------
    morph : instance of SourceMorph
        The :class:`mne.SourceMorph` object.

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
    <https://surfer.nmr.mgh.harvard.edu/fswiki/Xhemi>`_. For statistical
    comparisons between hemispheres, use of the symmetric ``fsaverage_sym``
    model is recommended to minimize bias :footcite:`GreveEtAl2013`.

    .. versionadded:: 0.17.0

    .. versionadded:: 0.21.0
       Support for morphing mixed source estimates.

    References
    ----------
    .. footbibliography::
    """
    src_data, kind, src_subject = _get_src_data(src)
    subject_from = _check_subject_src(subject_from, src_subject)
    del src
    _validate_type(src_to, (SourceSpaces, None), 'src_to')
    _validate_type(subject_to, (str, None), 'subject_to')
    if src_to is None and subject_to is None:
        raise ValueError('subject_to cannot be None when src_to is None')
    subject_to = _check_subject_src(subject_to, src_to, 'subject_to')

    # Params
    warn = False if sparse else warn

    if kind not in 'surface' and xhemi:
        raise ValueError('Inter-hemispheric morphing can only be used '
                         'with surface source estimates.')
    if sparse and kind != 'surface':
        raise ValueError('Only surface source estimates can compute a '
                         'sparse morph.')

    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    shape = affine = pre_affine = sdr_morph = morph_mat = None
    vertices_to_surf, vertices_to_vol = list(), list()

    if kind in ('volume', 'mixed'):
        _check_dep(nibabel='2.1.0', dipy='0.10.1')
        import nibabel as nib

        logger.info('Volume source space(s) present...')

        # load moving MRI
        mri_subpath = op.join('mri', 'brain.mgz')
        mri_path_from = op.join(subjects_dir, subject_from, mri_subpath)

        logger.info('    Loading %s as "from" volume' % mri_path_from)
        with warnings.catch_warnings():
            mri_from = nib.load(mri_path_from)

        # eventually we could let this be some other volume, but for now
        # let's KISS and use `brain.mgz`, too
        mri_path_to = op.join(subjects_dir, subject_to, mri_subpath)
        if not op.isfile(mri_path_to):
            raise IOError('cannot read file: %s' % mri_path_to)
        logger.info('    Loading %s as "to" volume' % mri_path_to)
        with warnings.catch_warnings():
            mri_to = nib.load(mri_path_to)

        # deal with `src_to` subsampling
        zooms_src_to = None
        if src_to is None:
            if kind == 'mixed':
                raise ValueError('src_to must be provided when using a '
                                 'mixed source space')
        else:
            surf_offset = 2 if src_to.kind == 'mixed' else 0
            # All of our computations are in RAS (like img.affine), so we need
            # to get the transformation from RAS to the source space
            # subsampling of vox (src), not MRI (FreeSurfer surface RAS) to src
            src_ras_t = np.dot(src_to[-1]['mri_ras_t']['trans'],
                               src_to[-1]['src_mri_t']['trans'])
            src_ras_t[:3] *= 1e3
            src_data['to_vox_map'] = (src_to[-1]['shape'], src_ras_t)
            vertices_to_vol = [s['vertno'] for s in src_to[surf_offset:]]
            zooms_src_to = np.diag(src_to[-1]['src_mri_t']['trans'])[:3] * 1000
            assert (zooms_src_to[0] == zooms_src_to).all()
            zooms_src_to = tuple(zooms_src_to)

        # pre-compute non-linear morph
        zooms = _check_zooms(mri_from, zooms, zooms_src_to)
        shape, zooms, affine, pre_affine, sdr_morph = _compute_morph_sdr(
            mri_from, mri_to, niter_affine, niter_sdr, zooms)

    if kind in ('surface', 'mixed'):
        logger.info('surface source space present ...')
        vertices_from = src_data['vertices_from']
        if sparse:
            if spacing is not None:
                raise ValueError('spacing must be set to None if '
                                 'sparse=True.')
            if xhemi:
                raise ValueError('xhemi=True can only be used with '
                                 'sparse=False')
            vertices_to_surf, morph_mat = _compute_sparse_morph(
                vertices_from, subject_from, subject_to, subjects_dir)
        else:
            if src_to is not None:
                assert src_to.kind in ('surface', 'mixed')
                vertices_to_surf = [s['vertno'].copy() for s in src_to[:2]]
            else:
                vertices_to_surf = grade_to_vertices(
                    subject_to, spacing, subjects_dir, 1)
            morph_mat = _compute_morph_matrix(
                subject_from=subject_from, subject_to=subject_to,
                vertices_from=vertices_from, vertices_to=vertices_to_surf,
                subjects_dir=subjects_dir, smooth=smooth, warn=warn,
                xhemi=xhemi)
            n_verts = sum(len(v) for v in vertices_to_surf)
            assert morph_mat.shape[0] == n_verts

    vertices_to = vertices_to_surf + vertices_to_vol
    if src_to is not None:
        assert len(vertices_to) == len(src_to)
    morph = SourceMorph(subject_from, subject_to, kind, zooms,
                        niter_affine, niter_sdr, spacing, smooth, xhemi,
                        morph_mat, vertices_to, shape, affine,
                        pre_affine, sdr_morph, src_data)
    logger.info('[done]')
    return morph


def _compute_sparse_morph(vertices_from, subject_from, subject_to,
                          subjects_dir=None):
    """Get nearest vertices from one subject to another."""
    maps = read_morph_map(subject_to, subject_from, subjects_dir)
    cnt = 0
    vertices = list()
    cols = list()
    for verts, map_hemi in zip(vertices_from, maps):
        vertno_h = _sparse_argmax_nnz_row(map_hemi[verts])
        order = np.argsort(vertno_h)
        cols.append(cnt + order)
        vertices.append(vertno_h[order])
        cnt += len(vertno_h)
    cols = np.concatenate(cols)
    rows = np.arange(len(cols))
    data = np.ones(len(cols))
    morph_mat = sparse.coo_matrix((data, (rows, cols)),
                                  shape=(len(cols), len(cols))).tocsr()
    return vertices, morph_mat


_SOURCE_MORPH_ATTRIBUTES = [  # used in writing
    'subject_from', 'subject_to', 'kind', 'zooms', 'niter_affine', 'niter_sdr',
    'spacing', 'smooth', 'xhemi', 'morph_mat', 'vertices_to',
    'shape', 'affine', 'pre_affine', 'sdr_morph', 'src_data', 'verbose']


@fill_doc
class SourceMorph(object):
    """Morph source space data from one subject to another.

    .. note:: This class should not be instantiated directly.
              Use :func:`mne.compute_source_morph` instead.

    Parameters
    ----------
    subject_from : str | None
        Name of the subject from which to morph as named in the SUBJECTS_DIR.
    subject_to : str | array | list of array
        Name of the subject on which to morph as named in the SUBJECTS_DIR.
        The default is 'fsaverage'. If morphing a volume source space,
        subject_to can be the path to a MRI volume. Can also be a list of
        two arrays if morphing to hemisphere surfaces.
    kind : str | None
        Kind of source estimate. E.g. 'volume' or 'surface'.
    zooms : float | tuple
        See :func:`mne.compute_source_morph`.
    niter_affine : tuple of int
        Number of levels (``len(niter_affine)``) and number of
        iterations per level - for each successive stage of iterative
        refinement - to perform the affine transform.
    niter_sdr : tuple of int
        Number of levels (``len(niter_sdr)``) and number of
        iterations per level - for each successive stage of iterative
        refinement - to perform the Symmetric Diffeomorphic Registration (sdr)
        transform :footcite:`AvantsEtAl2008`.
    spacing : int | list | None
        See :func:`mne.compute_source_morph`.
    smooth : int | str | None
        See :func:`mne.compute_source_morph`.
    xhemi : bool
        Morph across hemisphere.
    morph_mat : scipy.sparse.csr_matrix
        The sparse surface morphing matrix for spherical surface
        based morphing :footcite:`GreveEtAl2013`.
    vertices_to : list of ndarray
        The destination surface vertices.
    shape : tuple
        The volume MRI shape.
    affine : ndarray
        The volume MRI affine.
    pre_affine : instance of dipy.align.imaffine.AffineMap
        The :class:`dipy.align.imaffine.AffineMap` transformation that is
        applied before the before ``sdr_morph``.
    sdr_morph : instance of dipy.align.imwarp.DiffeomorphicMap
        The :class:`dipy.align.imwarp.DiffeomorphicMap` that applies the
        the symmetric diffeomorphic registration (SDR) morph.
    src_data : dict
        Additional source data necessary to perform morphing.
    %(verbose)s

    Notes
    -----
    .. versionadded:: 0.17

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, subject_from, subject_to, kind, zooms,
                 niter_affine, niter_sdr, spacing, smooth, xhemi,
                 morph_mat, vertices_to, shape,
                 affine, pre_affine, sdr_morph, src_data, verbose=None):
        # universal
        self.subject_from = subject_from
        self.subject_to = subject_to
        self.kind = kind
        # vol input
        self.zooms = zooms
        self.niter_affine = niter_affine
        self.niter_sdr = niter_sdr
        # surf input
        self.spacing = spacing
        self.smooth = smooth
        self.xhemi = xhemi
        # surf computed
        self.morph_mat = morph_mat
        # vol computed
        self.shape = shape
        self.affine = affine
        self.sdr_morph = sdr_morph
        self.pre_affine = pre_affine
        # used by both
        self.src_data = src_data
        self.verbose = verbose
        # compute vertices_to here (partly for backward compat and no src
        # provided)
        if vertices_to is None or len(vertices_to) == 0 and kind == 'volume':
            assert src_data['to_vox_map'] is None
            vertices_to = self._get_vol_vertices_to_nz()
        self.vertices_to = vertices_to

    @property
    def _vol_vertices_from(self):
        assert isinstance(self.src_data['inuse'], list)
        vertices_from = [np.where(in_)[0] for in_ in self.src_data['inuse']]
        return vertices_from

    def _get_vol_vertices_to_nz(self):
        logger.info('Computing nonzero vertices after morph ...')
        n_vertices = sum(len(v) for v in self._vol_vertices_from)
        ones = np.ones((n_vertices, 1))
        return [np.where(self._morph_one_vol(ones))[0]]

    @verbose
    def apply(self, stc_from, output='stc', mri_resolution=False,
              mri_space=None, verbose=None):
        """Morph source space data.

        Parameters
        ----------
        stc_from : VolSourceEstimate | VolVectorSourceEstimate | SourceEstimate | VectorSourceEstimate
            The source estimate to morph.
        output : str
            Can be 'stc' (default) or possibly 'nifti1', or 'nifti2'
            when working with a volume source space defined on a regular
            grid.
        mri_resolution : bool | tuple | int | float
            If True the image is saved in MRI resolution. Default False.
            WARNING: if you have many time points the file produced can be
            huge. The default is mri_resolution=False.
        mri_space : bool | None
            Whether the image to world registration should be in mri space. The
            default (None) is mri_space=mri_resolution.
        %(verbose_meth)s

        Returns
        -------
        stc_to : VolSourceEstimate | SourceEstimate | VectorSourceEstimate | Nifti1Image | Nifti2Image
            The morphed source estimates.
        """  # noqa: E501
        _validate_type(output, str, 'output')
        _validate_type(stc_from, _BaseSourceEstimate, 'stc_from',
                       'source estimate')
        if isinstance(stc_from, _BaseSurfaceSourceEstimate):
            allowed_kinds = ('stc',)
            extra = 'when stc is a surface source estimate'
        else:
            allowed_kinds = ('stc', 'nifti1', 'nifti2')
            extra = ''
        _check_option('output', output, allowed_kinds, extra)
        stc = copy.deepcopy(stc_from)

        mri_space = mri_resolution if mri_space is None else mri_space
        if stc.subject is None:
            stc.subject = self.subject_from
        if self.subject_from is None:
            self.subject_from = stc.subject
        if stc.subject != self.subject_from:
            raise ValueError('stc_from.subject and '
                             'morph.subject_from must match. (%s != %s)' %
                             (stc.subject, self.subject_from))
        out = _apply_morph_data(self, stc)
        if output != 'stc':  # convert to volume
            out = _morphed_stc_as_volume(
                self, out, mri_resolution=mri_resolution, mri_space=mri_space,
                output=output)
        return out

    def _morph_one_vol(self, one):
        # prepare data to be morphed
        # here we use mri_resolution=True, mri_space=True because
        # we will slice afterward
        from dipy.align.reslice import reslice
        from nibabel.processing import resample_from_to
        from nibabel.spatialimages import SpatialImage
        assert isinstance(one, np.ndarray)
        assert one.shape[1] == 1
        stc_one = VolSourceEstimate(
            one, self._vol_vertices_from, 0., 1., self.subject_from)
        img_to = _interpolate_data(stc_one, self, mri_resolution=True,
                                   mri_space=True, output='nifti1')
        img_to = _get_img_fdata(img_to)
        assert img_to.ndim == 4 and img_to.shape[-1] == 1
        img_to = img_to[:, :, :, 0]

        attrs = ('real', 'imag') if np.iscomplexobj(img_to) else ('real',)
        img_complex = 0.
        for attr in attrs:
            # reslice to match morph
            img_real, _ = reslice(
                getattr(img_to, attr), self.affine,
                _get_zooms_orig(self), self.zooms)

            # morph data
            img_real = self.pre_affine.transform(img_real)
            if self.sdr_morph is not None:
                img_real = self.sdr_morph.transform(img_real)

            # subselect the correct cube if src_to is provided
            if self.src_data['to_vox_map'] is not None:
                # order=0 (nearest) should be fine since it's just subselecting
                img_real = SpatialImage(img_real, self.affine)
                img_real = resample_from_to(
                    img_real, self.src_data['to_vox_map'], 1)
                img_real = _get_img_fdata(img_real)

            # combine real and complex parts
            if attr == 'real':
                img_complex = img_complex + img_real
            else:
                img_complex = img_complex + 1j * img_real
        img_to = img_complex

        # reshape to nvoxel x nvol:
        # in the MNE definition of volume source spaces,
        # x varies fastest, then y, then z, so we need order='F' here
        img_to = img_to.reshape(-1, order='F')
        return img_to

    def __repr__(self):  # noqa: D105
        s = u"%s" % self.kind
        s += u", %s -> %s" % (self.subject_from, self.subject_to)
        if self.kind == 'volume':
            s += ", zooms : {}".format(self.zooms)
            s += ", niter_affine : {}".format(self.niter_affine)
            s += ", niter_sdr : {}".format(self.niter_sdr)
        elif self.kind in ('surface', 'vector'):
            s += ", spacing : {}".format(self.spacing)
            s += ", smooth : %s" % self.smooth
            s += ", xhemi" if self.xhemi else ""

        return "<SourceMorph | %s>" % s

    @verbose
    def save(self, fname, overwrite=False, verbose=None):
        """Save the morph for source estimates to a file.

        Parameters
        ----------
        fname : str
            The stem of the file name. '-morph.h5' will be added if fname does
            not end with '.h5'.
        overwrite : bool
            If True, overwrite existing file.
        %(verbose_meth)s
        """
        fname = _check_fname(fname, overwrite=overwrite, must_exist=False)
        if not fname.endswith('.h5'):
            fname = '%s-morph.h5' % fname

        out_dict = {k: getattr(self, k) for k in _SOURCE_MORPH_ATTRIBUTES}
        for key in ('pre_affine', 'sdr_morph'):  # classes
            if out_dict[key] is not None:
                out_dict[key] = out_dict[key].__dict__
        write_hdf5(fname, out_dict, overwrite=overwrite)


def _check_zooms(mri_from, zooms, zooms_src_to):
    # use voxel size of mri_from
    if isinstance(zooms, str) and zooms == 'auto':
        zooms = zooms_src_to if zooms_src_to is not None else 5.
    if zooms is None:
        zooms = mri_from.header.get_zooms()[:3]
    zooms = np.atleast_1d(zooms).astype(float)
    if zooms.shape == (1,):
        zooms = np.repeat(zooms, 3)
    if zooms.shape != (3,):
        raise ValueError('zooms must be None, a singleton, or have shape (3,),'
                         ' got shape %s' % (zooms.shape,))
    zooms = tuple(zooms)
    if zooms_src_to is not None:
        if not np.allclose(zooms_src_to, zooms, atol=1e-6):
            raise ValueError('If src_to is provided, zooms should be "auto" '
                             'or match the src_to zooms (%s), got %s'
                             % (zooms_src_to, zooms))
        zooms = zooms_src_to
    return zooms


###############################################################################
# I/O
def _check_subject_src(subject, src, name='subject_from', src_name='src'):
    if isinstance(src, str):
        subject_check = src
    elif src is None:  # assume it's correct although dangerous but unlikely
        subject_check = subject
    else:
        subject_check = src._subject
        if subject_check is None:
            warn('The source space does not contain the subject name, we '
                 'recommend regenerating the source space (and forward / '
                 'inverse if applicable) for better code reliability')
    if subject is None:
        subject = subject_check
    elif subject_check is not None and subject != subject_check:
        raise ValueError('%s does not match %s subject (%s != %s)'
                         % (name, src_name, subject, subject_check))
    if subject is None:
        raise ValueError('%s could not be inferred from %s, it must be '
                         'specified' % (name, src_name))
    return subject


def read_source_morph(fname):
    """Load the morph for source estimates from a file.

    Parameters
    ----------
    fname : str
        Full filename including path.

    Returns
    -------
    source_morph : instance of SourceMorph
        The loaded morph.
    """
    vals = read_hdf5(fname)
    if vals['pre_affine'] is not None:  # reconstruct
        from dipy.align.imaffine import AffineMap
        affine = vals['pre_affine']
        vals['pre_affine'] = AffineMap(None)
        vals['pre_affine'].__dict__ = affine
    if vals['sdr_morph'] is not None:
        from dipy.align.imwarp import DiffeomorphicMap
        morph = vals['sdr_morph']
        vals['sdr_morph'] = DiffeomorphicMap(None, [])
        vals['sdr_morph'].__dict__ = morph
    # Backward compat with when it used to be a list
    if isinstance(vals['vertices_to'], np.ndarray):
        vals['vertices_to'] = [vals['vertices_to']]
    # Backward compat with when it used to be a single array
    if isinstance(vals['src_data'].get('inuse', None), np.ndarray):
        vals['src_data']['inuse'] = [vals['src_data']['inuse']]
    return SourceMorph(**vals)


###############################################################################
# Helper functions for SourceMorph methods
def _check_dep(nibabel='2.1.0', dipy='0.10.1'):
    """Check dependencies."""
    for lib, ver in zip(['nibabel', 'dipy'],
                        [nibabel, dipy]):
        passed = True if not ver else check_version(lib, ver)

        if not passed:
            raise ImportError('%s %s or higher must be correctly '
                              'installed and accessible from Python' % (lib,
                                                                        ver))


def _morphed_stc_as_volume(morph, stc, mri_resolution, mri_space, output):
    """Return volume source space as Nifti1Image and/or save to disk."""
    assert isinstance(stc, _BaseVolSourceEstimate)  # should be guaranteed
    if stc._data_ndim == 3:
        stc = stc.magnitude()
    _check_dep(nibabel='2.1.0', dipy=False)

    NiftiImage, NiftiHeader = _triage_output(output)

    # if MRI resolution is set manually as a single value, convert to tuple
    if isinstance(mri_resolution, (int, float)):
        # use iso voxel size
        new_zooms = (float(mri_resolution),) * 3
    elif isinstance(mri_resolution, tuple):
        new_zooms = mri_resolution
    # if full MRI resolution, compute zooms from shape and MRI zooms
    if isinstance(mri_resolution, bool):
        new_zooms = _get_zooms_orig(morph) if mri_resolution else None

    # create header
    hdr = NiftiHeader()
    hdr.set_xyzt_units('mm', 'msec')
    hdr['pixdim'][4] = 1e3 * stc.tstep

    # setup empty volume
    if morph.src_data['to_vox_map'] is not None:
        shape = morph.src_data['to_vox_map'][0]
        affine = morph.src_data['to_vox_map'][1]
    else:
        shape = morph.shape
        affine = morph.affine
    assert stc.data.ndim == 2
    n_times = stc.data.shape[1]
    img = np.zeros((np.prod(shape), n_times))
    img[stc.vertices[0], :] = stc.data
    img = img.reshape(shape + (n_times,), order='F')  # match order='F' above
    del shape

    # make nifti from data
    with warnings.catch_warnings():  # nibabel<->numpy warning
        img = NiftiImage(img, affine, header=hdr)

    # reslice in case of manually defined voxel size
    zooms = morph.zooms[:3]
    if new_zooms is not None:
        from dipy.align.reslice import reslice
        new_zooms = new_zooms[:3]
        img, affine = reslice(_get_img_fdata(img),
                              img.affine,  # MRI to world registration
                              zooms,  # old voxel size in mm
                              new_zooms)  # new voxel size in mm
        with warnings.catch_warnings():  # nibabel<->numpy warning
            img = NiftiImage(img, affine)
        zooms = new_zooms

    #  set zooms in header
    img.header.set_zooms(tuple(zooms) + (1,))
    return img


def _get_src_data(src, mri_resolution=True):
    # copy data to avoid conflicts
    _validate_type(
        src, (_BaseSurfaceSourceEstimate, 'path-like', SourceSpaces),
        'src', 'source space or surface source estimate')
    if isinstance(src, _BaseSurfaceSourceEstimate):
        src_t = [dict(vertno=src.vertices[0]), dict(vertno=src.vertices[1])]
        src_kind = 'surface'
        src_subject = src.subject
    else:
        src_t = _ensure_src(src).copy()
        src_kind = src_t.kind
        src_subject = src_t._subject
    del src
    _check_option('src kind', src_kind, ('surface', 'volume', 'mixed'))

    # extract all relevant data for volume operations
    src_data = dict()
    if src_kind in ('volume', 'mixed'):
        use_src = src_t[-1]
        shape = use_src['shape']
        start = 0 if src_kind == 'volume' else 2
        for si, s in enumerate(src_t[start:], start):
            if s['interpolator'] is None:
                if mri_resolution:
                    raise RuntimeError(
                        'MRI interpolator not present in src[%d], '
                        'cannot use mri_resolution=True' % (si,))
                interpolator = None
                break
        else:
            interpolator = sum((s['interpolator'] for s in src_t[start:]), 0.)
        inuses = [s['inuse'] for s in src_t[start:]]
        src_data.update({'src_shape': (shape[2], shape[1], shape[0]),  # SAR
                         'src_affine_vox': use_src['vox_mri_t']['trans'],
                         'src_affine_src': use_src['src_mri_t']['trans'],
                         'src_affine_ras': use_src['mri_ras_t']['trans'],
                         'src_shape_full': (  # SAR
                             use_src['mri_height'], use_src['mri_depth'],
                             use_src['mri_width']),
                         'interpolator': interpolator,
                         'inuse': inuses,
                         'to_vox_map': None,
                         })
    if src_kind in ('surface', 'mixed'):
        src_data.update(vertices_from=[s['vertno'].copy() for s in src_t[:2]])

    # delete copy
    return src_data, src_kind, src_subject


def _triage_output(output):
    _check_option('output', output, ['nifti', 'nifti1', 'nifti2'])
    if output in ('nifti', 'nifti1'):
        from nibabel import (Nifti1Image as NiftiImage,
                             Nifti1Header as NiftiHeader)
    else:
        assert output == 'nifti2'
        from nibabel import (Nifti2Image as NiftiImage,
                             Nifti2Header as NiftiHeader)
    return NiftiImage, NiftiHeader


def _csr_dot(csr, other, result):
    # Adapted from SciPy to allow "out" specification
    assert isinstance(csr, sparse.csr_matrix)
    M, N = csr.shape
    n_vecs = other.shape[1]  # number of column vectors
    assert result.shape == (M, n_vecs)
    sparse._sparsetools.csr_matvecs(
        M, N, n_vecs, csr.indptr, csr.indices, csr.data,
        other.ravel(), result.ravel())
    return result


def _interpolate_data(stc, morph, mri_resolution, mri_space, output):
    """Interpolate source estimate data to MRI."""
    _check_dep(nibabel='2.1.0', dipy=False)
    NiftiImage, NiftiHeader = _triage_output(output)
    _validate_type(stc, _BaseVolSourceEstimate, 'stc',
                   'volume source estimate')
    assert morph.kind in ('volume', 'mixed')

    voxel_size_defined = False

    if isinstance(mri_resolution, (int, float)) and not isinstance(
            mri_resolution, bool):
        # use iso voxel size
        mri_resolution = (float(mri_resolution),) * 3

    if isinstance(mri_resolution, tuple):
        _check_dep(nibabel=False, dipy='0.10.1')  # nibabel was already checked
        from dipy.align.reslice import reslice

        voxel_size = mri_resolution
        voxel_size_defined = True
        mri_resolution = True

    # if data wasn't morphed yet - necessary for call of
    # stc_unmorphed.as_volume. Since only the shape of src is known, it cannot
    # be resliced to a given voxel size without knowing the original.
    if isinstance(morph, SourceSpaces):
        assert morph.kind in ('volume', 'mixed')
        offset = 2 if morph.kind == 'mixed' else 0
        if voxel_size_defined:
            raise ValueError(
                "Cannot infer original voxel size for reslicing... "
                "set mri_resolution to boolean value or apply morph first.")
        # Now deal with the fact that we may have multiple sub-volumes
        inuse = [s['inuse'] for s in morph[offset:]]
        src_shape = [s['shape'] for s in morph[offset:]]
        assert len(set(map(tuple, src_shape))) == 1
        src_subject = morph._subject
        morph = BunchConst(src_data=_get_src_data(morph, mri_resolution)[0])
    else:
        # Make a list as we may have many inuse when using multiple sub-volumes
        inuse = morph.src_data['inuse']
        src_subject = morph.subject_from
    assert isinstance(inuse, list)
    if stc.subject is not None:
        _check_subject_src(stc.subject, src_subject, 'stc.subject')

    n_times = stc.data.shape[1]
    shape = morph.src_data['src_shape'][::-1] + (n_times,)  # SAR->RAST
    dtype = np.complex128 if np.iscomplexobj(stc.data) else np.float64
    # order='F' so that F-order flattening is faster
    vols = np.zeros((np.prod(shape[:3]), shape[3]), dtype=dtype, order='F')
    n_vertices_seen = 0
    for this_inuse in inuse:
        this_inuse = this_inuse.astype(bool)
        n_vertices = np.sum(this_inuse)
        stc_slice = slice(n_vertices_seen, n_vertices_seen + n_vertices)
        vols[this_inuse] = stc.data[stc_slice]
        n_vertices_seen += n_vertices

    # use mri resolution as represented in src
    if mri_resolution:
        if morph.src_data['interpolator'] is None:
            raise RuntimeError(
                'Cannot morph with mri_resolution when add_interpolator=False '
                'was used with setup_volume_source_space')
        shape = morph.src_data['src_shape_full'][::-1] + (n_times,)
        vols = _csr_dot(
            morph.src_data['interpolator'], vols,
            np.zeros((np.prod(shape[:3]), shape[3]), dtype=dtype, order='F'))

    # reshape back to proper shape
    vols = np.reshape(vols, shape, order='F')

    # set correct space
    if mri_resolution:
        affine = morph.src_data['src_affine_vox']
    else:
        affine = morph.src_data['src_affine_src']

    if mri_space:
        affine = np.dot(morph.src_data['src_affine_ras'], affine)

    affine[:3] *= 1e3

    # pre-define header
    header = NiftiHeader()
    header.set_xyzt_units('mm', 'msec')
    header['pixdim'][4] = 1e3 * stc.tstep

    with warnings.catch_warnings():  # nibabel<->numpy warning
        img = NiftiImage(vols, affine, header=header)

    # if a specific voxel size was targeted (only possible after morphing)
    if voxel_size_defined:
        # reslice mri
        img, img_affine = reslice(
            _get_img_fdata(img), img.affine, _get_zooms_orig(morph),
            voxel_size)
        with warnings.catch_warnings():  # nibabel<->numpy warning
            img = NiftiImage(img, img_affine, header=header)

    return img


###############################################################################
# Morph for VolSourceEstimate

def _compute_r2(a, b):
    return 100 * (a.ravel() @ b.ravel()) / \
        (np.linalg.norm(a) * np.linalg.norm(b))


def _compute_morph_sdr(mri_from, mri_to, niter_affine, niter_sdr, zooms):
    """Get a matrix that morphs data from one subject to another."""
    import nibabel as nib
    with np.testing.suppress_warnings():
        from dipy.align import imaffine, imwarp, metrics, transforms
    from dipy.align.reslice import reslice

    logger.info('Computing nonlinear Symmetric Diffeomorphic Registration...')

    # reslice mri_from
    mri_from_res, mri_from_res_affine = reslice(
        _get_img_fdata(mri_from), mri_from.affine,
        mri_from.header.get_zooms()[:3], zooms)

    with warnings.catch_warnings():  # nibabel<->numpy warning
        mri_from = nib.Nifti1Image(mri_from_res, mri_from_res_affine)

    # reslice mri_to
    mri_to_res, mri_to_res_affine = reslice(
        _get_img_fdata(mri_to), mri_to.affine, mri_to.header.get_zooms()[:3],
        zooms)

    with warnings.catch_warnings():  # nibabel<->numpy warning
        mri_to = nib.Nifti1Image(mri_to_res, mri_to_res_affine)

    affine = mri_to.affine
    mri_to = _get_img_fdata(mri_to).copy()  # to ndarray
    mri_to /= mri_to.max()
    mri_from_affine = mri_from.affine  # get mri_from to world transform
    mri_from = _get_img_fdata(mri_from)  # to ndarray
    mri_from /= mri_from.max()  # normalize

    # compute center of mass
    c_of_mass = imaffine.transform_centers_of_mass(
        mri_to, affine, mri_from, mri_from_affine)

    # set up Affine Registration
    affreg = imaffine.AffineRegistration(
        metric=imaffine.MutualInformationMetric(nbins=32),
        level_iters=list(niter_affine),
        sigmas=[3.0, 1.0, 0.0],
        factors=[4, 2, 1])

    # translation
    logger.info('Optimizing translation:')
    with wrapped_stdout(indent='    ', cull_newlines=True):
        translation = affreg.optimize(
            mri_to, mri_from, transforms.TranslationTransform3D(), None,
            affine, mri_from_affine, starting_affine=c_of_mass.affine)

    # rigid body transform (translation + rotation)
    logger.info('Optimizing rigid-body:')
    with wrapped_stdout(indent='    ', cull_newlines=True):
        rigid = affreg.optimize(
            mri_to, mri_from, transforms.RigidTransform3D(), None,
            affine, mri_from_affine, starting_affine=translation.affine)
    mri_from_to = rigid.transform(mri_from)
    dist = np.linalg.norm(rigid.affine[:3, 3])
    angle = np.rad2deg(_angle_between_quats(
        np.zeros(3), rot_to_quat(rigid.affine[:3, :3])))

    logger.info(f'    Translation: {dist:6.1f} mm')
    logger.info(f'    Rotation:    {angle:6.1f}°')
    logger.info(f'    R²:          {_compute_r2(mri_to, mri_from_to):6.1f}%')

    # affine transform (translation + rotation + scaling)
    logger.info('Optimizing full affine:')
    with wrapped_stdout(indent='    ', cull_newlines=True):
        pre_affine = affreg.optimize(
            mri_to, mri_from, transforms.AffineTransform3D(), None,
            affine, mri_from_affine, starting_affine=rigid.affine)
    mri_from_to = pre_affine.transform(mri_from)
    logger.info(f'    R²:          {_compute_r2(mri_to, mri_from_to):6.1f}%')

    # SDR
    shape = tuple(pre_affine.domain_shape)
    if len(niter_sdr):
        sdr = imwarp.SymmetricDiffeomorphicRegistration(
            metrics.CCMetric(3), list(niter_sdr))
        logger.info('Optimizing SDR:')
        with wrapped_stdout(indent='    ', cull_newlines=True):
            sdr_morph = sdr.optimize(mri_to, pre_affine.transform(mri_from))
        assert shape == tuple(sdr_morph.domain_shape)  # should be tuple of int
        mri_from_to = sdr_morph.transform(mri_from_to)
    else:
        sdr_morph = None

    logger.info(f'    R²:          {_compute_r2(mri_to, mri_from_to):6.1f}%')

    # To debug to_vox_map, this can be used:
    # from nibabel.processing import resample_from_to
    # mri_from_to = sdr_morph.transform(pre_affine.transform(mri_from))
    # mri_from_to = nib.Nifti1Image(mri_from_to, affine)
    # fig1 = mri_from_to.orthoview()
    # mri_from_to_cut = resample_from_to(mri_from_to, to_vox_map, 1)
    # fig2 = mri_from_to_cut.orthoview()

    return shape, zooms, affine, pre_affine, sdr_morph


def _compute_morph_matrix(subject_from, subject_to, vertices_from, vertices_to,
                          smooth=None, subjects_dir=None, warn=True,
                          xhemi=False):
    """Compute morph matrix."""
    logger.info('Computing morph matrix...')
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)

    tris = _get_subject_sphere_tris(subject_from, subjects_dir)
    maps = read_morph_map(subject_from, subject_to, subjects_dir, xhemi)

    # morph the data

    morpher = []
    for hemi_to in range(2):  # iterate over to / block-rows of CSR matrix
        hemi_from = (1 - hemi_to) if xhemi else hemi_to
        morpher.append(_hemi_morph(
            tris[hemi_from], vertices_to[hemi_to], vertices_from[hemi_from],
            smooth, maps[hemi_from], warn))

    shape = (sum(len(v) for v in vertices_to),
             sum(len(v) for v in vertices_from))
    data = [m.data for m in morpher]
    indices = [m.indices.copy() for m in morpher]
    indptr = [m.indptr.copy() for m in morpher]
    # column indices need to be adjusted
    indices[0 if xhemi else 1] += len(vertices_from[0])
    indices = np.concatenate(indices)
    # row index pointers need to be adjusted
    indptr[1] = indptr[1][1:] + len(data[0])
    indptr = np.concatenate(indptr)
    # data does not need to be adjusted
    data = np.concatenate(data)
    # this is equivalent to morpher = sparse_block_diag(morpher).tocsr(),
    # but works for xhemi mode
    morpher = sparse.csr_matrix((data, indices, indptr), shape=shape)
    logger.info('[done]')
    return morpher


def _hemi_morph(tris, vertices_to, vertices_from, smooth, maps, warn):
    if len(vertices_from) == 0:
        return sparse.csr_matrix((len(vertices_to), 0))
    e = mesh_edges(tris)
    e.data[e.data == 2] = 1
    n_vertices = e.shape[0]
    e = e + sparse.eye(n_vertices)
    m = sparse.eye(len(vertices_from), format='csr')
    mm = _morph_buffer(m, vertices_from, e, smooth, n_vertices,
                       vertices_to, maps, warn=warn)
    assert mm.shape == (len(vertices_to), len(vertices_from))
    return mm


@verbose
def grade_to_vertices(subject, grade, subjects_dir=None, n_jobs=1,
                      verbose=None):
    """Convert a grade to source space vertices for a given subject.

    Parameters
    ----------
    subject : str
        Name of the subject.
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
    %(subjects_dir)s
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    vertices : list of array of int
        Vertex numbers for LH and RH.
    """
    _validate_type(grade, (list, 'int-like', None), 'grade')
    # add special case for fsaverage for speed
    if subject == 'fsaverage' and isinstance(grade, int) and grade == 5:
        return [np.arange(10242), np.arange(10242)]
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)

    spheres_to = [op.join(subjects_dir, subject, 'surf',
                          xh + '.sphere.reg') for xh in ['lh', 'rh']]
    lhs, rhs = [read_surface(s)[0] for s in spheres_to]

    if grade is not None:  # fill a subset of vertices
        if isinstance(grade, list):
            if not len(grade) == 2:
                raise ValueError('grade as a list must have two elements '
                                 '(arrays of output vertices)')
            vertices = grade
        else:
            grade = _ensure_int(grade)
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


def _surf_nearest(vertices, adj_mat):
    from scipy.sparse.csgraph import dijkstra
    if not check_version('scipy', '1.3'):
        raise ValueError('scipy >= 1.3 is required to use nearest smoothing, '
                         'consider upgrading SciPy or using a different '
                         'smoothing value')
    # Vertices can be out of order, so sort them to start ...
    order = np.argsort(vertices)
    vertices = vertices[order]
    _, _, sources = dijkstra(adj_mat, False, indices=vertices, min_only=True,
                             return_predecessors=True)
    col = np.searchsorted(vertices, sources)
    # ... then get things back to the correct configuration.
    col = order[col]
    row = np.arange(len(col))
    data = np.ones(len(col))
    mat = sparse.coo_matrix((data, (row, col)))
    assert mat.shape == (adj_mat.shape[0], len(vertices)), mat.shape
    return mat


def _morph_buffer(data, idx_use, e, smooth, n_vertices, nearest, maps,
                  warn=True):
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
        Vertices on the reference surface to use.
    maps : sparse matrix
        Morph map from one subject to the other.
    warn : bool
        If True, warn if not all vertices were used.
    %(verbose)s The default
        is verbose=None.

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
                n_vertices=n_vertices, nearest=nearest, maps=maps, warn=warn)
        return data_morphed

    n_iter = 99  # max nb of smoothing iterations (minus one)
    _validate_type(smooth, ('int-like', str, None), 'smooth')
    if isinstance(smooth, str):
        _check_option('smooth', smooth, ('nearest',),
                      extra=' when used as a string.')
    if smooth is not None:
        if smooth == 'nearest':
            return (maps[nearest, :] * _surf_nearest(idx_use, e)) * data
        smooth = _ensure_int(smooth)
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
    idx = np.empty(n_rows, dtype=np.int64)
    for k in range(n_rows):
        row = csr_mat[k].tocoo()
        idx[k] = row.col[np.argmax(row.data)]
    return idx


def _get_subject_sphere_tris(subject, subjects_dir):
    spheres = [op.join(subjects_dir, subject, 'surf',
                       xh + '.sphere.reg') for xh in ['lh', 'rh']]
    tris = [read_surface(s)[1] for s in spheres]
    return tris


###############################################################################
# Apply morph to source estimate
def _get_zooms_orig(morph):
    """Compute src zooms from morph zooms, morph shape and src shape."""
    # zooms_to = zooms_from / shape_to * shape_from for each spatial dimension
    return [mz / ss * ms for mz, ms, ss in
            zip(morph.zooms, morph.shape, morph.src_data['src_shape_full'])]


def _check_vertices_match(v1, v2, name):
    if not np.array_equal(v1, v2):
        ext = ''
        if np.in1d(v2, v1).all():
            ext = ' Vertices were likely excluded during forward computation.'
        raise ValueError(
            'vertices do not match between morph (%s) and stc (%s) for %s:\n%s'
            '\n%s\nPerhaps src_to=fwd["src"] needs to be passed when calling '
            'compute_source_morph.%s' % (len(v1), len(v2), name, v1, v2, ext))


def _apply_morph_data(morph, stc_from):
    """Morph a source estimate from one subject to another."""
    if stc_from.subject is not None and stc_from.subject != morph.subject_from:
        raise ValueError('stc.subject (%s) != morph.subject_from (%s)'
                         % (stc_from.subject, morph.subject_from))
    _check_option('morph.kind', morph.kind, ('surface', 'volume', 'mixed'))
    if morph.kind == 'surface':
        _validate_type(stc_from, _BaseSurfaceSourceEstimate, 'stc_from',
                       'volume source estimate when using a surface morph')
    elif morph.kind == 'volume':
        _validate_type(stc_from, _BaseVolSourceEstimate, 'stc_from',
                       'surface source estimate when using a volume morph')
    else:
        assert morph.kind == 'mixed'  # can handle any
        _validate_type(stc_from, _BaseSourceEstimate, 'stc_from',
                       'source estimate when using a mixed source morph')

    # figure out what to actually morph
    do_vol = not isinstance(stc_from, _BaseSurfaceSourceEstimate)
    do_surf = not isinstance(stc_from, _BaseVolSourceEstimate)

    vol_src_offset = 2 if do_surf else 0
    from_surf_stop = sum(len(v) for v in stc_from.vertices[:vol_src_offset])
    to_surf_stop = sum(len(v) for v in morph.vertices_to[:vol_src_offset])
    from_vol_stop = stc_from.data.shape[0]
    vertices_to = morph.vertices_to
    if morph.kind == 'mixed':
        vertices_to = vertices_to[0 if do_surf else 2:None if do_vol else 2]
    to_vol_stop = sum(len(v) for v in vertices_to)

    data_from = np.reshape(stc_from.data, (stc_from.data.shape[0], -1))
    n_times = data_from.shape[1]  # oris treated as times
    data = np.empty((to_vol_stop, n_times), stc_from.data.dtype)
    to_used = np.zeros(data.shape[0], bool)
    from_used = np.zeros(data_from.shape[0], bool)
    if do_vol:
        stc_from_vertices = stc_from.vertices[vol_src_offset:]
        vertices_from = morph._vol_vertices_from
        for ii, (v1, v2) in enumerate(zip(vertices_from, stc_from_vertices)):
            _check_vertices_match(v1, v2, 'volume[%d]' % (ii,))
        vol_verts = np.concatenate(
            morph.vertices_to[0 if morph.kind == 'volume' else 2:])
        from_sl = slice(from_surf_stop, from_vol_stop)
        assert not from_used[from_sl].any()
        from_used[from_sl] = True
        to_sl = slice(to_surf_stop, to_vol_stop)
        assert not to_used[to_sl].any()
        to_used[to_sl] = True
        # Loop over time points to save memory
        for k in range(n_times):
            this_data = data_from[from_sl, k:k + 1]
            this_img_to = morph._morph_one_vol(this_data)
            data[to_sl, k] = this_img_to[vol_verts]
    if do_surf:
        for hemi, v1, v2 in zip(('left', 'right'),
                                morph.src_data['vertices_from'],
                                stc_from.vertices[:2]):
            _check_vertices_match(v1, v2, '%s hemisphere' % (hemi,))
        from_sl = slice(0, from_surf_stop)
        assert not from_used[from_sl].any()
        from_used[from_sl] = True
        to_sl = slice(0, to_surf_stop)
        assert not to_used[to_sl].any()
        to_used[to_sl] = True
        data[to_sl] = morph.morph_mat * data_from[from_sl]
    assert to_used.all()
    assert from_used.all()
    data.shape = (data.shape[0],) + stc_from.data.shape[1:]
    klass = stc_from.__class__
    stc_to = klass(data, vertices_to, stc_from.tmin, stc_from.tstep,
                   morph.subject_to)
    return stc_to
