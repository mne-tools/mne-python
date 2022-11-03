# -*- coding: utf-8 -*-
"""Freesurfer handling functions."""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import os.path as op
import numpy as np
from gzip import GzipFile

from .bem import _bem_find_surface, read_bem_surfaces
from .io.constants import FIFF
from .io.meas_info import read_fiducials
from .transforms import (apply_trans, invert_transform, combine_transforms,
                         _ensure_trans, read_ras_mni_t, Transform)
from .surface import read_surface, _read_mri_surface
from .utils import (verbose, _validate_type, _check_fname, _check_option,
                    get_subjects_dir, _require_version, logger, warn)


def _check_subject_dir(subject, subjects_dir):
    """Check that the Freesurfer subject directory is as expected."""
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    for img_name in ('T1', 'brain', 'aseg'):
        if not op.isfile(op.join(subjects_dir, subject, 'mri',
                                 f'{img_name}.mgz')):
            raise ValueError('Freesurfer recon-all subject folder '
                             'is incorrect or improperly formatted, '
                             f'got {op.join(subjects_dir, subject)}')
    return op.join(subjects_dir, subject)


def _get_aseg(aseg, subject, subjects_dir):
    """Check that the anatomical segmentation file exists and load it."""
    _require_version('nibabel', 'load aseg', '2.1.0')
    import nibabel as nib
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    if not aseg.endswith('aseg'):
        raise RuntimeError(
            f'`aseg` file path must end with "aseg", got {aseg}')
    aseg = _check_fname(op.join(subjects_dir, subject, 'mri', aseg + '.mgz'),
                        overwrite='read', must_exist=True)
    aseg = nib.load(aseg)
    aseg_data = np.array(aseg.dataobj)
    return aseg, aseg_data


def _import_nibabel(why='use MRI files'):
    try:
        import nibabel as nib
    except ImportError as exp:
        msg = 'nibabel is required to %s, got:\n%s' % (why, exp)
    else:
        msg = ''
    if msg:
        raise ImportError(msg)
    return nib


def _reorient_image(img, axcodes='RAS'):
    """Reorient an image to a given orientation.

    Parameters
    ----------
    img : instance of SpatialImage
        The MRI image.
    axcodes : tuple | str
        The axis codes specifying the orientation, e.g. "RAS".
        See :func:`nibabel.orientations.aff2axcodes`.

    Returns
    -------
    img_data : ndarray
        The reoriented image data.
    vox_ras_t : ndarray
        The new transform from the new voxels to surface RAS.

    Notes
    -----
    .. versionadded:: 0.24
    """
    import nibabel as nib
    orig_data = np.array(img.dataobj).astype(np.float32)
    # reorient data to RAS
    ornt = nib.orientations.axcodes2ornt(
        nib.orientations.aff2axcodes(img.affine)).astype(int)
    ras_ornt = nib.orientations.axcodes2ornt(axcodes)
    ornt_trans = nib.orientations.ornt_transform(ornt, ras_ornt)
    img_data = nib.orientations.apply_orientation(orig_data, ornt_trans)
    orig_mgh = nib.MGHImage(orig_data, img.affine)
    aff_trans = nib.orientations.inv_ornt_aff(ornt_trans, img.shape)
    vox_ras_t = np.dot(orig_mgh.header.get_vox2ras_tkr(), aff_trans)
    return img_data, vox_ras_t


def _mri_orientation(orientation):
    """Get MRI orientation information from an image.

    Parameters
    ----------
    orientation : str
        Orientation that you want. Can be "axial", "sagittal", or "coronal".

    Returns
    -------
    axis : int
        The dimension of the axis to take slices over when plotting.
    x : int
        The dimension of the x axis.
    y : int
        The dimension of the y axis.

    Notes
    -----
    .. versionadded:: 0.21
    .. versionchanged:: 0.24
    """
    _check_option('orientation', orientation, ('coronal', 'axial', 'sagittal'))
    axis = dict(coronal=1, axial=2, sagittal=0)[orientation]
    x, y = sorted(set([0, 1, 2]).difference(set([axis])))
    return axis, x, y


def _get_mri_info_data(mri, data):
    # Read the segmentation data using nibabel
    if data:
        _import_nibabel('load MRI atlas data')
    out = dict()
    _, out['vox_mri_t'], out['mri_ras_t'], dims, _, mgz = _read_mri_info(
        mri, return_img=True)
    out.update(
        mri_width=dims[0], mri_height=dims[1],
        mri_depth=dims[1], mri_volume_name=mri)
    if data:
        assert mgz is not None
        out['mri_vox_t'] = invert_transform(out['vox_mri_t'])
        out['data'] = np.asarray(mgz.dataobj)
    return out


def _get_mgz_header(fname):
    """Adapted from nibabel to quickly extract header info."""
    fname = _check_fname(fname, overwrite='read', must_exist=True,
                         name='MRI image')
    if not fname.endswith('.mgz'):
        raise IOError('Filename must end with .mgz')
    header_dtd = [('version', '>i4'), ('dims', '>i4', (4,)),
                  ('type', '>i4'), ('dof', '>i4'), ('goodRASFlag', '>i2'),
                  ('delta', '>f4', (3,)), ('Mdc', '>f4', (3, 3)),
                  ('Pxyz_c', '>f4', (3,))]
    header_dtype = np.dtype(header_dtd)
    with GzipFile(fname, 'rb') as fid:
        hdr_str = fid.read(header_dtype.itemsize)
    header = np.ndarray(shape=(), dtype=header_dtype,
                        buffer=hdr_str)
    # dims
    dims = header['dims'].astype(int)
    dims = dims[:3] if len(dims) == 4 else dims
    # vox2ras_tkr
    delta = header['delta']
    ds = np.array(delta, float)
    ns = np.array(dims * ds) / 2.0
    v2rtkr = np.array([[-ds[0], 0, 0, ns[0]],
                       [0, 0, ds[2], -ns[2]],
                       [0, -ds[1], 0, ns[1]],
                       [0, 0, 0, 1]], dtype=np.float32)
    # ras2vox
    d = np.diag(delta)
    pcrs_c = dims / 2.0
    Mdc = header['Mdc'].T
    pxyz_0 = header['Pxyz_c'] - np.dot(Mdc, np.dot(d, pcrs_c))
    M = np.eye(4, 4)
    M[0:3, 0:3] = np.dot(Mdc, d)
    M[0:3, 3] = pxyz_0.T
    header = dict(dims=dims, vox2ras_tkr=v2rtkr, vox2ras=M,
                  zooms=header['delta'])
    return header


def _get_atlas_values(vol_info, rr):
    # Transform MRI coordinates (where our surfaces live) to voxels
    rr_vox = apply_trans(vol_info['mri_vox_t'], rr)
    good = ((rr_vox >= -.5) &
            (rr_vox < np.array(vol_info['data'].shape, int) - 0.5)).all(-1)
    idx = np.round(rr_vox[good].T).astype(np.int64)
    values = np.full(rr.shape[0], np.nan)
    values[good] = vol_info['data'][tuple(idx)]
    return values


def get_volume_labels_from_aseg(mgz_fname, return_colors=False,
                                atlas_ids=None):
    """Return a list of names and colors of segmented volumes.

    Parameters
    ----------
    mgz_fname : str
        Filename to read. Typically aseg.mgz or some variant in the freesurfer
        pipeline.
    return_colors : bool
        If True returns also the labels colors.
    atlas_ids : dict | None
        A lookup table providing a mapping from region names (str) to ID values
        (int). Can be None to use the standard Freesurfer LUT.

        .. versionadded:: 0.21.0

    Returns
    -------
    label_names : list of str
        The names of segmented volumes included in this mgz file.
    label_colors : list of str
        The RGB colors of the labels included in this mgz file.

    See Also
    --------
    read_freesurfer_lut

    Notes
    -----
    .. versionchanged:: 0.21.0
       The label names are now sorted in the same order as their corresponding
       values in the MRI file.

    .. versionadded:: 0.9.0
    """
    import nibabel as nib
    if not op.isfile(mgz_fname):
        raise IOError('aseg file "%s" not found' % mgz_fname)
    atlas = nib.load(mgz_fname)
    data = np.asarray(atlas.dataobj)  # don't need float here
    want = np.unique(data)
    if atlas_ids is None:
        atlas_ids, colors = read_freesurfer_lut()
    elif return_colors:
        raise ValueError('return_colors must be False if atlas_ids are '
                         'provided')
    # restrict to the ones in the MRI, sorted by label name
    keep = np.in1d(list(atlas_ids.values()), want)
    keys = sorted((key for ki, key in enumerate(atlas_ids.keys()) if keep[ki]),
                  key=lambda x: atlas_ids[x])
    if return_colors:
        colors = [colors[k] for k in keys]
        out = keys, colors
    else:
        out = keys
    return out


##############################################################################
# Head to MRI volume conversion


@verbose
def head_to_mri(pos, subject, mri_head_t, subjects_dir=None, *,
                kind=None, unscale=False, verbose=None):
    """Convert pos from head coordinate system to MRI ones.

    Parameters
    ----------
    pos : array, shape (n_pos, 3)
        The  coordinates (in m) in head coordinate system.
    %(subject)s
    mri_head_t : instance of Transform
        MRI<->Head coordinate transformation.
    %(subjects_dir)s
    kind : str
        The  MRI coordinate frame kind, can be ``'ras'`` (default in 1.2) to
        use MRI RAS (scanner RAS) or ``'mri'`` (default in 1.3) for
        FreeSurfer surface RAS.

        .. versionadded:: 1.2
    unscale : bool
        For surrogate MRIs (e.g., scaled using ``mne coreg``), if True
        (default False), use the MRI scaling parameters to obtain points in
        the original/surrogate subject's MRI space.

        .. versionadded:: 1.2
    %(verbose)s

    Returns
    -------
    coordinates : array, shape (n_pos, 3)
        The MRI RAS coordinates (in mm) of pos.

    Notes
    -----
    This function requires nibabel.
    """
    from .coreg import read_mri_cfg
    if kind is None:
        warn('kind defaults to "ras" in 1.2 and will change to "mri" in 1.3, '
             'set it explicitly to avoid this warning', FutureWarning)
        kind = 'ras'
    _validate_type(kind, str, 'kind')
    _check_option('kind', kind, ('ras', 'mri'))
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    t1_fname = op.join(subjects_dir, subject, 'mri', 'T1.mgz')
    head_mri_t = _ensure_trans(mri_head_t, 'head', 'mri')
    if kind == 'ras':
        _, _, mri_ras_t, _, _ = _read_mri_info(t1_fname)
        head_ras_t = combine_transforms(head_mri_t, mri_ras_t, 'head', 'ras')
        head_dest_t = head_ras_t
    else:
        assert kind == 'mri'
        head_dest_t = head_mri_t
    pos_dest = apply_trans(head_dest_t, pos)
    # unscale if requested
    if unscale:
        params = read_mri_cfg(subject, subjects_dir)
        pos_dest /= params['scale']
    pos_dest *= 1e3  # mm
    return pos_dest


##############################################################################
# Surface to MNI conversion

@verbose
def vertex_to_mni(vertices, hemis, subject, subjects_dir=None, verbose=None):
    """Convert the array of vertices for a hemisphere to MNI coordinates.

    Parameters
    ----------
    vertices : int, or list of int
        Vertex number(s) to convert.
    hemis : int, or list of int
        Hemisphere(s) the vertices belong to.
    %(subject)s
    subjects_dir : str, or None
        Path to SUBJECTS_DIR if it is not set in the environment.
    %(verbose)s

    Returns
    -------
    coordinates : array, shape (n_vertices, 3)
        The MNI coordinates (in mm) of the vertices.
    """
    singleton = False
    if not isinstance(vertices, list) and not isinstance(vertices, np.ndarray):
        singleton = True
        vertices = [vertices]

    if not isinstance(hemis, list) and not isinstance(hemis, np.ndarray):
        hemis = [hemis] * len(vertices)

    if not len(hemis) == len(vertices):
        raise ValueError('hemi and vertices must match in length')

    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)

    surfs = [op.join(subjects_dir, subject, 'surf', '%s.white' % h)
             for h in ['lh', 'rh']]

    # read surface locations in MRI space
    rr = [read_surface(s)[0] for s in surfs]

    # take point locations in MRI space and convert to MNI coordinates
    xfm = read_talxfm(subject, subjects_dir)
    xfm['trans'][:3, 3] *= 1000.  # m->mm
    data = np.array([rr[h][v, :] for h, v in zip(hemis, vertices)])
    if singleton:
        data = data[0]
    return apply_trans(xfm['trans'], data)


##############################################################################
# Volume to MNI conversion

@verbose
def head_to_mni(pos, subject, mri_head_t, subjects_dir=None,
                verbose=None):
    """Convert pos from head coordinate system to MNI ones.

    Parameters
    ----------
    pos : array, shape (n_pos, 3)
        The  coordinates (in m) in head coordinate system.
    %(subject)s
    mri_head_t : instance of Transform
        MRI<->Head coordinate transformation.
    %(subjects_dir)s
    %(verbose)s

    Returns
    -------
    coordinates : array, shape (n_pos, 3)
        The MNI coordinates (in mm) of pos.

    Notes
    -----
    This function requires either nibabel.
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)

    # before we go from head to MRI (surface RAS)
    head_mni_t = combine_transforms(
        _ensure_trans(mri_head_t, 'head', 'mri'),
        read_talxfm(subject, subjects_dir), 'head', 'mni_tal')
    return apply_trans(head_mni_t, pos) * 1000.


@verbose
def get_mni_fiducials(subject, subjects_dir=None, verbose=None):
    """Estimate fiducials for a subject.

    Parameters
    ----------
    %(subject)s
    %(subjects_dir)s
    %(verbose)s

    Returns
    -------
    fids_mri : list
        List of estimated fiducials (each point in a dict), in the order
        LPA, nasion, RPA.

    Notes
    -----
    This takes the ``fsaverage-fiducials.fif`` file included with MNE—which
    contain the LPA, nasion, and RPA for the ``fsaverage`` subject—and
    transforms them to the given FreeSurfer subject's MRI space.
    The MRI of ``fsaverage`` is already in MNI Talairach space, so applying
    the inverse of the given subject's MNI Talairach affine transformation
    (``$SUBJECTS_DIR/$SUBJECT/mri/transforms/talairach.xfm``) is used
    to estimate the subject's fiducial locations.

    For more details about the coordinate systems and transformations involved,
    see https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems and
    :ref:`tut-source-alignment`.
    """
    # Eventually we might want to allow using the MNI Talairach with-skull
    # transformation rather than the standard brain-based MNI Talaranch
    # transformation, and/or project the points onto the head surface
    # (if available).
    fname_fids_fs = op.join(op.dirname(__file__), 'data',
                            'fsaverage', 'fsaverage-fiducials.fif')

    # Read fsaverage fiducials file and subject Talairach.
    fids, coord_frame = read_fiducials(fname_fids_fs)
    assert coord_frame == FIFF.FIFFV_COORD_MRI
    if subject == 'fsaverage':
        return fids  # special short-circuit for fsaverage
    mni_mri_t = invert_transform(read_talxfm(subject, subjects_dir))
    for f in fids:
        f['r'] = apply_trans(mni_mri_t, f['r'])
    return fids


@verbose
def estimate_head_mri_t(subject, subjects_dir=None, verbose=None):
    """Estimate the head->mri transform from fsaverage fiducials.

    A subject's fiducials can be estimated given a Freesurfer ``recon-all``
    by transforming ``fsaverage`` fiducials using the inverse Talairach
    transform, see :func:`mne.coreg.get_mni_fiducials`.

    Parameters
    ----------
    %(subject)s
    %(subjects_dir)s
    %(verbose)s

    Returns
    -------
    %(trans_not_none)s
    """
    from .channels.montage import make_dig_montage, compute_native_head_t
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    lpa, nasion, rpa = get_mni_fiducials(subject, subjects_dir)
    montage = make_dig_montage(lpa=lpa['r'], nasion=nasion['r'], rpa=rpa['r'],
                               coord_frame='mri')
    return invert_transform(compute_native_head_t(montage))


def _ensure_image_in_surface_RAS(image, subject, subjects_dir):
    """Check if the image is in Freesurfer surface RAS space."""
    import nibabel as nib
    if not isinstance(image, nib.spatialimages.SpatialImage):
        image = nib.load(image)
    image = nib.MGHImage(image.dataobj.astype(np.float32), image.affine)
    fs_img = nib.load(op.join(subjects_dir, subject, 'mri', 'brain.mgz'))
    if not np.allclose(image.affine, fs_img.affine, atol=1e-6):
        raise RuntimeError('The `image` is not aligned to Freesurfer '
                           'surface RAS space. This space is required as '
                           'it is the space where the anatomical '
                           'segmentation and reconstructed surfaces are')
    return image  # returns MGH image for header


@verbose
def read_lta(fname, verbose=None):
    """Read a Freesurfer linear transform array file.

    Parameters
    ----------
    fname : str | None
        The transform filename.
    %(verbose)s

    Returns
    -------
    affine : ndarray
        The affine transformation described by the lta file.
    """
    _validate_type(fname, ('path-like', None), 'fname')
    _check_fname(fname, 'read', must_exist=True)
    with open(fname, 'r') as fid:
        affine = np.loadtxt(fid.readlines()[5:9])
    return affine


@verbose
def read_talxfm(subject, subjects_dir=None, verbose=None):
    """Compute MRI-to-MNI transform from FreeSurfer talairach.xfm file.

    Parameters
    ----------
    %(subject)s
    %(subjects_dir)s
    %(verbose)s

    Returns
    -------
    mri_mni_t : instance of Transform
        The affine transformation from MRI to MNI space for the subject.
    """
    # Adapted from freesurfer m-files. Altered to deal with Norig
    # and Torig correctly
    subjects_dir = get_subjects_dir(subjects_dir)
    # Setup the RAS to MNI transform
    ras_mni_t = read_ras_mni_t(subject, subjects_dir)
    ras_mni_t['trans'][:3, 3] /= 1000.  # mm->m

    # We want to get from Freesurfer surface RAS ('mri') to MNI ('mni_tal').
    # This file only gives us RAS (non-zero origin) ('ras') to MNI ('mni_tal').
    # Se we need to get the ras->mri transform from the MRI headers.

    # To do this, we get Norig and Torig
    # (i.e. vox_ras_t and vox_mri_t, respectively)
    path = op.join(subjects_dir, subject, 'mri', 'orig.mgz')
    if not op.isfile(path):
        path = op.join(subjects_dir, subject, 'mri', 'T1.mgz')
    if not op.isfile(path):
        raise IOError('mri not found: %s' % path)
    _, _, mri_ras_t, _, _ = _read_mri_info(path)
    mri_mni_t = combine_transforms(mri_ras_t, ras_mni_t, 'mri', 'mni_tal')
    return mri_mni_t


def _check_mri(mri, subject, subjects_dir):
    """Check whether an mri exists in the Freesurfer subject directory."""
    _validate_type(mri, 'path-like', 'mri')
    if op.isfile(mri) and op.basename(mri) != mri:
        return mri
    if not op.isfile(mri):
        if subject is None:
            raise FileNotFoundError(
                f'MRI file {mri!r} not found and no subject provided')
        subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
        mri = op.join(subjects_dir, subject, 'mri', mri)
        if not op.isfile(mri):
            raise FileNotFoundError(f'MRI file {mri!r} not found')
    if op.basename(mri) == mri:
        err = (f'Ambiguous filename - found {mri!r} in current folder.\n'
               'If this is correct prefix name with relative or absolute path')
        raise IOError(err)
    return mri


def _read_mri_info(path, units='m', return_img=False, use_nibabel=False):
    # This is equivalent but 100x slower, so only use nibabel if we need to
    # (later):
    if use_nibabel:
        import nibabel
        hdr = nibabel.load(path).header
        n_orig = hdr.get_vox2ras()
        t_orig = hdr.get_vox2ras_tkr()
        dims = hdr.get_data_shape()
        zooms = hdr.get_zooms()[:3]
    else:
        hdr = _get_mgz_header(path)
        n_orig = hdr['vox2ras']
        t_orig = hdr['vox2ras_tkr']
        dims = hdr['dims']
        zooms = hdr['zooms']

    # extract the MRI_VOXEL to RAS (non-zero origin) transform
    vox_ras_t = Transform('mri_voxel', 'ras', n_orig)

    # extract the MRI_VOXEL to MRI transform
    vox_mri_t = Transform('mri_voxel', 'mri', t_orig)

    # construct the MRI to RAS (non-zero origin) transform
    mri_ras_t = combine_transforms(
        invert_transform(vox_mri_t), vox_ras_t, 'mri', 'ras')

    assert units in ('m', 'mm')
    if units == 'm':
        conv = np.array([[1e-3, 1e-3, 1e-3, 1]]).T
        # scaling and translation terms
        vox_ras_t['trans'] *= conv
        vox_mri_t['trans'] *= conv
        # just the translation term
        mri_ras_t['trans'][:, 3:4] *= conv

    out = (vox_ras_t, vox_mri_t, mri_ras_t, dims, zooms)
    if return_img:
        nibabel = _import_nibabel()
        out += (nibabel.load(path),)
    return out


def read_freesurfer_lut(fname=None):
    """Read a Freesurfer-formatted LUT.

    Parameters
    ----------
    fname : str | None
        The filename. Can be None to read the standard Freesurfer LUT.

    Returns
    -------
    atlas_ids : dict
        Mapping from label names to IDs.
    colors : dict
        Mapping from label names to colors.
    """
    lut = _get_lut(fname)
    names, ids = lut['name'], lut['id']
    colors = np.array([lut['R'], lut['G'], lut['B'], lut['A']], float).T
    atlas_ids = dict(zip(names, ids))
    colors = dict(zip(names, colors))
    return atlas_ids, colors


def _get_lut(fname=None):
    """Get a FreeSurfer LUT."""
    _validate_type(fname, ('path-like', None), 'fname')
    if fname is None:
        fname = op.join(op.dirname(__file__), 'data',
                        'FreeSurferColorLUT.txt')
    _check_fname(fname, 'read', must_exist=True)
    dtype = [('id', '<i8'), ('name', 'U'),
             ('R', '<i8'), ('G', '<i8'), ('B', '<i8'), ('A', '<i8')]
    lut = {d[0]: list() for d in dtype}
    with open(fname, 'r') as fid:
        for line in fid:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            line = line.split()
            if len(line) != len(dtype):
                raise RuntimeError(f'LUT is improperly formatted: {fname}')
            for d, part in zip(dtype, line):
                lut[d[0]].append(part)
    lut = {d[0]: np.array(lut[d[0]], dtype=d[1]) for d in dtype}
    assert len(lut['name']) > 0
    return lut


@verbose
def _get_head_surface(surf, subject, subjects_dir, bem=None, verbose=None):
    """Get a head surface from the Freesurfer subject directory.

    Parameters
    ----------
    surf : str
        The name of the surface 'auto', 'head', 'outer_skin', 'head-dense'
        or 'seghead'.
    %(subject)s
    %(subjects_dir)s
    bem : mne.bem.ConductorModel | None
        The conductor model that stores information about the head surface.
    %(verbose)s

    Returns
    -------
    head_surf : dict | None
        A dictionary with keys 'rr', 'tris', 'ntri', 'use_tris', 'np'
        and 'coord_frame' that store information for mesh plotting and other
        useful information about the head surface.

    Notes
    -----
    .. versionadded: 0.24
    """
    _check_option(
        'surf', surf, ('auto', 'head', 'outer_skin', 'head-dense', 'seghead'))
    if surf in ('auto', 'head', 'outer_skin'):
        if bem is not None:
            try:
                return _bem_find_surface(bem, 'head')
            except RuntimeError:
                logger.info('Could not find the surface for '
                            'head in the provided BEM model, '
                            'looking in the subject directory.')
    if subject is None:
        if surf == 'auto':
            return
        raise ValueError('To plot the head surface, the BEM/sphere'
                         ' model must contain a head surface '
                         'or "subject" must be provided (got '
                         'None)')
    subject_dir = op.join(
        get_subjects_dir(subjects_dir, raise_error=True), subject)
    if surf in ('head-dense', 'seghead'):
        try_fnames = [op.join(subject_dir, 'bem', f'{subject}-head-dense.fif'),
                      op.join(subject_dir, 'surf', 'lh.seghead')]
    else:
        try_fnames = [
            op.join(subject_dir, 'bem', 'outer_skin.surf'),
            op.join(subject_dir, 'bem', 'flash', 'outer_skin.surf'),
            op.join(subject_dir, 'bem', f'{subject}-head-sparse.fif'),
            op.join(subject_dir, 'bem', f'{subject}-head.fif'),
        ]
    for fname in try_fnames:
        if op.exists(fname):
            logger.info(f'Using {op.basename(fname)} for head surface.')
            if op.splitext(fname)[-1] == '.fif':
                return read_bem_surfaces(fname, on_defects='warn')[0]
            else:
                return _read_mri_surface(fname)
    raise IOError('No head surface found for subject '
                  f'{subject} after trying:\n' + '\n'.join(try_fnames))


@verbose
def _get_skull_surface(surf, subject, subjects_dir, bem=None, verbose=None):
    """Get a skull surface from the Freesurfer subject directory.

    Parameters
    ----------
    surf : str
        The name of the surface 'outer' or 'inner'.
    %(subject)s
    %(subjects_dir)s
    bem : mne.bem.ConductorModel | None
        The conductor model that stores information about the skull surface.
    %(verbose)s

    Returns
    -------
    skull_surf : dict | None
        A dictionary with keys 'rr', 'tris', 'ntri', 'use_tris', 'np'
        and 'coord_frame' that store information for mesh plotting and other
        useful information about the head surface.

    Notes
    -----
    .. versionadded: 0.24
    """
    if bem is not None:
        try:
            return _bem_find_surface(bem, surf + '_skull')
        except RuntimeError:
            logger.info('Could not find the surface for '
                        'skull in the provided BEM model, '
                        'looking in the subject directory.')
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    fname = _check_fname(op.join(subjects_dir, subject, 'bem',
                                 surf + '_skull.surf'),
                         overwrite='read', must_exist=True,
                         name=f'{surf} skull surface')
    return _read_mri_surface(fname)


def _estimate_talxfm_rigid(subject, subjects_dir):
    from .coreg import fit_matched_points, _trans_from_params
    xfm = read_talxfm(subject, subjects_dir)
    # XYZ+origin + halfway
    pts_tal = np.concatenate([np.eye(4)[:, :3], np.eye(3) * 0.5])
    pts_subj = apply_trans(invert_transform(xfm), pts_tal)
    # we fit with scaling enabled, but then discard it (we just need
    # the rigid-body components)
    params = fit_matched_points(pts_subj, pts_tal, scale=3, out='params')
    rigid = _trans_from_params((True, True, False), params[:6])
    return rigid
