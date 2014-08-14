# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from .externals.six import string_types
import numpy as np
import os
import os.path as op
from scipy import sparse, linalg
from copy import deepcopy

from .io.constants import FIFF
from .io.tree import dir_tree_find
from .io.tag import find_tag, read_tag
from .io.open import fiff_open
from .io.write import (start_block, end_block, write_int,
                       write_float_sparse_rcs, write_string,
                       write_float_matrix, write_int_matrix,
                       write_coord_trans, start_file, end_file, write_id)
from .surface import (read_surface, _create_surf_spacing, _get_ico_surface,
                      _tessellate_sphere_surf, read_bem_surfaces,
                      _read_surface_geom, _normalize_vectors,
                      _complete_surface_info, _compute_nearest,
                      fast_cross_3d)
from .source_estimate import mesh_dist
from .utils import (get_subjects_dir, run_subprocess, has_freesurfer,
                    has_nibabel, check_fname, logger, verbose,
                    check_scipy_version)
from .fixes import in1d, partial, gzip_open
from .parallel import parallel_func, check_n_jobs
from .transforms import (invert_transform, apply_trans, _print_coord_trans,
                         combine_transforms, _get_mri_head_t_from_trans_file,
                         read_trans, _coord_frame_name)


class SourceSpaces(list):
    """Represent a list of source space

    Currently implemented as a list of dictionaries containing the source
    space information

    Parameters
    ----------
    source_spaces : list
        A list of dictionaries containing the source space information.
    info : dict
        Dictionary with information about the creation of the source space
        file. Has keys 'working_dir' and 'command_line'.

    Attributes
    ----------
    info : dict
        Dictionary with information about the creation of the source space
        file. Has keys 'working_dir' and 'command_line'.
    """
    def __init__(self, source_spaces, info=None):
        super(SourceSpaces, self).__init__(source_spaces)
        if info is None:
            self.info = dict()
        else:
            self.info = dict(info)

    def __repr__(self):
        ss_repr = []
        for ss in self:
            ss_type = ss['type']
            if ss_type == 'vol':
                if 'seg_name' in ss:
                    r = ("'vol' (%s), n_used=%i"
                         % (ss['seg_name'], ss['nuse']))
                else:
                    r = ("'vol', shape=%s, n_used=%i"
                         % (repr(ss['shape']), ss['nuse']))
            elif ss_type == 'surf':
                r = "'surf', n_vertices=%i, n_used=%i" % (ss['np'], ss['nuse'])
            else:
                r = "%r" % ss_type
            coord_frame = ss['coord_frame']
            if isinstance(coord_frame, np.ndarray):
                coord_frame = coord_frame[0]
            r += ', coordinate_frame=%s' % _coord_frame_name(coord_frame)
            ss_repr.append('<%s>' % r)
        ss_repr = ', '.join(ss_repr)
        return "<SourceSpaces: [{ss}]>".format(ss=ss_repr)

    def __add__(self, other):
        return SourceSpaces(list.__add__(self, other))

    def copy(self):
        """Make a copy of the source spaces

        Returns
        -------
        src : instance of SourceSpaces
            The copied source spaces.
        """
        src = deepcopy(self)
        return src

    def save(self, fname):
        """Save the source spaces to a fif file

        Parameters
        ----------
        fname : str
            File to write.
        """
        write_source_spaces(fname, self)

    @verbose
    def export_volume(self, fname, include_surfaces=True,
                      include_discrete=True, dest='mri', trans=None,
                      mri_resolution=False, use_lut=True):
        """Exports source spaces to nifti or mgz file

        Parameters
        ----------
        fname : str
            Name of nifti or mgz file to write.
        include_surfaces : bool
            If True, include surface source spaces.
        include_discrete : bool
            If True, include discrete source spaces.
        dest : 'mri' | 'surf'
            If 'mri' the volume is defined in the coordinate system of the
            original T1 image. If 'surf' the coordinate system of the
            FreeSurfer surface is used (Surface RAS).
        trans : dict, str, or None
            Either a transformation filename (usually made using mne_analyze)
            or an info dict (usually opened using read_trans()).
            If string, an ending of `.fif` or `.fif.gz` will be assumed to be
            in FIF format, any other ending will be assumed to be a text file
            with a 4x4 transformation matrix (like the `--trans` MNE-C option.
            Must be provided if source spaces are in head coordinates and
            include_surfaces and mri_resolution are True.
        mri_resolution : bool
            If True, the image is saved in MRI resolution
            (e.g. 256 x 256 x 256).
        use_lut : bool
            If True, assigns a numeric value to each source space that
            corresponds to a color on the freesurfer lookup table.

        Notes
        -----
        This method requires nibabel.
        """

        # import nibabel or raise error
        try:
            import nibabel as nib
        except ImportError:
            raise RuntimeError('This function requires nibabel.')

        # Check coordinate frames of each source space
        coord_frames = np.array([s['coord_frame'] for s in self])

        # Raise error if trans is not provided when head coordinates are used
        # and mri_resolution and include_surfaces are true
        if (coord_frames == FIFF.FIFFV_COORD_HEAD).all():
            coords = 'head'  # all sources in head coordinates
            if mri_resolution and include_surfaces:
                if trans is None:
                    raise ValueError('trans containing mri to head transform '
                                     'must be provided if mri_resolution and '
                                     'include_surfaces are true and surfaces '
                                     'are in head coordinates')

            elif trans is not None:
                logger.info('trans is not needed and will not be used unless '
                            'include_surfaces and mri_resolution are True.')

        elif (coord_frames == FIFF.FIFFV_COORD_MRI).all():
            coords = 'mri'  # all sources in mri coordinates
            if trans is not None:
                logger.info('trans is not needed and will not be used unless '
                            'sources are in head coordinates.')
        # Raise error if all sources are not in the same space, or sources are
        # not in mri or head coordinates
        else:
            raise ValueError('All sources must be in head coordinates or all '
                             'sources must be in mri coordinates.')

        # use lookup table to assign values to source spaces
        logger.info('Reading FreeSurfer lookup table')
        # read the lookup table
        data_dir = op.join(op.dirname(__file__), 'data')
        lut_fname = op.join(data_dir, 'FreeSurferColorLUT.txt')
        lut = np.genfromtxt(lut_fname, dtype=None,
                            usecols=(0, 1), names=['id', 'name'])

        # Setup a dictionary of source types
        src_types = dict(volume=[], surface=[], discrete=[])

        # Populate dictionary of source types
        for src in self:
            # volume sources
            if src['type'] == 'vol':
                src_types['volume'].append(src)
            # surface sources
            elif src['type'] == 'surf':
                src_types['surface'].append(src)
            # discrete sources
            elif src['type'] == 'discrete':
                src_types['discrete'].append(src)
            # raise an error if dealing with source type other than volume
            # surface or discrete
            else:
                raise ValueError('Unrecognized source type: %s.' % src['type'])

        # Get shape, inuse array and interpolation matrix from volume sources
        first_vol = True  # mark the first volume source
        # Loop through the volume sources
        for vs in src_types['volume']:
            # read the lookup table value for segmented volume
            if 'seg_name' in vs:
                # find the color value for this volume
                i = lut['id'][lut['name'] == vs['seg_name']]
            else:
                # raise error for whole brain volume
                raise ValueError('Volume sources should be segments, '
                                 'not the entire volume.')

            if not use_lut:
                i = 1  # default value for no lookup table color matching

            if first_vol:
                # get the inuse array
                if mri_resolution:
                    # read the mri file used to generate volumes
                    aseg = nib.load(vs['mri_file'])

                    # get the voxel space shape
                    shape3d = (vs['mri_height'], vs['mri_depth'],
                               vs['mri_width'])

                    # get the values for this volume
                    inuse = i * (aseg.get_data() == i).astype(int)
                    # store as 1D array
                    inuse = inuse.ravel((2, 1, 0))

                else:
                    inuse = i * vs['inuse']

                    # get the volume source space shape
                    shape = vs['shape']

                    # read the shape in reverse order
                    # (otherwise results are scrambled)
                    shape3d = (shape[2], shape[1], shape[0])

                first_vol = False

            else:
                # update the inuse array
                if mri_resolution:

                    # get the values for this volume
                    use = i * (aseg.get_data() == i).astype(int)
                    inuse += use.ravel((2, 1, 0))
                else:
                    inuse += i * vs['inuse']

        # Raise error if there are no volume source spaces
        if first_vol:
            raise ValueError('Source spaces must contain at least one volume.')

        # create 3d grid in the MRI_VOXEL coordinate frame
        # len of inuse array should match shape regardless of mri_resolution
        assert len(inuse) == np.prod(shape3d)

        # setup the image in 3d space
        img = inuse.reshape(shape3d).T

        # include surface and/or discrete source spaces
        if include_surfaces or include_discrete:

            # setup affine transform for source spaces
            if mri_resolution:
                # get the MRI to MRI_VOXEL transform
                affine = invert_transform(vs['vox_mri_t'])
            else:
                # get the MRI to SOURCE (MRI_VOXEL) transform
                affine = invert_transform(vs['src_mri_t'])

            # modify affine if in head coordinates
            if coords == 'head':

                # read transformation
                if isinstance(trans, string_types):
                    if not op.isfile(trans):
                        raise IOError('trans file "%s" not found' % trans)
                    if op.splitext(trans)[1] in ['.fif', '.gz']:
                        mri_head_t = read_trans(trans)
                    else:
                        mri_head_t = _get_mri_head_t_from_trans_file(trans)
                else:  # dict
                    mri_head_t = trans

                # make sure its an MRI to HEAD transform
                if mri_head_t['from'] == FIFF.FIFFV_COORD_HEAD:
                    mri_head_t = invert_transform(mri_head_t)
                if not (mri_head_t['from'] == FIFF.FIFFV_COORD_MRI and
                        mri_head_t['to'] == FIFF.FIFFV_COORD_HEAD):
                    raise RuntimeError('Incorrect MRI transform provided')

                # get the HEAD to MRI transform
                head_mri_t = invert_transform(mri_head_t)

                # combine transforms, from HEAD to MRI_VOXEL
                affine = combine_transforms(head_mri_t, affine,
                                            FIFF.FIFFV_COORD_HEAD,
                                            FIFF.FIFFV_MNE_COORD_MRI_VOXEL)

            # loop through the surface source spaces
            if include_surfaces:

                # get the surface names (assumes left, right order. may want
                # to add these names during source space generation
                surf_names = ['Left-Cerebral-Cortex', 'Right-Cerebral-Cortex']

                for i, surf in enumerate(src_types['surface']):
                    # convert vertex positions from their native space
                    # (either HEAD or MRI) to MRI_VOXEL space
                    srf_rr = apply_trans(affine['trans'], surf['rr'])
                    # convert to numeric indices
                    ix_orig, iy_orig, iz_orig = srf_rr.T.round().astype(int)
                    # clip indices outside of volume space
                    ix_clip = np.maximum(np.minimum(ix_orig, shape3d[2]-1), 0)
                    iy_clip = np.maximum(np.minimum(iy_orig, shape3d[1]-1), 0)
                    iz_clip = np.maximum(np.minimum(iz_orig, shape3d[0]-1), 0)
                    # compare original and clipped indices
                    n_diff = np.array((ix_orig != ix_clip, iy_orig != iy_clip,
                                       iz_orig != iz_clip)).any(0).sum()
                    # generate use warnings for clipping
                    if n_diff > 0:
                        logger.warning('%s surface vertices lay outside '
                                       'of volume space. Consider using a '
                                       'larger volume space.' % n_diff)
                    # get surface id or use default value
                    if use_lut:
                        i = lut['id'][lut['name'] == surf_names[i]]
                    else:
                        i = 1
                    # update image to include surface voxels
                    img[ix_clip, iy_clip, iz_clip] = i

            # loop through discrete source spaces
            if include_discrete:
                for i, disc in enumerate(src_types['discrete']):
                    # convert vertex positions from their native space
                    # (either HEAD or MRI) to MRI_VOXEL space
                    disc_rr = apply_trans(affine['trans'], disc['rr'])
                    # convert to numeric indices
                    ix_orig, iy_orig, iz_orig = disc_rr.T.astype(int)
                    # clip indices outside of volume space
                    ix_clip = np.maximum(np.minimum(ix_orig, shape3d[2]-1), 0)
                    iy_clip = np.maximum(np.minimum(iy_orig, shape3d[1]-1), 0)
                    iz_clip = np.maximum(np.minimum(iz_orig, shape3d[0]-1), 0)
                    # compare original and clipped indices
                    n_diff = np.array((ix_orig != ix_clip, iy_orig != iy_clip,
                                       iz_orig != iz_clip)).any(0).sum()
                    # generate use warnings for clipping
                    if n_diff > 0:
                        logger.warning('%s discrete vertices lay outside '
                                       'of volume space. Consider using a '
                                       'larger volume space.' % n_diff)
                    # set default value
                    img[ix_clip, iy_clip, iz_clip] = 1
                    if use_lut:
                        logger.info('Discrete sources do not have values on '
                                    'the lookup table. Defaulting to 1.')

        # calculate affine transform for image (MRI_VOXEL to RAS)
        if mri_resolution:
            # MRI_VOXEL to MRI transform
            transform = vs['vox_mri_t'].copy()
        else:
            # MRI_VOXEL to MRI transform
            # NOTE: 'src' indicates downsampled version of MRI_VOXEL
            transform = vs['src_mri_t'].copy()
        if dest == 'mri':
            # combine with MRI to RAS transform
            transform = combine_transforms(transform, vs['mri_ras_t'],
                                           transform['from'],
                                           vs['mri_ras_t']['to'])
        # now setup the affine for volume image
        affine = transform['trans']
        # make sure affine converts from m to mm
        affine[:3] *= 1e3

        # save volume data

        # setup image for file
        if fname.endswith(('.nii', '.nii.gz')):  # save as nifit
            # setup the nifti header
            hdr = nib.Nifti1Header()
            hdr.set_xyzt_units('mm')
            # save the nifti image
            img = nib.Nifti1Image(img, affine, header=hdr)
        elif fname.endswith('.mgz'):  # save as mgh
            # convert to float32 (float64 not currently supported)
            img = img.astype('float32')
            # save the mgh image
            img = nib.freesurfer.mghformat.MGHImage(img, affine)
        else:
            raise(ValueError('Unrecognized file extension'))

        # write image to file
        nib.save(img, fname)


def _add_patch_info(s):
    """Patch information in a source space

    Generate the patch information from the 'nearest' vector in
    a source space. For vertex in the source space it provides
    the list of neighboring vertices in the high resolution
    triangulation.

    Parameters
    ----------
    s : dict
        The source space.
    """
    nearest = s['nearest']
    if nearest is None:
        s['pinfo'] = None
        s['patch_inds'] = None
        return

    logger.info('    Computing patch statistics...')

    indn = np.argsort(nearest)
    nearest_sorted = nearest[indn]

    steps = np.where(nearest_sorted[1:] != nearest_sorted[:-1])[0] + 1
    starti = np.r_[[0], steps]
    stopi = np.r_[steps, [len(nearest)]]

    pinfo = list()
    for start, stop in zip(starti, stopi):
        pinfo.append(np.sort(indn[start:stop]))
    s['pinfo'] = pinfo

    # compute patch indices of the in-use source space vertices
    patch_verts = nearest_sorted[steps - 1]
    s['patch_inds'] = np.searchsorted(patch_verts, s['vertno'])

    logger.info('    Patch information added...')


@verbose
def read_source_spaces_from_tree(fid, tree, add_geom=False, verbose=None):
    """Read the source spaces from a FIF file

    Parameters
    ----------
    fid : file descriptor
        An open file descriptor.
    tree : dict
        The FIF tree structure if source is a file id.
    add_geom : bool, optional (default False)
        Add geometry information to the surfaces.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    src : SourceSpaces
        The source spaces.
    """
    #   Find all source spaces
    spaces = dir_tree_find(tree, FIFF.FIFFB_MNE_SOURCE_SPACE)
    if len(spaces) == 0:
        raise ValueError('No source spaces found')

    src = list()
    for s in spaces:
        logger.info('    Reading a source space...')
        this = _read_one_source_space(fid, s)
        logger.info('    [done]')
        if add_geom:
            _complete_source_space_info(this)

        src.append(this)

    src = SourceSpaces(src)
    logger.info('    %d source spaces read' % len(spaces))

    return src


@verbose
def read_source_spaces(fname, add_geom=False, verbose=None):
    """Read the source spaces from a FIF file

    Parameters
    ----------
    fname : str
        The name of the file, which should end with -src.fif or
        -src.fif.gz.
    add_geom : bool, optional (default False)
        Add geometry information to the surfaces.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    src : SourceSpaces
        The source spaces.
    """
    check_fname(fname, 'source space', ('-src.fif', '-src.fif.gz'))

    ff, tree, _ = fiff_open(fname)
    with ff as fid:
        src = read_source_spaces_from_tree(fid, tree, add_geom=add_geom,
                                           verbose=verbose)
        src.info['fname'] = fname
        node = dir_tree_find(tree, FIFF.FIFFB_MNE_ENV)
        if node:
            node = node[0]
            for p in range(node['nent']):
                kind = node['directory'][p].kind
                pos = node['directory'][p].pos
                tag = read_tag(fid, pos)
                if kind == FIFF.FIFF_MNE_ENV_WORKING_DIR:
                    src.info['working_dir'] = tag.data
                elif kind == FIFF.FIFF_MNE_ENV_COMMAND_LINE:
                    src.info['command_line'] = tag.data
    return src


@verbose
def _read_one_source_space(fid, this, verbose=None):
    """Read one source space
    """
    FIFF_BEM_SURF_NTRI = 3104
    FIFF_BEM_SURF_TRIANGLES = 3106

    res = dict()

    tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_ID)
    if tag is None:
        res['id'] = int(FIFF.FIFFV_MNE_SURF_UNKNOWN)
    else:
        res['id'] = int(tag.data)

    tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_TYPE)
    if tag is None:
        raise ValueError('Unknown source space type')
    else:
        src_type = int(tag.data)
        if src_type == FIFF.FIFFV_MNE_SPACE_SURFACE:
            res['type'] = 'surf'
        elif src_type == FIFF.FIFFV_MNE_SPACE_VOLUME:
            res['type'] = 'vol'
        elif src_type == FIFF.FIFFV_MNE_SPACE_DISCRETE:
            res['type'] = 'discrete'
        else:
            raise ValueError('Unknown source space type (%d)' % src_type)

    if res['type'] == 'vol':

        tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_VOXEL_DIMS)
        if tag is not None:
            res['shape'] = tuple(tag.data)

        tag = find_tag(fid, this, FIFF.FIFF_COORD_TRANS)
        if tag is not None:
            res['src_mri_t'] = tag.data

        parent_mri = dir_tree_find(this, FIFF.FIFFB_MNE_PARENT_MRI_FILE)
        if len(parent_mri) == 0:
            # MNE 2.7.3 (and earlier) didn't store necessary information
            # about volume coordinate translations. Although there is a
            # FFIF_COORD_TRANS in the higher level of the FIFF file, this
            # doesn't contain all the info we need. Safer to return an
            # error unless a user really wants us to add backward compat.
            raise ValueError('Can not find parent MRI location. The volume '
                             'source space may have been made with an MNE '
                             'version that is too old (<= 2.7.3). Consider '
                             'updating and regenerating the inverse.')

        mri = parent_mri[0]
        for d in mri['directory']:
            if d.kind == FIFF.FIFF_COORD_TRANS:
                tag = read_tag(fid, d.pos)
                trans = tag.data
                if trans['from'] == FIFF.FIFFV_MNE_COORD_MRI_VOXEL:
                    res['vox_mri_t'] = tag.data
                if trans['to'] == FIFF.FIFFV_MNE_COORD_RAS:
                    res['mri_ras_t'] = tag.data

        tag = find_tag(fid, mri, FIFF.FIFF_MNE_SOURCE_SPACE_INTERPOLATOR)
        if tag is not None:
            res['interpolator'] = tag.data
        else:
            logger.info("Interpolation matrix for MRI not found.")

        tag = find_tag(fid, mri, FIFF.FIFF_MNE_SOURCE_SPACE_MRI_FILE)
        if tag is not None:
            res['mri_file'] = tag.data

        tag = find_tag(fid, mri, FIFF.FIFF_MRI_WIDTH)
        if tag is not None:
            res['mri_width'] = int(tag.data)

        tag = find_tag(fid, mri, FIFF.FIFF_MRI_HEIGHT)
        if tag is not None:
            res['mri_height'] = int(tag.data)

        tag = find_tag(fid, mri, FIFF.FIFF_MRI_DEPTH)
        if tag is not None:
            res['mri_depth'] = int(tag.data)

        tag = find_tag(fid, this, FIFF.FIFF_COMMENT)
        if tag is not None:
            res['seg_name'] = tag.data

    tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_NPOINTS)
    if tag is None:
        raise ValueError('Number of vertices not found')

    res['np'] = int(tag.data)

    tag = find_tag(fid, this, FIFF_BEM_SURF_NTRI)
    if tag is None:
        tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_NTRI)
        if tag is None:
            res['ntri'] = 0
        else:
            res['ntri'] = int(tag.data)
    else:
        res['ntri'] = tag.data

    tag = find_tag(fid, this, FIFF.FIFF_MNE_COORD_FRAME)
    if tag is None:
        raise ValueError('Coordinate frame information not found')

    res['coord_frame'] = tag.data

    #   Vertices, normals, and triangles
    tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_POINTS)
    if tag is None:
        raise ValueError('Vertex data not found')

    res['rr'] = tag.data.astype(np.float)  # double precision for mayavi
    if res['rr'].shape[0] != res['np']:
        raise ValueError('Vertex information is incorrect')

    tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_NORMALS)
    if tag is None:
        raise ValueError('Vertex normals not found')

    res['nn'] = tag.data
    if res['nn'].shape[0] != res['np']:
        raise ValueError('Vertex normal information is incorrect')

    if res['ntri'] > 0:
        tag = find_tag(fid, this, FIFF_BEM_SURF_TRIANGLES)
        if tag is None:
            tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_TRIANGLES)
            if tag is None:
                raise ValueError('Triangulation not found')
            else:
                res['tris'] = tag.data - 1  # index start at 0 in Python
        else:
            res['tris'] = tag.data - 1  # index start at 0 in Python

        if res['tris'].shape[0] != res['ntri']:
            raise ValueError('Triangulation information is incorrect')
    else:
        res['tris'] = None

    #   Which vertices are active
    tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_NUSE)
    if tag is None:
        res['nuse'] = 0
        res['inuse'] = np.zeros(res['nuse'], dtype=np.int)
        res['vertno'] = None
    else:
        res['nuse'] = int(tag.data)
        tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_SELECTION)
        if tag is None:
            raise ValueError('Source selection information missing')

        res['inuse'] = tag.data.astype(np.int).T
        if len(res['inuse']) != res['np']:
            raise ValueError('Incorrect number of entries in source space '
                             'selection')

        res['vertno'] = np.where(res['inuse'])[0]

    #   Use triangulation
    tag1 = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_NUSE_TRI)
    tag2 = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_USE_TRIANGLES)
    if tag1 is None or tag2 is None:
        res['nuse_tri'] = 0
        res['use_tris'] = None
    else:
        res['nuse_tri'] = tag1.data
        res['use_tris'] = tag2.data - 1  # index start at 0 in Python

    #   Patch-related information
    tag1 = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_NEAREST)
    tag2 = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_NEAREST_DIST)

    if tag1 is None or tag2 is None:
        res['nearest'] = None
        res['nearest_dist'] = None
    else:
        res['nearest'] = tag1.data
        res['nearest_dist'] = tag2.data.T

    _add_patch_info(res)

    #   Distances
    tag1 = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_DIST)
    tag2 = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_DIST_LIMIT)
    if tag1 is None or tag2 is None:
        res['dist'] = None
        res['dist_limit'] = None
    else:
        res['dist'] = tag1.data
        res['dist_limit'] = tag2.data
        #   Add the upper triangle
        res['dist'] = res['dist'] + res['dist'].T
    if (res['dist'] is not None):
        logger.info('    Distance information added...')

    tag = find_tag(fid, this, FIFF.FIFF_SUBJ_HIS_ID)
    if tag is not None:
        res['subject_his_id'] = tag.data

    return res


@verbose
def _complete_source_space_info(this, verbose=None):
    """Add more info on surface
    """
    #   Main triangulation
    logger.info('    Completing triangulation info...')
    this['tri_area'] = np.zeros(this['ntri'])
    r1 = this['rr'][this['tris'][:, 0], :]
    r2 = this['rr'][this['tris'][:, 1], :]
    r3 = this['rr'][this['tris'][:, 2], :]
    this['tri_cent'] = (r1 + r2 + r3) / 3.0
    this['tri_nn'] = fast_cross_3d((r2 - r1), (r3 - r1))
    size = np.sqrt(np.sum(this['tri_nn'] ** 2, axis=1))
    this['tri_area'] = size / 2.0
    this['tri_nn'] /= size[:, None]
    logger.info('[done]')

    #   Selected triangles
    logger.info('    Completing selection triangulation info...')
    if this['nuse_tri'] > 0:
        r1 = this['rr'][this['use_tris'][:, 0], :]
        r2 = this['rr'][this['use_tris'][:, 1], :]
        r3 = this['rr'][this['use_tris'][:, 2], :]
        this['use_tri_cent'] = (r1 + r2 + r3) / 3.0
        this['use_tri_nn'] = fast_cross_3d((r2 - r1), (r3 - r1))
        this['use_tri_area'] = np.sqrt(np.sum(this['use_tri_nn'] ** 2, axis=1)
                                       ) / 2.0
    logger.info('[done]')


def find_source_space_hemi(src):
    """Return the hemisphere id for a source space

    Parameters
    ----------
    src : dict
        The source space to investigate

    Returns
    -------
    hemi : int
        Deduced hemisphere id
    """
    xave = src['rr'][:, 0].sum()

    if xave < 0:
        hemi = int(FIFF.FIFFV_MNE_SURF_LEFT_HEMI)
    else:
        hemi = int(FIFF.FIFFV_MNE_SURF_RIGHT_HEMI)

    return hemi


def label_src_vertno_sel(label, src):
    """ Find vertex numbers and indices from label

    Parameters
    ----------
    label : Label
        Source space label
    src : dict
        Source space

    Returns
    -------
    vertno : list of length 2
        Vertex numbers for lh and rh
    src_sel : array of int (len(idx) = len(vertno[0]) + len(vertno[1]))
        Indices of the selected vertices in sourse space
    """
    if src[0]['type'] != 'surf':
        return Exception('Label are only supported with surface source spaces')

    vertno = [src[0]['vertno'], src[1]['vertno']]

    if label.hemi == 'lh':
        vertno_sel = np.intersect1d(vertno[0], label.vertices)
        src_sel = np.searchsorted(vertno[0], vertno_sel)
        vertno[0] = vertno_sel
        vertno[1] = np.array([])
    elif label.hemi == 'rh':
        vertno_sel = np.intersect1d(vertno[1], label.vertices)
        src_sel = np.searchsorted(vertno[1], vertno_sel) + len(vertno[0])
        vertno[0] = np.array([])
        vertno[1] = vertno_sel
    elif label.hemi == 'both':
        vertno_sel_lh = np.intersect1d(vertno[0], label.lh.vertices)
        src_sel_lh = np.searchsorted(vertno[0], vertno_sel_lh)
        vertno_sel_rh = np.intersect1d(vertno[1], label.rh.vertices)
        src_sel_rh = np.searchsorted(vertno[1], vertno_sel_rh) + len(vertno[0])
        src_sel = np.hstack((src_sel_lh, src_sel_rh))
        vertno = [vertno_sel_lh, vertno_sel_rh]
    else:
        raise Exception("Unknown hemisphere type")

    return vertno, src_sel


def _get_vertno(src):
    return [s['vertno'] for s in src]


###############################################################################
# Write routines

@verbose
def _write_source_spaces_to_fid(fid, src, verbose=None):
    """Write the source spaces to a FIF file

    Parameters
    ----------
    fid : file descriptor
        An open file descriptor.
    src : list
        The list of source spaces.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    """
    for s in src:
        logger.info('    Write a source space...')
        start_block(fid, FIFF.FIFFB_MNE_SOURCE_SPACE)
        _write_one_source_space(fid, s, verbose)
        end_block(fid, FIFF.FIFFB_MNE_SOURCE_SPACE)
        logger.info('    [done]')
    logger.info('    %d source spaces written' % len(src))


@verbose
def write_source_spaces(fname, src, verbose=None):
    """Write source spaces to a file

    Parameters
    ----------
    fname : str
        The name of the file, which should end with -src.fif or
        -src.fif.gz.
    src : SourceSpaces
        The source spaces (as returned by read_source_spaces).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    """
    check_fname(fname, 'source space', ('-src.fif', '-src.fif.gz'))

    fid = start_file(fname)
    start_block(fid, FIFF.FIFFB_MNE)

    if src.info:
        start_block(fid, FIFF.FIFFB_MNE_ENV)

        write_id(fid, FIFF.FIFF_BLOCK_ID)

        data = src.info.get('working_dir', None)
        if data:
            write_string(fid, FIFF.FIFF_MNE_ENV_WORKING_DIR, data)
        data = src.info.get('command_line', None)
        if data:
            write_string(fid, FIFF.FIFF_MNE_ENV_COMMAND_LINE, data)

        end_block(fid, FIFF.FIFFB_MNE_ENV)

    _write_source_spaces_to_fid(fid, src, verbose)

    end_block(fid, FIFF.FIFFB_MNE)
    end_file(fid)


def _write_one_source_space(fid, this, verbose=None):
    """Write one source space"""
    if this['type'] == 'surf':
        src_type = FIFF.FIFFV_MNE_SPACE_SURFACE
    elif this['type'] == 'vol':
        src_type = FIFF.FIFFV_MNE_SPACE_VOLUME
    elif this['type'] == 'discrete':
        src_type = FIFF.FIFFV_MNE_SPACE_DISCRETE
    else:
        raise ValueError('Unknown source space type (%s)' % this['type'])
    write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_TYPE, src_type)
    if this['id'] >= 0:
        write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_ID, this['id'])

    data = this.get('subject_his_id', None)
    if data:
        write_string(fid, FIFF.FIFF_SUBJ_HIS_ID, data)
    write_int(fid, FIFF.FIFF_MNE_COORD_FRAME, this['coord_frame'])

    write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NPOINTS, this['np'])
    write_float_matrix(fid, FIFF.FIFF_MNE_SOURCE_SPACE_POINTS, this['rr'])
    write_float_matrix(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NORMALS, this['nn'])

    #   Which vertices are active
    write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_SELECTION, this['inuse'])
    write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NUSE, this['nuse'])

    if this['ntri'] > 0:
        write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NTRI, this['ntri'])
        write_int_matrix(fid, FIFF.FIFF_MNE_SOURCE_SPACE_TRIANGLES,
                         this['tris'] + 1)

    if this['type'] != 'vol' and this['use_tris'] is not None:
        #   Use triangulation
        write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NUSE_TRI, this['nuse_tri'])
        write_int_matrix(fid, FIFF.FIFF_MNE_SOURCE_SPACE_USE_TRIANGLES,
                         this['use_tris'] + 1)

    if this['type'] == 'vol':
        neighbor_vert = this.get('neighbor_vert', None)
        if neighbor_vert is not None:
            nneighbors = np.array([len(n) for n in neighbor_vert])
            neighbors = np.concatenate(neighbor_vert)
            write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NNEIGHBORS, nneighbors)
            write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NEIGHBORS, neighbors)

        write_coord_trans(fid, this['src_mri_t'])

        write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_VOXEL_DIMS, this['shape'])

        start_block(fid, FIFF.FIFFB_MNE_PARENT_MRI_FILE)
        write_coord_trans(fid, this['mri_ras_t'])
        write_coord_trans(fid, this['vox_mri_t'])

        mri_volume_name = this.get('mri_volume_name', None)
        if mri_volume_name is not None:
            write_string(fid, FIFF.FIFF_MNE_FILE_NAME, mri_volume_name)

        write_float_sparse_rcs(fid, FIFF.FIFF_MNE_SOURCE_SPACE_INTERPOLATOR,
                               this['interpolator'])

        if 'mri_file' in this and this['mri_file'] is not None:
            write_string(fid, FIFF.FIFF_MNE_SOURCE_SPACE_MRI_FILE,
                         this['mri_file'])

        write_int(fid, FIFF.FIFF_MRI_WIDTH, this['mri_width'])
        write_int(fid, FIFF.FIFF_MRI_HEIGHT, this['mri_height'])
        write_int(fid, FIFF.FIFF_MRI_DEPTH, this['mri_depth'])

        end_block(fid, FIFF.FIFFB_MNE_PARENT_MRI_FILE)

    #   Patch-related information
    if this['nearest'] is not None:
        write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NEAREST, this['nearest'])
        write_float_matrix(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NEAREST_DIST,
                           this['nearest_dist'])

    #   Distances
    if this['dist'] is not None:
        # Save only upper triangular portion of the matrix
        dists = this['dist'].copy()
        dists = sparse.triu(dists, format=dists.format)
        write_float_sparse_rcs(fid, FIFF.FIFF_MNE_SOURCE_SPACE_DIST, dists)
        write_float_matrix(fid, FIFF.FIFF_MNE_SOURCE_SPACE_DIST_LIMIT,
                           this['dist_limit'])

    #   Segmentation data
    if this['type'] == 'vol' and ('seg_name' in this):
        # Save the name of the segment
        write_string(fid, FIFF.FIFF_COMMENT, this['seg_name'])


##############################################################################
# Surface to MNI conversion

@verbose
def vertex_to_mni(vertices, hemis, subject, subjects_dir=None, mode=None,
                  verbose=None):
    """Convert the array of vertices for a hemisphere to MNI coordinates

    Parameters
    ----------
    vertices : int, or list of int
        Vertex number(s) to convert
    hemis : int, or list of int
        Hemisphere(s) the vertices belong to
    subject : string
        Name of the subject to load surfaces from.
    subjects_dir : string, or None
        Path to SUBJECTS_DIR if it is not set in the environment.
    mode : string | None
        Either 'nibabel' or 'freesurfer' for the software to use to
        obtain the transforms. If None, 'nibabel' is tried first, falling
        back to 'freesurfer' if it fails. Results should be equivalent with
        either option, but nibabel may be quicker (and more pythonic).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    coordinates : n_vertices x 3 array of float
        The MNI coordinates (in mm) of the vertices

    Notes
    -----
    This function requires either nibabel (in Python) or Freesurfer
    (with utility "mri_info") to be correctly installed.
    """
    if not has_freesurfer and not has_nibabel():
        raise RuntimeError('NiBabel (Python) or Freesurfer (Unix) must be '
                           'correctly installed and accessible from Python')

    if not isinstance(vertices, list) and not isinstance(vertices, np.ndarray):
        vertices = [vertices]

    if not isinstance(hemis, list) and not isinstance(hemis, np.ndarray):
        hemis = [hemis] * len(vertices)

    if not len(hemis) == len(vertices):
        raise ValueError('hemi and vertices must match in length')

    subjects_dir = get_subjects_dir(subjects_dir)

    surfs = [op.join(subjects_dir, subject, 'surf', '%s.white' % h)
             for h in ['lh', 'rh']]

    # read surface locations in MRI space
    rr = [read_surface(s)[0] for s in surfs]

    # take point locations in MRI space and convert to MNI coordinates
    xfm = _read_talxfm(subject, subjects_dir, mode)
    data = np.array([rr[h][v, :] for h, v in zip(hemis, vertices)])
    return apply_trans(xfm['trans'], data)


@verbose
def _read_talxfm(subject, subjects_dir, mode=None, verbose=None):
    """Read MNI transform from FreeSurfer talairach.xfm file

    Adapted from freesurfer m-files. Altered to deal with Norig
    and Torig correctly.
    """
    if mode is not None and mode not in ['nibabel', 'freesurfer']:
        raise ValueError('mode must be "nibabel" or "freesurfer"')
    fname = op.join(subjects_dir, subject, 'mri', 'transforms',
                    'talairach.xfm')
    # read the RAS to MNI transform from talairach.xfm
    with open(fname, 'r') as fid:
        logger.debug('Reading FreeSurfer talairach.xfm file:\n%s' % fname)

        # read lines until we get the string 'Linear_Transform', which precedes
        # the data transformation matrix
        got_it = False
        comp = 'Linear_Transform'
        for line in fid:
            if line[:len(comp)] == comp:
                # we have the right line, so don't read any more
                got_it = True
                break

        if got_it:
            xfm = list()
            # read the transformation matrix (3x4)
            for ii, line in enumerate(fid):
                digs = [float(s) for s in line.strip('\n;').split()]
                xfm.append(digs)
                if ii == 2:
                    break
            xfm.append([0., 0., 0., 1.])
            xfm = np.array(xfm, dtype=float)
        else:
            raise ValueError('failed to find \'Linear_Transform\' string in '
                             'xfm file:\n%s' % fname)

    # Setup the RAS to MNI transform
    ras_mni_t = {'from': FIFF.FIFFV_MNE_COORD_RAS,
                 'to': FIFF.FIFFV_MNE_COORD_MNI_TAL, 'trans': xfm}

    # now get Norig and Torig
    # (i.e. vox_ras_t and vox_mri_t, respectively)
    path = op.join(subjects_dir, subject, 'mri', 'orig.mgz')

    if has_nibabel():
        use_nibabel = True
    else:
        use_nibabel = False
        if mode == 'nibabel':
            raise ImportError('Tried to import nibabel but failed, try using '
                              'mode=None or mode=Freesurfer')

    # note that if mode == None, then we default to using nibabel
    if use_nibabel is True and mode == 'freesurfer':
        use_nibabel = False
    if use_nibabel:
        import nibabel as nib
        img = nib.load(path)
        hdr = img.get_header()
        # read the MRI_VOXEL to RAS transform
        n_orig = hdr.get_vox2ras()
        # read the MRI_VOXEL to MRI transform
        ds = np.array(hdr.get_zooms())
        ns = (np.array(hdr.get_data_shape()[:3]) * ds) / 2.0
        t_orig = np.array([[-ds[0], 0, 0, ns[0]],
                           [0, 0, ds[2], -ns[2]],
                           [0, -ds[1], 0, ns[1]],
                           [0, 0, 0, 1]], dtype=float)
        nt_orig = [n_orig, t_orig]
    else:
        nt_orig = list()
        for conv in ['--vox2ras', '--vox2ras-tkr']:
            stdout, stderr = run_subprocess(['mri_info', conv, path])
            stdout = np.fromstring(stdout, sep=' ').astype(float)
            if not stdout.size == 16:
                raise ValueError('Could not parse Freesurfer mri_info output')
            nt_orig.append(stdout.reshape(4, 4))
    # extract the MRI_VOXEL to RAS transform
    n_orig = nt_orig[0]
    vox_ras_t = {'from': FIFF.FIFFV_MNE_COORD_MRI_VOXEL,
                 'to': FIFF.FIFFV_MNE_COORD_RAS,
                 'trans': n_orig}

    # extract the MRI_VOXEL to MRI transform
    t_orig = nt_orig[1]
    vox_mri_t = {'from': FIFF.FIFFV_MNE_COORD_MRI_VOXEL,
                 'to': FIFF.FIFFV_COORD_MRI,
                 'trans': t_orig}

    # invert MRI_VOXEL to MRI to get the MRI to MRI_VOXEL transform
    mri_vox_t = invert_transform(vox_mri_t)

    # construct an MRI to RAS transform
    mri_ras_t = combine_transforms(mri_vox_t, vox_ras_t,
                                   FIFF.FIFFV_COORD_MRI,
                                   FIFF.FIFFV_MNE_COORD_RAS)

    # construct the MRI to MNI transform
    mri_mni_t = combine_transforms(mri_ras_t, ras_mni_t,
                                   FIFF.FIFFV_COORD_MRI,
                                   FIFF.FIFFV_MNE_COORD_MNI_TAL)
    return mri_mni_t


###############################################################################
# Creation and decimation

@verbose
def setup_source_space(subject, fname=True, spacing='oct6', surface='white',
                       overwrite=False, subjects_dir=None, add_dist=None,
                       verbose=None):
    """Setup a source space with subsampling

    Parameters
    ----------
    subject : str
        Subject to process.
    fname : str | None | bool
        Filename to use. If True, a default name will be used. If None,
        the source space will not be saved (only returned).
    spacing : str
        The spacing to use. Can be ``'ico#'`` for a recursively subdivided
        icosahedron, ``'oct#'`` for a recursively subdivided octahedron,
        or ``'all'`` for all points.
    surface : str
        The surface to use.
    overwrite: bool
        If True, overwrite output file (if it exists).
    subjects_dir : string, or None
        Path to SUBJECTS_DIR if it is not set in the environment.
    add_dist : bool
        Add distance and patch information to the source space. This takes some
        time so precomputing it is recommended. The default is currently False
        but will change to True in release 0.9.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    src : list
        The source space for each hemisphere.
    """
    if add_dist is None:
        msg = ("The add_dist parameter to mne.setup_source_space currently "
               "defaults to False, but the default will change to True in "
               "release 0.9. Specify the parameter explicitly to avoid this "
               "warning.")
        logger.warning(msg)

    cmd = ('setup_source_space(%s, fname=%s, spacing=%s, surface=%s, '
           'overwrite=%s, subjects_dir=%s, add_dist=%s, verbose=%s)'
           % (subject, fname, spacing, surface, overwrite,
              subjects_dir, add_dist, verbose))
    # check to make sure our parameters are good, parse 'spacing'
    space_err = ('"spacing" must be a string with values '
                 '"ico#", "oct#", or "all", and "ico" and "oct"'
                 'numbers must be integers')
    if not isinstance(spacing, string_types) or len(spacing) < 3:
        raise ValueError(space_err)
    if spacing == 'all':
        stype = 'all'
        sval = ''
    elif spacing[:3] == 'ico':
        stype = 'ico'
        sval = spacing[3:]
    elif spacing[:3] == 'oct':
        stype = 'oct'
        sval = spacing[3:]
    else:
        raise ValueError(space_err)
    try:
        if stype in ['ico', 'oct']:
            sval = int(sval)
        elif stype == 'spacing':  # spacing
            sval = float(sval)
    except:
        raise ValueError(space_err)
    subjects_dir = get_subjects_dir(subjects_dir)
    surfs = [op.join(subjects_dir, subject, 'surf', hemi + surface)
             for hemi in ['lh.', 'rh.']]
    bem_dir = op.join(subjects_dir, subject, 'bem')

    for surf, hemi in zip(surfs, ['LH', 'RH']):
        if surf is not None and not op.isfile(surf):
            raise IOError('Could not find the %s surface %s'
                          % (hemi, surf))

    if not (fname is True or fname is None or isinstance(fname, string_types)):
        raise ValueError('"fname" must be a string, True, or None')
    if fname is True:
        extra = '%s-%s' % (stype, sval) if sval != '' else stype
        fname = op.join(bem_dir, '%s-%s-src.fif' % (subject, extra))
    if fname is not None and op.isfile(fname) and overwrite is False:
        raise IOError('file "%s" exists, use overwrite=True if you want '
                      'to overwrite the file' % fname)

    logger.info('Setting up the source space with the following parameters:\n')
    logger.info('SUBJECTS_DIR = %s' % subjects_dir)
    logger.info('Subject      = %s' % subject)
    logger.info('Surface      = %s' % surface)
    if stype == 'ico':
        src_type_str = 'ico = %s' % sval
        logger.info('Icosahedron subdivision grade %s\n' % sval)
    elif stype == 'oct':
        src_type_str = 'oct = %s' % sval
        logger.info('Octahedron subdivision grade %s\n' % sval)
    else:
        src_type_str = 'all'
        logger.info('Include all vertices\n')

    # Create the fif file
    if fname is not None:
        logger.info('>>> 1. Creating the source space file %s...' % fname)
    else:
        logger.info('>>> 1. Creating the source space...\n')

    # mne_make_source_space ... actually make the source spaces
    src = []

    # pre-load ico/oct surf (once) for speed, if necessary
    if stype in ['ico', 'oct']:
        # ### from mne_ico_downsample.c ###
        if stype == 'ico':
            logger.info('Doing the icosahedral vertex picking...')
            ico_surf = _get_ico_surface(sval)
        else:
            logger.info('Doing the octahedral vertex picking...')
            ico_surf = _tessellate_sphere_surf(sval)
    else:
        ico_surf = None

    for hemi, surf in zip(['lh', 'rh'], surfs):
        logger.info('Loading %s...' % surf)
        # Setup the surface spacing in the MRI coord frame
        s = _create_surf_spacing(surf, hemi, subject, stype, sval, ico_surf,
                                 subjects_dir)
        logger.info('loaded %s %d/%d selected to source space (%s)'
                    % (op.split(surf)[1], s['nuse'], s['np'], src_type_str))
        src.append(s)
        logger.info('')  # newline after both subject types are run

    # Fill in source space info
    hemi_ids = [FIFF.FIFFV_MNE_SURF_LEFT_HEMI, FIFF.FIFFV_MNE_SURF_RIGHT_HEMI]
    for s, s_id in zip(src, hemi_ids):
        # Add missing fields
        s.update(dict(dist=None, dist_limit=None, nearest=None, type='surf',
                      nearest_dist=None, pinfo=None, patch_inds=None, id=s_id,
                      coord_frame=np.array((FIFF.FIFFV_COORD_MRI,), np.int32)))
        s['rr'] /= 1000.0
        del s['tri_area']
        del s['tri_cent']
        del s['tri_nn']
        del s['neighbor_tri']

    # upconvert to object format from lists
    src = SourceSpaces(src, dict(working_dir=os.getcwd(), command_line=cmd))

    if add_dist:
        add_source_space_distances(src, verbose=verbose)

    # write out if requested, then return the data
    if fname is not None:
        write_source_spaces(fname, src)
        logger.info('Wrote %s' % fname)
    logger.info('You are now one step closer to computing the gain matrix')
    return src


@verbose
def setup_volume_source_space(subject, fname=None, pos=5.0, mri=None,
                              sphere=(0.0, 0.0, 0.0, 90.0), bem=None,
                              surface=None, mindist=5.0, exclude=0.0,
                              overwrite=False, subjects_dir=None,
                              volume_label=None, verbose=None):
    """Setup a volume source space with grid spacing or discrete source space

    Parameters
    ----------
    subject : str
        Subject to process.
    fname : str | None
        Filename to use. If None, the source space will not be saved
        (only returned).
    pos : float | dict
        Positions to use for sources. If float, a grid will be constructed
        with the spacing given by `pos` in mm, generating a volume source
        space. If dict, pos['rr'] and pos['nn'] will be used as the source
        space locations (in meters) and normals, respectively, creating a
        discrete source space. NOTE: For a discrete source space (`pos` is
        a dict), `mri` must be None.
    mri : str | None
        The filename of an MRI volume (mgh or mgz) to create the
        interpolation matrix over. Source estimates obtained in the
        volume source space can then be morphed onto the MRI volume
        using this interpolator. If pos is a dict, this can be None.
    sphere : array_like (length 4)
        Define spherical source space bounds using origin and radius given
        by (ox, oy, oz, rad) in mm. Only used if `bem` and `surface` are
        both None.
    bem : str | None
        Define source space bounds using a BEM file (specifically the inner
        skull surface).
    surface : str | dict | None
        Define source space bounds using a FreeSurfer surface file. Can
        also be a dictionary with entries `'rr'` and `'tris'`, such as
        those returned by `read_surface()`.
    mindist : float
        Exclude points closer than this distance (mm) to the bounding surface.
    exclude : float
        Exclude points closer than this distance (mm) from the center of mass
        of the bounding surface.
    overwrite: bool
        If True, overwrite output file (if it exists).
    subjects_dir : string, or None
        Path to SUBJECTS_DIR if it is not set in the environment.
    volume_label : str | None
        Region of interest corresponding with freesurfer lookup table.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    src : list
        The source space. Note that this list will have length 1 for
        compatibility reasons, as most functions expect source spaces
        to be provided as lists).

    Notes
    -----
    To create a discrete source space, `pos` must be a dict, 'mri' must be
    None, and 'volume_label' must be None. To create a whole brain volume
    source space, `pos` must be a float and 'mri' must be provided. To create
    a volume source space from label, 'pos' must be a float, 'volume_label'
    must be provided, and 'mri' must refer to a .mgh or .mgz file with values
    corresponding to the freesurfer lookup-table (typically aseg.mgz).
    """

    subjects_dir = get_subjects_dir(subjects_dir)

    if bem is not None and surface is not None:
        raise ValueError('Only one of "bem" and "surface" should be '
                         'specified')
    if mri is not None:
        if not op.isfile(mri):
            raise IOError('mri file "%s" not found' % mri)
        if isinstance(pos, dict):
            raise ValueError('Cannot create interpolation matrix for '
                             'discrete source space, mri must be None if '
                             'pos is a dict')
    elif not isinstance(pos, dict):
        # "pos" will create a discrete src, so we don't need "mri"
        # if "pos" is None, we must have "mri" b/c it will be vol src
        raise RuntimeError('"mri" must be provided if "pos" is not a dict '
                           '(i.e., if a volume instead of discrete source '
                           'space is desired)')

    if volume_label is not None:
        if isinstance(pos, dict):
            raise ValueError('If volume label is not none, pos must be '
                             'float, not a dict.')
        # Check that volume label is found in .mgz file
        volume_labels = get_volume_labels_from_aseg(mri)
        if volume_label not in volume_labels:
            raise ValueError('Volume %s not found in file %s. Double check '
                             'freesurfer lookup table.' % (volume_label, mri))

    sphere = np.asarray(sphere)
    if sphere.size != 4:
        raise ValueError('"sphere" must be array_like with 4 elements')

    # triage bounding argument
    if bem is not None:
        logger.info('BEM file              : %s', bem)
    elif surface is not None:
        if isinstance(surface, dict):
            if not all([key in surface for key in ['rr', 'tris']]):
                raise KeyError('surface, if dict, must have entries "rr" '
                               'and "tris"')
            # let's make sure we have geom info
            surface = _read_surface_geom(surface, verbose=False)
            surf_extra = 'dict()'
        elif isinstance(surface, string_types):
            if not op.isfile(surface):
                raise IOError('surface file "%s" not found' % surface)
            surf_extra = surface
        logger.info('Boundary surface file : %s', surf_extra)
    else:
        logger.info('Sphere                : origin at (%.1f %.1f %.1f) mm'
                    % (sphere[0], sphere[1], sphere[2]))
        logger.info('              radius  : %.1f mm' % sphere[3])

    # triage pos argument
    if isinstance(pos, dict):
        if not all([key in pos for key in ['rr', 'nn']]):
            raise KeyError('pos, if dict, must contain "rr" and "nn"')
        pos_extra = 'dict()'
    else:  # pos should be float-like
        try:
            pos = float(pos)
        except (TypeError, ValueError):
            raise ValueError('pos must be a dict, or something that can be '
                             'cast to float()')
    if not isinstance(pos, float):
        logger.info('Source location file  : %s', pos_extra)
        logger.info('Assuming input in millimeters')
        logger.info('Assuming input in MRI coordinates')

    logger.info('Output file           : %s', fname)
    if isinstance(pos, float):
        logger.info('grid                  : %.1f mm' % pos)
        logger.info('mindist               : %.1f mm' % mindist)
        pos /= 1000.0  # convert pos from m to mm
    if exclude > 0.0:
        logger.info('Exclude               : %.1f mm' % exclude)
    if mri is not None:
        logger.info('MRI volume            : %s' % mri)
    exclude /= 1000.0  # convert exclude from m to mm
    logger.info('')

    # Explicit list of points
    if not isinstance(pos, float):
        # Make the grid of sources
        sp = _make_discrete_source_space(pos)
    else:
        # Load the brain surface as a template
        if bem is not None:
            # read bem surface in the MRI coordinate frame
            surf = read_bem_surfaces(bem, s_id=FIFF.FIFFV_BEM_SURF_ID_BRAIN,
                                     verbose=False)
            logger.info('Loaded inner skull from %s (%d nodes)'
                        % (bem, surf['np']))
        elif surface is not None:
            if isinstance(surface, string_types):
                # read the surface in the MRI coordinate frame
                surf = _read_surface_geom(surface)
            else:
                surf = surface
            logger.info('Loaded bounding surface from %s (%d nodes)'
                        % (surface, surf['np']))
            surf = deepcopy(surf)
            surf['rr'] *= 1e-3  # must be converted to meters
        else:  # Load an icosahedron and use that as the surface
            logger.info('Setting up the sphere...')
            surf = _get_ico_surface(3)

            # Scale and shift

            # center at origin and make radius 1
            _normalize_vectors(surf['rr'])

            # normalize to sphere (in MRI coord frame)
            surf['rr'] *= sphere[3] / 1000.0  # scale by radius
            surf['rr'] += sphere[:3] / 1000.0  # move by center
            _complete_surface_info(surf, True)
        # Make the grid of sources in MRI space
        sp = _make_volume_source_space(surf, pos, exclude, mindist, mri,
                                       volume_label)

    # Compute an interpolation matrix to show data in MRI_VOXEL coord frame
    if mri is not None:
        _add_interpolator(sp, mri)

    if 'vol_dims' in sp:
        del sp['vol_dims']

    # Save it
    sp.update(dict(nearest=None, dist=None, use_tris=None, patch_inds=None,
                   dist_limit=None, pinfo=None, ntri=0, nearest_dist=None,
                   nuse_tri=0, tris=None))
    sp = SourceSpaces([sp], dict(working_dir=os.getcwd(), command_line='None'))
    if fname is not None:
        write_source_spaces(fname, sp, verbose=False)
    return sp


def _make_voxel_ras_trans(move, ras, voxel_size):
    """Make a transformation from MRI_VOXEL to MRI surface RAS (i.e. MRI)"""
    assert voxel_size.ndim == 1
    assert voxel_size.size == 3
    rot = ras.T * voxel_size[np.newaxis, :]
    assert rot.ndim == 2
    assert rot.shape[0] == 3
    assert rot.shape[1] == 3
    trans = np.c_[np.r_[rot, np.zeros((1, 3))], np.r_[move, 1.0]]
    t = {'from': FIFF.FIFFV_MNE_COORD_MRI_VOXEL, 'to': FIFF.FIFFV_COORD_MRI,
         'trans': trans}
    return t


def _make_discrete_source_space(pos):
    """Use a discrete set of source locs/oris to make src space

    Parameters
    ----------
    pos : dict
        Must have entries "rr" and "nn". Data should be in meters.

    Returns
    -------
    src : dict
        The source space.
    """
    # process points
    rr = pos['rr'].copy()
    nn = pos['nn'].copy()
    if not (rr.ndim == nn.ndim == 2 and nn.shape[0] == nn.shape[0] and
            rr.shape[1] == nn.shape[1]):
        raise RuntimeError('"rr" and "nn" must both be 2D arrays with '
                           'the same number of rows and 3 columns')
    npts = rr.shape[0]
    _normalize_vectors(nn)
    nz = np.sum(np.sum(nn * nn, axis=1) == 0)
    if nz != 0:
        raise RuntimeError('%d sources have zero length normal' % nz)
    logger.info('Positions (in meters) and orientations')
    logger.info('%d sources' % npts)

    # Ready to make the source space
    coord_frame = FIFF.FIFFV_COORD_MRI
    sp = dict(coord_frame=coord_frame, type='discrete', nuse=npts, np=npts,
              inuse=np.ones(npts, int), vertno=np.arange(npts), rr=rr, nn=nn,
              id=-1)
    return sp


def _make_volume_source_space(surf, grid, exclude, mindist, mri, volume_label):
    """Make a source space which covers the volume bounded by surf"""

    # Figure out the grid size in the MRI coordinate frame
    mins = np.min(surf['rr'], axis=0)
    maxs = np.max(surf['rr'], axis=0)
    cm = np.mean(surf['rr'], axis=0)  # center of mass

    # Define the sphere which fits the surface
    maxdist = np.sqrt(np.max(np.sum((surf['rr'] - cm) ** 2, axis=1)))

    logger.info('Surface CM = (%6.1f %6.1f %6.1f) mm'
                % (1000 * cm[0], 1000 * cm[1], 1000 * cm[2]))
    logger.info('Surface fits inside a sphere with radius %6.1f mm'
                % (1000 * maxdist))
    logger.info('Surface extent:')
    for c, mi, ma in zip('xyz', mins, maxs):
        logger.info('    %s = %6.1f ... %6.1f mm' % (c, 1000 * mi, 1000 * ma))
    maxn = np.zeros(3, int)
    minn = np.zeros(3, int)
    for c in range(3):
        if maxs[c] > 0:
            maxn[c] = np.floor(np.abs(maxs[c]) / grid) + 1
        else:
            maxn[c] = -np.floor(np.abs(maxs[c]) / grid) - 1
        if mins[c] > 0:
            minn[c] = np.floor(np.abs(mins[c]) / grid) + 1
        else:
            minn[c] = -np.floor(np.abs(mins[c]) / grid) - 1

    logger.info('Grid extent:')
    for c, mi, ma in zip('xyz', minn, maxn):
        logger.info('    %s = %6.1f ... %6.1f mm'
                    % (c, 1000 * mi * grid, 1000 * ma * grid))

    # Now make the initial grid
    ns = maxn - minn + 1
    npts = np.prod(ns)
    nrow = ns[0]
    ncol = ns[1]
    nplane = nrow * ncol
    sp = dict(np=npts, rr=np.zeros((npts, 3)), nn=np.zeros((npts, 3)),
              inuse=np.ones(npts, int), type='vol', nuse=npts,
              coord_frame=FIFF.FIFFV_COORD_MRI, id=-1, shape=ns)
    sp['nn'][:, 2] = 1.0  # Source orientation is immaterial

    x = np.arange(minn[0], maxn[0] + 1)[np.newaxis, np.newaxis, :]
    y = np.arange(minn[1], maxn[1] + 1)[np.newaxis, :, np.newaxis]
    z = np.arange(minn[2], maxn[2] + 1)[:, np.newaxis, np.newaxis]
    z = np.tile(z, (1, ns[1], ns[0])).ravel()
    y = np.tile(y, (ns[2], 1, ns[0])).ravel()
    x = np.tile(x, (ns[2], ns[1], 1)).ravel()
    k = np.arange(npts)
    sp['rr'] = np.c_[x, y, z] * grid
    neigh = np.empty((26, npts), int)
    neigh.fill(-1)

    # Figure out each neighborhood:
    # 6-neighborhood first
    idxs = [z > minn[2], x < maxn[0], y < maxn[1],
            x > minn[0], y > minn[1], z < maxn[2]]
    offsets = [-nplane, 1, nrow, -1, -nrow, nplane]
    for n, idx, offset in zip(neigh[:6], idxs, offsets):
        n[idx] = k[idx] + offset

    # Then the rest to complete the 26-neighborhood

    # First the plane below
    idx1 = z > minn[2]

    idx2 = np.logical_and(idx1, x < maxn[0])
    neigh[6, idx2] = k[idx2] + 1 - nplane
    idx3 = np.logical_and(idx2, y < maxn[1])
    neigh[7, idx3] = k[idx3] + 1 + nrow - nplane

    idx2 = np.logical_and(idx1, y < maxn[1])
    neigh[8, idx2] = k[idx2] + nrow - nplane

    idx2 = np.logical_and(idx1, x > minn[0])
    idx3 = np.logical_and(idx2, y < maxn[1])
    neigh[9, idx3] = k[idx3] - 1 + nrow - nplane
    neigh[10, idx2] = k[idx2] - 1 - nplane
    idx3 = np.logical_and(idx2, y > minn[1])
    neigh[11, idx3] = k[idx3] - 1 - nrow - nplane

    idx2 = np.logical_and(idx1, y > minn[1])
    neigh[12, idx2] = k[idx2] - nrow - nplane
    idx3 = np.logical_and(idx2, x < maxn[0])
    neigh[13, idx3] = k[idx3] + 1 - nrow - nplane

    # Then the same plane
    idx1 = np.logical_and(x < maxn[0], y < maxn[1])
    neigh[14, idx1] = k[idx1] + 1 + nrow

    idx1 = x > minn[0]
    idx2 = np.logical_and(idx1, y < maxn[1])
    neigh[15, idx2] = k[idx2] - 1 + nrow
    idx2 = np.logical_and(idx1, y > minn[1])
    neigh[16, idx2] = k[idx2] - 1 - nrow

    idx1 = np.logical_and(y > minn[1], x < maxn[0])
    neigh[17, idx1] = k[idx1] + 1 - nrow - nplane

    # Finally one plane above
    idx1 = z < maxn[2]

    idx2 = np.logical_and(idx1, x < maxn[0])
    neigh[18, idx2] = k[idx2] + 1 + nplane
    idx3 = np.logical_and(idx2, y < maxn[1])
    neigh[19, idx3] = k[idx3] + 1 + nrow + nplane

    idx2 = np.logical_and(idx1, y < maxn[1])
    neigh[20, idx2] = k[idx2] + nrow + nplane

    idx2 = np.logical_and(idx1, x > minn[0])
    idx3 = np.logical_and(idx2, y < maxn[1])
    neigh[21, idx3] = k[idx3] - 1 + nrow + nplane
    neigh[22, idx2] = k[idx2] - 1 + nplane
    idx3 = np.logical_and(idx2, y > minn[1])
    neigh[23, idx3] = k[idx3] - 1 - nrow + nplane

    idx2 = np.logical_and(idx1, y > minn[1])
    neigh[24, idx2] = k[idx2] - nrow + nplane
    idx3 = np.logical_and(idx2, x < maxn[0])
    neigh[25, idx3] = k[idx3] + 1 - nrow + nplane

    logger.info('%d sources before omitting any.', sp['nuse'])

    # Exclude infeasible points
    dists = np.sqrt(np.sum((sp['rr'] - cm) ** 2, axis=1))
    bads = np.where(np.logical_or(dists < exclude, dists > maxdist))[0]
    sp['inuse'][bads] = False
    sp['nuse'] -= len(bads)
    logger.info('%d sources after omitting infeasible sources.', sp['nuse'])

    _filter_source_spaces(surf, mindist, None, [sp])
    logger.info('%d sources remaining after excluding the sources outside '
                'the surface and less than %6.1f mm inside.'
                % (sp['nuse'], mindist))

    # Restrict sources to volume of interest
    if volume_label is not None:
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("nibabel is required to read segmentation file.")

        logger.info('Selecting voxels from %s' % volume_label)

        # Read the segmentation data using nibabel
        mgz = nib.load(mri)
        mgz_hdr = mgz.get_header()
        mgz_data = mgz.get_data()

        # Read the freesurfer look up table
        lut_fname = op.join(op.split(__file__)[0], 'data',
                            'FreeSurferColorLUT.txt')
        lut_data = np.genfromtxt(lut_fname, dtype=None, usecols=(0, 1),
                                 names=['id', 'name'])

        # Get the numeric index for this volume label
        vol_id = lut_data['id'][lut_data['name'] == volume_label]

        # Get indices for this volume label in voxel space
        vox_bool = mgz_data == vol_id

        # Get the 3 dimensional indices in voxel space
        vox_xyz = np.array(np.where(vox_bool)).T

        # Transform to RAS coordinates
        # (use tkr normalization or volume won't align with surface sources)
        trans = mgz_hdr.get_vox2ras_tkr()
        # Convert transform from mm to m
        trans[:3] /= 1000.
        rr_voi = apply_trans(trans, vox_xyz)  # positions of VOI in RAS space
        # Filter out points too far from volume region voxels
        dists = _compute_nearest(rr_voi, sp['rr'], return_dists=True)[1]
        # Maximum distance from center of mass of a voxel to any of its corners
        maxdist = np.sqrt(((trans[:3, :3].sum(0) / 2.) ** 2).sum())
        bads = np.where(dists > maxdist)[0]

        # Update source info
        sp['inuse'][bads] = False
        sp['vertno'] = np.where(sp['inuse'] > 0)[0]
        sp['nuse'] = len(sp['vertno'])
        sp['seg_name'] = volume_label
        sp['mri_file'] = mri

        # Update log
        logger.info('%d sources remaining after excluding sources too far '
                    'from VOI voxels', sp['nuse'])

    # Omit unused vertices from the neighborhoods
    logger.info('Adjusting the neighborhood info...')
    # remove non source-space points
    log_inuse = sp['inuse'] > 0
    neigh[:, np.logical_not(log_inuse)] = -1
    # remove these points from neigh
    vertno = np.where(log_inuse)[0]
    sp['vertno'] = vertno
    old_shape = neigh.shape
    neigh = neigh.ravel()
    checks = np.where(neigh >= 0)[0]
    removes = np.logical_not(in1d(checks, vertno))
    neigh[checks[removes]] = -1
    neigh.shape = old_shape
    neigh = neigh.T
    # Thought we would need this, but C code keeps -1 vertices, so we will:
    # neigh = [n[n >= 0] for n in enumerate(neigh[vertno])]
    sp['neighbor_vert'] = neigh

    # Set up the volume data (needed for creating the interpolation matrix)
    r0 = minn * grid
    voxel_size = grid * np.ones(3)
    ras = np.eye(3)
    sp['src_mri_t'] = _make_voxel_ras_trans(r0, ras, voxel_size)
    sp['vol_dims'] = maxn - minn + 1
    return sp


def _vol_vertex(width, height, jj, kk, pp):
    return jj + width * kk + pp * (width * height)


def _get_mgz_header(fname):
    """Adapted from nibabel to quickly extract header info"""
    if not fname.endswith('.mgz'):
        raise IOError('Filename must end with .mgz')
    header_dtd = [('version', '>i4'), ('dims', '>i4', (4,)),
                  ('type', '>i4'), ('dof', '>i4'), ('goodRASFlag', '>i2'),
                  ('delta', '>f4', (3,)), ('Mdc', '>f4', (3, 3)),
                  ('Pxyz_c', '>f4', (3,))]
    header_dtype = np.dtype(header_dtd)
    with gzip_open(fname, 'rb') as fid:
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
    M = linalg.inv(M)
    header = dict(dims=dims, vox2ras_tkr=v2rtkr, ras2vox=M)
    return header


def _add_interpolator(s, mri_name):
    """Compute a sparse matrix to interpolate the data into an MRI volume"""
    # extract transformation information from mri
    logger.info('Reading %s...' % mri_name)
    header = _get_mgz_header(mri_name)
    mri_width, mri_height, mri_depth = header['dims']

    s.update(dict(mri_width=mri_width, mri_height=mri_height,
                  mri_depth=mri_depth))
    trans = header['vox2ras_tkr'].copy()
    trans[:3, :] /= 1000.0
    s['vox_mri_t'] = {'trans': trans, 'from': FIFF.FIFFV_MNE_COORD_MRI_VOXEL,
                      'to': FIFF.FIFFV_COORD_MRI}  # ras_tkr
    trans = linalg.inv(np.dot(header['vox2ras_tkr'], header['ras2vox']))
    trans[:3, 3] /= 1000.0
    s['mri_ras_t'] = {'trans': trans, 'from': FIFF.FIFFV_COORD_MRI,
                      'to': FIFF.FIFFV_MNE_COORD_RAS}  # ras

    _print_coord_trans(s['src_mri_t'], 'Source space : ')
    _print_coord_trans(s['vox_mri_t'], 'MRI volume : ')
    _print_coord_trans(s['mri_ras_t'], 'MRI volume : ')

    #
    # Convert MRI voxels from destination (MRI volume) to source (volume
    # source space subset) coordinates
    #
    combo_trans = combine_transforms(s['vox_mri_t'],
                                     invert_transform(s['src_mri_t']),
                                     FIFF.FIFFV_MNE_COORD_MRI_VOXEL,
                                     FIFF.FIFFV_MNE_COORD_MRI_VOXEL)
    combo_trans['trans'] = combo_trans['trans'].astype(np.float32)

    logger.info('Setting up interpolation...')

    # Take *all* MRI vertices...
    # This is equivalent to using mgrid and reshaping, but faster
    js = np.arange(mri_width, dtype=np.float32)
    js = np.tile(js[np.newaxis, np.newaxis, :],
                 (mri_depth, mri_height, 1)).ravel()
    ks = np.arange(mri_height, dtype=np.float32)
    ks = np.tile(ks[np.newaxis, :, np.newaxis],
                 (mri_depth, 1, mri_width)).ravel()
    ps = np.arange(mri_depth, dtype=np.float32)
    ps = np.tile(ps[:, np.newaxis, np.newaxis],
                 (1, mri_height, mri_width)).ravel()
    r0 = np.c_[js, ks, ps]
    # note we have the correct number of vertices
    assert len(r0) == mri_width * mri_height * mri_depth

    # ...and transform them from their MRI space into our source space's frame
    # (this is labeled as FIFFV_MNE_COORD_MRI_VOXEL, but it's really a subset
    # of the entire volume!)
    r0 = apply_trans(combo_trans['trans'], r0)
    rn = np.floor(r0).astype(int)
    maxs = (s['vol_dims'] - 1)[np.newaxis, :]
    good = np.logical_and(np.all(rn >= 0, axis=1), np.all(rn < maxs, axis=1))
    rn = rn[good]
    r0 = r0[good]
    # now we take each MRI voxel *in this space*, and figure out how to make
    # its value the weighted sum of voxels in the volume source space. This
    # is a 3D weighting scheme based (presumably) on the fact that we know
    # we're interpolating from one volumetric grid into another.
    jj = rn[:, 0]
    kk = rn[:, 1]
    pp = rn[:, 2]
    vss = np.empty((8, len(jj)), int)
    width = s['vol_dims'][0]
    height = s['vol_dims'][1]
    vss[0, :] = _vol_vertex(width, height, jj, kk, pp)
    vss[1, :] = _vol_vertex(width, height, jj + 1, kk, pp)
    vss[2, :] = _vol_vertex(width, height, jj + 1, kk + 1, pp)
    vss[3, :] = _vol_vertex(width, height, jj, kk + 1, pp)
    vss[4, :] = _vol_vertex(width, height, jj, kk, pp + 1)
    vss[5, :] = _vol_vertex(width, height, jj + 1, kk, pp + 1)
    vss[6, :] = _vol_vertex(width, height, jj + 1, kk + 1, pp + 1)
    vss[7, :] = _vol_vertex(width, height, jj, kk + 1, pp + 1)
    del jj, kk, pp
    uses = np.any(s['inuse'][vss], axis=0)

    verts = vss[:, uses].ravel()  # vertex (col) numbers in csr matrix
    row_idx = np.tile(np.where(good)[0][uses], (8, 1)).ravel()

    # figure out weights for each vertex
    r0 = r0[uses]
    rn = rn[uses]
    xf = r0[:, 0] - rn[:, 0].astype(np.float32)
    yf = r0[:, 1] - rn[:, 1].astype(np.float32)
    zf = r0[:, 2] - rn[:, 2].astype(np.float32)
    omxf = 1.0 - xf
    omyf = 1.0 - yf
    omzf = 1.0 - zf
    weights = np.concatenate([omxf * omyf * omzf,  # correspond to rows of vss
                              xf * omyf * omzf,
                              xf * yf * omzf,
                              omxf * yf * omzf,
                              omxf * omyf * zf,
                              xf * omyf * zf,
                              xf * yf * zf,
                              omxf * yf * zf])
    del xf, yf, zf, omxf, omyf, omzf

    # Compose the sparse matrix
    ij = (row_idx, verts)
    nvox = mri_width * mri_height * mri_depth
    interp = sparse.csr_matrix((weights, ij), shape=(nvox, s['np']))
    s['interpolator'] = interp
    s['mri_volume_name'] = mri_name
    logger.info(' %d/%d nonzero values [done]' % (len(weights), nvox))


@verbose
def _filter_source_spaces(surf, limit, mri_head_t, src, n_jobs=1,
                          verbose=None):
    """Remove all source space points closer than a given limit"""
    if src[0]['coord_frame'] == FIFF.FIFFV_COORD_HEAD and mri_head_t is None:
        raise RuntimeError('Source spaces are in head coordinates and no '
                           'coordinate transform was provided!')

    # How close are the source points to the surface?
    out_str = 'Source spaces are in '

    if src[0]['coord_frame'] == FIFF.FIFFV_COORD_HEAD:
        inv_trans = invert_transform(mri_head_t)
        out_str += 'head coordinates.'
    elif src[0]['coord_frame'] == FIFF.FIFFV_COORD_MRI:
        out_str += 'MRI coordinates.'
    else:
        out_str += 'unknown (%d) coordinates.' % src[0]['coord_frame']
    logger.info(out_str)
    out_str = 'Checking that the sources are inside the bounding surface'
    if limit > 0.0:
        out_str += ' and at least %6.1f mm away' % (limit)
    logger.info(out_str + ' (will take a few...)')

    for s in src:
        vertno = np.where(s['inuse'])[0]  # can't trust s['vertno'] this deep
        # Convert all points here first to save time
        r1s = s['rr'][vertno]
        if s['coord_frame'] == FIFF.FIFFV_COORD_HEAD:
            r1s = apply_trans(inv_trans['trans'], r1s)

        # Check that the source is inside surface (often the inner skull)
        x = _sum_solids_div(r1s, surf, n_jobs)
        outside = np.abs(x - 1.0) > 1e-5
        omit_outside = np.sum(outside)

        # vectorized nearest using BallTree (or cdist)
        omit = 0
        if limit > 0.0:
            dists = _compute_nearest(surf['rr'], r1s, return_dists=True)[1]
            close = np.logical_and(dists < limit / 1000.0,
                                   np.logical_not(outside))
            omit = np.sum(close)
            outside = np.logical_or(outside, close)
        s['inuse'][vertno[outside]] = False
        s['nuse'] -= (omit + omit_outside)
        s['vertno'] = np.where(s['inuse'])[0]

        if omit_outside > 0:
            extras = [omit_outside]
            extras += ['s', 'they are'] if omit_outside > 1 else ['', 'it is']
            logger.info('%d source space point%s omitted because %s '
                        'outside the inner skull surface.' % tuple(extras))
        if omit > 0:
            extras = [omit]
            extras += ['s'] if omit_outside > 1 else ['']
            extras += [limit]
            logger.info('%d source space point%s omitted because of the '
                        '%6.1f-mm distance limit.' % tuple(extras))
    logger.info('Thank you for waiting.')


def _sum_solids_div(fros, surf, n_jobs):
    """Compute sum of solid angles according to van Oosterom for all tris"""
    parallel, p_fun, _ = parallel_func(_get_solids, n_jobs)
    tot_angles = parallel(p_fun(surf['rr'][tris], fros)
                          for tris in np.array_split(surf['tris'], n_jobs))
    return np.sum(tot_angles, axis=0) / (2 * np.pi)


def _get_solids(tri_rrs, fros):
    """Helper for computing _sum_solids_div total angle in chunks"""
    # NOTE: This incorporates the division by 4PI that used to be separate
    tot_angle = np.zeros((len(fros)))
    for tri_rr in tri_rrs:
        v1 = fros - tri_rr[0]
        v2 = fros - tri_rr[1]
        v3 = fros - tri_rr[2]
        triple = np.sum(fast_cross_3d(v1, v2) * v3, axis=1)
        l1 = np.sqrt(np.sum(v1 * v1, axis=1))
        l2 = np.sqrt(np.sum(v2 * v2, axis=1))
        l3 = np.sqrt(np.sum(v3 * v3, axis=1))
        s = (l1 * l2 * l3 +
             np.sum(v1 * v2, axis=1) * l3 +
             np.sum(v1 * v3, axis=1) * l2 +
             np.sum(v2 * v3, axis=1) * l1)
        tot_angle -= np.arctan2(triple, s)
    return tot_angle


@verbose
def add_source_space_distances(src, dist_limit=np.inf, n_jobs=1, verbose=None):
    """Compute inter-source distances along the cortical surface

    This function will also try to add patch info for the source space.
    It will only occur if the ``dist_limit`` is sufficiently high that all
    points on the surface are within ``dist_limit`` of a point in the
    source space.

    Parameters
    ----------
    src : instance of SourceSpaces
        The source spaces to compute distances for.
    dist_limit : float
        The upper limit of distances to include (in meters).
        Note: if limit < np.inf, scipy > 0.13 (bleeding edge as of
        10/2013) must be installed.
    n_jobs : int
        Number of jobs to run in parallel. Will only use (up to) as many
        cores as there are source spaces.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    src : instance of SourceSpaces
        The original source spaces, with distance information added.
        The distances are stored in src[n]['dist'].
        Note: this function operates in-place.

    Notes
    -----
    Requires scipy >= 0.11 (> 0.13 for `dist_limit < np.inf`).

    This function can be memory- and CPU-intensive. On a high-end machine
    (2012) running 6 jobs in parallel, an ico-5 (10242 per hemi) source space
    takes about 10 minutes to compute all distances (`dist_limit = np.inf`).
    With `dist_limit = 0.007`, computing distances takes about 1 minute.

    We recommend computing distances once per source space and then saving
    the source space to disk, as the computed distances will automatically be
    stored along with the source space data for future use.
    """
    n_jobs = check_n_jobs(n_jobs)
    if not isinstance(src, SourceSpaces):
        raise ValueError('"src" must be an instance of SourceSpaces')
    if not np.isscalar(dist_limit):
        raise ValueError('limit must be a scalar')
    if not check_scipy_version('0.11'):
        raise RuntimeError('scipy >= 0.11 must be installed (or > 0.13 '
                           'if dist_limit < np.inf')

    if not all([s['type'] == 'surf' for s in src]):
        raise RuntimeError('Currently all source spaces must be of surface '
                           'type')

    if dist_limit < np.inf:
        # can't do introspection on dijkstra function because it's Cython,
        # so we'll just try quickly here
        try:
            sparse.csgraph.dijkstra(sparse.csr_matrix(np.zeros((2, 2))),
                                    limit=1.0)
        except TypeError:
            raise RuntimeError('Cannot use "limit < np.inf" unless scipy '
                               '> 0.13 is installed')

    parallel, p_fun, _ = parallel_func(_do_src_distances, n_jobs)
    min_dists = list()
    min_idxs = list()
    logger.info('Calculating source space distances (limit=%s mm)...'
                % (1000 * dist_limit))
    for s in src:
        connectivity = mesh_dist(s['tris'], s['rr'])
        d = parallel(p_fun(connectivity, s['vertno'], r, dist_limit)
                     for r in np.array_split(np.arange(len(s['vertno'])),
                                             n_jobs))
        # deal with indexing so we can add patch info
        min_idx = np.array([dd[1] for dd in d])
        min_dist = np.array([dd[2] for dd in d])
        midx = np.argmin(min_dist, axis=0)
        range_idx = np.arange(len(s['rr']))
        min_dist = min_dist[midx, range_idx]
        min_idx = min_idx[midx, range_idx]
        min_dists.append(min_dist)
        min_idxs.append(min_idx)
        # now actually deal with distances, convert to sparse representation
        d = np.concatenate([dd[0] for dd in d], axis=0)
        i, j = np.meshgrid(s['vertno'], s['vertno'])
        d = d.ravel()
        i = i.ravel()
        j = j.ravel()
        idx = d > 0
        d = sparse.csr_matrix((d[idx], (i[idx], j[idx])),
                              shape=(s['np'], s['np']), dtype=np.float32)
        s['dist'] = d
        s['dist_limit'] = np.array([dist_limit], np.float32)

    # Let's see if our distance was sufficient to allow for patch info
    if not any([np.any(np.isinf(md)) for md in min_dists]):
        # Patch info can be added!
        for s, min_dist, min_idx in zip(src, min_dists, min_idxs):
            s['nearest'] = min_idx
            s['nearest_dist'] = min_dist
            _add_patch_info(s)
    else:
        logger.info('Not adding patch information, dist_limit too small')
    return src


def _do_src_distances(con, vertno, run_inds, limit):
    """Helper to compute source space distances in chunks"""
    if limit < np.inf:
        func = partial(sparse.csgraph.dijkstra, limit=limit)
    else:
        func = sparse.csgraph.dijkstra
    chunk_size = 100  # save memory by chunking (only a little slower)
    lims = np.r_[np.arange(0, len(run_inds), chunk_size), len(run_inds)]
    n_chunks = len(lims) - 1
    d = np.empty((len(run_inds), len(vertno)))
    min_dist = np.empty((n_chunks, con.shape[0]))
    min_idx = np.empty((n_chunks, con.shape[0]), np.int32)
    range_idx = np.arange(con.shape[0])
    for li, (l1, l2) in enumerate(zip(lims[:-1], lims[1:])):
        idx = vertno[run_inds[l1:l2]]
        out = func(con, indices=idx)
        midx = np.argmin(out, axis=0)
        min_idx[li] = idx[midx]
        min_dist[li] = out[midx, range_idx]
        d[l1:l2] = out[:, vertno]
    midx = np.argmin(min_dist, axis=0)
    min_dist = min_dist[midx, range_idx]
    min_idx = min_idx[midx, range_idx]
    d[d == np.inf] = 0  # scipy will give us np.inf for uncalc. distances
    return d, min_idx, min_dist


def get_volume_labels_from_aseg(mgz_fname):
    """Returns a list of names of segmented volumes.

    Parameters
    ----------
    mgz_fname : str
        Filename to read. Typically aseg.mgz or some variant in the freesurfer
        pipeline.

    Returns
    -------
    label_names : list of str
        The names of segmented volumes included in this mgz file.
    """
    import nibabel as nib

    # Read the mgz file using nibabel
    mgz_data = nib.load(mgz_fname).get_data()

    # Read the freesurfer lookup table
    lut_fname = op.join(op.split(__file__)[0], 'data',
                        'FreeSurferColorLUT.txt')
    lut_data = np.genfromtxt(lut_fname, dtype=None, usecols=(0, 1),
                             names=['id', 'name'])

    # Get the unique label names
    label_names = [lut_data[lut_data['id'] == ii]['name'][0] for ii in
                   np.unique(mgz_data)]
    label_names = sorted(label_names, key=lambda n: n.lower())

    return label_names
