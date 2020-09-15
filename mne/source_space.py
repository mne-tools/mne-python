# Authors: Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

# Many of the computations in this code were derived from Matti Hämäläinen's
# C code.

from copy import deepcopy
from functools import partial
from gzip import GzipFile
import os
import os.path as op

import numpy as np
from scipy import sparse, linalg

from .io.constants import FIFF
from .io.meas_info import create_info
from .io.tree import dir_tree_find
from .io.tag import find_tag, read_tag
from .io.open import fiff_open
from .io.write import (start_block, end_block, write_int,
                       write_float_sparse_rcs, write_string,
                       write_float_matrix, write_int_matrix,
                       write_coord_trans, start_file, end_file, write_id)
from .bem import read_bem_surfaces, ConductorModel
from .fixes import _get_img_fdata
from .surface import (read_surface, _create_surf_spacing, _get_ico_surface,
                      _tessellate_sphere_surf, _get_surf_neighbors,
                      _normalize_vectors, _triangle_neighbors, mesh_dist,
                      complete_surface_info, _compute_nearest, fast_cross_3d,
                      _CheckInside)
from .utils import (get_subjects_dir, check_fname, logger, verbose,
                    _ensure_int, check_version, _get_call_line, warn,
                    _check_fname, _check_path_like, has_nibabel, _check_sphere,
                    _validate_type, _check_option, _is_numeric, _pl, _suggest,
                    object_size, sizeof_fmt)
from .parallel import parallel_func, check_n_jobs
from .transforms import (invert_transform, apply_trans, _print_coord_trans,
                         combine_transforms, _get_trans,
                         _coord_frame_name, Transform, _str_to_frame,
                         _ensure_trans, read_ras_mni_t)


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
        fname = op.join(op.dirname(__file__), 'data', 'FreeSurferColorLUT.txt')
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


def _get_lut_id(lut, label):
    """Convert a label to a LUT ID number."""
    assert isinstance(label, str)
    mask = (lut['name'] == label)
    assert mask.sum() == 1
    return lut['id'][mask]


_src_kind_dict = {
    'vol': 'volume',
    'surf': 'surface',
    'discrete': 'discrete',
}


class SourceSpaces(list):
    """Represent a list of source space.

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

    def __init__(self, source_spaces, info=None):  # noqa: D102
        # First check the types is actually a valid config
        _validate_type(source_spaces, list, 'source_spaces')
        super(SourceSpaces, self).__init__(source_spaces)  # list
        self.kind  # will raise an error if there is a problem
        if info is None:
            self.info = dict()
        else:
            self.info = dict(info)

    @property
    def kind(self):
        types = list()
        for si, s in enumerate(self):
            _validate_type(s, dict, 'source_spaces[%d]' % (si,))
            types.append(s.get('type', None))
            _check_option('source_spaces[%d]["type"]' % (si,),
                          types[-1], ('surf', 'discrete', 'vol'))
        if all(k == 'surf' for k in types[:2]):
            surf_check = 2
            if len(types) == 2:
                kind = 'surface'
            else:
                kind = 'mixed'
        else:
            surf_check = 0
            if all(k == 'discrete' for k in types):
                kind = 'discrete'
            else:
                kind = 'volume'
        if any(k == 'surf' for k in types[surf_check:]):
            raise RuntimeError('Invalid source space with kinds %s' % (types,))
        return kind

    @verbose
    def plot(self, head=False, brain=None, skull=None, subjects_dir=None,
             trans=None, verbose=None):
        """Plot the source space.

        Parameters
        ----------
        head : bool
            If True, show head surface.
        brain : bool | str
            If True, show the brain surfaces. Can also be a str for
            surface type (e.g., 'pial', same as True). Default is None,
            which means 'white' for surface source spaces and False otherwise.
        skull : bool | str | list of str | list of dict | None
            Whether to plot skull surface. If string, common choices would be
            'inner_skull', or 'outer_skull'. Can also be a list to plot
            multiple skull surfaces. If a list of dicts, each dict must
            contain the complete surface info (such as you get from
            :func:`mne.make_bem_model`). True is an alias of 'outer_skull'.
            The subjects bem and bem/flash folders are searched for the 'surf'
            files. Defaults to None, which is False for surface source spaces,
            and True otherwise.
        subjects_dir : str | None
            Path to SUBJECTS_DIR if it is not set in the environment.
        trans : str | 'auto' | dict | None
            The full path to the head<->MRI transform ``*-trans.fif`` file
            produced during coregistration. If trans is None, an identity
            matrix is assumed. This is only needed when the source space is in
            head coordinates.
        %(verbose_meth)s

        Returns
        -------
        fig : instance of mayavi.mlab.Figure
            The figure.
        """
        from .viz import plot_alignment

        surfaces = list()
        bem = None

        if brain is None:
            brain = 'white' if any(ss['type'] == 'surf'
                                   for ss in self) else False

        if isinstance(brain, str):
            surfaces.append(brain)
        elif brain:
            surfaces.append('brain')

        if skull is None:
            skull = False if self.kind == 'surface' else True

        if isinstance(skull, str):
            surfaces.append(skull)
        elif skull is True:
            surfaces.append('outer_skull')
        elif skull is not False:  # list
            if isinstance(skull[0], dict):  # bem
                skull_map = {FIFF.FIFFV_BEM_SURF_ID_BRAIN: 'inner_skull',
                             FIFF.FIFFV_BEM_SURF_ID_SKULL: 'outer_skull',
                             FIFF.FIFFV_BEM_SURF_ID_HEAD: 'outer_skin'}
                for this_skull in skull:
                    surfaces.append(skull_map[this_skull['id']])
                bem = skull
            else:  # list of str
                for surf in skull:
                    surfaces.append(surf)

        if head:
            surfaces.append('head')

        if self[0]['coord_frame'] == FIFF.FIFFV_COORD_HEAD:
            coord_frame = 'head'
            if trans is None:
                raise ValueError('Source space is in head coordinates, but no '
                                 'head<->MRI transform was given. Please '
                                 'specify the full path to the appropriate '
                                 '*-trans.fif file as the "trans" parameter.')
        else:
            coord_frame = 'mri'

        info = create_info(0, 1000., 'eeg')

        return plot_alignment(
            info, trans=trans, subject=self._subject,
            subjects_dir=subjects_dir, surfaces=surfaces,
            coord_frame=coord_frame, meg=(), eeg=False, dig=False, ecog=False,
            bem=bem, src=self
        )

    def __getitem__(self, *args, **kwargs):
        """Get an item."""
        out = super().__getitem__(*args, **kwargs)
        if isinstance(out, list):
            out = SourceSpaces(out)
        return out

    def __repr__(self):  # noqa: D105
        ss_repr = []
        extra = []
        for si, ss in enumerate(self):
            ss_type = ss['type']
            r = _src_kind_dict[ss_type]
            if ss_type == 'vol':
                if 'seg_name' in ss:
                    r += " (%s)" % (ss['seg_name'],)
                else:
                    r += ", shape=%s" % (ss['shape'],)
            elif ss_type == 'surf':
                r += (" (%s), n_vertices=%i" % (_get_hemi(ss)[0], ss['np']))
            r += ', n_used=%i' % (ss['nuse'],)
            if si == 0:
                extra += ['%s coords'
                          % (_coord_frame_name(int(ss['coord_frame'])))]
            ss_repr.append('<%s>' % r)
        subj = self._subject
        if subj is not None:
            extra += ['subject %r' % (subj,)]
        sz = object_size(self)
        if sz is not None:
            extra += [f'~{sizeof_fmt(sz)}']
        return "<SourceSpaces: [%s] %s>" % (
            ', '.join(ss_repr), ', '.join(extra))

    @property
    def _subject(self):
        return self[0].get('subject_his_id', None)

    def __add__(self, other):
        """Combine source spaces."""
        out = self.copy()
        out += other
        return SourceSpaces(out)

    def copy(self):
        """Make a copy of the source spaces.

        Returns
        -------
        src : instance of SourceSpaces
            The copied source spaces.
        """
        return deepcopy(self)

    def save(self, fname, overwrite=False):
        """Save the source spaces to a fif file.

        Parameters
        ----------
        fname : str
            File to write.
        overwrite : bool
            If True, the destination file (if it exists) will be overwritten.
            If False (default), an error will be raised if the file exists.
        """
        write_source_spaces(fname, self, overwrite)

    @verbose
    def export_volume(self, fname, include_surfaces=True,
                      include_discrete=True, dest='mri', trans=None,
                      mri_resolution=False, use_lut=True, overwrite=False,
                      verbose=None):
        """Export source spaces to nifti or mgz file.

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
            or an info dict (usually opened using read_trans()). If string, an
            ending of ``.fif`` or ``.fif.gz`` will be assumed to be in FIF
            format, any other ending will be assumed to be a text file with a
            4x4 transformation matrix (like the ``--trans`` MNE-C option.
            Must be provided if source spaces are in head coordinates and
            include_surfaces and mri_resolution are True.
        mri_resolution : bool | str
            If True, the image is saved in MRI resolution
            (e.g. 256 x 256 x 256), and each source region (surface or
            segmentation volume) filled in completely. If "sparse", only a
            single voxel in the high-resolution MRI is filled in for each
            source point.

            .. versionchanged:: 0.21.0
               Support for "sparse" was added.
        use_lut : bool
            If True, assigns a numeric value to each source space that
            corresponds to a color on the freesurfer lookup table.
        overwrite : bool
            If True, overwrite the file if it exists.

            .. versionadded:: 0.19
        %(verbose_meth)s

        Notes
        -----
        This method requires nibabel.
        """
        _check_fname(fname, overwrite)
        _validate_type(mri_resolution, (bool, str), 'mri_resolution')
        if isinstance(mri_resolution, str):
            _check_option('mri_resolution', mri_resolution, ["sparse"],
                          extra='when mri_resolution is a string')
        else:
            mri_resolution = bool(mri_resolution)
        fname = str(fname)
        # import nibabel or raise error
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError('This function requires nibabel.')

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
        lut = _get_lut()

        # Setup a dictionary of source types
        src_types = dict(volume=[], surface_discrete=[])

        # Populate dictionary of source types
        for src in self:
            # volume sources
            if src['type'] == 'vol':
                src_types['volume'].append(src)
            # surface and discrete sources
            elif src['type'] in ('surf', 'discrete'):
                src_types['surface_discrete'].append(src)
            else:
                raise ValueError('Unrecognized source type: %s.' % src['type'])

        # Raise error if there are no volume source spaces
        if len(src_types['volume']) == 0:
            raise ValueError('Source spaces must contain at least one volume.')

        # Get shape, inuse array and interpolation matrix from volume sources
        src = src_types['volume'][0]
        aseg_data = None
        if mri_resolution:
            # read the mri file used to generate volumes
            if mri_resolution is True:
                aseg_data = _get_img_fdata(nib.load(src['mri_file']))
            # get the voxel space shape
            shape3d = (src['mri_width'], src['mri_depth'],
                       src['mri_height'])
        else:
            # get the volume source space shape
            # read the shape in reverse order
            # (otherwise results are scrambled)
            shape3d = src['shape']

        # calculate affine transform for image (MRI_VOXEL to RAS)
        if mri_resolution:
            # MRI_VOXEL to MRI transform
            transform = src['vox_mri_t']
        else:
            # MRI_VOXEL to MRI transform
            # NOTE: 'src' indicates downsampled version of MRI_VOXEL
            transform = src['src_mri_t']

        # Figure out how to get from our input source space to output voxels
        fro_dst_t = invert_transform(transform)
        dest = transform['to']
        if coords == 'head':
            head_mri_t = _get_trans(trans, 'head', 'mri')[0]
            fro_dst_t = combine_transforms(head_mri_t, fro_dst_t, 'head', dest)
        else:
            fro_dst_t = fro_dst_t

        # Fill in the volumes
        img = np.zeros(shape3d)
        for ii, vs in enumerate(src_types['volume']):
            # read the lookup table value for segmented volume
            if 'seg_name' not in vs:
                raise ValueError('Volume sources should be segments, '
                                 'not the entire volume.')
            # find the color value for this volume
            use_id = 1.
            if mri_resolution is True or use_lut:
                id_ = _get_lut_id(lut, vs['seg_name'])
                if use_lut:
                    use_id = id_

            if mri_resolution == 'sparse':
                idx = apply_trans(fro_dst_t, vs['rr'][vs['vertno']])
                idx = tuple(idx.round().astype(int).T)
            elif mri_resolution is True:  # fill the represented vol
                # get the values for this volume
                idx = (aseg_data == id_)
            else:
                assert mri_resolution is False
                idx = vs['inuse'].reshape(shape3d, order='F').astype(bool)
            img[idx] = use_id

        # loop through the surface and discrete source spaces

        # get the surface names (assumes left, right order. may want
        # to add these names during source space generation
        for src in src_types['surface_discrete']:
            val = 1
            if src['type'] == 'surf':
                if not include_surfaces:
                    continue
                if use_lut:
                    surf_name = {
                        FIFF.FIFFV_MNE_SURF_LEFT_HEMI: 'Left',
                        FIFF.FIFFV_MNE_SURF_RIGHT_HEMI: 'Right',
                    }[src['id']] + '-Cerebral-Cortex'
                    val = _get_lut_id(lut, surf_name)
            else:
                assert src['type'] == 'discrete'
                if not include_discrete:
                    continue
                if use_lut:
                    logger.info('Discrete sources do not have values on '
                                'the lookup table. Defaulting to 1.')
            # convert vertex positions from their native space
            # (either HEAD or MRI) to MRI_VOXEL space
            if mri_resolution is True:
                use_rr = src['rr']
            else:
                assert mri_resolution is False or mri_resolution == 'sparse'
                use_rr = src['rr'][src['vertno']]
            srf_vox = apply_trans(fro_dst_t['trans'], use_rr)
            # convert to numeric indices
            ix_, iy_, iz_ = srf_vox.T.round().astype(int)
            # clip indices outside of volume space
            ix = np.clip(ix_, 0, shape3d[0] - 1),
            iy = np.clip(iy_, 0, shape3d[1] - 1)
            iz = np.clip(iz_, 0, shape3d[2] - 1)
            # compare original and clipped indices
            n_diff = ((ix_ != ix) | (iy_ != iy) | (iz_ != iz)).sum()
            # generate use warnings for clipping
            if n_diff > 0:
                warn(f'{n_diff} {src["type"]} vertices lay outside of volume '
                     f'space. Consider using a larger volume space.')
            # get surface id or use default value
            # update image to include surface voxels
            img[ix, iy, iz] = val

        if dest == 'mri':
            # combine with MRI to RAS transform
            transform = combine_transforms(
                transform, vs['mri_ras_t'],
                transform['from'], vs['mri_ras_t']['to'])
        # now setup the affine for volume image
        affine = transform['trans'].copy()
        # make sure affine converts from m to mm
        affine[:3] *= 1e3

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
    """Patch information in a source space.

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
def _read_source_spaces_from_tree(fid, tree, patch_stats=False, verbose=None):
    """Read the source spaces from a FIF file.

    Parameters
    ----------
    fid : file descriptor
        An open file descriptor.
    tree : dict
        The FIF tree structure if source is a file id.
    patch_stats : bool, optional (default False)
        Calculate and add cortical patch statistics to the surfaces.
    %(verbose)s

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
        if patch_stats:
            _complete_source_space_info(this)

        src.append(this)

    logger.info('    %d source spaces read' % len(spaces))
    return SourceSpaces(src)


@verbose
def read_source_spaces(fname, patch_stats=False, verbose=None):
    """Read the source spaces from a FIF file.

    Parameters
    ----------
    fname : str
        The name of the file, which should end with -src.fif or
        -src.fif.gz.
    patch_stats : bool, optional (default False)
        Calculate and add cortical patch statistics to the surfaces.
    %(verbose)s

    Returns
    -------
    src : SourceSpaces
        The source spaces.

    See Also
    --------
    write_source_spaces, setup_source_space, setup_volume_source_space
    """
    # be more permissive on read than write (fwd/inv can contain src)
    check_fname(fname, 'source space', ('-src.fif', '-src.fif.gz',
                                        '_src.fif', '_src.fif.gz',
                                        '-fwd.fif', '-fwd.fif.gz',
                                        '_fwd.fif', '_fwd.fif.gz',
                                        '-inv.fif', '-inv.fif.gz',
                                        '_inv.fif', '_inv.fif.gz'))

    ff, tree, _ = fiff_open(fname)
    with ff as fid:
        src = _read_source_spaces_from_tree(fid, tree, patch_stats=patch_stats,
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


def _read_one_source_space(fid, this):
    """Read one source space."""
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

        tag = find_tag(fid, mri, FIFF.FIFF_MNE_FILE_NAME)
        if tag is not None:
            res['mri_volume_name'] = tag.data

        tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_NNEIGHBORS)
        if tag is not None:
            nneighbors = tag.data
            tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_NEIGHBORS)
            offset = 0
            neighbors = []
            for n in nneighbors:
                neighbors.append(tag.data[offset:offset + n])
                offset += n
            res['neighbor_vert'] = neighbors

        tag = find_tag(fid, this, FIFF.FIFF_COMMENT)
        if tag is not None:
            res['seg_name'] = tag.data

    tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_NPOINTS)
    if tag is None:
        raise ValueError('Number of vertices not found')

    res['np'] = int(tag.data)

    tag = find_tag(fid, this, FIFF.FIFF_BEM_SURF_NTRI)
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

    res['coord_frame'] = tag.data[0]

    #   Vertices, normals, and triangles
    tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_POINTS)
    if tag is None:
        raise ValueError('Vertex data not found')

    res['rr'] = tag.data.astype(np.float64)  # double precision for mayavi
    if res['rr'].shape[0] != res['np']:
        raise ValueError('Vertex information is incorrect')

    tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_NORMALS)
    if tag is None:
        raise ValueError('Vertex normals not found')

    res['nn'] = tag.data.copy()
    if res['nn'].shape[0] != res['np']:
        raise ValueError('Vertex normal information is incorrect')

    if res['ntri'] > 0:
        tag = find_tag(fid, this, FIFF.FIFF_BEM_SURF_TRIANGLES)
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
        res['inuse'] = np.zeros(res['nuse'], dtype=np.int64)
        res['vertno'] = None
    else:
        res['nuse'] = int(tag.data)
        tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_SELECTION)
        if tag is None:
            raise ValueError('Source selection information missing')

        res['inuse'] = tag.data.astype(np.int64).T
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
    if tag is None:
        res['subject_his_id'] = None
    else:
        res['subject_his_id'] = tag.data

    return res


@verbose
def _complete_source_space_info(this, verbose=None):
    """Add more info on surface."""
    #   Main triangulation
    logger.info('    Completing triangulation info...')
    this['tri_area'] = np.zeros(this['ntri'])
    r1 = this['rr'][this['tris'][:, 0], :]
    r2 = this['rr'][this['tris'][:, 1], :]
    r3 = this['rr'][this['tris'][:, 2], :]
    this['tri_cent'] = (r1 + r2 + r3) / 3.0
    this['tri_nn'] = fast_cross_3d((r2 - r1), (r3 - r1))
    this['tri_area'] = _normalize_vectors(this['tri_nn']) / 2.0
    logger.info('[done]')

    #   Selected triangles
    logger.info('    Completing selection triangulation info...')
    if this['nuse_tri'] > 0:
        r1 = this['rr'][this['use_tris'][:, 0], :]
        r2 = this['rr'][this['use_tris'][:, 1], :]
        r3 = this['rr'][this['use_tris'][:, 2], :]
        this['use_tri_cent'] = (r1 + r2 + r3) / 3.0
        this['use_tri_nn'] = fast_cross_3d((r2 - r1), (r3 - r1))
        this['use_tri_area'] = np.linalg.norm(this['use_tri_nn'], axis=1) / 2.
    logger.info('[done]')


def find_source_space_hemi(src):
    """Return the hemisphere id for a source space.

    Parameters
    ----------
    src : dict
        The source space to investigate.

    Returns
    -------
    hemi : int
        Deduced hemisphere id.
    """
    xave = src['rr'][:, 0].sum()

    if xave < 0:
        hemi = int(FIFF.FIFFV_MNE_SURF_LEFT_HEMI)
    else:
        hemi = int(FIFF.FIFFV_MNE_SURF_RIGHT_HEMI)

    return hemi


def label_src_vertno_sel(label, src):
    """Find vertex numbers and indices from label.

    Parameters
    ----------
    label : Label
        Source space label.
    src : dict
        Source space.

    Returns
    -------
    vertices : list of length 2
        Vertex numbers for lh and rh.
    src_sel : array of int (len(idx) = len(vertices[0]) + len(vertices[1]))
        Indices of the selected vertices in sourse space.
    """
    if src[0]['type'] != 'surf':
        return Exception('Labels are only supported with surface source '
                         'spaces')

    vertno = [src[0]['vertno'], src[1]['vertno']]

    if label.hemi == 'lh':
        vertno_sel = np.intersect1d(vertno[0], label.vertices)
        src_sel = np.searchsorted(vertno[0], vertno_sel)
        vertno[0] = vertno_sel
        vertno[1] = np.array([], int)
    elif label.hemi == 'rh':
        vertno_sel = np.intersect1d(vertno[1], label.vertices)
        src_sel = np.searchsorted(vertno[1], vertno_sel) + len(vertno[0])
        vertno[0] = np.array([], int)
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
    """Write the source spaces to a FIF file.

    Parameters
    ----------
    fid : file descriptor
        An open file descriptor.
    src : list
        The list of source spaces.
    %(verbose)s
    """
    for s in src:
        logger.info('    Write a source space...')
        start_block(fid, FIFF.FIFFB_MNE_SOURCE_SPACE)
        _write_one_source_space(fid, s, verbose)
        end_block(fid, FIFF.FIFFB_MNE_SOURCE_SPACE)
        logger.info('    [done]')
    logger.info('    %d source spaces written' % len(src))


@verbose
def write_source_spaces(fname, src, overwrite=False, verbose=None):
    """Write source spaces to a file.

    Parameters
    ----------
    fname : str
        The name of the file, which should end with -src.fif or
        -src.fif.gz.
    src : SourceSpaces
        The source spaces (as returned by read_source_spaces).
    overwrite : bool
        If True, the destination file (if it exists) will be overwritten.
        If False (default), an error will be raised if the file exists.
    %(verbose)s

    See Also
    --------
    read_source_spaces
    """
    check_fname(fname, 'source space', ('-src.fif', '-src.fif.gz',
                                        '_src.fif', '_src.fif.gz'))
    _check_fname(fname, overwrite=overwrite)

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
    """Write one source space."""
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
# Head to MRI volume conversion


@verbose
def head_to_mri(pos, subject, mri_head_t, subjects_dir=None,
                verbose=None):
    """Convert pos from head coordinate system to MRI ones.

    This function converts to MRI RAS coordinates and not to surface
    RAS.

    Parameters
    ----------
    pos : array, shape (n_pos, 3)
        The  coordinates (in m) in head coordinate system.
    subject : str
        Name of the subject.
    mri_head_t : instance of Transform
        MRI<->Head coordinate transformation.
    %(subjects_dir)s
    %(verbose)s

    Returns
    -------
    coordinates : array, shape (n_pos, 3)
        The MRI RAS coordinates (in mm) of pos.

    Notes
    -----
    This function requires nibabel.
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    t1_fname = op.join(subjects_dir, subject, 'mri', 'T1.mgz')
    head_mri_t = _ensure_trans(mri_head_t, 'head', 'mri')
    _, _, mri_ras_t, _, _ = _read_mri_info(t1_fname)
    head_ras_t = combine_transforms(head_mri_t, mri_ras_t, 'head', 'ras')
    return 1e3 * apply_trans(head_ras_t, pos)  # mm


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
    subject : str
        Name of the subject to load surfaces from.
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
    xfm = _read_talxfm(subject, subjects_dir)
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
    subject : str
        Name of the subject.
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
    head_mri_t = _ensure_trans(mri_head_t, 'head', 'mri')
    coo_MRI_RAS = apply_trans(head_mri_t, pos)

    # convert to MNI coordinates
    xfm = _read_talxfm(subject, subjects_dir)
    return apply_trans(xfm['trans'], coo_MRI_RAS * 1000)


@verbose
def _read_talxfm(subject, subjects_dir, verbose=None):
    """Compute MNI transform from FreeSurfer talairach.xfm file.

    Adapted from freesurfer m-files. Altered to deal with Norig
    and Torig correctly.
    """
    # Setup the RAS to MNI transform
    ras_mni_t = read_ras_mni_t(subject, subjects_dir)

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
    _, _, mri_ras_t, _, _ = _read_mri_info(path, units='mm')
    mri_mni_t = combine_transforms(mri_ras_t, ras_mni_t, 'mri', 'mni_tal')
    return mri_mni_t


def _read_mri_info(path, units='m', return_img=False):
    if has_nibabel():
        import nibabel
        mgz = nibabel.load(path)
        hdr = mgz.header
        n_orig = hdr.get_vox2ras()
        t_orig = hdr.get_vox2ras_tkr()
        dims = hdr.get_data_shape()
        zooms = hdr.get_zooms()[:3]
    else:
        mgz = None
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
        out += (mgz,)
    return out


###############################################################################
# Creation and decimation

@verbose
def _check_spacing(spacing, verbose=None):
    """Check spacing parameter."""
    # check to make sure our parameters are good, parse 'spacing'
    types = ('a string with values "ico#", "oct#", "all", or an int >= 2')
    space_err = ('"spacing" must be %s, got type %s (%r)'
                 % (types, type(spacing), spacing))
    if isinstance(spacing, str):
        if spacing == 'all':
            stype = 'all'
            sval = ''
        elif isinstance(spacing, str) and spacing[:3] in ('ico', 'oct'):
            stype = spacing[:3]
            sval = spacing[3:]
            try:
                sval = int(sval)
            except Exception:
                raise ValueError('%s subdivision must be an integer, got %r'
                                 % (stype, sval))
            lim = 0 if stype == 'ico' else 1
            if sval < lim:
                raise ValueError('%s subdivision must be >= %s, got %s'
                                 % (stype, lim, sval))
        else:
            raise ValueError(space_err)
    else:
        stype = 'spacing'
        sval = _ensure_int(spacing, 'spacing', types)
        if sval < 2:
            raise ValueError('spacing must be >= 2, got %d' % (sval,))
    if stype == 'all':
        logger.info('Include all vertices')
        ico_surf = None
        src_type_str = 'all'
    else:
        src_type_str = '%s = %s' % (stype, sval)
        if stype == 'ico':
            logger.info('Icosahedron subdivision grade %s' % sval)
            ico_surf = _get_ico_surface(sval)
        elif stype == 'oct':
            logger.info('Octahedron subdivision grade %s' % sval)
            ico_surf = _tessellate_sphere_surf(sval)
        else:
            assert stype == 'spacing'
            logger.info('Approximate spacing %s mm' % sval)
            ico_surf = sval
    return stype, sval, ico_surf, src_type_str


@verbose
def setup_source_space(subject, spacing='oct6', surface='white',
                       subjects_dir=None, add_dist=True, n_jobs=1,
                       verbose=None):
    """Set up bilateral hemisphere surface-based source space with subsampling.

    Parameters
    ----------
    subject : str
        Subject to process.
    spacing : str
        The spacing to use. Can be ``'ico#'`` for a recursively subdivided
        icosahedron, ``'oct#'`` for a recursively subdivided octahedron,
        ``'all'`` for all points, or an integer to use approximate
        distance-based spacing (in mm).

        .. versionchanged:: 0.18
           Support for integers for distance-based spacing.
    surface : str
        The surface to use.
    %(subjects_dir)s
    add_dist : bool | str
        Add distance and patch information to the source space. This takes some
        time so precomputing it is recommended. Can also be 'patch' to only
        compute patch information (requires SciPy 1.3+).

        .. versionchanged:: 0.20
           Support for add_dist='patch'.
    %(n_jobs)s
        Ignored if ``add_dist=='patch'``.
    %(verbose)s

    Returns
    -------
    src : SourceSpaces
        The source space for each hemisphere.

    See Also
    --------
    setup_volume_source_space
    """
    cmd = ('setup_source_space(%s, spacing=%s, surface=%s, '
           'subjects_dir=%s, add_dist=%s, verbose=%s)'
           % (subject, spacing, surface, subjects_dir, add_dist, verbose))

    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    surfs = [op.join(subjects_dir, subject, 'surf', hemi + surface)
             for hemi in ['lh.', 'rh.']]
    for surf, hemi in zip(surfs, ['LH', 'RH']):
        if surf is not None and not op.isfile(surf):
            raise IOError('Could not find the %s surface %s'
                          % (hemi, surf))

    logger.info('Setting up the source space with the following parameters:\n')
    logger.info('SUBJECTS_DIR = %s' % subjects_dir)
    logger.info('Subject      = %s' % subject)
    logger.info('Surface      = %s' % surface)
    stype, sval, ico_surf, src_type_str = _check_spacing(spacing)
    logger.info('')
    del spacing

    logger.info('>>> 1. Creating the source space...\n')

    # mne_make_source_space ... actually make the source spaces
    src = []

    # pre-load ico/oct surf (once) for speed, if necessary
    if stype not in ('spacing', 'all'):
        logger.info('Doing the %shedral vertex picking...'
                    % (dict(ico='icosa', oct='octa')[stype],))
    for hemi, surf in zip(['lh', 'rh'], surfs):
        logger.info('Loading %s...' % surf)
        # Setup the surface spacing in the MRI coord frame
        if stype != 'all':
            logger.info('Mapping %s %s -> %s (%d) ...'
                        % (hemi, subject, stype, sval))
        s = _create_surf_spacing(surf, hemi, subject, stype, ico_surf,
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
                      coord_frame=FIFF.FIFFV_COORD_MRI))
        s['rr'] /= 1000.0
        del s['tri_area']
        del s['tri_cent']
        del s['tri_nn']
        del s['neighbor_tri']

    # upconvert to object format from lists
    src = SourceSpaces(src, dict(working_dir=os.getcwd(), command_line=cmd))

    if add_dist:
        dist_limit = 0. if add_dist == 'patch' else np.inf
        add_source_space_distances(src, dist_limit=dist_limit,
                                   n_jobs=n_jobs, verbose=verbose)

    # write out if requested, then return the data
    logger.info('You are now one step closer to computing the gain matrix')
    return src


def _check_mri(mri, subject, subjects_dir):
    _validate_type(mri, 'path-like', 'mri')
    if not op.isfile(mri):
        if subject is None:
            raise FileNotFoundError(
                'MRI file %r not found and no subject provided' % (mri,))
        subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
        mri = op.join(subjects_dir, subject, 'mri', mri)
        if not op.isfile(mri):
            raise FileNotFoundError('MRI file %r not found' % (mri,))
    return mri


def _check_volume_labels(volume_label, mri, name='volume_label'):
    _validate_type(mri, 'path-like', 'mri when %s is not None' % (name,))
    mri = _check_fname(mri, overwrite='read', must_exist=True)
    if isinstance(volume_label, str):
        volume_label = [volume_label]
    _validate_type(volume_label, (list, tuple, dict), name)  # should be
    if not isinstance(volume_label, dict):
        # Turn it into a dict
        if not mri.endswith('aseg.mgz'):
            raise RuntimeError(
                'Must use a *aseg.mgz file unless %s is a dict, got %s'
                % (name, op.basename(mri)))
        lut, _ = read_freesurfer_lut()
        use_volume_label = dict()
        for label in volume_label:
            if label not in lut:
                raise ValueError(
                    'Volume %r not found in file %s. Double check '
                    'FreeSurfer lookup table.%s'
                    % (label, mri, _suggest(label, lut)))
            use_volume_label[label] = lut[label]
        volume_label = use_volume_label
    for label, id_ in volume_label.items():
        _validate_type(label, str, 'volume_label keys')
        _validate_type(id_, 'int-like', 'volume_labels[%r]' % (label,))
    volume_label = {k: _ensure_int(v) for k, v in volume_label.items()}
    return volume_label


@verbose
def setup_volume_source_space(subject=None, pos=5.0, mri=None,
                              sphere=None, bem=None,
                              surface=None, mindist=5.0, exclude=0.0,
                              subjects_dir=None, volume_label=None,
                              add_interpolator=True, sphere_units='m',
                              single_volume=False, verbose=None):
    """Set up a volume source space with grid spacing or discrete source space.

    Parameters
    ----------
    subject : str | None
        Subject to process. If None, the path to the MRI volume must be
        absolute to get a volume source space. If a subject name
        is provided the T1.mgz file will be found automatically.
        Defaults to None.
    pos : float | dict
        Positions to use for sources. If float, a grid will be constructed
        with the spacing given by ``pos`` in mm, generating a volume source
        space. If dict, pos['rr'] and pos['nn'] will be used as the source
        space locations (in meters) and normals, respectively, creating a
        discrete source space.

        .. note:: For a discrete source space (``pos`` is a dict),
                  ``mri`` must be None.
    mri : str | None
        The filename of an MRI volume (mgh or mgz) to create the
        interpolation matrix over. Source estimates obtained in the
        volume source space can then be morphed onto the MRI volume
        using this interpolator. If pos is a dict, this cannot be None.
        If subject name is provided, ``pos`` is a float or ``volume_label``
        are not provided then the ``mri`` parameter will default to 'T1.mgz'
        or ``aseg.mgz``, respectively, else it will stay None.
    sphere : ndarray, shape (4,) | ConductorModel | None
        Define spherical source space bounds using origin and radius given
        by (ox, oy, oz, rad) in ``sphere_units``.
        Only used if ``bem`` and ``surface`` are both None. Can also be a
        spherical ConductorModel, which will use the origin and radius.
        None (the default) uses a head-digitization fit.
    bem : str | None | ConductorModel
        Define source space bounds using a BEM file (specifically the inner
        skull surface) or a ConductorModel for a 1-layer of 3-layers BEM.
    surface : str | dict | None
        Define source space bounds using a FreeSurfer surface file. Can
        also be a dictionary with entries ``'rr'`` and ``'tris'``, such as
        those returned by :func:`mne.read_surface`.
    mindist : float
        Exclude points closer than this distance (mm) to the bounding surface.
    exclude : float
        Exclude points closer than this distance (mm) from the center of mass
        of the bounding surface.
    %(subjects_dir)s
    volume_label : str | dict | list | None
        Region(s) of interest to use. None (default) will create a single
        whole-brain source space. Otherwise, a separate source space will be
        created for each entry in the list or dict (str will be turned into
        a single-element list). If list of str, standard Freesurfer labels
        are assumed. If dict, should be a mapping of region names to atlas
        id numbers, allowing the use of other atlases.

        .. versionchanged:: 0.21.0
           Support for dict added.
    add_interpolator : bool
        If True and ``mri`` is not None, then an interpolation matrix
        will be produced.
    sphere_units : str
        Defaults to ``"m"``.

        .. versionadded:: 0.20
    single_volume : bool
        If True, multiple values of ``volume_label`` will be merged into a
        a single source space instead of occupying multiple source spaces
        (one for each sub-volume), i.e., ``len(src)`` will be ``1`` instead of
        ``len(volume_label)``. This can help conserve memory and disk space
        when many labels are used.

        .. versionadded:: 0.21
    %(verbose)s

    Returns
    -------
    src : SourceSpaces
        A :class:`SourceSpaces` object containing one source space for each
        entry of ``volume_labels``, or a single source space if
        ``volume_labels`` was not specified.

    See Also
    --------
    setup_source_space

    Notes
    -----
    Volume source spaces are related to an MRI image such as T1 and allow to
    visualize source estimates overlaid on MRIs and to morph estimates
    to a template brain for group analysis. Discrete source spaces
    don't allow this. If you provide a subject name the T1 MRI will be
    used by default.

    When you work with a source space formed from a grid you need to specify
    the domain in which the grid will be defined. There are three ways
    of specifying this:
    (i) sphere, (ii) bem model, and (iii) surface.
    The default behavior is to use sphere model
    (``sphere=(0.0, 0.0, 0.0, 90.0)``) if ``bem`` or ``surface`` is not
    ``None`` then ``sphere`` is ignored.
    If you're going to use a BEM conductor model for forward model
    it is recommended to pass it here.

    To create a discrete source space, ``pos`` must be a dict, ``mri`` must be
    None, and ``volume_label`` must be None. To create a whole brain volume
    source space, ``pos`` must be a float and 'mri' must be provided.

    To create a volume source space from label, ``pos`` must be a float,
    ``volume_label`` must be provided, and 'mri' must refer to a .mgh or .mgz
    file with values corresponding to the freesurfer lookup-table (typically
    ``aseg.mgz``).
    """
    subjects_dir = get_subjects_dir(subjects_dir)
    _validate_type(
        volume_label, (str, list, tuple, dict, None), 'volume_label')

    if bem is not None and surface is not None:
        raise ValueError('Only one of "bem" and "surface" should be '
                         'specified')

    if mri is None and subject is not None:
        if volume_label is not None:
            mri = 'aseg.mgz'
        elif _is_numeric(pos):
            mri = 'T1.mgz'

    if mri is not None:
        mri = _check_mri(mri, subject, subjects_dir)
        if isinstance(pos, dict):
            raise ValueError('Cannot create interpolation matrix for '
                             'discrete source space, mri must be None if '
                             'pos is a dict')

    if volume_label is not None:
        volume_label = _check_volume_labels(volume_label, mri)
    assert volume_label is None or isinstance(volume_label, dict)

    need_warn = sphere_units is None and not isinstance(sphere, ConductorModel)
    sphere = _check_sphere(sphere, sphere_units=sphere_units)

    # triage bounding argument
    if bem is not None:
        logger.info('BEM              : %s', bem)
    elif surface is not None:
        if isinstance(surface, dict):
            if not all(key in surface for key in ['rr', 'tris']):
                raise KeyError('surface, if dict, must have entries "rr" '
                               'and "tris"')
            # let's make sure we have geom info
            complete_surface_info(surface, copy=False, verbose=False)
            surf_extra = 'dict()'
        elif isinstance(surface, str):
            if not op.isfile(surface):
                raise IOError('surface file "%s" not found' % surface)
            surf_extra = surface
        logger.info('Boundary surface file : %s', surf_extra)
    else:
        logger.info('Sphere                : origin at (%.1f %.1f %.1f) mm'
                    % (1000 * sphere[0], 1000 * sphere[1], 1000 * sphere[2]))
        logger.info('              radius  : %.1f mm' % (1000 * sphere[3],))
        if need_warn:
            warn('sphere_units defaults to mm in 0.20 but will change to m in '
                 '0.21, set it explicitly to avoid this warning',
                 DeprecationWarning)

    # triage pos argument
    if isinstance(pos, dict):
        if not all(key in pos for key in ['rr', 'nn']):
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

    if isinstance(pos, float):
        logger.info('grid                  : %.1f mm' % pos)
        logger.info('mindist               : %.1f mm' % mindist)
        pos /= 1000.0  # convert pos from m to mm
    if exclude > 0.0:
        logger.info('Exclude               : %.1f mm' % exclude)
    vol_info = dict()
    if mri is not None:
        logger.info('MRI volume            : %s' % mri)
        logger.info('')
        logger.info('Reading %s...' % mri)
        vol_info = _get_mri_info_data(mri, data=volume_label is not None)

    exclude /= 1000.0  # convert exclude from m to mm
    logger.info('')

    # Explicit list of points
    if not isinstance(pos, float):
        # Make the grid of sources
        sp = [_make_discrete_source_space(pos)]
    else:
        # Load the brain surface as a template
        if isinstance(bem, str):
            # read bem surface in the MRI coordinate frame
            surf = read_bem_surfaces(bem, s_id=FIFF.FIFFV_BEM_SURF_ID_BRAIN,
                                     verbose=False)
            logger.info('Loaded inner skull from %s (%d nodes)'
                        % (bem, surf['np']))
        elif bem is not None and bem.get('is_sphere') is False:
            # read bem surface in the MRI coordinate frame
            which = np.where([surf['id'] == FIFF.FIFFV_BEM_SURF_ID_BRAIN
                              for surf in bem['surfs']])[0]
            if len(which) != 1:
                raise ValueError('Could not get inner skull surface from BEM')
            surf = bem['surfs'][which[0]]
            assert surf['id'] == FIFF.FIFFV_BEM_SURF_ID_BRAIN
            if surf['coord_frame'] != FIFF.FIFFV_COORD_MRI:
                raise ValueError('BEM is not in MRI coordinates, got %s'
                                 % (_coord_frame_name(surf['coord_frame']),))
            logger.info('Taking inner skull from %s' % bem)
        elif surface is not None:
            if isinstance(surface, str):
                # read the surface in the MRI coordinate frame
                surf = read_surface(surface, return_dict=True)[-1]
            else:
                surf = surface
            logger.info('Loaded bounding surface from %s (%d nodes)'
                        % (surface, surf['np']))
            surf = deepcopy(surf)
            surf['rr'] *= 1e-3  # must be converted to meters
        else:  # Load an icosahedron and use that as the surface
            logger.info('Setting up the sphere...')
            surf = dict(R=sphere[3], r0=sphere[:3])
        # Make the grid of sources in MRI space
        sp = _make_volume_source_space(
            surf, pos, exclude, mindist, mri, volume_label,
            vol_info=vol_info, single_volume=single_volume)
    del sphere
    assert isinstance(sp, list)
    assert len(sp) == 1 if (volume_label is None or
                            single_volume) else len(volume_label)

    # Compute an interpolation matrix to show data in MRI_VOXEL coord frame
    if mri is not None:
        _add_interpolator(sp, add_interpolator)
    elif sp[0]['type'] == 'vol':
        # If there is no interpolator, it's actually a discrete source space
        sp[0]['type'] = 'discrete'

    # do some cleaning
    if volume_label is None and 'seg_name' in sp[0]:
        del sp[0]['seg_name']
    for s in sp:
        if 'vol_dims' in s:
            del s['vol_dims']

    # Save it
    sp = _complete_vol_src(sp, subject)
    return sp


def _complete_vol_src(sp, subject=None):
    for s in sp:
        s.update(dict(nearest=None, dist=None, use_tris=None, patch_inds=None,
                      dist_limit=None, pinfo=None, ntri=0, nearest_dist=None,
                      nuse_tri=0, tris=None, subject_his_id=subject))

    sp = SourceSpaces(sp, dict(working_dir=os.getcwd(), command_line='None'))
    return sp


def _make_voxel_ras_trans(move, ras, voxel_size):
    """Make a transformation from MRI_VOXEL to MRI surface RAS (i.e. MRI)."""
    assert voxel_size.ndim == 1
    assert voxel_size.size == 3
    rot = ras.T * voxel_size[np.newaxis, :]
    assert rot.ndim == 2
    assert rot.shape[0] == 3
    assert rot.shape[1] == 3
    trans = np.c_[np.r_[rot, np.zeros((1, 3))], np.r_[move, 1.0]]
    t = Transform('mri_voxel', 'mri', trans)
    return t


def _make_discrete_source_space(pos, coord_frame='mri'):
    """Use a discrete set of source locs/oris to make src space.

    Parameters
    ----------
    pos : dict
        Must have entries "rr" and "nn". Data should be in meters.
    coord_frame : str
        The coordinate frame in which the positions are given; default: 'mri'.
        The frame must be one defined in transforms.py:_str_to_frame

    Returns
    -------
    src : dict
        The source space.
    """
    # Check that coordinate frame is valid
    if coord_frame not in _str_to_frame:  # will fail if coord_frame not string
        raise KeyError('coord_frame must be one of %s, not "%s"'
                       % (list(_str_to_frame.keys()), coord_frame))
    coord_frame = _str_to_frame[coord_frame]  # now an int

    # process points (copy and cast)
    rr = np.array(pos['rr'], float)
    nn = np.array(pos['nn'], float)
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
    sp = dict(coord_frame=coord_frame, type='discrete', nuse=npts, np=npts,
              inuse=np.ones(npts, int), vertno=np.arange(npts), rr=rr, nn=nn,
              id=-1)
    return sp


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


def _mri_orientation(img, orientation):
    """Get MRI orientation information from an image.

    Parameters
    ----------
    img : instance of SpatialImage
        The MRI image.
    orientation : str
        Orientation that you want. Can be "axial", "saggital", or "coronal".

    Returns
    -------
    xyz : tuple, shape (3,)
        The dimension indices for X, Y, and Z.
    flips : tuple, shape (3,)
        Whether each dimension requires a flip.
    order : tuple, shape (3,)
        The resulting order of the data if the given ``xyz`` and ``flips``
        are used.

    Notes
    -----
    .. versionadded:: 0.21
    """
    import nibabel as nib
    _validate_type(img, nib.spatialimages.SpatialImage)
    _check_option('orientation', orientation, ('coronal', 'axial', 'sagittal'))
    axcodes = ''.join(nib.orientations.aff2axcodes(img.affine))
    flips = {o: (1 if o in axcodes else -1) for o in 'RAS'}
    axcodes = axcodes.replace('L', 'R').replace('P', 'A').replace('I', 'S')
    order = dict(
        coronal=('R', 'S', 'A'),
        axial=('R', 'A', 'S'),
        sagittal=('A', 'S', 'R'),
    )[orientation]
    xyz = tuple(axcodes.index(c) for c in order)
    flips = tuple(flips[c] for c in order)
    return xyz, flips, order


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


def _get_atlas_values(vol_info, rr):
    # Transform MRI coordinates (where our surfaces live) to voxels
    rr_vox = apply_trans(vol_info['mri_vox_t'], rr)
    good = ((rr_vox >= -.5) &
            (rr_vox < np.array(vol_info['data'].shape, int) - 0.5)).all(-1)
    idx = np.round(rr_vox[good].T).astype(np.int64)
    values = np.full(rr.shape[0], np.nan)
    values[good] = vol_info['data'][tuple(idx)]
    return values


def _make_volume_source_space(surf, grid, exclude, mindist, mri=None,
                              volume_labels=None, do_neighbors=True, n_jobs=1,
                              vol_info={}, single_volume=False):
    """Make a source space which covers the volume bounded by surf."""
    # Figure out the grid size in the MRI coordinate frame
    if 'rr' in surf:
        mins = np.min(surf['rr'], axis=0)
        maxs = np.max(surf['rr'], axis=0)
        cm = np.mean(surf['rr'], axis=0)  # center of mass
        maxdist = np.linalg.norm(surf['rr'] - cm, axis=1).max()
    else:
        mins = surf['r0'] - surf['R']
        maxs = surf['r0'] + surf['R']
        cm = surf['r0'].copy()
        maxdist = surf['R']

    # Define the sphere which fits the surface
    logger.info('Surface CM = (%6.1f %6.1f %6.1f) mm'
                % (1000 * cm[0], 1000 * cm[1], 1000 * cm[2]))
    logger.info('Surface fits inside a sphere with radius %6.1f mm'
                % (1000 * maxdist))
    logger.info('Surface extent:')
    for c, mi, ma in zip('xyz', mins, maxs):
        logger.info('    %s = %6.1f ... %6.1f mm'
                    % (c, 1000 * mi, 1000 * ma))
    maxn = np.array([np.floor(np.abs(m) / grid) + 1 if m > 0 else -
                     np.floor(np.abs(m) / grid) - 1 for m in maxs], int)
    minn = np.array([np.floor(np.abs(m) / grid) + 1 if m > 0 else -
                     np.floor(np.abs(m) / grid) - 1 for m in mins], int)
    logger.info('Grid extent:')
    for c, mi, ma in zip('xyz', minn, maxn):
        logger.info('    %s = %6.1f ... %6.1f mm'
                    % (c, 1000 * mi * grid, 1000 * ma * grid))

    # Now make the initial grid
    ns = tuple(maxn - minn + 1)
    npts = np.prod(ns)
    nrow = ns[0]
    ncol = ns[1]
    nplane = nrow * ncol
    # x varies fastest, then y, then z (can use unravel to do this)
    rr = np.meshgrid(np.arange(minn[2], maxn[2] + 1),
                     np.arange(minn[1], maxn[1] + 1),
                     np.arange(minn[0], maxn[0] + 1), indexing='ij')
    x, y, z = rr[2].ravel(), rr[1].ravel(), rr[0].ravel()
    rr = np.array([x * grid, y * grid, z * grid]).T
    sp = dict(np=npts, nn=np.zeros((npts, 3)), rr=rr,
              inuse=np.ones(npts, int), type='vol', nuse=npts,
              coord_frame=FIFF.FIFFV_COORD_MRI, id=-1, shape=ns)
    sp['nn'][:, 2] = 1.0
    assert sp['rr'].shape[0] == npts

    logger.info('%d sources before omitting any.', sp['nuse'])

    # Exclude infeasible points
    dists = np.linalg.norm(sp['rr'] - cm, axis=1)
    bads = np.where(np.logical_or(dists < exclude, dists > maxdist))[0]
    sp['inuse'][bads] = False
    sp['nuse'] -= len(bads)
    logger.info('%d sources after omitting infeasible sources not within '
                '%0.1f - %0.1f mm.',
                sp['nuse'], 1000 * exclude, 1000 * maxdist)
    if 'rr' in surf:
        _filter_source_spaces(surf, mindist, None, [sp], n_jobs)
    else:  # sphere
        vertno = np.where(sp['inuse'])[0]
        bads = (np.linalg.norm(sp['rr'][vertno] - surf['r0'], axis=-1) >=
                surf['R'] - mindist / 1000.)
        sp['nuse'] -= bads.sum()
        sp['inuse'][vertno[bads]] = False
        sp['vertno'] = np.where(sp['inuse'])[0]
        del vertno
    del surf
    logger.info('%d sources remaining after excluding the sources outside '
                'the surface and less than %6.1f mm inside.'
                % (sp['nuse'], mindist))

    # Restrict sources to volume of interest
    if volume_labels is None:
        sp['seg_name'] = 'the whole brain'
        sps = [sp]
    else:
        if not do_neighbors:
            raise RuntimeError('volume_label cannot be None unless '
                               'do_neighbors is True')
        sps = list()
        orig_sp = sp
        for volume_label, id_ in volume_labels.items():
            sp = deepcopy(orig_sp)
            good = _get_atlas_values(vol_info, sp['rr'][sp['vertno']]) == id_
            n_good = good.sum()
            logger.info('    Selected %d voxel%s from %s'
                        % (n_good, _pl(n_good), volume_label))
            # Update source info
            sp['inuse'][sp['vertno'][~good]] = False
            sp['vertno'] = sp['vertno'][good]
            sp['nuse'] = sp['inuse'].sum()
            sp['seg_name'] = volume_label
            sp['mri_file'] = mri
            sps.append(sp)
        assert len(sps) == len(volume_labels)
        # This will undo some of the work above, but the calculations are
        # pretty trivial so allow it
        if single_volume:
            for sp in sps[1:]:
                sps[0]['inuse'][sp['vertno']] = True
            sp = sps[0]
            sp['seg_name'] = '+'.join(s['seg_name'] for s in sps)
            sps = sps[:1]
            sp['vertno'] = np.where(sp['inuse'])[0]
            sp['nuse'] = len(sp['vertno'])
    del sp, volume_labels
    if not do_neighbors:
        return sps

    k = np.arange(npts)
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

    # Omit unused vertices from the neighborhoods
    logger.info('Adjusting the neighborhood info.')
    r0 = minn * grid
    voxel_size = grid * np.ones(3)
    ras = np.eye(3)
    src_mri_t = _make_voxel_ras_trans(r0, ras, voxel_size)
    neigh_orig = neigh
    for sp in sps:
        # remove non source-space points
        neigh = neigh_orig.copy()
        neigh[:, np.logical_not(sp['inuse'])] = -1
        # remove these points from neigh
        old_shape = neigh.shape
        neigh = neigh.ravel()
        checks = np.where(neigh >= 0)[0]
        removes = np.logical_not(np.in1d(checks, sp['vertno']))
        neigh[checks[removes]] = -1
        neigh.shape = old_shape
        neigh = neigh.T
        # Thought we would need this, but C code keeps -1 vertices, so we will:
        # neigh = [n[n >= 0] for n in enumerate(neigh[vertno])]
        sp['neighbor_vert'] = neigh

        # Set up the volume data (needed for creating the interpolation matrix)
        sp['src_mri_t'] = src_mri_t
        sp['vol_dims'] = maxn - minn + 1
        for key in ('mri_width', 'mri_height', 'mri_depth', 'mri_volume_name',
                    'vox_mri_t', 'mri_ras_t'):
            if key in vol_info:
                sp[key] = vol_info[key]
    _print_coord_trans(sps[0]['src_mri_t'], 'Source space : ')
    for key in ('vox_mri_t', 'mri_ras_t'):
        if key in sps[0]:
            _print_coord_trans(sps[0][key], 'MRI volume : ')
    return sps


def _vol_vertex(width, height, jj, kk, pp):
    return jj + width * kk + pp * (width * height)


def _get_mgz_header(fname):
    """Adapted from nibabel to quickly extract header info."""
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


def _add_interpolator(sp, add_interpolator):
    """Compute a sparse matrix to interpolate the data into an MRI volume."""
    # extract transformation information from mri
    s = sp[0]
    mri_width = s['mri_width']
    mri_height = s['mri_height']
    mri_depth = s['mri_depth']
    nvox = mri_width * mri_height * mri_depth
    if not add_interpolator:
        for s in sp:
            s['interpolator'] = sparse.csr_matrix((nvox, s['np']))
        return

    #
    # Convert MRI voxels from destination (MRI volume) to source (volume
    # source space subset) coordinates
    #
    vol_dims = s['vol_dims']
    combo_trans = combine_transforms(s['vox_mri_t'],
                                     invert_transform(s['src_mri_t']),
                                     'mri_voxel', 'mri_voxel')
    del s
    combo_trans['trans'] = combo_trans['trans'].astype(np.float32)

    logger.info('Setting up volume interpolation ...')
    # Loop over slices to save (lots of) memory
    # Note that it is the slowest incrementing index
    # This is equivalent to using mgrid and reshaping, but faster
    datas = [list() for _ in range(len(sp))]
    indicess = [list() for _ in range(len(sp))]
    indptrs = [np.zeros(nvox + 1, np.int32) for _ in range(len(sp))]
    for p in range(mri_depth):
        js = np.arange(mri_width, dtype=np.float32)
        js = np.tile(js[np.newaxis, :],
                     (mri_height, 1)).ravel()
        ks = np.arange(mri_height, dtype=np.float32)
        ks = np.tile(ks[:, np.newaxis],
                     (1, mri_width)).ravel()
        ps = np.empty((mri_height, mri_width), np.float32).ravel()
        ps.fill(p)
        r0 = np.c_[js, ks, ps]
        del js, ks, ps

        # Transform our vertices from their MRI space into our source space's
        # frame (this is labeled as FIFFV_MNE_COORD_MRI_VOXEL, but it's
        # really a subset of the entire volume!)
        r0 = apply_trans(combo_trans['trans'], r0)
        rn = np.floor(r0).astype(int)
        maxs = (vol_dims - 1)[np.newaxis, :]
        good = np.where(np.logical_and(np.all(rn >= 0, axis=1),
                                       np.all(rn < maxs, axis=1)))[0]
        good.flags['WRITEABLE'] = False
        rns = rn[good]
        r0s = r0[good]
        del rn, r0

        # now we take each MRI voxel *in this space*, and figure out how
        # to make its value the weighted sum of voxels in the volume source
        # space. This is a 3D weighting scheme based (presumably) on the
        # fact that we know we're interpolating from one volumetric grid
        # into another.
        jj = rns[:, 0]
        kk = rns[:, 1]
        pp = rns[:, 2]
        vss = np.empty((len(jj), 8), np.int32)
        width = vol_dims[0]
        height = vol_dims[1]
        jjp1 = jj + 1
        kkp1 = kk + 1
        ppp1 = pp + 1
        vss[:, 0] = _vol_vertex(width, height, jj, kk, pp)
        vss[:, 1] = _vol_vertex(width, height, jjp1, kk, pp)
        vss[:, 2] = _vol_vertex(width, height, jjp1, kkp1, pp)
        vss[:, 3] = _vol_vertex(width, height, jj, kkp1, pp)
        vss[:, 4] = _vol_vertex(width, height, jj, kk, ppp1)
        vss[:, 5] = _vol_vertex(width, height, jjp1, kk, ppp1)
        vss[:, 6] = _vol_vertex(width, height, jjp1, kkp1, ppp1)
        vss[:, 7] = _vol_vertex(width, height, jj, kkp1, ppp1)
        vss.flags['WRITEABLE'] = False
        del jj, kk, pp, jjp1, kkp1, ppp1
        for si, s in enumerate(sp):
            uses = np.any(s['inuse'][vss], axis=1)
            if uses.size == 0:
                continue
            # vertex (col) numbers in csr matrix
            indicess[si].append(vss[uses].ravel())
            indptrs[si][good[uses] + p * mri_height * mri_width + 1] = 8

            # figure out weights for each vertex
            r0 = r0s[uses]
            rn = rns[uses]
            del uses
            xf = r0[:, 0] - rn[:, 0].astype(np.float32)
            yf = r0[:, 1] - rn[:, 1].astype(np.float32)
            zf = r0[:, 2] - rn[:, 2].astype(np.float32)
            omxf = 1.0 - xf
            omyf = 1.0 - yf
            omzf = 1.0 - zf
            # each entry in the concatenation corresponds to a row of vss
            datas[si].append(
                np.array([omxf * omyf * omzf,
                          xf * omyf * omzf,
                          xf * yf * omzf,
                          omxf * yf * omzf,
                          omxf * omyf * zf,
                          xf * omyf * zf,
                          xf * yf * zf,
                          omxf * yf * zf], order='F').T.ravel())
            del r0, rn, xf, yf, zf, omxf, omyf, omzf

    # Compose the sparse matrices
    for si, s in enumerate(sp):
        indptr = np.cumsum(indptrs[si], out=indptrs[si])
        indices = np.concatenate(indicess[si])
        data = np.concatenate(datas[si])
        s['interpolator'] = sparse.csr_matrix((data, indices, indptr),
                                              shape=(nvox, s['np']))
        logger.info('    %d/%d nonzero values for %s'
                    % (len(data), nvox, s['seg_name']))
    logger.info('[done]')


def _pts_in_hull(pts, hull, tolerance=1e-12):
    return np.all([np.dot(eq[:-1], pts.T) + eq[-1] <= tolerance
                   for eq in hull.equations], axis=0)


@verbose
def _filter_source_spaces(surf, limit, mri_head_t, src, n_jobs=1,
                          verbose=None):
    """Remove all source space points closer than a given limit (in mm)."""
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
    out_str = 'Checking that the sources are inside the surface'
    if limit > 0.0:
        out_str += ' and at least %6.1f mm away' % (limit)
    logger.info(out_str + ' (will take a few...)')

    # fit a sphere to a surf quickly
    check_inside = _CheckInside(surf)

    # Check that the source is inside surface (often the inner skull)
    for s in src:
        vertno = np.where(s['inuse'])[0]  # can't trust s['vertno'] this deep
        # Convert all points here first to save time
        r1s = s['rr'][vertno]
        if s['coord_frame'] == FIFF.FIFFV_COORD_HEAD:
            r1s = apply_trans(inv_trans['trans'], r1s)

        inside = check_inside(r1s, n_jobs)
        omit_outside = (~inside).sum()

        # vectorized nearest using BallTree (or cdist)
        omit_limit = 0
        if limit > 0.0:
            # only check "inside" points
            idx = np.where(inside)[0]
            check_r1s = r1s[idx]
            if check_inside.inner_r is not None:
                # ... and those that are at least inner_sphere + limit away
                mask = (np.linalg.norm(check_r1s - check_inside.cm, axis=-1) >=
                        check_inside.inner_r - limit / 1000.)
                idx = idx[mask]
                check_r1s = check_r1s[mask]
            dists = _compute_nearest(
                surf['rr'], check_r1s, return_dists=True, method='cKDTree')[1]
            close = (dists < limit / 1000.0)
            omit_limit = np.sum(close)
            inside[idx[close]] = False
        s['inuse'][vertno[~inside]] = False
        del vertno
        s['nuse'] -= (omit_outside + omit_limit)
        s['vertno'] = np.where(s['inuse'])[0]

        if omit_outside > 0:
            extras = [omit_outside]
            extras += ['s', 'they are'] if omit_outside > 1 else ['', 'it is']
            logger.info('    %d source space point%s omitted because %s '
                        'outside the inner skull surface.' % tuple(extras))
        if omit_limit > 0:
            extras = [omit_limit]
            extras += ['s'] if omit_outside > 1 else ['']
            extras += [limit]
            logger.info('    %d source space point%s omitted because of the '
                        '%6.1f-mm distance limit.' % tuple(extras))
        # Adjust the patch inds as well if necessary
        if omit_limit + omit_outside > 0:
            _adjust_patch_info(s)


@verbose
def _adjust_patch_info(s, verbose=None):
    """Adjust patch information in place after vertex omission."""
    if s.get('patch_inds') is not None:
        if s['nearest'] is None:
            # This shouldn't happen, but if it does, we can probably come
            # up with a more clever solution
            raise RuntimeError('Cannot adjust patch information properly, '
                               'please contact the mne-python developers')
        _add_patch_info(s)


@verbose
def _ensure_src(src, kind=None, extra='', verbose=None):
    """Ensure we have a source space."""
    _check_option(
        'kind', kind, (None, 'surface', 'volume', 'mixed', 'discrete'))
    msg = 'src must be a string or instance of SourceSpaces%s' % (extra,)
    if _check_path_like(src):
        src = str(src)
        if not op.isfile(src):
            raise IOError('Source space file "%s" not found' % src)
        logger.info('Reading %s...' % src)
        src = read_source_spaces(src, verbose=False)
    if not isinstance(src, SourceSpaces):
        raise ValueError('%s, got %s (type %s)' % (msg, src, type(src)))
    if kind is not None:
        if src.kind != kind and src.kind == 'mixed':
            if kind == 'surface':
                src = src[:2]
            elif kind == 'volume':
                src = src[2:]
        if src.kind != kind:
            raise ValueError('Source space must contain %s type, got '
                             '%s' % (kind, src.kind))
    return src


def _ensure_src_subject(src, subject):
    src_subject = src._subject
    if subject is None:
        subject = src_subject
        if subject is None:
            raise ValueError('source space is too old, subject must be '
                             'provided')
    elif src_subject is not None and subject != src_subject:
        raise ValueError('Mismatch between provided subject "%s" and subject '
                         'name "%s" in the source space'
                         % (subject, src_subject))
    return subject


_DIST_WARN_LIMIT = 10242  # warn for anything larger than ICO-5


@verbose
def add_source_space_distances(src, dist_limit=np.inf, n_jobs=1, verbose=None):
    """Compute inter-source distances along the cortical surface.

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
        10/2013) must be installed. If 0, then only patch (nearest vertex)
        information is added.
    %(n_jobs)s
        Ignored if ``dist_limit==0.``.
    %(verbose)s

    Returns
    -------
    src : instance of SourceSpaces
        The original source spaces, with distance information added.
        The distances are stored in src[n]['dist'].
        Note: this function operates in-place.

    Notes
    -----
    This function can be memory- and CPU-intensive. On a high-end machine
    (2012) running 6 jobs in parallel, an ico-5 (10242 per hemi) source space
    takes about 10 minutes to compute all distances (``dist_limit = np.inf``).
    With ``dist_limit = 0.007``, computing distances takes about 1 minute.

    We recommend computing distances once per source space and then saving
    the source space to disk, as the computed distances will automatically be
    stored along with the source space data for future use.
    """
    from scipy.sparse.csgraph import dijkstra
    n_jobs = check_n_jobs(n_jobs)
    src = _ensure_src(src)
    dist_limit = float(dist_limit)
    if dist_limit < 0:
        raise ValueError('dist_limit must be non-negative, got %s'
                         % (dist_limit,))
    patch_only = (dist_limit == 0)
    if patch_only and not check_version('scipy', '1.3'):
        raise RuntimeError('scipy >= 1.3 is required to calculate patch '
                           'information only, consider upgrading SciPy or '
                           'using dist_limit=np.inf when running '
                           'add_source_space_distances')
    if src.kind != 'surface':
        raise RuntimeError('Currently all source spaces must be of surface '
                           'type')

    parallel, p_fun, _ = parallel_func(_do_src_distances, n_jobs)
    min_dists = list()
    min_idxs = list()
    msg = 'patch information' if patch_only else 'source space distances'
    logger.info('Calculating %s (limit=%s mm)...' % (msg, 1000 * dist_limit))
    max_n = max(s['nuse'] for s in src)
    if not patch_only and max_n > _DIST_WARN_LIMIT:
        warn('Computing distances for %d source space points (in one '
             'hemisphere) will be very slow, consider using add_dist=False'
             % (max_n,))
    for s in src:
        adjacency = mesh_dist(s['tris'], s['rr'])
        if patch_only:
            min_dist, _, min_idx = dijkstra(
                adjacency, indices=s['vertno'],
                min_only=True, return_predecessors=True)
            min_dists.append(min_dist.astype(np.float32))
            min_idxs.append(min_idx)
            for key in ('dist', 'dist_limit'):
                s[key] = None
        else:
            d = parallel(p_fun(adjacency, s['vertno'], r, dist_limit)
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
            # convert to sparse representation
            d = np.concatenate([dd[0] for dd in d]).ravel()  # already float32
            idx = d > 0
            d = d[idx]
            i, j = np.meshgrid(s['vertno'], s['vertno'])
            i = i.ravel()[idx]
            j = j.ravel()[idx]
            s['dist'] = sparse.csr_matrix(
                (d, (i, j)), shape=(s['np'], s['np']), dtype=np.float32)
            s['dist_limit'] = np.array([dist_limit], np.float32)

    # Let's see if our distance was sufficient to allow for patch info
    if not any(np.any(np.isinf(md)) for md in min_dists):
        # Patch info can be added!
        for s, min_dist, min_idx in zip(src, min_dists, min_idxs):
            s['nearest'] = min_idx
            s['nearest_dist'] = min_dist
            _add_patch_info(s)
    else:
        logger.info('Not adding patch information, dist_limit too small')
    return src


def _do_src_distances(con, vertno, run_inds, limit):
    """Compute source space distances in chunks."""
    from scipy.sparse.csgraph import dijkstra
    func = partial(dijkstra, limit=limit)
    chunk_size = 20  # save memory by chunking (only a little slower)
    lims = np.r_[np.arange(0, len(run_inds), chunk_size), len(run_inds)]
    n_chunks = len(lims) - 1
    # eventually we want this in float32, so save memory by only storing 32-bit
    d = np.empty((len(run_inds), len(vertno)), np.float32)
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


# XXX this should probably be deprecated because it returns surface Labels,
# and probably isn't the way to go moving forward
# XXX this also assumes that the first two source spaces are surf without
# checking, which might not be the case (could be all volumes)
def get_volume_labels_from_src(src, subject, subjects_dir):
    """Return a list of Label of segmented volumes included in the src space.

    Parameters
    ----------
    src : instance of SourceSpaces
        The source space containing the volume regions.
    subject : str
        Subject name.
    subjects_dir : str
        Freesurfer folder of the subjects.

    Returns
    -------
    labels_aseg : list of Label
        List of Label of segmented volumes included in src space.
    """
    from . import Label
    from . import get_volume_labels_from_aseg

    # Read the aseg file
    aseg_fname = op.join(subjects_dir, subject, 'mri', 'aseg.mgz')
    if not op.isfile(aseg_fname):
        raise IOError('aseg file "%s" not found' % aseg_fname)
    all_labels_aseg = get_volume_labels_from_aseg(
        aseg_fname, return_colors=True)

    # Create a list of Label
    if len(src) < 2:
        raise ValueError('No vol src space in src')

    if any(np.any(s['type'] != 'vol') for s in src[2:]):
        raise ValueError('source spaces have to be of vol type')

    labels_aseg = list()
    for nr in range(2, len(src)):
        vertices = src[nr]['vertno']

        pos = src[nr]['rr'][src[nr]['vertno'], :]
        roi_str = src[nr]['seg_name']
        try:
            ind = all_labels_aseg[0].index(roi_str)
            color = np.array(all_labels_aseg[1][ind]) / 255
        except ValueError:
            pass

        if 'left' in roi_str.lower():
            hemi = 'lh'
            roi_str = roi_str.replace('Left-', '') + '-lh'
        elif 'right' in roi_str.lower():
            hemi = 'rh'
            roi_str = roi_str.replace('Right-', '') + '-rh'
        else:
            hemi = 'both'

        label = Label(vertices=vertices, pos=pos, hemi=hemi,
                      name=roi_str, color=color,
                      subject=subject)
        labels_aseg.append(label)

    return labels_aseg


def _get_hemi(s):
    """Get a hemisphere from a given source space."""
    if s['type'] != 'surf':
        raise RuntimeError('Only surface source spaces supported')
    if s['id'] == FIFF.FIFFV_MNE_SURF_LEFT_HEMI:
        return 'lh', 0, s['id']
    elif s['id'] == FIFF.FIFFV_MNE_SURF_RIGHT_HEMI:
        return 'rh', 1, s['id']
    else:
        raise ValueError('unknown surface ID %s' % s['id'])


def _get_vertex_map_nn(fro_src, subject_from, subject_to, hemi, subjects_dir,
                       to_neighbor_tri=None):
    """Get a nearest-neigbor vertex match for a given hemi src.

    The to_neighbor_tri can optionally be passed in to avoid recomputation
    if it's already available.
    """
    # adapted from mne_make_source_space.c, knowing accurate=False (i.e.
    # nearest-neighbor mode should be used)
    logger.info('Mapping %s %s -> %s (nearest neighbor)...'
                % (hemi, subject_from, subject_to))
    regs = [op.join(subjects_dir, s, 'surf', '%s.sphere.reg' % hemi)
            for s in (subject_from, subject_to)]
    reg_fro, reg_to = [read_surface(r, return_dict=True)[-1] for r in regs]
    if to_neighbor_tri is not None:
        reg_to['neighbor_tri'] = to_neighbor_tri
    if 'neighbor_tri' not in reg_to:
        reg_to['neighbor_tri'] = _triangle_neighbors(reg_to['tris'],
                                                     reg_to['np'])

    morph_inuse = np.zeros(len(reg_to['rr']), bool)
    best = np.zeros(fro_src['np'], int)
    ones = _compute_nearest(reg_to['rr'], reg_fro['rr'][fro_src['vertno']])
    for v, one in zip(fro_src['vertno'], ones):
        # if it were actually a proper morph map, we would do this, but since
        # we know it's nearest neighbor list, we don't need to:
        # this_mm = mm[v]
        # one = this_mm.indices[this_mm.data.argmax()]
        if morph_inuse[one]:
            # Try the nearest neighbors
            neigh = _get_surf_neighbors(reg_to, one)  # on demand calc
            was = one
            one = neigh[np.where(~morph_inuse[neigh])[0]]
            if len(one) == 0:
                raise RuntimeError('vertex %d would be used multiple times.'
                                   % one)
            one = one[0]
            logger.info('Source space vertex moved from %d to %d because of '
                        'double occupation.' % (was, one))
        best[v] = one
        morph_inuse[one] = True
    return best


@verbose
def morph_source_spaces(src_from, subject_to, surf='white', subject_from=None,
                        subjects_dir=None, verbose=None):
    """Morph an existing source space to a different subject.

    .. warning:: This can be used in place of morphing source estimates for
                 multiple subjects, but there may be consequences in terms
                 of dipole topology.

    Parameters
    ----------
    src_from : instance of SourceSpaces
        Surface source spaces to morph.
    subject_to : str
        The destination subject.
    surf : str
        The brain surface to use for the new source space.
    subject_from : str | None
        The "from" subject. For most source spaces this shouldn't need
        to be provided, since it is stored in the source space itself.
    subjects_dir : str | None
        Path to SUBJECTS_DIR if it is not set in the environment.
    %(verbose)s

    Returns
    -------
    src : instance of SourceSpaces
        The morphed source spaces.

    Notes
    -----
    .. versionadded:: 0.10.0
    """
    # adapted from mne_make_source_space.c
    src_from = _ensure_src(src_from)
    subject_from = _ensure_src_subject(src_from, subject_from)
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    src_out = list()
    for fro in src_from:
        hemi, idx, id_ = _get_hemi(fro)
        to = op.join(subjects_dir, subject_to, 'surf', '%s.%s' % (hemi, surf,))
        logger.info('Reading destination surface %s' % (to,))
        to = read_surface(to, return_dict=True, verbose=False)[-1]
        complete_surface_info(to, copy=False)
        # Now we morph the vertices to the destination
        # The C code does something like this, but with a nearest-neighbor
        # mapping instead of the weighted one::
        #
        #     >>> mm = read_morph_map(subject_from, subject_to, subjects_dir)
        #
        # Here we use a direct NN calculation, since picking the max from the
        # existing morph map (which naively one might expect to be equivalent)
        # differs for ~3% of vertices.
        best = _get_vertex_map_nn(fro, subject_from, subject_to, hemi,
                                  subjects_dir, to['neighbor_tri'])
        for key in ('neighbor_tri', 'tri_area', 'tri_cent', 'tri_nn',
                    'use_tris'):
            del to[key]
        to['vertno'] = np.sort(best[fro['vertno']])
        to['inuse'] = np.zeros(len(to['rr']), int)
        to['inuse'][to['vertno']] = True
        to['use_tris'] = best[fro['use_tris']]
        to.update(nuse=len(to['vertno']), nuse_tri=len(to['use_tris']),
                  nearest=None, nearest_dist=None, patch_inds=None, pinfo=None,
                  dist=None, id=id_, dist_limit=None, type='surf',
                  coord_frame=FIFF.FIFFV_COORD_MRI, subject_his_id=subject_to,
                  rr=to['rr'] / 1000.)
        src_out.append(to)
        logger.info('[done]\n')
    info = dict(working_dir=os.getcwd(), command_line=_get_call_line())
    return SourceSpaces(src_out, info=info)


@verbose
def _get_morph_src_reordering(vertices, src_from, subject_from, subject_to,
                              subjects_dir=None, verbose=None):
    """Get the reordering indices for a morphed source space.

    Parameters
    ----------
    vertices : list
        The vertices for the left and right hemispheres.
    src_from : instance of SourceSpaces
        The original source space.
    subject_from : str
        The source subject.
    subject_to : str
        The destination subject.
    %(subjects_dir)s
    %(verbose)s

    Returns
    -------
    data_idx : ndarray, shape (n_vertices,)
        The array used to reshape the data.
    from_vertices : list
        The right and left hemisphere vertex numbers for the "from" subject.
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    from_vertices = list()
    data_idxs = list()
    offset = 0
    for ii, hemi in enumerate(('lh', 'rh')):
        # Get the mapping from the original source space to the destination
        # subject's surface vertex numbers
        best = _get_vertex_map_nn(src_from[ii], subject_from, subject_to,
                                  hemi, subjects_dir)
        full_mapping = best[src_from[ii]['vertno']]
        # Tragically, we might not have all of our vertno left (e.g. because
        # some are omitted during fwd calc), so we must do some indexing magic:

        # From all vertices, a subset could be chosen by fwd calc:
        used_vertices = np.in1d(full_mapping, vertices[ii])
        from_vertices.append(src_from[ii]['vertno'][used_vertices])
        remaining_mapping = full_mapping[used_vertices]
        if not np.array_equal(np.sort(remaining_mapping), vertices[ii]) or \
                not np.in1d(vertices[ii], full_mapping).all():
            raise RuntimeError('Could not map vertices, perhaps the wrong '
                               'subject "%s" was provided?' % subject_from)

        # And our data have been implicitly remapped by the forced ascending
        # vertno order in source spaces
        implicit_mapping = np.argsort(remaining_mapping)  # happens to data
        data_idx = np.argsort(implicit_mapping)  # to reverse the mapping
        data_idx += offset  # hemisphere offset
        data_idxs.append(data_idx)
        offset += len(implicit_mapping)
    data_idx = np.concatenate(data_idxs)
    # this one is really just a sanity check for us, should never be violated
    # by users
    assert np.array_equal(np.sort(data_idx),
                          np.arange(sum(len(v) for v in vertices)))
    return data_idx, from_vertices


def _compare_source_spaces(src0, src1, mode='exact', nearest=True,
                           dist_tol=1.5e-3):
    """Compare two source spaces.

    Note: this function is also used by forward/tests/test_make_forward.py
    """
    from numpy.testing import (assert_allclose, assert_array_equal,
                               assert_equal, assert_)
    from scipy.spatial.distance import cdist
    if mode != 'exact' and 'approx' not in mode:  # 'nointerp' can be appended
        raise RuntimeError('unknown mode %s' % mode)

    for si, (s0, s1) in enumerate(zip(src0, src1)):
        # first check the keys
        a, b = set(s0.keys()), set(s1.keys())
        assert_equal(a, b, str(a ^ b))
        for name in ['nuse', 'ntri', 'np', 'type', 'id']:
            a, b = s0[name], s1[name]
            if name == 'id':  # workaround for old NumPy bug
                a, b = int(a), int(b)
            assert_equal(a, b, name)
        for name in ['subject_his_id']:
            if name in s0 or name in s1:
                assert_equal(s0[name], s1[name], name)
        for name in ['interpolator']:
            if name in s0 or name in s1:
                diffs = (s0['interpolator'] - s1['interpolator']).data
                if len(diffs) > 0 and 'nointerp' not in mode:
                    # 5%
                    assert_(np.sqrt(np.mean(diffs ** 2)) < 0.10, name)
        for name in ['nn', 'rr', 'nuse_tri', 'coord_frame', 'tris']:
            if s0[name] is None:
                assert_(s1[name] is None, name)
            else:
                if mode == 'exact':
                    assert_array_equal(s0[name], s1[name], name)
                else:  # 'approx' in mode
                    atol = 1e-3 if name == 'nn' else 1e-4
                    assert_allclose(s0[name], s1[name], rtol=1e-3, atol=atol,
                                    err_msg=name)
        for name in ['seg_name']:
            if name in s0 or name in s1:
                assert_equal(s0[name], s1[name], name)
        # these fields will exist if patch info was added
        if nearest:
            for name in ['nearest', 'nearest_dist', 'patch_inds']:
                if s0[name] is None:
                    assert_(s1[name] is None, name)
                else:
                    atol = 0 if mode == 'exact' else 1e-6
                    assert_allclose(s0[name], s1[name],
                                    atol=atol, err_msg=name)
            for name in ['pinfo']:
                if s0[name] is None:
                    assert_(s1[name] is None, name)
                else:
                    assert_(len(s0[name]) == len(s1[name]), name)
                    for p1, p2 in zip(s0[name], s1[name]):
                        assert_(all(p1 == p2), name)
        if mode == 'exact':
            for name in ['inuse', 'vertno', 'use_tris']:
                assert_array_equal(s0[name], s1[name], err_msg=name)
            for name in ['dist_limit']:
                assert_(s0[name] == s1[name], name)
            for name in ['dist']:
                if s0[name] is not None:
                    assert_equal(s1[name].shape, s0[name].shape)
                    assert_(len((s0['dist'] - s1['dist']).data) == 0)
        else:  # 'approx' in mode:
            # deal with vertno, inuse, and use_tris carefully
            for ii, s in enumerate((s0, s1)):
                assert_array_equal(s['vertno'], np.where(s['inuse'])[0],
                                   'src%s[%s]["vertno"] != '
                                   'np.where(src%s[%s]["inuse"])[0]'
                                   % (ii, si, ii, si))
            assert_equal(len(s0['vertno']), len(s1['vertno']))
            agreement = np.mean(s0['inuse'] == s1['inuse'])
            assert_(agreement >= 0.99, "%s < 0.99" % agreement)
            if agreement < 1.0:
                # make sure mismatched vertno are within 1.5mm
                v0 = np.setdiff1d(s0['vertno'], s1['vertno'])
                v1 = np.setdiff1d(s1['vertno'], s0['vertno'])
                dists = cdist(s0['rr'][v0], s1['rr'][v1])
                assert_allclose(np.min(dists, axis=1), np.zeros(len(v0)),
                                atol=dist_tol, err_msg='mismatched vertno')
            if s0['use_tris'] is not None:  # for "spacing"
                assert_array_equal(s0['use_tris'].shape, s1['use_tris'].shape)
            else:
                assert_(s1['use_tris'] is None)
            assert_(np.mean(s0['use_tris'] == s1['use_tris']) > 0.99)
    # The above "if s0[name] is not None" can be removed once the sample
    # dataset is updated to have a source space with distance info
    for name in ['working_dir', 'command_line']:
        if mode == 'exact':
            assert_equal(src0.info[name], src1.info[name])
        else:  # 'approx' in mode:
            if name in src0.info:
                assert_(name in src1.info, '"%s" missing' % name)
            else:
                assert_(name not in src1.info, '"%s" should not exist' % name)


def _set_source_space_vertices(src, vertices):
    """Reset the list of source space vertices."""
    assert len(src) == len(vertices)
    for s, v in zip(src, vertices):
        s['inuse'].fill(0)
        s['nuse'] = len(v)
        s['vertno'] = np.array(v)
        s['inuse'][s['vertno']] = 1
        s['use_tris'] = np.array([[]], int)
        s['nuse_tri'] = np.array([0])
        # This will fix 'patch_info' and 'pinfo'
        _adjust_patch_info(s, verbose=False)
    return src


def _get_src_nn(s, use_cps=True, vertices=None):
    vertices = s['vertno'] if vertices is None else vertices
    if use_cps and s.get('patch_inds') is not None:
        nn = np.empty((len(vertices), 3))
        for vp, p in enumerate(np.searchsorted(s['vertno'], vertices)):
            assert(s['vertno'][p] == vertices[vp])
            #  Project out the surface normal and compute SVD
            nn[vp] = np.sum(
                s['nn'][s['pinfo'][s['patch_inds'][p]], :], axis=0)
        nn /= linalg.norm(nn, axis=-1, keepdims=True)
    else:
        nn = s['nn'][vertices, :]
    return nn
