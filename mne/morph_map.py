# Authors: Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD-3-Clause

# Many of the computations in this code were derived from Matti Hämäläinen's
# C code.

import os
import os.path as op

import numpy as np

from .io.constants import FIFF
from .io.open import fiff_open
from .io.tag import find_tag
from .io.tree import dir_tree_find
from .io.write import (start_block, end_block, write_string,
                       start_and_end_file, write_float_sparse_rcs, write_int)
from .surface import (read_surface, _triangle_neighbors, _compute_nearest,
                      _normalize_vectors, _get_tri_supp_geom,
                      _find_nearest_tri_pts)
from .utils import get_subjects_dir, warn, logger, verbose


@verbose
def read_morph_map(subject_from, subject_to, subjects_dir=None, xhemi=False,
                   verbose=None):
    """Read morph map.

    Morph maps can be generated with mne_make_morph_maps. If one isn't
    available, it will be generated automatically and saved to the
    ``subjects_dir/morph_maps`` directory.

    Parameters
    ----------
    subject_from : str
        Name of the original subject as named in the SUBJECTS_DIR.
    subject_to : str
        Name of the subject on which to morph as named in the SUBJECTS_DIR.
    subjects_dir : str
        Path to SUBJECTS_DIR is not set in the environment.
    xhemi : bool
        Morph across hemisphere. Currently only implemented for
        ``subject_to == subject_from``. See notes of
        :func:`mne.compute_source_morph`.
    %(verbose)s

    Returns
    -------
    left_map, right_map : ~scipy.sparse.csr_matrix
        The morph maps for the 2 hemispheres.
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)

    # First check for morph-map dir existence
    mmap_dir = op.join(subjects_dir, 'morph-maps')
    if not op.isdir(mmap_dir):
        try:
            os.mkdir(mmap_dir)
        except Exception:
            warn('Could not find or make morph map directory "%s"' % mmap_dir)

    # filename components
    if xhemi:
        if subject_to != subject_from:
            raise NotImplementedError(
                "Morph-maps between hemispheres are currently only "
                "implemented for subject_to == subject_from")
        map_name_temp = '%s-%s-xhemi'
        log_msg = 'Creating morph map %s -> %s xhemi'
    else:
        map_name_temp = '%s-%s'
        log_msg = 'Creating morph map %s -> %s'

    map_names = [map_name_temp % (subject_from, subject_to),
                 map_name_temp % (subject_to, subject_from)]

    # find existing file
    fname = None
    for map_name in map_names:
        fname = op.join(mmap_dir, '%s-morph.fif' % map_name)
        if op.exists(fname):
            return _read_morph_map(fname, subject_from, subject_to)
    # if file does not exist, make it
    logger.info('Morph map "%s" does not exist, creating it and saving it to '
                'disk' % fname)
    logger.info(log_msg % (subject_from, subject_to))
    mmap_1 = _make_morph_map(subject_from, subject_to, subjects_dir, xhemi)
    if subject_to == subject_from:
        mmap_2 = None
    else:
        logger.info(log_msg % (subject_to, subject_from))
        mmap_2 = _make_morph_map(subject_to, subject_from, subjects_dir,
                                 xhemi)
    _write_morph_map(fname, subject_from, subject_to, mmap_1, mmap_2)
    return mmap_1


def _read_morph_map(fname, subject_from, subject_to):
    """Read a morph map from disk."""
    f, tree, _ = fiff_open(fname)
    with f as fid:
        # Locate all maps
        maps = dir_tree_find(tree, FIFF.FIFFB_MNE_MORPH_MAP)
        if len(maps) == 0:
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
                        logger.info('    Left-hemisphere map read.')
                    elif tag.data == FIFF.FIFFV_MNE_SURF_RIGHT_HEMI:
                        tag = find_tag(fid, m, FIFF.FIFF_MNE_MORPH_MAP)
                        right_map = tag.data
                        logger.info('    Right-hemisphere map read.')

    if left_map is None or right_map is None:
        raise ValueError('Could not find both hemispheres in %s' % fname)

    return left_map, right_map


def _write_morph_map(fname, subject_from, subject_to, mmap_1, mmap_2):
    """Write a morph map to disk."""
    try:
        with start_and_end_file(fname) as fid:
            _write_morph_map_(fid, subject_from, subject_to, mmap_1, mmap_2)
    except Exception as exp:
        warn('Could not write morph-map file "%s" (error: %s)'
             % (fname, exp))


def _write_morph_map_(fid, subject_from, subject_to, mmap_1, mmap_2):
    assert len(mmap_1) == 2
    hemis = [FIFF.FIFFV_MNE_SURF_LEFT_HEMI, FIFF.FIFFV_MNE_SURF_RIGHT_HEMI]
    for m, hemi in zip(mmap_1, hemis):
        start_block(fid, FIFF.FIFFB_MNE_MORPH_MAP)
        write_string(fid, FIFF.FIFF_MNE_MORPH_MAP_FROM, subject_from)
        write_string(fid, FIFF.FIFF_MNE_MORPH_MAP_TO, subject_to)
        write_int(fid, FIFF.FIFF_MNE_HEMI, hemi)
        write_float_sparse_rcs(fid, FIFF.FIFF_MNE_MORPH_MAP, m)
        end_block(fid, FIFF.FIFFB_MNE_MORPH_MAP)
    # don't write mmap_2 if it is identical (subject_to == subject_from)
    if mmap_2 is not None:
        assert len(mmap_2) == 2
        for m, hemi in zip(mmap_2, hemis):
            start_block(fid, FIFF.FIFFB_MNE_MORPH_MAP)
            write_string(fid, FIFF.FIFF_MNE_MORPH_MAP_FROM, subject_to)
            write_string(fid, FIFF.FIFF_MNE_MORPH_MAP_TO, subject_from)
            write_int(fid, FIFF.FIFF_MNE_HEMI, hemi)
            write_float_sparse_rcs(fid, FIFF.FIFF_MNE_MORPH_MAP, m)
            end_block(fid, FIFF.FIFFB_MNE_MORPH_MAP)


def _make_morph_map(subject_from, subject_to, subjects_dir, xhemi):
    """Construct morph map from one subject to another.

    Note that this is close, but not exactly like the C version.
    For example, parts are more accurate due to double precision,
    so expect some small morph-map differences!

    Note: This seems easily parallelizable, but the overhead
    of pickling all the data structures makes it less efficient
    than just running on a single core :(
    """
    subjects_dir = get_subjects_dir(subjects_dir)
    if xhemi:
        reg = '%s.sphere.left_right'
        hemis = (('lh', 'rh'), ('rh', 'lh'))
    else:
        reg = '%s.sphere.reg'
        hemis = (('lh', 'lh'), ('rh', 'rh'))

    return [_make_morph_map_hemi(subject_from, subject_to, subjects_dir,
                                 reg % hemi_from, reg % hemi_to)
            for hemi_from, hemi_to in hemis]


def _make_morph_map_hemi(subject_from, subject_to, subjects_dir, reg_from,
                         reg_to):
    """Construct morph map for one hemisphere."""
    from scipy.sparse import csr_matrix, eye as speye
    # add speedy short-circuit for self-maps
    if subject_from == subject_to and reg_from == reg_to:
        fname = op.join(subjects_dir, subject_from, 'surf', reg_from)
        n_pts = len(read_surface(fname, verbose=False)[0])
        return speye(n_pts, n_pts, format='csr')

    # load surfaces and normalize points to be on unit sphere
    fname = op.join(subjects_dir, subject_from, 'surf', reg_from)
    from_rr, from_tri = read_surface(fname, verbose=False)
    fname = op.join(subjects_dir, subject_to, 'surf', reg_to)
    to_rr = read_surface(fname, verbose=False)[0]
    _normalize_vectors(from_rr)
    _normalize_vectors(to_rr)

    # from surface: get nearest neighbors, find triangles for each vertex
    nn_pts_idx = _compute_nearest(from_rr, to_rr, method='cKDTree')
    from_pt_tris = _triangle_neighbors(from_tri, len(from_rr))
    from_pt_tris = [from_pt_tris[pt_idx].astype(int) for pt_idx in nn_pts_idx]
    from_pt_lens = np.cumsum([0] + [len(x) for x in from_pt_tris])
    from_pt_tris = np.concatenate(from_pt_tris)
    assert from_pt_tris.ndim == 1
    assert from_pt_lens[-1] == len(from_pt_tris)

    # find triangle in which point lies and assoc. weights
    tri_inds = []
    weights = []
    tri_geom = _get_tri_supp_geom(dict(rr=from_rr, tris=from_tri))
    weights, tri_inds = _find_nearest_tri_pts(
        to_rr, from_pt_tris, from_pt_lens, run_all=False, reproject=False,
        **tri_geom)

    nn_idx = from_tri[tri_inds]
    weights = np.array(weights)

    row_ind = np.repeat(np.arange(len(to_rr)), 3)
    this_map = csr_matrix((weights.ravel(), (row_ind, nn_idx.ravel())),
                          shape=(len(to_rr), len(from_rr)))
    return this_map
