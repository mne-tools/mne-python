# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg

from .fiff.constants import FIFF
from .fiff.open import fiff_open
from .fiff.tree import dir_tree_find
from .fiff.tag import find_tag, read_tag
from .fiff.matrix import _read_named_matrix, _transpose_named_matrix
from .fiff.pick import pick_channels_forward

from .source_space import read_source_spaces_from_tree, find_source_space_hemi
from .transforms import transform_source_space_to, invert_transform


def _block_diag(A, n):
    """Constructs a block diagonal from a packed structure

    You have to try it on a matrix to see what it's doing.

    If A is not sparse, then returns a sparse block diagonal "bd",
    diagonalized from the
    elements in "A".
    "A" is ma x na, comprising bdn=(na/"n") blocks of submatrices.
    Each submatrix is ma x "n", and these submatrices are
    placed down the diagonal of the matrix.

    If A is already sparse, then the operation is reversed, yielding
    a block
    row matrix, where each set of n columns corresponds to a block element
    from the block diagonal.

    Parameters
    ----------
    A: array
        The matrix
    n: int
        The block size
    Returns
    -------
    bd: sparse matrix
        The block diagonal matrix
    """
    from scipy import sparse

    if not sparse.issparse(A):  # then make block sparse
        ma, na = A.shape
        bdn = na / int(n)  # number of submatrices

        if na % n > 0:
            raise ValueError('Width of matrix must be a multiple of n')

        tmp = np.arange(ma * bdn, dtype=np.int).reshape(bdn, ma)
        tmp = np.tile(tmp, (1, n))
        ii = tmp.ravel()

        jj = np.arange(na, dtype=np.int)[None, :]
        jj = jj * np.ones(ma, dtype=np.int)[:, None]
        jj = jj.T.ravel()  # column indices foreach sparse bd

        bd = sparse.coo_matrix((A.T.ravel(), np.c_[ii, jj].T)).tocsc()
    else:  # already is sparse, unblock it
        import pdb; pdb.set_trace()
        # [mA,na] = size(A);        % matrix always has na columns
        # % how many entries in the first column?
        # bdn = na/n;           % number of blocks
        # ma = mA/bdn;          % rows in first block
        #
        # % blocks may themselves contain zero entries. Build indexing as above
        # tmp = reshape([1:(ma*bdn)]',ma,bdn);
        # i = zeros(ma*n,bdn);
        # for iblock = 1:n,
        # i((iblock-1)*ma+[1:ma],:) = tmp;
        # end
        #
        # i = i(:);             % row indices foreach sparse bd
        #
        #
        # j = [0:mA:(mA*(na-1))];
        # j = j(ones(ma,1),:);
        # j = j.ravel()
        #
        # i += j
        #
        # bd = full(A(i));  % column vector
        # bd = reshape(bd,ma,na);   % full matrix

    return bd


def _read_one(fid, node):
    """Read all interesting stuff for one forward solution
    """
    if node is None:
        return None

    tag = find_tag(fid, node, FIFF.FIFF_MNE_SOURCE_ORIENTATION)
    if tag is None:
        fid.close()
        raise ValueError('Source orientation tag not found')

    one = dict()
    one['source_ori'] = tag.data
    tag = find_tag(fid, node, FIFF.FIFF_MNE_COORD_FRAME)
    if tag is None:
        fid.close()
        raise ValueError('Coordinate frame tag not found')

    one['coord_frame'] = tag.data
    tag = find_tag(fid, node, FIFF.FIFF_MNE_SOURCE_SPACE_NPOINTS)
    if tag is None:
        fid.close()
        raise ValueError('Number of sources not found')

    one['nsource'] = tag.data
    tag = find_tag(fid, node, FIFF.FIFF_NCHAN)
    if tag is None:
        fid.close()
        raise ValueError('Number of channels not found')

    one['nchan'] = tag.data
    try:
        one['sol'] = _read_named_matrix(fid, node,
                                            FIFF.FIFF_MNE_FORWARD_SOLUTION)
        one['sol'] = _transpose_named_matrix(one['sol'])
    except Exception as inst:
        fid.close()
        raise 'Forward solution data not found (%s)' % inst

    try:
        one['sol_grad'] = _read_named_matrix(fid, node,
                                        FIFF.FIFF_MNE_FORWARD_SOLUTION_GRAD)
        one['sol_grad'] = _transpose_named_matrix(one['sol_grad'])
    except Exception as inst:
        one['sol_grad'] = None

    if one['sol']['data'].shape[0] != one['nchan'] or \
                (one['sol']['data'].shape[1] != one['nsource'] and
                 one['sol']['data'].shape[1] != 3 * one['nsource']):
        fid.close()
        raise ValueError('Forward solution matrix has wrong dimensions')

    if one['sol_grad'] is not None:
        if one['sol_grad']['data'].shape[0] != one['nchan'] or \
                (one['sol_grad']['data'].shape[1] != 3 * one['nsource'] and
                 one['sol_grad']['data'].shape[1] != 3 * 3 * one['nsource']):
            fid.close()
            raise ValueError('Forward solution gradient matrix has '
                             'wrong dimensions')

    return one


def read_forward_solution(fname, force_fixed=False, surf_ori=False,
                              include=[], exclude=[]):
    """Read a forward solution a.k.a. lead field

    Parameters
    ----------
    fname: string
        The file name.

    force_fixed: bool, optional (default False)
        Force fixed source orientation mode?

    surf_ori: bool, optional (default False)
        Use surface based source coordinate system?

    include: list, optional
        List of names of channels to include. If empty all channels
        are included.

    exclude: list, optional
        List of names of channels to exclude. If empty include all
        channels.

    Returns
    -------
    fwd: dict
        The forward solution

    """

    #   Open the file, create directory
    print 'Reading forward solution from %s...' % fname
    fid, tree, _ = fiff_open(fname)

    #   Find all forward solutions
    fwds = dir_tree_find(tree, FIFF.FIFFB_MNE_FORWARD_SOLUTION)
    if len(fwds) == 0:
        fid.close()
        raise ValueError('No forward solutions in %s' % fname)

    #   Parent MRI data
    parent_mri = dir_tree_find(tree, FIFF.FIFFB_MNE_PARENT_MRI_FILE)
    if len(fwds) == 0:
        fid.close()
        raise ValueError('No parent MRI information in %s' % fname)
    parent_mri = parent_mri[0]

    #   Parent MEG data
    parent_meg = dir_tree_find(tree, FIFF.FIFFB_MNE_PARENT_MEAS_FILE)
    if len(parent_meg) == 0:
        fid.close()
        raise ValueError('No parent MEG information in %s' % fname)
    parent_meg = parent_meg[0]

    chs = list()
    for k in range(parent_meg['nent']):
        kind = parent_meg['directory'][k].kind
        pos = parent_meg['directory'][k].pos
        if kind == FIFF.FIFF_CH_INFO:
            tag = read_tag(fid, pos)
            chs.append(tag.data)

    try:
        src = read_source_spaces_from_tree(fid, tree, add_geom=False)
    except Exception as inst:
        fid.close()
        raise ValueError('Could not read the source spaces (%s)' % inst)

    for s in src:
        s['id'] = find_source_space_hemi(s)

    fwd = None

    #   Locate and read the forward solutions
    megnode = None
    eegnode = None
    for k in range(len(fwds)):
        tag = find_tag(fid, fwds[k], FIFF.FIFF_MNE_INCLUDED_METHODS)
        if tag is None:
            fid.close()
            raise ValueError('Methods not listed for one of the forward '
                             'solutions')

        if tag.data == FIFF.FIFFV_MNE_MEG:
            megnode = fwds[k]
        elif tag.data == FIFF.FIFFV_MNE_EEG:
            eegnode = fwds[k]

    megfwd = _read_one(fid, megnode)
    if megfwd is not None:
        if megfwd['source_ori'] == FIFF.FIFFV_MNE_FIXED_ORI:
            ori = 'fixed'
        else:
            ori = 'free'

        print '\tRead MEG forward solution (%d sources, %d channels, ' \
              '%s orientations)' % (megfwd['nsource'], megfwd['nchan'], ori)

    eegfwd = _read_one(fid, eegnode)
    if eegfwd is not None:
        if eegfwd['source_ori'] == FIFF.FIFFV_MNE_FIXED_ORI:
            ori = 'fixed'
        else:
            ori = 'free'

        print '\tRead EEG forward solution (%d sources, %d channels, ' \
               '%s orientations)' % (eegfwd['nsource'], eegfwd['nchan'], ori)

    #   Merge the MEG and EEG solutions together
    if megfwd is not None and eegfwd is not None:
        if (megfwd['sol']['data'].shape[1] != eegfwd['sol']['data'].shape[1] or
                megfwd['source_ori'] != eegfwd['source_ori'] or
                megfwd['nsource'] != eegfwd['nsource'] or
                megfwd['coord_frame'] != eegfwd['coord_frame']):
            fid.close()
            raise ValueError('The MEG and EEG forward solutions do not match')

        fwd = megfwd
        fwd['sol']['data'] = np.r_[fwd['sol']['data'], eegfwd['sol']['data']]
        fwd['sol']['nrow'] = fwd['sol']['nrow'] + eegfwd['sol']['nrow']

        fwd['sol']['row_names'] = fwd['sol']['row_names'] + \
                                  eegfwd['sol']['row_names']
        if fwd['sol_grad'] is not None:
            fwd['sol_grad']['data'] = np.r_[fwd['sol_grad']['data'],
                                            eegfwd['sol_grad']['data']]
            fwd['sol_grad']['nrow'] = fwd['sol_grad']['nrow'] + \
                                      eegfwd['sol_grad']['nrow']
            fwd['sol_grad']['row_names'] = fwd['sol_grad']['row_names'] + \
                                           eegfwd['sol_grad']['row_names']

        fwd['nchan'] = fwd['nchan'] + eegfwd['nchan']
        print '\tMEG and EEG forward solutions combined'
    elif megfwd is not None:
        fwd = megfwd
    else:
        fwd = eegfwd

    del megfwd
    del eegfwd

    #   Get the MRI <-> head coordinate transformation
    tag = find_tag(fid, parent_mri, FIFF.FIFF_COORD_TRANS)
    if tag is None:
        fid.close()
        raise ValueError('MRI/head coordinate transformation not found')
    else:
        mri_head_t = tag.data
        if (mri_head_t['from'] != FIFF.FIFFV_COORD_MRI or
                mri_head_t['to'] != FIFF.FIFFV_COORD_HEAD):
            mri_head_t = invert_transform(mri_head_t)
            if (mri_head_t['from'] != FIFF.FIFFV_COORD_MRI
                or mri_head_t['to'] != FIFF.FIFFV_COORD_HEAD):
                fid.close()
                raise ValueError('MRI/head coordinate transformation not '
                                 'found')

    fid.close()

    fwd['mri_head_t'] = mri_head_t

    #   Transform the source spaces to the correct coordinate frame
    #   if necessary

    if (fwd['coord_frame'] != FIFF.FIFFV_COORD_MRI and
            fwd['coord_frame'] != FIFF.FIFFV_COORD_HEAD):
        raise ValueError('Only forward solutions computed in MRI or head '
                         'coordinates are acceptable')

    nuse = 0
    for s in src:
        try:
            s = transform_source_space_to(s, fwd['coord_frame'], mri_head_t)
        except Exception as inst:
            raise ValueError('Could not transform source space (%s)' % inst)

        nuse += s['nuse']

    if nuse != fwd['nsource']:
        raise ValueError('Source spaces do not match the forward solution.')

    print '\tSource spaces transformed to the forward solution ' \
          'coordinate frame'
    fwd['src'] = src

    #   Handle the source locations and orientations
    if (fwd['source_ori'] == FIFF.FIFFV_MNE_FIXED_ORI) or force_fixed:
        nuse = 0
        fwd['source_rr'] = np.zeros((fwd['nsource'], 3))
        fwd['source_nn'] = np.zeros((fwd['nsource'], 3))
        for s in src:
            fwd['source_rr'][nuse:nuse + s['nuse'], :] = \
                                                    s['rr'][s['vertno'], :]
            fwd['source_nn'][nuse:nuse + s['nuse'], :] = \
                                                    s['nn'][s['vertno'], :]
            nuse += s['nuse']

        #   Modify the forward solution for fixed source orientations
        if fwd['source_ori'] != FIFF.FIFFV_MNE_FIXED_ORI:
            print '\tChanging to fixed-orientation forward solution...'
            fix_rot = _block_diag(fwd['source_nn'].T, 1)
            fwd['sol']['data'] *= fix_rot
            fwd['sol']['ncol'] = fwd['nsource']
            fwd['source_ori'] = FIFF.FIFFV_MNE_FIXED_ORI

            if fwd['sol_grad'] is not None:
                fwd['sol_grad']['data'] = np.dot(fwd['sol_grad']['data'],
                                                 np.kron(fix_rot, np.eye(3)))
                fwd['sol_grad']['ncol'] = 3 * fwd['nsource']

            print '[done]'
    elif surf_ori:
        #   Rotate the local source coordinate systems
        print '\tConverting to surface-based source orientations...'
        nuse = 0
        pp = 0
        nuse_total = sum([s['nuse'] for s in src])
        fwd['source_rr'] = np.zeros((fwd['nsource'], 3))
        fwd['source_nn'] = np.empty((3 * nuse_total, 3), dtype=np.float)
        for s in src:
            rr = s['rr'][s['vertno'], :]
            fwd['source_rr'][nuse:nuse + s['nuse'], :] = rr
            for p in range(s['nuse']):
                #  Project out the surface normal and compute SVD
                nn = s['nn'][s['vertno'][p], :][:, None]
                U, S, _ = linalg.svd(np.eye(3, 3) - nn * nn.T)
                #  Make sure that ez is in the direction of nn
                if np.sum(nn.ravel() * U[:, 2].ravel()) < 0:
                    U *= -1.0
                fwd['source_nn'][pp:pp + 3, :] = U.T
                pp += 3

            nuse += s['nuse']

        surf_rot = _block_diag(fwd['source_nn'].T, 3)
        fwd['sol']['data'] = fwd['sol']['data'] * surf_rot
        if fwd['sol_grad'] is not None:
            fwd['sol_grad']['data'] = np.dot(fwd['sol_grad']['data'] * \
                                             np.kron(surf_rot, np.eye(3)))

        print '[done]'
    else:
        print '\tCartesian source orientations...'
        nuse = 0
        fwd['source_rr'] = np.zeros((fwd['nsource'], 3))
        for s in src:
            rr = s['rr'][s['vertno'], :]
            fwd['source_rr'][nuse:nuse + s['nuse'], :] = rr
            nuse += s['nuse']

        fwd['source_nn'] = np.kron(np.ones((fwd['nsource'], 1)), np.eye(3))
        print '[done]'

    fwd['surf_ori'] = surf_ori

    # Add channel information
    fwd['chs'] = chs

    fwd = pick_channels_forward(fwd, include=include, exclude=exclude)

    return fwd


def compute_depth_prior(G, exp=0.8, limit=10.0):
    """Compute weighting for depth prior
    """
    n_pos = G.shape[1] // 3
    d = np.zeros(n_pos)
    for k in xrange(n_pos):
        Gk = G[:, 3 * k:3 * (k + 1)]
        d[k] = linalg.svdvals(np.dot(Gk.T, Gk))[0]
    w = 1.0 / d
    wmax = np.min(w) * (limit ** 2)
    wp = np.minimum(w, wmax)
    wpp = (wp / wmax) ** exp
    depth_prior = np.ravel(wpp[:, None] * np.ones((1, 3)))
    return depth_prior
