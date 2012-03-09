# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import warnings
from copy import deepcopy
from math import sqrt
import numpy as np
from scipy import linalg

from ..fiff.constants import FIFF
from ..fiff.open import fiff_open
from ..fiff.tag import find_tag
from ..fiff.matrix import _read_named_matrix, _transpose_named_matrix, \
                          write_named_matrix
from ..fiff.proj import read_proj, make_projector, write_proj
from ..fiff.tree import dir_tree_find
from ..fiff.write import write_int, write_float_matrix, start_file, \
                         start_block, end_block, end_file, write_float, \
                         write_coord_trans

from ..fiff.cov import read_cov, write_cov
from ..cov import prepare_noise_cov
from ..forward import compute_depth_prior, compute_depth_prior_fixed
from ..source_space import read_source_spaces_from_tree, \
                           find_source_space_hemi, _get_vertno, \
                           write_source_spaces
from ..transforms import invert_transform, transform_source_space_to
from ..source_estimate import SourceEstimate


def _pick_channels_inverse_operator(ch_names, inv):
    """Gives the indices of the data channel to be used knowing
    an inverse operator
    """
    sel = []
    for name in inv['noise_cov']['names']:
        if name in ch_names:
            sel.append(ch_names.index(name))
        else:
            raise ValueError('The inverse operator was computed with '
                             'channel %s which is not present in '
                             'the data. You should compute a new inverse '
                             'operator restricted to the good data '
                             'channels.' % name)
    return sel


def read_inverse_operator(fname):
    """Read the inverse operator decomposition from a FIF file

    Parameters
    ----------
    fname: string
        The name of the FIF file.

    Returns
    -------
    inv: dict
        The inverse operator
    """
    #
    #   Open the file, create directory
    #
    print 'Reading inverse operator decomposition from %s...' % fname
    fid, tree, _ = fiff_open(fname)
    #
    #   Find all inverse operators
    #
    invs = dir_tree_find(tree, FIFF.FIFFB_MNE_INVERSE_SOLUTION)
    if invs is None:
        fid.close()
        raise Exception('No inverse solutions in %s' % fname)

    invs = invs[0]
    #
    #   Parent MRI data
    #
    parent_mri = dir_tree_find(tree, FIFF.FIFFB_MNE_PARENT_MRI_FILE)
    if len(parent_mri) == 0:
        fid.close()
        raise Exception('No parent MRI information in %s' % fname)
    parent_mri = parent_mri[0]  # take only first one

    print '    Reading inverse operator info...',
    #
    #   Methods and source orientations
    #
    tag = find_tag(fid, invs, FIFF.FIFF_MNE_INCLUDED_METHODS)
    if tag is None:
        fid.close()
        raise Exception('Modalities not found')

    inv = dict()
    inv['methods'] = tag.data

    tag = find_tag(fid, invs, FIFF.FIFF_MNE_SOURCE_ORIENTATION)
    if tag is None:
        fid.close()
        raise Exception('Source orientation constraints not found')

    inv['source_ori'] = int(tag.data)

    tag = find_tag(fid, invs, FIFF.FIFF_MNE_SOURCE_SPACE_NPOINTS)
    if tag is None:
        fid.close()
        raise Exception('Number of sources not found')

    inv['nsource'] = tag.data
    inv['nchan'] = 0
    #
    #   Coordinate frame
    #
    tag = find_tag(fid, invs, FIFF.FIFF_MNE_COORD_FRAME)
    if tag is None:
        fid.close()
        raise Exception('Coordinate frame tag not found')

    inv['coord_frame'] = tag.data
    #
    #   The actual source orientation vectors
    #
    tag = find_tag(fid, invs, FIFF.FIFF_MNE_INVERSE_SOURCE_ORIENTATIONS)
    if tag is None:
        fid.close()
        raise Exception('Source orientation information not found')

    inv['source_nn'] = tag.data
    print '[done]'
    #
    #   The SVD decomposition...
    #
    print '    Reading inverse operator decomposition...',
    tag = find_tag(fid, invs, FIFF.FIFF_MNE_INVERSE_SING)
    if tag is None:
        fid.close()
        raise Exception('Singular values not found')

    inv['sing'] = tag.data
    inv['nchan'] = len(inv['sing'])
    #
    #   The eigenleads and eigenfields
    #
    inv['eigen_leads_weighted'] = False
    eigen_leads = _read_named_matrix(fid, invs, FIFF.FIFF_MNE_INVERSE_LEADS)
    if eigen_leads is None:
        inv['eigen_leads_weighted'] = True
        eigen_leads = _read_named_matrix(fid, invs,
                                          FIFF.FIFF_MNE_INVERSE_LEADS_WEIGHTED)
    if eigen_leads is None:
        raise ValueError('Eigen leads not found in inverse operator.')
    #
    #   Having the eigenleads as columns is better for the inverse calculations
    #
    inv['eigen_leads'] = _transpose_named_matrix(eigen_leads)
    inv['eigen_fields'] = _read_named_matrix(fid, invs,
                                             FIFF.FIFF_MNE_INVERSE_FIELDS)
    print '[done]'
    #
    #   Read the covariance matrices
    #
    inv['noise_cov'] = read_cov(fid, invs, FIFF.FIFFV_MNE_NOISE_COV)
    print '    Noise covariance matrix read.'

    inv['source_cov'] = read_cov(fid, invs, FIFF.FIFFV_MNE_SOURCE_COV)
    print '    Source covariance matrix read.'
    #
    #   Read the various priors
    #
    inv['orient_prior'] = read_cov(fid, invs,
                                   FIFF.FIFFV_MNE_ORIENT_PRIOR_COV)
    if inv['orient_prior'] is not None:
        print '    Orientation priors read.'

    inv['depth_prior'] = read_cov(fid, invs,
                                      FIFF.FIFFV_MNE_DEPTH_PRIOR_COV)
    if inv['depth_prior'] is not None:
        print '    Depth priors read.'

    inv['fmri_prior'] = read_cov(fid, invs, FIFF.FIFFV_MNE_FMRI_PRIOR_COV)
    if inv['fmri_prior'] is not None:
        print '    fMRI priors read.'

    #
    #   Read the source spaces
    #

    inv['src'] = read_source_spaces_from_tree(fid, tree, add_geom=False)

    for s in inv['src']:
        s['id'] = find_source_space_hemi(s)

    #
    #   Get the MRI <-> head coordinate transformation
    #
    tag = find_tag(fid, parent_mri, FIFF.FIFF_COORD_TRANS)
    if tag is None:
        fid.close()
        raise Exception('MRI/head coordinate transformation not found')
    else:
        mri_head_t = tag.data
        if mri_head_t['from'] != FIFF.FIFFV_COORD_MRI or \
                        mri_head_t['to'] != FIFF.FIFFV_COORD_HEAD:
            mri_head_t = invert_transform(mri_head_t)
            if mri_head_t['from'] != FIFF.FIFFV_COORD_MRI or \
                        mri_head_t['to'] != FIFF.FIFFV_COORD_HEAD:
                fid.close()
                raise Exception('MRI/head coordinate transformation '
                                 'not found')

    inv['mri_head_t'] = mri_head_t

    #
    #   Transform the source spaces to the correct coordinate frame
    #   if necessary
    #
    if inv['coord_frame'] != FIFF.FIFFV_COORD_MRI and \
            inv['coord_frame'] != FIFF.FIFFV_COORD_HEAD:
        fid.close()
        raise Exception('Only inverse solutions computed in MRI or '
                         'head coordinates are acceptable')

    #
    #  Number of averages is initially one
    #
    inv['nave'] = 1
    #
    #  We also need the SSP operator
    #
    inv['projs'] = read_proj(fid, tree)
    #
    #  Some empty fields to be filled in later
    #
    inv['proj'] = []       # This is the projector to apply to the data
    inv['whitener'] = []   # This whitens the data
    inv['reginv'] = []     # This the diagonal matrix implementing
                           # regularization and the inverse
    inv['noisenorm'] = []  # These are the noise-normalization factors
    #
    nuse = 0
    for k in range(len(inv['src'])):
        try:
            inv['src'][k] = transform_source_space_to(inv['src'][k],
                                                inv['coord_frame'], mri_head_t)
        except Exception as inst:
            fid.close()
            raise Exception('Could not transform source space (%s)' % inst)

        nuse += inv['src'][k]['nuse']

    print ('    Source spaces transformed to the inverse solution '
           'coordinate frame')
    #
    #   Done!
    #
    fid.close()

    return inv


def write_inverse_operator(fname, inv):
    """Write an inverse operator to a FIF file

    Parameters
    ----------
    fname: string
        The name of the FIF file.

    inv: dict
        The inverse operator
    """
    #
    #   Open the file, create directory
    #
    print 'Write inverse operator decomposition in %s...' % fname

    # Create the file and save the essentials
    fid = start_file(fname)

    start_block(fid, FIFF.FIFFB_MNE_INVERSE_SOLUTION)

    print '    Writing inverse operator info...',

    write_int(fid, FIFF.FIFF_MNE_INCLUDED_METHODS, inv['methods'])
    write_int(fid, FIFF.FIFF_MNE_SOURCE_ORIENTATION, inv['source_ori'])
    write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NPOINTS, inv['nsource'])
    write_int(fid, FIFF.FIFF_MNE_COORD_FRAME, inv['coord_frame'])
    write_float_matrix(fid, FIFF.FIFF_MNE_INVERSE_SOURCE_ORIENTATIONS,
                       inv['source_nn'])
    write_float(fid, FIFF.FIFF_MNE_INVERSE_SING, inv['sing'])

    #
    #   The eigenleads and eigenfields
    #
    if inv['eigen_leads_weighted']:
        write_named_matrix(fid, FIFF.FIFF_MNE_INVERSE_LEADS_WEIGHTED,
                           _transpose_named_matrix(inv['eigen_leads']))
    else:
        write_named_matrix(fid, FIFF.FIFF_MNE_INVERSE_LEADS,
                           _transpose_named_matrix(inv['eigen_leads']))

    write_named_matrix(fid, FIFF.FIFF_MNE_INVERSE_FIELDS, inv['eigen_fields'])
    print '[done]'
    #
    #   write the covariance matrices
    #
    print '    Writing noise covariance matrix.'
    write_cov(fid, inv['noise_cov'])

    print '    Writing source covariance matrix.'
    write_cov(fid, inv['source_cov'])
    #
    #   write the various priors
    #
    print '    Writing orientation priors.'
    if inv['orient_prior'] is not None:
        write_cov(fid, inv['orient_prior'])
    write_cov(fid, inv['depth_prior'])

    if inv['fmri_prior'] is not None:
        write_cov(fid, inv['fmri_prior'])

    #
    #   Parent MRI data
    #
    start_block(fid, FIFF.FIFFB_MNE_PARENT_MRI_FILE)
    #   write the MRI <-> head coordinate transformation
    write_coord_trans(fid, inv['mri_head_t'])
    end_block(fid, FIFF.FIFFB_MNE_PARENT_MRI_FILE)

    #
    #   Write the source spaces
    #
    if 'src' in inv:
        write_source_spaces(fid, inv['src'])

    #
    #  We also need the SSP operator
    #
    write_proj(fid, inv['projs'])
    #
    #   Done!
    #

    end_block(fid, FIFF.FIFFB_MNE_INVERSE_SOLUTION)
    end_file(fid)

    fid.close()


###############################################################################
# Compute inverse solution


def combine_xyz(vec, square=False):
    """Compute the three Cartesian components of a vector or matrix together

    Parameters
    ----------
    vec: 2d array of shape [3 n x p]
        Input [ x1 y1 z1 ... x_n y_n z_n ] where x1 ... z_n
        can be vectors

    Returns
    -------
    comb: array
        Output vector [sqrt(x1^2+y1^2+z1^2), ..., sqrt(x_n^2+y_n^2+z_n^2)]
    """
    if vec.ndim != 2:
        raise Exception('Input must be 2D')
    if (vec.shape[0] % 3) != 0:
        raise Exception('Input must have 3N rows')

    n, p = vec.shape
    if np.iscomplexobj(vec):
        vec = np.abs(vec)
    comb = vec[0::3] ** 2
    comb += vec[1::3] ** 2
    comb += vec[2::3] ** 2
    if not square:
        comb = np.sqrt(comb)
    return comb


def _chech_ch_names(inv, info):
    """Check that channels in inverse operator are measurements"""

    inv_ch_names = inv['eigen_fields']['col_names']

    if inv['noise_cov']['names'] != inv_ch_names:
        raise ValueError('Channels in inverse operator eigen fields do not '
                         'match noise covariance channels.')
    data_ch_names = info['ch_names']

    missing_ch_names = list()
    for ch_name in inv_ch_names:
        if ch_name not in data_ch_names:
            missing_ch_names.append(ch_name)
    n_missing = len(missing_ch_names)
    if n_missing > 0:
        raise ValueError('%d channels in inverse operator ' % n_missing +
                         'are not present in the data (%s)' % missing_ch_names)


def prepare_inverse_operator(orig, nave, lambda2, dSPM):
    """Prepare an inverse operator for actually computing the inverse

    Parameters
    ----------
    orig: dict
        The inverse operator structure read from a file
    nave: int
        Number of averages (scales the noise covariance)
    lambda2: float
        The regularization factor. Recommended to be 1 / SNR**2
    dSPM: bool
        If True, compute the noise-normalization factors for dSPM.

    Returns
    -------
    inv: dict
        Prepared inverse operator
    """

    if nave <= 0:
        raise ValueError('The number of averages should be positive')

    print 'Preparing the inverse operator for use...'
    inv = deepcopy(orig)
    #
    #   Scale some of the stuff
    #
    scale = float(inv['nave']) / nave
    inv['noise_cov']['data'] = scale * inv['noise_cov']['data']
    inv['noise_cov']['eig'] = scale * inv['noise_cov']['eig']
    inv['source_cov']['data'] = scale * inv['source_cov']['data']
    #
    if inv['eigen_leads_weighted']:
        inv['eigen_leads']['data'] = sqrt(scale) * inv['eigen_leads']['data']

    print ('    Scaled noise and source covariance from nave = %d to '
          'nave = %d' % (inv['nave'], nave))
    inv['nave'] = nave
    #
    #   Create the diagonal matrix for computing the regularized inverse
    #
    sing = np.array(inv['sing'], dtype=np.float64)
    inv['reginv'] = sing / (sing ** 2 + lambda2)
    print '    Created the regularized inverter'
    #
    #   Create the projection operator
    #
    inv['proj'], ncomp, _ = make_projector(inv['projs'],
                                           inv['noise_cov']['names'])
    if ncomp > 0:
        print '    Created an SSP operator (subspace dimension = %d)' % ncomp
    else:
        print '    The projection vectors do not apply to these channels.'

    #
    #   Create the whitener
    #
    if not inv['noise_cov']['diag']:
        inv['whitener'] = np.zeros((inv['noise_cov']['dim'],
                                    inv['noise_cov']['dim']))
        #
        #   Omit the zeroes due to projection
        #
        eig = inv['noise_cov']['eig']
        nzero = (eig > 0)
        inv['whitener'][nzero, nzero] = 1.0 / np.sqrt(eig[nzero])
        #
        #   Rows of eigvec are the eigenvectors
        #
        inv['whitener'] = np.dot(inv['whitener'], inv['noise_cov']['eigvec'])
        print ('    Created the whitener using a full noise covariance matrix '
               '(%d small eigenvalues omitted)' % (inv['noise_cov']['dim']
                                                  - np.sum(nzero)))
    else:
        #
        #   No need to omit the zeroes due to projection
        #
        inv['whitener'] = np.diag(1.0 /
                                  np.sqrt(inv['noise_cov']['data'].ravel()))
        print ('    Created the whitener using a diagonal noise covariance '
               'matrix (%d small eigenvalues discarded)' % ncomp)

    #
    #   Finally, compute the noise-normalization factors
    #
    if dSPM:
        print '    Computing noise-normalization factors...',
        noise_norm = np.zeros(inv['eigen_leads']['nrow'])
        nrm2, = linalg.get_blas_funcs(('nrm2',), (noise_norm,))
        if inv['eigen_leads_weighted']:
            for k in range(inv['eigen_leads']['nrow']):
                one = inv['eigen_leads']['data'][k, :] * inv['reginv']
                noise_norm[k] = nrm2(one)
        else:
            for k in range(inv['eigen_leads']['nrow']):
                one = sqrt(inv['source_cov']['data'][k]) * \
                            inv['eigen_leads']['data'][k, :] * inv['reginv']
                noise_norm[k] = nrm2(one)

        #
        #   Compute the final result
        #
        if inv['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
            #
            #   The three-component case is a little bit more involved
            #   The variances at three consequtive entries must be squared and
            #   added together
            #
            #   Even in this case return only one noise-normalization factor
            #   per source location
            #
            noise_norm = combine_xyz(noise_norm[:, None]).ravel()

        inv['noisenorm'] = 1.0 / np.abs(noise_norm)
        print '[done]'
    else:
        inv['noisenorm'] = []

    return inv


def _assemble_kernel(inv, label, dSPM, pick_normal):
    #
    #   Simple matrix multiplication followed by combination of the
    #   current components
    #
    #   This does all the data transformations to compute the weights for the
    #   eigenleads
    #
    eigen_leads = inv['eigen_leads']['data']
    source_cov = inv['source_cov']['data'][:, None]
    if dSPM:
        noise_norm = inv['noisenorm'][:, None]

    src = inv['src']
    vertno = _get_vertno(src)

    if label is not None:
        vertno, src_sel = _get_label_sel(label, inv)

        if dSPM:
            noise_norm = noise_norm[src_sel]

        if inv['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
            src_sel = 3 * src_sel
            src_sel = np.c_[src_sel, src_sel + 1, src_sel + 2]
            src_sel = src_sel.ravel()

        eigen_leads = eigen_leads[src_sel]
        source_cov = source_cov[src_sel]

    if pick_normal:
        if not inv['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
            raise ValueError('Pick normal can only be used with a free '
                             'orientation inverse operator.')

        is_loose = 0 < inv['orient_prior']['data'][0] < 1
        if not is_loose:
            raise ValueError('The pick_normal parameter is only valid '
                             'when working with loose orientations.')

        # keep only the normal components
        eigen_leads = eigen_leads[2::3]
        source_cov = source_cov[2::3]

    trans = inv['reginv'][:, None] * reduce(np.dot,
                                            [inv['eigen_fields']['data'],
                                            inv['whitener'],
                                            inv['proj']])
    #
    #   Transformation into current distributions by weighting the eigenleads
    #   with the weights computed above
    #
    if inv['eigen_leads_weighted']:
        #
        #     R^0.5 has been already factored in
        #
        print '(eigenleads already weighted)...',
        K = np.dot(eigen_leads, trans)
    else:
        #
        #     R^0.5 has to be factored in
        #
        print '(eigenleads need to be weighted)...',
        K = np.sqrt(source_cov) * np.dot(eigen_leads, trans)

    if not dSPM:
        noise_norm = None

    return K, noise_norm, vertno


def _make_stc(sol, tmin, tstep, vertno):
    stc = SourceEstimate(None)
    stc.data = sol
    stc.tmin = tmin
    stc.tstep = tstep
    stc.vertno = vertno
    stc._init_times()
    return stc


def _get_label_sel(label, inv):
    src = inv['src']

    if src[0]['type'] != 'surf':
        return Exception('Label are only supported with surface source spaces')

    vertno = [src[0]['vertno'], src[1]['vertno']]

    if label['hemi'] == 'lh':
        vertno_sel = np.intersect1d(vertno[0], label['vertices'])
        src_sel = np.searchsorted(vertno[0], vertno_sel)
        vertno[0] = vertno_sel
        vertno[1] = np.array([])
    elif label['hemi'] == 'rh':
        vertno_sel = np.intersect1d(vertno[1], label['vertices'])
        src_sel = np.searchsorted(vertno[1], vertno_sel) + len(vertno[0])
        vertno[0] = np.array([])
        vertno[1] = vertno_sel
    else:
        raise Exception("Unknown hemisphere type")

    return vertno, src_sel


def apply_inverse(evoked, inverse_operator, lambda2, dSPM=True,
                  pick_normal=False):
    """Apply inverse operator to evoked data

    Computes a L2-norm inverse solution
    Actual code using these principles might be different because
    the inverse operator is often reused across data sets.

    Parameters
    ----------
    evoked: Evoked object
        Evoked data
    inverse_operator: dict
        Inverse operator read with mne.read_inverse_operator
    lambda2: float
        The regularization parameter
    dSPM: bool
        do dSPM ?
    pick_normal: bool
        If True, rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations.

    Returns
    -------
    stc: SourceEstimate
        The source estimates
    """
    #
    #   Set up the inverse according to the parameters
    #
    nave = evoked.nave

    _chech_ch_names(inverse_operator, evoked.info)

    inv = prepare_inverse_operator(inverse_operator, nave, lambda2, dSPM)
    #
    #   Pick the correct channels from the data
    #
    sel = _pick_channels_inverse_operator(evoked.ch_names, inv)
    print 'Picked %d channels from the data' % len(sel)

    print 'Computing inverse...',
    K, noise_norm, _ = _assemble_kernel(inv, None, dSPM, pick_normal)
    sol = np.dot(K, evoked.data[sel])  # apply imaging kernel

    is_free_ori = (inverse_operator['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI
                   and not pick_normal)

    if is_free_ori:
        print 'combining the current components...',
        sol = combine_xyz(sol)

    if noise_norm is not None:
        print '(dSPM)...',
        sol *= noise_norm

    tstep = 1.0 / evoked.info['sfreq']
    tmin = float(evoked.first) / evoked.info['sfreq']
    vertno = _get_vertno(inv['src'])
    stc = _make_stc(sol, tmin, tstep, vertno)
    print '[done]'

    return stc


def apply_inverse_raw(raw, inverse_operator, lambda2, dSPM=True,
                      label=None, start=None, stop=None, nave=1,
                      time_func=None, pick_normal=False,
                      buffer_size=None):
    """Apply inverse operator to Raw data

    Computes a L2-norm inverse solution
    Actual code using these principles might be different because
    the inverse operator is often reused across data sets.

    Parameters
    ----------
    raw: Raw object
        Raw data
    inverse_operator: dict
        Inverse operator read with mne.read_inverse_operator
    lambda2: float
        The regularization parameter
    dSPM: bool
        do dSPM ?
    label: Label
        Restricts the source estimates to a given label
    start: int
        Index of first time sample (index not time is seconds)
    stop: int
        Index of first time sample not to include (index not time is seconds)
    nave: int
        Number of averages used to regularize the solution.
        Set to 1 on raw data.
    time_func: callable
        Linear function applied to sensor space time series.
    pick_normal: bool
        If True, rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations.
    buffer_size: int (or None)
        If not None, the computation of the inverse and the combination of the
        current components is performed in segments of length buffer_size
        samples. While slightly slower, this is useful for long datasets as it
        reduces the memory requirements by approx. a factor of 3 (assuming
        buffer_size << data length).
        Note that this setting has no effect for fixed-orientation inverse
        operators.
    Returns
    -------
    stc: SourceEstimate
        The source estimates
    """
    _chech_ch_names(inverse_operator, raw.info)

    #
    #   Set up the inverse according to the parameters
    #
    inv = prepare_inverse_operator(inverse_operator, nave, lambda2, dSPM)
    #
    #   Pick the correct channels from the data
    #
    sel = _pick_channels_inverse_operator(raw.ch_names, inv)
    print 'Picked %d channels from the data' % len(sel)
    print 'Computing inverse...',

    data, times = raw[sel, start:stop]

    if time_func is not None:
        data = time_func(data)

    K, noise_norm, vertno = _assemble_kernel(inv, label, dSPM, pick_normal)

    is_free_ori = (inverse_operator['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI
                   and not pick_normal)

    if buffer_size is not None and is_free_ori:
        # Process the data in segments to conserve memory
        n_seg = int(np.ceil(data.shape[1] / float(buffer_size)))
        print 'computing inverse and combining the current components'\
              ' (using %d segments)...' % (n_seg)

        # Allocate space for inverse solution
        n_times = data.shape[1]
        sol = np.empty((K.shape[0] / 3, n_times),
                        dtype=(K[0, 0] * data[0, 0]).dtype)

        for pos in xrange(0, n_times, buffer_size):
            sol[:, pos:pos + buffer_size] = \
                combine_xyz(np.dot(K, data[:, pos:pos + buffer_size]))

            print 'segment %d / %d done..' % (pos / buffer_size + 1, n_seg)
    else:
        sol = np.dot(K, data)
        if is_free_ori:
            print 'combining the current components...',
            sol = combine_xyz(sol)

    if noise_norm is not None:
        sol *= noise_norm

    tmin = float(times[0])
    tstep = 1.0 / raw.info['sfreq']
    stc = _make_stc(sol, tmin, tstep, vertno)
    print '[done]'

    return stc


def apply_inverse_epochs(epochs, inverse_operator, lambda2, dSPM=True,
                         label=None, nave=1, pick_normal=False):
    """Apply inverse operator to Epochs

    Computes a L2-norm inverse solution on each epochs and returns
    single trial source estimates.

    Parameters
    ----------
    epochs: Epochs object
        Single trial epochs
    inverse_operator: dict
        Inverse operator read with mne.read_inverse_operator
    lambda2: float
        The regularization parameter
    dSPM: bool
        do dSPM ?
    label: Label
        Restricts the source estimates to a given label
    nave: int
        Number of averages used to regularize the solution.
        Set to 1 on single Epoch by default.
    pick_normal: bool
        If True, rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations.

    Returns
    -------
    stc: list of SourceEstimate
        The source estimates for all epochs
    """
    _chech_ch_names(inverse_operator, epochs.info)

    #
    #   Set up the inverse according to the parameters
    #
    inv = prepare_inverse_operator(inverse_operator, nave, lambda2, dSPM)
    #
    #   Pick the correct channels from the data
    #
    sel = _pick_channels_inverse_operator(epochs.ch_names, inv)
    print 'Picked %d channels from the data' % len(sel)

    print 'Computing inverse...',
    K, noise_norm, vertno = _assemble_kernel(inv, label, dSPM, pick_normal)

    stcs = list()
    tstep = 1.0 / epochs.info['sfreq']
    tmin = epochs.times[0]

    is_free_ori = (inverse_operator['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI
                   and not pick_normal)

    for k, e in enumerate(epochs):
        print "Processing epoch : %d" % (k + 1)
        sol = np.dot(K, e[sel])  # apply imaging kernel

        if is_free_ori:
            print 'combining the current components...',
            sol = combine_xyz(sol)

        if noise_norm is not None:
            sol *= noise_norm

        stcs.append(_make_stc(sol, tmin, tstep, vertno))

    print '[done]'

    return stcs


def _xyz2lf(Lf_xyz, normals):
    """Reorient leadfield to one component matching the normal to the cortex

    This program takes a leadfield matix computed for dipole components
    pointing in the x, y, and z directions, and outputs a new lead field
    matrix for dipole components pointing in the normal direction of the
    cortical surfaces and in the two tangential directions to the cortex
    (that is on the tangent cortical space). These two tangential dipole
    components are uniquely determined by the SVD (reduction of variance).

    Parameters
    ----------
    Lf_xyz: array of shape [n_sensors, n_positions x 3]
        Leadfield
    normals: array of shape [n_positions, 3]
        Normals to the cortex

    Returns
    -------
    Lf_cortex: array of shape [n_sensors, n_positions x 3]
        Lf_cortex is a leadfield matrix for dipoles in rotated orientations, so
        that the first column is the gain vector for the cortical normal dipole
        and the following two column vectors are the gain vectors for the
        tangential orientations (tangent space of cortical surface).
    """
    n_sensors, n_dipoles = Lf_xyz.shape
    n_positions = n_dipoles / 3
    Lf_xyz = Lf_xyz.reshape(n_sensors, n_positions, 3)
    n_sensors, n_positions, _ = Lf_xyz.shape
    Lf_cortex = np.zeros_like(Lf_xyz)

    for k in range(n_positions):
        lf_normal = np.dot(Lf_xyz[:, k, :], normals[k])
        lf_normal_n = lf_normal[:, None] / linalg.norm(lf_normal)
        P = np.eye(n_sensors, n_sensors) - np.dot(lf_normal_n, lf_normal_n.T)
        lf_p = np.dot(P, Lf_xyz[:, k, :])
        U, s, Vh = linalg.svd(lf_p)
        Lf_cortex[:, k, 0] = lf_normal
        Lf_cortex[:, k, 1:] = np.c_[U[:, 0] * s[0], U[:, 1] * s[1]]

    Lf_cortex = Lf_cortex.reshape(n_sensors, n_dipoles)
    return Lf_cortex


###############################################################################
# Assemble the inverse operator

def make_inverse_operator(info, forward, noise_cov, loose=0.2, depth=0.8):
    """Assemble inverse operator

    Parameters
    ----------
    info: dict
        The measurement info to specify the channels to include.
        Bad channels in info['bads'] are ignored.
    forward: dict
        Forward operator
    noise_cov: Covariance
        The noise covariance matrix
    loose: float in [0, 1]
        Value that weights the source variances of the dipole components
        defining the tangent space of the cortical surfaces.
    depth: None | float in [0, 1]
        Depth weighting coefficients. If None, no depth weighting is performed.

    # XXX : add support for megreg=0.0, eegreg=0.0

    Returns
    -------
    stc: dict
        Source time courses
    """
    is_fixed_ori = (forward['source_ori'] == FIFF.FIFFV_MNE_FIXED_ORI)
    if is_fixed_ori and loose is not None:
        warnings.warn('Ignoring loose parameter with forward operator with '
                      'fixed orientation.')
        loose = None

    if not forward['surf_ori'] and loose is not None:
        raise ValueError('Forward operator is not oriented in surface '
                         'coordinates. loose parameter should be None '
                         'not %s.' % loose)

    if loose is not None and not (0 <= loose <= 1):
        raise ValueError('loose value should be smaller than 1 and bigger than'
                         ' 0, or None for not loose orientations.')
    if depth is not None and not (0 < depth <= 1):
        raise ValueError('depth should be a scalar between 0 and 1')

    fwd_ch_names = [c['ch_name'] for c in forward['chs']]
    ch_names = [c['ch_name'] for c in info['chs']
                                    if (c['ch_name'] not in info['bads'])
                                        and (c['ch_name'] in fwd_ch_names)]
    n_chan = len(ch_names)

    print "Computing inverse operator with %d channels." % n_chan

    noise_cov = prepare_noise_cov(noise_cov, info, ch_names)

    W = np.zeros((n_chan, n_chan), dtype=np.float)
    #
    #   Omit the zeroes due to projection
    #
    eig = noise_cov['eig']
    nzero = (eig > 0)
    W[nzero, nzero] = 1.0 / np.sqrt(eig[nzero])
    n_nzero = sum(nzero)
    #
    #   Rows of eigvec are the eigenvectors
    #
    W = np.dot(W, noise_cov['eigvec'])

    gain = forward['sol']['data']

    fwd_idx = [fwd_ch_names.index(name) for name in ch_names]
    gain = gain[fwd_idx]

    n_dipoles = gain.shape[1]

    # Handle depth prior scaling
    depth_prior = np.ones(n_dipoles, dtype=gain.dtype)
    if depth is not None:
        if is_fixed_ori:
            depth_prior = compute_depth_prior_fixed(gain, exp=depth)
        else:
            depth_prior = compute_depth_prior(gain, exp=depth)

    print "Computing inverse operator with %d channels." % len(ch_names)

    # Whiten lead field.
    print 'Whitening lead field matrix.'
    gain = np.dot(W, gain)

    source_cov = depth_prior.copy()
    depth_prior = dict(data=depth_prior)

    # apply loose orientations
    if not is_fixed_ori:
        orient_prior = np.ones(n_dipoles, dtype=gain.dtype)
        if loose is not None:
            print ('Applying loose dipole orientations. Loose value of %s.'
                                                                    % loose)
            orient_prior[np.mod(np.arange(n_dipoles), 3) != 2] *= loose
            source_cov *= orient_prior
        orient_prior = dict(data=orient_prior)
    else:
        orient_prior = None

    # Adjusting Source Covariance matrix to make trace of G*R*G' equal
    # to number of sensors.
    print 'Adjusting source covariance matrix.'
    source_std = np.sqrt(source_cov)
    gain *= source_std[None, :]
    trace_GRGT = linalg.norm(gain, ord='fro') ** 2
    scaling_source_cov = n_nzero / trace_GRGT
    source_cov *= scaling_source_cov
    gain *= sqrt(scaling_source_cov)

    source_cov = dict(data=source_cov)

    # now np.trace(np.dot(gain, gain.T)) == n_nzero
    # print np.trace(np.dot(gain, gain.T)), n_nzero

    print 'Computing SVD of whitened and weighted lead field matrix.'
    eigen_fields, sing, eigen_leads = linalg.svd(gain, full_matrices=False)

    eigen_fields = dict(data=eigen_fields.T, col_names=ch_names)
    eigen_leads = dict(data=eigen_leads.T, nrow=eigen_leads.shape[1])
    nave = 1.0

    inv_op = dict(eigen_fields=eigen_fields, eigen_leads=eigen_leads,
                  sing=sing, nave=nave, depth_prior=depth_prior,
                  source_cov=source_cov, noise_cov=noise_cov,
                  orient_prior=orient_prior, projs=deepcopy(info['projs']),
                  eigen_leads_weighted=False, source_ori=forward['source_ori'],
                  mri_head_t=deepcopy(forward['mri_head_t']),
                  src=deepcopy(forward['src']))

    return inv_op
