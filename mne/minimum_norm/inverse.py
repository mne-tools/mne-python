# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Rey Rene Ramirez, Ph.D. <rrramirez at mcw.edu>
#
# License: BSD (3-clause)

from math import sqrt
import numpy as np
from scipy import linalg

from ..fiff.constants import FIFF
from ..fiff.open import fiff_open
from ..fiff.tag import find_tag
from ..fiff.matrix import _read_named_matrix, _transpose_named_matrix
from ..fiff.proj import read_proj, make_projector
from ..fiff.tree import dir_tree_find
from ..fiff.pick import pick_channels_evoked, pick_channels

from ..cov import read_cov
from ..source_space import read_source_spaces_from_tree, find_source_space_hemi
from ..forward import _block_diag
from ..transforms import invert_transform, transform_source_space_to
from ..source_estimate import SourceEstimate


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
    parent_mri = parent_mri[0]

    print '\tReading inverse operator info...',
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
    print '\tReading inverse operator decomposition...',
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
    try:
        inv['eigen_leads'] = _read_named_matrix(fid, invs,
                                               FIFF.FIFF_MNE_INVERSE_LEADS)
    except:
        inv['eigen_leads_weighted'] = True
        inv['eigen_leads'] = _read_named_matrix(fid, invs,
                                    FIFF.FIFF_MNE_INVERSE_LEADS_WEIGHTED)
    #
    #   Having the eigenleads as columns is better for the inverse calculations
    #
    inv['eigen_leads'] = _transpose_named_matrix(inv['eigen_leads'])
    inv['eigen_fields'] = _read_named_matrix(fid, invs,
                                            FIFF.FIFF_MNE_INVERSE_FIELDS)

    print '[done]'
    #
    #   Read the covariance matrices
    #
    inv['noise_cov'] = read_cov(fid, invs, FIFF.FIFFV_MNE_NOISE_COV)
    print '\tNoise covariance matrix read.'

    inv['source_cov'] = read_cov(fid, invs, FIFF.FIFFV_MNE_SOURCE_COV)
    print '\tSource covariance matrix read.'
    #
    #   Read the various priors
    #
    inv['orient_prior'] = read_cov(fid, invs,
                                   FIFF.FIFFV_MNE_ORIENT_PRIOR_COV)
    if inv['orient_prior'] is not None:
        print '\tOrientation priors read.'

    inv['depth_prior'] = read_cov(fid, invs,
                                      FIFF.FIFFV_MNE_DEPTH_PRIOR_COV)
    if inv['depth_prior'] is not None:
        print '\tDepth priors read.'

    inv['fmri_prior'] = read_cov(fid, invs, FIFF.FIFFV_MNE_FMRI_PRIOR_COV)
    if inv['fmri_prior'] is not None:
        print '\tfMRI priors read.'

    #
    #   Read the source spaces
    #
    try:
        inv['src'] = read_source_spaces_from_tree(fid, tree, add_geom=False)
    except Exception as inst:
        fid.close()
        raise Exception('Could not read the source spaces (%s)' % inst)

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

    print ('\tSource spaces transformed to the inverse solution '
           'coordinate frame')
    #
    #   Done!
    #
    fid.close()

    return inv

###############################################################################
# Compute inverse solution


def combine_xyz(vec, square=False):
    """Compute the three Cartesian components of a vector or matrix together

    Parameters
    ----------
    vec : 2d array of shape [3 n x p]
        Input [ x1 y1 z1 ... x_n y_n z_n ] where x1 ... z_n
        can be vectors

    Returns
    -------
    comb : array
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


def prepare_inverse_operator(orig, nave, lambda2, dSPM):
    """Prepare an inverse operator for actually computing the inverse

    Parameters
    ----------
    orig : dict
        The inverse operator structure read from a file
    nave : int
        Number of averages (scales the noise covariance)
    lambda2 : float
        The regularization factor. Recommended to be 1 / SNR**2
    dSPM : bool
        If True, compute the noise-normalization factors for dSPM.

    Returns
    -------
    inv : dict
        Prepared inverse operator
    """

    if nave <= 0:
        raise ValueError('The number of averages should be positive')

    print 'Preparing the inverse operator for use...'
    inv = orig.copy()
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

    print ('\tScaled noise and source covariance from nave = %d to '
          'nave = %d' % (inv['nave'], nave))
    inv['nave'] = nave
    #
    #   Create the diagonal matrix for computing the regularized inverse
    #
    inv['reginv'] = inv['sing'] / (inv['sing'] ** 2 + lambda2)
    print '\tCreated the regularized inverter'
    #
    #   Create the projection operator
    #
    inv['proj'], ncomp, _ = make_projector(inv['projs'],
                                        inv['noise_cov']['names'])
    if ncomp > 0:
        print '\tCreated an SSP operator (subspace dimension = %d)' % ncomp

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
        print ('\tCreated the whitener using a full noise covariance matrix '
              '(%d small eigenvalues omitted)' % (inv['noise_cov']['dim']
                                                  - np.sum(nzero)))
    else:
        #
        #   No need to omit the zeroes due to projection
        #
        inv['whitener'] = np.diag(1.0 / np.sqrt(inv['noise_cov']['data'].ravel()))
        print ('\tCreated the whitener using a diagonal noise covariance '
               'matrix (%d small eigenvalues discarded)' % ncomp)

    #
    #   Finally, compute the noise-normalization factors
    #
    if dSPM:
        print '\tComputing noise-normalization factors...',
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


def apply_inverse(evoked, inverse_operator, lambda2, dSPM=True):
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

    Returns
    -------
    stc: SourceEstimate
        The source estimates
    """

    #
    #   Set up the inverse according to the parameters
    #
    nave = evoked.nave

    inv = prepare_inverse_operator(inverse_operator, nave, lambda2, dSPM)
    #
    #   Pick the correct channels from the data
    #
    evoked = pick_channels_evoked(evoked, inv['noise_cov']['names'])
    print 'Picked %d channels from the data' % evoked.info['nchan']
    print 'Computing inverse...',
    #
    #   Simple matrix multiplication followed by combination of the
    #   three current components
    #
    #   This does all the data transformations to compute the weights for the
    #   eigenleads
    #
    trans = inv['reginv'][:, None] * reduce(np.dot,
                                           [inv['eigen_fields']['data'],
                                           inv['whitener'],
                                           inv['proj'],
                                           evoked.data])

    #
    #   Transformation into current distributions by weighting the eigenleads
    #   with the weights computed above
    #
    if inv['eigen_leads_weighted']:
        #
        #     R^0.5 has been already factored in
        #
        print '(eigenleads already weighted)...',
        sol = np.dot(inv['eigen_leads']['data'], trans)
    else:
        #
        #     R^0.5 has to be factored in
        #
        print '(eigenleads need to be weighted)...',
        sol = np.sqrt(inv['source_cov']['data'])[:, None] * \
                             np.dot(inv['eigen_leads']['data'], trans)

    if inv['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
        print 'combining the current components...',
        sol = combine_xyz(sol)

    if dSPM:
        print '(dSPM)...',
        sol *= inv['noisenorm'][:, None]

    src = inv['src']
    stc = SourceEstimate(None)
    stc.data = sol
    stc.tmin = float(evoked.first) / evoked.info['sfreq']
    stc.tstep = 1.0 / evoked.info['sfreq']
    stc.lh_vertno = src[0]['vertno']
    stc.rh_vertno = src[1]['vertno']
    stc._init_times()
    print '[done]'

    return stc


def apply_inverse_raw(raw, inverse_operator, lambda2, dSPM=True,
                      label=None, start=None, stop=None, nave=1,
                      time_func=None):
    """Apply inverse operator to Raw data

    Computes a L2-norm inverse solution
    Actual code using these principles might be different because
    the inverse operator is often reused across data sets.

    Parameters
    ----------
    raw: Raw object
        Evoked data
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
        Index of last time sample (index not time is seconds)
    nave: int
        Number of averages used to regularize the solution.
        Set to 1 on raw data.
    time_func: callable
        Linear function applied to sensor space time series.
    Returns
    -------
    stc: SourceEstimate
        The source estimates
    """

    #
    #   Set up the inverse according to the parameters
    #
    inv = prepare_inverse_operator(inverse_operator, nave, lambda2, dSPM)
    #
    #   Pick the correct channels from the data
    #
    sel = pick_channels(raw.ch_names, include=inv['noise_cov']['names'])
    print 'Picked %d channels from the data' % len(sel)
    print 'Computing inverse...',
    #
    #   Simple matrix multiplication followed by combination of the
    #   three current components
    #
    #   This does all the data transformations to compute the weights for the
    #   eigenleads
    #

    src = inv['src']
    lh_vertno = src[0]['vertno']
    rh_vertno = src[1]['vertno']

    data, times = raw[sel, start:stop]

    if time_func is not None:
        data = time_func(data)

    trans = inv['reginv'][:, None] * reduce(np.dot,
                                            [inv['eigen_fields']['data'],
                                            inv['whitener'],
                                            inv['proj'],
                                            data])

    eigen_leads = inv['eigen_leads']['data']
    source_cov = inv['source_cov']['data'][:, None]
    noise_norm = inv['noisenorm'][:, None]

    if label is not None:
        if label['hemi'] == 'lh':
            vertno_sel = np.intersect1d(lh_vertno, label['vertices'])
            src_sel = np.searchsorted(lh_vertno, vertno_sel)
            lh_vertno = vertno_sel
            rh_vertno = np.array([])
        elif label['hemi'] == 'rh':
            vertno_sel = np.intersect1d(rh_vertno, label['vertices'])
            src_sel = np.searchsorted(rh_vertno, vertno_sel) + len(lh_vertno)
            lh_vertno = np.array([])
            rh_vertno = vertno_sel

        noise_norm = noise_norm[src_sel]

        if inv['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
            src_sel = 3 * src_sel
            src_sel = np.c_[src_sel, src_sel + 1, src_sel + 2]
            src_sel = src_sel.ravel()

        eigen_leads = eigen_leads[src_sel]
        source_cov = source_cov[src_sel]

    #
    #   Transformation into current distributions by weighting the eigenleads
    #   with the weights computed above
    #
    if inv['eigen_leads_weighted']:
        #
        #     R^0.5 has been already factored in
        #
        print '(eigenleads already weighted)...',
        sol = np.dot(eigen_leads, trans)
    else:
        #
        #     R^0.5 has to be factored in
        #
        print '(eigenleads need to be weighted)...',
        sol = np.sqrt(source_cov) * np.dot(eigen_leads, trans)

    if inv['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
        print 'combining the current components...',
        sol = combine_xyz(sol)

    if dSPM:
        print '(dSPM)...',
        sol *= noise_norm

    stc = SourceEstimate(None)
    stc.data = sol
    stc.tmin = float(times[0]) / raw.info['sfreq']
    stc.tstep = 1.0 / raw.info['sfreq']
    stc.lh_vertno = lh_vertno
    stc.rh_vertno = rh_vertno
    stc._init_times()
    print '[done]'

    return stc


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
    Lf_xyz : array of shape [n_sensors, n_positions x 3]
        Leadfield
    normals : array of shape [n_positions, 3]
        Normals to the cortex

    Returns
    -------
    Lf_cortex : array of shape [n_sensors, n_positions x 3]
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


def minimum_norm(evoked, forward, whitener, method='dspm',
                 orientation='fixed', snr=3, loose=0.2, depth=True,
                 weight_exp=0.8, weight_limit=10, fmri=None, fmri_thresh=None,
                 fmri_off=0.1):
    """Minimum norm estimate (MNE)

    Compute MNE, dSPM and sLORETA on evoked data starting from
    a forward operator.

    Parameters
    ----------
    evoked : Evoked
        Evoked data to invert
    forward : dict
        Forward operator
    cov : Covariance
        Noise covariance matrix
    method : 'wmne' | 'dspm' | 'sloreta'
        The method to use
    orientation : 'fixed' | 'free' | 'loose'
        Type of orientation constraints 'fixed'.
    snr : float
        Signal-to noise ratio defined as in MNE (default: 3).
    loose : float in [0, 1]
        Value that weights the source variances of the dipole components
        defining the tangent space of the cortical surfaces.
    depth : bool
        Flag to do depth weighting (default: True).
    weight_exp : float
        Order of the depth weighting. {0=no, 1=full normalization, default=0.8}
    weight_limit : float
        Maximal amount depth weighting (default: 10).
    mag_reg : float
        Amount of regularization of the magnetometer noise covariance matrix
    grad_reg : float
        Amount of regularization of the gradiometer noise covariance matrix.
    eeg_reg : float
        Amount of regularization of the EEG noise covariance matrix.
    fmri : array of shape [n_sources]
        Vector of fMRI values are the source points.
    fmri_thresh : float
        fMRI threshold. The source variances of source points with fmri smaller
        than fmri_thresh will be multiplied by fmri_off.
    fmri_off : float
        Weight assigned to non-active source points according to fmri
        and fmri_thresh.

    Returns
    -------
    stc : dict
        Source time courses
    """
    assert method in ['wmne', 'dspm', 'sloreta']
    assert orientation in ['fixed', 'free', 'loose']

    if not 0 <= loose <= 1:
        raise ValueError('loose value should be smaller than 1 and bigger than'
                         ' 0, or empty for no loose orientations.')
    if not 0 <= weight_exp <= 1:
        raise ValueError('weight_exp should be a scalar between 0 and 1')

    # Set regularization parameter based on SNR
    lambda2 = 1.0 / snr ** 2

    normals = []
    for s in forward['src']:
        normals.append(s['nn'][s['inuse'] != 0])
    normals = np.concatenate(normals)

    W, ch_names = whitener.W, whitener.ch_names

    gain = forward['sol']['data']
    fwd_ch_names = [forward['chs'][k]['ch_name'] for k in range(gain.shape[0])]
    fwd_idx = [fwd_ch_names.index(name) for name in ch_names]
    gain = gain[fwd_idx]

    print "Computing inverse solution with %d channels." % len(ch_names)

    rank_noise = len(W)
    print 'Total rank is %d' % rank_noise

    # processing lead field matrices, weights, areas & orientations
    # Initializing.
    n_positions = gain.shape[1] / 3

    if orientation == 'fixed':
        n_dip_per_pos = 1
    elif orientation in ['free', 'loose']:
        n_dip_per_pos = 3

    n_dipoles = n_positions * n_dip_per_pos

    w = np.ones(n_dipoles)

    # compute power
    if depth:
        w = np.sum(gain ** 2, axis=0)
        w = w.reshape(-1, 3).sum(axis=1)
        w = w[:, None] * np.ones((1, n_dip_per_pos))
        w = w.ravel()

    if orientation == 'fixed':
        print 'Appying fixed dipole orientations.'
        gain = gain * _block_diag(normals.ravel()[None, :], 3).T
    elif orientation == 'free':
        print 'Using free dipole orientations. No constraints.'
    elif orientation == 'loose':
        print 'Transforming lead field matrix to cortical coordinate system.'
        gain = _xyz2lf(gain, normals)
        # Getting indices for tangential dipoles: [1, 2, 4, 5]
        itangential = [k for k in range(n_dipoles) if n_dipoles % 3 != 0]

    # Whiten lead field.
    print 'Whitening lead field matrix.'
    gain = np.dot(W, gain)

    # Computing reciprocal of power.
    w = 1.0 / w

    # apply areas
    # if ~isempty(areas)
    #     display('wMNE> Applying areas to compute current source density.')
    #     areas = areas.^2;
    #     w = w .* areas;
    # end
    # clear areas

    # apply depth weighthing
    if depth:
        # apply weight limit
        # Applying weight limit.
        print 'Applying weight limit.'
        weight_limit2 = weight_limit ** 2
        # limit = min(w(w>min(w) * weight_limit2));  % This is the Matti way.
        # we do the Rey way (robust to possible weight discontinuity).
        limit = np.min(w) * weight_limit2
        w[w > limit] = limit

        # apply weight exponent
        # Applying weight exponent.
        print 'Applying weight exponent.'
        w = w ** weight_exp

    # apply loose orientations
    if orientation == 'loose':
        print 'Applying loose dipole orientations. Loose value of %s.' % loose
        w[itangential] *= loose

    # Apply fMRI Priors
    if fmri is not None:
        print 'Applying fMRI priors.'
        w[fmri < fmri_thresh] *= fmri_off

    # Adjusting Source Covariance matrix to make trace of L*C_J*L' equal
    # to number of sensors.
    print 'Adjusting source covariance matrix.'
    source_std = np.sqrt(w)  # sqrt(C_J)
    trclcl = linalg.norm(gain * source_std[None, :], ord='fro')
    source_std *= sqrt(rank_noise) / trclcl  # correct C_J
    gain *= source_std[None, :]

    # Compute SVD.
    print 'Computing SVD of whitened and weighted lead field matrix.'
    U, s, Vh = linalg.svd(gain, full_matrices=False)
    ss = s / (s ** 2 + lambda2)

    # Compute whitened MNE operator.
    Kernel = source_std[:, None] * np.dot(Vh.T, ss[:, None] * U.T)

    # Compute dSPM operator.
    if method == 'dspm':
        print 'Computing dSPM inverse operator.'
        dspm_diag = np.sum(Kernel ** 2, axis=1)
        if n_dip_per_pos == 1:
            dspm_diag = np.sqrt(dspm_diag)
        elif n_dip_per_pos in [2, 3]:
            dspm_diag = dspm_diag.reshape(-1, n_dip_per_pos)
            dspm_diag = np.sqrt(np.sum(dspm_diag, axis=1))
            dspm_diag = (np.ones((1, n_dip_per_pos)) *
                         dspm_diag[:, None]).ravel()

        Kernel /= dspm_diag[:, None]

    # whitened sLORETA imaging kernel
    elif method == 'sloreta':
        print 'Computing sLORETA inverse operator.'
        if n_dip_per_pos == 1:
            sloreta_diag = np.sqrt(np.sum(Kernel * gain.T, axis=1))
            Kernel /= sloreta_diag[:, None]
        elif n_dip_per_pos in [2, 3]:
            for k in n_positions:
                start = k * n_dip_per_pos
                stop = start + n_dip_per_pos
                R = np.dot(Kernel[start:stop, :], gain[:, start:stop])
                SIR = linalg.matfuncs.sqrtm(R, linalg.pinv(R))
                Kernel[start:stop] = np.dot(SIR, Kernel[start:stop])

    # Multiply inverse operator by whitening matrix, so no need to whiten data
    Kernel = np.dot(Kernel, W)
    sel = [evoked.ch_names.index(name) for name in ch_names]
    sol = np.dot(Kernel, evoked.data[sel])

    if n_dip_per_pos > 1:
        print 'combining the current components...',
        sol = combine_xyz(sol)

    src = forward['src']
    stc = SourceEstimate(None)
    stc.data = sol
    stc.tmin = float(evoked.first) / evoked.info['sfreq']
    stc.tstep = 1.0 / evoked.info['sfreq']
    stc.lh_vertno = src[0]['vertno']
    stc.rh_vertno = src[1]['vertno']
    stc._init_times()
    print '[done]'

    return stc
