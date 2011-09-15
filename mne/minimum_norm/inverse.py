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
from ..fiff.matrix import _read_named_matrix, _transpose_named_matrix
from ..fiff.proj import read_proj, make_projector
from ..fiff.tree import dir_tree_find
from ..fiff.pick import pick_channels

from ..cov import read_cov, prepare_noise_cov
from ..forward import compute_depth_prior
from ..source_space import read_source_spaces_from_tree, find_source_space_hemi
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


def _combine_ori(sol, inverse_operator, pick_normal):
    if inverse_operator['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
        print 'combining the current components...',
        if pick_normal:
            is_loose = 0 < inverse_operator['orient_prior']['data'][0] < 1
            if not is_loose:
                raise ValueError('The pick_normal parameter is only valid '
                                 'when working with loose orientations.')
            sol = sol[2::3]  # take one every 3 sources ie. only the normal
        else:
            sol = combine_xyz(sol)
    return sol


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
        inv['whitener'] = np.diag(1.0 /
                                  np.sqrt(inv['noise_cov']['data'].ravel()))
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


def _assemble_kernel(inv, label, dSPM):
    #
    #   Simple matrix multiplication followed by combination of the
    #   three current components
    #
    #   This does all the data transformations to compute the weights for the
    #   eigenleads
    #
    eigen_leads = inv['eigen_leads']['data']
    source_cov = inv['source_cov']['data'][:, None]
    noise_norm = inv['noisenorm'][:, None]

    src = inv['src']
    lh_vertno = src[0]['vertno']
    rh_vertno = src[1]['vertno']

    if label is not None:
        lh_vertno, rh_vertno, src_sel = _get_label_sel(label, inv)

        noise_norm = noise_norm[src_sel]

        if inv['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
            src_sel = 3 * src_sel
            src_sel = np.c_[src_sel, src_sel + 1, src_sel + 2]
            src_sel = src_sel.ravel()

        eigen_leads = eigen_leads[src_sel]
        source_cov = source_cov[src_sel]

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

    return K, noise_norm, lh_vertno, rh_vertno


def _make_stc(sol, tmin, tstep, lh_vertno, rh_vertno):
    stc = SourceEstimate(None)
    stc.data = sol
    stc.tmin = tmin
    stc.tstep = tstep
    stc.lh_vertno = lh_vertno
    stc.rh_vertno = rh_vertno
    stc._init_times()
    return stc


def _get_label_sel(label, inv):
    src = inv['src']
    lh_vertno = src[0]['vertno']
    rh_vertno = src[1]['vertno']

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
    else:
        raise Exception("Unknown hemisphere type")

    return lh_vertno, rh_vertno, src_sel


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

    inv = prepare_inverse_operator(inverse_operator, nave, lambda2, dSPM)
    #
    #   Pick the correct channels from the data
    #
    sel = pick_channels(evoked.ch_names, include=inv['noise_cov']['names'])
    print 'Picked %d channels from the data' % len(sel)

    print 'Computing inverse...',
    K, noise_norm, _, _ = _assemble_kernel(inv, None, dSPM)
    sol = np.dot(K, evoked.data[sel])  # apply imaging kernel
    sol = _combine_ori(sol, inv, pick_normal)

    if noise_norm is not None:
        print '(dSPM)...',
        sol *= noise_norm

    tstep = 1.0 / evoked.info['sfreq']
    tmin = float(evoked.first) / evoked.info['sfreq']
    src = inv['src']
    stc = _make_stc(sol, tmin, tstep, src[0]['vertno'], src[1]['vertno'])
    print '[done]'

    return stc


def apply_inverse_raw(raw, inverse_operator, lambda2, dSPM=True,
                      label=None, start=None, stop=None, nave=1,
                      time_func=None, pick_normal=False):
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
        Index of last time sample (index not time is seconds)
    nave: int
        Number of averages used to regularize the solution.
        Set to 1 on raw data.
    time_func: callable
        Linear function applied to sensor space time series.
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
    inv = prepare_inverse_operator(inverse_operator, nave, lambda2, dSPM)
    #
    #   Pick the correct channels from the data
    #
    sel = pick_channels(raw.ch_names, include=inv['noise_cov']['names'])
    print 'Picked %d channels from the data' % len(sel)
    print 'Computing inverse...',

    src = inv['src']
    lh_vertno = src[0]['vertno']
    rh_vertno = src[1]['vertno']

    data, times = raw[sel, start:stop]

    if time_func is not None:
        data = time_func(data)

    K, noise_norm, lh_vertno, rh_vertno = _assemble_kernel(inv, label, dSPM)
    sol = np.dot(K, data)
    sol = _combine_ori(sol, inv, pick_normal)

    if noise_norm is not None:
        sol *= noise_norm

    tmin = float(times[0]) / raw.info['sfreq']
    tstep = 1.0 / raw.info['sfreq']
    stc = _make_stc(sol, tmin, tstep, lh_vertno, rh_vertno)
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
    #
    #   Set up the inverse according to the parameters
    #
    inv = prepare_inverse_operator(inverse_operator, nave, lambda2, dSPM)
    #
    #   Pick the correct channels from the data
    #
    sel = pick_channels(epochs.ch_names, include=inv['noise_cov']['names'])
    print 'Picked %d channels from the data' % len(sel)

    print 'Computing inverse...',
    K, noise_norm, lh_vertno, rh_vertno = _assemble_kernel(inv, label, dSPM)

    stcs = list()
    tstep = 1.0 / epochs.info['sfreq']
    tmin = epochs.times[0]

    for k, e in enumerate(epochs):
        print "Processing epoch : %d" % (k + 1)
        sol = np.dot(K, e[sel])  # apply imaging kernel
        sol = _combine_ori(sol, inv, pick_normal)

        if noise_norm is not None:
            sol *= noise_norm

        stcs.append(_make_stc(sol, tmin, tstep, lh_vertno, rh_vertno))

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

    n_positions = gain.shape[1] / 3

    fwd_idx = [fwd_ch_names.index(name) for name in ch_names]
    gain = gain[fwd_idx]

    # Handle depth prior scaling
    depth_prior = np.ones(gain.shape[1])
    if depth is not None:
        depth_prior = compute_depth_prior(gain, exp=depth)

    print "Computing inverse operator with %d channels." % len(ch_names)

    if is_fixed_ori:
        n_dip_per_pos = 1
    else:
        n_dip_per_pos = 3

    n_dipoles = n_positions * n_dip_per_pos

    # Whiten lead field.
    print 'Whitening lead field matrix.'
    gain = np.dot(W, gain)

    # apply loose orientations
    orient_prior = np.ones(n_dipoles, dtype=np.float)
    if loose is not None:
        print 'Applying loose dipole orientations. Loose value of %s.' % loose
        orient_prior[np.mod(np.arange(n_dipoles), 3) != 2] *= loose

    source_cov = orient_prior * depth_prior

    # Adjusting Source Covariance matrix to make trace of G*R*G' equal
    # to number of sensors.
    print 'Adjusting source covariance matrix.'
    source_std = np.sqrt(source_cov)
    gain *= source_std[None, :]
    trace_GRGT = linalg.norm(gain, ord='fro') ** 2
    scaling_source_cov = n_nzero / trace_GRGT
    source_cov *= scaling_source_cov
    gain *= sqrt(scaling_source_cov)

    # now np.trace(np.dot(gain, gain.T)) == n_nzero
    # print np.trace(np.dot(gain, gain.T)), n_nzero

    print 'Computing SVD of whitened and weighted lead field matrix.'
    eigen_fields, sing, eigen_leads = linalg.svd(gain, full_matrices=False)

    eigen_fields = dict(data=eigen_fields.T)
    eigen_leads = dict(data=eigen_leads.T, nrow=eigen_leads.shape[1])
    depth_prior = dict(data=depth_prior)
    orient_prior = dict(data=orient_prior)
    source_cov = dict(data=source_cov)
    nave = 1.0

    inv_op = dict(eigen_fields=eigen_fields, eigen_leads=eigen_leads,
                  sing=sing, nave=nave, depth_prior=depth_prior,
                  source_cov=source_cov, noise_cov=noise_cov,
                  orient_prior=orient_prior, projs=deepcopy(info['projs']),
                  eigen_leads_weighted=False, source_ori=forward['source_ori'],
                  mri_head_t=deepcopy(forward['mri_head_t']),
                  src=deepcopy(forward['src']))

    return inv_op
