# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

from math import sqrt
import numpy as np

from .fiff.constants import FIFF
from .fiff.open import fiff_open
from .fiff.tag import find_tag
from .fiff.matrix import _read_named_matrix, _transpose_named_matrix
from .fiff.proj import read_proj, make_projector
from .fiff.tree import dir_tree_find
from .fiff.evoked import read_evoked
from .fiff.pick import pick_channels_evoked

from .cov import read_cov
from .source_space import read_source_spaces, find_source_space_hemi
from .forward import _invert_transform, _transform_source_space_to, \
                     _block_diag


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
        raise ValueError, 'No inverse solutions in %s' % fname

    invs = invs[0]
    #
    #   Parent MRI data
    #
    parent_mri = dir_tree_find(tree, FIFF.FIFFB_MNE_PARENT_MRI_FILE)
    if len(parent_mri) == 0:
        fid.close()
        raise ValueError, 'No parent MRI information in %s' % fname
    parent_mri = parent_mri[0]

    print '\tReading inverse operator info...'
    #
    #   Methods and source orientations
    #
    tag = find_tag(fid, invs, FIFF.FIFF_MNE_INCLUDED_METHODS)
    if tag is None:
        fid.close()
        raise ValueError, 'Modalities not found'

    inv = dict()
    inv['methods'] = tag.data

    tag = find_tag(fid, invs, FIFF.FIFF_MNE_SOURCE_ORIENTATION)
    if tag is None:
        fid.close()
        raise ValueError, 'Source orientation constraints not found'

    inv['source_ori'] = tag.data

    tag = find_tag(fid, invs, FIFF.FIFF_MNE_SOURCE_SPACE_NPOINTS)
    if tag is None:
        fid.close()
        raise ValueError, 'Number of sources not found'

    inv['nsource'] = tag.data
    inv['nchan'] = 0
    #
    #   Coordinate frame
    #
    tag = find_tag(fid, invs, FIFF.FIFF_MNE_COORD_FRAME)
    if tag is None:
        fid.close()
        raise ValueError, 'Coordinate frame tag not found'

    inv['coord_frame'] = tag.data
    #
    #   The actual source orientation vectors
    #
    tag = find_tag(fid, invs, FIFF.FIFF_MNE_INVERSE_SOURCE_ORIENTATIONS)
    if tag is None:
        fid.close()
        raise ValueError, 'Source orientation information not found'

    inv['source_nn'] = tag.data
    print '[done]'
    #
    #   The SVD decomposition...
    #
    print '\tReading inverse operator decomposition...'
    tag = find_tag(fid, invs, FIFF.FIFF_MNE_INVERSE_SING)
    if tag is None:
        fid.close()
        raise ValueError, 'Singular values not found'

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
        try:
            inv.eigen_leads = _read_named_matrix(fid, invs,
                                        FIFF.FIFF_MNE_INVERSE_LEADS_WEIGHTED)
        except Exception as inst:
            raise ValueError, '%s' % inst
    #
    #   Having the eigenleads as columns is better for the inverse calculations
    #
    inv['eigen_leads'] = _transpose_named_matrix(inv['eigen_leads'])
    try:
        inv['eigen_fields'] = _read_named_matrix(fid, invs,
                                                FIFF.FIFF_MNE_INVERSE_FIELDS)
    except Exception as inst:
        raise ValueError, '%s' % inst

    print '[done]'
    #
    #   Read the covariance matrices
    #
    try:
        inv['noise_cov'] = read_cov(fid, invs, FIFF.FIFFV_MNE_NOISE_COV)
        print '\tNoise covariance matrix read.'
    except Exception as inst:
        fid.close()
        raise ValueError, '%s' % inst

    try:
        inv['source_cov'] = read_cov(fid, invs, FIFF.FIFFV_MNE_SOURCE_COV)
        print '\tSource covariance matrix read.'
    except Exception as inst:
        fid.close()
        raise ValueError, '%s' % inst
    #
    #   Read the various priors
    #
    try:
        inv.orient_prior = read_cov(fid, invs, FIFF.FIFFV_MNE_ORIENT_PRIOR_COV)
        print '\tOrientation priors read.'
    except Exception as inst:
        inv['orient_prior'] = []

    try:
        inv['depth_prior'] = read_cov(fid, invs,
                                          FIFF.FIFFV_MNE_DEPTH_PRIOR_COV)
        print '\tDepth priors read.'
    except:
        inv['depth_prior'] = []

    try:
        inv['fmri_prior'] = read_cov(fid, invs, FIFF.FIFFV_MNE_FMRI_PRIOR_COV)
        print '\tfMRI priors read.'
    except:
        inv['fmri_prior'] = []

    #
    #   Read the source spaces
    #
    try:
        inv['src'] = read_source_spaces(fid, False, tree)
    except Exception as inst:
        fid.close()
        raise ValueError, 'Could not read the source spaces (%s)' % inst

    for s in inv['src']:
        s['id'] = find_source_space_hemi(s)

    #
    #   Get the MRI <-> head coordinate transformation
    #
    tag = find_tag(fid, parent_mri, FIFF.FIFF_COORD_TRANS)
    if tag is None:
        fid.close()
        raise ValueError, 'MRI/head coordinate transformation not found'
    else:
        mri_head_t = tag.data
        if mri_head_t['from_'] != FIFF.FIFFV_COORD_MRI or \
                        mri_head_t['to'] != FIFF.FIFFV_COORD_HEAD:
            mri_head_t = _invert_transform(mri_head_t)
            if mri_head_t['from_'] != FIFF.FIFFV_COORD_MRI or \
                        mri_head_t['to'] != FIFF.FIFFV_COORD_HEAD:
                fid.close()
                raise ValueError, ('MRI/head coordinate transformation '
                                   'not found')

    inv['mri_head_t'] = mri_head_t
    #
    #   Transform the source spaces to the correct coordinate frame
    #   if necessary
    #
    if inv['coord_frame'] != FIFF.FIFFV_COORD_MRI and \
            inv['coord_frame'] != FIFF.FIFFV_COORD_HEAD:
        fid.close()
        raise ValueError, 'Only inverse solutions computed in MRI or ' \
                          'head coordinates are acceptable'

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
    inv['proj'] = []      #   This is the projector to apply to the data
    inv['whitener'] = []      #   This whitens the data
    inv['reginv'] = []      #   This the diagonal matrix implementing
                             #   regularization and the inverse
    inv['noisenorm'] = []      #   These are the noise-normalization factors
    #
    nuse = 0
    for k in range(len(inv['src'])):
        try:
            inv['src'][k] = _transform_source_space_to(inv['src'][k],
                                                inv['coord_frame'], mri_head_t)
        except Exception as inst:
            fid.close()
            raise ValueError, 'Could not transform source space (%s)', inst

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

def combine_xyz(vec):
    """
    %
    % function [comb] = mne_combine_xyz(vec)
    %
    % Compute the three Cartesian components of a vector together
    %
    %
    % vec         - Input row or column vector [ x1 y1 z1 ... x_n y_n z_n ]
    % comb        - Output vector [x1^2+y1^2+z1^2 ... x_n^2+y_n^2+z_n^2 ]
    %
    """
    if vec.ndim != 1 or (vec.size % 3) != 0:
        raise ValueError, ('Input must be a 1D vector with '
                           '3N entries')

    s = _block_diag(vec[None, :], 3)
    comb = (s * s.T).diagonal()
    return comb


def prepare_inverse_operator(orig, nave, lambda2, dSPM):
    """
    %
    % [inv] = mne_prepare_inverse_operator(orig,nave,lambda2,dSPM)
    %
    % Prepare for actually computing the inverse
    %
    % orig        - The inverse operator structure read from a file
    % nave        - Number of averages (scales the noise covariance)
    % lambda2     - The regularization factor
    % dSPM        - Compute the noise-normalization factors for dSPM?
    %
    """

    if nave <= 0:
        raise ValueError, 'The number of averages should be positive'

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
    inv['reginv'] = inv['sing'] / (inv['sing'] * inv['sing'] + lambda2)
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
    inv['whitener'] = np.zeros((inv['noise_cov']['dim'],
                                inv['noise_cov']['dim']))
    if inv['noise_cov']['diag'] == 0:
        #
        #   Omit the zeroes due to projection
        #
        nnzero = 0
        for k in range(ncomp, inv['noise_cov']['dim']):
            if inv['noise_cov']['eig'][k] > 0:
                inv['whitener'][k, k] = 1.0 / sqrt(inv['noise_cov']['eig'][k])
                nnzero += 1

        #
        #   Rows of eigvec are the eigenvectors
        #
        inv['whitener'] = np.dot(inv['whitener'], inv['noise_cov']['eigvec'])
        print ('\tCreated the whitener using a full noise covariance matrix '
              '(%d small eigenvalues omitted)' % (inv['noise_cov']['dim']
                                                  - nnzero))
    else:
        #
        #   No need to omit the zeroes due to projection
        #
        for k in range(inv['noise_cov']['dim']):
            inv['whitener'][k, k] = 1.0 / sqrt(inv['noise_cov']['data'][k])

        print ('\tCreated the whitener using a diagonal noise covariance '
               'matrix (%d small eigenvalues discarded)' % ncomp)

    #
    #   Finally, compute the noise-normalization factors
    #
    if dSPM:
        print '\tComputing noise-normalization factors...'
        noise_norm = np.zeros(inv['eigen_leads']['nrow'])
        if inv['eigen_leads_weighted']:
            for k in range(inv['eigen_leads']['nrow']):
                one = inv['eigen_leads']['data'][k, :] * inv['reginv']
                noise_norm[k] = np.sum(one**2)
        else:
            for k in range(inv['eigen_leads']['nrow']):
                one = sqrt(inv['source_cov']['data'][k]) * \
                            np.sum(inv['eigen_leads']['data'][k, :]
                                   * inv['reginv'])
                noise_norm[k] = np.sum(one**2)

        #
        #   Compute the final result
        #
        if inv['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
            #
            #   The three-component case is a little bit more involved
            #   The variances at three consequtive entries must be squeared and
            #   added together
            #
            #   Even in this case return only one noise-normalization factor
            #   per source location
            #
            noise_norm = np.sqrt(combine_xyz(noise_norm))
            #
            #   This would replicate the same value on three consequtive
            #   entries
            #
            #   noise_norm = kron(sqrt(mne_combine_xyz(noise_norm)),ones(3,1));

        inv['noisenorm'] = np.diag(1.0 / np.abs(noise_norm)) # XXX
        print '[done]'
    else:
        inv['noisenorm'] = []

    return inv


def compute_inverse(fname_data, setno, fname_inv, lambda2, dSPM=True,
                    nave=None):
    """Compute inverse solution

    Computes a L2-norm inverse solution
    Actual code using these principles might be different because
    the inverse operator is often reused across data sets.

    Parameters
    ----------
    fname: string
        File name of the data file
    setno: int
        Data set number
    fname_inv: string
        File name of the inverse operator
    nave: int
        Number of averages (scales the noise covariance)
        If negative, the number of averages in the data will be used XXX
    lambda2: float
        The regularization parameter
    dSPM: bool
        do dSPM ?

    Returns
    -------
    res: dict
        Inverse solution
    """

    #
    #   Read the data first
    #
    data = read_evoked(fname_data, setno)
    #
    #   Then the inverse operator
    #
    inv = read_inverse_operator(fname_inv)
    #
    #   Set up the inverse according to the parameters
    #
    if nave is None:
        nave = data['evoked']['nave']

    inv = prepare_inverse_operator(inv, nave, lambda2, dSPM)
    #
    #   Pick the correct channels from the data
    #
    data = pick_channels_evoked(data, inv['noise_cov']['names'])
    print 'Picked %d channels from the data' % data['info']['nchan']
    print 'Computing inverse...',
    #
    #   Simple matrix multiplication followed by combination of the
    #   three current components
    #
    #   This does all the data transformations to compute the weights for the
    #   eigenleads
    #
    trans = reduce(np.dot, [np.diag(inv['reginv']),
                            inv['eigen_fields']['data'],
                            inv['whitener'],
                            inv['proj'],
                            data['evoked']['epochs']])
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
        #     R^0.5 has to factored in
        #
        print '(eigenleads need to be weighted)...',
        sol = np.sqrt(inv['source_cov']['data'])[:, None] * \
                             np.dot(inv['eigen_leads']['data'], trans)


    if inv['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
        print 'combining the current components...',
        sol1 = np.zeros((sol.shape[0]/3, sol.shape[1]))
        for k in range(sol.shape[1]):
            sol1[:, k] = np.sqrt(combine_xyz(sol[:, k]))

        sol = sol1

    if dSPM:
        print '(dSPM)...',
        sol = np.dot(inv['noisenorm'], sol)

    res = dict()
    res['inv'] = inv
    res['sol'] = sol
    res['tmin'] = float(data['evoked']['first']) / data['info']['sfreq']
    res['tstep'] = 1.0 / data['info']['sfreq']
    print '[done]'

    return res
