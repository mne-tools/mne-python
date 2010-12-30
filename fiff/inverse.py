from .constants import FIFF
from .open import fiff_open
from .tag import find_tag
from .matrix import read_named_matrix, transpose_named_matrix
from .cov import read_cov
from .proj import read_proj
from .tree import dir_tree_find
from .source_space import read_source_spaces, find_source_space_hemi
from .forward import invert_transform, transform_source_space_to


def read_inverse_operator(fname):
    """
    %
    % [inv] = mne_read_inverse_operator(fname)
    %
    % Reads the inverse operator decomposition from a fif file
    %
    % fname        - The name of the file
    %
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
    inv['methods'] = tag.data;

    tag = find_tag(fid, invs, FIFF.FIFF_MNE_SOURCE_ORIENTATION)
    if tag is None:
        fid.close()
        raise ValueError, 'Source orientation constraints not found'

    inv['source_ori'] = tag.data;

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
    print '[done]\n'
    #
    #   The SVD decomposition...
    #
    print '\tReading inverse operator decomposition...'
    tag = find_tag(fid, invs, FIFF.FIFF_MNE_INVERSE_SING)
    if tag is None:
        fid.close()
        raise ValueError, 'Singular values not found'

    inv['sing']  = tag.data
    inv['nchan'] = len(inv['sing'])
    #
    #   The eigenleads and eigenfields
    #
    inv['eigen_leads_weighted'] = False
    try:
       inv['eigen_leads'] = read_named_matrix(fid, invs, FIFF.FIFF_MNE_INVERSE_LEADS)
    except:
       inv['eigen_leads_weighted'] = True
       try:
          inv.eigen_leads = read_named_matrix(fid,invs,FIFF.FIFF_MNE_INVERSE_LEADS_WEIGHTED);
       except Exception as inst:
          raise ValueError, '%s' % inst
    #
    #   Having the eigenleads as columns is better for the inverse calculations
    #
    inv['eigen_leads'] = transpose_named_matrix(inv['eigen_leads'])
    try:
        inv['eigen_fields'] = read_named_matrix(fid, invs, FIFF.FIFF_MNE_INVERSE_FIELDS)
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
        print '\tDepth priors read.\n'
    except:
        inv['depth_prior'] = [];

    try:
        inv['fmri_prior'] = read_cov(fid, invs, FIFF.FIFFV_MNE_FMRI_PRIOR_COV)
        print '\tfMRI priors read.\n'
    except:
        inv['fmri_prior'] = [];

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
        mri_head_t = tag.data;
        if mri_head_t['from_'] != FIFF.FIFFV_COORD_MRI or \
                        mri_head_t['to'] != FIFF.FIFFV_COORD_HEAD:
            mri_head_t = invert_transform(mri_head_t)
            if mri_head_t['from_'] != FIFF.FIFFV_COORD_MRI or \
                        mri_head_t['to'] != FIFF.FIFFV_COORD_HEAD:
                fid.close()
                raise ValueError, 'MRI/head coordinate transformation not found'

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
    inv['nave'] = 1;
    #
    #  We also need the SSP operator
    #
    inv['projs'] = read_proj(fid, tree)
    #
    #  Some empty fields to be filled in later
    #
    inv['proj']      = []      #   This is the projector to apply to the data
    inv['whitener']  = []      #   This whitens the data
    inv['reginv']    = []      #   This the diagonal matrix implementing
                             #   regularization and the inverse
    inv['noisenorm'] = []      #   These are the noise-normalization factors
    #
    nuse = 0
    for k in range(len(inv['src'])):
       try:
          inv['src'][k] = transform_source_space_to(inv['src'][k],
                                                inv['coord_frame'], mri_head_t)
       except Exception as inst:
          fid.close()
          raise ValueError, 'Could not transform source space (%s)', inst

       nuse += inv['src'][k]['nuse']

    print '\tSource spaces transformed to the inverse solution coordinate frame'
    #
    #   Done!
    #
    fid.close()

    return inv