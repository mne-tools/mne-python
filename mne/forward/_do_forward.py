# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

from os import path as op
import numpy as np

from ..fiff.raw import Raw
from ..fiff.pick import pick_types, pick_info
from ..fiff.constants import FIFF
from ..fiff.open import fiff_open
from ..fiff.tree import dir_tree_find
from ..fiff.tag import find_tag
from .forward import write_forward_solution, _merge_meg_eeg_fwds
from ._compute_forward import _compute_forward
from ..transforms import (invert_transform, transform_source_space_to,
                          read_trans, read_transform_ascii, combine_transforms,
                          apply_trans)
from ..utils import logger, verbose
from ..source_space import read_source_spaces
from ..surface import read_bem_surfaces, _compute_nearest, fast_cross_3d


# Not currently supported:
#    Correct FWD calculation (hah!)
#    Correct fwd['info'] population/restriction
#    EEG Sphere model
#    Label specification
#    Channel compensation


def _coord_frame_name(cframe):
    """Map integers to human-readable names"""
    types = [FIFF.FIFFV_COORD_UNKNOWN, FIFF.FIFFV_COORD_DEVICE,
             FIFF.FIFFV_COORD_ISOTRAK, FIFF.FIFFV_COORD_HPI,
             FIFF.FIFFV_COORD_HEAD, FIFF.FIFFV_COORD_MRI,
             FIFF.FIFFV_MNE_COORD_MRI_VOXEL, FIFF.FIFFV_COORD_MRI_SLICE,
             FIFF.FIFFV_COORD_MRI_DISPLAY, FIFF.FIFFV_MNE_COORD_CTF_DEVICE,
             FIFF.FIFFV_MNE_COORD_CTF_HEAD, FIFF.FIFFV_MNE_COORD_RAS,
             FIFF.FIFFV_MNE_COORD_MNI_TAL, FIFF.FIFFV_MNE_COORD_FS_TAL_GTZ,
             FIFF.FIFFV_MNE_COORD_FS_TAL_LTZ, -1]
    strs = ['unknown', 'MEG device', 'isotrak', 'hpi', 'head',
            'MRI (surface RAS)', 'MRI voxel', 'MRI slice', 'MRI display',
            'CTF MEG device', 'CTF/4D/KIT head', 'RAS (non-zero origin)',
            'MNI Talairach', 'Talairach (MNI z > 0)', 'Talairach (MNI z < 0)',
            'unknown']
    assert len(types) == len(strs)
    for t, s in zip(types, strs):
        if cframe == t:
            return s
    return strs[-1]


def _print_coord_trans(t):
    logger.info('Coordinate transformation: ')
    logger.info('%s -> %s'
                % (_coord_frame_name(t['from']), _coord_frame_name(t['to'])))
    for tt in t['trans']:
        logger.info('    % 8.6f % 8.6f % 8.6f    %7.2f mm' %
                    (tt[0], tt[1], tt[2], 1000 * tt[3]))


def _read_coil_defs(fname):
    """Read a coil definition file"""
    big_val = 0.5
    with open(fname, 'r') as fid:
        lines = fid.readlines()
        res = dict(coils=list())
        lines = lines[::-1]
        while len(lines) > 0:
            line = lines.pop()
            if line[0] != '#':
                vals = np.fromstring(line, sep=' ')
                assert len(vals) == 7
                start = line.find('"')
                end = len(line.strip()) - 1
                assert line.strip()[end] == '"'
                desc = line[start:end]
                coil = dict(coil_type=vals[1], coil_class=vals[0],
                            accuracy=vals[2], np=int(vals[3]),
                            size=vals[4], base=vals[5], desc=desc)
                coil['rmag'] = np.zeros((coil['np'], 3))
                coil['cosmag'] = np.zeros((coil['np'], 3))
                coil['w'] = np.zeros(coil['np'])
                for p in xrange(coil['np']):
                    # get next non-comment line
                    line = lines.pop()
                    while(line[0] == '#'):
                        line = lines.pop()
                    vals = np.fromstring(line, sep=' ')
                    assert len(vals) == 7
                    # Read and verify data for each integration point
                    coil['w'][p] = vals[0]
                    coil['rmag'][p] = vals[[1, 2, 3]]
                    coil['cosmag'][p] = vals[[4, 5, 6]]
                    if np.sqrt(np.sum(coil['rmag'][p] ** 2)) > big_val:
                        raise RuntimeError('Unreasonable integration point')
                    size = np.sqrt(np.sum(coil['cosmag'][p]))
                    if size <= 0:
                        raise RuntimeError('Unreasonable normal')
                    coil['cosmag'][p] /= size
                res['coils'].append(coil)
    res['ncoil'] = len(res['coils'])
    logger.info('%d coil definitions read', res['ncoil'])
    return res


def _bem_load_solution(fname, m):
    """Load the solution matrix"""
    # convert from surfaces to solution
    m = dict(surfs=m)
    f, tree, _ = fiff_open(fname)
    with f as fid:
        # Find the BEM data
        nodes = dir_tree_find(tree, FIFF.FIFFB_BEM)
        if len(nodes) == 0:
            raise RuntimeError('No BEM data in %s' % fname)
        bem_node = nodes[0]

        # Approximation method
        tag = find_tag(f, bem_node, FIFF.FIFF_BEM_APPROX)
        method = tag.data[0]
        if method == FIFF.FIFFV_BEM_APPROX_CONST:
            method = 'constant collocation'
        elif method == FIFF.FIFFV_BEM_APPROX_LINEAR:
            method = 'linear collocation'
        else:
            raise RuntimeError('Cannot handle BEM approximation method : %d'
                               % method)

        tag = find_tag(fid, bem_node, FIFF.FIFF_BEM_POT_SOLUTION)
        dims = tag.data.shape
        if len(dims) != 2:
            raise RuntimeError('Expected a two-dimensional solution matrix '
                               'instead of a %d dimensional one' % dims[0])

        dim = 0
        for mm in m['surfs']:
            if method == 'linear collocation':
                dim += mm['np']
            else:
                dim += mm['ntri']

        if dims[0] != dim or dims[1] != dim:
            raise RuntimeError('Expected a %d x %d solution matrix instead of '
                               'a %d x %d one' % (dim, dim, dims[1], dims[0]))
        sol = tag.data
        nsol = dims[0]

    # Gamma factors and multipliers
    m['sigma'] = np.array([s['sigma'] for s in m['surfs']])
    # Dirty trick for the zero conductivity outside
    sigma = np.r_[0.0, m['sigma']]
    m['source_mult'] = 2.0 / (sigma[1:] + sigma[:-1])
    m['field_mult'] = sigma[1:] - sigma[:-1]
    m['gamma'] = ((sigma[1:] - sigma[:-1])[np.newaxis, :] /
                  (sigma[1:] + sigma[:-1])[:, np.newaxis])
    m['sol_name'] = fname
    m['solution'] = sol
    m['nsol'] = nsol
    m['bem_method'] = method
    logger.info('Loaded %s BEM solution from %s', m['bem_method'], fname)
    return m


def _create_meg_coil(coilset, ch, acc, t):
    """Create a coil definition using templates, transform if necessary"""
    # Also change the coordinate frame if so desired

    if ch['kind'] not in [FIFF.FIFFV_MEG_CH, FIFF.FIFFV_REF_MEG_CH]:
        raise RuntimeError('%s is not a MEG channel' % ch['ch_name'])

    # Simple linear search from the coil definitions
    d = None
    for coil in coilset['coils']:
        if coil['coil_type'] == (ch['coil_type'] & 0xFFFF) and \
                coil['accuracy'] == acc:
            d = coil
    if d is None:
        raise RuntimeError('Desired coil definition not found '
                           '(type = %d acc = %d)' % (ch['coil_type'], acc))

    # Create the result
    res = dict(chname=ch['ch_name'], desc=None, coil_class=d['coil_class'],
               accuracy=d['accuracy'], base=d['base'], size=d['size'],
               type=d['coil_type'], np=d['np'], w=d['w'])

    if d['desc']:
        res['desc'] = d['desc']

    # Apply a coordinate transformation if so desired
    coil_trans = ch['coil_trans'].copy()  # make sure we don't botch it
    if t is not None:
        coil_trans = np.dot(t['trans'], coil_trans)
        res['coord_frame'] = t['to']
    else:
        res['coord_frame'] = FIFF.FIFFV_COORD_DEVICE

    res['rmag'] = apply_trans(coil_trans, d['rmag'])
    res['cosmag'] = apply_trans(coil_trans, d['cosmag'], False)
    res.update(ex=coil_trans[:3, 0], ey=coil_trans[:3, 1],
               ez=coil_trans[:3, 2], r0=coil_trans[:3, 3])
    return res


def _create_eeg_el(ch, t):
    """Create an electrode definition, transform coords if necessary"""
    if ch['kind'] != FIFF.FIFFV_EEG_CH:
        raise RuntimeError('%s is not an EEG channel. Cannot create an '
                           'electrode definition.' % ch['ch_name'])
    if t is not None and t['from'] != FIFF.FIFFV_COORD_HEAD:
        raise RuntimeError('Inappropriate coordinate transformation')

    r0 = ch['eeg_loc'][:, 0].copy()
    ex = ch['eeg_loc'][:, 1].copy()
    ex_size = np.sqrt(np.sum(ex ** 2))
    if ex_size < 1e-4:
        w = np.array([1.])
    else:
        w = np.array([1., -1.])
    res = dict(chname=ch['ch_name'], desc='EEG electrode',
               coil_class=FIFF.FWD_COILC_EEG,
               accuracy=FIFF.FWD_COIL_ACCURACY_NORMAL,
               base=0.0, size=0.0, type=ch['coil_type'], np=len(w), w=w)

    # Optional coordinate transformation
    if t is not None:
        r0 = apply_trans(t['trans'], r0)
        ex = apply_trans(t['trans'], ex)
        res['coord_frame'] = t['to']
    else:
        res['coord_frame'] = FIFF.FIFFV_COORD_HEAD

    # The electrode location
    res.update(dict(ex=ex, r0=r0))
    res['rmag'] = r0.copy()
    res['cosmag'] = r0.copy() / np.sqrt(np.sum(r0 ** 2))

    # Add the reference electrode, if appropriate
    if res['np'] == 2:
        res['rmag'] = np.array([res['rmag'], ex])
        res['cosmag'] = np.array([res['cosmag'], ex / ex_size])
    return res


def _create_coils(coilset, chs, acc, t, coil_type='meg'):
    """Create a set of MEG or EEG coils"""
    coils = list()
    for ch in chs:
        if coil_type == 'meg':
            coils.append(_create_meg_coil(coilset, ch, acc, t))
        elif coil_type == 'eeg':
            coils.append(_create_eeg_el(ch, t))
        else:
            raise RuntimeError('unknown coil type')
    res = dict(coils=coils, ncoil=len(coils), coord_frame=t['to'])
    return res


def _sum_solids_div(fros, surf):
    """Compute sum of solid angles according to van Oosterom for all tris"""
    # NOTE: This incorporates the division by 4PI that used to be separate
    tot_angle = np.zeros((len(fros)))
    for tri in surf['tris']:
        v1 = fros - surf['rr'][tri[0]]
        v2 = fros - surf['rr'][tri[1]]
        v3 = fros - surf['rr'][tri[2]]
        triple = np.sum(fast_cross_3d(v1, v2) * v3, axis=1)
        l1 = np.sqrt(np.sum(v1 * v1, axis=1))
        l2 = np.sqrt(np.sum(v2 * v2, axis=1))
        l3 = np.sqrt(np.sum(v3 * v3, axis=1))
        s = (l1 * l2 * l3 +
             np.sum(v1 * v2, axis=1) * l3 +
             np.sum(v1 * v3, axis=1) * l2 +
             np.sum(v2 * v3, axis=1) * l1)
        tot_angle -= np.arctan2(triple, s)
    return tot_angle / (2 * np.pi)


@verbose
def _filter_source_spaces(surf, limit, mri_head_t, src, verbose=None):
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
        out_str += 'and at least %6.1f mm away' % (limit)
    logger.info(out_str + ' (will take a few...)')

    for s in src:
        vertno = np.where(s['inuse'])[0]  # can't trust s['vertno'] this deep
        # Convert all points here first to save time
        r1s = s['rr'][vertno]
        if s['coord_frame'] == FIFF.FIFFV_COORD_HEAD:
            r1s = apply_trans(inv_trans['trans'], r1s)

        # Check that the source is inside the inner skull surface
        x = _sum_solids_div(r1s, surf)
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

        if omit_outside > 0:
            logger.info('%d source space points omitted because they are '
                        'outside the inner skull surface.' % omit_outside)
        if omit > 0:
            logger.info('%d source space points omitted because of the '
                        '%6.1f-mm distance limit.' % (omit, limit))
    logger.info('Thank you for waiting.')


@verbose
def do_forward_solution2(subject, meas, fname=None, src=None, spacing=None,
                         mindist=0.0, bem=None, mri=None, trans=None,
                         eeg=True, meg=True, fixed=False, grad=False,
                         mricoord=False, do_all=False, overwrite=False,
                         subjects_dir=None, verbose=None):
    if src is None:
        raise ValueError('Source space file "src" must be specified')
    elif not op.isfile(src):
        raise IOError('Source space file "%s" not found' % src)
    if bem is None:
        raise ValueError('BEM file "bem" must be specified')
    elif not op.isfile(bem):
        raise IOError('BEM file "%s" not found' % bem)
    if sum([mri is None, trans is None]) != 1:
        raise ValueError('Either "mri" or "trans" must be specified '
                         '(but not both)')
    else:
        if mri is not None and not op.isfile(mri):
            raise IOError('mri file "%s" not found' % mri)
        if trans is not None and not op.isfile(trans):
            raise IOError('trans file "%s" not found' % trans)

    coord_frame = FIFF.FIFFV_COORD_HEAD

    # Report the setup
    logger.info('Source space                 : %s' % src)
    logger.info('MRI -> head transform source : %s'
                % (mri if mri is not None else trans))
    logger.info('Measurement data             : %s' % meas)
    logger.info('BEM model                    : %s' % bem)
    # XXX Left out EEG sphere model code in mne_forward_solution.c
    logger.info('Accurate field computations')
    logger.info('Do computations in %s coordinates',
                _coord_frame_name(coord_frame))
    logger.info("%s source orientations" % ('Fixed' if fixed else 'Free'))
    if grad is True:
        logger.info('Compute derivatives with respect to source location '
                    'coordinates')
    logger.info('Destination for the solution : %s' % fname)
    if do_all is True:
        logger.info('Calculate solution for all source locations.')

    # Read the source locations
    logger.info('')
    logger.info('Reading %s...' % src)
    src = read_source_spaces(src, verbose=False)
    if do_all is True:
        for s in src:
            s['inuse'].fill(True)
            s['nuse'] = s['np']
    nsource = sum(s['nuse'] for s in src)
    if nsource == 0:
        raise RuntimeError('No sources are active in these source spaces. '
                           '"do_all" option should be used.')
    logger.info('Read %d source spaces a total of %d active source locations'
                % (len(src), nsource))

    # Read the MRI -> head coordinate transformation
    logger.info('')
    if mri is not None:
        mri_head_t = read_trans(mri)
        if mri_head_t['from'] == FIFF.FIFFV_COORD_HEAD:
            mri_head_t = invert_transform(mri_head_t)
        if not (mri_head_t['from'] == FIFF.FIFFV_COORD_MRI and
                mri_head_t['to'] == FIFF.FIFFV_COORD_HEAD):
            raise RuntimeError('Incorrect MRI transform provided')
    elif trans is not None:
        t = read_transform_ascii(trans, FIFF.FIFFV_COORD_HEAD,
                                 FIFF.FIFFV_COORD_MRI)
        mri_head_t = invert_transform(t)
    _print_coord_trans(mri_head_t)

    # Read the channel information & MEG device -> head coord trans
    raw = Raw(meas, verbose=False)
    info = raw.info
    meg_head_t = info['dev_head_t']

    # MEG channels
    logger.info('')
    picks = pick_types(info, meg=True, eeg=False, exclude=[])
    nmeg = len(picks)
    megchs = pick_info(info, picks)['chs']
    megnames = [info['ch_names'][p] for p in picks]
    if nmeg > 0:
        logger.info('Read %3d MEG channels from %s' % (len(picks), meas))

    # comp channels
    compchs = info['comps']
    ncomp = len(compchs)
    if (ncomp > 0):
        logger.info('Read %3d MEG compensation channels from %s'
                    % (ncomp, meas))
    comp_data = None  # ACTUALLY NEED THIS XXX

    # EEG channels
    picks = pick_types(info, meg=False, eeg=True, exclude=[])
    neeg = len(picks)
    eegchs = pick_info(info, picks)['chs']
    eegnames = [info['ch_names'][p] for p in picks]
    if neeg > 0:
        logger.info('Read %3d EEG channels from %s' % (len(picks), meas))

    if not meg:
        logger.info('MEG not requested. MEG channels omitted.')
        nmeg = 0
    else:
        _print_coord_trans(meg_head_t)
    if not eeg:
        logger.info('EEG not requested. EEG channels omitted.')
        neeg = 0

    # Create coil descriptions with transformation to head or MRI frame
    templates = _read_coil_defs(op.join(op.split(__file__)[0],
                                        '..', 'data', 'coil_def.dat'))
    if meg is True:
        # Compensation data
        if ncomp > 0:  # Compensation channel information may be needed
            logger.info('%d compensation data sets in %s' % (ncomp, meas))

    if coord_frame == FIFF.FIFFV_COORD_MRI:
        head_mri_t = invert_transform(mri_head_t)
        meg_xform = combine_transforms(FIFF.FIFFV_COORD_DEVICE,
                                       FIFF.FIFFV_COORD_MRI,
                                       meg_head_t, head_mri_t)
        eeg_xform = head_mri_t
        extra_str = 'MRI'
    else:
        meg_xform = meg_head_t
        eeg_xform = {'trans': np.eye(4), 'to': FIFF.FIFFV_COORD_HEAD,
                     'from': FIFF.FIFFV_COORD_HEAD}
        extra_str = 'Head'

    megcoils = _create_coils(templates, megchs,
                             FIFF.FWD_COIL_ACCURACY_ACCURATE,
                             meg_xform, coil_type='meg')
    if ncomp > 0:
        compcoils = _create_coils(templates, compchs,
                                  FIFF.FWD_COIL_ACCURACY_NORMAL,
                                  meg_xform, coil_type='meg')
    else:
        compcoils = None
    eegels = _create_coils(templates, eegchs, None,
                           eeg_xform, coil_type='eeg')
    logger.info('%s coordinate coil definitions created.' % extra_str)

    # Transform the source spaces into the appropriate coordinates
    for s in src:
        transform_source_space_to(s, coord_frame, mri_head_t)
    logger.info('Source spaces are now in %s coordinates.'
                % _coord_frame_name(coord_frame))

    # Prepare the BEM model
    logger.info('')
    logger.info('Setting up the BEM model using %s...' % bem)
    logger.info('\nLoading surfaces...')
    bem_model = read_bem_surfaces(bem, add_geom=True, verbose=False)
    if len(bem_model) == 3:
        logger.info('Three-layer model surfaces loaded.')
        needed = np.array([FIFF.FIFFV_BEM_SURF_ID_HEAD,
                           FIFF.FIFFV_BEM_SURF_ID_SKULL,
                           FIFF.FIFFV_BEM_SURF_ID_BRAIN])
        if not all([x['id'] in needed for x in bem_model]):
            raise RuntimeError('Could not find necessary BEM surfaces')
        # reorder surfaces as necessary (shouldn't need to?)
        reorder = [None] * 3
        for x in bem_model:
            reorder[np.where(x['id'] == needed)[0]] = x
        bem_model = reorder
    elif len(bem_model) == 1:
        if not bem_model[0]['id'] == FIFF.FIFFV_BEM_SURF_ID_BRAIN:
            raise RuntimeError('BEM Surfaces not found')
        logger.info('Homogeneous model surface loaded.')
        if neeg > 0:
            raise RuntimeError('Cannot use a homogeneous model in EEG '
                               'calculations')

    logger.info('\nLoading the solution matrix...\n')
    bem_model = _bem_load_solution(bem, bem_model)
    if coord_frame == FIFF.FIFFV_COORD_HEAD:
        logger.info('Employing the head->MRI coordinate transform with the '
                    'BEM model.')
        # fwd_bem_set_head_mri_t: Set the coordinate transformation
        to, fro = mri_head_t['to'], mri_head_t['from']
        if fro == FIFF.FIFFV_COORD_HEAD and to == FIFF.FIFFV_COORD_MRI:
            bem_model['head_mri_t'] = mri_head_t
        elif fro == FIFF.FIFFV_COORD_MRI and to == FIFF.FIFFV_COORD_HEAD:
            bem_model['head_mri_t'] = invert_transform(mri_head_t)
        else:
            raise RuntimeError('Improper coordinate transform')
    logger.info('BEM model %s is now set up' % bem)
    logger.info('')

    # Circumvent numerical problems by excluding points too close to the skull
    idx = np.where(np.array([s['id'] for s in bem_model['surfs']])
                   == FIFF.FIFFV_BEM_SURF_ID_BRAIN)[0]
    if len(idx) != 1:
        raise RuntimeError('BEM model does not have the inner skull '
                           'triangulation')
    _filter_source_spaces(bem_model['surfs'][idx], mindist, mri_head_t, src)
    logger.info('')

    # Do the actual computation
    megfwd, megfwd_grad, eegfwd, eegfwd_grad = None, None, None, None
    ori = FIFF.FIFFV_MNE_FIXED_ORI if fixed else FIFF.FIFFV_MNE_FREE_ORI
    cf = coord_frame
    if nmeg > 0:
        out = _compute_forward(src, megcoils, compcoils, comp_data, fixed,
                               bem_model, None, coil_type='meg', grad=grad)
        megfwd, megfwd_grad = out
        megfwd = _to_forward_dict(megfwd, megfwd_grad, megnames, cf, ori)
    if neeg > 0:
        out = _compute_forward(src, eegels, None, None, fixed,
                               bem_model, None, coil_type='eeg', grad=grad)
        eegfwd, eegfwd_grad = out
        eegfwd = _to_forward_dict(eegfwd, eegfwd_grad, eegnames, cf, ori)

    # merge forwards into one
    fwd = _merge_meg_eeg_fwds(megfwd, eegfwd, verbose=False)
    logger.info('')

    # pick out final dict info
    picks = pick_types(info, meg=meg, eeg=eeg, ref_meg=meg, exclude=[])
    fwd_info = pick_info(info, picks)
    source_rr = np.concatenate([s['rr'][s['inuse'] == 1] for s in src])
    if fixed:
        nsource = fwd['sol']['data'].shape[1]
        source_nn = np.concatenate([s['nn'][s['inuse'] == 1] for s in src])
    else:
        nsource = fwd['sol']['data'].shape[1] / 3
        source_nn = np.tile(np.eye(3), (nsource, 1))

    # Transform the source spaces back into MRI coordinates
    for s in src:
        transform_source_space_to(s, FIFF.FIFFV_COORD_MRI, mri_head_t)

    fwd.update(dict(nchan=fwd['sol']['data'].shape[0], nsource=nsource,
                    info=fwd_info, src=src, source_nn=source_nn,
                    source_rr=source_rr, surf_ori=False,
                    mri_head_t=mri_head_t))
    if fname is not None:
        logger.info('writing %s...', fname)
        write_forward_solution(fname, fwd, overwrite)

    logger.info('Finished.')
    return fwd


def _to_forward_dict(fwd, fwd_grad, names, coord_frame, source_ori):
    """Convert forward solution matrices to mne-python dictionary"""
    sol = dict(data=fwd.T, nrow=fwd.shape[1], ncol=fwd.shape[0],
               row_names=names, col_names=[])
    fwd = dict(sol=sol, source_ori=source_ori, nsource=sol['ncol'],
               coord_frame=coord_frame, sol_grad=None,
               nchan=sol['nrow'])
    if fwd_grad is not None:
        sol_grad = dict(data=fwd_grad.T, nrow=fwd_grad.shape[1],
                        ncol=fwd_grad.shape[0], row_names=names,
                        col_names=[])
        fwd.update(dict(sol_grad=sol_grad))
    return fwd
