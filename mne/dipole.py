# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import numpy as np
from scipy import optimize, linalg
import re

from .cov import read_cov, whiten_evoked, _whiten_data
from .io.pick import pick_types
from .io.constants import FIFF
from .bem import make_sphere_model, _fit_sphere
from .transforms import (_print_coord_trans, apply_trans, invert_transform,
                         _coord_frame_name)
from .forward._make_forward import (_get_mri_head_t, _setup_bem,
                                    _prep_channels)
from .forward._compute_forward import (_compute_forwards_meeg,
                                       _prep_field_computation)

from .externals.six import string_types
from .surface import (_bem_find_surface, transform_surface_to,
                      _normalize_vectors, _get_ico_surface,
                      _bem_explain_surface)
from .source_space import _filter_source_spaces
from .parallel import parallel_func
from .fixes import partial
from .utils import logger, verbose, deprecated


class Dipole(dict):
    """Dipole class

    Used to store positions, orientations, amplitudes, times, goodness of fit
    of dipoles, typically obtained with Neuromag/xfit, mne_dipole_fit
    or certain inverse solvers.

    Parameters
    ----------
    times : array, shape (n_dipoles,)
        The time instants at which each dipole was fitted.
    pos : array, shape (n_dipoles, 3)
        The dipoles positions in meters
    amplitude : array, shape (n_dipoles,)
        The amplitude of the dipoles in nAm
    ori : array, shape (n_dipoles, 3)
        The dipolar moments. Amplitude of the moment is in nAm.
    gof : array, shape (n_dipoles,)
        The goodness of fit
    """
    def __init__(self, times, pos, amplitude, ori, gof, name=None):
        self['times'] = times
        self['pos'] = pos
        self['amplitude'] = amplitude
        self['ori'] = ori
        self['gof'] = gof
        if name is not None:
            self['name'] = name

    def __repr__(self):
        s = "n_times : %s" % len(self['times'])
        s += ", tmin : %s" % np.min(self['times'])
        s += ", tmax : %s" % np.max(self['times'])
        return "<Dipole  |  %s>" % s

    def save(self, fname):
        """Save dipole in a .dip file

        Parameters
        ----------
        fname : str
            The name of the .dip file.
        """
        fmt = "  %7.1f %7.1f %8.2f %8.2f %8.2f %8.3f %8.3f %8.3f %8.3f %6.1f"
        with open(fname, 'wb') as fid:
            fid.write('# CoordinateSystem "Head"\n'.encode('utf-8'))
            fid.write('#   begin     end   X (mm)   Y (mm)   Z (mm)'
                      '   Q(nAm)  Qx(nAm)  Qy(nAm)  Qz(nAm)    g/%\n'
                      .encode('utf-8'))
            t = self['times'][:, np.newaxis]
            gof = self['gof'][:, np.newaxis]
            amp = self['amplitude'][:, np.newaxis]
            out = np.concatenate((t, t, self['pos'] / 1e-3, amp,
                                  self['ori'], gof), axis=-1)
            np.savetxt(fid, out, fmt=fmt)
            if self.get('name') is not None:
                fid.write(('## Name "%s dipoles" Style "Dipoles"'
                           % self['name']).encode('utf-8'))


# #############################################################################
# IO

@deprecated("'read_dip' will be removed in version 0.10, please use "
            "'read_dipole' instead")
def read_dip(fname, verbose=None):
    """Read .dip file from Neuromag/xfit or MNE

    Parameters
    ----------
    fname : str
        The name of the .dip file.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    time : array, shape (n_dipoles,)
        The time instants at which each dipole was fitted.
    pos : array, shape (n_dipoles, 3)
        The dipoles positions in meters
    amplitude : array, shape (n_dipoles,)
        The amplitude of the dipoles in nAm
    ori : array, shape (n_dipoles, 3)
        The dipolar moments. Amplitude of the moment is in nAm.
    gof : array, shape (n_dipoles,)
        The goodness of fit
    """
    dipole = read_dipole(fname)
    return (dipole['times'], dipole['pos'], dipole['amplitude'],
            dipole['ori'], dipole['gof'])


@verbose
def read_dipole(fname, verbose=None):
    """Read .dip file from Neuromag/xfit or MNE

    Parameters
    ----------
    fname : str
        The name of the .dip file.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    time : array, shape (n_dipoles,)
        The time instants at which each dipole was fitted.
    pos : array, shape (n_dipoles, 3)
        The dipoles positions in meters
    amplitude : array, shape (n_dipoles,)
        The amplitude of the dipoles in nAm
    ori : array, shape (n_dipoles, 3)
        The dipolar moments. Amplitude of the moment is in nAm.
    gof : array, shape (n_dipoles,)
        The goodness of fit
    """
    try:
        data = np.loadtxt(fname, comments='%')
    except:
        data = np.loadtxt(fname, comments='#')  # handle 2 types of comments...
    name = None
    with open(fname, 'r') as fid:
        for line in fid.readlines():
            if line.startswith('##') or line.startswith('%%'):
                m = re.search('Name "(.*) dipoles"', line)
                if m:
                    name = m.group(1)
                    break
    if data.ndim == 1:
        data = data[None, :]
    logger.info("%d dipole(s) found" % len(data))
    times = data[:, 0]
    pos = 1e-3 * data[:, 2:5]  # put data in meters
    amplitude = data[:, 5]
    ori = data[:, 6:9]
    gof = data[:, 9]
    dipole = Dipole(times=times, pos=pos, amplitude=amplitude, ori=ori,
                    gof=gof, name=name)
    return dipole


# #############################################################################
# Fitting

def _dipole_forwards(fit_data, rr, n_jobs=1):
    """Compute the forward solution and do other nice stuff"""
    B = _compute_forwards_meeg(rr, fit_data['fwd_data'], n_jobs, verbose=False)
    B = np.concatenate(B, axis=1)

    # Apply projection and whiten
    _whiten_data(B.T, fit_data['info_nobads'], fit_data['cov'], verbose=False)

    # column normalization
    S = np.sum(B ** 2, axis=1)  # across channels
    scales = np.repeat(np.sqrt(np.sum(np.reshape(S, (len(rr), 3)), axis=1)) /
                       3., 3)
    B *= scales[:, np.newaxis]
    return B, scales


def _make_volume_source_space(surf, grid, exclude, mindist, n_jobs):
    """Make a source space which covers the volume bounded by surf"""
    # Figure out the grid size
    cm = np.mean(surf['rr'], axis=0)
    min_ = surf['rr'].min(axis=0)
    max_ = surf['rr'].max(axis=0)

    # Define the sphere which fits the surface
    maxdist = surf['rr'] - cm[np.newaxis, :]
    maxdist = np.sqrt(np.sum(maxdist * maxdist, axis=1).max())

    logger.info('\tSurface CM = (%6.1f %6.1f %6.1f) mm'
                % (1000 * cm[0], 1000 * cm[1], 1000 * cm[2]))
    logger.info('\tSurface fits inside a sphere with radius %6.1f mm'
                % (1000 * maxdist))
    logger.info('\tSurface extent:\n'
                '\t\tx = %6.1f ... %6.1f mm\n'
                '\t\ty = %6.1f ... %6.1f mm\n'
                '\t\tz = %6.1f ... %6.1f mm'
                % (1000 * min_[0], 1000 * max_[0],
                   1000 * min_[1], 1000 * max_[1],
                   1000 * min_[2], 1000 * max_[2]))
    maxn = np.array([np.floor(np.abs(m) / grid) + 1 if m > 0 else -
                     np.floor(np.abs(m) / grid) - 1 for m in max_], int)
    minn = np.array([np.floor(np.abs(m) / grid) + 1 if m > 0 else -
                     np.floor(np.abs(m) / grid) - 1 for m in min_], int)
    logger.info('\tGrid extent:\n'
                '\t\tx = %6.1f ... %6.1f mm\n'
                '\t\ty = %6.1f ... %6.1f mm\n'
                '\t\tz = %6.1f ... %6.1f mm\n'
                % (1000 * (minn[0] * grid), 1000 * (maxn[0] * grid),
                   1000 * (minn[1] * grid), 1000 * (maxn[1] * grid),
                   1000 * (minn[2] * grid), 1000 * (maxn[2] * grid)))
    # Now make the initial grid
    npts = np.prod(maxn - minn + 1)
    sp = dict(rr=np.zeros((npts, 3)), nn=np.zeros((npts, 3)), nuse=npts,
              inuse=np.ones(npts, bool), coord_frame=FIFF.FIFFV_COORD_MRI)
    sp['nn'][:, 2] = 1.0
    # x varies fastest, then y, then z (can use unravel to do this)
    rr = np.meshgrid(np.arange(minn[2], maxn[2] + 1) * grid,
                     np.arange(minn[1], maxn[1] + 1) * grid,
                     np.arange(minn[0], maxn[0] + 1) * grid, indexing='ij')
    sp['rr'] = np.array([rr[2].ravel(), rr[1].ravel(), rr[0].ravel()]).T
    assert sp['rr'].shape[0] == npts
    logger.info('\t%d sources before omitting any.' % sp['nuse'])

    # Exclude infeasible points
    bad = np.sqrt(np.sum((sp['rr'] - cm) ** 2, axis=1))
    bad = np.logical_or(bad > maxdist, bad < exclude)
    sp['inuse'][bad] = False
    sp['nuse'] -= np.sum(bad)
    sp['vertno'] = np.where(sp['inuse'])[0]
    logger.info('\t%d sources after omitting infeasible sources.' % sp['nuse'])

    _filter_source_spaces(surf, 1000 * mindist, None, [sp], n_jobs=n_jobs)
    logger.info('\t%d sources remaining after excluding the sources '
                'outside the surface and less than %6.1f mm inside.'
                % (sp['nuse'], 1000 * mindist))
    return sp


def _make_guesses(surf, radius, r0, grid, exclude, mindist, n_jobs):
    """Make a guess space inside a sphere or BEM surface"""
    if surf is None:
        logger.info('Making a spherical guess space with radius %7.1f mm...'
                    % (1000 * radius))
        surf = _get_ico_surface(3)
        _normalize_vectors(surf['rr'])
        surf['rr'] *= radius
        surf['rr'] += r0
    else:
        logger.info('Guess surface (%s) is in %s coordinates'
                    % (_bem_explain_surface(surf['id']),
                       _coord_frame_name(surf['coord_frame'])))
    logger.info('Filtering (grid = %6.f mm)...' % (1000 * grid))
    return _make_volume_source_space(surf, grid, exclude, mindist, n_jobs)


def _fit_eval(rd, B, B2, fwd=None, fit_data=None):
    """Calculate the residual sum of squares"""
    if fwd is None:
        fwd = _dipole_forwards(fit_data, rd[np.newaxis, :])[0]
    uu, sing, vv = linalg.svd(fwd.T, full_matrices=False)
    ncomp = 3 if sing[2] / sing[0] > 0.2 else 2
    Bm2 = np.sum(np.dot(uu.T[:ncomp], B) ** 2)
    return B2 - Bm2


def _find_best_guess(B, B2, rrs, guess_fwd):
    """Find a good starting point"""
    goodnesses = [1.0 - _fit_eval(rr, B, B2, fwd) / B2
                  for rr, fwd in zip(rrs,
                                     np.array_split(guess_fwd, len(rrs)))]
    return rrs[np.argmax(goodnesses)]


def _fit_Q(fit, B, B2, rd):
    """Fit the dipole moment once the location is known"""
    fwd, scales = _dipole_forwards(fit, rd[np.newaxis, :])
    uu, sing, vv = linalg.svd(fwd.T, full_matrices=False)
    ncomp = 3 if sing[2] / sing[0] > 0.2 else 2
    one = np.dot(uu.T[:ncomp], B)
    # Counteract the effect of column normalization
    Q = scales[0] * np.sum(vv[:ncomp] * (one / sing[:ncomp][:, np.newaxis]),
                           axis=0)
    Bm2 = np.sum(one ** 2)
    residual = B2 - Bm2
    return Q, residual


def _fit_dipoles(data, rrs, guess_fwd, fit_data, n_jobs):
    """Fit a single dipole to the given whitened, projected data"""
    parallel, p_fun, _ = parallel_func(_fit_dipole, n_jobs)
    # parallel over time points
    res = parallel(p_fun(B, rrs, guess_fwd, fit_data)
                   for B in data.T)
    pos = np.array([r[0] for r in res])
    amp = np.array([r[1] for r in res])
    ori = np.array([r[2] for r in res])
    gof = np.array([r[3] for r in res])
    return pos, amp, ori, gof


def _fit_dipole(B, rrs, guess_fwd, fit_data):
    """Fit a single bit of data"""
    B2 = np.dot(B, B)
    rd_guess = _find_best_guess(B, B2, rrs, guess_fwd)
    fun = partial(_fit_eval, B=B, B2=B2, fit_data=fit_data)
    rd_final = optimize.minimize(fun, rd_guess).x
    # Compute the dipole moment at the final point
    Q, residual = _fit_Q(fit_data, B, B2, rd_final)
    gof = 1.0 - residual / B2
    amp = np.sqrt(np.sum(Q * Q))
    ori = Q / amp
    raise RuntimeError
    return rd_final, amp, ori, gof


@verbose
def fit_dipole(evoked, cov, bem, mri=None, n_jobs=1, verbose=None):
    """Fit a dipole

    Parameters
    ----------
    evoked : instance of Evoked
        The dataset to fit.
    cov : str | instance of Covariance
        The noise covariance.
    bem : str | dict
        The BEM filename (str) or a loaded sphere model (dict).
    mri : str
        The head<->MRI transform filename. Must be provided if BEM is
        not a sphere model.
    n_jobs : int
        Number of jobs to run in parallel (used in field computation
        and fitting).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    dip : dict
        The dipole fits.
    """
    info = evoked.info
    neeg = len(pick_types(info, meg=False, eeg=True, exclude=[]))
    if isinstance(bem, string_types):
        logger.info('BEM              : %s' % bem)
    if mri is not None:
        logger.info('MRI transform    : %s' % mri)
        mri_head_t, mri = _get_mri_head_t(mri)
    else:
        mri_head_t = {'from': FIFF.FIFFV_COORD_HEAD,
                      'to': FIFF.FIFFV_COORD_MRI, 'trans': np.eye(4)}
    bem = _setup_bem(bem, bem, neeg, mri_head_t)
    if not bem['is_sphere']:
        if mri is None:
            raise ValueError('mri must not be None if BEM is provided')
        # Find the best-fitting sphere
        inner_skull = _bem_find_surface(bem, 'inner_skull')
        R, r0 = _fit_sphere(inner_skull['rr'], disp=False)
        assert mri_head_t['to'] == FIFF.FIFFV_COORD_HEAD
        r0 = apply_trans(mri_head_t['trans'], r0[np.newaxis, :])[0]
        logger.info('Fitted sphere model origin : '
                    '%6.1f %6.1f %6.1f mm rad = %6.1f mm.'
                    % (1000 * r0[0], 1000 * r0[1], 1000 * r0[2], 1000 * R))
        sphere_model = make_sphere_model(r0, R)
    else:
        logger.info('Sphere model     : origin at (% 7.2f % 7.2f % 7.2f) mm'
                    % (1000 * r0[0], 1000 * r0[1], 1000 * r0[2]))
        inner_skull = None
        sphere_model = bem

    # Eventually these could be parameters, but they are just used for
    # the initial grid anyway
    guess_grid = 0.01
    guess_mindist = 0.01
    guess_exclude = 0.02
    accurate = False  # can be made an option later (shouldn't make big diff)

    logger.info('Guess grid       : %6.1f mm' % (1000 * guess_grid,))
    if guess_mindist > 0.0:
        logger.info('Guess mindist    : %6.1f mm' % (1000 * guess_mindist,))
    if guess_exclude > 0:
        logger.info('Guess exclude    : %6.1f mm' % (1000 * guess_exclude,))
    logger.info('Using %s MEG coil definitions.'
                % ("accurate" if accurate else "standard"))
    if isinstance(cov, string_types):
        logger.info('Noise covariance : %s' % (cov,))
        cov = read_cov(cov, verbose=False)
    logger.info('')

    _print_coord_trans(mri_head_t)
    _print_coord_trans(info['dev_head_t'])
    logger.info('%d bad channels total' % len(info['bads']))

    # Forward model setup (setup_forward_model from setup.c)
    megcoils, compcoils, eegels, megnames, eegnames, meg_info = \
        _prep_channels(info, exclude='bads', accurate=accurate)
    from mne import pick_info
    info_nobads = pick_info(info, pick_types(info, meg=True, eeg=True,
                                             exclude='bads'))

    # whiten data
    evoked = whiten_evoked(evoked, cov, verbose=False)

    # Proceed to computing the fits (make_guess_data)
    logger.info('\n---- Computing the forward solution for the guesses...')
    r0_mri = apply_trans(invert_transform(mri_head_t)['trans'], r0)
    src = _make_guesses(inner_skull, 0.08, r0_mri, guess_grid, guess_exclude,
                        guess_mindist, n_jobs=n_jobs)
    transform_surface_to(src, 'head', mri_head_t)
    logger.info('Guess locations are now in head coordinates.\n')

    logger.info('Go through all guess source locations...')
    # Compute the guesses using the sphere model for speed
    fwd_data = dict(coils_list=[megcoils, eegels], infos=[meg_info, None],
                    ccoils_list=[compcoils, None], coil_types=['meg', 'eeg'])
    fit_data = dict(cov=cov, info_nobads=info_nobads, fwd_data=fwd_data)
    _prep_field_computation(src['rr'], sphere_model, fit_data['fwd_data'],
                            n_jobs, verbose=False)
    guess_fwd = _dipole_forwards(fit_data, src['rr'], n_jobs=n_jobs)[0]
    logger.info('[done %d sources]' % src['nuse'])

    # Do actual fits using the real BEM (if available)
    tstep = 1. / evoked.info['sfreq']
    logger.info('---- Fitting : %7.1f ... %7.1f ms '
                '(step: %6.1f ms)'
                % (1000 * evoked.times[0], 1000 * evoked.times[-1],
                   1000 * tstep))
    _prep_field_computation(src['rr'], bem, fwd_data,
                            n_jobs, verbose=False)
    picks = pick_types(evoked.info, meg=True, eeg=True, exclude='bads')
    out = _fit_dipoles(evoked.data[picks], src['rr'], guess_fwd, fit_data,
                       n_jobs)
    dipoles = dict(time=np.array(evoked.times), name=evoked.comment,
                   pos=out[0], amplitude=out[1], ori=out[2], gof=out[3])
    logger.info('%d dipoles fitted' % len(dipoles['time']))
    return dipoles
