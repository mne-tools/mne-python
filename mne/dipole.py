# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import numpy as np
from scipy import optimize, linalg
import re

from .cov import read_cov, _get_whitener_data
from .io.pick import pick_types
from .io.constants import FIFF
from .bem import _fit_sphere
from .transforms import (_print_coord_trans, _coord_frame_name,
                         apply_trans, invert_transform)

from .forward._make_forward import (_get_mri_head_t, _setup_bem,
                                    _prep_channels)
from .forward._compute_forward import (_compute_forwards_meeg,
                                       _prep_field_computation)

from .externals.six import string_types
from .surface import (_bem_find_surface, transform_surface_to,
                      _normalize_vectors, _get_ico_surface,
                      _bem_explain_surface, _compute_nearest)
from .source_space import (_filter_source_spaces, SourceSpaces,
                           _points_outside_surface)
from .parallel import parallel_func
from .fixes import partial
from .utils import logger, verbose, deprecated


class Dipole(object):
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
        self.times = times
        self.pos = pos
        self.amplitude = amplitude
        self.ori = ori
        self.gof = gof
        self.name = name

    def __repr__(self):
        s = "n_times : %s" % len(self.times)
        s += ", tmin : %s" % np.min(self.times)
        s += ", tmax : %s" % np.max(self.times)
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
            t = self.times[:, np.newaxis] * 1000.
            gof = self.gof[:, np.newaxis]
            amp = 1e9 * self.amplitude[:, np.newaxis]
            out = np.concatenate((t, t, self.pos / 1e-3, amp,
                                  self.ori * amp, gof), axis=-1)
            np.savetxt(fid, out, fmt=fmt)
            if self.name is not None:
                fid.write(('## Name "%s dipoles" Style "Dipoles"'
                           % self.name).encode('utf-8'))

    def crop(self, tmin=None, tmax=None):
        """Crop data to a given time interval
        """
        times = self.times
        mask = np.ones(len(times), dtype=np.bool)
        if tmin is not None:
            mask = mask & (times >= tmin)
        if tmax is not None:
            mask = mask & (times <= tmax)
        for attr in ('times', 'pos', 'gof', 'amplitude', 'ori', 'gof'):
            setattr(self, attr, getattr(self, attr)[mask])


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
    return (dipole.times * 1000., dipole.pos, dipole.amplitude,
            1e9 * dipole.ori * dipole.amplitude[:, np.newaxis], dipole.gof)


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
        The time instants at which each dipole was fitted (in sec).
    pos : array, shape (n_dipoles, 3)
        The dipoles positions in meters
    amplitude : array, shape (n_dipoles,)
        The amplitude of the dipoles in nAm
    ori : array, shape (n_dipoles, 3)
        The dipolar moments. Amplitude of the moment is in nAm.
    gof : array, shape (n_dipoles,)
        The goodness of fit (in percent).
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
    times = data[:, 0] / 1000.
    pos = 1e-3 * data[:, 2:5]  # put data in meters
    amplitude = data[:, 5]
    norm = amplitude.copy()
    amplitude /= 1e9
    norm[norm == 0] = 1
    ori = data[:, 6:9] / norm[:, np.newaxis]
    gof = data[:, 9]
    dipole = Dipole(times, pos, amplitude, ori, gof, name)
    return dipole


# #############################################################################
# Fitting

def _dipole_forwards(fwd_data, whitener, rr, n_jobs=1):
    """Compute the forward solution and do other nice stuff"""
    B = _compute_forwards_meeg(rr, fwd_data, n_jobs, verbose=False)
    B = np.concatenate(B, axis=1)

    # Apply projection and whiten (cov has projections already)
    B = np.dot(B, whitener.T)

    # column normalization
    S = np.sum(B * B, axis=1)  # across channels
    scales = np.repeat(3. / np.sqrt(np.sum(np.reshape(S, (len(rr), 3)),
                                           axis=1)), 3)
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
    sp['rr'] = sp['rr'][sp['inuse']]
    sp['nn'] = sp['nn'][sp['inuse']]
    sp['nuse'] = len(sp['rr'])
    sp['vertno'] = np.arange(sp['nuse'])
    sp['inuse'] = np.ones(sp['nuse'], bool)
    sp.update({'type': 'discrete', 'id': -1, 'np': sp['nuse'], 'ntri': 0,
               'use_tris': None, 'nearest': None, 'dist': None})
    return sp


def _make_guesses(surf_or_rad, r0, grid, exclude, mindist, n_jobs):
    """Make a guess space inside a sphere or BEM surface"""
    if isinstance(surf_or_rad, dict):
        surf = surf_or_rad
        logger.info('Guess surface (%s) is in %s coordinates'
                    % (_bem_explain_surface(surf['id']),
                       _coord_frame_name(surf['coord_frame'])))
    else:
        radius = surf_or_rad[0]
        logger.info('Making a spherical guess space with radius %7.1f mm...'
                    % (1000 * radius))
        surf = _get_ico_surface(3)
        _normalize_vectors(surf['rr'])
        surf['rr'] *= radius
        surf['rr'] += r0
    logger.info('Filtering (grid = %6.f mm)...' % (1000 * grid))
    src = _make_volume_source_space(surf, grid, exclude, mindist, n_jobs)
    return SourceSpaces([src])


def _fit_eval(rd, B, B2, fwd_svd=None, fwd_data=None, whitener=None,
              constraint=None):
    """Calculate the residual sum of squares"""
    if fwd_svd is None:
        dist = constraint(rd)
        if dist <= 0:
            return 1. - 100 * dist
        r1s = rd[np.newaxis, :]
        fwd = _dipole_forwards(fwd_data, whitener, r1s)[0]
        uu, sing, vv = linalg.svd(fwd, full_matrices=False)
    else:
        uu, sing, vv = fwd_svd
    ncomp = 3 if sing[2] / sing[0] > 0.2 else 2
    Bm2 = np.sum(np.dot(vv[:ncomp], B) ** 2)
    return 1. - Bm2 / B2


def _fit_Q(fwd_data, whitener, B, B2, rd):
    """Fit the dipole moment once the location is known"""
    fwd, scales = _dipole_forwards(fwd_data, whitener, rd[np.newaxis, :])
    uu, sing, vv = linalg.svd(fwd, full_matrices=False)
    ncomp = 3 if sing[2] / sing[0] > 0.2 else 2
    one = np.dot(vv[:ncomp], B)
    # Counteract the effect of column normalization
    Q = scales[0] * np.sum(uu.T[:ncomp] * (one / sing[:ncomp])[:, np.newaxis],
                           axis=0)
    Bm2 = np.sum(one ** 2)
    residual = B2 - Bm2
    gof = 1. - residual / B2
    return Q, gof


def _fit_dipoles(data, times, rrs, guess_fwd_svd, fwd_data, whitener, n_jobs):
    """Fit a single dipole to the given whitened, projected data"""
    parallel, p_fun, _ = parallel_func(_fit_dipole, n_jobs)
    # parallel over time points
    res = parallel(p_fun(B, t, rrs, guess_fwd_svd, fwd_data, whitener)
                   for B, t in zip(data.T, times))
    pos = np.array([r[0] for r in res])
    amp = np.array([r[1] for r in res])
    ori = np.array([r[2] for r in res])
    gof = np.array([r[3] for r in res]) * 100  # convert to percentage
    return pos, amp, ori, gof


def _fit_dipole(B, t, rrs, guess_fwd_svd, fwd_data, whitener):
    """Fit a single bit of data"""
    logger.info('---- Fitting : %7.1f ms' % (1000 * t,))

    # make constraint function to keep the solver within the inner skull
    if isinstance(fwd_data['inner_skull'], dict):  # bem
        surf = fwd_data['inner_skull']

        def constraint(rd):
            if _points_outside_surface(rd[np.newaxis, :], surf, 1)[0]:
                dist = _compute_nearest(surf['rr'], rd[np.newaxis, :],
                                        return_dists=True)[1][0]
                return -dist
            else:
                return 1.
    else:  # sphere
        R, r0 = fwd_data['inner_skull']
        R_adj = R - 1e-5  # to be sure we don't hit the innermost surf

        def constraint(rd):
            return R_adj - np.sqrt(np.sum((rd - r0) ** 2))

    # Find a good starting point (find_best_guess in C)
    B2 = np.dot(B, B)
    if B2 == 0:
        logger.warning('Zero field found for time %s' % t)
        return np.zeros(3), 0, np.zeros(3), 0
    x0 = rrs[np.argmin([_fit_eval(rrs[fi][np.newaxis, :], B, B2, fwd_svd)
                        for fi, fwd_svd in enumerate(guess_fwd_svd)])]
    fun = partial(_fit_eval, B=B, B2=B2, fwd_data=fwd_data, whitener=whitener,
                  constraint=constraint)

    # These were tested on a grid of 26 known locations
    # Params                    corr, dist, time
    # c = clean (100 snr), n = noisy (20 snr)

    # Simplex (fmin)
    # c xtol=1e-4, ftol=1e-4:     0.999, 3.1, 49 sec
    # c xtol=1e-3, ftol=1e-3:     0.999, 3.4, 28 sec
    # c xtol=5e-3, ftol=5e-3:     0.998, 3.8, 20 sec
    # c xtol=1e-3, ftol=1e-2:     0.997, 5.3, 25 sec
    # n xtol=1e-3, ftol=1e-3:     0.994, 8.4, 30 sec
    # n xtol=5e-3, ftol=5e-3:     0.993, 9.3, 15 sec
    # rd_final = optimize.fmin(fun, x0, xtol=5e-3, ftol=5e-3, disp=False)

    # BFGS
    # c gtol=1e-3, epsilon=1e-4:  0.997, 5.8, 101 sec
    # c rd_final = optimize.fmin_bfgs(fun, x0, gtol=1e-3, epsilon=1e-4)

    # CG
    # c gtol=1e-3, epsilon=1e-4:  0.997, 5.8, 140 sec
    # c rd_final = optimize.fmin_cg(fun, x0, gtol=1e-3, epsilon=1e-4)

    # COBYLA
    # c rhobeg=1e0,  rhoend=1e-4: 0.999, 3.1, 31 sec
    # c rhobeg=1e0,  rhoend=1e-3: 0.999, 3.3, 21 sec
    # c rhobeg=1e-2, rhoend=1e-3: 0.999, 3.3, 19 sec
    # c rhobeg=1e-2, rhoend=5e-4: 0.999, 3.3, 23 sec
    # c rhobeg=1e-2, rhoend=1e-4: 0.999, 3.1, 37 sec
    # c rhobeg=5e-3, rhoend=1e-3: 0.998, 4.0, 18 sec
    # c rhobeg=2e-2, rhoend=1e-3: 0.998, 4.1, 21 sec
    # n rhobeg=1e-2, rhoend=1e-3: 0.994, 8.1, 19 sec
    # n rhobeg=1e-2, rhoend=1e-4: 0.994, 8.0, 29 sec
    # n rhobeg=1e-2, rhoend=5e-4: 0.995, 8.0, 22 sec
    rd_final = optimize.fmin_cobyla(fun, x0, (constraint,), consargs=(),
                                    rhobeg=5e-2, rhoend=1e-4, disp=False)

    # L-BFGS-B
    # c epsilon=1e-4, factr=1e12: 0.998, 4.6, 46 sec
    # rd_final = optimize.fmin_l_bfgs_b(fun, x0, approx_grad=True,
    #                                   epsilon=1e-4, factr=1e12)[0]

    # Powell (too many function evals)
    # c xtol=1e-4, ftol=1e-4:     0.999, 3.2, 166 sec
    # rd_final = optimize.fmin_powell(fun, x0, xtol=1e-4, ftol=1e-4)

    # SLSQP
    # c acc=1e-5, epsilon=1e-6:   0.999, 3.4, 44 sec
    # c acc=1e-4, epsilon=1e-6:   0.999, 3.5, 26 sec
    # c           epsilon=1e-4:   0.999, 3.5, 26 sec
    # c acc=1e-3, epsilon=1e-6:   0.997, 5.0, 19 sec
    # c           epsilon=1e-4:   0.998, 4.0, 20 sec
    # n acc=1e-4, epsilon=1e-4:   0.994, 8.2, 26 sec
    # rd_final = optimize.fmin_slsqp(fun, x0, acc=1e-4, epsilon=1e-4,
    #                                disp=False)

    # TNC
    # c epsilon=1e-4,ftol=1e-4,xtol=1e-4: 0.998, 4.62, 107 sec
    # rd_final = optimize.fmin_tnc(fun, x0, epsilon=1e-4, ftol=1e-4, xtol=1e-4,
    #                              approx_grad=True)[0]

    # Compute the dipole moment at the final point
    Q, gof = _fit_Q(fwd_data, whitener, B, B2, rd_final)
    amp = np.sqrt(np.sum(Q * Q))
    norm = 1 if amp == 0 else amp
    ori = Q / norm
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
    mri : str | None
        The head<->MRI transform filename. Must be provided unless BEM
        is a sphere model.
    n_jobs : int
        Number of jobs to run in parallel (used in field computation
        and fitting).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    dip : instance of Dipole
        The dipole fits.
    """
    neeg = len(pick_types(evoked.info, meg=False, eeg=True, exclude=[]))
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
        inner_skull = inner_skull.copy()
        R, r0 = _fit_sphere(inner_skull['rr'], disp=False)
        r0 = apply_trans(mri_head_t['trans'], r0[np.newaxis, :])[0]
        logger.info('Grid origin      : '
                    '%6.1f %6.1f %6.1f mm rad = %6.1f mm.'
                    % (1000 * r0[0], 1000 * r0[1], 1000 * r0[2], 1000 * R))
    else:
        r0 = bem['r0']
        logger.info('Sphere model     : origin at (% 7.2f % 7.2f % 7.2f) mm'
                    % (1000 * r0[0], 1000 * r0[1], 1000 * r0[2]))
        if 'layers' in bem:
            R = bem['layers'][0]['rad']
        else:
            R = np.inf
        inner_skull = [R, r0]
    r0_mri = apply_trans(invert_transform(mri_head_t)['trans'],
                         r0[np.newaxis, :])[0]

    # Eventually these could be parameters, but they are just used for
    # the initial grid anyway
    guess_grid = 0.02
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
    _print_coord_trans(evoked.info['dev_head_t'])
    logger.info('%d bad channels total' % len(evoked.info['bads']))

    # Forward model setup (setup_forward_model from setup.c)
    megcoils, compcoils, eegels, megnames, eegnames, meg_info = \
        _prep_channels(evoked.info, exclude='bads', accurate=accurate)

    # Whitener for the data
    logger.info('Decomposing the sensor noise covariance matrix...')
    picks = pick_types(evoked.info, meg=True, eeg=True, exclude='bads')

    from .io.pick import pick_info
    from .cov import prepare_noise_cov
    info_nb = pick_info(evoked.info, picks)
    cov = prepare_noise_cov(cov, info_nb, info_nb['ch_names'], verbose=False)
    nzero = (cov['eig'] > 0)
    n_chan = len(info_nb['ch_names'])
    whitener = np.zeros((n_chan, n_chan), dtype=np.float)
    whitener[nzero, nzero] = 1.0 / np.sqrt(cov['eig'][nzero])
    whitener = np.dot(whitener, cov['eigvec'])
    del cov, info_nb

    # whitener = _get_whitener_data(evoked.info, cov, picks, verbose=False)

    # Proceed to computing the fits (make_guess_data)
    logger.info('\n---- Computing the forward solution for the guesses...')
    src = _make_guesses(inner_skull, r0_mri,
                        guess_grid, guess_exclude, guess_mindist,
                        n_jobs=n_jobs)[0]
    if isinstance(inner_skull, dict):
        transform_surface_to(inner_skull, 'head', mri_head_t)
    transform_surface_to(src, 'head', mri_head_t)

    # C code computes guesses using a sphere model for speed, don't bother here
    logger.info('Go through all guess source locations...')
    fwd_data = dict(coils_list=[megcoils, eegels], infos=[meg_info, None],
                    ccoils_list=[compcoils, None], coil_types=['meg', 'eeg'],
                    inner_skull=inner_skull)
    _prep_field_computation(src['rr'], bem, fwd_data, n_jobs, verbose=False)
    guess_fwd = _dipole_forwards(fwd_data, whitener, src['rr'],
                                 n_jobs=n_jobs)[0]
    # decompose ahead of time
    guess_fwd_svd = [linalg.svd(fwd, full_matrices=False)
                     for fwd in np.array_split(guess_fwd, len(src['rr']))]
    logger.info('[done %d sources]' % src['nuse'])

    # Do actual fits
    picks = pick_types(evoked.info, meg=True, eeg=True, exclude='bads')
    data = np.dot(whitener, evoked.data[picks])
    out = _fit_dipoles(data, evoked.times, src['rr'], guess_fwd_svd, fwd_data,
                       whitener, n_jobs)
    dipoles = Dipole(evoked.times.copy(), out[0], out[1], out[2], out[3],
                     evoked.comment)
    logger.info('%d dipoles fitted' % len(dipoles.times))
    return dipoles
