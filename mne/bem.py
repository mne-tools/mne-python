# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import sys
import os
import os.path as op
import shutil
import numpy as np
from scipy import linalg

from .fixes import partial
from .utils import (verbose, logger, run_subprocess, get_config,
                    get_subjects_dir)
from .io.constants import FIFF
from .externals.six import string_types


# ############################################################################
# Compute EEG sphere model

def _fwd_eeg_get_multi_sphere_model_coeffs(m, n_terms):
    """Get the model depended weighting factor for n"""
    nlayer = len(m['layers'])
    if nlayer in (0, 1):
        return 1.

    # Initialize the arrays
    c1 = np.zeros(nlayer - 1)
    c2 = np.zeros(nlayer - 1)
    cr = np.zeros(nlayer - 1)
    cr_mult = np.zeros(nlayer - 1)
    for k in range(nlayer - 1):
        c1[k] = m['layers'][k]['sigma'] / m['layers'][k + 1]['sigma']
        c2[k] = c1[k] - 1.0
        cr_mult[k] = m['layers'][k]['rel_rad']
        cr[k] = cr_mult[k]
        cr_mult[k] *= cr_mult[k]

    coeffs = np.zeros(n_terms - 1)
    for n in range(1, n_terms):
        # Increment the radius coefficients
        for k in range(nlayer - 1):
            cr[k] *= cr_mult[k]

        # Multiply the matrices
        M = np.eye(2)
        n1 = n + 1.0
        for k in range(nlayer - 2, -1, -1):
            M = np.dot([[n + n1 * c1[k], n1 * c2[k] / cr[k]],
                        [n * c2[k] * cr[k], n1 + n * c1[k]]], M)
        num = n * (2.0 * n + 1.0) ** (nlayer - 1)
        coeffs[n - 1] = num / (n * M[1, 1] + n1 * M[1, 0])
    return coeffs


def _compose_linear_fitting_data(mu, u):
    # y is the data to be fitted (nterms-1 x 1)
    # M is the model matrix      (nterms-1 x nfit-1)
    for k in range(u['nterms'] - 1):
        k1 = k + 1
        mu1n = np.power(mu[0], k1)
        u['y'][k] = u['w'][k] * (u['fn'][k1] - mu1n * u['fn'][0])
        for p in range(u['nfit'] - 1):
            u['M'][k][p] = u['w'][k] * (np.power(mu[p + 1], k1) - mu1n)


def _compute_linear_parameters(mu, u):
    """Compute the best-fitting linear parameters"""
    _compose_linear_fitting_data(mu, u)
    uu, sing, vv = linalg.svd(u['M'], full_matrices=False)

    # Compute the residuals
    u['resi'] = u['y'].copy()

    vec = np.empty(u['nfit'] - 1)
    for p in range(u['nfit'] - 1):
        vec[p] = np.dot(uu[:, p], u['y'])
        for k in range(u['nterms'] - 1):
            u['resi'][k] -= uu[k, p] * vec[p]
        vec[p] = vec[p] / sing[p]

    lambda_ = np.zeros(u['nfit'])
    for p in range(u['nfit'] - 1):
        sum_ = 0.
        for q in range(u['nfit'] - 1):
            sum_ += vv[q, p] * vec[q]
        lambda_[p + 1] = sum_
    lambda_[0] = u['fn'][0] - np.sum(lambda_[1:])
    rv = np.dot(u['resi'], u['resi']) / np.dot(u['y'], u['y'])
    return rv, lambda_


def _one_step(mu, u):
    """Evaluate the residual sum of squares fit for one set of mu values"""
    if np.abs(mu).max() > 1.0:
        return 1.0

    # Compose the data for the linear fitting, compute SVD, then residuals
    _compose_linear_fitting_data(mu, u)
    u['uu'], u['sing'], u['vv'] = linalg.svd(u['M'])
    u['resi'][:] = u['y'][:]
    for p in range(u['nfit'] - 1):
        dot = np.dot(u['uu'][p], u['y'])
        for k in range(u['nterms'] - 1):
            u['resi'][k] = u['resi'][k] - u['uu'][p, k] * dot

    # Return their sum of squares
    return np.dot(u['resi'], u['resi'])


def _fwd_eeg_fit_berg_scherg(m, nterms, nfit):
    """Fit the Berg-Scherg equivalent spherical model dipole parameters"""
    from scipy.optimize import minimize
    assert nfit >= 2
    u = dict(y=np.zeros(nterms - 1), resi=np.zeros(nterms - 1),
             nfit=nfit, nterms=nterms, M=np.zeros((nterms - 1, nfit - 1)))

    # (1) Calculate the coefficients of the true expansion
    u['fn'] = _fwd_eeg_get_multi_sphere_model_coeffs(m, nterms + 1)

    # (2) Calculate the weighting
    f = (min([layer['rad'] for layer in m['layers']]) /
         max([layer['rad'] for layer in m['layers']]))

    # correct weighting
    k = np.arange(1, nterms + 1)
    u['w'] = np.sqrt((2.0 * k + 1) * (3.0 * k + 1.0) /
                     k) * np.power(f, (k - 1.0))
    u['w'][-1] = 0

    # Do the nonlinear minimization, constraining mu to the interval [-1, +1]
    mu_0 = np.random.RandomState(0).rand(nfit) * f
    fun = partial(_one_step, u=u)
    cons = []
    for ii in range(nfit):
        for val in [1., -1.]:
            cons.append({'type': 'ineq',
                         'fun': lambda x: np.array([val * x[ii] + 1.]),
                         'jac': lambda x: np.array([0.] * ii + [val] +
                                                   [0.] * (nfit - ii - 1))})
    mu = minimize(fun, mu_0, constraints=cons, method='COBYLA', tol=1e-2).x

    # (6) Do the final step: calculation of the linear parameters
    rv, lambda_ = _compute_linear_parameters(mu, u)
    order = np.argsort(mu)[::-1]
    mu, lambda_ = mu[order], lambda_[order]  # sort: largest mu first

    m['mu'] = mu
    # This division takes into account the actual conductivities
    m['lambda'] = lambda_ / m['layers'][-1]['sigma']
    m['nfit'] = nfit
    return rv


@verbose
def make_sphere_model(r0=(0., 0., 0.04), head_radius=0.09, info=None,
                      relative_radii=(0.90, 0.92, 0.97, 1.0),
                      sigmas=(0.33, 1.0, 0.004, 0.33), verbose=None):
    """Create a spherical model for forward solution calculation

    Parameters
    ----------
    r0 : array-like | str
        Head center to use (in head coordinates). If 'auto', the head
        center will be calculated from the digitization points in info.
    head_radius : float | str | None
        If float, compute spherical shells for EEG using the given radius.
        If 'auto', estimate an approriate radius from the dig points in Info,
        If None, exclude shells.
    info : instance of mne.io.meas_info.Info | None
        Measurement info. Only needed if ``r0`` or ``head_radius`` are
        ``'auto'``.
    relative_radii : array-like
        Relative radii for the spherical shells.
    sigmas : array-like
        Sigma values for the spherical shells.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    sphere : dict
        A spherical BEM.

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    for name in ('r0', 'head_radius'):
        param = locals()[name]
        if isinstance(param, string_types):
            if param != 'auto':
                raise ValueError('%s, if str, must be "auto" not "%s"'
                                 % (name, param))

    if (isinstance(r0, string_types) and r0 == 'auto') or \
       (isinstance(head_radius, string_types) and head_radius == 'auto'):
        if info is None:
            raise ValueError('Info must not be None for auto mode')
        head_radius_fit, r0_fit = fit_sphere_to_headshape(info)[:2]
        if isinstance(r0, string_types):
            r0 = r0_fit / 1000.
        if isinstance(head_radius, string_types):
            head_radius = head_radius_fit / 1000.
    sphere = dict(r0=np.array(r0), is_sphere=True,
                  coord_frame=FIFF.FIFFV_COORD_HEAD)
    sphere['layers'] = []
    if head_radius is not None:
        # Eventually these could be configurable...
        relative_radii = np.array(relative_radii, float)
        sigmas = np.array(sigmas, float)
        order = np.argsort(relative_radii)
        relative_radii = relative_radii[order]
        sigmas = sigmas[order]
        layers = sphere['layers']
        for rel_rad, sig in zip(relative_radii, sigmas):
            # sort layers by (relative) radius, and scale radii
            layer = dict(rad=rel_rad, sigma=sig)
            layer['rel_rad'] = layer['rad'] = rel_rad
            layers.append(layer)

        # scale the radii
        R = layers[-1]['rad']
        rR = layers[-1]['rel_rad']
        for layer in layers:
            layer['rad'] /= R
            layer['rel_rad'] /= rR

        #
        # Setup the EEG sphere model calculations
        #

        # Scale the relative radii
        for k in range(len(relative_radii)):
            layers[k]['rad'] = (head_radius * layers[k]['rel_rad'])
        rv = _fwd_eeg_fit_berg_scherg(sphere, 200, 3)
        logger.info('\nEquiv. model fitting -> RV = %g %%' % (100 * rv))
        for k in range(3):
            logger.info('mu%d = %g\tlambda%d = %g'
                        % (k + 1, sphere['mu'][k], k + 1,
                           layers[-1]['sigma'] * sphere['lambda'][k]))
        logger.info('Set up EEG sphere model with scalp radius %7.1f mm\n'
                    % (1000 * head_radius,))
    return sphere


# #############################################################################
# Helpers

@verbose
def fit_sphere_to_headshape(info, dig_kinds=(FIFF.FIFFV_POINT_EXTRA,),
                            verbose=None):
    """Fit a sphere to the headshape points to determine head center

    Parameters
    ----------
    info : instance of mne.io.meas_info.Info
        Measurement info.
    dig_kinds : tuple of int
        Kind of digitization points to use in the fitting. These can be
        any kind defined in io.constants.FIFF:
            FIFFV_POINT_CARDINAL
            FIFFV_POINT_HPI
            FIFFV_POINT_EEG
            FIFFV_POINT_ECG
            FIFFV_POINT_EXTRA
        Defaults to (FIFFV_POINT_EXTRA,).

    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    radius : float
        Sphere radius in mm.
    origin_head: ndarray, shape (3,)
        Head center in head coordinates (mm).
    origin_device: ndarray, shape (3,)
        Head center in device coordinates (mm).
    """
    # get head digization points of the specified kind
    hsp = [p['r'] for p in info['dig'] if p['kind'] in dig_kinds]

    # exclude some frontal points (nose etc.)
    hsp = [p for p in hsp if not (p[2] < 0 and p[1] > 0)]

    if len(hsp) == 0:
        raise ValueError('No head digitization points of the specified '
                         'kinds (%s) found.' % dig_kinds)

    hsp = 1e3 * np.array(hsp)

    radius, origin_head = _fit_sphere(hsp, disp=False)
    # compute origin in device coordinates
    trans = info['dev_head_t']
    if trans['from'] != FIFF.FIFFV_COORD_DEVICE \
            or trans['to'] != FIFF.FIFFV_COORD_HEAD:
        raise RuntimeError('device to head transform not found')

    head_to_dev = linalg.inv(trans['trans'])
    origin_device = 1e3 * np.dot(head_to_dev,
                                 np.r_[1e-3 * origin_head, 1.0])[:3]

    logger.info('Fitted sphere radius:'.ljust(30) + '%0.1f mm' % radius)
    logger.info('Origin head coordinates:'.ljust(30) +
                '%0.1f %0.1f %0.1f mm' % tuple(origin_head))
    logger.info('Origin device coordinates:'.ljust(30) +
                '%0.1f %0.1f %0.1f mm' % tuple(origin_device))

    return radius, origin_head, origin_device


def _fit_sphere(points, disp='auto'):
    """Aux function to fit points to a sphere"""
    from scipy.optimize import fmin_powell
    if isinstance(disp, string_types) and disp == 'auto':
        disp = True if logger.level <= 20 else False
    # initial guess for center and radius
    xradius = (np.max(points[:, 0]) - np.min(points[:, 0])) / 2.
    yradius = (np.max(points[:, 1]) - np.min(points[:, 1])) / 2.

    radius_init = (xradius + yradius) / 2.
    center_init = np.array([0.0, 0.0, np.max(points[:, 2]) - radius_init])

    # optimization
    x0 = np.r_[center_init, radius_init]

    def cost_fun(x, points):
        return np.sum((np.sqrt(np.sum((points - x[:3]) ** 2, axis=1)) -
                      x[3]) ** 2)

    x_opt = fmin_powell(cost_fun, x0, args=(points,), disp=disp)

    origin = x_opt[:3]
    radius = x_opt[3]
    return radius, origin


# ############################################################################
# Create BEM surfaces

@verbose
def make_watershed_bem(subject=None, subjects_dir=None, overwrite=False,
                       volume='T1', atlas=False, gcaatlas=False, preflood=None,
                       verbose=None):
    """
    Create BEM surfaces using the watershed algorithm included with FreeSurfer

    Parameters
    ----------
    subject : string
        Subject name (SUBJECT environment variable)
    subjects_dir : string
        Directory containing subjects data. If None use
        the Freesurfer SUBJECTS_DIR environment variable.
    overwrite : bool
        Write over existing files
    volume : string
        Defaults to T1
    atlas : bool
        Specify the --atlas option for mri_watershed
    gcaatlas : bool
        Use the subcortical atlas
    preflood : int
        Change the preflood height
    verbose : bool, str or None
        If not None, override default verbose level
    """
    if not os.environ.get('FREESURFER_HOME'):
        raise RuntimeError('FREESURFER_HOME environment variable not set')
    if subject:
        os.environ['SUBJECT'] = subject
    else:
        subject = get_config('SUBJECT', raise_error=True)
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    os.environ['SUBJECTS_DIR'] = subjects_dir

    env = os.environ.copy()

    subject_dir = op.join(subjects_dir, subject)
    mri_dir = op.join(subject_dir, 'mri')
    T1_dir = op.join(mri_dir, volume)
    T1_mgz = op.join(mri_dir, volume + '.mgz')
    bem_dir = op.join(subject_dir, 'bem')
    ws_dir = op.join(subject_dir, 'bem', 'watershed')

    if not op.exists(subject_dir):
        raise RuntimeError('Could not find the MRI data directory "%s"'
                           % subject_dir)
    if not op.exists(bem_dir):
        os.makedirs(bem_dir)
    if (not op.exists(T1_dir) and not op.exists(T1_mgz)):
        raise RuntimeError('Could not find the MRI data')
    if op.exists(ws_dir):
        if not overwrite:
            raise RuntimeError('%s already exists. Use the --overwrite option'
                               'to recreate it.' % ws_dir)
        else:
            shutil.rmtree(ws_dir)
    # put together the command
    cmd = ['mri_watershed']
    if preflood:
        cmd += [preflood]
    if gcaatlas:
        cmd += ['-atlas', '-T1', '-brain_atlas', env['FREESURFER_HOME'] +
                '/average/RB_all_withskull_2007-08-08.gca',
                subject_dir + '/mri/transforms/talairach_with_skull.lta']
    elif atlas:
        cmd += ['-atlas']
    if op.exists(T1_mgz):
        cmd += ['-useSRAS', '-surf', op.join(ws_dir, subject), T1_mgz,
                op.join(ws_dir, 'ws')]
    else:
        cmd += ['-useSRAS', '-surf', op.join(ws_dir, subject), T1_dir,
                op.join(ws_dir, 'ws')]
    # report and run
    logger.info('\nRunning mri_watershed for BEM segmentation with the '
                'following parameters:\n\n'
                'SUBJECTS_DIR = %s\n'
                'SUBJECT = %s\n'
                'Result dir = %s\n' % (subjects_dir, subject, ws_dir))
    os.makedirs(op.join(ws_dir, 'ws'))
    run_subprocess(cmd, env=env, stdout=sys.stdout)
    #
    os.chdir(ws_dir)
    if op.exists(T1_mgz):
        surfaces = [subject + '_brain_surface', subject +
                    '_inner_skull_surface', subject + '_outer_skull_surface',
                    subject + '_outer_skin_surface']
        for s in surfaces:
            cmd = ['mne_convert_surface', '--surf', s, '--mghmri', T1_mgz,
                   '--surfout', s]
            run_subprocess(cmd, env=env, stdout=sys.stdout)
    os.chdir(bem_dir)
    if op.exists(subject + '-head.fif'):
        os.remove(subject + '-head.fif')
    cmd = ['mne_surf2bem', '--surf', op.join(ws_dir,
           subject + '_outer_skin_surface'), '--id', '4', '--fif',
           subject + '-head.fif']
    run_subprocess(cmd, env=env, stdout=sys.stdout)
    logger.info('Created %s/%s-head.fif\n\nComplete.' % (bem_dir, subject))
