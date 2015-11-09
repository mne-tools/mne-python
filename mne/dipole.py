# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import numpy as np
from scipy import linalg
from copy import deepcopy
import re

from .cov import read_cov, _get_whitener_data
from .io.pick import pick_types, channel_type
from .io.proj import make_projector, _has_eeg_average_ref_proj
from .bem import _fit_sphere
from .transforms import (_print_coord_trans, _coord_frame_name,
                         apply_trans, invert_transform, Transform)

from .forward._make_forward import (_get_mri_head_t, _setup_bem,
                                    _prep_meg_channels, _prep_eeg_channels)
from .forward._compute_forward import (_compute_forwards_meeg,
                                       _prep_field_computation)

from .externals.six import string_types
from .surface import (transform_surface_to, _normalize_vectors,
                      _get_ico_surface, _compute_nearest)
from .bem import _bem_find_surface, _bem_explain_surface
from .source_space import (_make_volume_source_space, SourceSpaces,
                           _points_outside_surface)
from .parallel import parallel_func
from .fixes import partial
from .utils import logger, verbose, _time_mask


class Dipole(object):
    """Dipole class

    Used to store positions, orientations, amplitudes, times, goodness of fit
    of dipoles, typically obtained with Neuromag/xfit, mne_dipole_fit
    or certain inverse solvers.

    Parameters
    ----------
    times : array, shape (n_dipoles,)
        The time instants at which each dipole was fitted (sec).
    pos : array, shape (n_dipoles, 3)
        The dipoles positions (m).
    amplitude : array, shape (n_dipoles,)
        The amplitude of the dipoles (nAm).
    ori : array, shape (n_dipoles, 3)
        The dipole orientations (normalized to unit length).
    gof : array, shape (n_dipoles,)
        The goodness of fit.
    name : str | None
        Name of the dipole.
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

        Parameters
        ----------
        tmin : float | None
            Start time of selection in seconds.
        tmax : float | None
            End time of selection in seconds.
        """
        mask = _time_mask(self.times, tmin, tmax)
        for attr in ('times', 'pos', 'gof', 'amplitude', 'ori'):
            setattr(self, attr, getattr(self, attr)[mask])

    def copy(self):
        """Copy the Dipoles object

        Returns
        -------
        dip : instance of Dipole
            The copied dipole instance.
        """
        return deepcopy(self)

    @verbose
    def plot_locations(self, trans, subject, subjects_dir=None,
                       bgcolor=(1, 1, 1), opacity=0.3,
                       brain_color=(0.7, 0.7, 0.7), mesh_color=(1, 1, 0),
                       fig_name=None, fig_size=(600, 600), mode='cone',
                       scale_factor=0.1e-1, colors=None, verbose=None):
        """Plot dipole locations as arrows

        Parameters
        ----------
        trans : dict
            The mri to head trans.
        subject : str
            The subject name corresponding to FreeSurfer environment
            variable SUBJECT.
        subjects_dir : None | str
            The path to the freesurfer subjects reconstructions.
            It corresponds to Freesurfer environment variable SUBJECTS_DIR.
            The default is None.
        bgcolor : tuple of length 3
            Background color in 3D.
        opacity : float in [0, 1]
            Opacity of brain mesh.
        brain_color : tuple of length 3
            Brain color.
        mesh_color : tuple of length 3
            Mesh color.
        fig_name : tuple of length 2
            Mayavi figure name.
        fig_size : tuple of length 2
            Mayavi figure size.
        mode : str
            Should be ``'cone'`` or ``'sphere'`` to specify how the
            dipoles should be shown.
        scale_factor : float
            The scaling applied to amplitudes for the plot.
        colors: list of colors | None
            Color to plot with each dipole. If None defaults colors are used.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).

        Returns
        -------
        fig : instance of mlab.Figure
            The mayavi figure.
        """
        from .viz import plot_dipole_locations
        dipoles = []
        for t in self.times:
            dipoles.append(self.copy())
            dipoles[-1].crop(t, t)
        return plot_dipole_locations(
            dipoles, trans, subject, subjects_dir, bgcolor, opacity,
            brain_color, mesh_color, fig_name, fig_size, mode, scale_factor,
            colors)

    def plot_amplitudes(self, color='k', show=True):
        """Plot the dipole amplitudes as a function of time

        Parameters
        ----------
        color: matplotlib Color
            Color to use for the trace.
        show : bool
            Show figure if True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        """
        from .viz import plot_dipole_amplitudes
        return plot_dipole_amplitudes([self], [color], show)

    def __getitem__(self, idx_slice):
        """Handle indexing"""
        if isinstance(idx_slice, int):  # make sure attributes stay 2d
            idx_slice = [idx_slice]

        selected_times = self.times[idx_slice].copy()
        selected_pos = self.pos[idx_slice, :].copy()
        selected_amplitude = self.amplitude[idx_slice].copy()
        selected_ori = self.ori[idx_slice, :].copy()
        selected_gof = self.gof[idx_slice].copy()
        selected_name = self.name

        new_dipole = Dipole(selected_times, selected_pos,
                            selected_amplitude, selected_ori,
                            selected_gof, selected_name)
        return new_dipole

    def __len__(self):
        """Handle len function"""
        return self.pos.shape[0]


# #############################################################################
# IO

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
    dipole : instance of Dipole
        The dipole.
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
    return Dipole(times, pos, amplitude, ori, gof, name)


# #############################################################################
# Fitting

def _dipole_forwards(fwd_data, whitener, rr, n_jobs=1):
    """Compute the forward solution and do other nice stuff"""
    B = _compute_forwards_meeg(rr, fwd_data, n_jobs, verbose=False)
    B = np.concatenate(B, axis=1)
    B_orig = B.copy()

    # Apply projection and whiten (cov has projections already)
    B = np.dot(B, whitener.T)

    # column normalization doesn't affect our fitting, so skip for now
    # S = np.sum(B * B, axis=1)  # across channels
    # scales = np.repeat(3. / np.sqrt(np.sum(np.reshape(S, (len(rr), 3)),
    #                                        axis=1)), 3)
    # B *= scales[:, np.newaxis]
    scales = np.ones(3)
    return B, B_orig, scales


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
    src = _make_volume_source_space(surf, grid, exclude, 1000 * mindist,
                                    do_neighbors=False, n_jobs=n_jobs)
    # simplify the result to make things easier later
    src = dict(rr=src['rr'][src['vertno']], nn=src['nn'][src['vertno']],
               nuse=src['nuse'], coord_frame=src['coord_frame'],
               vertno=np.arange(src['nuse']))
    return SourceSpaces([src])


def _fit_eval(rd, B, B2, fwd_svd=None, fwd_data=None, whitener=None):
    """Calculate the residual sum of squares"""
    if fwd_svd is None:
        fwd = _dipole_forwards(fwd_data, whitener, rd[np.newaxis, :])[0]
        uu, sing, vv = linalg.svd(fwd, overwrite_a=True, full_matrices=False)
    else:
        uu, sing, vv = fwd_svd
    gof = _dipole_gof(uu, sing, vv, B, B2)[0]
    # mne-c uses fitness=B2-Bm2, but ours (1-gof) is just a normalized version
    return 1. - gof


def _dipole_gof(uu, sing, vv, B, B2):
    """Calculate the goodness of fit from the forward SVD"""
    ncomp = 3 if sing[2] / sing[0] > 0.2 else 2
    one = np.dot(vv[:ncomp], B)
    Bm2 = np.sum(one * one)
    gof = Bm2 / B2
    return gof, one


def _fit_Q(fwd_data, whitener, proj_op, B, B2, B_orig, rd):
    """Fit the dipole moment once the location is known"""
    fwd, fwd_orig, scales = _dipole_forwards(fwd_data, whitener,
                                             rd[np.newaxis, :])
    uu, sing, vv = linalg.svd(fwd, full_matrices=False)
    gof, one = _dipole_gof(uu, sing, vv, B, B2)
    ncomp = len(one)
    # Counteract the effect of column normalization
    Q = scales[0] * np.sum(uu.T[:ncomp] * (one / sing[:ncomp])[:, np.newaxis],
                           axis=0)
    # apply the projector to both elements
    B_residual = np.dot(proj_op, B_orig) - np.dot(np.dot(Q, fwd_orig),
                                                  proj_op.T)
    return Q, gof, B_residual


def _fit_dipoles(min_dist_to_inner_skull, data, times, guess_rrs,
                 guess_fwd_svd, fwd_data, whitener, proj_op, n_jobs):
    """Fit a single dipole to the given whitened, projected data"""
    from scipy.optimize import fmin_cobyla
    parallel, p_fun, _ = parallel_func(_fit_dipole, n_jobs)
    # parallel over time points
    res = parallel(p_fun(min_dist_to_inner_skull, B, t, guess_rrs,
                         guess_fwd_svd, fwd_data, whitener, proj_op,
                         fmin_cobyla)
                   for B, t in zip(data.T, times))
    pos = np.array([r[0] for r in res])
    amp = np.array([r[1] for r in res])
    ori = np.array([r[2] for r in res])
    gof = np.array([r[3] for r in res]) * 100  # convert to percentage
    residual = np.array([r[4] for r in res]).T

    return pos, amp, ori, gof, residual


'''Simplex code in case we ever want/need it for testing

def _make_tetra_simplex():
    """Make the initial tetrahedron"""
    #
    # For this definition of a regular tetrahedron, see
    #
    # http://mathworld.wolfram.com/Tetrahedron.html
    #
    x = np.sqrt(3.0) / 3.0
    r = np.sqrt(6.0) / 12.0
    R = 3 * r
    d = x / 2.0
    simplex = 1e-2 * np.array([[x, 0.0, -r],
                               [-d, 0.5, -r],
                               [-d, -0.5, -r],
                               [0., 0., R]])
    return simplex


def try_(p, y, psum, ndim, fun, ihi, neval, fac):
    """Helper to try a value"""
    ptry = np.empty(ndim)
    fac1 = (1.0 - fac) / ndim
    fac2 = fac1 - fac
    ptry = psum * fac1 - p[ihi] * fac2
    ytry = fun(ptry)
    neval += 1
    if ytry < y[ihi]:
        y[ihi] = ytry
        psum[:] += ptry - p[ihi]
        p[ihi] = ptry
    return ytry, neval


def _simplex_minimize(p, ftol, stol, fun, max_eval=1000):
    """Minimization with the simplex algorithm

    Modified from Numerical recipes"""
    y = np.array([fun(s) for s in p])
    ndim = p.shape[1]
    assert p.shape[0] == ndim + 1
    mpts = ndim + 1
    neval = 0
    psum = p.sum(axis=0)

    loop = 1
    while(True):
        ilo = 1
        if y[1] > y[2]:
            ihi = 1
            inhi = 2
        else:
            ihi = 2
            inhi = 1
        for i in range(mpts):
            if y[i] < y[ilo]:
                ilo = i
            if y[i] > y[ihi]:
                inhi = ihi
                ihi = i
            elif y[i] > y[inhi]:
                if i != ihi:
                    inhi = i

        rtol = 2 * np.abs(y[ihi] - y[ilo]) / (np.abs(y[ihi]) + np.abs(y[ilo]))
        if rtol < ftol:
            break
        if neval >= max_eval:
            raise RuntimeError('Maximum number of evaluations exceeded.')
        if stol > 0:  # Has the simplex collapsed?
            dsum = np.sqrt(np.sum((p[ilo] - p[ihi]) ** 2))
            if loop > 5 and dsum < stol:
                break

        ytry, neval = try_(p, y, psum, ndim, fun, ihi, neval, -1.)
        if ytry <= y[ilo]:
            ytry, neval = try_(p, y, psum, ndim, fun, ihi, neval, 2.)
        elif ytry >= y[inhi]:
            ysave = y[ihi]
            ytry, neval = try_(p, y, psum, ndim, fun, ihi, neval, 0.5)
            if ytry >= ysave:
                for i in range(mpts):
                    if i != ilo:
                        psum[:] = 0.5 * (p[i] + p[ilo])
                        p[i] = psum
                        y[i] = fun(psum)
                neval += ndim
                psum = p.sum(axis=0)
        loop += 1
'''


def _fit_dipole(min_dist_to_inner_skull, B_orig, t, guess_rrs,
                guess_fwd_svd, fwd_data, whitener, proj_op,
                fmin_cobyla):
    """Fit a single bit of data"""
    B = np.dot(whitener, B_orig)

    # make constraint function to keep the solver within the inner skull
    if isinstance(fwd_data['inner_skull'], dict):  # bem
        surf = fwd_data['inner_skull']

        def constraint(rd):

            dist = _compute_nearest(surf['rr'], rd[np.newaxis, :],
                                    return_dists=True)[1][0]

            if _points_outside_surface(rd[np.newaxis, :], surf, 1)[0]:
                dist *= -1.

            # Once we know the dipole is below the inner skull,
            # let's check if its distance to the inner skull is at least
            # min_dist_to_inner_skull. This can be enforced by adding a
            # constrain proportional to its distance.
            dist -= min_dist_to_inner_skull
            return dist

    else:  # sphere
        surf = None
        R, r0 = fwd_data['inner_skull']
        R_adj = R - min_dist_to_inner_skull

        def constraint(rd):
            return R_adj - np.sqrt(np.sum((rd - r0) ** 2))

    # Find a good starting point (find_best_guess in C)
    B2 = np.dot(B, B)
    if B2 == 0:
        logger.warning('Zero field found for time %s' % t)
        return np.zeros(3), 0, np.zeros(3), 0

    idx = np.argmin([_fit_eval(guess_rrs[[fi], :], B, B2, fwd_svd)
                     for fi, fwd_svd in enumerate(guess_fwd_svd)])
    x0 = guess_rrs[idx]
    fun = partial(_fit_eval, B=B, B2=B2, fwd_data=fwd_data, whitener=whitener)

    # Tested minimizers:
    #    Simplex, BFGS, CG, COBYLA, L-BFGS-B, Powell, SLSQP, TNC
    # Several were similar, but COBYLA won for having a handy constraint
    # function we can use to ensure we stay inside the inner skull /
    # smallest sphere
    rd_final = fmin_cobyla(fun, x0, (constraint,), consargs=(),
                           rhobeg=5e-2, rhoend=5e-5, disp=False)

    # simplex = _make_tetra_simplex() + x0
    # _simplex_minimize(simplex, 1e-4, 2e-4, fun)
    # rd_final = simplex[0]

    # Compute the dipole moment at the final point
    Q, gof, residual = _fit_Q(fwd_data, whitener, proj_op, B, B2, B_orig,
                              rd_final)
    amp = np.sqrt(np.dot(Q, Q))
    norm = 1. if amp == 0. else amp
    ori = Q / norm

    msg = '---- Fitted : %7.1f ms' % (1000. * t)
    if surf is not None:
        dist_to_inner_skull = _compute_nearest(surf['rr'],
                                               rd_final[np.newaxis, :],
                                               return_dists=True)[1][0]
        msg += (", distance to inner skull : %2.4f mm"
                % (dist_to_inner_skull * 1000.))

    logger.info(msg)
    return rd_final, amp, ori, gof, residual


@verbose
def fit_dipole(evoked, cov, bem, trans=None, min_dist=5., n_jobs=1,
               verbose=None):
    """Fit a dipole

    Parameters
    ----------
    evoked : instance of Evoked
        The dataset to fit.
    cov : str | instance of Covariance
        The noise covariance.
    bem : str | dict
        The BEM filename (str) or a loaded sphere model (dict).
    trans : str | None
        The head<->MRI transform filename. Must be provided unless BEM
        is a sphere model.
    min_dist : float
        Minimum distance (in milimeters) from the dipole to the inner skull.
        Must be positive. Note that because this is a constraint passed to
        a solver it is not strict but close, i.e. for a ``min_dist=5.`` the
        fits could be 4.9 mm from the inner skull.
    n_jobs : int
        Number of jobs to run in parallel (used in field computation
        and fitting).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    dip : instance of Dipole
        The dipole fits.
    residual : ndarray, shape (n_meeg_channels, n_times)
        The good M-EEG data channels with the fitted dipolar activity
        removed.

    See Also
    --------
    mne.beamformer.rap_music

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    # This could eventually be adapted to work with other inputs, these
    # are what is needed:

    evoked = evoked.copy()

    # Determine if a list of projectors has an average EEG ref
    if "eeg" in evoked and not _has_eeg_average_ref_proj(evoked.info['projs']):
        raise ValueError('EEG average reference is mandatory for dipole '
                         'fitting.')

    if min_dist < 0:
        raise ValueError('min_dist should be positive. Got %s' % min_dist)

    data = evoked.data
    info = evoked.info
    times = evoked.times.copy()
    comment = evoked.comment

    # Convert the min_dist to meters
    min_dist_to_inner_skull = min_dist / 1000.
    del min_dist

    # Figure out our inputs
    neeg = len(pick_types(info, meg=False, eeg=True, exclude=[]))
    if isinstance(bem, string_types):
        logger.info('BEM              : %s' % bem)
    if trans is not None:
        logger.info('MRI transform    : %s' % trans)
        mri_head_t, trans = _get_mri_head_t(trans)
    else:
        mri_head_t = Transform('head', 'mri', np.eye(4))
    bem = _setup_bem(bem, bem, neeg, mri_head_t)
    if not bem['is_sphere']:
        if trans is None:
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
    guess_grid = 0.02  # MNE-C uses 0.01, but this is faster w/similar perf
    guess_mindist = max(0.005, min_dist_to_inner_skull)
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
    ch_types = [channel_type(info, idx) for idx in range(info['nchan'])]

    megcoils, compcoils, megnames, meg_info = [], [], [], None
    eegels, eegnames = [], []
    if 'grad' in ch_types or 'mag' in ch_types:
        megcoils, compcoils, megnames, meg_info = \
            _prep_meg_channels(info, exclude='bads',
                               accurate=accurate, verbose=verbose)
    if 'eeg' in ch_types:
        eegels, eegnames = _prep_eeg_channels(info, exclude='bads',
                                              verbose=verbose)

    # Ensure that MEG and/or EEG channels are present
    if len(megcoils + eegels) == 0:
        raise RuntimeError('No MEG or EEG channels found.')

    # Whitener for the data
    logger.info('Decomposing the sensor noise covariance matrix...')
    picks = pick_types(info, meg=True, eeg=True)

    # In case we want to more closely match MNE-C for debugging:
    # from .io.pick import pick_info
    # from .cov import prepare_noise_cov
    # info_nb = pick_info(info, picks)
    # cov = prepare_noise_cov(cov, info_nb, info_nb['ch_names'], verbose=False)
    # nzero = (cov['eig'] > 0)
    # n_chan = len(info_nb['ch_names'])
    # whitener = np.zeros((n_chan, n_chan), dtype=np.float)
    # whitener[nzero, nzero] = 1.0 / np.sqrt(cov['eig'][nzero])
    # whitener = np.dot(whitener, cov['eigvec'])

    whitener = _get_whitener_data(info, cov, picks, verbose=False)

    # Proceed to computing the fits (make_guess_data)
    logger.info('\n---- Computing the forward solution for the guesses...')
    guess_src = _make_guesses(inner_skull, r0_mri,
                              guess_grid, guess_exclude, guess_mindist,
                              n_jobs=n_jobs)[0]
    if isinstance(inner_skull, dict):
        transform_surface_to(inner_skull, 'head', mri_head_t)
    transform_surface_to(guess_src, 'head', mri_head_t)

    # C code computes guesses using a sphere model for speed, don't bother here
    logger.info('Go through all guess source locations...')
    fwd_data = dict(coils_list=[megcoils, eegels], infos=[meg_info, None],
                    ccoils_list=[compcoils, None], coil_types=['meg', 'eeg'],
                    inner_skull=inner_skull)
    _prep_field_computation(guess_src['rr'], bem, fwd_data, n_jobs,
                            verbose=False)
    guess_fwd = _dipole_forwards(fwd_data, whitener, guess_src['rr'],
                                 n_jobs=n_jobs)[0]
    # decompose ahead of time
    guess_fwd_svd = [linalg.svd(fwd, overwrite_a=True, full_matrices=False)
                     for fwd in np.array_split(guess_fwd,
                                               len(guess_src['rr']))]
    del guess_fwd  # destroyed
    logger.info('[done %d sources]' % guess_src['nuse'])

    # Do actual fits
    data = data[picks]
    ch_names = [info['ch_names'][p] for p in picks]
    proj_op = make_projector(info['projs'], ch_names, info['bads'])[0]
    out = _fit_dipoles(min_dist_to_inner_skull, data, times, guess_src['rr'],
                       guess_fwd_svd, fwd_data,
                       whitener, proj_op, n_jobs)
    dipoles = Dipole(times, out[0], out[1], out[2], out[3], comment)
    residual = out[4]

    logger.info('%d dipoles fitted' % len(dipoles.times))
    return dipoles, residual
