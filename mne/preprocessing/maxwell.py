# -*- coding: utf-8 -*-
# Authors: Mark Wronkiewicz <wronk.mark@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#          Jussi Nurminen <jnu@iki.fi>


# License: BSD (3-clause)

from __future__ import division
from copy import deepcopy
import numpy as np
from scipy import linalg
from math import factorial
from os import path as op
import inspect

from .. import __version__
from ..forward._compute_forward import _concatenate_coils
from ..forward._make_forward import _prep_meg_channels
from ..surface import _normalize_vectors
from ..io.constants import FIFF
from ..io.open import fiff_open
from ..io.tree import dir_tree_find
from ..io.write import _generate_meas_id, _date_now
from ..io.tag import find_tag
from ..io.pick import pick_types, pick_info, pick_channels
from ..utils import verbose, logger


@verbose
def maxwell_filter(raw, origin=(0., 0., 40.), int_order=8, ext_order=3,
                   fine_cal=None, ctc=None, st_dur=None, st_corr=0.98,
                   verbose=None):
    """Apply Maxwell filter to data using spherical harmonics.

    Parameters
    ----------
    raw : instance of mne.io.Raw
        Data to be filtered
    origin : array-like, shape (3,)
        Origin of internal and external multipolar moment space in head coords
        and in millimeters
    int_order : int
        Order of internal component of spherical expansion
    ext_order : int
        Order of external component of spherical expansion
    fine_cal : str | None
        Path to the ``'.dat'`` file with fine calibration coefficients.
        File can have 1D or 3D gradiometer imbalance correction.
        This file is machine/site-specific.
    ctc : str | None
        Path to the FIF file with cross-talk correction information.
    st_dur : float | None
        If not None, apply spatiotemporal SSS with specified buffer duration
        (in seconds). Elekta's default is 10.0 seconds in MaxFilter v2.2.
        Spatiotemporal SSS acts as implicitly as a high-pass filter where the
        cut-off frequency is 1/st_dur Hz. For this (and other) reasons, longer
        buffers are generally better as long as your system can handle the
        higher memory usage. To ensure that each window is processed
        identically, choose a buffer length that divides evenly into your data.
        Any data at the trailing edge that doesn't fit evenly into a whole
        buffer window will be lumped into the previous buffer.
    st_corr : float
        Correlation limit between inner and outer subspaces used to reject
        ovwrlapping intersecting inner/outer signals during spatiotemporal SSS.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose)

    Returns
    -------
    raw_sss : instance of mne.io.Raw
        The raw data with Maxwell filtering applied

    Notes
    -----
    .. versionadded:: 0.10

    Equation numbers refer to Taulu and Kajola, 2005 [1]_ unless otherwise
    noted.

    Some of this code was adapted and relicensed (with BSD form) with
    permission from Jussi Nurminen.

    References
    ----------
    .. [1] Taulu S. and Kajola M. "Presentation of electromagnetic
           multichannel data: The signal space separation method,"
           Journal of Applied Physics, vol. 97, pp. 124905 1-10, 2005.

           http://lib.tkk.fi/Diss/2008/isbn9789512295654/article2.pdf

    .. [2] Taulu S. and Simola J. "Spatiotemporal signal space separation
           method for rejecting nearby interference in MEG measurements,"
           Physics in Medicine and Biology, vol. 51, pp. 1759-1768, 2006.

           http://lib.tkk.fi/Diss/2008/isbn9789512295654/article3.pdf
    """
    # There are an absurd number of different possible notations for spherical
    # coordinates, which confounds the notation for spherical harmonics.  Here,
    # we purposefully stay away from shorthand notation in both and use
    # explicit terms (like 'azimuth' and 'polar') to avoid confusion.
    # See mathworld.wolfram.com/SphericalHarmonic.html for more discussion.
    # Our code follows the same standard that ``scipy`` uses for ``sph_harm``.

    if raw.proj:
        raise RuntimeError('Projectors cannot be applied to raw data.')
    if len(raw.info.get('comps', [])) > 0:
        raise RuntimeError('Maxwell filter cannot handle compensated '
                           'channels.')
    st_corr = float(st_corr)
    if st_corr <= 0. or st_corr > 1.:
        raise ValueError('Need 0 < st_corr <= 1., got %s' % st_corr)
    logger.info('Maxwell filtering raw data')
    if len(raw.info['bads']) > 0:
        logger.info('    Bad channels being reconstructed: %s'
                    % raw.info['bads'])
    else:
        logger.info('    No bad channels')
    raw_sss = raw.copy().load_data(verbose=False)
    del raw

    # If necessary, load fine calibration and overwrite sensor geometry
    if fine_cal is not None:
        logger.info('    Using fine calibration %s' % op.basename(fine_cal))
        cal_chans = _read_fine_cal(raw_sss.info, fine_cal)
        raw_sss.info = _update_sensor_geometry(raw_sss.info, cal_chans)

    times = raw_sss.times

    # Get indices of channels to use in multipolar moment calculation
    good_picks = pick_types(raw_sss.info, meg=True, exclude='bads')
    # Get indices of MEG channels
    meg_picks = pick_types(raw_sss.info, meg=True, exclude=[])
    meg_coils, _, _, meg_info = _prep_meg_channels(
        raw_sss.info, accurate=True, elekta_defs=True, verbose=False)

    # Magnetometers (with coil_class == 1.0) must be scaled by 100 to improve
    # numerical stability as they have different scales than gradiometers
    mag_scale = 100.
    coil_scale = np.ones((len(meg_picks), 1))
    coil_scale[np.array([coil['coil_class'] == 1.0
                         for coil in meg_coils])] = mag_scale

    # Compute multipolar moment bases
    origin = np.array(origin) / 1000.  # Convert scale from mm to m
    # Compute in/out bases and create copies containing only good chs
    S_tot = _sss_basis(origin, meg_coils, int_order, ext_order)

    # Cross-talk processing
    if ctc is not None:
        ctc = _read_ctc(ctc, raw_sss.info, meg_picks, good_picks)

    # Fine calibration processing (point-like magnetometers and calib. coeffs)
    if fine_cal is not None:
        mag_inds = pick_types(raw_sss.info, meg='mag')
        grad_inds = pick_types(raw_sss.info, meg='grad')
        grad_info = pick_info(raw_sss.info, grad_inds)

        # Compute point-like mags to incorporate gradiometer imbalance
        grad_imbalances = np.array([cal_chans[ii]['calib_coeff']
                                    for ii in grad_inds])
        S_fine = _sss_basis_point(origin, grad_info, int_order, ext_order,
                                  grad_imbalances)

        # Add point like magnetometer data to bases.
        S_tot[grad_inds, :] += S_fine

        # Scale magnetometers by calibration coefficient
        mag_calib_coeffs = np.array([cal_chans[ii]['calib_coeff']
                                     for ii in mag_inds])
        S_tot[mag_inds, :] /= mag_calib_coeffs

    S_tot_good = S_tot[good_picks, :]
    S_tot_good_norm = np.sqrt(np.sum(S_tot_good *
                                     S_tot_good, axis=0))[:, np.newaxis]
    S_tot_good /= S_tot_good_norm.T

    # Pseudo-inverse of total multipolar moment basis set (Part of Eq. 37)
    pS_tot_good = linalg.pinv(S_tot_good, cond=1e-15)

    # Compute multipolar moments of (magnetometer scaled) data (Eq. 37)
    # XXX eventually we can refactor this to work in chunks
    data = raw_sss[good_picks][0]
    if ctc is not None:
        data = ctc.dot(data)
    mm = np.dot(pS_tot_good, data * coil_scale[good_picks])
    # Reconstruct data from internal space (Eq. 38)
    n_in = _get_n_moments(int_order)
    raw_sss._data[meg_picks] = np.dot(S_tot[:, :n_in],
                                      mm[:n_in] / S_tot_good_norm[:n_in])
    raw_sss._data[meg_picks] /= coil_scale

    # Reset 'bads' for any MEG channels since they've been reconstructed
    bad_inds = [raw_sss.info['ch_names'].index(ch)
                for ch in raw_sss.info['bads']]
    raw_sss.info['bads'] = [raw_sss.info['ch_names'][bi] for bi in bad_inds
                            if bi not in meg_picks]

    # Reconstruct raw file object with spatiotemporal processed data
    if st_dur is not None:
        if st_dur > times[-1]:
            raise ValueError('st_dur (%0.1fs) longer than length of signal in '
                             'raw (%0.1fs).' % (st_dur, times[-1]))
        logger.info('Processing data using tSSS with st_dur=%s' % st_dur)

        # Generate time points to break up data in to windows
        lims = raw_sss.time_as_index(np.arange(times[0], times[-1], st_dur))
        len_last_buf = raw_sss.times[-1] - raw_sss.index_as_time(lims[-1])[0]
        if len_last_buf == st_dur:
            lims = np.concatenate([lims, [len(raw_sss.times)]])
        else:
            # len_last_buf < st_dur so fold it into the previous buffer
            lims[-1] = len(raw_sss.times)
            logger.info('Spatiotemporal window did not fit evenly into raw '
                        'object. The final %0.2f seconds were lumped onto '
                        'the previous window.' % len_last_buf)

        # Loop through buffer windows of data
        for win in zip(lims[:-1], lims[1:]):
            # Reconstruct data from external space and compute residual
            resid = data[:, win[0]:win[1]]
            resid -= raw_sss._data[meg_picks, win[0]:win[1]]
            resid -= np.dot(S_tot[:, n_in:], mm[n_in:, win[0]:win[1]] /
                            S_tot_good_norm[n_in:]) / coil_scale
            _check_finite(resid)

            # Compute SSP-like projector. Set overlap limit to 0.02
            this_data = raw_sss._data[meg_picks, win[0]:win[1]]
            _check_finite(this_data)
            V = _overlap_projector(this_data, resid, st_corr)

            # Apply projector according to Eq. 12 in [2]_
            logger.info('    Projecting out %s tSSS components for %s-%s'
                        % (V.shape[1], win[0] / raw_sss.info['sfreq'],
                           win[1] / raw_sss.info['sfreq']))
            this_data -= np.dot(np.dot(this_data, V), V.T)
            raw_sss._data[meg_picks, win[0]:win[1]] = this_data

    # Update info
    raw_sss = _update_sss_info(raw_sss, origin, int_order, ext_order,
                               len(good_picks))

    return raw_sss


def _check_finite(data):
    """Helper to ensure data is finite"""
    if not np.isfinite(data).all():
        raise RuntimeError('data contains non-finite numbers')


def _sph_harm(order, degree, az, pol):
    """Evaluate point in specified multipolar moment. [1]_ Equation 4.

    When using, pay close attention to inputs. Spherical harmonic notation for
    order/degree, and theta/phi are both reversed in original SSS work compared
    to many other sources. See mathworld.wolfram.com/SphericalHarmonic.html for
    more discussion.

    Note that scipy has ``scipy.special.sph_harm``, but that function is
    too slow on old versions (< 0.15) and has a weird bug on newer versions.
    At some point we should track it down and open a bug report...

    Parameters
    ----------
    order : int
        Order of spherical harmonic. (Usually) corresponds to 'm'
    degree : int
        Degree of spherical harmonic. (Usually) corresponds to 'l'
    az : float
        Azimuthal (longitudinal) spherical coordinate [0, 2*pi]. 0 is aligned
        with x-axis.
    pol : float
        Polar (or colatitudinal) spherical coordinate [0, pi]. 0 is aligned
        with z-axis.

    Returns
    -------
    base : complex float
        The spherical harmonic value at the specified azimuth and polar angles
    """
    from scipy.special import lpmv

    # Error checks
    if np.abs(order) > degree:
        raise ValueError('Absolute value of expansion coefficient must be <= '
                         'degree')
    # Ensure that polar and azimuth angles are arrays
    az = np.asarray(az)
    pol = np.asarray(pol)
    if (az < -2 * np.pi).any() or (az > 2 * np.pi).any():
        raise ValueError('Azimuth coords must lie in [-2*pi, 2*pi]')
    if(pol < 0).any() or (pol > np.pi).any():
        raise ValueError('Polar coords must lie in [0, pi]')

    base = np.sqrt((2 * degree + 1) / (4 * np.pi) * factorial(degree - order) /
                   factorial(degree + order)) * \
        lpmv(order, degree, np.cos(pol)) * np.exp(1j * order * az)
    return base


def _sss_basis(origin, coils, int_order, ext_order):
    """Compute SSS basis for given conditions.

    Parameters
    ----------
    origin : ndarray, shape (3,)
        Origin of the multipolar moment space in millimeters
    coils : list
        List of MEG coils. Each should contain coil information dict specifying
        position, normals, weights, number of integration points and channel
        type. All position info must be in the 'origin' coordinate frame
    int_order : int
        Order of the internal multipolar moment space
    ext_order : int
        Order of the external multipolar moment space

    Returns
    -------
    bases : ndarray, shape (n_coils, n_mult_moments)
        Internal and external basis sets as a single ndarray.

    Notes
    -----
    Incorporates magnetometer scaling factor. Does not normalize spaces.
    """

    # Get position, normal, weights, and number of integration pts.
    r_int_pts, ncoils, wcoils, counts = _concatenate_coils(coils)
    bins = np.repeat(np.arange(len(counts)), counts)
    n_sens = len(counts)
    n_bases = _get_n_moments([int_order, ext_order]).sum()

    n_in, n_out = _get_n_moments([int_order, ext_order])
    S_tot = np.empty((n_sens, n_in + n_out))
    S_tot.fill(np.nan)
    S_in = S_tot[:, :n_in]
    S_out = S_tot[:, n_in:]

    # Set all magnetometers (with 'coil_class' == 1.0) to be scaled by 100
    coil_scale = np.ones((len(coils)))
    coil_scale[np.array([coil['coil_class'] == 1.0 for coil in coils])] = 100.

    if n_bases > n_sens:
        raise ValueError('Number of requested bases (%s) exceeds number of '
                         'sensors (%s)' % (str(n_bases), str(n_sens)))

    # Compute position vector between origin and coil integration pts
    cvec_cart = r_int_pts - origin[np.newaxis, :]
    # Convert points to spherical coordinates
    cvec_sph = _cart_to_sph(cvec_cart)

    # Compute internal/external basis vectors (exclude degree 0; L/RHS Eq. 5)
    for spc, g_func, order in zip([S_in, S_out],
                                  [_grad_in_components, _grad_out_components],
                                  [int_order, ext_order]):
        for deg in range(1, order + 1):
            for order in range(-deg, deg + 1):

                # Compute gradient for all integration points
                grads = -1 * g_func(deg, order, cvec_sph[:, 0], cvec_sph[:, 1],
                                    cvec_sph[:, 2])

                # Gradients dotted with integration point normals and weighted
                all_grads = wcoils * np.einsum('ij,ij->i', grads, ncoils)

                # For order and degree, sum over each sensor's integration pts
                # for pt_i in range(0, len(int_lens) - 1):
                #    int_pts_sum = \
                #        np.sum(all_grads[int_lens[pt_i]:int_lens[pt_i + 1]])
                #    spc[pt_i, deg ** 2 + deg + order - 1] = int_pts_sum
                spc[:, deg ** 2 + deg + order - 1] = \
                    np.bincount(bins, weights=all_grads, minlength=len(counts))

        # Scale magnetometers
        spc *= coil_scale[:, np.newaxis]

    return S_tot


def _alegendre_deriv(degree, order, val):
    """Compute the derivative of the associated Legendre polynomial at a value.

    Parameters
    ----------
    degree : int
        Degree of spherical harmonic. (Usually) corresponds to 'l'
    order : int
        Order of spherical harmonic. (Usually) corresponds to 'm'
    val : float
        Value to evaluate the derivative at

    Returns
    -------
    dPlm : float
        Associated Legendre function derivative
    """
    from scipy.special import lpmv

    C = 1
    if order < 0:
        order = abs(order)
        C = (-1) ** order * factorial(degree - order) / factorial(degree +
                                                                  order)
    return C * (order * val * lpmv(order, degree, val) + (degree + order) *
                (degree - order + 1) * np.sqrt(1 - val ** 2) *
                lpmv(order - 1, degree, val)) / (1 - val ** 2)


def _grad_in_components(degree, order, rad, az, pol):
    """Compute gradient of internal component of V(r) spherical expansion.

    Internal component has form: Ylm(pol, az) / (rad ** (degree + 1))

    Parameters
    ----------
    degree : int
        Degree of spherical harmonic. (Usually) corresponds to 'l'
    order : int
        Order of spherical harmonic. (Usually) corresponds to 'm'
    rad : ndarray, shape (n_samples,)
        Array of radii
    az : ndarray, shape (n_samples,)
        Array of azimuthal (longitudinal) spherical coordinates [0, 2*pi]. 0 is
        aligned with x-axis.
    pol : ndarray, shape (n_samples,)
        Array of polar (or colatitudinal) spherical coordinates [0, pi]. 0 is
        aligned with z-axis.

    Returns
    -------
    grads : ndarray, shape (n_samples, 3)
        Gradient of the spherical harmonic and vector specified in rectangular
        coordinates
    """
    # Compute gradients for all spherical coordinates (Eq. 6)
    g_rad = (-(degree + 1) / rad ** (degree + 2) *
             _sph_harm(order, degree, az, pol))

    g_az = (1 / (rad ** (degree + 2) * np.sin(pol)) * 1j * order *
            _sph_harm(order, degree, az, pol))

    g_pol = (1 / rad ** (degree + 2) *
             np.sqrt((2 * degree + 1) * factorial(degree - order) /
                     (4 * np.pi * factorial(degree + order))) *
             np.sin(-pol) * _alegendre_deriv(degree, order, np.cos(pol)) *
             np.exp(1j * order * az))

    # Get real component of vectors, convert to cartesian coords, and return
    real_grads = _get_real_grad(np.c_[g_rad, g_az, g_pol], order)
    return _sph_to_cart_partials(np.c_[rad, az, pol], real_grads)


def _grad_out_components(degree, order, rad, az, pol):
    """Compute gradient of external component of V(r) spherical expansion.

    External component has form: Ylm(azimuth, polar) * (radius ** degree)

    Parameters
    ----------
    degree : int
        Degree of spherical harmonic. (Usually) corresponds to 'l'
    order : int
        Order of spherical harmonic. (Usually) corresponds to 'm'
    rad : ndarray, shape (n_samples,)
        Array of radii
    az : ndarray, shape (n_samples,)
        Array of azimuthal (longitudinal) spherical coordinates [0, 2*pi]. 0 is
        aligned with x-axis.
    pol : ndarray, shape (n_samples,)
        Array of polar (or colatitudinal) spherical coordinates [0, pi]. 0 is
        aligned with z-axis.

    Returns
    -------
    grads : ndarray, shape (n_samples, 3)
        Gradient of the spherical harmonic and vector specified in rectangular
        coordinates
    """
    # Compute gradients for all spherical coordinates (Eq. 7)
    g_rad = degree * rad ** (degree - 1) * _sph_harm(order, degree, az, pol)

    g_az = (rad ** (degree - 1) / np.sin(pol) * 1j * order *
            _sph_harm(order, degree, az, pol))

    g_pol = (rad ** (degree - 1) *
             np.sqrt((2 * degree + 1) * factorial(degree - order) /
                     (4 * np.pi * factorial(degree + order))) *
             np.sin(-pol) * _alegendre_deriv(degree, order, np.cos(pol)) *
             np.exp(1j * order * az))

    # Get real component of vectors, convert to cartesian coords, and return
    real_grads = _get_real_grad(np.c_[g_rad, g_az, g_pol], order)
    return _sph_to_cart_partials(np.c_[rad, az, pol], real_grads)


def _get_real_grad(grad_vec_raw, order):
    """Helper function to convert gradient vector to real basis functions.

    Parameters
    ----------
    grad_vec_raw : ndarray, shape (n_gradients, 3)
        Gradient array with columns for radius, azimuth, polar points
    order : int
        Order (usually 'm') of multipolar moment.

    Returns
    -------
    grad_vec : ndarray, shape (n_gradients, 3)
        Gradient vectors with only real componnet
    """

    if order > 0:
        grad_vec = np.sqrt(2) * np.real(grad_vec_raw)
    elif order < 0:
        grad_vec = np.sqrt(2) * np.imag(grad_vec_raw)
    else:
        grad_vec = grad_vec_raw

    return np.real(grad_vec)


def _get_n_moments(order):
    """Compute the number of multipolar moments.

    Equivalent to [1]_ Eq. 32.

    Parameters
    ----------
    order : array-like
        Expansion orders, often ``[int_order, ext_order]``.

    Returns
    -------
    M : ndarray
        Number of moments due to each order.
    """
    order = np.asarray(order, int)
    return (order + 2) * order


def _sph_to_cart_partials(sph_pts, sph_grads):
    """Convert spherical partial derivatives to cartesian coords.

    Note: Because we are dealing with partial derivatives, this calculation is
    not a static transformation. The transformation matrix itself is dependent
    on azimuth and polar coord.

    See the 'Spherical coordinate sytem' section here:
    wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates

    Parameters
    ----------
    sph_pts : ndarray, shape (n_points, 3)
        Array containing spherical coordinates points (rad, azimuth, polar)
    sph_grads : ndarray, shape (n_points, 3)
        Array containing partial derivatives at each spherical coordinate

    Returns
    -------
    cart_grads : ndarray, shape (n_points, 3)
        Array containing partial derivatives in Cartesian coordinates (x, y, z)
    """
    cart_grads = np.zeros_like(sph_grads)
    c_as, s_as = np.cos(sph_pts[:, 1]), np.sin(sph_pts[:, 1])
    c_ps, s_ps = np.cos(sph_pts[:, 2]), np.sin(sph_pts[:, 2])
    trans = np.array([[c_as * s_ps, -s_as, c_as * c_ps],
                      [s_as * s_ps, c_as, c_ps * s_as],
                      [c_ps, np.zeros_like(c_as), -s_ps]])
    cart_grads = np.einsum('ijk,kj->ki', trans, sph_grads)
    return cart_grads


def _cart_to_sph(cart_pts):
    """Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    cart_pts : ndarray, shape (n_points, 3)
        Array containing points in Cartesian coordinates (x, y, z)

    Returns
    -------
    sph_pts : ndarray, shape (n_points, 3)
        Array containing points in spherical coordinates (rad, azimuth, polar)
    """
    rad = np.sqrt(np.sum(cart_pts * cart_pts, axis=1))
    az = np.arctan2(cart_pts[:, 1], cart_pts[:, 0])
    pol = np.arccos(cart_pts[:, 2] / rad)
    return np.array([rad, az, pol]).T


def _update_sss_info(raw, origin, int_order, ext_order, nsens):
    """Helper function to update info after Maxwell filtering.

    Parameters
    ----------
    raw : instance of mne.io.Raw
        Data to be filtered
    origin : array-like, shape (3,)
        Origin of internal and external multipolar moment space in head coords
        and in millimeters
    int_order : int
        Order of internal component of spherical expansion
    ext_order : int
        Order of external component of spherical expansion
    nsens : int
        Number of sensors

    Returns
    -------
    raw : mne.io.Raw
        raw file object with raw.info modified
    """
    # TODO: Continue to fill out bookkeeping info as additional features
    # are added (fine calibration, cross-talk calibration, etc.)
    int_moments, ext_moments = _get_n_moments([int_order, ext_order])

    raw.info['maxshield'] = False
    sss_info_dict = dict(in_order=int_order, out_order=ext_order,
                         nsens=nsens, origin=origin.astype('float32'),
                         n_int_moments=int_moments,
                         frame=raw.info['dev_head_t']['to'],
                         components=np.ones(int_moments +
                                            ext_moments).astype('int32'))

    max_info_dict = dict(max_st={}, sss_cal={}, sss_ctc={},
                         sss_info=sss_info_dict)

    block_id = _generate_meas_id()
    proc_block = dict(max_info=max_info_dict, block_id=block_id,
                      creator='mne-python v%s' % __version__,
                      date=_date_now(), experimentor='')

    # Insert information in raw.info['proc_info']
    raw.info['proc_history'] = [proc_block] + raw.info.get('proc_history', [])
    return raw


check_disable = dict()  # not available on really old versions of SciPy
if 'check_finite' in inspect.getargspec(linalg.svd)[0]:
    check_disable['check_finite'] = False


def _orth_overwrite(A):
    """Helper to create a slightly more efficient 'orth'"""
    # adapted from scipy/linalg/decomp_svd.py
    u, s = linalg.svd(A, overwrite_a=True, full_matrices=False,
                      **check_disable)[:2]
    M, N = A.shape
    eps = np.finfo(float).eps
    tol = max(M, N) * np.amax(s) * eps
    num = np.sum(s > tol, dtype=int)
    return u[:, :num]


def _overlap_projector(data_int, data_res, corr):
    """Calculate projector for removal of subspace intersection in tSSS"""
    # corr necessary to deal with noise when finding identical signal
    # directions in the subspace. See the end of the Results section in [2]_

    # Note that the procedure here is an updated version of [2]_ (and used in
    # Elekta's tSSS) that uses residuals instead of internal/external spaces
    # directly. This provides more degrees of freedom when analyzing for
    # intersections between internal and external spaces.

    # Normalize data, then compute orth to get temporal bases. Matrices
    # must have shape (n_samps x effective_rank) when passed into svd
    # computation
    Q_int = linalg.qr(_orth_overwrite((data_int / np.linalg.norm(data_int)).T),
                      overwrite_a=True, mode='economic', **check_disable)[0].T
    Q_res = linalg.qr(_orth_overwrite((data_res / np.linalg.norm(data_res)).T),
                      overwrite_a=True, mode='economic', **check_disable)[0]
    assert data_int.shape[1] > 0
    C_mat = np.dot(Q_int, Q_res)
    del Q_int

    # Compute angles between subspace and which bases to keep
    S_intersect, Vh_intersect = linalg.svd(C_mat, overwrite_a=True,
                                           full_matrices=False,
                                           **check_disable)[1:]
    del C_mat
    intersect_mask = (S_intersect >= corr)
    del S_intersect

    # Compute projection operator as (I-LL_T) Eq. 12 in [2]_
    # V_principal should be shape (n_time_pts x n_retained_inds)
    Vh_intersect = Vh_intersect[intersect_mask].T
    V_principal = np.dot(Q_res, Vh_intersect)
    return V_principal


def _read_ctc(ctc, info, meg_picks, good_picks):
    """Helper to read a cross-talk correction matrix"""
    f, tree, _ = fiff_open(ctc)
    with f as fid:
        node = dir_tree_find(tree, FIFF.FIFFB_DATA_CORRECTION)
        comment = find_tag(fid, node[0], FIFF.FIFF_COMMENT).data
        assert comment == 'cross-talk compensation matrix'
        node = dir_tree_find(node[0], FIFF.FIFFB_CHANNEL_DECOUPLER)
        ctc = find_tag(fid, node[0], FIFF.FIFF_DECOUPLER_MATRIX)
        assert ctc is not None
        ctc = ctc.data
        chs = find_tag(fid, node[0], FIFF.FIFF_PROJ_ITEM_CH_NAME_LIST)
        assert chs is not None
        chs = chs.data.strip().split(':')
        # XXX for some reason this list has a bunch of junk in the last entry:
        # [..., u'MEG2642', u'MEG2643', u'MEG2641\x00\x00\x00 ... \x00']
        chs[-1] = chs[-1].split('\x00')[0]
        if set(info['ch_names'][p] for p in meg_picks) != set(chs):
            raise RuntimeError('ctc channels and raw channels do not match')
    ctc_picks = pick_channels(chs, [info['ch_names'][c] for c in good_picks])
    ctc = ctc[ctc_picks][:, ctc_picks]
    return ctc


def _read_fine_cal(info, fine_cal):
    """Read sensor locations and calib. coeffs from fine calibration file."""

    # Read new sensor locations
    with open(fine_cal, 'r') as fid:
        lines = [line for line in fid if line[0] not in '#\n']
        cal_chans = [None] * len(lines)
        for line in lines:
            # `vals` contains channel number, (x, y, z), x-norm 3-vec, y-norm
            # 3-vec, z-norm 3-vec, and (1 or 3) imbalance terms
            vals = np.fromstring(line, sep=' ').astype(np.float64)

            # Check for correct number of items
            if len(vals) not in [14, 16]:
                raise RuntimeError('Error reading fine calibration file')

            ch_name = 'MEG' + '%04d' % vals[0]  # Zero-pad names to 4 char
            loc = vals[1:13]  # Get orientation information for 'loc'

            # Get orientation information for 'coil_trans'
            coil_trans = np.eye(4)
            coil_trans[:3, :3] = loc[3:].reshape(3, 3).T
            coil_trans[:3, 3] = loc[:3]
            calib_coeff = vals[13:].copy()  # Get imbalance/calibration coeff
            idx = info['ch_names'].index(ch_name)
            assert cal_chans[idx] is None
            cal_chans[idx] = dict(ch_name=ch_name, coil_trans=coil_trans,
                                  calib_coeff=calib_coeff,
                                  coord_frame=FIFF.FIFFV_COORD_DEVICE)

    # Check that we ended up with correct channels
    meg_info = pick_info(info, pick_types(info, exclude=[]))
    order = pick_channels([c['ch_name'] for c in cal_chans],
                          meg_info['ch_names'])
    if not (len(cal_chans) == meg_info['nchan'] == len(order)):
        raise RuntimeError('Number of channels in fine calibration file (%i) '
                           'does not equal number of channels in info (%i)' %
                           (len(cal_chans), info['nchan']))
    # ensure they're ordered like our data
    cal_chans = [cal_chans[ii] for ii in order]
    return cal_chans


def _update_sensor_geometry(info, cal_chans):
    """Helper to replace sensor geometry information"""

    # Replace sensor locations (and track differences) for fine calibration
    ang_shift = np.zeros((len(cal_chans), 3))
    used = np.zeros(len(info['chs']))
    for ci, cal_chan in enumerate(cal_chans):
        idx = info['ch_names'].index(cal_chan['ch_name'])
        assert not used[idx]
        used[idx] = True
        info_chan = info['chs'][idx]

        # calculate shift angle
        v1 = cal_chan['coil_trans'][:3, :3]
        _normalize_vectors(v1)
        v2 = info_chan['coil_trans'][:3, :3]
        _normalize_vectors(v2)
        ang_shift[ci] = np.sum(v1 * v2, axis=1)

        # Adjust channel orientation with those from fine calibration
        info_chan['coil_trans'] = cal_chan['coil_trans']
        assert info_chan['coord_frame'] == cal_chan['coord_frame']

    # Deal with numerical precision giving absolute vals slightly more than 1.
    np.clip(ang_shift, -1., 1., ang_shift)
    np.rad2deg(np.arccos(ang_shift), ang_shift)  # Convert to degrees

    # Log quantification of sensor changes
    logger.info('    Fine calibration adjusted coil positions by (μ ± σ): '
                '%0.1f° ± %0.1f° (max: %0.1f°)' %
                (np.mean(ang_shift), np.std(ang_shift),
                 np.max(np.abs(ang_shift))))

    return info


def _sss_basis_point(origin, info, int_order, ext_order, imbalance,
                     head_frame=True):
    """Compute multipolar moments for point-like magnetometers (in fine cal)"""

    # Construct 'coils' with r, weights, normal vecs, # integration pts, and
    # channel type.
    if imbalance.ndim == 1:
        imbalance = imbalance[:, np.newaxis]
    if imbalance.shape[1] not in [1, 3]:
        raise ValueError('Must have 1 (x) or 3 (x, y, z) point-like ' +
                         'magnetometers. Currently have %i' %
                         imbalance.shape[1])

    # Coil_type values for x, y, z point magnetometers
    pt_types = [FIFF.FIFFV_COIL_POINT_MAGNETOMETER_X,
                FIFF.FIFFV_COIL_POINT_MAGNETOMETER_Y,
                FIFF.FIFFV_COIL_POINT_MAGNETOMETER]

    # Loop over all coordinate directions desired and create point mags
    S_tot = None
    for dir_ind in range(imbalance.shape[1]):
        temp_info = deepcopy(info)
        for ch in temp_info['chs']:
            ch['coil_type'] = pt_types[dir_ind]
        coils_add = _prep_meg_channels(temp_info, accurate=True,
                                       elekta_defs=True, head_frame=head_frame,
                                       verbose=False)[0]
        S_add = _sss_basis(origin, coils_add, int_order, ext_order)
        S_add *= imbalance[:, [dir_ind]]
        if S_tot is None:
            S_tot = S_add
        else:
            S_tot += S_add
        # Scale spaces by gradiometer imbalance

    # Return point-like mag bases
    return S_tot
