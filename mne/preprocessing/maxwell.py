# -*- coding: utf-8 -*-
# Authors: Mark Wronkiewicz <wronk.mark@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#          Jussi Nurminen <jnu@iki.fi>


# License: BSD (3-clause)

from copy import deepcopy
import numpy as np
from scipy import linalg
from math import factorial
from os import path as op
import inspect

from .. import __version__
from ..transforms import _str_to_frame, _get_trans
from ..forward._compute_forward import _concatenate_coils
from ..forward._make_forward import _prep_meg_channels
from ..surface import _normalize_vectors
from ..io.constants import FIFF
from ..io.open import fiff_open
from ..io.tree import dir_tree_find
from ..io.write import _generate_meas_id, _date_now
from ..io.tag import find_tag, _loc_to_coil_trans
from ..io.pick import pick_types, pick_info, pick_channels
from ..utils import verbose, logger
from ..externals.six import string_types
from ..channels.channels import _get_T1T2_mag_inds


@verbose
def maxwell_filter(raw, origin='default', int_order=8, ext_order=3,
                   fine_cal=None, ctc=None, st_dur=None, st_corr=0.98,
                   coord_frame='head', destination=None, verbose=None):
    """Apply Maxwell filter to data using spherical harmonics.

    Parameters
    ----------
    raw : instance of mne.io.Raw
        Data to be filtered
    origin : array-like, shape (3,) | str
        Origin of internal and external multipolar moment space in head coords
        and in meters. The default is ``'default'``, which means
        ``(0., 0., 40e-3)`` for ``coord_frame='head'`` and
        ``(0., 13e-3, -6e-3)`` for ``coord_frame='meg'``.
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
    coord_frame : str
        The coordinate frame that the ``origin`` is specified in, either
        ``'meg'`` or ``'head'``. For empty-room recordings that do not have
        a head<->meg transform ``info['dev_head_t']``, the MEG coordinate
        frame should be used.
    destination : str | array-like, shape (3,) | None
        The destination location for the head. Can be ``None``, which
        will not change the head position, or a string path to a FIF file
        containing a MEG device<->head transformation, or a 3-element array
        giving the coordinates to translate to (with no rotations).
        For example, ``destination=(0, 0, 0.04)`` would translate the bases
        as ``--trans default`` would in ``maxfilter`` (i.e., to the default
        head location).
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

    Compared to Elekta's implementation of ``maxfilter``, our algorithm
    currently provides the following features:

        * Basic Maxwell filtering
        * Cross-talk cancellation
        * tSSS
        * Bad channel reconstruction
        * Coordinate frame translation

    The following features are not yet implemented:

        * Movement compensation
        * Automatic bad channel detection
        * Regularization of in/out components
        * cHPI subtraction

    Our algorithm has the following enhancements:

        * double floating point precision
        * handling of 3D (in additon to 1D) fine calibration files

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
    if coord_frame not in ('head', 'meg'):
        raise ValueError('coord_frame must be either "head" or "meg", not "%s"'
                         % coord_frame)
    logger.info('Maxwell filtering raw data')
    raw_sss = raw.copy().load_data(verbose=False)
    del raw
    info, times, ch_names = raw_sss.info, raw_sss.times, raw_sss.ch_names

    # Check for T1/T2 mag types
    mag_inds_T1T2 = _get_T1T2_mag_inds(info)
    if len(mag_inds_T1T2) > 0:
        logger.warning('%d T1/T2 magnetometer channel types found. If using '
                       ' SSS, it is advised to replace coil types using '
                       ' `fix_mag_coil_types`.' % len(mag_inds_T1T2))
    if len(info['bads']) > 0:
        logger.info('    Bad channels being reconstructed: %s' % info['bads'])
    else:
        logger.info('    No bad channels')
    #
    # Fine calibration processing (load fine cal and overwrite sensor geometry)
    #
    if fine_cal is not None:
        logger.warning('Fine calibration is experimental despite similar '
                       ' shielding factor to Elekta\'s processing.')
        grad_imbalances, mag_cals = _update_sensor_geometry(info, fine_cal)

    # Get indices of channels to use in multipolar moment calculation
    good_picks = pick_types(info, meg=True, exclude='bads')
    # Get indices of MEG channels
    meg_picks = pick_types(info, meg=True, exclude=[])
    mag_picks = pick_types(info, meg='mag', exclude=[])
    grad_picks = pick_types(info, meg='grad', exclude=[])
    grad_info = pick_info(info, grad_picks)
    if info['dev_head_t'] is None and coord_frame == 'head':
        raise RuntimeError('coord_frame cannot be "head" because '
                           'info["dev_head_t"] is None; if this is an '
                           'empty room recording, consider using '
                           'coord_frame="meg"')
    head_frame = True if coord_frame == 'head' else False
    meg_coils = _prep_meg_channels(info, accurate=True, elekta_defs=True,
                                   head_frame=head_frame, verbose=False)[0]

    # Magnetometers (with coil_class == 1.0) must be scaled by 100 to improve
    # numerical stability as they have different scales than gradiometers
    coil_scale = np.ones((len(meg_picks), 1))
    coil_scale[mag_picks] = 100.

    # Compute multipolar moment bases
    if isinstance(origin, string_types):
        # XXX eventually we could add "auto" mode here
        if origin != 'default':
            raise ValueError('origin must be a numerical array, or "default"')
        origin = (0, 0, 0.04) if coord_frame == 'head' else (0, 13e-3, -6e-3)
    origin = np.array(origin, float)
    if origin.shape != (3,):
        raise ValueError('origin must be a 3-element array')
    # Compute in/out bases and create copies containing only good chs
    S_decomp = _sss_basis(origin, meg_coils, int_order, ext_order)
    # We always want to reconstruct with non-corrected defs
    n_in = _get_n_moments(int_order)
    S_recon = S_decomp[:, :n_in].copy()

    #
    # Cross-talk processing
    #
    if ctc is not None:
        ctc = _read_ctc(ctc, raw_sss.info, meg_picks, good_picks)

    #
    # Fine calibration processing (point-like magnetometers and calib. coeffs)
    #
    if fine_cal is not None:
        # Compute point-like mags to incorporate gradiometer imbalance
        S_fine = _sss_basis_point(origin, grad_info, int_order, ext_order,
                                  grad_imbalances, head_frame=head_frame)
        # Add point like magnetometer data to bases.
        S_decomp[grad_picks, :] += S_fine
        # Scale magnetometers by calibration coefficient
        S_decomp[mag_picks, :] /= mag_cals

    S_decomp_good = S_decomp[good_picks, :]
    S_decomp_good_norm = np.sqrt(np.sum(S_decomp_good *
                                 S_decomp_good, axis=0))[np.newaxis, :]
    S_decomp_good /= S_decomp_good_norm

    # Pseudo-inverse of total multipolar moment basis set (Part of Eq. 37)
    pS_decomp_good = linalg.pinv(S_decomp_good, cond=1e-15)

    # Compute multipolar moments of (magnetometer scaled) data (Eq. 37)
    # XXX eventually we can refactor this to work in chunks
    data = raw_sss[good_picks][0]
    if ctc is not None:
        data = ctc.dot(data)
    mm_norm = np.dot(pS_decomp_good, data * coil_scale[good_picks])
    mm_norm /= S_decomp_good_norm.T

    #
    # Translate to destination frame
    #
    if destination is not None:
        if not head_frame:
            raise RuntimeError('destination can only be set if using the '
                               'head coordinate frame')
        if isinstance(destination, string_types):
            recon_trans = _get_trans(destination, 'meg', 'head')[0]['trans']
        else:
            destination = np.array(destination, float)
            if destination.shape != (3,):
                raise ValueError('destination must be a 3-element vector, '
                                 'str, or None')
            recon_trans = np.eye(4)
            recon_trans[:3, 3] = destination
        info_recon = deepcopy(info)
        info_recon['dev_head_t']['trans'] = recon_trans
        recon_coils = _prep_meg_channels(info_recon,
                                         accurate=True, elekta_defs=True,
                                         head_frame=head_frame,
                                         verbose=False)[0]
        S_recon = _sss_basis(origin, recon_coils, int_order, 0)
        # warn if we have translated too far
        diff = 1000 * (info['dev_head_t']['trans'][:3, 3] -
                       info_recon['dev_head_t']['trans'][:3, 3])
        dist = np.sqrt(np.sum((diff) ** 2))
        if dist > 0.025:
            logger.warning('Head position change is over 25 mm (%s) = %0.1f mm'
                           % (', '.join('%0.1f' % x for x in diff), dist))

    # Reconstruct data from internal space only (Eq. 38), first rescale S_recon
    S_recon /= coil_scale
    raw_sss._data[meg_picks] = np.dot(S_recon, mm_norm[:n_in])

    # Reset 'bads' for any MEG channels since they've been reconstructed
    bad_inds = [ch_names.index(ch) for ch in info['bads']]
    info['bads'] = [ch_names[bi] for bi in bad_inds if bi not in meg_picks]

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
        for start, stop in zip(lims[:-1], lims[1:]):
            # Reconstruct data from external space and compute residual
            resid = data[:, start:stop]
            resid -= raw_sss._data[meg_picks, start:stop]
            resid -= np.dot(S_decomp[:, n_in:],
                            mm_norm[n_in:, start:stop]) / coil_scale
            _check_finite(resid)

            # Compute SSP-like projection vectors based on minimal correlation
            this_data = raw_sss._data[meg_picks, start:stop]
            _check_finite(this_data)
            V = _overlap_projector(this_data, resid, st_corr)

            # Apply projector according to Eq. 12 in [2]_
            logger.info('    Projecting out %s tSSS components for %s-%s'
                        % (V.shape[1], start / raw_sss.info['sfreq'],
                           stop / raw_sss.info['sfreq']))
            this_data -= np.dot(np.dot(this_data, V), V.T)
            raw_sss._data[meg_picks, start:stop] = this_data

    # Update info
    raw_sss = _update_sss_info(raw_sss, origin, int_order, ext_order,
                               len(good_picks), coord_frame)

    return raw_sss


def _check_finite(data):
    """Helper to ensure data is finite"""
    if not np.isfinite(data).all():
        raise RuntimeError('data contains non-finite numbers')


def _sph_harm_norm(order, degree):
    """Normalization factor for spherical harmonics"""
    # we could use scipy.special.poch(degree + order + 1, -2 * order)
    # here, but it's slower for our fairly small degree
    # do in two steps like scipy for better precision
    norm = np.sqrt((2 * degree + 1.) / (4 * np.pi))
    norm *= np.sqrt(factorial(degree - order) /
                    float(factorial(degree + order)))
    return norm


def _sph_harm(order, degree, az, pol, norm=True):
    """Evaluate point in specified multipolar moment. [1]_ Equation 4.

    When using, pay close attention to inputs. Spherical harmonic notation for
    order/degree, and theta/phi are both reversed in original SSS work compared
    to many other sources. See mathworld.wolfram.com/SphericalHarmonic.html for
    more discussion.

    Note that scipy has ``scipy.special.sph_harm``, but that function is
    too slow on old versions (< 0.15) for heavy use.

    Parameters
    ----------
    order : int
        Order of spherical harmonic. (Usually) corresponds to 'm'.
    degree : int
        Degree of spherical harmonic. (Usually) corresponds to 'l'.
    az : float
        Azimuthal (longitudinal) spherical coordinate [0, 2*pi]. 0 is aligned
        with x-axis.
    pol : float
        Polar (or colatitudinal) spherical coordinate [0, pi]. 0 is aligned
        with z-axis.
    norm : bool
        If True, include normalization factor.

    Returns
    -------
    base : complex float
        The spherical harmonic value.
    """
    from scipy.special import lpmv

    # Error checks
    if np.abs(order) > degree:
        raise ValueError('Absolute value of order must be <= degree')
    # Ensure that polar and azimuth angles are arrays
    az = np.asarray(az)
    pol = np.asarray(pol)
    if (np.abs(az) > 2 * np.pi).any():
        raise ValueError('Azimuth coords must lie in [-2*pi, 2*pi]')
    if(pol < 0).any() or (pol > np.pi).any():
        raise ValueError('Polar coords must lie in [0, pi]')
    # This is the "seismology" convention on Wikipedia, w/o Condon-Shortley
    if norm:
        norm = _sph_harm_norm(order, degree)
    else:
        norm = 1.
    return norm * lpmv(order, degree, np.cos(pol)) * np.exp(1j * order * az)


def _sss_basis(origin, coils, int_order, ext_order, mag_scale=100.):
    """Compute SSS basis for given conditions.

    Parameters
    ----------
    origin : ndarray, shape (3,)
        Origin of the multipolar moment space in millimeters
    coils : list
        List of MEG coils. Each should contain coil information dict specifying
        position, normals, weights, number of integration points and channel
        type. All coil geometry must be in the same coordinate frame
        as ``origin`` (``head`` or ``meg``).
    int_order : int
        Order of the internal multipolar moment space
    ext_order : int
        Order of the external multipolar moment space
    scale : float
        Scale factor for magnetometers.

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
    coil_scale[np.array([coil['coil_class'] == FIFF.FWD_COILC_MAG
                         for coil in coils])] = mag_scale

    if n_bases > n_sens:
        raise ValueError('Number of requested bases (%s) exceeds number of '
                         'sensors (%s)' % (str(n_bases), str(n_sens)))

    # Compute position vector between origin and coil integration pts
    cvec_cart = r_int_pts - origin[np.newaxis, :]
    # Convert points to spherical coordinates
    rad, az, pol = _cart_to_sph(cvec_cart).T

    # Compute internal/external basis vectors (exclude degree 0; L/RHS Eq. 5)
    for degree in range(1, max(int_order, ext_order) + 1):
        # Only loop over positive orders, negative orders are handled
        # for efficiency within
        for order in range(degree + 1):
            S_in_out = list()
            grads_in_out = list()
            # Same spherical harmonic is used for both internal and external
            sph = _sph_harm(order, degree, az, pol, norm=False)
            sph_norm = _sph_harm_norm(order, degree)
            sph *= sph_norm
            # Compute complex gradient for all integration points
            # in spherical coordinates (Eq. 6). The gradient for rad, az, pol
            # is obtained by taking the partial derivative of Eq. 4 w.r.t. each
            # coordinate.
            az_factor = 1j * order * sph / np.sin(pol)
            pol_factor = (-sph_norm * np.sin(pol) * np.exp(1j * order * az) *
                          _alegendre_deriv(order, degree, np.cos(pol)))
            if degree <= int_order:
                S_in_out.append(S_in)
                in_norm = rad ** -(degree + 2)
                g_rad = in_norm * (-(degree + 1.) * sph)
                g_az = in_norm * az_factor
                g_pol = in_norm * pol_factor
                grads_in_out.append(_sph_to_cart_partials(az, pol,
                                                          g_rad, g_az, g_pol))
            if degree <= ext_order:
                S_in_out.append(S_out)
                out_norm = rad ** (degree - 1)
                g_rad = out_norm * degree * sph
                g_az = out_norm * az_factor
                g_pol = out_norm * pol_factor
                grads_in_out.append(_sph_to_cart_partials(az, pol,
                                                          g_rad, g_az, g_pol))
            for spc, grads in zip(S_in_out, grads_in_out):
                # We could convert to real at the end, but it's more efficient
                # to do it now
                grads_pos_neg = [_sh_complex_to_real(grads, order)]
                orders_pos_neg = [order]
                # Deal with the negative orders
                if order > 0:
                    # it's faster to use the conjugation property for
                    # our normalized spherical harmonics than recalculate...
                    grads_pos_neg.append(_sh_complex_to_real(
                        _sh_negate(grads, order), -order))
                    orders_pos_neg.append(-order)
                for gr, oo in zip(grads_pos_neg, orders_pos_neg):
                    # Gradients dotted w/integration point normals and weighted
                    gr = wcoils * np.einsum('ij,ij->i', gr, ncoils, order='C')
                    # For order/degree, sum over each sensor's integration pts
                    # for pt_i in range(0, len(int_lens) - 1):
                    #  int_pts_sum = \
                    #    np.sum(all_grads[int_lens[pt_i]:int_lens[pt_i + 1]])
                    #  spc[pt_i, deg ** 2 + deg + oo - 1] = int_pts_sum
                    vals = np.bincount(bins, weights=gr, minlength=len(counts))
                    spc[:, _deg_order_idx(degree, oo)] = -vals

    # Scale magnetometers
    S_tot *= coil_scale[:, np.newaxis]
    return S_tot


def _deg_order_idx(deg, order):
    """Helper to get the index into S_in or S_out given a degree and order"""
    return deg ** 2 + deg + order - 1


def _alegendre_deriv(order, degree, val):
    """Compute the derivative of the associated Legendre polynomial at a value.

    Parameters
    ----------
    order : int
        Order of spherical harmonic. (Usually) corresponds to 'm'.
    degree : int
        Degree of spherical harmonic. (Usually) corresponds to 'l'.
    val : float
        Value to evaluate the derivative at.

    Returns
    -------
    dPlm : float
        Associated Legendre function derivative
    """
    from scipy.special import lpmv
    assert order >= 0
    return (order * val * lpmv(order, degree, val) + (degree + order) *
            (degree - order + 1.) * np.sqrt(1. - val * val) *
            lpmv(order - 1, degree, val)) / (1. - val * val)


def _sh_negate(sh, order):
    """Helper to get the negative spherical harmonic from a positive one"""
    assert order >= 0
    return sh.conj() * (-1. if order % 2 else 1.)  # == (-1) ** order


def _sh_complex_to_real(sh, order):
    """Helper function to convert complex to real basis functions.

    Parameters
    ----------
    sh : array-like
        Spherical harmonics. Must be from order >=0 even if negative orders
        are used.
    order : int
        Order (usually 'm') of multipolar moment.

    Returns
    -------
    real_sh : array-like
        The real version of the spherical harmonics.

    Notes
    -----
    This does not include the Condon-Shortely phase.
    """

    if order == 0:
        return np.real(sh)
    else:
        return np.sqrt(2.) * (np.real if order > 0 else np.imag)(sh)


def _sh_real_to_complex(shs, order):
    """Convert real spherical harmonic pair to complex

    Parameters
    ----------
    shs : ndarray, shape (2, ...)
        The real spherical harmonics at ``[order, -order]``.
    order : int
        Order (usually 'm') of multipolar moment.

    Returns
    -------
    sh : array-like, shape (...)
        The complex version of the spherical harmonics.
    """
    if order == 0:
        return shs[0]
    else:
        return (shs[0] + 1j * np.sign(order) * shs[1]) / np.sqrt(2.)


def _bases_complex_to_real(complex_tot, int_order, ext_order):
    """Convert complex spherical harmonics to real"""
    n_in, n_out = _get_n_moments([int_order, ext_order])
    complex_in = complex_tot[:, :n_in]
    complex_out = complex_tot[:, n_in:]
    real_tot = np.empty(complex_tot.shape, np.float64)
    real_in = real_tot[:, :n_in]
    real_out = real_tot[:, n_in:]
    for comp, real, exp_order in zip([complex_in, complex_out],
                                     [real_in, real_out],
                                     [int_order, ext_order]):
        for deg in range(1, exp_order + 1):
            for order in range(deg + 1):
                idx_pos = _deg_order_idx(deg, order)
                idx_neg = _deg_order_idx(deg, -order)
                real[:, idx_pos] = _sh_complex_to_real(comp[:, idx_pos], order)
                if order != 0:
                    # This extra mult factor baffles me a bit, but it works
                    # in round-trip testing, so we'll keep it :(
                    mult = (-1 if order % 2 == 0 else 1)
                    real[:, idx_neg] = mult * _sh_complex_to_real(
                        comp[:, idx_neg], -order)
    return real_tot


def _bases_real_to_complex(real_tot, int_order, ext_order):
    """Convert real spherical harmonics to complex"""
    n_in, n_out = _get_n_moments([int_order, ext_order])
    real_in = real_tot[:, :n_in]
    real_out = real_tot[:, n_in:]
    comp_tot = np.empty(real_tot.shape, np.complex128)
    comp_in = comp_tot[:, :n_in]
    comp_out = comp_tot[:, n_in:]
    for real, comp, exp_order in zip([real_in, real_out],
                                     [comp_in, comp_out],
                                     [int_order, ext_order]):
        for deg in range(1, exp_order + 1):
            # only loop over positive orders, figure out neg from pos
            for order in range(deg + 1):
                idx_pos = _deg_order_idx(deg, order)
                idx_neg = _deg_order_idx(deg, -order)
                this_comp = _sh_real_to_complex([real[:, idx_pos],
                                                 real[:, idx_neg]], order)
                comp[:, idx_pos] = this_comp
                comp[:, idx_neg] = _sh_negate(this_comp, order)
    return comp_tot


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


def _sph_to_cart_partials(az, pol, g_rad, g_az, g_pol):
    """Convert spherical partial derivatives to cartesian coords.

    Note: Because we are dealing with partial derivatives, this calculation is
    not a static transformation. The transformation matrix itself is dependent
    on azimuth and polar coord.

    See the 'Spherical coordinate sytem' section here:
    wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates

    Parameters
    ----------
    az : ndarray, shape (n_points,)
        Array containing spherical coordinates points (azimuth).
    pol : ndarray, shape (n_points,)
        Array containing spherical coordinates points (polar).
    sph_grads : ndarray, shape (n_points, 3)
        Array containing partial derivatives at each spherical coordinate
        (radius, azimuth, polar).

    Returns
    -------
    cart_grads : ndarray, shape (n_points, 3)
        Array containing partial derivatives in Cartesian coordinates (x, y, z)
    """
    sph_grads = np.c_[g_rad, g_az, g_pol]
    cart_grads = np.zeros_like(sph_grads)
    c_as, s_as = np.cos(az), np.sin(az)
    c_ps, s_ps = np.cos(pol), np.sin(pol)
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


def _update_sss_info(raw, origin, int_order, ext_order, nsens, coord_frame):
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
                         frame=_str_to_frame[coord_frame],
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
    n = np.sqrt(np.sum(data_int * data_int))
    Q_int = linalg.qr(_orth_overwrite((data_int / n).T),
                      overwrite_a=True, mode='economic', **check_disable)[0].T
    n = np.sqrt(np.sum(data_res * data_res))
    Q_res = linalg.qr(_orth_overwrite((data_res / n).T),
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


def _read_fine_cal(fine_cal):
    """Read sensor locations and calib. coeffs from fine calibration file."""

    # Read new sensor locations
    cal_chs = list()
    with open(fine_cal, 'r') as fid:
        lines = [line for line in fid if line[0] not in '#\n']
        for line in lines:
            # `vals` contains channel number, (x, y, z), x-norm 3-vec, y-norm
            # 3-vec, z-norm 3-vec, and (1 or 3) imbalance terms
            vals = np.fromstring(line, sep=' ').astype(np.float64)

            # Check for correct number of items
            if len(vals) not in [14, 16]:
                raise RuntimeError('Error reading fine calibration file')

            ch_name = 'MEG' + '%04d' % vals[0]  # Zero-pad names to 4 char

            # Get orientation information for coil transformation
            loc = vals[1:13].copy()  # Get orientation information for 'loc'
            calib_coeff = vals[13:].copy()  # Get imbalance/calibration coeff
            cal_chs.append(dict(ch_name=ch_name,
                                loc=loc, calib_coeff=calib_coeff,
                                coord_frame=FIFF.FIFFV_COORD_DEVICE))
    return cal_chs


def _update_sensor_geometry(info, fine_cal):
    """Helper to replace sensor geometry information and reorder cal_chs"""
    logger.info('    Using fine calibration %s' % op.basename(fine_cal))
    cal_chs = _read_fine_cal(fine_cal)

    # Check that we ended up with correct channels
    meg_info = pick_info(info, pick_types(info, meg=True, exclude=[]))
    order = pick_channels([c['ch_name'] for c in cal_chs],
                          meg_info['ch_names'])
    if not (len(cal_chs) == meg_info['nchan'] == len(order)):
        raise RuntimeError('Number of channels in fine calibration file (%i) '
                           'does not equal number of channels in info (%i)' %
                           (len(cal_chs), info['nchan']))
    # ensure they're ordered like our data
    cal_chs = [cal_chs[ii] for ii in order]

    # Replace sensor locations (and track differences) for fine calibration
    ang_shift = np.zeros((len(cal_chs), 3))
    used = np.zeros(len(info['chs']), bool)
    for ci, cal_ch in enumerate(cal_chs):
        idx = info['ch_names'].index(cal_ch['ch_name'])
        assert not used[idx]
        used[idx] = True
        info_ch = info['chs'][idx]

        # calculate shift angle
        v1 = _loc_to_coil_trans(cal_ch['loc'])[:3, :3]
        _normalize_vectors(v1)
        v2 = _loc_to_coil_trans(info_ch['loc'])[:3, :3]
        _normalize_vectors(v2)
        ang_shift[ci] = np.sum(v1 * v2, axis=0)

        # Adjust channel normal orientations with those from fine calibration
        # Channel positions are not changed
        info_ch['loc'][3:] = cal_ch['loc'][3:]
        assert (info_ch['coord_frame'] == cal_ch['coord_frame'] ==
                FIFF.FIFFV_COORD_DEVICE)

    # Deal with numerical precision giving absolute vals slightly more than 1.
    np.clip(ang_shift, -1., 1., ang_shift)
    np.rad2deg(np.arccos(ang_shift), ang_shift)  # Convert to degrees

    # Log quantification of sensor changes
    logger.info('    Fine calibration adjusted coil positions by (μ ± σ): '
                '%0.1f° ± %0.1f° (max: %0.1f°)' %
                (np.mean(ang_shift), np.std(ang_shift),
                 np.max(np.abs(ang_shift))))

    # Determine gradiometer imbalances and magnetometer calibrations
    grad_picks = pick_types(info, meg='grad', exclude=[])
    mag_picks = pick_types(info, meg='mag', exclude=[])
    grad_imbalances = np.array([cal_chs[ii]['calib_coeff']
                                for ii in grad_picks]).T
    mag_cals = np.array([cal_chs[ii]['calib_coeff'] for ii in mag_picks])
    return grad_imbalances, mag_cals


def _sss_basis_point(origin, info, int_order, ext_order, imbalances,
                     head_frame=True):
    """Compute multipolar moments for point-like magnetometers (in fine cal)"""

    # Construct 'coils' with r, weights, normal vecs, # integration pts, and
    # channel type.
    if imbalances.shape[0] not in [1, 3]:
        raise ValueError('Must have 1 (x) or 3 (x, y, z) point-like ' +
                         'magnetometers. Currently have %i' %
                         imbalances.shape[0])

    # Coil_type values for x, y, z point magnetometers
    # Note: 1D correction files only have x-direction corrections
    pt_types = [FIFF.FIFFV_COIL_POINT_MAGNETOMETER_X,
                FIFF.FIFFV_COIL_POINT_MAGNETOMETER_Y,
                FIFF.FIFFV_COIL_POINT_MAGNETOMETER]

    # Loop over all coordinate directions desired and create point mags
    S_tot = 0.
    for imb, pt_type in zip(imbalances, pt_types):
        temp_info = deepcopy(info)
        for ch in temp_info['chs']:
            ch['coil_type'] = pt_type
        coils_add = _prep_meg_channels(
            temp_info, accurate=True, elekta_defs=True, head_frame=head_frame,
            verbose=False)[0]
        # Scale spaces by gradiometer imbalance
        S_add = _sss_basis(origin, coils_add, int_order,
                           ext_order) * imb[:, np.newaxis]
        S_tot += S_add

    # Return point-like mag bases
    return S_tot
