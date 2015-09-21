# Authors: Mark Wronkiewicz <wronk.mark@gmail.com>
#          Jussi Nurminen <jnu@iki.fi>


# License: BSD (3-clause)

from __future__ import division
import numpy as np
from scipy.linalg import pinv
from math import factorial

from .. import pick_types
from ..forward._compute_forward import _concatenate_coils
from ..forward._make_forward import _prep_meg_channels
from ..io.write import _generate_meas_id, _date_now
from ..utils import logger, verbose


@verbose
def maxwell_filter(raw, origin=(0, 0, 40), int_order=8, ext_order=3,
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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose)

    Returns
    -------
    raw_sss : instance of mne.io.Raw
        The raw data with Maxwell filtering applied

    Notes
    -----
    .. versionadded:: 0.10

    Equation numbers refer to Taulu and Kajola, 2005 [1]_.

    This code was adapted and relicensed (with BSD form) with permission from
    Jussi Nurminen.

    References
    ----------
    .. [1] Taulu S. and Kajola M. "Presentation of electromagnetic
           multichannel data: The signal space separation method,"
           Journal of Applied Physics, vol. 97, pp. 124905 1-10, 2005.

           http://lib.tkk.fi/Diss/2008/isbn9789512295654/article2.pdf
    """

    # There are an absurd number of different possible notations for spherical
    # coordinates, which confounds the notation for spherical harmonics.  Here,
    # we purposefully stay away from shorthand notation in both and use
    # explicit terms (like 'azimuth' and 'polar') to avoid confusion.
    # See mathworld.wolfram.com/SphericalHarmonic.html for more discussion.
    # Our code follows the same standard that ``scipy`` uses for ``sph_harm``.

    if raw.proj:
        raise RuntimeError('Projectors cannot be applied to raw data.')
    if 'dev_head_t' not in raw.info.keys():
        raise RuntimeError("Raw.info must contain 'dev_head_t' to transform "
                           "device to head coords")
    if len(raw.info.get('comps', [])) > 0:
        raise RuntimeError('Maxwell filter cannot handle compensated '
                           'channels.')
    logger.info('Bad channels being reconstructed: ' + str(raw.info['bads']))

    raw.preload_data()

    # Get indices of channels to use in multipolar moment calculation
    good_chs = pick_types(raw.info, meg=True, exclude='bads')
    # Get indices of MEG channels
    meg_chs = pick_types(raw.info, meg=True, exclude=[])
    data, _ = raw[good_chs, :]

    meg_coils, _, _, meg_info = _prep_meg_channels(raw.info, accurate=True,
                                                   elekta_defs=True)

    # Magnetometers (with coil_class == 1.0) must be scaled by 100 to improve
    # numerical stability as they have different scales than gradiometers
    coil_scale = np.ones(len(meg_coils))
    coil_scale[np.array([coil['coil_class'] == 1.0
                         for coil in meg_coils])] = 100.

    # Compute multipolar moment bases
    origin = np.array(origin) / 1000.  # Convert scale from mm to m
    # Compute in/out bases and create copies containing only good chs
    S_in, S_out = _sss_basis(origin, meg_coils, int_order, ext_order)

    S_in_good, S_out_good = S_in[good_chs, :], S_out[good_chs, :]
    S_in_good_norm = np.sqrt(np.sum(S_in_good * S_in_good, axis=0))[:,
                                                                    np.newaxis]

    # Pseudo-inverse of total multipolar moment basis set (Part of Eq. 37)
    S_tot_good = np.c_[S_in_good, S_out_good]
    S_tot_good /= np.sqrt(np.sum(S_tot_good * S_tot_good, axis=0))[np.newaxis,
                                                                   :]
    pS_tot_good = pinv(S_tot_good, cond=1e-15)

    # Compute multipolar moments of (magnetometer scaled) data (Eq. 37)
    mm = np.dot(pS_tot_good, data * coil_scale[good_chs][:, np.newaxis])

    # Reconstruct data from internal space (Eq. 38)
    recon = np.dot(S_in, mm[:S_in.shape[1], :] / S_in_good_norm)

    # Return reconstructed raw file object
    raw_sss = _update_sss_info(raw.copy(), origin, int_order, ext_order,
                               data.shape[0])
    raw_sss._data[meg_chs, :] = recon / coil_scale[:, np.newaxis]

    # Reset 'bads' for any MEG channels since they've been reconstructed
    bad_inds = [raw_sss.info['ch_names'].index(ch)
                for ch in raw_sss.info['bads']]
    raw_sss.info['bads'] = [raw_sss.info['ch_names'][bi] for bi in bad_inds
                            if bi not in meg_chs]

    return raw_sss


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
        List of MEG coils. Each should contain coil information dict. All
        position info must be in the same coordinate frame as 'origin'
    int_order : int
        Order of the internal multipolar moment space
    ext_order : int
        Order of the external multipolar moment space

    Returns
    -------
    bases: tuple, len (2)
        Internal and external basis sets ndarrays with shape
        (n_coils, n_mult_moments)
    """
    r_int_pts, ncoils, wcoils, counts = _concatenate_coils(coils)
    bins = np.repeat(np.arange(len(counts)), counts)
    n_sens = len(counts)
    n_bases = get_num_moments(int_order, ext_order)
    # int_lens = np.insert(np.cumsum(counts), obj=0, values=0)

    S_in = np.empty((n_sens, (int_order + 1) ** 2 - 1))
    S_out = np.empty((n_sens, (ext_order + 1) ** 2 - 1))
    S_in.fill(np.nan)
    S_out.fill(np.nan)

    # Set all magnetometers (with 'coil_type' == 1.0) to be scaled by 100
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

    return S_in, S_out


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
    """Helper function to convert gradient vector to to real basis functions.

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


def get_num_moments(int_order, ext_order):
    """Compute total number of multipolar moments. Equivalent to [1]_ Eq. 32.

    Parameters
    ----------
    int_order : int
        Internal expansion order
    ext_order : int
        External expansion order

    Returns
    -------
    M : int
        Total number of multipolar moments
    """

    # TODO: Eventually, reuse code in field_interpolation

    return int_order ** 2 + 2 * int_order + ext_order ** 2 + 2 * ext_order


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

    return np.c_[rad, az, pol]


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
    from .. import __version__
    # TODO: Continue to fill out bookkeeping info as additional features
    # are added (fine calibration, cross-talk calibration, etc.)
    int_moments = get_num_moments(int_order, 0)
    ext_moments = get_num_moments(0, ext_order)

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
    if 'proc_history' in raw.info.keys():
        raw.info['proc_history'].insert(0, proc_block)
    else:
        raw.info['proc_history'] = [proc_block]

    return raw
