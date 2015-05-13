# Authors: Mark Wronkiewicz <wronk.mark@gmail.com>
#          Jussi Nurminen
#
# License: BSD (3-clause)

#TODO: write in equation numbers from Samu's paper(s)
#XXX: Why do any relative imports? Why not mne.package
import numpy as np
from scipy.special import sph_harm, lpmv
from scipy.misc import factorial
from mne.forward._lead_dots import (_get_legen_table, _get_legen_lut_fast,
                                    _get_legen_lut_accurate)
from ..fixes import partial
from ..transforms import _sphere_to_cartesian as sph_to_cart
from ..transforms import _cartesian_to_sphere as cart_to_sph


def sph_harmonic(l, m, theta, phi):
    """
    Compute the spherical harmonic function at point in spherical coordinates.

    Parameters
    ----------
    l : int
        Order of spherical harmonic (indicating spatial frequency complexity)
    m : int
        Expansion coefficient
    theta : float
        Azimuth spherical coordinate (?)
    phi : float
        Elevation spherical coordinate (?)

    Returns
    -------
    y : complex float
        The spherical harmonic value at the specified theta and phi
    """
    assert np.abs(m) <= l, 'Absolute value of expansion coefficient must be <= l'

    # Get function for Legendre derivatives
    lut, n_fact = _get_legen_table('meg', False, 100)
    lut_deriv_fun = partial(_get_legen_lut_accurate, lut=lut)

    #TODO: Should factorial function use floating or long precision?

    # Real valued spherical expansion as given by Wikso, (11)
    # TODO fix look up of Legendre value
    base = np.sqrt(((2 * l + 1) / (4 * np.pi) *
                    factorial(l - m) / factorial(l + m)) *
                   lut_deriv_fun(l, m, np.cos(theta)))
    if m < 0:
        return base * np.sin(m * phi)
    else:
        return base * np.cos(m * phi)

    #TODO: Check how fast taking the real part of scipy's sph harmonic function is


def get_num_harmonics(L_in, L_out):
    """
    Compute total number of spherical harmonics.

    Parameters
    ---------
    L_in : int
        Spherical harmonic expansion order of 'in' space.
    L_out : int
        Spherical harmonic expansion order of 'out' space.

    Returns
    -------
    M : int
        Total number of spherical harmonics
    """

    return L_in ** 2 + 2 * L_in + L_out ** 2 + 2 * L_out


def grad_in_comp(l, m, r, theta, phi):
    """
    Compute gradient of in-component of V(r) spherical expansion having form
    Ylm(theta, phi) / (r ** (l+1))
    """

    # Get function for Legendre derivatives
    lut, n_fact = _get_legen_table('meg', False, 100)
    lut_deriv_fun = partial(_get_legen_lut_accurate, lut=lut)

    # Compute gradients for r, theta, and phi
    # TODO: speed up by computing sph_harmonic only once and passing twice
    r1 = -(l + 1) / r ** (l + 2) * sph_harmonic(l, m, theta, phi)

    theta1 = 1 / r ** (l + 2) * np.sqrt((2 * l + 1) * factorial(l - m) /
                                        (4 * np.pi * factorial(l + m))) * \
        -np.sin(theta) * lut_deriv_fun(l, m, np.cos(theta)) * np.exp(1j * m * phi)

    phi1 = 1 / (r ** (l + 2) * np.sin(theta)) * 1j * m * \
        sph_harmonic(l, m, theta, phi)

    # Get real component of vectors, convert to cartesian coords, and return
    return _to_real_and_cart(np.concatenate((r1, theta1, phi1)), m)


def grad_out_comp(l, m, r, theta, phi):
    """
    Compute gradient of out-component of V(r) spherical expansion having form
    Ylm(theta, phi) * (r ** l)
    """

    # Get function for Legendre derivatives
    lut, n_fact = _get_legen_table('meg', False, 100)
    lut_deriv_fun = partial(_get_legen_lut_accurate, lut=lut)

    # Compute gradients for r, theta, and phi
    r1 = l * r ** (l - 1) * sph_harmonic(l, m, theta, phi)

    theta1 = r ** (l - 1) * np.sqrt((2 * l + 1) * factorial(l - m) /
                                    (4 * np.pi * factorial(1 + m))) * \
        -np.sin(theta) * lut_deriv_fun(l, m, np.cos(theta)) * np.exp(1j * m * phi)

    phi1 = r ** (l - 1) / np.sin(theta) * 1j * m * sph_harmonic(l, m, theta, phi)

    # Get real component of vectors, convert to cartesian coords, and return
    return _to_real_and_cart(np.concatenate((r1, theta1, phi1)), m)


def _to_real_and_cart(grad_vec_raw, m):
    """
    Helper function to take real component of gradient vector and convert from
    spherical to cartesian coords
    """

    # Get real component of gradient based on m
    if m > 0:
        grad_vec = np.sqrt(2) * np.real(grad_vec_raw)
    elif m < 0:
        grad_vec = np.sqrt(2) * np.imag(grad_vec_raw)
    else:
        grad_vec = grad_vec_raw

    # Convert from spherical to rectanglar coords
    return sph_to_cart(grad_vec[:, 0], grad_vec[:, 1], grad_vec[:, 2])


def sss_basis(h0_int, h0_ext, L_in, L_out, Ipm, Iam, Ipind, Snaug):
    """
    Compute SSS basis vectors
    """
    Np = Ipm.shape[0]
    n_sensors = Ipind.shape[0]
    n_vectors = (L_in + 1) ** 2 - 1 + (L_out + 1) ** 2 - 1

    # Initialize internal and external SSS basis vectors
    A = np.zeros((n_sensors, (L_in + 1) ** 2 - 1))
    B = np.zeros((n_sensors, (L_out + 1) ** 2 - 1))

    if n_vectors > n_sensors:
        print('Too many SSS basis vectors requested. %s vectors requested',
              ' exceeds %s sensors.' % (n_vectors, n_sensors))

    for l in range(1, L_in):
        for m in range(-l, l):
            c_vec = Ipm - np.ones((Np, 1)) * h0_int
            r1, theta1, phi1 = cart_to_sph(c_vec[:, 0], c_vec[:, 1], c_vec[:, 2])

            # TODO, dot over only 1 dimension
            a1_all = Iam * np.dot(-grad_in_comp(l, m, r1, theta1, phi1), Snaug)
            a1 = np.sum(a1_all, 1)  # XXX Check that this sum is correct
            A[:, l ** 2 + l + m] = a1  # TODO, inds might be off by 1

    for l in range(1, L_out):
        for m in range(-l, l):
            c_vec = Ipm - np.ones((Np, 1)) * h0_ext  # XXX h0_int in Jussi's code (mistake?)
            r1, theta1, phi1 = cart_to_sph(c_vec[:, 0], c_vec[:, 1], c_vec[:, 2])

            # TODO, dot over only 1 dimension
            b1_all = Iam * np.dot(-grad_out_comp(l, m, r1, theta1, phi1), Snaug)
            b1 = np.sum(b1_all, 1)  # XXX Check that this sum is correct with Jussi's code
            B[:, l ** 2 + l + m] = b1


def maxwell_filter(Raw):
    pass
