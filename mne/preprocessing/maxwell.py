# Authors: Mark Wronkiewicz <wronk.mark@gmail.com>
#          Jussi Nurminen
#
# License: BSD (3-clause)

#TODO: write in equation numbers from Samu's paper

import numpy as np
from scipy.special import sph_harm, lpmv
from scipy.misc import factorial
from ..forward._lead_dots import (_get_legen_table, _get_legen_lut_fast,
                                  _get_legen_lut_accurate)
from ..fixes import partial
from ..transforms import _cartesian_to_sphere as cart_to_sph


def maxwell_filter(Raw):
    pass


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


def legendre(theta):
    """
    """


def grad_in_comp(l, m, r, theta, phi):
    """
    Compute gradient of LHS of V(r) spherical expansion having form
    Ylm(theta, phi) / (r ** (l+1))
    """

    # Get function for Legendre derivatives
    lut, n_fact = _get_legen_table('meg', False, 100)
    lut_deriv_fun = partial(_get_legen_lut_accurate, lut=lut)

    # Compute gradients for r, theta, and phi
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
    Compute gradient of RHS of V(r) spherical expansion having form
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

    if m > 0:
        grad_vec = np.sqrt(2) * np.real(grad_vec_raw)
    elif m < 0:
        grad_vec = np.sqrt(2) * np.imag(grad_vec_raw)
    else:
        grad_vec = grad_vec_raw

    # Convert to rectanglar coords
    return cart_to_sph(grad_vec[:, 0], grad_vec[:, 1], grad_vec[:, 2])
