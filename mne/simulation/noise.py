# author: ngayraud
#
# Created on Fri Mar 16 10:31:51 2018.

import warnings
import numpy as np

from ..cov import make_ad_hoc_cov, read_cov, make_custom_cov
from ..externals.six import string_types
from ..io.pick import pick_channels_cov
from ..utils import check_random_state

from mne.cov import Covariance


def _check_cov(info, cov):
    """Check that the user provided a valid covariance matrix for the noise."""
    if isinstance(cov, string_types):
        if cov == 'simple':
            cov = make_ad_hoc_cov(info, verbose=False)
        else:
            cov = read_cov(cov, verbose=False)
    elif isinstance(cov, Covariance):
        pass
    elif isinstance(cov, dict):
        cov = make_custom_cov(info, cov, verbose=False)
    else:
        raise ValueError('Covariance Matrix type not recognized. Valid input'
                         'types are: instance of Covariance, array'
                         'string(covariance filename | \'simple\'')
    return cov


def generate_noise_data(info, cov, n_samples, random_state, iir_filter=None,
                        zi=None):
    """Create spatially colored and temporally IIR-filtered noise.

    Parameters
    ----------
    cov : instance of Covariance | str
        The sensor covariance matrix used to generate noise.
        If 'simple', a basic (diagonal) ad-hoc noise covariance will be used.
        If a string (filename), then the covariance will be loaded.
        If dict, a covariance matrix will be generated from it.
    """
    from scipy.signal import lfilter

    # All checks here
    cov = _check_cov(info, cov)
    rng = check_random_state(random_state)

    noise_cov = pick_channels_cov(cov, include=info['ch_names'], exclude=[])

    if set(info['ch_names']) != set(noise_cov.ch_names):
        raise ValueError('Info and covariance channel names are not '
                         'identical. Cannot generate the noise matrix. '
                         'Channels missing in covariance %s.' %
                         np.setdiff1d(info['ch_names'], noise_cov.ch_names))

    noise_cov['data'] = (np.diag(noise_cov.data) if noise_cov['diag'] else
                         noise_cov.data)

    # Parameters
    n_channels = len(noise_cov.data)
    mu_channels = np.zeros(n_channels)

    # Generate the noise
    # we almost always get a positive semidefinite warning here, so squash it
    with warnings.catch_warnings(record=True):
        noise = rng.multivariate_normal(mu_channels, noise_cov.data,
                                        n_samples).T

    # Apply the filter if any
    if iir_filter is not None:
        if zi is None:
            zi = np.zeros((n_channels, len(iir_filter) - 1))
        noise, zf = lfilter([1], iir_filter, noise, axis=-1, zi=zi)
    else:
        zf = None

    return noise, zf
