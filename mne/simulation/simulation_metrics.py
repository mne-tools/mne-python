
import numpy as np


def _check_stc(stc1, stc2):
    # XXX What should we check? that the data is having the same size?
    if stc1.data.shape != stc2.data.shape:
        raise ValueError('data in stcs must have the same size')


def source_estimate_quantification(stc1, stc2, metric='rms'):

    _check_stc(stc1, stc2)

    if metric == 'rms':
        return np.mean((stc1.data - stc2.data) ** 2)
