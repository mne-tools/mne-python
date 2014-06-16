""" Utils that are implemented in scikit-learn.
"""

from __future__ import absolute_import

import sklearn.utils.extmath as extmath

if hasattr(extmath, 'cartesian_'):
    cartesian = extmath.cartesian
else:
    # fall back to builtin cartesian since older
    # sklearn versions don't have this function.
    from ..builtin import utils
    cartesian = utils.cartesian

