"""Non-Linear sparse inverse solvers."""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: Simplified BSD

from .mxne_inverse import (mixed_norm, tf_mixed_norm,
                           make_stc_from_dipoles)
from ._gamma_map import gamma_map
