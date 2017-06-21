"""Non-Linear sparse inverse solvers."""

# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: Simplified BSD

from .mxne_inverse import mixed_norm, tf_mixed_norm
from ._gamma_map import gamma_map
