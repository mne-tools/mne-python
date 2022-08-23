"""NIRS specific preprocessing functions."""

# Authors: Robert Luke <mail@robertluke.net>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause

from .nirs import (short_channels, source_detector_distances,
                   _check_channels_ordered, _channel_frequencies,
                   _fnirs_spread_bads, _channel_chromophore,
                   _validate_nirs_info, _fnirs_optode_names, _optode_position,
                   _reorder_nirx)
from ._optical_density import optical_density
from ._beer_lambert_law import beer_lambert_law
from ._scalp_coupling_index import scalp_coupling_index
from ._tddr import temporal_derivative_distribution_repair, tddr
