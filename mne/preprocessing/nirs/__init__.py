"""NIRS specific preprocessing functions."""

# Authors: Robert Luke <mail@robertluke.net>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

from .nirs import short_channels, source_detector_distances, _check_channels_ordered,\
    _channel_frequencies, _fnirs_check_bads, _fnirs_spread_bads
from ._optical_density import optical_density
from ._beer_lambert_law import beer_lambert_law
from ._scalp_coupling_index import scalp_coupling_index
