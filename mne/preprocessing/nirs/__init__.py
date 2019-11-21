"""NIRS specific preprocessing functions."""

# Authors: Robert Luke <mail@robertluke.net>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

from .nirs import short_channels, source_detector_distances
from ._optical_density import optical_density
from ._beer_lambert_law import beer_lambert_law
