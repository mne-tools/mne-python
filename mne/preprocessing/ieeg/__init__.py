"""Intracranial EEG specific preprocessing functions."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from ._projection import project_sensors_onto_brain
from ._volume import make_montage_volume, warp_montage
