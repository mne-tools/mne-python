# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""Forward modeling code."""

from ._source_space import (
    SourceSpaces,
    add_source_space_distances,
    compute_distance_to_sensors,
    get_decimated_surfaces,
    get_volume_labels_from_src,
    morph_source_spaces,
    read_source_spaces,
    setup_source_space,
    setup_volume_source_space,
    write_source_spaces,
)
