# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""Forward modeling code."""

from ._source_space import (
    SourceSpaces,
    add_source_space_distances,
    compute_distance_to_sensors,
    find_source_space_hemi,
    get_decimated_surfaces,
    get_volume_labels_from_src,
    label_src_vertno_sel,
    morph_source_spaces,
    read_source_spaces,
    setup_source_space,
    setup_volume_source_space,
    write_source_spaces,
)
