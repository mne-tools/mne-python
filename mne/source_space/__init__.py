"""Forward modeling code."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["_source_space"],
    submod_attrs={
        "_source_space": [
            "compute_distance_to_sensors",
            "get_decimated_surfaces",
            # These are documented in the MNE namespace but it doesn't hurt to
            # keep them here as well
            "SourceSpaces",
            "read_source_spaces",
            "write_source_spaces",
            "setup_source_space",
            "setup_volume_source_space",
            "add_source_space_distances",
        ],
    },
)
