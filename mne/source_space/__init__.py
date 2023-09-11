"""Forward modeling code."""

import lazy_loader as lazy

__getattr_lz__, __dir__, __all__ = lazy.attach(
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


from . import _source_space
from ..utils import warn as _warn


def __getattr__(name):
    msg = out = None
    try:
        return __getattr_lz__(name)
    except AttributeError:
        try:
            out = getattr(_source_space, name)
        except AttributeError:
            pass  # will raise original error below
        else:
            # These should be removed (they're in the MNE namespace)
            msg = f"mne.source_space.{name} is deprecated and will be removed in 1.6, "
            if name in (
                "read_freesurfer_lut",
                "get_mni_fiducials",
                "get_volume_labels_from_aseg",
                "get_volume_labels_from_src",
            ):
                msg += f"use mne.{name} instead"
            else:
                msg += "use public API instead"
        if out is None:
            raise
    if msg is not None:
        _warn(msg, FutureWarning)
    return out
