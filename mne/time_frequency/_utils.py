"""Utility functions for spectral and spectrotemporal analysis."""
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.


def _get_instance_type_string(inst):
    """Get string representation of the originating instance type."""
    from ..epochs import BaseEpochs
    from ..evoked import Evoked, EvokedArray
    from ..io import BaseRaw

    parent_classes = inst._inst_type.__bases__
    if BaseRaw in parent_classes:
        inst_type_str = "Raw"
    elif BaseEpochs in parent_classes:
        inst_type_str = "Epochs"
    elif inst._inst_type in (Evoked, EvokedArray):
        inst_type_str = "Evoked"
    else:
        raise RuntimeError(
            f"Unknown instance type {inst._inst_type} in {type(inst).__name__}"
        )
    return inst_type_str
