"""Utility functions for spectral and spectrotemporal analysis."""
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import inspect


def _ensure_output_not_in_method_kw(inst, method_kw):
    legacy = inspect.currentframe().f_back.f_back.f_back.f_code.co_name == "_tfr_aux"
    if legacy:
        return method_kw
    if "output" in method_kw:
        raise ValueError(
            f"{type(inst).__name__}.compute_tfr() got an unexpected keyword argument "
            '"output". if you need more control over the output computation, please '
            "use the array interfaces (mne.time_frequency.tfr_array_morlet() or "
            "mne.time_frequency.tfr_array_multitaper())."
        )
    method_kw["output"] = "power"
    return method_kw


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
