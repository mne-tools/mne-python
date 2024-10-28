"""Utility functions for spectral and spectrotemporal analysis."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from inspect import currentframe, getargvalues, signature

from ..utils import warn


def _get_instance_type_string(inst):
    """Get string representation of the originating instance type."""
    from numpy import ndarray

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
    elif inst._inst_type == ndarray:
        inst_type_str = "Array"
    else:
        raise RuntimeError(
            f"Unknown instance type {inst._inst_type} in {type(inst).__name__}"
        )
    return inst_type_str


def _pop_with_fallback(mapping, key, fallback_fun):
    """Pop from a dict and fallback to a function parameter's default value."""
    fallback = signature(fallback_fun).parameters[key].default
    return mapping.pop(key, fallback)


def _update_old_psd_kwargs(kwargs):
    """Modify passed-in kwargs to match new API.

    NOTE: using plot_raw_psd as fallback (even for epochs) is fine because
    their kwargs are the same (and will stay the same: both are @legacy funcs).
    """
    from ..viz import plot_raw_psd as fallback_fun

    may_change = ("axes", "alpha", "ci_alpha", "amplitude", "ci")
    for kwarg in may_change:
        if kwarg in kwargs:
            warn(
                "The legacy plot_psd() method got an unexpected keyword argument "
                f"'{kwarg}', which is a parameter of Spectrum.plot(). Try rewriting as "
                f"object.compute_psd(...).plot(..., {kwarg}=<whatever>)."
            )
    kwargs.setdefault("axes", _pop_with_fallback(kwargs, "ax", fallback_fun))
    kwargs.setdefault("alpha", _pop_with_fallback(kwargs, "line_alpha", fallback_fun))
    kwargs.setdefault(
        "ci_alpha", _pop_with_fallback(kwargs, "area_alpha", fallback_fun)
    )
    est = _pop_with_fallback(kwargs, "estimate", fallback_fun)
    kwargs.setdefault("amplitude", est == "amplitude")
    area_mode = _pop_with_fallback(kwargs, "area_mode", fallback_fun)
    kwargs.setdefault("ci", "sd" if area_mode == "std" else area_mode)


def _split_psd_kwargs(*, plot_fun=None, kwargs=None):
    from ..io import BaseRaw
    from ..time_frequency import Spectrum

    # if no kwargs supplied, get them from calling func
    if kwargs is None:
        frame = currentframe().f_back
        arginfo = getargvalues(frame)
        kwargs = {k: v for k, v in arginfo.locals.items() if k in arginfo.args}
        if arginfo.keywords is not None:  # add in **method_kw
            kwargs.update(arginfo.locals[arginfo.keywords])

    # for compatibility with `plot_raw_psd`, `plot_epochs_psd` and
    # `plot_epochs_psd_topomap` functions (not just the instance methods/mixin)
    if "raw" in kwargs:
        kwargs["self"] = kwargs.pop("raw")
    elif "epochs" in kwargs:
        kwargs["self"] = kwargs.pop("epochs")

    # `reject_by_annotation` not needed for Epochs or Evoked
    if not isinstance(kwargs.pop("self", None), BaseRaw):
        kwargs.pop("reject_by_annotation", None)

    # handle API changes from .plot_psd(...) to .compute_psd(...).plot(...)
    if plot_fun is Spectrum.plot:
        _update_old_psd_kwargs(kwargs)

    # split off the plotting kwargs
    plot_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in signature(plot_fun).parameters and k != "picks"
    }
    for k in plot_kwargs:
        del kwargs[k]
    return kwargs, plot_kwargs
