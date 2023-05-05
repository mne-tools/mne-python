from inspect import currentframe, getargvalues, signature


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

    kwargs["axes"] = _pop_with_fallback(kwargs, "ax", fallback_fun)
    kwargs["alpha"] = _pop_with_fallback(kwargs, "line_alpha", fallback_fun)
    kwargs["ci_alpha"] = _pop_with_fallback(kwargs, "area_alpha", fallback_fun)
    est = _pop_with_fallback(kwargs, "estimate", fallback_fun)
    kwargs["amplitude"] = "auto" if est == "auto" else (est == "amplitude")
    area_mode = _pop_with_fallback(kwargs, "area_mode", fallback_fun)
    kwargs["ci"] = "sd" if area_mode == "std" else area_mode


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
    # user-defined picks should only be passed to the Spectrum constructor
    # (otherwise integer picks could be wrong, `None` will be handled wrong
    # for `misc` data, etc)
    if plot_fun is Spectrum.plot:
        plot_kwargs["picks"] = "all"  # TODO: this will be the default in v1.5
    return kwargs, plot_kwargs
