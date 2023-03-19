from inspect import currentframe, getargvalues, signature


def _pop_with_fallback(mapping, key, fallback_fun):
    """Pop from a dict and fallback to a function parameter's default value."""
    fallback = signature(fallback_fun).parameters[key].default
    return mapping.pop(key, fallback)


def _translate_old_psd_kwargs(fallback_fun, kwargs):
    """Modify passed-in kwargs to match new API."""
    kwargs['axes'] = _pop_with_fallback(kwargs, 'ax', fallback_fun)
    kwargs['alpha'] = _pop_with_fallback(kwargs, 'line_alpha', fallback_fun)
    kwargs['ci_alpha'] = _pop_with_fallback(kwargs, 'area_alpha', fallback_fun)
    est = _pop_with_fallback(kwargs, 'estimate', fallback_fun)
    kwargs['amplitude'] = 'auto' if est == 'auto' else (est == 'amplitude')
    area_mode = _pop_with_fallback(kwargs, 'area_mode', fallback_fun)
    kwargs['ci'] = 'sd' if area_mode == 'std' else area_mode
    return kwargs


def _triage_old_psd_kwargs(*, fallback_fun, plot_fun=None, kwargs=None):
    """Convert .plot_psd(params) into .compute_psd(params).plot(other_params).

    ``fallback_fun`` should be a callable or "self" or None. ``plot_fun`` is
    ``Spectrum.plot`` or ``Spectrum.plot_topomap``. ``kwargs`` should be a
    dict; if it's not passed, the calling function's arguments will be inferred
    using the ``inspect`` module. Returns a tuple of dicts (1 for
    ``inst.compute_psd()`` and 1 for ``[Epochs]Spectrum.plot()``.
    """
    from ..io import BaseRaw
    from ..time_frequency import Spectrum
    from ..viz import plot_epochs_psd, plot_raw_psd

    if plot_fun is None:
        plot_fun = Spectrum.plot

    # for the SpectrumMixin we can't easily pass in the right fallback without
    # introspection; could have done that in the Mixin but better to keep it
    # tucked away here (less chance to confuse folks).
    if fallback_fun == 'self':
        calling_obj = currentframe().f_back.f_locals['self']
        if isinstance(calling_obj, BaseRaw):
            fallback_fun = plot_raw_psd
        else:
            fallback_fun = plot_epochs_psd

    # if no kwargs supplied, get them from calling func
    if kwargs is None:
        frame = currentframe().f_back
        arginfo = getargvalues(frame)
        kwargs = {k: v for k, v in arginfo.locals.items() if k in arginfo.args}
        if arginfo.keywords is not None:  # add in **method_kw
            kwargs.update(arginfo.locals[arginfo.keywords])

    # for compatibility with `plot_raw_psd`, `plot_epochs_psd` and
    # `plot_epochs_psd_topomap` functions (not just the instance methods/mixin)
    if 'raw' in kwargs:
        kwargs['self'] = kwargs.pop('raw')
    elif 'epochs' in kwargs:
        kwargs['self'] = kwargs.pop('epochs')

    # `reject_by_annotation` not needed for Epochs or Evoked
    if not isinstance(kwargs.pop('self', None), BaseRaw):
        kwargs.pop('reject_by_annotation', None)

    # handle API changes from .plot_psd(...) to .compute_psd(...).plot(...)
    if fallback_fun in (plot_raw_psd, plot_epochs_psd):
        kwargs = _translate_old_psd_kwargs(fallback_fun, kwargs)

    # split off the plotting kwargs. user-defined picks should only be passed
    # to the Spectrum constructor (otherwise integer picks could be wrong,
    # `None` will be handled wrong for `misc` data, etc)
    plot_kwargs = {k: v for k, v in kwargs.items() if
                   k in signature(plot_fun).parameters and k != 'picks'}
    for k in plot_kwargs:
        del kwargs[k]
    if plot_fun is Spectrum.plot:
        plot_kwargs['picks'] = 'all'  # TODO: this should be the default

    # sanity check
    overlapping_kwargs = set(kwargs).intersection(set(plot_kwargs))
    if len(overlapping_kwargs):  # might be 0 for mne.Reports or plot_topomap
        assert overlapping_kwargs == set(['picks'])

    return kwargs, plot_kwargs
