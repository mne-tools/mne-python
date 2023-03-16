from inspect import signature


def _translate_old_psd_kwargs(estimate, area_mode):
    amplitude = 'auto' if estimate == 'auto' else (estimate == 'amplitude')
    ci = 'sd' if area_mode == 'std' else area_mode
    return amplitude, ci


def _triage_old_psd_kwargs(**kwargs):
    from ..time_frequency import Spectrum

    plot_kwargs = {key: val for key, val in kwargs.items()
                   if key in signature(Spectrum.plot)}
    init_kwargs = {key: val for key, val in kwargs.items()
                   if key not in plot_kwargs}
    # user-defined picks should only be passed to the Spectrum constructor
    if 'picks' in kwargs:
        init_kwargs['picks'] = kwargs['picks']
        plot_kwargs['picks'] = 'all'
    return init_kwargs, plot_kwargs
