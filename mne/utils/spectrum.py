def _translate_old_psd_kwargs(estimate, area_mode):
    amplitude = 'auto' if estimate == 'auto' else (estimate == 'amplitude')
    ci = 'sd' if area_mode == 'std' else area_mode
    return amplitude, ci
