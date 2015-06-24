# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis A. Engemann <denis.engemann@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy

DEFAULTS = dict(
    color=dict(mag='darkblue', grad='b', eeg='k', eog='k', ecg='m',
               emg='k', ref_meg='steelblue', misc='k', stim='k',
               resp='k', chpi='k', exci='k', ias='k', syst='k',
               seeg='k'),
    config_opts=dict(),
    units=dict(eeg='uV', grad='fT/cm', mag='fT', misc='AU',
               seeg='uV'),
    scalings=dict(mag=1e15, grad=1e13, eeg=1e6,
                  misc=1.0, seeg=1e4),
    scalings_plot_raw=dict(mag=1e-12, grad=4e-11, eeg=20e-6,
                           eog=150e-6, ecg=5e-4, emg=1e-3,
                           ref_meg=1e-12, misc=1e-3,
                           stim=1, resp=1, chpi=1e-4, exci=1,
                           ias=1, syst=1, seeg=1e-5),
    scalings_cov_rank=dict(mag=1e12, grad=1e11, eeg=1e5),
    ylim=dict(mag=(-600., 600.), grad=(-200., 200.),
              eeg=(-200., 200.), misc=(-5., 5.),
              seeg=(-200., 200.)),
    titles=dict(eeg='EEG', grad='Gradiometers',
                mag='Magnetometers', misc='misc', seeg='sEEG'),
    mask_params=dict(marker='o',
                     markerfacecolor='w',
                     markeredgecolor='k',
                     linewidth=0,
                     markeredgewidth=1,
                     markersize=4),
)


def _handle_default(k, v=None):
    """Helper to avoid dicts as default keyword arguments

    Use this function instead to resolve default dict values. Example usage::

        scalings = _handle_default('scalings', scalings)

    """
    this_mapping = deepcopy(DEFAULTS[k])
    if v is not None:
        if isinstance(v, dict):
            this_mapping.update(v)
        else:
            for key in this_mapping.keys():
                this_mapping[key] = v
    return this_mapping
