# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis A. Engemann <denis.engemann@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy

DEFAULTS = dict(
    color=dict(mag='darkblue', grad='b', eeg='k', eog='k', ecg='m', emg='k',
               ref_meg='steelblue', misc='k', stim='k', resp='k', chpi='k',
               exci='k', ias='k', syst='k', seeg='k', dipole='k', gof='k',
               bio='k', ecog='k', hbo='darkblue', hbr='b'),
    units=dict(eeg='uV', grad='fT/cm', mag='fT', eog='uV', misc='AU',
               seeg='uV', dipole='nAm', gof='GOF', emg='uV', ecg='uV',
               bio='uV', ecog='uV', hbo='uM', hbr='uM'),
    scalings=dict(mag=1e15, grad=1e13, eeg=1e6, eog=1e6, emg=1e6, ecg=1e6,
                  misc=1.0, seeg=1e4, dipole=1e9, gof=1.0, bio=1e6, ecog=1e6,
                  hbo=1e6, hbr=1e6),
    scalings_plot_raw=dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6,
                           ecg=5e-4, emg=1e-3, ref_meg=1e-12, misc=1e-3,
                           stim=1, resp=1, chpi=1e-4, exci=1, ias=1, syst=1,
                           seeg=1e-5, bio=1e-6, ecog=1e-4, hbo=10e-6,
                           hbr=10e-6),
    scalings_cov_rank=dict(mag=1e12, grad=1e11, eeg=1e5),
    ylim=dict(mag=(-600., 600.), grad=(-200., 200.), eeg=(-200., 200.),
              misc=(-5., 5.), seeg=(-200., 200.), dipole=(-100., 100.),
              gof=(0., 1.), bio=(-500., 500.), ecog=(-200., 200.), hbo=(0, 20),
              hbr=(0, 20)),
    titles=dict(eeg='EEG', grad='Gradiometers', mag='Magnetometers',
                misc='misc', seeg='sEEG', dipole='Dipole', eog='EOG',
                gof='Goodness of fit', ecg='ECG', emg='EMG', bio='BIO',
                ecog='ECoG', hbo='Oxyhemoglobin', hbr='Deoxyhemoglobin'),
    mask_params=dict(marker='o',
                     markerfacecolor='w',
                     markeredgecolor='k',
                     linewidth=0,
                     markeredgewidth=1,
                     markersize=4),
    coreg=dict(
        mri_fid_opacity=1.0,
        dig_fid_opacity=0.3,

        mri_fid_scale=1e-2,
        dig_fid_scale=3e-2,
        extra_scale=4e-3,
        eeg_scale=4e-3, eegp_scale=20e-3, eegp_height=0.1,
        ecog_scale=5e-3,
        hpi_scale=15e-3,

        head_color=(0.988, 0.89, 0.74),
        hpi_color=(1., 0., 1.),
        extra_color=(1., 1., 1.),
        eeg_color=(1., 0.596, 0.588), eegp_color=(0.839, 0.15, 0.16),
        ecog_color=(1., 1., 1.),
        lpa_color=(1., 0., 0.),
        nasion_color=(0., 1., 0.),
        rpa_color=(0., 0., 1.),
    ),
)


def _handle_default(k, v=None):
    """Avoid dicts as default keyword arguments.

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
