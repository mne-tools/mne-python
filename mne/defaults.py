# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis A. Engemann <denis.engemann@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

from copy import deepcopy

DEFAULTS = dict(
    color=dict(mag='darkblue', grad='b', eeg='k', eog='k', ecg='m', emg='k',
               ref_meg='steelblue', misc='k', stim='k', resp='k', chpi='k',
               exci='k', ias='k', syst='k', seeg='saddlebrown', dbs='seagreen',
               dipole='k', gof='k', bio='k', ecog='k', hbo='#AA3377', hbr='b',
               fnirs_cw_amplitude='k', fnirs_fd_ac_amplitude='k',
               fnirs_fd_phase='k', fnirs_od='k', csd='k', whitened='k'),
    si_units=dict(mag='T', grad='T/m', eeg='V', eog='V', ecg='V', emg='V',
                  misc='AU', seeg='V', dbs='V', dipole='Am', gof='GOF',
                  bio='V', ecog='V', hbo='M', hbr='M', ref_meg='T',
                  fnirs_cw_amplitude='V', fnirs_fd_ac_amplitude='V',
                  fnirs_fd_phase='rad', fnirs_od='V', csd='V/m²',
                  whitened='Z'),
    units=dict(mag='fT', grad='fT/cm', eeg='µV', eog='µV', ecg='µV', emg='µV',
               misc='AU', seeg='mV', dbs='µV', dipole='nAm', gof='GOF',
               bio='µV', ecog='µV', hbo='µM', hbr='µM', ref_meg='fT',
               fnirs_cw_amplitude='V', fnirs_fd_ac_amplitude='V',
               fnirs_fd_phase='rad', fnirs_od='V', csd='mV/m²',
               whitened='Z'),
    # scalings for the units
    scalings=dict(mag=1e15, grad=1e13, eeg=1e6, eog=1e6, emg=1e6, ecg=1e6,
                  misc=1.0, seeg=1e3, dbs=1e6, ecog=1e6, dipole=1e9, gof=1.0,
                  bio=1e6, hbo=1e6, hbr=1e6, ref_meg=1e15,
                  fnirs_cw_amplitude=1.0, fnirs_fd_ac_amplitude=1.0,
                  fnirs_fd_phase=1., fnirs_od=1.0, csd=1e3, whitened=1.),
    # rough guess for a good plot
    scalings_plot_raw=dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6,
                           ecg=5e-4, emg=1e-3, ref_meg=1e-12, misc='auto',
                           stim=1, resp=1, chpi=1e-4, exci=1, ias=1, syst=1,
                           seeg=1e-4, dbs=1e-4, bio=1e-6, ecog=1e-4, hbo=10e-6,
                           hbr=10e-6, whitened=10., fnirs_cw_amplitude=2e-2,
                           fnirs_fd_ac_amplitude=2e-2, fnirs_fd_phase=2e-1,
                           fnirs_od=2e-2, csd=200e-4,
                           dipole=1e-7, gof=1e2),
    scalings_cov_rank=dict(mag=1e12, grad=1e11, eeg=1e5,  # ~100x scalings
                           seeg=1e1, dbs=1e4, ecog=1e4, hbo=1e4, hbr=1e4),
    ylim=dict(mag=(-600., 600.), grad=(-200., 200.), eeg=(-200., 200.),
              misc=(-5., 5.), seeg=(-20., 20.), dbs=(-200., 200.),
              dipole=(-100., 100.), gof=(0., 1.), bio=(-500., 500.),
              ecog=(-200., 200.), hbo=(0, 20), hbr=(0, 20), csd=(-50., 50.)),
    titles=dict(mag='Magnetometers', grad='Gradiometers', eeg='EEG', eog='EOG',
                ecg='ECG', emg='EMG', misc='misc', seeg='sEEG', dbs='DBS',
                bio='BIO', dipole='Dipole', ecog='ECoG', hbo='Oxyhemoglobin',
                ref_meg='Reference Magnetometers',
                fnirs_cw_amplitude='fNIRS (CW amplitude)',
                fnirs_fd_ac_amplitude='fNIRS (FD AC amplitude)',
                fnirs_fd_phase='fNIRS (FD phase)',
                fnirs_od='fNIRS (OD)', hbr='Deoxyhemoglobin',
                gof='Goodness of fit', csd='Current source density',
                stim='Stimulus',
                ),
    mask_params=dict(marker='o',
                     markerfacecolor='w',
                     markeredgecolor='k',
                     linewidth=0,
                     markeredgewidth=1,
                     markersize=4),
    coreg=dict(
        mri_fid_opacity=1.0,
        dig_fid_opacity=1.0,

        mri_fid_scale=5e-3,
        dig_fid_scale=8e-3,
        extra_scale=4e-3,
        eeg_scale=4e-3, eegp_scale=20e-3, eegp_height=0.1,
        ecog_scale=5e-3,
        seeg_scale=5e-3,
        dbs_scale=5e-3,
        fnirs_scale=5e-3,
        source_scale=5e-3,
        detector_scale=5e-3,
        hpi_scale=15e-3,

        head_color=(0.988, 0.89, 0.74),
        hpi_color=(1., 0., 1.),
        extra_color=(1., 1., 1.),
        meg_color=(0., 0.25, 0.5), ref_meg_color=(0.5, 0.5, 0.5),
        helmet_color=(0.0, 0.0, 0.6),
        eeg_color=(1., 0.596, 0.588), eegp_color=(0.839, 0.15, 0.16),
        ecog_color=(1., 1., 1.),
        dbs_color=(0.82, 0.455, 0.659),
        seeg_color=(1., 1., .3),
        fnirs_color=(1., .647, 0.),
        source_color=(1., .05, 0.),
        detector_color=(.3, .15, .15),
        lpa_color=(1., 0., 0.),
        nasion_color=(0., 1., 0.),
        rpa_color=(0., 0., 1.),
    ),
    noise_std=dict(grad=5e-13, mag=20e-15, eeg=0.2e-6),
    eloreta_options=dict(eps=1e-6, max_iter=20, force_equal=False),
    depth_mne=dict(exp=0.8, limit=10., limit_depth_chs=True,
                   combine_xyz='spectral', allow_fixed_depth=False),
    depth_sparse=dict(exp=0.8, limit=None, limit_depth_chs='whiten',
                      combine_xyz='fro', allow_fixed_depth=True),
    interpolation_method=dict(eeg='spline', meg='MNE', fnirs='nearest'),
    volume_options=dict(
        alpha=None, resolution=1., surface_alpha=None, blending='mip',
        silhouette_alpha=None, silhouette_linewidth=2.),
    prefixes={'': 1e0, 'd': 1e1, 'c': 1e2, 'm': 1e3, 'µ': 1e6, 'u': 1e6,
              'n': 1e9, 'p': 1e12, 'f': 1e15},
    transform_zooms=dict(
        translation=None, rigid=None, affine=None, sdr=None),
    transform_niter=dict(
        translation=(10000, 1000, 100),
        rigid=(10000, 1000, 100),
        affine=(10000, 1000, 100),
        sdr=(10, 10, 5)),
    volume_label_indices=(
        # Left and middle
        4,  # Left-Lateral-Ventricle
        5,  # Left-Inf-Lat-Vent

        8,  # Left-Cerebellum-Cortex

        10,  # Left-Thalamus-Proper
        11,  # Left-Caudate
        12,  # Left-Putamen
        13,  # Left-Pallidum
        14,  # 3rd-Ventricle
        15,  # 4th-Ventricle
        16,  # Brain-Stem
        17,  # Left-Hippocampus
        18,  # Left-Amygdala

        26,  # Left-Accumbens-area

        28,  # Left-VentralDC

        # Right
        43,  # Right-Lateral-Ventricle
        44,  # Right-Inf-Lat-Vent

        47,  # Right-Cerebellum-Cortex

        49,  # Right-Thalamus-Proper
        50,  # Right-Caudate
        51,  # Right-Putamen
        52,  # Right-Pallidum
        53,  # Right-Hippocampus
        54,  # Right-Amygdala

        58,  # Right-Accumbens-area

        60,  # Right-VentralDC
    ),
    report_stc_plot_kwargs=dict(
        views=('lateral', 'medial'),
        hemi='split',
        backend='pyvistaqt',
        time_viewer=False,
        show_traces=False,
        size=(450, 450),
        background='white',
        time_label=None
    )
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
            for key in this_mapping:
                this_mapping[key] = v
    return this_mapping


HEAD_SIZE_DEFAULT = 0.095  # in [m]
_BORDER_DEFAULT = 'mean'
_EXTRAPOLATE_DEFAULT = 'auto'
