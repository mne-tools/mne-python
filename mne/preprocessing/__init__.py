"""Preprocessing with artifact detection, SSP, and ICA."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD-3-Clause

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["eyetracking", "ieeg", "nirs"],
    submod_attrs={
        "_annotate_amplitude": ["annotate_amplitude"],
        "maxfilter": ["apply_maxfilter"],
        "ssp": ["compute_proj_ecg", "compute_proj_eog"],
        "eog": ["find_eog_events", "create_eog_epochs"],
        "ecg": ["find_ecg_events", "create_ecg_epochs"],
        "ica": [
            "ICA",
            "ica_find_eog_events",
            "ica_find_ecg_events",
            "get_score_funcs",
            "read_ica",
            "corrmap",
            "read_ica_eeglab",
        ],
        "otp": ["oversampled_temporal_projection"],
        "_peak_finder": ["peak_finder"],
        "infomax_": ["infomax"],
        "stim": ["fix_stim_artifact"],
        "maxwell": [
            "maxwell_filter",
            "find_bad_channels_maxwell",
            "compute_maxwell_basis",
            "maxwell_filter_prepare_emptyroom",
        ],
        "realign": ["realign_raw"],
        "xdawn": ["Xdawn"],
        "_csd": ["compute_current_source_density", "compute_bridged_electrodes"],
        "artifact_detection": [
            "annotate_movement",
            "compute_average_dev_head_t",
            "annotate_muscle_zscore",
            "annotate_break",
        ],
        "_regress": ["regress_artifact", "EOGRegression", "read_eog_regression"],
        "_fine_cal": [
            "compute_fine_calibration",
            "read_fine_calibration",
            "write_fine_calibration",
        ],
        "annotate_nan": ["annotate_nan"],
        "interpolate": ["equalize_bads", "interpolate_bridged_electrodes"],
        "_css": ["cortical_signal_suppression"],
        "hfc": ["compute_proj_hfc"],
        "bads": ["unify_bad_channels"]
    },
)
