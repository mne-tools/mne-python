"""Beamformers for source localization."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[],
    submod_attrs={
        "_lcmv": [
            "make_lcmv",
            "apply_lcmv",
            "apply_lcmv_epochs",
            "apply_lcmv_raw",
            "apply_lcmv_cov",
        ],
        "_dics": [
            "make_dics",
            "apply_dics",
            "apply_dics_epochs",
            "apply_dics_tfr_epochs",
            "apply_dics_csd",
        ],
        "_rap_music": [
            "rap_music",
            "trap_music",
        ],
        "_compute_beamformer": [
            "Beamformer",
            "read_beamformer",
        ],
        "resolution_matrix": [
            "make_lcmv_resolution_matrix",
        ],
    },
)
