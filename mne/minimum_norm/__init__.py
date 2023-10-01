"""Linear inverse solvers based on L2 Minimum Norm Estimates (MNE)."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[],
    submod_attrs={
        "inverse": [
            "InverseOperator",
            "read_inverse_operator",
            "apply_inverse",
            "apply_inverse_raw",
            "make_inverse_operator",
            "apply_inverse_epochs",
            "apply_inverse_tfr_epochs",
            "write_inverse_operator",
            "compute_rank_inverse",
            "prepare_inverse_operator",
            "estimate_snr",
            "apply_inverse_cov",
            "INVERSE_METHODS",
        ],
        "time_frequency": [
            "source_band_induced_power",
            "source_induced_power",
            "compute_source_psd",
            "compute_source_psd_epochs",
        ],
        "resolution_matrix": [
            "make_inverse_resolution_matrix",
            "get_point_spread",
            "get_cross_talk",
        ],
        "spatial_resolution": ["resolution_metrics"],
    },
)
