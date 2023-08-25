"""Data simulation code."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["metrics"],
    submod_attrs={
        "evoked": ["simulate_evoked", "add_noise"],
        "raw": ["simulate_raw", "add_ecg", "add_eog", "add_chpi"],
        "source": [
            "select_source_in_label",
            "simulate_stc",
            "simulate_sparse_stc",
            "SourceSimulator",
        ],
    },
)
