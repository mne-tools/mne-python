__all__ = [
    "SourceSimulator",
    "add_chpi",
    "add_ecg",
    "add_eog",
    "add_noise",
    "metrics",
    "select_source_in_label",
    "simulate_evoked",
    "simulate_raw",
    "simulate_sparse_stc",
    "simulate_stc",
]
from . import metrics
from .evoked import simulate_evoked, add_noise
from .raw import simulate_raw, add_ecg, add_eog, add_chpi
from .source import (
    select_source_in_label,
    simulate_stc,
    simulate_sparse_stc,
    SourceSimulator,
)
