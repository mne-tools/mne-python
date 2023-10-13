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
from .evoked import add_noise, simulate_evoked
from .raw import add_chpi, add_ecg, add_eog, simulate_raw
from .source import (
    SourceSimulator,
    select_source_in_label,
    simulate_sparse_stc,
    simulate_stc,
)
