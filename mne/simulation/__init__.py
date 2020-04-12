"""Data simulation code."""

from .evoked import simulate_evoked, add_noise
from .raw import simulate_raw, add_ecg, add_eog, add_chpi
from .source import select_source_in_label, simulate_stc, simulate_sparse_stc
from .source import SourceSimulator
from .metrics import source_estimate_quantification
