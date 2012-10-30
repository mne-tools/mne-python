"""Artifacts finding/correction related functions
"""

from .eog import find_eog_events
from .ecg import find_ecg_events
from .ica import ICA, ica_find_eog_events, ica_find_ecg_events, score_funcs