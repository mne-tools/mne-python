"""Artifacts finding/correction related functions
"""

from ..utils import deprecated
from ..preprocessing import find_eog_events, find_ecg_events

_preprocessing = ('The artifacts module will no longer be supported in v0.6.'
                  ' Please use the preprocessing module instead.')

find_eog_events = deprecated(_preprocessing)(find_eog_events)
find_ecg_events = deprecated(_preprocessing)(find_ecg_events)
