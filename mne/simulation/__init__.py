"""Data simulation code
"""

from .evoked import (generate_evoked, generate_noise_evoked, add_noise_evoked,
                     simulate_evoked, simulate_noise_evoked)
from .raw import simulate_raw
from .source import (select_source_in_label, generate_sparse_stc, generate_stc,
                     simulate_sparse_stc)
from .metrics import source_estimate_quantification
