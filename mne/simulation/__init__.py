"""Data simulation code
"""

from .evoked import generate_evoked, generate_noise_evoked, add_noise_evoked

from .epochs import generate_epochs, generate_noise_epochs, add_noise_epochs

from .source import (select_source_in_label, generate_sparse_stc, generate_stc,
                     simulate_sparse_stc)

from .simulation_metrics import source_estimate_quantification
