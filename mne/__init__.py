"""MNE for MEG and EEG data analysis
"""

__version__ = '0.3'

from .cov import read_cov, write_cov, Covariance, \
                 compute_raw_data_covariance, compute_covariance
from .event import read_events, write_events, find_events, merge_events, \
                   pick_events
from .forward import read_forward_solution, apply_forward, apply_forward_raw
from .source_estimate import read_stc, write_stc, read_w, write_w, \
                             SourceEstimate, morph_data, \
                             spatio_temporal_src_connectivity, \
                             spatio_temporal_tris_connectivity, \
                             save_stc_as_volume
from .surface import read_bem_surfaces, read_surface, write_bem_surface
from .source_space import read_source_spaces
from .epochs import Epochs
from .label import label_time_courses, read_label, label_sign_flip, \
                   write_label, stc_to_label
from .misc import parse_config, read_reject_parameters
from .transforms import transform_coordinates
from .proj import read_proj
from . import fiff
from . import artifacts
from . import stats
from . import viz
