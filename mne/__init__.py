__version__ = '0.1'

from .cov import read_cov, write_cov, write_cov_file, Covariance, \
                 compute_raw_data_covariance, compute_covariance
from .event import read_events, write_events, find_events, merge_events
from .forward import read_forward_solution
from .source_estimate import read_stc, write_stc, SourceEstimate, morph_data, \
                             spatio_temporal_src_connectivity, \
                             spatio_temporal_tris_connectivity
from .surface import read_bem_surfaces, read_surface
from .source_space import read_source_spaces
from .epochs import Epochs
from .label import label_time_courses, read_label
from .misc import parse_config, read_reject_parameters
from .transforms import transform_coordinates
import fiff
import artifacts
import stats
