__version__ = '0.1.git'

from .cov import read_cov, write_cov, write_cov_file, Covariance, \
                 compute_raw_data_covariance, compute_covariance
from .event import read_events, write_events, find_events
from .forward import read_forward_solution
from .stc import read_stc, write_stc
from .bem_surfaces import read_bem_surfaces
from .source_space import read_source_spaces
from .inverse import read_inverse_operator, apply_inverse, minimum_norm
from .epochs import Epochs
from .label import label_time_courses, read_label
from .misc import parse_config, read_reject_parameters
import fiff
