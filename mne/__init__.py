"""MNE for MEG and EEG data analysis
"""

__version__ = '0.7.git'

# have to import verbose first since it's needed by many things
from .utils import set_log_level, set_log_file, verbose, set_config, \
                   get_config, get_config_path, set_cache_dir,\
                   set_memmap_min_size

from .cov import read_cov, write_cov, Covariance, \
                 compute_covariance, compute_raw_data_covariance, \
                 whiten_evoked
from .event import read_events, write_events, find_events, merge_events, \
                   pick_events, make_fixed_length_events, concatenate_events, \
                   find_stim_steps
from .forward import read_forward_solution, apply_forward, apply_forward_raw, \
                     do_forward_solution, average_forward_solutions, \
                     write_forward_solution
from .source_estimate import read_stc, write_stc, read_w, write_w, \
                             read_source_estimate, \
                             SourceEstimate, VolSourceEstimate, morph_data, \
                             morph_data_precomputed, compute_morph_matrix, \
                             grade_to_tris, grade_to_vertices, \
                             spatial_src_connectivity, \
                             spatial_tris_connectivity, \
                             spatial_dist_connectivity, \
                             spatio_temporal_src_connectivity, \
                             spatio_temporal_tris_connectivity, \
                             spatio_temporal_dist_connectivity, \
                             save_stc_as_volume, extract_label_time_course
from .surface import read_bem_surfaces, read_surface, write_bem_surface, \
                     write_surface, decimate_surface
from .source_space import read_source_spaces, vertex_to_mni, \
                          write_source_spaces
from .epochs import Epochs, read_epochs
from .label import label_time_courses, read_label, label_sign_flip, \
                   write_label, stc_to_label, grow_labels, Label, \
                   BiHemiLabel, labels_from_parc
from .misc import parse_config, read_reject_parameters
from .transforms import transform_coordinates, read_trans, write_trans
from .proj import read_proj, write_proj, compute_proj_epochs, \
                  compute_proj_evoked, compute_proj_raw, sensitivity_map
from .selection import read_selection
from .dipole import read_dip
from . import beamformer
from . import connectivity
from . import cuda
from . import datasets
from . import epochs
from . import fiff
from . import filter
from . import layouts
from . import minimum_norm
from . import mixed_norm
from . import preprocessing
from . import simulation
from . import stats
from . import tests
from . import time_frequency
from . import viz

# initialize logging
set_log_level(None, False)
set_log_file()

# initialize CUDA
if get_config('MNE_USE_CUDA', 'false').lower() == 'true':
    cuda.init_cuda()
