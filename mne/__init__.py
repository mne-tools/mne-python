"""MNE for MEG and EEG data analysis
"""

__version__ = '0.8.git'

# have to import verbose first since it's needed by many things
from .utils import (set_log_level, set_log_file, verbose, set_config,
                    get_config, get_config_path, set_cache_dir,
                    set_memmap_min_size)
from .io.pick import (pick_types, pick_channels, pick_types_evoked,
                      pick_channels_regexp, pick_channels_forward,
                      pick_types_forward, pick_channels_cov,
                      pick_channels_evoked, pick_info)
from .io.base import concatenate_raws, get_chpi_positions
from .io.meas_info import create_info
from .cov import (read_cov, write_cov, Covariance,
                  compute_covariance, compute_raw_data_covariance,
                  whiten_evoked)
from .event import (read_events, write_events, find_events, merge_events,
                    pick_events, make_fixed_length_events, concatenate_events,
                    find_stim_steps)
from .forward import (read_forward_solution, apply_forward, apply_forward_raw,
                      do_forward_solution, average_forward_solutions,
                      write_forward_solution, make_forward_solution,
                      convert_forward_solution, make_field_map)
from .source_estimate import (read_source_estimate, MixedSourceEstimate,
                              SourceEstimate, VolSourceEstimate, morph_data,
                              morph_data_precomputed, compute_morph_matrix,
                              grade_to_tris, grade_to_vertices,
                              spatial_src_connectivity,
                              spatial_tris_connectivity,
                              spatial_dist_connectivity,
                              spatio_temporal_src_connectivity,
                              spatio_temporal_tris_connectivity,
                              spatio_temporal_dist_connectivity,
                              save_stc_as_volume, extract_label_time_course)
from .surface import (read_bem_surfaces, read_surface, write_bem_surface,
                      write_surface, decimate_surface, read_morph_map,
                      read_bem_solution, get_head_surf,
                      get_meg_helmet_surf)
from .source_space import (read_source_spaces, vertex_to_mni,
                           write_source_spaces, setup_source_space,
                           setup_volume_source_space,
                           add_source_space_distances)
from .epochs import Epochs, EpochsArray, read_epochs
from .evoked import (Evoked, EvokedArray, read_evoked, write_evoked,
                     read_evokeds, write_evokeds)
from .label import (label_time_courses, read_label, label_sign_flip,
                    write_label, stc_to_label, grow_labels, Label, split_label,
                    BiHemiLabel, labels_from_parc, parc_from_labels,
                    read_labels_from_annot, write_labels_to_annot)
from .misc import parse_config, read_reject_parameters
from .coreg import (create_default_subject, scale_bem, scale_mri, scale_labels,
                    scale_source_space)
from .transforms import (transform_coordinates, read_trans, write_trans,
                         transform_surface_to)
from .proj import (read_proj, write_proj, compute_proj_epochs,
                   compute_proj_evoked, compute_proj_raw, sensitivity_map)
from .selection import read_selection
from .dipole import read_dip
from .layouts.layout import find_layout
from .channels import (equalize_channels, rename_channels,
                       read_ch_connectivity)

from . import beamformer
from . import connectivity
from . import coreg
from . import cuda
from . import datasets
from . import epochs
from . import externals
from . import fiff  # XXX : to be deprecated in 0.9
from . import io
from . import filter
from . import gui
from . import layouts
from . import minimum_norm
from . import mixed_norm
from . import preprocessing
from . import simulation
from . import stats
from . import time_frequency
from . import viz
from . import decoding
from . import realtime

# initialize logging
set_log_level(None, False)
set_log_file()

# initialize CUDA
if get_config('MNE_USE_CUDA', 'false').lower() == 'true':
    cuda.init_cuda()
