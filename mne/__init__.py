"""MNE software for MEG and EEG data analysis."""

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.devN' where N is an integer.
#

from ._version import __version__

# have to import verbose first since it's needed by many things
from .utils import (set_log_level, set_log_file, verbose, set_config,
                    get_config, get_config_path, set_cache_dir,
                    set_memmap_min_size, grand_average, sys_info, open_docs,
                    use_log_level)
from .io.pick import (pick_types, pick_channels,
                      pick_channels_regexp, pick_channels_forward,
                      pick_types_forward, pick_channels_cov,
                      pick_channels_evoked, pick_info,
                      channel_type, channel_indices_by_type)
from .io.base import concatenate_raws
from .io.meas_info import create_info, Info
from .io.proj import Projection
from .io.kit import read_epochs_kit
from .io.eeglab import read_epochs_eeglab
from .io.reference import (set_eeg_reference, set_bipolar_reference,
                           add_reference_channels)
from .io.what import what
from .bem import (make_sphere_model, make_bem_model, make_bem_solution,
                  read_bem_surfaces, write_bem_surfaces, write_head_bem,
                  read_bem_solution, write_bem_solution)
from .cov import (read_cov, write_cov, Covariance, compute_raw_covariance,
                  compute_covariance, whiten_evoked, make_ad_hoc_cov)
from .event import (read_events, write_events, find_events, merge_events,
                    pick_events, make_fixed_length_events, concatenate_events,
                    find_stim_steps, AcqParserFIF)
from ._freesurfer import (head_to_mni, head_to_mri, read_talxfm,
                          get_volume_labels_from_aseg, read_freesurfer_lut,
                          vertex_to_mni, read_lta)
from .forward import (read_forward_solution, apply_forward, apply_forward_raw,
                      average_forward_solutions, Forward,
                      write_forward_solution, make_forward_solution,
                      convert_forward_solution, make_field_map,
                      make_forward_dipole, use_coil_def)
from .source_estimate import (read_source_estimate,
                              SourceEstimate, VectorSourceEstimate,
                              VolSourceEstimate, VolVectorSourceEstimate,
                              MixedSourceEstimate, MixedVectorSourceEstimate,
                              grade_to_tris,
                              spatial_src_adjacency,
                              spatial_tris_adjacency,
                              spatial_dist_adjacency,
                              spatial_inter_hemi_adjacency,
                              spatio_temporal_src_adjacency,
                              spatio_temporal_tris_adjacency,
                              spatio_temporal_dist_adjacency,
                              extract_label_time_course, stc_near_sensors)
from .surface import (read_surface, write_surface, decimate_surface, read_tri,
                      get_head_surf, get_meg_helmet_surf, dig_mri_distances,
                      warp_montage_volume, get_montage_volume_labels)
from .morph_map import read_morph_map
from .morph import (SourceMorph, read_source_morph, grade_to_vertices,
                    compute_source_morph)
from .source_space import (read_source_spaces,
                           write_source_spaces, setup_source_space,
                           setup_volume_source_space, SourceSpaces,
                           add_source_space_distances, morph_source_spaces,
                           get_volume_labels_from_src)
from .annotations import (Annotations, read_annotations, annotations_from_events,
                          events_from_annotations)
from .epochs import (BaseEpochs, Epochs, EpochsArray, read_epochs,
                     concatenate_epochs, make_fixed_length_epochs)
from .evoked import (Evoked, EvokedArray, read_evokeds, write_evokeds,
                     combine_evoked)
from .label import (read_label, label_sign_flip,
                    write_label, stc_to_label, grow_labels, Label, split_label,
                    BiHemiLabel, read_labels_from_annot, write_labels_to_annot,
                    random_parcellation, morph_labels, labels_to_stc)
from .misc import parse_config, read_reject_parameters
from .coreg import (create_default_subject, scale_bem, scale_mri, scale_labels,
                    scale_source_space)
from .transforms import (read_trans, write_trans,
                         transform_surface_to, Transform)
from .proj import (read_proj, write_proj, compute_proj_epochs,
                   compute_proj_evoked, compute_proj_raw, sensitivity_map)
from .dipole import read_dipole, Dipole, DipoleFixed, fit_dipole
from .channels import (equalize_channels, rename_channels, find_layout,
                       read_vectorview_selection)
from .report import Report, open_report

from .io import (read_epochs_fieldtrip, read_evoked_besa,
                 read_evoked_fieldtrip, read_evokeds_mff)
from .rank import compute_rank

from . import beamformer
from . import channels
from . import chpi
from . import commands
from . import coreg
from . import cuda
from . import datasets
from . import dipole
from . import epochs
from . import event
from . import io
from . import filter
from . import gui
from . import inverse_sparse
from . import minimum_norm
from . import preprocessing
from . import simulation
from . import stats
from . import surface
from . import time_frequency
from . import viz
from . import decoding
from . import export

# initialize logging
set_log_level(None, False)
set_log_file()
