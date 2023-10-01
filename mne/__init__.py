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
import lazy_loader as lazy

try:
    from importlib.metadata import version

    __version__ = version("mne")
except Exception:
    try:
        from ._version import __version__
    except ImportError:
        __version__ = "0.0.0"


__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "beamformer",
        "channels",
        "chpi",
        "commands",
        "coreg",
        "cuda",
        "datasets",
        "decoding",
        "defaults",
        "dipole",
        "epochs",
        "event",
        "export",
        "filter",
        "forward",
        "gui",
        "inverse_sparse",
        "io",
        "minimum_norm",
        "preprocessing",
        "report",
        "source_space",
        "simulation",
        "stats",
        "surface",
        "time_frequency",
        "viz",
    ],
    submod_attrs={
        "_freesurfer": [
            "get_volume_labels_from_aseg",
            "head_to_mni",
            "head_to_mri",
            "read_freesurfer_lut",
            "read_lta",
            "read_talxfm",
            "vertex_to_mni",
        ],
        "annotations": [
            "Annotations",
            "annotations_from_events",
            "count_annotations",
            "events_from_annotations",
            "read_annotations",
        ],
        "bem": [
            "make_bem_model",
            "make_bem_solution",
            "make_sphere_model",
            "read_bem_solution",
            "read_bem_surfaces",
            "write_bem_solution",
            "write_bem_surfaces",
            "write_head_bem",
        ],
        "channels": [
            "equalize_channels",
            "rename_channels",
            "find_layout",
            "read_vectorview_selection",
        ],
        "coreg": [
            "create_default_subject",
            "scale_bem",
            "scale_labels",
            "scale_mri",
            "scale_source_space",
        ],
        "cov": [
            "Covariance",
            "compute_covariance",
            "compute_raw_covariance",
            "make_ad_hoc_cov",
            "read_cov",
            "whiten_evoked",
            "write_cov",
        ],
        "dipole": [
            "Dipole",
            "DipoleFixed",
            "fit_dipole",
            "read_dipole",
        ],
        "epochs": [
            "BaseEpochs",
            "Epochs",
            "EpochsArray",
            "concatenate_epochs",
            "make_fixed_length_epochs",
            "read_epochs",
        ],
        "event": [
            "AcqParserFIF",
            "concatenate_events",
            "count_events",
            "find_events",
            "find_stim_steps",
            "make_fixed_length_events",
            "merge_events",
            "pick_events",
            "read_events",
            "write_events",
        ],
        "evoked": [
            "Evoked",
            "EvokedArray",
            "combine_evoked",
            "read_evokeds",
            "write_evokeds",
        ],
        "forward": [
            "Forward",
            "apply_forward_raw",
            "apply_forward",
            "average_forward_solutions",
            "convert_forward_solution",
            "make_field_map",
            "make_forward_dipole",
            "make_forward_solution",
            "read_forward_solution",
            "use_coil_def",
            "write_forward_solution",
        ],
        "io": [
            "read_epochs_fieldtrip",
            "read_evoked_besa",
            "read_evoked_fieldtrip",
            "read_evokeds_mff",
        ],
        "io.base": [
            "concatenate_raws",
            "match_channel_orders",
        ],
        "io.eeglab": [
            "read_epochs_eeglab",
        ],
        "io.kit": [
            "read_epochs_kit",
        ],
        "_fiff.meas_info": [
            "Info",
            "create_info",
        ],
        "_fiff.pick": [
            "channel_indices_by_type",
            "channel_type",
            "pick_channels_cov",
            "pick_channels_forward",
            "pick_channels_regexp",
            "pick_channels",
            "pick_info",
            "pick_types_forward",
            "pick_types",
        ],
        "_fiff.proj": [
            "Projection",
        ],
        "_fiff.reference": [
            "add_reference_channels",
            "set_bipolar_reference",
            "set_eeg_reference",
        ],
        "_fiff.what": [
            "what",
        ],
        "label": [
            "BiHemiLabel",
            "grow_labels",
            "label_sign_flip",
            "Label",
            "labels_to_stc",
            "morph_labels",
            "random_parcellation",
            "read_label",
            "read_labels_from_annot",
            "split_label",
            "stc_to_label",
            "write_label",
            "write_labels_to_annot",
        ],
        "misc": [
            "parse_config",
            "read_reject_parameters",
        ],
        "morph_map": [
            "read_morph_map",
        ],
        "morph": [
            "SourceMorph",
            "compute_source_morph",
            "grade_to_vertices",
            "read_source_morph",
        ],
        "proj": [
            "compute_proj_epochs",
            "compute_proj_evoked",
            "compute_proj_raw",
            "read_proj",
            "sensitivity_map",
            "write_proj",
        ],
        "rank": [
            "compute_rank",
        ],
        "report": [
            "Report",
            "open_report",
        ],
        "source_estimate": [
            "MixedSourceEstimate",
            "MixedVectorSourceEstimate",
            "SourceEstimate",
            "VectorSourceEstimate",
            "VolSourceEstimate",
            "VolVectorSourceEstimate",
            "extract_label_time_course",
            "grade_to_tris",
            "read_source_estimate",
            "spatial_dist_adjacency",
            "spatial_inter_hemi_adjacency",
            "spatial_src_adjacency",
            "spatial_tris_adjacency",
            "spatio_temporal_dist_adjacency",
            "spatio_temporal_src_adjacency",
            "spatio_temporal_tris_adjacency",
            "stc_near_sensors",
        ],
        "source_space._source_space": [
            "SourceSpaces",
            "add_source_space_distances",
            "get_volume_labels_from_src",
            "morph_source_spaces",
            "read_source_spaces",
            "setup_source_space",
            "setup_volume_source_space",
            "write_source_spaces",
        ],
        "surface": [
            "decimate_surface",
            "dig_mri_distances",
            "get_head_surf",
            "get_meg_helmet_surf",
            "get_montage_volume_labels",
            "read_surface",
            "read_tri",
            "write_surface",
        ],
        "transforms": [
            "Transform",
            "read_trans",
            "transform_surface_to",
            "write_trans",
        ],
        "utils": [
            "get_config_path",
            "get_config",
            "grand_average",
            "open_docs",
            "set_cache_dir",
            "set_config",
            "set_memmap_min_size",
            "sys_info",
            "use_log_level",
            "verbose",
        ],
    },
)

# initialize logging
from .utils import set_log_level, set_log_file

set_log_level(None, False)
set_log_file()
