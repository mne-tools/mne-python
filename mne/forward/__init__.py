"""Forward modeling code."""

from .forward import (Forward, read_forward_solution, write_forward_solution,
                      is_fixed_orient, _read_forward_meas_info,
                      _select_orient_forward,
                      compute_orient_prior, compute_depth_prior,
                      apply_forward, apply_forward_raw,
                      restrict_forward_to_stc, restrict_forward_to_label,
                      average_forward_solutions, _stc_src_sel,
                      _fill_measurement_info, _apply_forward,
                      _subject_from_forward, convert_forward_solution,
                      _merge_meg_eeg_fwds, _do_forward_solution)
from ._make_forward import (make_forward_solution, _prepare_for_forward,
                            _prep_meg_channels, _prep_eeg_channels,
                            _to_forward_dict, _create_meg_coils,
                            _read_coil_defs, _transform_orig_meg_coils,
                            make_forward_dipole, use_coil_def)
from ._compute_forward import (_magnetic_dipole_field_vec, _compute_forwards,
                               _concatenate_coils)
from ._field_interpolation import (_make_surface_mapping, make_field_map,
                                   _as_meg_type_inst, _map_meg_or_eeg_channels)
from . import _lead_dots  # for testing purposes
