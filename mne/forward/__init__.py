from .forward import (Forward, read_forward_solution, write_forward_solution,
                      is_fixed_orient, _read_forward_meas_info,
                      write_forward_meas_info,
                      compute_orient_prior, compute_depth_prior,
                      apply_forward, apply_forward_raw,
                      restrict_forward_to_stc, restrict_forward_to_label,
                      do_forward_solution, average_forward_solutions,
                      _restrict_gain_matrix, _stc_src_sel,
                      _fill_measurement_info, _apply_forward,
                      _subject_from_forward, convert_forward_solution,
                      _to_fixed_ori, prepare_bem_model)
from ._make_forward import make_forward_solution
from ._field_interpolation import (_make_surface_mapping, make_field_map,
                                   _as_meg_type_evoked, _map_meg_channels)
from . import _lead_dots  # for testing purposes
