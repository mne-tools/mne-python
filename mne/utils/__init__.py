# # # WARNING # # #
# This list must also be updated in doc/_templates/autosummary/class.rst if it
# is changed here!
_doc_special_members = ('__contains__', '__getitem__', '__iter__', '__len__',
                        '__add__', '__sub__', '__mul__', '__div__',
                        '__neg__')

from ._bunch import Bunch, BunchConst, BunchConstNamed
from .check import (check_fname, check_version, check_random_state,
                    _check_fname, _check_subject, _check_pandas_installed,
                    _check_pandas_index_arguments,
                    _check_event_id, _check_ch_locs, _check_compensation_grade,
                    _check_if_nan, _is_numeric, _ensure_int, _check_preload,
                    _validate_type, _check_info_inv,
                    _check_channels_spatial_filter, _check_one_ch_type,
                    _check_rank, _check_option, _check_depth, _check_combine,
                    _path_like, _check_src_normal, _check_stc_units,
                    _check_qt_version, _check_sphere, _check_time_format,
                    _check_freesurfer_home, _suggest, _require_version,
                    _on_missing, _check_on_missing, int_like, _safe_input,
                    _check_all_same_channel_names, path_like, _ensure_events,
                    _check_eeglabio_installed, _check_pybv_installed,
                    _check_edflib_installed, _to_rgb, _soft_import,
                    _check_dict_keys, _check_pymatreader_installed,
                    _import_h5py, _import_h5io_funcs,
                    _import_pymatreader_funcs, _check_head_radius)
from .config import (set_config, get_config, get_config_path, set_cache_dir,
                     set_memmap_min_size, get_subjects_dir, _get_stim_channel,
                     sys_info, _get_extra_data_path, _get_root_dir,
                     _get_numpy_libs)
from .docs import (copy_function_doc_to_method_doc, copy_doc, linkcode_resolve,
                   open_docs, deprecated, fill_doc, deprecated_alias, legacy,
                   copy_base_doc_to_subclass_doc, docdict as _docdict)
from .fetching import _url_to_local_path
from ._logging import (verbose, logger, set_log_level, set_log_file,
                       use_log_level, catch_logging, warn, filter_out_warnings,
                       wrapped_stdout, _get_call_line, _record_warnings,
                       ClosingStringIO, _verbose_safe_false)
from .misc import (run_subprocess, _pl, _clean_names, pformat, _file_like,
                   _explain_exception, _get_argvalues, sizeof_fmt,
                   running_subprocess, _DefaultEventParser,
                   _assert_no_instances, _resource_path, repr_html)
from .progressbar import ProgressBar
from ._testing import (run_command_if_main, requires_sklearn,
                       requires_version, requires_nibabel, requires_mne,
                       requires_good_network, requires_pandas, requires_h5py,
                       ArgvSetter, SilenceStdout, has_freesurfer, has_mne_c,
                       _TempDir, has_nibabel, buggy_mkl_svd,
                       requires_numpydoc, requires_freesurfer,
                       requires_nitime, requires_dipy, requires_mne_mark,
                       requires_neuromag2ft, requires_pylsl,
                       assert_object_equal, assert_and_remove_boundary_annot,
                       _raw_annot, assert_dig_allclose, assert_meg_snr,
                       assert_snr, assert_stcs_equal, _click_ch_name,
                       requires_openmeeg_mark)
from .numerics import (hashfunc, _compute_row_norms,
                       _reg_pinv, random_permutation, _reject_data_segments,
                       compute_corr, _get_inst_data, array_split_idx,
                       sum_squared, split_list, _gen_events, create_slices,
                       _time_mask, _freq_mask, grand_average, object_diff,
                       object_hash, object_size, _apply_scaling_cov,
                       _undo_scaling_cov, _apply_scaling_array,
                       _undo_scaling_array, _scaled_array, _replace_md5, _PCA,
                       _mask_to_onsets_offsets, _array_equal_nan,
                       _julian_to_cal, _cal_to_julian, _dt_to_julian,
                       _julian_to_dt, _dt_to_stamp, _stamp_to_dt,
                       _check_dt, _ReuseCycle, _arange_div, _hashable_ndarray,
                       _custom_lru_cache)
from .mixin import (SizeMixin, GetEpochsMixin, TimeMixin,
                    _prepare_read_metadata, _prepare_write_metadata,
                    _FakeNoPandas)
from .linalg import (_svd_lwork, _repeated_svd, _sym_mat_pow, sqrtm_sym, eigh,
                     _get_blas_funcs)
from .dataframe import (_set_pandas_dtype, _scale_dataframe_data,
                        _convert_times, _build_data_frame)
