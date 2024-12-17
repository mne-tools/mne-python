"""The check functions."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numbers
import operator
import os
import re
from builtins import input  # noqa: A004, UP029
from difflib import get_close_matches
from importlib import import_module
from inspect import signature
from pathlib import Path

import numpy as np

from ..defaults import HEAD_SIZE_DEFAULT, _handle_default
from ..fixes import _compare_version, _median_complex
from ._logging import _record_warnings, _verbose_safe_false, logger, verbose, warn


def _ensure_int(x, name="unknown", must_be="an int", *, extra=""):
    """Ensure a variable is an integer."""
    # This is preferred over numbers.Integral, see:
    # https://github.com/scipy/scipy/pull/7351#issuecomment-299713159
    extra = f" {extra}" if extra else extra
    try:
        # someone passing True/False is much more likely to be an error than
        # intentional usage
        if isinstance(x, bool):
            raise TypeError()
        x = int(operator.index(x))
    except TypeError:
        raise TypeError(f"{name} must be {must_be}{extra}, got {type(x)}")
    return x


def _check_integer_or_list(arg, name):
    """Validate arguments that should be an integer or a list.

    Always returns a list.
    """
    if not isinstance(arg, list):
        arg = [_ensure_int(arg, name=name, must_be="an integer or a list")]
    return arg


def check_fname(fname, filetype, endings, endings_err=()):
    """Enforce MNE filename conventions.

    Parameters
    ----------
    fname : path-like
        Name of the file.
    filetype : str
        Type of file. e.g., ICA, Epochs, etc.
    endings : tuple
        Acceptable endings for the filename.
    endings_err : tuple
        Obligatory possible endings for the filename.
    """
    _validate_type(fname, "path-like", "fname")
    fname = str(fname)
    if len(endings_err) > 0 and not fname.endswith(endings_err):
        print_endings = " or ".join([", ".join(endings_err[:-1]), endings_err[-1]])
        raise OSError(
            f"The filename ({fname}) for file type {filetype} must end "
            f"with {print_endings}"
        )
    print_endings = " or ".join([", ".join(endings[:-1]), endings[-1]])
    if not fname.endswith(endings):
        warn(
            f"This filename ({fname}) does not conform to MNE naming conventions. "
            f"All {filetype} files should end with {print_endings}"
        )


def check_version(library, min_version="0.0", *, strip=True, return_version=False):
    r"""Check minimum library version required.

    Parameters
    ----------
    library : str
        The library name to import. Must have a ``__version__`` property.
    min_version : str
        The minimum version string. Anything that matches
        ``'(\d+ | [a-z]+ | \.)'``. Can also be empty to skip version
        check (just check for library presence).
    strip : bool
        If True (default), then PEP440 development markers like ``.devN``
        will be stripped from the version. This makes it so that
        ``check_version('mne', '1.1')`` will be ``True`` even when on version
        ``'1.1.dev0'`` (prerelease/dev version). This option is provided for
        backward compatibility with the behavior of ``LooseVersion``, and
        diverges from how modern parsing in ``packaging.version.parse`` works.

        .. versionadded:: 1.0
    return_version : bool
        If True (default False), also return the version (can be None if the
        library is missing).

        .. versionadded:: 1.0

    Returns
    -------
    ok : bool
        True if the library exists with at least the specified version.
    version : str | None
        The version. Only returned when ``return_version=True``.
    """
    ok = True
    version = None
    try:
        library = import_module(library)
    except ImportError:
        ok = False
    else:
        check_version = min_version and min_version != "0.0"
        get_version = check_version or return_version
        if get_version:
            version = library.__version__
            if strip:
                version = _strip_dev(version)
        if check_version:
            if _compare_version(version, "<", min_version):
                ok = False
    out = (ok, version) if return_version else ok
    return out


def _strip_dev(version):
    # First capturing group () is what we want to keep, at the beginning:
    #
    # - at least one numeral, then
    # - repeats of {dot, at least one numeral}
    #
    # The rest (consume to the end of the string) is the stuff we want to cut
    # off:
    #
    # - A period (maybe), then
    # - "dev", "rc", or "+", then
    # - numerals, periods, dashes, and "a" through "g" (hex chars)
    #
    # Thanks https://www.regextester.com !
    exp = r"^([0-9]+(?:\.[0-9]+)*)\.?(?:dev|rc|\+)[0-9+a-g\.\-]+$"
    match = re.match(exp, version)
    return match.groups()[0] if match is not None else version


def _require_version(lib, what, version="0.0"):
    """Require library for a purpose."""
    ok, got = check_version(lib, version, return_version=True)
    if not ok:
        extra = f" (version >= {version})" if version != "0.0" else ""
        why = "package was not found" if got is None else f"got {repr(got)}"
        raise ImportError(f"The {lib} package{extra} is required to {what}, {why}")


def _import_h5py():
    _require_version("h5py", "read MATLAB files >= v7.3")
    import h5py

    return h5py


def _import_h5io_funcs():
    h5io = _soft_import("h5io", "HDF5-based I/O")

    # Saving to HDF5 does not support pathlib.Path objects, which are more and more used
    # in MNE-Python.
    # Related issue in h5io: https://github.com/h5io/h5io/issues/113
    def cast_path_to_str(data: dict) -> dict:
        """Cast all paths value to string in data."""
        keys2cast = []
        for key, value in data.items():
            if isinstance(value, dict):
                cast_path_to_str(value)
            if isinstance(value, Path):
                data[key] = value.as_posix()
            if isinstance(key, Path):
                keys2cast.append(key)
        for key in keys2cast:
            data[key.as_posix()] = data.pop(key)
        return data

    def write_hdf5(fname, data, *args, **kwargs):
        """Write h5 and cast all paths to string in data."""
        if isinstance(data, dict):
            data = cast_path_to_str(data)
        elif isinstance(data, list):
            for k, elt in enumerate(data):
                if isinstance(elt, dict):
                    data[k] = cast_path_to_str(elt)
        h5io.write_hdf5(fname, data, *args, **kwargs)

    return h5io.read_hdf5, write_hdf5


def _import_pymatreader_funcs(purpose):
    pymatreader = _soft_import("pymatreader", purpose)
    return pymatreader.read_mat


# adapted from scikit-learn utils/validation.py
def check_random_state(seed):
    """Turn seed into a numpy.random.mtrand.RandomState instance.

    If seed is None, return the RandomState singleton used by np.random.mtrand.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, int | np.integer):
        return np.random.mtrand.RandomState(seed)
    if isinstance(seed, np.random.mtrand.RandomState):
        return seed
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError(
        f"{seed!r} cannot be used to seed a numpy.random.mtrand.RandomState instance"
    )


def _check_event_id(event_id, events):
    """Check event_id and convert to default format."""
    # check out event_id dict
    if event_id is None:  # convert to int to make typing-checks happy
        event_id = list(np.unique(events[:, 2]))
    if isinstance(event_id, dict):
        for key in event_id.keys():
            _validate_type(key, str, "Event names")
        event_id = {
            key: _ensure_int(val, f"event_id[{key}]") for key, val in event_id.items()
        }
    elif isinstance(event_id, list):
        event_id = [_ensure_int(v, f"event_id[{vi}]") for vi, v in enumerate(event_id)]
        event_id = dict(zip((str(i) for i in event_id), event_id))
    else:
        event_id = _ensure_int(event_id, "event_id")
        event_id = {str(event_id): event_id}
    return event_id


@verbose
def _check_fname(
    fname,
    overwrite=False,
    must_exist=False,
    name="File",
    need_dir=False,
    *,
    check_bids_split=False,
    verbose=None,
) -> Path:
    """Check for file existence, and return its absolute path."""
    _validate_type(fname, "path-like", name)
    # special case for MNE-BIDS, check split
    fname_path = Path(fname)
    if check_bids_split:
        try:
            from mne_bids import BIDSPath
        except Exception:
            pass
        else:
            if isinstance(fname, BIDSPath) and fname.split is not None:
                raise ValueError(
                    f"Passing a BIDSPath {name} with `{fname.split=}` is unsafe as it "
                    "can unexpectedly lead to invalid BIDS split naming. Explicitly "
                    f"set `{name}.split = None` to avoid ambiguity. If you want the "
                    f"old misleading split naming, you can pass `str({name})`."
                )

    fname = fname_path.expanduser().absolute()
    del fname_path

    if fname.exists():
        if not overwrite:
            raise FileExistsError(
                "Destination file exists. Please use option "
                '"overwrite=True" to force overwriting.'
            )
        elif overwrite != "read":
            logger.info("Overwriting existing file.")
        if must_exist:
            if need_dir:
                if not fname.is_dir():
                    raise OSError(
                        f"Need a directory for {name} but found a file at {fname}"
                    )
            else:
                if not fname.is_file():
                    raise OSError(
                        f"Need a file for {name} but found a directory at {fname}"
                    )
            if not os.access(fname, os.R_OK):
                raise PermissionError(f"{name} does not have read permissions: {fname}")
    elif must_exist:
        raise FileNotFoundError(f'{name} does not exist: "{fname}"')

    return fname


def _check_subject(
    first,
    second,
    *,
    raise_error=True,
    first_kind="class subject attribute",
    second_kind="input subject",
):
    """Get subject name from class."""
    if second is not None:
        _validate_type(second, "str", "subject input")
        if first is not None and first != second:
            raise ValueError(
                f"{first_kind} ({repr(first)}) did not match "
                f"{second_kind} ({second})"
            )
        return second
    elif first is not None:
        _validate_type(first, "str", f"Either {second_kind} subject or {first_kind}")
        return first
    elif raise_error is True:
        raise ValueError(f"Neither {second_kind} subject nor {first_kind} was a string")
    return None


def _check_preload(inst, msg):
    """Ensure data are preloaded."""
    from ..epochs import BaseEpochs
    from ..evoked import Evoked
    from ..source_estimate import _BaseSourceEstimate
    from ..time_frequency import BaseTFR
    from ..time_frequency.spectrum import BaseSpectrum

    if isinstance(inst, BaseTFR | Evoked | BaseSpectrum | _BaseSourceEstimate):
        pass
    else:
        name = "epochs" if isinstance(inst, BaseEpochs) else "raw"
        if not inst.preload:
            raise RuntimeError(
                "By default, MNE does not load data into main memory to "
                "conserve resources. " + msg + f" requires {name} data to be "
                "loaded. Use preload=True (or string) in the constructor or "
                f"{name}.load_data()."
            )
        if name == "epochs":
            inst._handle_empty("raise", msg)


def _check_compensation_grade(info1, info2, name1, name2="data", ch_names=None):
    """Ensure that objects have same compensation_grade."""
    from .._fiff.compensator import get_current_comp
    from .._fiff.meas_info import Info
    from .._fiff.pick import pick_channels, pick_info

    for t_info in (info1, info2):
        if t_info is None:
            return
        assert isinstance(t_info, Info), t_info  # or internal code is wrong

    if ch_names is not None:
        info1 = info1.copy()
        info2 = info2.copy()
        # pick channels
        for t_info in [info1, info2]:
            if t_info["comps"]:
                with t_info._unlock():
                    t_info["comps"] = []
            picks = pick_channels(t_info["ch_names"], ch_names, ordered=False)
            pick_info(t_info, picks, copy=False)
    # "or 0" here aliases None -> 0, as they are equivalent
    grade1 = get_current_comp(info1) or 0
    grade2 = get_current_comp(info2) or 0

    # perform check
    if grade1 != grade2:
        raise RuntimeError(
            f"Compensation grade of {name1} ({grade1}) and {name2} ({grade2}) "
            "do not match"
        )


def _soft_import(name, purpose, strict=True, *, min_version=None):
    """Import soft dependencies, providing informative errors on failure.

    Parameters
    ----------
    name : str
        Name of the module to be imported. For example, 'pandas'.
    purpose : str
        A very brief statement (formulated as a noun phrase) explaining what
        functionality the package provides to MNE-Python.
    strict : bool
        Whether to raise an error if module import fails.
    """
    # Mapping import namespaces to their pypi package name
    pip_name = dict(
        sklearn="scikit-learn",
        mne_bids="mne-bids",
        mne_nirs="mne-nirs",
        mne_features="mne-features",
        mne_qt_browser="mne-qt-browser",
        mne_connectivity="mne-connectivity",
        mne_gui_addons="mne-gui-addons",
        pyvista="pyvistaqt",
    ).get(name, name)

    got_version = None
    try:
        mod = import_module(name)
    except (ImportError, ModuleNotFoundError):
        mod = False
    else:
        have, got_version = check_version(
            name,
            min_version=min_version,
            return_version=True,
        )
        if not have:
            mod = False
    if mod is False and strict:
        extra = "" if min_version is None else f">={min_version}"
        if got_version is not None:
            extra += f" (found version {got_version})"
        raise RuntimeError(
            f"For {purpose} to work, the module {name}{extra} is needed, "
            "but it could not be imported. Use the following installation method "
            "appropriate for your environment:\n\n"
            f"    pip install {pip_name}\n"
            f"    conda install -c conda-forge {pip_name}"
        )
    return mod


def _check_pandas_installed(strict=True):
    """Aux function."""
    return _soft_import("pandas", "dataframe integration", strict=strict)


def _check_eeglabio_installed(strict=True):
    """Aux function."""
    return _soft_import("eeglabio", "exporting to EEGLab", strict=strict)


def _check_edfio_installed(strict=True):
    """Aux function."""
    return _soft_import("edfio", "exporting to EDF", strict=strict)


def _check_pybv_installed(strict=True):
    """Aux function."""
    return _soft_import("pybv", "exporting to BrainVision", strict=strict)


def _check_pymatreader_installed(strict=True):
    """Aux function."""
    return _soft_import("pymatreader", "loading v7.3 (HDF5) .MAT files", strict=strict)


def _check_pandas_index_arguments(index, valid):
    """Check pandas index arguments."""
    if index is None:
        return
    if isinstance(index, str):
        index = [index]
    if not isinstance(index, list):
        raise TypeError(
            "index must be `None` or a string or list of strings, got type "
            f"{type(index)}."
        )
    invalid = set(index) - set(valid)
    if invalid:
        plural = ("is not a valid option", "are not valid options")[
            int(len(invalid) > 1)
        ]
        raise ValueError(
            '"{}" {}. Valid index options are `None`, "{}".'.format(
                '", "'.join(invalid), plural, '", "'.join(valid)
            )
        )
    return index


def _check_time_format(time_format, valid, meas_date=None):
    """Check time_format argument."""
    if time_format not in valid and time_format is not None:
        valid_str = '", "'.join(valid)
        raise ValueError(
            f'"{time_format}" is not a valid time format. Valid options are '
            f'"{valid_str}" and None.'
        )
    # allow datetime only if meas_date available
    if time_format == "datetime" and meas_date is None:
        warn(
            "Cannot convert to Datetime when raw.info['meas_date'] is "
            "None. Falling back to Timedelta."
        )
        time_format = "timedelta"
    return time_format


def _check_ch_locs(info, picks=None, ch_type=None):
    """Check if channel locations exist.

    Parameters
    ----------
    info : Info | None
        `~mne.Info` instance.
    picks : list of int
        Channel indices to consider. If provided, ``ch_type`` must be ``None``.
    ch_type : str | None
        The channel type to restrict the check to. If ``None``, check all
        channel types. If provided, ``picks`` must be ``None``.
    """
    from .._fiff.pick import _picks_to_idx, pick_info

    if picks is not None and ch_type is not None:
        raise ValueError("Either picks or ch_type may be provided, not both")

    if picks is not None:
        info = pick_info(info=info, sel=picks)
    elif ch_type is not None:
        picks = _picks_to_idx(info=info, picks=ch_type, none=ch_type)
        info = pick_info(info=info, sel=picks)

    chs = info["chs"]
    locs3d = np.array([ch["loc"][:3] for ch in chs])
    return not (
        (locs3d == 0).all() or (~np.isfinite(locs3d)).all() or np.allclose(locs3d, 0.0)
    )


def _is_numeric(n):
    return isinstance(n, numbers.Number)


class _IntLike:
    @classmethod
    def __instancecheck__(cls, other):
        try:
            _ensure_int(other)
        except TypeError:
            return False
        else:
            return True


int_like = _IntLike()
path_like = (str, Path, os.PathLike)


class _Callable:
    @classmethod
    def __instancecheck__(cls, other):
        return callable(other)


class _Sparse:
    @classmethod
    def __instancecheck__(cls, other):
        from scipy import sparse

        return sparse.issparse(other)


_multi = {
    "str": (str,),
    "numeric": (np.floating, float, int_like),
    "path-like": path_like,
    "int-like": (int_like,),
    "callable": (_Callable(),),
    "array-like": (list, tuple, set, np.ndarray),
    "sparse": (_Sparse(),),
}


def _validate_type(item, types=None, item_name=None, type_name=None, *, extra=""):
    """Validate that `item` is an instance of `types`.

    Parameters
    ----------
    item : object
        The thing to be checked.
    types : type | str | tuple of types | tuple of str
         The types to be checked against.
         If str, must be one of {'int', 'int-like', 'str', 'numeric', 'info',
         'path-like', 'callable', 'array-like'}.
         If a tuple of str is passed, use 'int-like' and not 'int' for integers.
    item_name : str | None
        Name of the item to show inside the error message.
    type_name : str | None
        Possible types to show inside the error message that the checked item
        can be.
    extra : str
        Extra text to append to the warning.
    """
    if types == "int":
        _ensure_int(item, name=item_name, extra=extra)
        return  # terminate prematurely
    elif types == "info":
        from .._fiff.meas_info import Info as types

    if not isinstance(types, list | tuple):
        types = [types]

    check_types = sum(
        (
            (type(None),)
            if type_ is None
            else (type_,)
            if not isinstance(type_, str)
            else _multi[type_]
            for type_ in types
        ),
        (),
    )
    extra = f" {extra}" if extra else extra
    if not isinstance(item, check_types):
        if type_name is None:
            type_name = [
                "None"
                if cls_ is None
                else cls_.__name__
                if not isinstance(cls_, str)
                else cls_
                for cls_ in types
            ]
            if len(type_name) == 1:
                type_name = type_name[0]
            elif len(type_name) == 2:
                type_name = " or ".join(type_name)
            else:
                type_name[-1] = "or " + type_name[-1]
                type_name = ", ".join(type_name)
        _item_name = "Item" if item_name is None else item_name
        raise TypeError(
            f"{_item_name} must be an instance of {type_name}{extra}, "
            f"got {type(item)} instead."
        )


def _check_range(val, min_val, max_val, name, min_inclusive=True, max_inclusive=True):
    """Check that item is within range.

    Parameters
    ----------
    val : int | float
        The value to be checked.
    min_val : int | float
        The minimum value allowed.
    max_val : int | float
        The maximum value allowed.
    name : str
        The name of the value.
    min_inclusive : bool
        Whether ``val`` is allowed to be ``min_val``.
    max_inclusive : bool
        Whether ``val`` is allowed to be ``max_val``.
    """
    below_min = val < min_val if min_inclusive else val <= min_val
    above_max = val > max_val if max_inclusive else val >= max_val
    if below_min or above_max:
        error_str = f"The value of {name} must be between {min_val} "
        if min_inclusive:
            error_str += "inclusive "
        error_str += f"and {max_val}"
        if max_inclusive:
            error_str += "inclusive "
        raise ValueError(error_str)


def _path_like(item):
    """Validate that `item` is `path-like`.

    Parameters
    ----------
    item : object
        The thing to be checked.

    Returns
    -------
    bool
        ``True`` if `item` is a `path-like` object; ``False`` otherwise.
    """
    try:
        _validate_type(item, types="path-like")
        return True
    except TypeError:
        return False


def _check_if_nan(data, msg=" to be plotted"):
    """Raise if any of the values are NaN."""
    if not np.isfinite(data).all():
        raise ValueError(f"Some of the values {msg} are NaN.")


@verbose
def _check_info_inv(info, forward, data_cov=None, noise_cov=None, verbose=None):
    """Return good channels common to forward model and covariance matrices."""
    from .._fiff.pick import pick_types

    # get a list of all channel names:
    fwd_ch_names = forward["info"]["ch_names"]

    # handle channels from forward model and info:
    ch_names = _compare_ch_names(info["ch_names"], fwd_ch_names, info["bads"])

    # make sure that no reference channels are left:
    ref_chs = pick_types(info, meg=False, ref_meg=True)
    ref_chs = [info["ch_names"][ch] for ch in ref_chs]
    ch_names = [ch for ch in ch_names if ch not in ref_chs]

    # inform about excluding channels:
    if (
        data_cov is not None
        and set(info["bads"]) != set(data_cov["bads"])
        and (len(set(ch_names).intersection(data_cov["bads"])) > 0)
    ):
        logger.info(
            'info["bads"] and data_cov["bads"] do not match, '
            "excluding bad channels from both."
        )
    if (
        noise_cov is not None
        and set(info["bads"]) != set(noise_cov["bads"])
        and (len(set(ch_names).intersection(noise_cov["bads"])) > 0)
    ):
        logger.info(
            'info["bads"] and noise_cov["bads"] do not match, '
            "excluding bad channels from both."
        )

    # handle channels from data cov if data cov is not None
    # Note: data cov is supposed to be None in tf_lcmv
    if data_cov is not None:
        ch_names = _compare_ch_names(ch_names, data_cov.ch_names, data_cov["bads"])

    # handle channels from noise cov if noise cov available:
    if noise_cov is not None:
        ch_names = _compare_ch_names(ch_names, noise_cov.ch_names, noise_cov["bads"])

    # inform about excluding any channels apart from bads and reference
    all_bads = info["bads"] + ref_chs
    if data_cov is not None:
        all_bads += data_cov["bads"]
    if noise_cov is not None:
        all_bads += noise_cov["bads"]
    dropped_nonbads = set(info["ch_names"]) - set(ch_names) - set(all_bads)
    if dropped_nonbads:
        logger.info(
            f"Excluding {len(dropped_nonbads)} channel(s) missing from the "
            "provided forward operator and/or covariance matrices"
        )

    picks = [info["ch_names"].index(k) for k in ch_names if k in info["ch_names"]]
    return picks


def _compare_ch_names(names1, names2, bads):
    """Return channel names of common and good channels."""
    ch_names = [ch for ch in names1 if ch not in bads and ch in names2]
    return ch_names


def _check_channels_spatial_filter(ch_names, filters):
    """Return data channel indices to be used with spatial filter.

    Unlike ``pick_channels``, this respects the order of ch_names.
    """
    sel = []
    # first check for channel discrepancies between filter and data:
    for ch_name in filters["ch_names"]:
        if ch_name not in ch_names:
            raise ValueError(
                f"The spatial filter was computed with channel {ch_name} "
                "which is not present in the data. You should "
                "compute a new spatial filter restricted to the "
                "good data channels."
            )
    # then compare list of channels and get selection based on data:
    sel = [ii for ii, ch_name in enumerate(ch_names) if ch_name in filters["ch_names"]]
    return sel


def _check_rank(rank):
    """Check rank parameter."""
    _validate_type(rank, (None, dict, str), "rank")
    if isinstance(rank, str):
        if rank not in ["full", "info"]:
            raise ValueError(f'rank, if str, must be "full" or "info", got {rank}')
    return rank


def _check_one_ch_type(method, info, forward, data_cov=None, noise_cov=None):
    """Check number of sensor types and presence of noise covariance matrix."""
    from .._fiff.pick import _contains_ch_type, pick_info
    from ..cov import Covariance, make_ad_hoc_cov
    from ..time_frequency.csd import CrossSpectralDensity

    if isinstance(data_cov, CrossSpectralDensity):
        _validate_type(noise_cov, [None, CrossSpectralDensity], "noise_cov")
        # FIXME
        picks = list(range(len(data_cov.ch_names)))
        info_pick = info
    else:
        _validate_type(noise_cov, [None, Covariance], "noise_cov")
        picks = _check_info_inv(
            info,
            forward,
            data_cov=data_cov,
            noise_cov=noise_cov,
            verbose=_verbose_safe_false(),
        )
        info_pick = pick_info(info, picks)
    ch_types = [_contains_ch_type(info_pick, tt) for tt in ("mag", "grad", "eeg")]
    if sum(ch_types) > 1:
        if noise_cov is None:
            raise ValueError(
                "Source reconstruction with several sensor types"
                " requires a noise covariance matrix to be "
                "able to apply whitening."
            )
    if noise_cov is None:
        noise_cov = make_ad_hoc_cov(info_pick, std=1.0)
        allow_mismatch = True
    else:
        noise_cov = noise_cov.copy()
        if isinstance(noise_cov, Covariance) and "estimator" in noise_cov:
            del noise_cov["estimator"]
        allow_mismatch = False
    _validate_type(noise_cov, (Covariance, CrossSpectralDensity), "noise_cov")
    return noise_cov, picks, allow_mismatch


def _check_depth(depth, kind="depth_mne"):
    """Check depth options."""
    if not isinstance(depth, dict):
        depth = dict(exp=None if depth is None else float(depth))
    return _handle_default(kind, depth)


def _check_dict_keys(mapping, valid_keys, key_description, valid_key_source):
    """Check that the keys in dictionary are valid against a set list.

    Return the input dictionary if it is valid,
    otherwise raise a ValueError with a readable error message.

    Parameters
    ----------
    mapping : dict
        The user-provided dict whose keys we want to check.
    valid_keys : iterable
        The valid keys.
    key_description : str
        Description of the keys in ``mapping``, e.g., "channel name(s)" or
        "annotation(s)".
    valid_key_source : str
        Description of the ``valid_keys`` source, e.g., "info dict" or
        "annotations in the data".

    Returns
    -------
    mapping
        If all keys are valid the input dict is returned unmodified.
    """
    missing = set(mapping) - set(valid_keys)
    if len(missing):
        _is = "are" if len(missing) > 1 else "is"
        msg = (
            f"Invalid {key_description} {missing} {_is} not present in "
            f"{valid_key_source}"
        )
        raise ValueError(msg)

    return mapping


def _check_option(parameter, value, allowed_values, extra=""):
    """Check the value of a parameter against a list of valid options.

    Return the value if it is valid, otherwise raise a ValueError with a
    readable error message.

    Parameters
    ----------
    parameter : str
        The name of the parameter to check. This is used in the error message.
    value : any type
        The value of the parameter to check.
    allowed_values : list
        The list of allowed values for the parameter.
    extra : str
        Extra string to append to the invalid value sentence, e.g.
        "when using ico mode".

    Raises
    ------
    ValueError
        When the value of the parameter is not one of the valid options.

    Returns
    -------
    value : any type
        The value if it is valid.
    """
    if value in allowed_values:
        return value

    # Prepare a nice error message for the user
    extra = f" {extra}" if extra else extra
    msg = (
        "Invalid value for the '{parameter}' parameter{extra}. "
        "{options}, but got {value!r} instead."
    )
    allowed_values = list(allowed_values)  # e.g., if a dict was given
    if len(allowed_values) == 1:
        options = f"The only allowed value is {repr(allowed_values[0])}"
    else:
        options = "Allowed values are "
        if len(allowed_values) == 2:
            options += " and ".join(repr(v) for v in allowed_values)
        else:
            options += ", ".join(repr(v) for v in allowed_values[:-1])
            options += f", and {repr(allowed_values[-1])}"
    raise ValueError(
        msg.format(parameter=parameter, options=options, value=value, extra=extra)
    )


def _check_all_same_channel_names(instances):
    """Check if a collection of instances all have the same channels."""
    ch_names = instances[0].info["ch_names"]
    for inst in instances:
        if ch_names != inst.info["ch_names"]:
            return False
    return True


def _check_combine(mode, valid=("mean", "median", "std"), axis=0):
    # XXX TODO Possibly de-duplicate with _make_combine_callable of mne/viz/utils.py
    if mode == "mean":

        def fun(data):
            return np.mean(data, axis=axis)

    elif mode == "std":

        def fun(data):
            return np.std(data, axis=axis)

    elif mode == "median" or mode == np.median:

        def fun(data):
            return _median_complex(data, axis=axis)

    elif callable(mode):
        fun = mode
    else:
        raise ValueError(
            "Combine option must be "
            + ", ".join(valid)
            + f" or callable, got {mode} (type {type(mode)})."
        )
    return fun


def _check_src_normal(pick_ori, src):
    from ..source_space import SourceSpaces

    _validate_type(src, SourceSpaces, "src")
    if pick_ori == "normal" and src.kind not in ("surface", "discrete"):
        raise RuntimeError(
            "Normal source orientation is supported only for "
            "surface or discrete SourceSpaces, got type "
            f"{src.kind}"
        )


def _check_stc_units(stc, threshold=1e-7):  # 100 nAm threshold for warning
    max_cur = np.max(np.abs(stc.data))
    if max_cur > threshold:
        warn(
            f"The maximum current magnitude is {1e9 * max_cur:.1f} nAm, which is very "
            "large. Are you trying to apply the forward model to noise-normalized "
            "(dSPM, sLORETA, or eLORETA) values? The result will only be "
            "correct if currents (in units of Am) are used."
        )


def _check_qt_version(*, return_api=False, check_usable_display=True):
    """Check if Qt is installed."""
    from ..viz.backends._utils import _init_mne_qtapp

    try:
        from qtpy import API_NAME as api
        from qtpy import QtCore
    except Exception:
        api = version = None
    else:
        try:  # pyside
            version = QtCore.__version__
        except AttributeError:
            version = QtCore.QT_VERSION_STR
        # Having Qt installed is not enough -- sometimes the app is unusable
        # for example because there is no usable display (e.g., on a server),
        # so we have to try instantiating one to actually know.
        if check_usable_display:
            try:
                _init_mne_qtapp()
            except Exception:
                api = version = None
    if return_api:
        return version, api
    else:
        return version


def _check_sphere(sphere, info=None, sphere_units="m"):
    from ..bem import ConductorModel, fit_sphere_to_headshape, get_fitting_dig

    if sphere is None:
        sphere = HEAD_SIZE_DEFAULT
        if info is not None:
            # Decide if we have enough dig points to do the auto fit
            try:
                get_fitting_dig(info, "extra", verbose="error")
            except (RuntimeError, ValueError):
                pass
            else:
                sphere = "auto"

    if isinstance(sphere, str):
        if sphere not in ("auto", "eeglab"):
            raise ValueError(
                f'sphere, if str, must be "auto" or "eeglab", got {sphere}'
            )
        assert info is not None

        if sphere == "auto":
            R, r0, _ = fit_sphere_to_headshape(
                info, verbose=_verbose_safe_false(), units="m"
            )
            sphere = tuple(r0) + (R,)
            sphere_units = "m"
        elif sphere == "eeglab":
            # We need coordinates for the 2D plane formed by
            # Fpz<->Oz and T7<->T8, as this plane will be the horizon (i.e. it
            # will determine the location of the head circle).
            #
            # We implement some special-handling in case Fpz is missing, as
            # this seems to be a quite common situation in numerous EEG labs.
            montage = info.get_montage()
            if montage is None:
                raise ValueError(
                    'No montage was set on your data, but sphere="eeglab" '
                    "can only work if digitization points for the EEG "
                    "channels are available. Consider calling set_montage() "
                    "to apply a montage."
                )
            ch_pos = montage.get_positions()["ch_pos"]
            horizon_ch_names = ("Fpz", "Oz", "T7", "T8")

            if "FPz" in ch_pos:  # "fix" naming
                ch_pos["Fpz"] = ch_pos["FPz"]
                del ch_pos["FPz"]
            elif "Fpz" not in ch_pos and "Oz" in ch_pos:
                logger.info(
                    "Approximating Fpz location by mirroring Oz along "
                    "the X and Y axes."
                )
                # This assumes Fpz and Oz have the same Z coordinate
                ch_pos["Fpz"] = ch_pos["Oz"] * [-1, -1, 1]

            for ch_name in horizon_ch_names:
                if ch_name not in ch_pos:
                    msg = (
                        f'sphere="eeglab" requires digitization points of '
                        f"the following electrode locations in the data: "
                        f'{", ".join(horizon_ch_names)}, but could not find: '
                        f"{ch_name}"
                    )
                    if ch_name == "Fpz":
                        msg += ", and was unable to approximate its location from Oz"
                    raise ValueError(msg)

            # Calculate the radius from: T7<->T8, Fpz<->Oz
            radius = np.abs(
                [
                    ch_pos["T7"][0],  # X axis
                    ch_pos["T8"][0],  # X axis
                    ch_pos["Fpz"][1],  # Y axis
                    ch_pos["Oz"][1],  # Y axis
                ]
            ).mean()

            # Calculate the center of the head sphere
            # Use 4 digpoints for each of the 3 axes to hopefully get a better
            # approximation than when using just 2 digpoints.
            sphere_locs = dict()
            for idx, axis in enumerate(("X", "Y", "Z")):
                sphere_locs[axis] = np.mean(
                    [
                        ch_pos["T7"][idx],
                        ch_pos["T8"][idx],
                        ch_pos["Fpz"][idx],
                        ch_pos["Oz"][idx],
                    ]
                )
            sphere = (sphere_locs["X"], sphere_locs["Y"], sphere_locs["Z"], radius)
            sphere_units = "m"
            del sphere_locs, radius, montage, ch_pos
    elif isinstance(sphere, ConductorModel):
        if not sphere["is_sphere"] or len(sphere["layers"]) == 0:
            raise ValueError(
                "sphere, if a ConductorModel, must be spherical "
                "with multiple layers, not a BEM or single-layer "
                f"sphere (got {sphere})"
            )
        sphere = tuple(sphere["r0"]) + (sphere["layers"][0]["rad"],)
        sphere_units = "m"
    sphere = np.array(sphere, dtype=float)
    if sphere.shape == ():
        sphere = np.concatenate([[0.0] * 3, [sphere]])
    if sphere.shape != (4,):
        raise ValueError(
            "sphere must be float or 1D array of shape (4,), got "
            f"array-like of shape {sphere.shape}"
        )
    _check_option("sphere_units", sphere_units, ("m", "mm"))
    if sphere_units == "mm":
        sphere /= 1000.0

    sphere = np.array(sphere, float)
    return sphere


def _check_head_radius(radius, add_info=""):
    """Check that head radius is within a reasonable range (5. - 10.85 cm).

    Parameters
    ----------
    radius : float
        Head radius in meters.
    add_info : str
        Additional info to add to the warning message.

    Notes
    -----
    The maximum value was taken from the head size percentiles given in the
    following Wikipedia infographic:
    https://upload.wikimedia.org/wikipedia/commons/0/06/AvgHeadSizes.png

    the maximum radius is taken from the 99th percentile for men Glabella
    to back of the head measurements (Glabella is a point just above the
    Nasion):

        21.7cm / 2 = 10.85 cm = 0.1085 m

    The minimum value was taken from The National Center for Health Statistics
    (USA) infant head circumference percentiles:
    https://www.cdc.gov/growthcharts/html_charts/hcageinf.htm
    we take the minimum to be the radius corresponding to the 3rd percentile
    head circumference of female 0-month infant, rounded down:
    31.9302 cm circumference / (2 * pi) = 5.08 cm radius -> 0.05 m
    """
    min_radius = 0.05
    max_radius = 0.1085
    if radius > max_radius:
        msg = (
            f"Estimated head radius ({1e2 * radius:0.1f} cm) is "
            "above the 99th percentile for adult head size."
        )
        warn(msg + add_info)
    elif radius < min_radius:
        msg = (
            f"Estimated head radius ({1e2 * radius:0.1f} cm) is "
            "below the 3rd percentile for infant head size."
        )
        warn(msg + add_info)


def _check_freesurfer_home():
    from .config import get_config

    fs_home = get_config("FREESURFER_HOME")
    if fs_home is None:
        raise RuntimeError("The FREESURFER_HOME environment variable is not set.")
    return fs_home


def _suggest(val, options, cutoff=0.66):
    options = get_close_matches(val, options, cutoff=cutoff)
    if len(options) == 0:
        return ""
    elif len(options) == 1:
        return f" Did you mean {repr(options[0])}?"
    else:
        return f" Did you mean one of {repr(options)}?"


def _check_on_missing(on_missing, name="on_missing", *, extras=()):
    _validate_type(on_missing, str, name)
    _check_option(name, on_missing, ["raise", "warn", "ignore"] + list(extras))


def _on_missing(on_missing, msg, name="on_missing", error_klass=None):
    _check_on_missing(on_missing, name)
    error_klass = ValueError if error_klass is None else error_klass
    on_missing = "raise" if on_missing == "error" else on_missing
    on_missing = "warn" if on_missing == "warning" else on_missing
    if on_missing == "raise":
        raise error_klass(msg)
    elif on_missing == "warn":
        warn(msg)
    else:  # Ignore
        assert on_missing == "ignore"


def _safe_input(msg, *, alt=None, use=None):
    try:
        return input(msg)
    except EOFError:  # MATLAB or other non-stdin
        if use is not None:
            return use
        raise RuntimeError(
            f"Could not use input() to get a response to:\n{msg}\n"
            f"You can {alt} to avoid this error."
        )


def _ensure_events(events):
    err_msg = f"events should be a NumPy array of integers, got {type(events)}"
    with _record_warnings():
        try:
            events = np.asarray(events)
        except ValueError as np_err:
            if str(np_err).startswith(
                "setting an array element with a sequence. The requested "
                "array has an inhomogeneous shape"
            ):
                raise TypeError(err_msg) from None
            else:
                raise
    if not np.issubdtype(events.dtype, np.integer):
        raise TypeError(err_msg)
    if events.ndim != 2 or events.shape[1] != 3:
        raise ValueError(f"events must be of shape (N, 3), got {events.shape}")
    return events


def _to_rgb(*args, name="color", alpha=False):
    from matplotlib.colors import colorConverter

    func = colorConverter.to_rgba if alpha else colorConverter.to_rgb
    try:
        return func(*args)
    except ValueError:
        args = args[0] if len(args) == 1 else args
        raise ValueError(
            f'Invalid RGB{"A" if alpha else ""} argument(s) for {name}: '
            f"{repr(args)}"
        ) from None


def _import_nibabel(why="use MRI files"):
    try:
        import nibabel as nib
    except ImportError as exp:
        raise exp.__class__(f"nibabel is required to {why}, got:\n{exp}") from None
    return nib


def _check_method_kwargs(func, kwargs, msg=None):
    """Ensure **kwargs are compatible with the function they're passed to."""
    from .misc import _pl

    valid = list(signature(func).parameters)
    is_invalid = np.isin(list(kwargs), valid, invert=True)
    if is_invalid.any():
        invalid_kw = np.array(list(kwargs))[is_invalid].tolist()
        s = _pl(invalid_kw)
        if msg is None:
            msg = f'function "{func}"'
        raise TypeError(
            f'Got unexpected keyword argument{s} {", ".join(invalid_kw)} for {msg}.'
        )
