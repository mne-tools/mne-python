# -*- coding: utf-8 -*-
"""The check functions."""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause

import importlib
from builtins import input  # no-op here but facilitates testing
from difflib import get_close_matches
from distutils.version import LooseVersion
import operator
import os
import os.path as op
from pathlib import Path
import sys
import warnings
import numbers

import numpy as np

from ..fixes import _median_complex
from ._logging import warn, logger


def _ensure_int(x, name='unknown', must_be='an int'):
    """Ensure a variable is an integer."""
    # This is preferred over numbers.Integral, see:
    # https://github.com/scipy/scipy/pull/7351#issuecomment-299713159
    try:
        # someone passing True/False is much more likely to be an error than
        # intentional usage
        if isinstance(x, bool):
            raise TypeError()
        x = int(operator.index(x))
    except TypeError:
        raise TypeError('%s must be %s, got %s' % (name, must_be, type(x)))
    return x


def check_fname(fname, filetype, endings, endings_err=()):
    """Enforce MNE filename conventions.

    Parameters
    ----------
    fname : str
        Name of the file.
    filetype : str
        Type of file. e.g., ICA, Epochs etc.
    endings : tuple
        Acceptable endings for the filename.
    endings_err : tuple
        Obligatory possible endings for the filename.
    """
    _validate_type(fname, 'path-like', 'fname')
    fname = str(fname)
    if len(endings_err) > 0 and not fname.endswith(endings_err):
        print_endings = ' or '.join([', '.join(endings_err[:-1]),
                                     endings_err[-1]])
        raise IOError('The filename (%s) for file type %s must end with %s'
                      % (fname, filetype, print_endings))
    print_endings = ' or '.join([', '.join(endings[:-1]), endings[-1]])
    if not fname.endswith(endings):
        warn('This filename (%s) does not conform to MNE naming conventions. '
             'All %s files should end with %s'
             % (fname, filetype, print_endings))


def check_version(library, min_version='0.0'):
    r"""Check minimum library version required.

    Parameters
    ----------
    library : str
        The library name to import. Must have a ``__version__`` property.
    min_version : str
        The minimum version string. Anything that matches
        ``'(\d+ | [a-z]+ | \.)'``. Can also be empty to skip version
        check (just check for library presence).

    Returns
    -------
    ok : bool
        True if the library exists with at least the specified version.
    """
    ok = True
    try:
        library = __import__(library)
    except ImportError:
        ok = False
    else:
        if min_version and \
                LooseVersion(library.__version__) < LooseVersion(min_version):
            ok = False
    return ok


def _require_version(lib, what, version='0.0'):
    """Require library for a purpose."""
    if not check_version(lib, version):
        extra = f' (version >= {version})' if version != '0.0' else ''
        raise ImportError(f'The {lib} package{extra} is required to {what}')


def _check_mayavi_version(min_version='4.3.0'):
    """Check mayavi version."""
    if not check_version('mayavi', min_version):
        raise RuntimeError("Need mayavi >= %s" % min_version)


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
    if isinstance(seed, (int, np.integer)):
        return np.random.mtrand.RandomState(seed)
    if isinstance(seed, np.random.mtrand.RandomState):
        return seed
    try:
        # Generator is only available in numpy >= 1.17
        if isinstance(seed, np.random.Generator):
            return seed
    except AttributeError:
        pass
    raise ValueError('%r cannot be used to seed a '
                     'numpy.random.mtrand.RandomState instance' % seed)


def _check_event_id(event_id, events):
    """Check event_id and convert to default format."""
    # check out event_id dict
    if event_id is None:  # convert to int to make typing-checks happy
        event_id = list(np.unique(events[:, 2]))
    if isinstance(event_id, dict):
        for key in event_id.keys():
            _validate_type(key, str, 'Event names')
        event_id = {key: _ensure_int(val, 'event_id[%s]' % key)
                    for key, val in event_id.items()}
    elif isinstance(event_id, list):
        event_id = [_ensure_int(v, 'event_id[%s]' % vi)
                    for vi, v in enumerate(event_id)]
        event_id = dict(zip((str(i) for i in event_id), event_id))
    else:
        event_id = _ensure_int(event_id, 'event_id')
        event_id = {str(event_id): event_id}
    return event_id


def _check_fname(fname, overwrite=False, must_exist=False, name='File',
                 need_dir=False):
    """Check for file existence, and return string of its absolute path."""
    _validate_type(fname, 'path-like', name)
    fname = str(
        Path(fname)
        .expanduser()
        .absolute()
    )

    if op.exists(fname):
        if not overwrite:
            raise FileExistsError('Destination file exists. Please use option '
                                  '"overwrite=True" to force overwriting.')
        elif overwrite != 'read':
            logger.info('Overwriting existing file.')
        if must_exist:
            if need_dir:
                if not op.isdir(fname):
                    raise IOError(
                        f'Need a directory for {name} but found a file '
                        f'at {fname}')
            else:
                if not op.isfile(fname):
                    raise IOError(
                        f'Need a file for {name} but found a directory '
                        f'at {fname}')
            if not os.access(fname, os.R_OK):
                raise PermissionError(
                    f'{name} does not have read permissions: {fname}')
    elif must_exist:
        raise FileNotFoundError(f'{name} does not exist: {fname}')

    return fname


def _check_subject(first, second, *, raise_error=True,
                   first_kind='class subject attribute',
                   second_kind='input subject'):
    """Get subject name from class."""
    if second is not None:
        _validate_type(second, 'str', "subject input")
        if first is not None and first != second:
            raise ValueError(
                f'{first_kind} ({repr(first)}) did not match '
                f'{second_kind} ({second})')
        return second
    elif first is not None:
        _validate_type(
            first, 'str', f"Either {second_kind} subject or {first_kind}")
        return first
    elif raise_error is True:
        raise ValueError(f'Neither {second_kind} subject nor {first_kind} '
                         'was a string')
    return None


def _check_preload(inst, msg):
    """Ensure data are preloaded."""
    from ..epochs import BaseEpochs
    from ..evoked import Evoked
    from ..time_frequency import _BaseTFR

    if isinstance(inst, (_BaseTFR, Evoked)):
        pass
    else:
        name = "epochs" if isinstance(inst, BaseEpochs) else 'raw'
        if not inst.preload:
            raise RuntimeError(
                "By default, MNE does not load data into main memory to "
                "conserve resources. " + msg + ' requires %s data to be '
                'loaded. Use preload=True (or string) in the constructor or '
                '%s.load_data().' % (name, name))


def _check_compensation_grade(info1, info2, name1,
                              name2='data', ch_names=None):
    """Ensure that objects have same compensation_grade."""
    from ..io import Info
    from ..io.pick import pick_channels, pick_info
    from ..io.compensator import get_current_comp

    for t_info in (info1, info2):
        if t_info is None:
            return
        assert isinstance(t_info, Info), t_info  # or internal code is wrong

    if ch_names is not None:
        info1 = info1.copy()
        info2 = info2.copy()
        # pick channels
        for t_info in [info1, info2]:
            if t_info['comps']:
                with t_info._unlock():
                    t_info['comps'] = []
            picks = pick_channels(t_info['ch_names'], ch_names)
            pick_info(t_info, picks, copy=False)
    # "or 0" here aliases None -> 0, as they are equivalent
    grade1 = get_current_comp(info1) or 0
    grade2 = get_current_comp(info2) or 0

    # perform check
    if grade1 != grade2:
        raise RuntimeError(
            'Compensation grade of %s (%s) and %s (%s) do not match'
            % (name1, grade1, name2, grade2))


def _soft_import(name, purpose, strict=True):
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
    try:
        mod = importlib.import_module(name)
        return mod
    except (ImportError, ModuleNotFoundError):
        if strict:
            raise RuntimeError(f'For {purpose} to work, the {name} module is '
                               'needed, but it could not be imported.')
        else:
            return False


def _check_pandas_installed(strict=True):
    """Aux function."""
    return _soft_import('pandas', 'dataframe integration', strict=strict)


def _check_eeglabio_installed(strict=True):
    """Aux function."""
    return _soft_import('eeglabio', 'exporting to EEGLab', strict=strict)


def _check_edflib_installed(strict=True):
    """Aux function."""
    return _soft_import('EDFlib', 'exporting to EDF', strict=strict)


def _check_pandas_index_arguments(index, valid):
    """Check pandas index arguments."""
    if index is None:
        return
    if isinstance(index, str):
        index = [index]
    if not isinstance(index, list):
        raise TypeError('index must be `None` or a string or list of strings,'
                        ' got type {}.'.format(type(index)))
    invalid = set(index) - set(valid)
    if invalid:
        plural = ('is not a valid option',
                  'are not valid options')[int(len(invalid) > 1)]
        raise ValueError('"{}" {}. Valid index options are `None`, "{}".'
                         .format('", "'.join(invalid), plural,
                                 '", "'.join(valid)))
    return index


def _check_time_format(time_format, valid, meas_date=None):
    """Check time_format argument."""
    if time_format not in valid and time_format is not None:
        valid_str = '", "'.join(valid)
        raise ValueError('"{}" is not a valid time format. Valid options are '
                         '"{}" and None.'.format(time_format, valid_str))
    # allow datetime only if meas_date available
    if time_format == 'datetime' and meas_date is None:
        warn("Cannot convert to Datetime when raw.info['meas_date'] is "
             "None. Falling back to Timedelta.")
        time_format = 'timedelta'
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
    from ..io.pick import _picks_to_idx, pick_info

    if picks is not None and ch_type is not None:
        raise ValueError('Either picks or ch_type may be provided, not both')

    if picks is not None:
        info = pick_info(info=info, sel=picks)
    elif ch_type is not None:
        picks = _picks_to_idx(info=info, picks=ch_type, none=ch_type)
        info = pick_info(info=info, sel=picks)

    chs = info['chs']
    locs3d = np.array([ch['loc'][:3] for ch in chs])
    return not ((locs3d == 0).all() or
                (~np.isfinite(locs3d)).all() or
                np.allclose(locs3d, 0.))


def _is_numeric(n):
    return isinstance(n, numbers.Number)


class _IntLike(object):
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


class _Callable(object):
    @classmethod
    def __instancecheck__(cls, other):
        return callable(other)


_multi = {
    'str': (str,),
    'numeric': (np.floating, float, int_like),
    'path-like': path_like,
    'int-like': (int_like,),
    'callable': (_Callable(),),
}


def _validate_type(item, types=None, item_name=None, type_name=None):
    """Validate that `item` is an instance of `types`.

    Parameters
    ----------
    item : object
        The thing to be checked.
    types : type | str | tuple of types | tuple of str
         The types to be checked against.
         If str, must be one of {'int', 'int-like', 'str', 'numeric', 'info',
         'path-like', 'callable'}.
         If a tuple of str is passed, use 'int-like' and not 'int' for
         integers.
    item_name : str | None
        Name of the item to show inside the error message.
    type_name : str | None
        Possible types to show inside the error message that the checked item
        can be.
    """
    if types == "int":
        _ensure_int(item, name=item_name)
        return  # terminate prematurely
    elif types == "info":
        from mne.io import Info as types

    if not isinstance(types, (list, tuple)):
        types = [types]

    check_types = sum(((type(None),) if type_ is None else (type_,)
                       if not isinstance(type_, str) else _multi[type_]
                       for type_ in types), ())
    if not isinstance(item, check_types):
        if type_name is None:
            type_name = ['None' if cls_ is None else cls_.__name__
                         if not isinstance(cls_, str) else cls_
                         for cls_ in types]
            if len(type_name) == 1:
                type_name = type_name[0]
            elif len(type_name) == 2:
                type_name = ' or '.join(type_name)
            else:
                type_name[-1] = 'or ' + type_name[-1]
                type_name = ', '.join(type_name)
        _item_name = 'Item' if item_name is None else item_name
        raise TypeError(f"{_item_name} must be an instance of {type_name}, "
                        f"got {type(item)} instead.")


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
        _validate_type(item, types='path-like')
        return True
    except TypeError:
        return False


def _check_if_nan(data, msg=" to be plotted"):
    """Raise if any of the values are NaN."""
    if not np.isfinite(data).all():
        raise ValueError("Some of the values {} are NaN.".format(msg))


def _check_info_inv(info, forward, data_cov=None, noise_cov=None):
    """Return good channels common to forward model and covariance matrices."""
    from .. import pick_types
    # get a list of all channel names:
    fwd_ch_names = forward['info']['ch_names']

    # handle channels from forward model and info:
    ch_names = _compare_ch_names(info['ch_names'], fwd_ch_names, info['bads'])

    # make sure that no reference channels are left:
    ref_chs = pick_types(info, meg=False, ref_meg=True)
    ref_chs = [info['ch_names'][ch] for ch in ref_chs]
    ch_names = [ch for ch in ch_names if ch not in ref_chs]

    # inform about excluding channels:
    if (data_cov is not None and set(info['bads']) != set(data_cov['bads']) and
            (len(set(ch_names).intersection(data_cov['bads'])) > 0)):
        logger.info('info["bads"] and data_cov["bads"] do not match, '
                    'excluding bad channels from both.')
    if (noise_cov is not None and
            set(info['bads']) != set(noise_cov['bads']) and
            (len(set(ch_names).intersection(noise_cov['bads'])) > 0)):
        logger.info('info["bads"] and noise_cov["bads"] do not match, '
                    'excluding bad channels from both.')

    # handle channels from data cov if data cov is not None
    # Note: data cov is supposed to be None in tf_lcmv
    if data_cov is not None:
        ch_names = _compare_ch_names(ch_names, data_cov.ch_names,
                                     data_cov['bads'])

    # handle channels from noise cov if noise cov available:
    if noise_cov is not None:
        ch_names = _compare_ch_names(ch_names, noise_cov.ch_names,
                                     noise_cov['bads'])

    picks = [info['ch_names'].index(k) for k in ch_names if k in
             info['ch_names']]
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
    for ch_name in filters['ch_names']:
        if ch_name not in ch_names:
            raise ValueError('The spatial filter was computed with channel %s '
                             'which is not present in the data. You should '
                             'compute a new spatial filter restricted to the '
                             'good data channels.' % ch_name)
    # then compare list of channels and get selection based on data:
    sel = [ii for ii, ch_name in enumerate(ch_names)
           if ch_name in filters['ch_names']]
    return sel


def _check_rank(rank):
    """Check rank parameter."""
    _validate_type(rank, (None, dict, str), 'rank')
    if isinstance(rank, str):
        if rank not in ['full', 'info']:
            raise ValueError('rank, if str, must be "full" or "info", '
                             'got %s' % (rank,))
    return rank


def _check_one_ch_type(method, info, forward, data_cov=None, noise_cov=None):
    """Check number of sensor types and presence of noise covariance matrix."""
    from ..cov import make_ad_hoc_cov, Covariance
    from ..time_frequency.csd import CrossSpectralDensity
    from ..io.pick import pick_info
    from ..channels.channels import _contains_ch_type
    if isinstance(data_cov, CrossSpectralDensity):
        _validate_type(noise_cov, [None, CrossSpectralDensity], 'noise_cov')
        # FIXME
        picks = list(range(len(data_cov.ch_names)))
        info_pick = info
    else:
        _validate_type(noise_cov, [None, Covariance], 'noise_cov')
        picks = _check_info_inv(info, forward, data_cov=data_cov,
                                noise_cov=noise_cov)
        info_pick = pick_info(info, picks)
    ch_types =\
        [_contains_ch_type(info_pick, tt) for tt in ('mag', 'grad', 'eeg')]
    if sum(ch_types) > 1:
        if noise_cov is None:
            raise ValueError('Source reconstruction with several sensor types'
                             ' requires a noise covariance matrix to be '
                             'able to apply whitening.')
    if noise_cov is None:
        noise_cov = make_ad_hoc_cov(info_pick, std=1.)
        allow_mismatch = True
    else:
        noise_cov = noise_cov.copy()
        if isinstance(noise_cov, Covariance) and 'estimator' in noise_cov:
            del noise_cov['estimator']
        allow_mismatch = False
    _validate_type(noise_cov, (Covariance, CrossSpectralDensity), 'noise_cov')
    return noise_cov, picks, allow_mismatch


def _check_depth(depth, kind='depth_mne'):
    """Check depth options."""
    from ..defaults import _handle_default
    if not isinstance(depth, dict):
        depth = dict(exp=None if depth is None else float(depth))
    return _handle_default(kind, depth)


def _check_dict_keys(mapping, valid_keys,
                     key_description, valid_key_source):
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
        _is = 'are' if len(missing) > 1 else 'is'
        msg = (f'Invalid {key_description} {missing} {_is} not present in '
               f'{valid_key_source}')
        raise ValueError(msg)

    return mapping


def _check_option(parameter, value, allowed_values, extra=''):
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
    extra = ' ' + extra if extra else extra
    msg = ("Invalid value for the '{parameter}' parameter{extra}. "
           '{options}, but got {value!r} instead.')
    allowed_values = list(allowed_values)  # e.g., if a dict was given
    if len(allowed_values) == 1:
        options = f'The only allowed value is {repr(allowed_values[0])}'
    else:
        options = 'Allowed values are '
        options += ', '.join([f'{repr(v)}' for v in allowed_values[:-1]])
        options += f', and {repr(allowed_values[-1])}'
    raise ValueError(msg.format(parameter=parameter, options=options,
                                value=value, extra=extra))


def _check_all_same_channel_names(instances):
    """Check if a collection of instances all have the same channels."""
    ch_names = instances[0].info["ch_names"]
    for inst in instances:
        if ch_names != inst.info["ch_names"]:
            return False
    return True


def _check_combine(mode, valid=('mean', 'median', 'std'), axis=0):
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
        raise ValueError("Combine option must be " + ", ".join(valid) +
                         " or callable, got %s (type %s)." %
                         (mode, type(mode)))
    return fun


def _check_src_normal(pick_ori, src):
    from ..source_space import SourceSpaces
    _validate_type(src, SourceSpaces, 'src')
    if pick_ori == 'normal' and src.kind not in ('surface', 'discrete'):
        raise RuntimeError('Normal source orientation is supported only for '
                           'surface or discrete SourceSpaces, got type '
                           '%s' % (src.kind,))


def _check_stc_units(stc, threshold=1e-7):  # 100 nAm threshold for warning
    max_cur = np.max(np.abs(stc.data))
    if max_cur > threshold:
        warn('The maximum current magnitude is %0.1f nAm, which is very large.'
             ' Are you trying to apply the forward model to noise-normalized '
             '(dSPM, sLORETA, or eLORETA) values? The result will only be '
             'correct if currents (in units of Am) are used.'
             % (1e9 * max_cur))


def _check_pyqt5_version():
    bad = True
    try:
        from PyQt5.Qt import PYQT_VERSION_STR as version
    except Exception:
        version = 'unknown'
    else:
        if LooseVersion(version) >= LooseVersion('5.10'):
            bad = False
    bad &= sys.platform == 'darwin'
    if bad:
        warn('macOS users should use PyQt5 >= 5.10 for GUIs, got %s. '
             'Please upgrade e.g. with:\n\n'
             '    pip install "PyQt5>=5.10,<5.14"\n'
             % (version,))

    return version


def _check_sphere(sphere, info=None, sphere_units='m'):
    from ..defaults import HEAD_SIZE_DEFAULT
    from ..bem import fit_sphere_to_headshape, ConductorModel, get_fitting_dig
    if sphere is None:
        sphere = HEAD_SIZE_DEFAULT
        if info is not None:
            # Decide if we have enough dig points to do the auto fit
            try:
                get_fitting_dig(info, 'extra', verbose='error')
            except (RuntimeError, ValueError):
                pass
            else:
                sphere = 'auto'
    if isinstance(sphere, str):
        if sphere != 'auto':
            raise ValueError('sphere, if str, must be "auto", got %r'
                             % (sphere))
        R, r0, _ = fit_sphere_to_headshape(info, verbose=False, units='m')
        sphere = tuple(r0) + (R,)
        sphere_units = 'm'
    elif isinstance(sphere, ConductorModel):
        if not sphere['is_sphere'] or len(sphere['layers']) == 0:
            raise ValueError('sphere, if a ConductorModel, must be spherical '
                             'with multiple layers, not a BEM or single-layer '
                             'sphere (got %s)' % (sphere,))
        sphere = tuple(sphere['r0']) + (sphere['layers'][0]['rad'],)
        sphere_units = 'm'
    sphere = np.array(sphere, dtype=float)
    if sphere.shape == ():
        sphere = np.concatenate([[0.] * 3, [sphere]])
    if sphere.shape != (4,):
        raise ValueError('sphere must be float or 1D array of shape (4,), got '
                         'array-like of shape %s' % (sphere.shape,))
    _check_option('sphere_units', sphere_units, ('m', 'mm'))
    if sphere_units == 'mm':
        sphere /= 1000.

    sphere = np.array(sphere, float)
    return sphere


def _check_freesurfer_home():
    from .config import get_config
    fs_home = get_config('FREESURFER_HOME')
    if fs_home is None:
        raise RuntimeError(
            'The FREESURFER_HOME environment variable is not set.')
    return fs_home


def _suggest(val, options, cutoff=0.66):
    options = get_close_matches(val, options, cutoff=cutoff)
    if len(options) == 0:
        return ''
    elif len(options) == 1:
        return ' Did you mean %r?' % (options[0],)
    else:
        return ' Did you mean one of %r?' % (options,)


def _check_on_missing(on_missing, name='on_missing'):
    _validate_type(on_missing, str, name)
    _check_option(name, on_missing, ['raise', 'warn', 'ignore'])


def _on_missing(on_missing, msg, name='on_missing', error_klass=None):
    _check_on_missing(on_missing, name)
    error_klass = ValueError if error_klass is None else error_klass
    on_missing = 'raise' if on_missing == 'error' else on_missing
    on_missing = 'warn' if on_missing == 'warning' else on_missing
    if on_missing == 'raise':
        raise error_klass(msg)
    elif on_missing == 'warn':
        warn(msg)
    else:  # Ignore
        assert on_missing == 'ignore'


def _safe_input(msg, *, alt=None, use=None):
    try:
        return input(msg)
    except EOFError:  # MATLAB or other non-stdin
        if use is not None:
            return use
        raise RuntimeError(
            f'Could not use input() to get a response to:\n{msg}\n'
            f'You can {alt} to avoid this error.')


def _ensure_events(events):
    events_type = type(events)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('ignore')  # deprecation for object array
        events = np.asarray(events)
    if not np.issubdtype(events.dtype, np.integer):
        raise TypeError('events should be a NumPy array of integers, '
                        f'got {events_type}')
    if events.ndim != 2 or events.shape[1] != 3:
        raise ValueError(
            f'events must be of shape (N, 3), got {events.shape}')
    return events


def _to_rgb(*args, name='color', alpha=False):
    from matplotlib.colors import colorConverter
    func = colorConverter.to_rgba if alpha else colorConverter.to_rgb
    try:
        return func(*args)
    except ValueError:
        args = args[0] if len(args) == 1 else args
        raise ValueError(
            f'Invalid RGB{"A" if alpha else ""} argument(s) for {name}: '
            f'{repr(args)}') from None
