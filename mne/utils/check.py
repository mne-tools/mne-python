# -*- coding: utf-8 -*-
"""The check functions."""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)


import operator
from distutils.version import LooseVersion
import os.path as op

import numpy as np

from ._logging import warn, logger


def _ensure_int(x, name='unknown', must_be='an int'):
    """Ensure a variable is an integer."""
    # This is preferred over numbers.Integral, see:
    # https://github.com/scipy/scipy/pull/7351#issuecomment-299713159
    try:
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


def check_version(library, min_version):
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
        if min_version:
            this_version = LooseVersion(library.__version__)
            if this_version < min_version:
                ok = False
    return ok


def _check_mayavi_version(min_version='4.3.0'):
    """Check mayavi version."""
    if not check_version('mayavi', min_version):
        raise RuntimeError("Need mayavi >= %s" % min_version)


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


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


def _check_fname(fname, overwrite=False, must_exist=False):
    """Check for file existence."""
    _validate_type(fname, 'str', 'fname')
    if must_exist and not op.isfile(fname):
        raise IOError('File "%s" does not exist' % fname)
    if op.isfile(fname):
        if not overwrite:
            raise IOError('Destination file exists. Please use option '
                          '"overwrite=True" to force overwriting.')
        elif overwrite != 'read':
            logger.info('Overwriting existing file.')


def _check_subject(class_subject, input_subject, raise_error=True):
    """Get subject name from class."""
    if input_subject is not None:
        _validate_type(input_subject, 'str', "subject input")
        return input_subject
    elif class_subject is not None:
        _validate_type(class_subject, 'str',
                       "Either subject input or class subject attribute")
        return class_subject
    else:
        if raise_error is True:
            raise ValueError('Neither subject input nor class subject '
                             'attribute was a string')
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


def _check_pylsl_installed(strict=True):
    """Aux function."""
    try:
        import pylsl
        return pylsl
    except ImportError:
        if strict is True:
            raise RuntimeError('For this functionality to work, the pylsl '
                               'library is required.')
        else:
            return False


def _check_pandas_installed(strict=True):
    """Aux function."""
    try:
        import pandas
        return pandas
    except ImportError:
        if strict is True:
            raise RuntimeError('For this functionality to work, the Pandas '
                               'library is required.')
        else:
            return False


def _check_pandas_index_arguments(index, defaults):
    """Check pandas index arguments."""
    if not any(isinstance(index, k) for k in (list, tuple)):
        index = [index]
    invalid_choices = [e for e in index if e not in defaults]
    if invalid_choices:
        options = [', '.join(e) for e in [invalid_choices, defaults]]
        raise ValueError('[%s] is not an valid option. Valid index'
                         'values are \'None\' or %s' % tuple(options))


def _check_ch_locs(chs):
    """Check if channel locations exist.

    Parameters
    ----------
    chs : dict
        The channels from info['chs']
    """
    locs3d = np.array([ch['loc'][:3] for ch in chs])
    return not ((locs3d == 0).all() or
                (~np.isfinite(locs3d)).all() or
                np.allclose(locs3d, 0.))


def _is_numeric(n):
    return isinstance(n, (np.integer, np.floating, int, float))


def _validate_type(item, types=None, item_name=None, type_name=None):
    """Validate that `item` is an instance of `types`.

    Parameters
    ----------
    item : object
        The thing to be checked.
    types : type | tuple of types | str
         The types to be checked against. If str, must be one of 'str', 'int',
         'numeric'.
    """
    if types == "int":
        _ensure_int(item, name=item_name)
        return  # terminate prematurely
    elif types == "str":
        types = str
        type_name = "str" if type_name is None else type_name
    elif types == "numeric":
        types = (np.integer, np.floating, int, float)
        type_name = "numeric" if type_name is None else type_name
    elif types == "info":
        from mne.io import Info as types
        type_name = "Info" if type_name is None else type_name
        item_name = "Info" if item_name is None else item_name
    if not isinstance(types, (list, tuple)):
        types = [types]

    check_types = tuple(type(None) if type_ is None else type_
                        for type_ in types)
    if not isinstance(item, check_types):
        if type_name is None:
            type_name = ['None' if cls_ is None else cls_.__name__
                         for cls_ in types]
            if len(type_name) == 1:
                type_name = type_name[0]
            elif len(type_name) == 2:
                type_name = ' or '.join(type_name)
            else:
                type_name[-1] = 'or ' + type_name[-1]
                type_name = ', '.join(type_name)
        raise TypeError('%s must be an instance of %s, got %s instead'
                        % (item_name, type_name, type(item),))


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
    """Check rank parameter and deal with deprecation."""
    err_msg = ('rank must be None, dict, "full", or int, '
               'got %s (type %s)' % (rank, type(rank)))
    if isinstance(rank, str):
        # XXX we can use rank='' to deprecate to get to None eventually:
        # if rank == '':
        #     warn('The rank parameter default in 0.18 of "full" will change '
        #          'to None in 0.19, set it explicitly to avoid this warning',
        #          DeprecationWarning)
        #     rank = 'full'
        if rank not in ['full', 'info']:
            raise ValueError('rank, if str, must be "full" or "info", '
                             'got %s' % (rank,))
    elif isinstance(rank, bool):
        raise TypeError(err_msg)
    elif rank is not None and not isinstance(rank, dict):
        try:
            rank = int(operator.index(rank))
        except TypeError:
            raise TypeError(err_msg)
        else:
            warn('rank as int is deprecated and will be removed in 0.19. '
                 'use rank=dict(meg=...) instead.', DeprecationWarning)
            rank = dict(meg=rank)
    return rank


def _check_one_ch_type(method, info, forward, data_cov=None, noise_cov=None):
    """Check number of sensor types and presence of noise covariance matrix."""
    from ..cov import make_ad_hoc_cov, Covariance
    from ..io.pick import pick_info
    from ..channels.channels import _contains_ch_type
    picks = _check_info_inv(info, forward, data_cov=data_cov,
                            noise_cov=noise_cov)
    info_pick = pick_info(info, picks)
    ch_types =\
        [_contains_ch_type(info_pick, tt) for tt in ('mag', 'grad', 'eeg')]
    if sum(ch_types) > 1:
        if method == 'lcmv' and noise_cov is None:
            raise ValueError('Source reconstruction with several sensor types'
                             ' requires a noise covariance matrix to be '
                             'able to apply whitening.')
        if method == 'dics':
            raise RuntimeError(
                'The use of several sensor types with the DICS beamformer is '
                'not supported yet.')
    if noise_cov is None:
        noise_cov = make_ad_hoc_cov(info_pick, std=1.)
    else:
        noise_cov = noise_cov.copy()
        if 'estimator' in noise_cov:
            del noise_cov['estimator']
    _validate_type(noise_cov, Covariance, 'noise_cov')
    return noise_cov, picks


def _check_depth(depth, kind='depth_mne'):
    """Check depth options."""
    from ..defaults import _handle_default
    if not isinstance(depth, dict):
        depth = dict(exp=None if depth is None else float(depth))
    return _handle_default(kind, depth)


def _check_option(parameter, value, allowed_values):
    """Check the value of a parameter against a list of valid options.

    Raises a ValueError with a readable error message if the value was invalid.

    Parameters
    ----------
    parameter : str
        The name of the parameter to check. This is used in the error message.
    value : any type
        The value of the parameter to check.
    allowed_values : list
        The list of allowed values for the parameter.

    Raises
    ------
    ValueError
        When the value of the parameter was not one of the valid options.
    """
    if value in allowed_values:
        return True

    # Prepare a nice error message for the user
    msg = ("Invalid value for the '{parameter}' parameter. "
           '{options}, but got {value!r} instead.')
    if len(allowed_values) == 1:
        options = 'The only allowed value is %r' % allowed_values[0]
    else:
        options = 'Allowed values are '
        options += ', '.join(['%r' % v for v in allowed_values[:-1]])
        options += ' and %r' % allowed_values[-1]
    raise ValueError(msg.format(parameter=parameter, options=options,
                                value=value))


def _check_all_same_channel_names(instances):
    """Check if a collection of instances all have the same channels."""
    ch_names = instances[0].info["ch_names"]
    for inst in instances:
        if ch_names != inst.info["ch_names"]:
            return False
    return True


def _check_combine(mode, valid=('mean', 'median', 'std')):
    if mode == "mean":
        def fun(data):
            return np.mean(data, axis=0)
    elif mode == "std":
        def fun(data):
            return np.std(data, axis=0)
    elif mode == "median":
        def fun(data):
            return np.median(data, axis=0)
    elif callable(mode):
        fun = mode
    else:
        raise ValueError("Combine option must be " + ", ".join(valid) +
                         " or callable, got %s (type %s)." %
                         (mode, type(mode)))
    return fun
