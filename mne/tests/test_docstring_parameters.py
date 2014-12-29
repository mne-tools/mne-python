# TODO inspect for Cython (see sagenb.misc.sageinspect)
from __future__ import print_function

from os import path as op
import inspect
import warnings
import imp
from importlib import import_module
import sys
from mne.utils import run_tests_if_main

public_modules = [
    # the list of modules users need to access for all functionality
    'mne',
    'mne.beamformer',
    'mne.connectivity',
    'mne.datasets',
    'mne.datasets.megsim',
    'mne.datasets.sample',
    'mne.datasets.spm_face',
    'mne.decoding',
    'mne.filter',
    'mne.gui',
    'mne.inverse_sparse',
    'mne.io',
    'mne.io.kit',
    'mne.minimum_norm',
    'mne.preprocessing',
    'mne.realtime',
    'mne.report',
    'mne.simulation',
    'mne.stats',
    'mne.time_frequency',
    'mne.viz',
    ]

docscrape_path = op.join(op.dirname(__file__), '..', '..', 'doc', 'sphinxext',
                         'numpy_ext', 'docscrape.py')
docscrape = imp.load_source('docscrape', docscrape_path)


def get_name(func):
    parts = []
    module = inspect.getmodule(func)
    if module:
        parts.append(module.__name__)
    if hasattr(func, 'im_class'):
        parts.append(func.im_class.__name__)
    parts.append(func.__name__)
    return '.'.join(parts)


def check_parameters_match(func, doc=None):
    """Helper to check docstring, returns list of incorrect results"""
    incorrect = []
    name_ = get_name(func)
    if not name_.startswith('mne.'):
        return incorrect
    if inspect.isdatadescriptor(func):
        return incorrect
    try:
        args, varargs, varkw, defaults = inspect.getargspec(func)
    except TypeError:
        return incorrect
    # drop self
    if len(args) > 0 and args[0] == 'self':
        args = args[1:]

    if doc is None:
        with warnings.catch_warnings(record=True) as w:
            doc = docscrape.FunctionDoc(func)
        if len(w):
            raise RuntimeError('Error for %s:\n%s' % (name_, w[0]))
    # check set
    param_names = [name for name, _, _ in doc['Parameters']]
    # clean up some docscrape output:
    param_names = [name.split(':')[0].strip('` ') for name in param_names]
    param_names = [name for name in param_names if '*' not in name]
    try:
        args_set = set(args)
    except TypeError:
        # TODO: handle arg tuples
        return incorrect
    extra_params = set(param_names) - args_set
    if extra_params and not varkw:
        incorrect += [get_name(func) + ' in doc ' + str(sorted(extra_params))]

    if defaults:
        none_defaults = [arg for arg, default in zip(args[-len(defaults):],
                                                     defaults)
                         if default is None]
    else:
        none_defaults = []
    extra_args = args_set - set(param_names) - set(none_defaults)
    if param_names and extra_args:
        incorrect += [get_name(func) + ' in argspec ' +
                      str(sorted(extra_args))]
    # check order?
    return incorrect


def test_docstring_parameters():
    """Test module docsting formatting"""
    incorrect = []
    for name in public_modules:
        module = import_module(name)
        classes = inspect.getmembers(module, inspect.isclass)
        for cname, cls in classes:
            if cname.startswith('_'):
                continue
            with warnings.catch_warnings(record=True) as w:
                cdoc = docscrape.ClassDoc(cls)
            if len(w):
                raise RuntimeError('Error for __init__ of %s in %s:\n%s'
                                   % (cls, name, w[0]))
            if hasattr(cls, '__init__'):
                incorrect += check_parameters_match(cls.__init__, cdoc)
            for method_name in cdoc.methods:
                method = getattr(cls, method_name)
                incorrect += check_parameters_match(method)
            if hasattr(cls, '__call__'):
                incorrect += check_parameters_match(cls.__call__)
        functions = inspect.getmembers(module, inspect.isfunction)
        for fname, func in functions:
            if fname.startswith('_'):
                continue
            incorrect += check_parameters_match(func)
    if len(incorrect) > 0:
        print('\n'.join(sorted(incorrect)))
        raise AssertionError('%s docstring errors' % len(incorrect))


run_tests_if_main()
