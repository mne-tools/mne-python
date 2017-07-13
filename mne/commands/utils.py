"""Some utility functions for commands (e.g. for cmdline handling)."""

# Authors: Yaroslav Halchenko <debian@onerussian.com>
#
# License: BSD (3-clause)

import sys
import os
import re
from optparse import OptionParser

import mne


def load_module(name, path):
    """Load module from .py/.pyc file.

    Parameters
    ----------
    name : str
        Name of the module.
    path : str
        Path to .py/.pyc file.

    Returns
    -------
    mod : module
        Imported module.
    """
    if sys.version_info < (3, 3):
        import imp
        if path.endswith('.pyc'):
            return imp.load_compiled(name, path)
        else:
            return imp.load_source(name, path)
    elif sys.version_info < (3, 5):
        if path.endswith('.pyc'):
            from importlib.machinery import SourcelessFileLoader
            return SourcelessFileLoader(name, path).load_module()
        else:
            from importlib.machinery import SourceFileLoader
            return SourceFileLoader(name, path).load_module()
    else:  # Python 3.5 or greater
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location(name, path)
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def get_optparser(cmdpath, usage=None):
    """Create OptionParser with cmd specific settings (e.g. prog value)."""
    command = os.path.basename(cmdpath)
    if re.match('mne_(.*).py', command):
        command = command[4:-3]
    elif re.match('mne_(.*).pyc', command):
        command = command[4:-4]

    # Fetch description
    mod = load_module('__temp', cmdpath)
    if mod.__doc__:
        doc, description, epilog = mod.__doc__, None, None

        doc_lines = doc.split('\n')
        description = doc_lines[0]
        if len(doc_lines) > 1:
            epilog = '\n'.join(doc_lines[1:])

    # monkey patch OptionParser to not wrap epilog
    OptionParser.format_epilog = lambda self, formatter: self.epilog
    parser = OptionParser(prog="mne %s" % command,
                          version=mne.__version__,
                          description=description,
                          epilog=epilog, usage=usage)

    return parser
