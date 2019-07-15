"""Some utility functions for commands (e.g., for cmdline handling)."""

# Authors: Yaroslav Halchenko <debian@onerussian.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)

import os
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
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location(name, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_optparser(cmdpath, usage=None, prog=None, version=None):
    """Create OptionParser with cmd specific settings (e.g., prog value)."""
    # Fetch description
    mod = load_module('__temp', cmdpath)
    if mod.__doc__:
        doc, description, epilog = mod.__doc__, None, None

        doc_lines = doc.split('\n')
        description = doc_lines[0]
        if len(doc_lines) > 1:
            epilog = '\n'.join(doc_lines[1:])

    # Set prog without command name for now
    if prog is None:
        prog = 'mne'

    # Get the name of the command and update prog with that name
    command = os.path.basename(cmdpath)
    command, _ = os.path.splitext(command)
    command = command[len(prog) + 1:]  # +1 is for `_` character
    prog += ' {}'.format(command)

    # Set version
    if version is None:
        version = mne.__version__

    # monkey patch OptionParser to not wrap epilog
    OptionParser.format_epilog = lambda self, formatter: self.epilog
    parser = OptionParser(prog=prog,
                          version=version,
                          description=description,
                          epilog=epilog, usage=usage)

    return parser
