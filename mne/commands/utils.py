#emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*- 
#ex: set sts=4 ts=4 sw=4 noet:
"""Some utility functions for commands (e.g. for cmdline handling)
"""

# Authors: Yaroslav Halchenko <debian@onerussian.com>
#
# License: BSD (3-clause)

import imp, os, re
from optparse import OptionParser

import mne

def get_optparser(cmdpath):
    """Create OptionParser with cmdsource specific settings (e.g. prog value)
    """
    command = os.path.basename(cmdpath)
    if re.match('mne_(.*).py', command):
        command = command[4:-3]

    # Fetch description
    mod = imp.load_source('__temp', cmdpath)
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
                          epilog=epilog)

    return parser
