"""Some utility functions for commands (e.g. for cmdline handling)
"""

# Authors: Yaroslav Halchenko <debian@onerussian.com>
#
# License: BSD (3-clause)

import imp
import os
import re
from optparse import OptionParser
from subprocess import Popen, PIPE

import mne


def get_optparser(cmdpath):
    """Create OptionParser with cmd source specific settings (e.g. prog value)
    """
    command = os.path.basename(cmdpath)
    if re.match('mne_(.*).py', command):
        command = command[4:-3]
    elif re.match('mne_(.*).pyc', command):
        command = command[4:-4]

    # Fetch description
    if cmdpath.endswith('.pyc'):
        mod = imp.load_compiled('__temp', cmdpath)
    else:
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


def get_status_output(cmd):
    """ Replacement for commands.getstatusoutput which has been deprecated since 2.6
        Returns the error status, output and error output"""
    pipe = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    output, error = pipe.communicate()
    status = pipe.returncode
    return status, output, error
