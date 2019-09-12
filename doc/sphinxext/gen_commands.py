# -*- coding: utf-8 -*-

import glob
from importlib import import_module
import os
from os import path as op

from mne.utils import _replace_md5, ArgvSetter


def setup(app):
    app.connect('builder-inited', generate_commands_rst)


def setup_module():
    # HACK: Stop nosetests running setup() above
    pass


# Header markings go:
# 1. =/= : Page title
# 2. =   : Command name
# 3. -/- : Command description
# 4. -   : Command sections (Examples, Notes)

header = """\
:orphan:

.. _python_commands:

===============================
Command line tools using Python
===============================

.. contents:: Page contents
   :local:
   :depth: 1

"""

command_rst = """

.. _gen_%s:

%s
%s

.. rst-class:: callout

%s

"""


def generate_commands_rst(app=None):
    from sphinx_gallery import sphinx_compatibility
    out_dir = op.abspath(op.join(op.dirname(__file__), '..', 'generated'))
    if not op.isdir(out_dir):
        os.mkdir(out_dir)
    out_fname = op.join(out_dir, 'commands.rst.new')

    command_path = op.abspath(
        op.join(os.path.dirname(__file__), '..', '..', 'mne', 'commands'))
    fnames = sorted([
        op.basename(fname)
        for fname in glob.glob(op.join(command_path, 'mne_*.py'))])
    iterator = sphinx_compatibility.status_iterator(
        fnames, 'generating MNE command help ... ', length=len(fnames))
    with open(out_fname, 'w') as f:
        f.write(header)
        for fname in iterator:
            cmd_name = fname[:-3]
            module = import_module('.' + cmd_name, 'mne.commands')
            with ArgvSetter(('mne', cmd_name, '--help')) as out:
                try:
                    module.run()
                except SystemExit:  # this is how these terminate
                    pass
            output = out.stdout.getvalue().splitlines()

            # Swap usage and title lines
            output[0], output[2] = output[2], output[0]

            # Add header marking
            for idx in (1, 0):
                output.insert(idx, '-' * len(output[0]))

            # Add code styling for the "Usage: " line
            for li, line in enumerate(output):
                if line.startswith('Usage: mne '):
                    output[li] = 'Usage: ``%s``' % line[7:]
                    break

            # Turn "Options:" into field list
            if 'Options:' in output:
                ii = output.index('Options:')
                output[ii] = 'Options'
                output.insert(ii + 1, '-------')
                output.insert(ii + 2, '')
                output.insert(ii + 3, '.. rst-class:: field-list cmd-list')
                output.insert(ii + 4, '')
            output = '\n'.join(output)
            f.write(command_rst % (cmd_name,
                                   cmd_name.replace('mne_', 'mne '),
                                   '=' * len(cmd_name),
                                   output))
    _replace_md5(out_fname)


# This is useful for testing/iterating to see what the result looks like
if __name__ == '__main__':
    generate_commands_rst()
