# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import glob
from os import path as op

from mne.utils import run_subprocess


def setup(app):
    app.connect('builder-inited', generate_commands_rst)


def setup_module():
    # HACK: Stop nosetests running setup() above
    pass


header = """.. _python_commands:

Command line tools using Python
===============================

.. contents:: Contents
   :local:
   :depth: 1

"""

command_rst = """

.. _gen_%s:

%s
----------------------------------------------------------

.. raw:: html

   <div>
   <pre>

%s

.. raw:: html

   </pre>
   </div>

"""


def generate_commands_rst(app):
    out_dir = op.abspath(op.join(op.dirname(__file__), '..', 'generated'))
    out_fname = op.join(out_dir, 'commands.rst')

    command_path = op.join(os.path.dirname(__file__), '..', '..', 'mne',
                           'commands')
    print('Generating commands for: %s ... ' % command_path, end='')
    fnames = glob.glob(op.join(command_path, 'mne_*.py'))

    with open(out_fname, 'w') as f:
        f.write(header)
        for fname in fnames:
            cmd_name = op.basename(fname)[:-3]

            output, _ = run_subprocess(['python', fname, '--help'])
            f.write(command_rst % (cmd_name, cmd_name.replace('mne_', 'mne '),
                                   output))
    print('[Done]')


# This is useful for testing/iterating to see what the result looks like
if __name__ == '__main__':
    generate_commands_rst()
