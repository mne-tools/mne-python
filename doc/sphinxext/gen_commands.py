# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import glob
from os import path as op
import subprocess
import sys

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
    from sphinx_gallery import sphinx_compatibility
    out_dir = op.abspath(op.join(op.dirname(__file__), '..', 'generated'))
    if not op.isdir(out_dir):
        os.mkdir(out_dir)
    out_fname = op.join(out_dir, 'commands.rst')

    command_path = op.abspath(
        op.join(os.path.dirname(__file__), '..', '..', 'mne', 'commands'))
    fnames = [op.basename(fname)
              for fname in glob.glob(op.join(command_path, 'mne_*.py'))]
    iterator = sphinx_compatibility.status_iterator(
        fnames, 'generating MNE command help ... ', length=len(fnames))
    with open(out_fname, 'w') as f:
        f.write(header)
        for fname in iterator:
            cmd_name = fname[:-3]
            run_name = op.join(command_path, fname)
            output, _ = run_subprocess([sys.executable, run_name, '--help'],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, verbose=False)
            f.write(command_rst % (cmd_name, cmd_name.replace('mne_', 'mne '),
                                   output))
    print('[Done]')


# This is useful for testing/iterating to see what the result looks like
if __name__ == '__main__':
    generate_commands_rst()
