# -*- coding: utf-8 -*-
# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

from pathlib import Path
from shutil import copytree
import subprocess
import sys

import pytest


def test_doc_filling(tmp_path):
    """Test that doc filling in setup.py works."""
    setup_path = Path(__file__).parent.parent.parent / 'setup.py'
    if not setup_path.is_file():
        pytest.skip(f'setup.py not found: {setup_path}')
    new_path = tmp_path / 'mne-python'
    repo_root = setup_path.parent
    mne_root = repo_root / 'mne'

    def _quick_copy_ignore(dirname, files):
        dirpath = Path(dirname)
        relname = dirpath.relative_to(repo_root)
        # allowlist
        if relname == Path('.'):
            files = list(set(files) - set(['setup.py', 'MANIFEST.in']))
            return dirname, files
        # denylist
        elif mne_root in dirpath.parents or mne_root == dirpath:
            files = [file for file in files if Path(file).suffix != '.py']
            return dirname, files
        # deny all
        else:
            return dirname, files

    copytree(repo_root, new_path, ignore=_quick_copy_ignore)
    out = subprocess.run(
        [sys.executable, 'setup.py', 'bdist'],
        cwd=new_path, capture_output=True)
    print(out.stdout.decode())
    print(out.stderr.decode())
    ret = out.returncode
    assert ret == 0
