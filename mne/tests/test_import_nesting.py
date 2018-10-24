import os.path as op
import sys
from subprocess import Popen, PIPE

from mne.utils import run_tests_if_main


run_script = """
from __future__ import print_function

import sys
import mne

out = []

# check scipy
ok_scipy_submodules = set(['scipy', 'numpy',  # these appear in old scipy
                           'fftpack', 'lib', 'linalg',
                           'misc', 'sparse', 'version'])
scipy_submodules = set(x.split('.')[1] for x in sys.modules.keys()
                       if x.startswith('scipy.') and '__' not in x and
                       not x.split('.')[1].startswith('_')
                       and sys.modules[x] is not None)
bad = scipy_submodules - ok_scipy_submodules
if len(bad) > 0:
    out.append('scipy submodules: %s' % list(bad))

# check sklearn and others
_sklearn = _pandas = _mayavi = _matplotlib = False
for x in sys.modules.keys():
    if x.startswith('sklearn') and not _sklearn:
        out.append('sklearn')
        _sklearn = True
    if x.startswith('pandas') and not _pandas:
        out.append('pandas')
        _pandas = True
    if x.startswith('mayavi') and not _mayavi:
        out.append('mayavi')
        _mayavi = True
    if x.startswith('matplotlib') and not _matplotlib:
        out.append('matplotlib')
        _matplotlib = True
if len(out) > 0:
    print('\\nFound un-nested imports for:\\n' + '\\n'.join(out), end='')
    exit(1)
"""


def test_module_nesting():
    """Test that module imports are properly nested."""
    proc = Popen([sys.executable, '-c', run_script], stdout=PIPE, stderr=PIPE)
    stdout, stderr = proc.communicate()
    stdout = stdout.decode('utf-8')
    stderr = stderr.decode('utf-8')
    assert not proc.returncode, stdout + stderr


mpl_script = """
import os
import os.path as op
import re
import sys
import mne

reg = re.compile('test_.*.py')
for dirpath, _, filenames in os.walk('{0}'):
    if dirpath.endswith('tests'):
        test_dir = op.join('{0}', dirpath)
        sys.path.insert(0, test_dir)
        for filename in filenames:
            if reg.match(filename) is not None:
                __import__(op.splitext(filename)[0])
                for x in sys.modules.keys():
                    if x.startswith('matplotlib.pyplot'):
                        print('\\nFound un-nested pyplot import: ' +
                              op.join(test_dir, filename))
                        exit(1)
        sys.path.pop(0)
"""


def test_mpl_nesting():
    """Test that matplotlib imports are properly nested in tests."""
    mne_path = op.abspath(op.join(op.dirname(__file__), '..'))
    proc = Popen([sys.executable, '-c', mpl_script.format(mne_path)],
                 stdout=PIPE, stderr=PIPE)
    stdout, stderr = proc.communicate()
    stdout = stdout.decode('utf-8')
    stderr = stderr.decode('utf-8')
    assert not proc.returncode, stdout + stderr


run_tests_if_main()
