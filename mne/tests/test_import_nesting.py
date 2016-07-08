import sys
from subprocess import Popen, PIPE

from mne.utils import run_tests_if_main, requires_version


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
                       not x.split('.')[1].startswith('_'))
bad = scipy_submodules - ok_scipy_submodules
if len(bad) > 0:
    out.append('Found un-nested scipy submodules: %s' % list(bad))

# check sklearn and others
_sklearn = _pandas = _nose = _mayavi = False
for x in sys.modules.keys():
    if x.startswith('sklearn') and not _sklearn:
        out.append('Found un-nested sklearn import')
        _sklearn = True
    if x.startswith('pandas') and not _pandas:
        out.append('Found un-nested pandas import')
        _pandas = True
    if x.startswith('nose') and not _nose:
        out.append('Found un-nested nose import')
        _nose = True
    if x.startswith('mayavi') and not _mayavi:
        out.append('Found un-nested mayavi import')
        _mayavi = True
if len(out) > 0:
    print('\\n' + '\\n'.join(out), end='')
    exit(1)
"""


@requires_version('scipy', '0.11')  # old ones not organized properly
def test_module_nesting():
    """Test that module imports are necessary
    """
    proc = Popen([sys.executable, '-c', run_script], stdout=PIPE, stderr=PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode:
        raise AssertionError(stdout)

run_tests_if_main()
