import sys
from subprocess import Popen, PIPE

from mne.utils import run_tests_if_main


run_script = """
from __future__ import print_function

import sys
import mne

# check scipy
ok_scipy_submodules = set(['scipy', 'numpy',  # these appear in old scipy
                           'fftpack', 'lib', 'linalg',
                           'misc', 'sparse', 'version'])
scipy_submodules = set(x.split('.')[1] for x in sys.modules.keys()
                       if x.startswith('scipy.') and '__' not in x and
                       not x.split('.')[1].startswith('_'))
bad = scipy_submodules - ok_scipy_submodules
if len(bad) > 0:
    print('Found un-nested scipy submodules:\\n%s' % bad, end='')
    exit(1)

# check sklearn
for x in sys.modules.keys():
    if x.startswith('sklearn'):
        print('Found un-nested sklearn import', end='')
        exit(1)
    if x.startswith('pandas'):
        print('Found un-nested pandas import', end='')
        exit(1)
    if x.startswith('nose'):
        print('Found un-nested nose import', end='')
        exit(1)
"""


def test_module_nesting():
    """Test that module imports are necessary
    """
    proc = Popen([sys.executable, '-c', run_script], stdout=PIPE, stderr=PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode:
        raise AssertionError(stdout)

run_tests_if_main()
