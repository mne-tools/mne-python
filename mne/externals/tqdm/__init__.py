from distutils.version import LooseVersion

try:
    from tqdm import *  # system version
except ImportError:
    from ._tqdm import *  # our copy
else:
    if LooseVersion(__version__) < LooseVersion('4.36'):
        from ._tqdm import *
