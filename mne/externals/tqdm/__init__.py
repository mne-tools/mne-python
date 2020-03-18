from distutils.version import LooseVersion

try:
    from tqdm import *  # system version
    from tqdm import auto
except ImportError:
    from ._tqdm import *  # our copy
    from ._tqdm import auto
else:
    if LooseVersion(__version__) < LooseVersion('4.36'):
        from ._tqdm import *
        from ._tqdm import auto
