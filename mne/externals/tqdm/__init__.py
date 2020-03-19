from distutils.version import LooseVersion

try:
    from tqdm import __version__
    if LooseVersion(__version__) < LooseVersion('4.36'):
        raise ImportError
except ImportError:  # use our copy
    from ._tqdm import *
    from ._tqdm import auto
else:  # use the system copy
    from tqdm import *
    from tqdm import auto
