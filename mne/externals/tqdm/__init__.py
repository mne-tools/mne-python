from ...fixes import _compare_version

try:
    from tqdm import __version__
    if _compare_version(__version__, '<', '4.36'):
        raise ImportError
except ImportError:  # use our copy
    from ._tqdm import *
    from ._tqdm import auto
else:  # use the system copy
    from tqdm import *
    from tqdm import auto
