# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" SCoT: The Source Connectivity Toolbox
"""

from . import config

# default backend
# TODO: set default backend in config
from .backend import builtin

from .ooapi import Workspace

from .connectivity import Connectivity

from . import datatools

__all__ = ['Workspace', 'Connectivity', 'datatools']