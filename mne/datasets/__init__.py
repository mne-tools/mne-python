"""Functions for fetching remote datasets.

See :ref:`datasets` for more information.
"""
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.
import lazy_loader as lazy

(__getattr__, __dir__, __all__) = lazy.attach_stub(__name__, __file__)
