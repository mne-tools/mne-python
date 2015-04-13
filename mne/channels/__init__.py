"""
Module dedicated to the manipulation of channels,
setting of sensors locations used for processing and plotting.
"""

from .layout import (Layout, make_eeg_layout, make_grid_layout, read_layout,
                     find_layout, read_montage, apply_montage)

from .channels import (equalize_channels, rename_channels, _set_channels_type,
                       read_ch_connectivity)
