"""
Module dedicated to the manipulation of channels,
setting of sensors locations used for processing and plotting.
"""

from .layout import (Layout, make_eeg_layout, make_grid_layout, read_layout,
                     find_layout, generate_2d_layout)
from .montage import read_montage, read_dig_montage, Montage, DigMontage

from .channels import (equalize_channels, rename_channels,
                       read_ch_connectivity, _get_ch_type)
