"""Module dedicated to manipulation of channels.

Can be used for setting of sensor locations used for processing and plotting.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[],
    submod_attrs={
        "channels": [
            "equalize_channels",
            "rename_channels",
            "fix_mag_coil_types",
            "read_ch_adjacency",
            "_get_ch_type",
            "find_ch_adjacency",
            "make_1020_channel_selections",
            "combine_channels",
            "read_vectorview_selection",
            "_SELECTIONS",
            "_EEG_SELECTIONS",
            "_divide_to_regions",
            "get_builtin_ch_adjacencies",
        ],
        "layout": [
            "Layout",
            "make_eeg_layout",
            "make_grid_layout",
            "read_layout",
            "find_layout",
            "generate_2d_layout",
        ],
        "montage": [
            "DigMontage",
            "get_builtin_montages",
            "make_dig_montage",
            "read_dig_dat",
            "read_dig_egi",
            "read_dig_captrak",
            "read_dig_fif",
            "read_dig_polhemus_isotrak",
            "read_polhemus_fastscan",
            "compute_dev_head_t",
            "make_standard_montage",
            "read_custom_montage",
            "read_dig_hpts",
            "read_dig_localite",
            "compute_native_head_t",
        ],
    },
)
