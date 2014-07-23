"""Visualization routines
"""

from .topomap import plot_evoked_topomap, plot_projs_topomap
from .topomap import plot_ica_components, plot_ica_topomap
from .topomap import plot_tfr_topomap, plot_topomap
from .topo import (plot_topo, plot_topo_tfr, plot_topo_image_epochs,
                   iter_topography)
from .utils import tight_layout, mne_analyze_colormap, compare_fiff
from ._3d import plot_sparse_source_estimates, plot_source_estimates
from ._3d import plot_trans, plot_evoked_field
from .misc import plot_cov, plot_bem, plot_events
from .misc import plot_source_spectrogram
from .utils import _mutable_defaults
from .evoked import plot_evoked, plot_evoked_image
from .circle import plot_connectivity_circle, circular_layout
from .epochs import plot_image_epochs, plot_drop_log, plot_epochs
from .epochs import _drop_log_stats
from .raw import plot_raw, plot_raw_psds
from .ica import plot_ica_scores, plot_ica_sources, plot_ica_overlay
