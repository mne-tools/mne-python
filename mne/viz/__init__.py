"""Visualization routines."""

from .topomap import (plot_evoked_topomap, plot_projs_topomap, plot_arrowmap,
                      plot_ica_components, plot_tfr_topomap, plot_topomap,
                      plot_epochs_psd_topomap, plot_layout)
from .topo import plot_topo_image_epochs, iter_topography
from .utils import (tight_layout, mne_analyze_colormap, compare_fiff,
                    ClickableImage, add_background_image, plot_sensors,
                    centers_to_edges, concatenate_images)
from ._3d import (plot_sparse_source_estimates, plot_source_estimates,
                  plot_vector_source_estimates, plot_evoked_field,
                  plot_dipole_locations, snapshot_brain_montage,
                  plot_head_positions, plot_alignment, plot_brain_colorbar,
                  plot_volume_source_estimates, plot_sensors_connectivity,
                  link_brains, set_3d_options)
from .misc import (plot_cov, plot_csd, plot_bem, plot_events,
                   plot_source_spectrogram, _get_presser,
                   plot_dipole_amplitudes, plot_ideal_filter, plot_filter,
                   adjust_axes, plot_chpi_snr)
from .evoked import (plot_evoked, plot_evoked_image, plot_evoked_white,
                     plot_snr_estimate, plot_evoked_topo,
                     plot_evoked_joint, plot_compare_evokeds)
from .circle import (plot_connectivity_circle, circular_layout,
                     plot_channel_labels_circle)
from .epochs import (plot_drop_log, plot_epochs, plot_epochs_psd,
                     plot_epochs_image)
from .raw import plot_raw, plot_raw_psd, plot_raw_psd_topo, _RAW_CLIP_DEF
from .ica import (plot_ica_scores, plot_ica_sources, plot_ica_overlay,
                  _plot_sources, plot_ica_properties)
from .montage import plot_montage
from .backends.renderer import (set_3d_backend, get_3d_backend, use_3d_backend,
                                set_3d_view, set_3d_title, create_3d_figure,
                                close_3d_figure, close_all_3d_figures,
                                get_brain_class)
from . import backends
from ._brain import Brain
from ._figure import (get_browser_backend, set_browser_backend,
                      use_browser_backend)
