"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import faulthandler
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import sphinx
from intersphinx_registry import get_intersphinx_mapping
from numpydoc import docscrape
from sphinx.config import is_serializable
from sphinx.domains.changeset import versionlabels
from sphinx_gallery.sorting import ExplicitOrder
from yaml import safe_load

import mne
import mne.html_templates._templates
from mne.utils import (
    linkcode_resolve,
    run_subprocess,
)

assert linkcode_resolve is not None  # avoid flake warnings, used by numpydoc
matplotlib.use("agg")
faulthandler.enable()
os.environ["_MNE_BROWSER_NO_BLOCK"] = "true"
os.environ["MNE_BROWSER_OVERVIEW_MODE"] = "hidden"
os.environ["MNE_BROWSER_THEME"] = "light"
os.environ["MNE_3D_OPTION_THEME"] = "light"
# https://numba.readthedocs.io/en/latest/reference/deprecation.html#deprecation-of-old-style-numba-captured-errors  # noqa: E501
os.environ["NUMBA_CAPTURED_ERRORS"] = "new_style"
mne.html_templates._templates._COLLAPSED = True  # collapse info _repr_html_

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
curpath = Path(__file__).parent.resolve(strict=True)
sys.path.append(str(curpath / "sphinxext"))

from build_lite_wheel import build_wheel, find_wheels  # noqa: E402
from credit_tools import generate_credit_rst  # noqa: E402
from jupyterlite_lite_renderer import LITE_RENDERER_CELL  # noqa: E402
from mne_doc_utils import report_scraper, reset_warnings, sphinx_logger  # noqa: E402

# -- Project information -----------------------------------------------------

project = "MNE"
td = datetime.now(tz=timezone.utc)

# We need to triage which date type we use so that incremental builds work
# (Sphinx looks at variable changes and rewrites all files if some change)
project_copyright = (
    f'2012–{td.year}, MNE Developers. Last updated <time datetime="{td.isoformat()}" class="localized">{td.strftime("%Y-%m-%d %H:%M %Z")}</time>.\n'  # noqa: E501
    """<script type="text/javascript">
function formatTimestamp() {
    document.querySelectorAll("time.localized").forEach(el => {
        const d = new Date(el.getAttribute("datetime"));
        el.textContent = d.toLocaleString("sv-SE", { "timeZoneName": "short" });
    });
}
if (document.readyState !== "loading") {
    formatTimestamp();
} else {
    document.addEventListener("DOMContentLoaded", formatTimestamp);
}
</script>"""
)
if os.getenv("MNE_FULL_DATE", "false").lower() != "true":
    project_copyright = f"2012–{td.year}, MNE Developers. Last updated locally."

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The full version, including alpha/beta/rc tags.
release = mne.__version__
sphinx_logger.info(f"Building documentation for MNE {release} ({mne.__file__})")
# The short X.Y version.
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "6.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # builtin
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.graphviz",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    # contrib
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "jupyterlite_sphinx",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.youtube",
    "sphinxcontrib.towncrier.ext",
    # homegrown
    "contrib_avatars",
    "gen_commands",
    "gen_names",
    "gh_substitutions",
    "mne_substitutions",
    "newcontrib_substitutions",
    "unit_role",
    "related_software",
    "directive_formatting",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.

# NB: changes here should also be made to the linkcheck target in the Makefile
exclude_patterns = [
    "_includes",
    "changes/dev",
    "jupyterlite_contents",
    "lite_extra",
    "pypi",
    "corrupt_*",
]

# The suffix of source filenames.
source_suffix = ".rst"

# The main toctree document.
master_doc = "index"

# List of documents that shouldn't be included in the build.
unused_docs = []

# List of directories, relative to source directory, that shouldn't be searched
# for source files.
exclude_trees = ["_build"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "py:obj"

# A list of ignored prefixes for module index sorting.
modindex_common_prefix = ["mne."]

# -- Sphinx-Copybutton configuration -----------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- sphinxcontrib-towncrier configuration -----------------------------------

towncrier_draft_working_directory = str(curpath.parent)

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    # More niche so didn't upstream to intersphinx_registry
    "nitime": ("https://nipy.org/nitime/", None),
    "mne_bids": ("https://mne.tools/mne-bids/stable", None),
    "mne-connectivity": ("https://mne.tools/mne-connectivity/stable", None),
    "mne-gui-addons": ("https://mne.tools/mne-gui-addons", None),
    "picard": ("https://mind-inria.github.io/picard/", None),
    "eeglabio": ("https://eeglabio.readthedocs.io/en/latest", None),
    "pybv": ("https://pybv.readthedocs.io/en/latest", None),
}
intersphinx_mapping.update(
    get_intersphinx_mapping(
        packages=set(
            """
imageio matplotlib numpy pandas python scipy statsmodels sklearn numba joblib nibabel
seaborn patsy pyvista dipy nilearn pyqtgraph
""".strip().split()
        ),
    )
)
# Broken as of 2026/06/08 (https://github.com/joblib/joblib/issues/1796)
intersphinx_mapping["joblib"] = ("https://joblib.readthedocs.io/en/stable", None)


# NumPyDoc configuration -----------------------------------------------------

# Define what extra methods numpydoc will document
docscrape.ClassDoc.extra_public_methods = mne.utils._doc_special_members
numpydoc_class_members_toctree = False
numpydoc_show_inherited_class_members = {
    "mne.Forward": False,
    "mne.Projection": False,
    "mne.SourceSpaces": False,
}
numpydoc_attributes_as_param_list = True
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    # Python
    "file-like": ":term:`file-like <python:file object>`",
    "iterator": ":term:`iterator <python:iterator>`",
    "path-like": ":term:`path-like`",
    "array-like": ":term:`array_like <numpy:array_like>`",
    "Path": ":class:`python:pathlib.Path`",
    "bool": ":ref:`bool <python:typebool>`",
    # Matplotlib
    "colormap": ":ref:`colormap <matplotlib:colormaps>`",
    "color": ":doc:`color <matplotlib:api/colors_api>`",
    "Axes": "matplotlib.axes.Axes",
    "Figure": "matplotlib.figure.Figure",
    "Axes3D": "mpl_toolkits.mplot3d.axes3d.Axes3D",
    "ColorbarBase": "matplotlib.colorbar.ColorbarBase",
    # sklearn
    "LeaveOneOut": "sklearn.model_selection.LeaveOneOut",
    "MetadataRequest": "sklearn.utils.metadata_routing.MetadataRequest",
    "estimator": "sklearn.base.BaseEstimator",
    # joblib
    "joblib.Parallel": "joblib.Parallel",
    # nibabel
    "Nifti1Image": "nibabel.nifti1.Nifti1Image",
    "Nifti2Image": "nibabel.nifti2.Nifti2Image",
    "SpatialImage": "nibabel.spatialimages.SpatialImage",
    # MNE
    "Label": "mne.Label",
    "Forward": "mne.Forward",
    "Evoked": "mne.Evoked",
    "Info": "mne.Info",
    "SourceSpaces": "mne.SourceSpaces",
    "Epochs": "mne.Epochs",
    "Layout": "mne.channels.Layout",
    "EvokedArray": "mne.EvokedArray",
    "BiHemiLabel": "mne.BiHemiLabel",
    "AverageTFR": "mne.time_frequency.AverageTFR",
    "AverageTFRArray": "mne.time_frequency.AverageTFRArray",
    "EpochsTFR": "mne.time_frequency.EpochsTFR",
    "EpochsTFRArray": "mne.time_frequency.EpochsTFRArray",
    "RawTFR": "mne.time_frequency.RawTFR",
    "RawTFRArray": "mne.time_frequency.RawTFRArray",
    "Raw": "mne.io.Raw",
    "ICA": "mne.preprocessing.ICA",
    "Covariance": "mne.Covariance",
    "Annotations": "mne.Annotations",
    "DigMontage": "mne.channels.DigMontage",
    "VectorSourceEstimate": "mne.VectorSourceEstimate",
    "VolSourceEstimate": "mne.VolSourceEstimate",
    "VolVectorSourceEstimate": "mne.VolVectorSourceEstimate",
    "MixedSourceEstimate": "mne.MixedSourceEstimate",
    "MixedVectorSourceEstimate": "mne.MixedVectorSourceEstimate",
    "SourceEstimate": "mne.SourceEstimate",
    "Projection": "mne.Projection",
    "ConductorModel": "mne.bem.ConductorModel",
    "Dipole": "mne.Dipole",
    "DipoleFixed": "mne.DipoleFixed",
    "InverseOperator": "mne.minimum_norm.InverseOperator",
    "CrossSpectralDensity": "mne.time_frequency.CrossSpectralDensity",
    "SourceMorph": "mne.SourceMorph",
    "Xdawn": "mne.preprocessing.Xdawn",
    "Report": "mne.Report",
    "TimeDelayingRidge": "mne.decoding.TimeDelayingRidge",
    "Vectorizer": "mne.decoding.Vectorizer",
    "UnsupervisedSpatialFilter": "mne.decoding.UnsupervisedSpatialFilter",
    "TemporalFilter": "mne.decoding.TemporalFilter",
    "SSD": "mne.decoding.SSD",
    "Scaler": "mne.decoding.Scaler",
    "SPoC": "mne.decoding.SPoC",
    "PSDEstimator": "mne.decoding.PSDEstimator",
    "LinearModel": "mne.decoding.LinearModel",
    "FilterEstimator": "mne.decoding.FilterEstimator",
    "EMS": "mne.decoding.EMS",
    "CSP": "mne.decoding.CSP",
    "Beamformer": "mne.beamformer.Beamformer",
    "Transform": "mne.transforms.Transform",
    "Coregistration": "mne.coreg.Coregistration",
    "Figure3D": "mne.viz.Figure3D",
    "EOGRegression": "mne.preprocessing.EOGRegression",
    "Spectrum": "mne.time_frequency.Spectrum",
    "EpochsSpectrum": "mne.time_frequency.EpochsSpectrum",
    "EpochsFIF": "mne.Epochs",
    "EpochsEEGLAB": "mne.Epochs",
    "EpochsKIT": "mne.Epochs",
    "BaseRaw": "mne.io.Raw",
    "RawANT": "mne.io.Raw",
    "RawArtemis123": "mne.io.Raw",
    "RawBCI2k": "mne.io.Raw",
    "RawBDF": "mne.io.Raw",
    "RawBOXY": "mne.io.Raw",
    "RawBrainVision": "mne.io.Raw",
    "RawBTi": "mne.io.Raw",
    "RawCNT": "mne.io.Raw",
    "RawCTF": "mne.io.Raw",
    "RawCurry": "mne.io.Raw",
    "RawEDF": "mne.io.Raw",
    "RawEEGLAB": "mne.io.Raw",
    "RawEGI": "mne.io.Raw",
    "RawEximia": "mne.io.Raw",
    "RawEyelink": "mne.io.Raw",
    "RawFIL": "mne.io.Raw",
    "RawGDF": "mne.io.Raw",
    "RawHitachi": "mne.io.Raw",
    "RawKIT": "mne.io.Raw",
    "RawNedf": "mne.io.Raw",
    "RawNeuralynx": "mne.io.Raw",
    "RawNicolet": "mne.io.Raw",
    "RawNihon": "mne.io.Raw",
    "RawNSX": "mne.io.Raw",
    "RawMEF": "mne.io.Raw",
    "RawNIRX": "mne.io.Raw",
    "RawPersyst": "mne.io.Raw",
    "RawSNIRF": "mne.io.Raw",
    "Calibration": "mne.preprocessing.eyetracking.Calibration",
    # dipy
    "dipy.align.AffineMap": "dipy.align.imaffine.AffineMap",
    "dipy.align.DiffeomorphicMap": "dipy.align.imwarp.DiffeomorphicMap",
}
numpydoc_xref_ignore = {
    # words
    "and",
    "as",
    "between",
    "class",
    "data",
    "instance",
    "instances",
    "input",
    "of",
    "default",
    "same",
    "shape",
    "or",
    "the",
    "with",
    "length",
    "pair",
    "matplotlib",
    "optional",
    "kwargs",
    "in",
    "dtype",
    "object",
    # shapes
    "n_vertices",
    "n_faces",
    "n_channels",
    "m",
    "n",
    "n_events",
    "n_colors",
    "n_times",
    "obj",
    "n_chan",
    "n_epochs",
    "n_picks",
    "n_ch_groups",
    "n_dipoles",
    "n_ica_components",
    "n_pos",
    "n_node_names",
    "n_tapers",
    "n_signals",
    "n_step",
    "n_freqs",
    "wsize",
    "Tx",
    "M",
    "N",
    "p",
    "q",
    "r",
    "n_observations",
    "n_regressors",
    "n_cols",
    "n_frequencies",
    "n_tests",
    "n_samples",
    "n_peaks",
    "n_permutations",
    "nchan",
    "n_points",
    "n_features",
    "n_parts",
    "n_features_new",
    "n_components",
    "n_labels",
    "n_events_in",
    "n_splits",
    "n_scores",
    "n_outputs",
    "n_trials",
    "n_estimators",
    "n_tasks",
    "nd_features",
    "n_classes",
    "n_targets",
    "n_slices",
    "n_hpi",
    "n_fids",
    "n_elp",
    "n_pts",
    "n_tris",
    "n_nodes",
    "n_nonzero",
    "n_events_out",
    "n_segments",
    "n_orient_inv",
    "n_orient_fwd",
    "n_orient",
    "n_dipoles_lcmv",
    "n_dipoles_fwd",
    "n_picks_ref",
    "n_coords",
    "n_meg",
    "n_good_meg",
    "n_moments",
    "n_patterns",
    "n_new_events",
    # sklearn subclasses
    "mapping",
    "to",
    "any",
    "pandas",
    "polars",
    "default",
    # unlinkable
    "_Renderer",
    "n_triangles",
    "CoregistrationUI",
    "mne_qt_browser.figure.MNEQtBrowser",
    # pooch, since its website is unreliable and users will rarely need the links
    "pooch.Unzip",
    "pooch.Untar",
    "pooch.HTTPDownloader",
}
numpydoc_validate = True
try:
    import tomllib
    # TODO VERSION: Can be removed once Python 3.11 is required
except Exception:
    pass
else:
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text("utf-8"))
    pyproject_nv = pyproject["tool"]["numpydoc_validation"]
    numpydoc_validation_checks = set(pyproject_nv["checks"])
    numpydoc_validation_exclude = set(pyproject_nv["exclude"])


# -- Sphinx-gallery configuration --------------------------------------------

examples_dirs = ["../tutorials", "../examples"]
gallery_dirs = ["auto_tutorials", "auto_examples"]
os.environ["_MNE_BUILDING_DOC"] = "true"

scrapers = (
    "matplotlib",
    "mne_doc_utils.gui_scraper",
    "mne_doc_utils.brain_scraper",
    "pyvista",
    "mne_doc_utils.report_scraper",
    "mne_doc_utils.mne_qt_browser_scraper",
)

compress_images = ("images", "thumbnails")
# let's make things easier on Windows users
# (on Linux and macOS it's easy enough to require this)
if sys.platform.startswith("win"):
    try:
        subprocess.check_call(["optipng", "--version"])
    except Exception:
        compress_images = ()

sphinx_gallery_parallel = int(os.getenv("MNE_DOC_BUILD_N_JOBS", "1"))
jupyterlite_contents = ["jupyterlite_contents"]
jupyterlite_bind_ipynb_suffix = False

# Inject the required subset of MNE-sample-data for JupyterLite. The data is
# placed under doc/lite_extra/mne_data and served at the docs root via
# html_extra_path (added below). The JupyterLite setup cell fetches these
# files over HTTP into the Pyodide kernel — the /drive virtual-filesystem
# bridge needs cross-origin-isolation (COOP/COEP) headers that static
# artifact servers (e.g. CircleCI) do not send, so it is unusable there.
# lite_data (mne.datasets.lite_data) extracts the curated subset here, with the
# files under their original dataset folders (MNE-sample-data/, ...).
mne_data_base = Path(os.path.expanduser("~/mne_data"))
lite_root = mne_data_base / "MNE-lite-data"
src_sample_data = lite_root / "MNE-sample-data"
lite_extra_base = (
    Path(os.path.abspath(os.path.dirname(__file__))) / "lite_extra" / "mne_data"
)
dst_sample_data = lite_extra_base / "MNE-sample-data"
dst_sample_data.mkdir(parents=True, exist_ok=True)


def _lite_src(folder, rel):
    """Return where a dataset file can be read from, or None if nowhere.

    The curated lite_data archive only carries the files it was published with,
    so look in whatever CI restored of the real dataset first and fall back to
    the archive. Sourcing from the archive alone means anything added since it
    was last uploaded goes missing without the build failing.
    """
    for root in (mne_data_base / folder, lite_root / folder):
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None


print(
    f"[JupyterLite] Sample data: real dataset="
    f"{(mne_data_base / 'MNE-sample-data').exists()}, "
    f"curated archive={src_sample_data.exists()}"
)
if (mne_data_base / "MNE-sample-data").exists() or src_sample_data.exists():
    required_files = [
        "version.txt",
        "MEG/sample/sample_audvis_raw.fif",
        "MEG/sample/sample_audvis_filt-0-40_raw.fif",
        "MEG/sample/sample_audvis_raw-eve.fif",
        "MEG/sample/sample_audvis_filt-0-40_raw-eve.fif",
        "MEG/sample/sample_audvis_ecg-proj.fif",
        "MEG/sample/sample_audvis-ave.fif",
        "MEG/sample/sample_audvis-cov.fif",
        "MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif",
        "MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif",
        "MEG/sample/sample_audvis-meg-oct-6-fwd.fif",
        "MEG/sample/sample_audvis-meg-oct-6-meg-fixed-inv.fif",
        "MEG/sample/ernoise_raw.fif",
        "MEG/sample/sample_audvis-no-filter-ave.fif",
        "MEG/sample/sample_audvis_raw-trans.fif",
        "MEG/sample/sample_audvis-shrunk-cov.fif",
        "MEG/sample/sample_audvis-meg-lh.stc",
        "MEG/sample/sample_audvis-meg-rh.stc",
        "MEG/sample/sample_audvis-meg-eeg-lh.stc",
        "MEG/sample/sample_audvis-meg-eeg-rh.stc",
        "MEG/sample/sample_audvis_ecg-eve.fif",
        # Maxwell-filter calibration pair, read from inside maxwell_filter
        # rather than through a shimmable reader (86 KB, so fetched eagerly)
        "SSS/sss_cal_mgh.dat",
        "SSS/ct_sparse_mgh.fif",
        "subjects/sample/mri/T1.mgz",
        "subjects/sample/mri/aseg.mgz",
        # read_talxfm builds this path itself, so nothing in the tutorials
        # names it; plot_alignment needs it to estimate MRI fiducials
        "subjects/sample/mri/transforms/talairach.xfm",
        "subjects/sample/bem/sample-oct-6-src.fif",
        # Head and skull surfaces for plot_alignment. outer_skin.surf is what
        # MNE picks first, so serving it makes the browser figure match the
        # rendered docs; sample-head.fif is the later fallback. There is no
        # sample-head-dense.fif in the dataset -- lh.seghead is the documented
        # second candidate for the dense surface. (These three .surf paths are
        # symlinks into bem/flash/, and copy2 follows them.)
        "subjects/sample/bem/outer_skin.surf",
        "subjects/sample/bem/outer_skull.surf",
        "subjects/sample/bem/inner_skull.surf",
        "subjects/sample/bem/sample-head.fif",
        "subjects/sample/surf/lh.seghead",
        # single-layer BEM solution (the 3-layer one is 237 MB, so notebooks
        # needing that are excluded instead)
        "subjects/sample/bem/sample-5120-bem-sol.fif",
        # fsaverage source space, used by the morphing and cluster-stats
        # notebooks; it ships inside MNE-sample-data
        "subjects/fsaverage/bem/fsaverage-ico-5-src.fif",
        "subjects/sample/surf/rh.pial",
        "subjects/sample/surf/lh.pial",
        "subjects/sample/surf/rh.white",
        "subjects/sample/surf/lh.white",
        "subjects/sample/surf/rh.inflated",
        "subjects/sample/surf/lh.inflated",
        "subjects/sample/surf/rh.curv",
        "subjects/sample/surf/lh.curv",
        # setup_source_space maps each hemisphere onto its sphere for any
        # ico/oct spacing, and _create_surf_spacing reads surf/{hemi}.sphere
        # by a path it builds itself (5.6 MB each)
        "subjects/sample/surf/lh.sphere",
        "subjects/sample/surf/rh.sphere",
        "subjects/sample/label/lh.aparc.annot",
        "subjects/sample/label/rh.aparc.annot",
        # the auditory/visual ROIs; about nine notebooks build these names with
        # an f-string, so a scan of the tutorial text never sees them
        "MEG/sample/labels/Aud-lh.label",
        "MEG/sample/labels/Aud-rh.label",
        "MEG/sample/labels/Vis-lh.label",
        "MEG/sample/labels/Vis-rh.label",
    ]
    for req in required_files:
        s = _lite_src("MNE-sample-data", req)
        d = dst_sample_data / req
        if s is not None:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
            print(f"[JupyterLite]   Copied: {req}")
        else:
            print(f"[JupyterLite]   MISSING: {req}")


# Also inject SSVEP and EEGLAB testing datasets for JupyterLite
lite_data_base = lite_extra_base
lite_data_base.mkdir(parents=True, exist_ok=True)

src_ssvep = mne_data_base / "ssvep-example-data"
dst_ssvep = lite_data_base / "ssvep-example-data"
print(f"[JupyterLite] SSVEP data source exists: {src_ssvep.exists()}")
if src_ssvep.exists() and not dst_ssvep.exists():
    shutil.copytree(src_ssvep, dst_ssvep, dirs_exist_ok=True)
    print("[JupyterLite]   Copied ssvep-example-data")

src_eeglab = mne_data_base / "MNE-testing-data" / "EEGLAB"
dst_eeglab = lite_data_base / "MNE-testing-data" / "EEGLAB"
print(f"[JupyterLite] EEGLAB data source exists: {src_eeglab.exists()}")
if src_eeglab.exists() and not dst_eeglab.exists():
    shutil.copytree(src_eeglab, dst_eeglab, dirs_exist_ok=True)
    print("[JupyterLite]   Copied MNE-testing-data/EEGLAB")

# The head-position and Maxwell-filtering tutorials read one continuous
# movement recording out of the testing dataset. CI already restores it from
# data-cache-testing, so only these two files are copied, not the 1.6 GB set.
testing_files = [
    "SSS/test_move_anon_raw.fif",
    "SSS/test_move_anon_raw.pos",
]
for testing_file in testing_files:
    s = _lite_src("MNE-testing-data", testing_file)
    d = lite_data_base / "MNE-testing-data" / testing_file
    if s is not None and not d.exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        _mb = s.stat().st_size / 1e6
        print(f"[JupyterLite]   Copied {testing_file} ({_mb:.1f} MB)")
    elif s is None:
        print(f"[JupyterLite]   MISSING {testing_file}")

# The remaining datasets are each used by one or two notebooks that read only a
# couple of files out of them. CI already downloads all of these in
# tools/circleci_download.sh, so copying is free -- but their sizes vary a lot,
# so refuse anything past this limit rather than bloat the artifact. For scale,
# the largest file already served (sample_audvis_raw.fif) is 128 MB.
LITE_MAX_FILE_MB = 150


def _lite_copy(folder, rel_paths):
    """Copy selected files of a dataset into the served tree."""
    for rel in rel_paths:
        s = _lite_src(folder, rel)
        if s is None:
            print(f"[JupyterLite]   MISSING {folder}/{rel}")
            continue
        size_mb = s.stat().st_size / 1e6
        if size_mb > LITE_MAX_FILE_MB:
            print(f"[JupyterLite]   SKIPPED {folder}/{rel} ({size_mb:.1f} MB)")
            continue
        d = lite_data_base / folder / rel
        if not d.exists():
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
            print(f"[JupyterLite]   Copied {folder}/{rel} ({size_mb:.1f} MB)")


def _lite_copy_tree(folder, rel_dir):
    """Copy a directory-shaped recording, leaving a manifest for the browser.

    read_raw_nirx and read_raw_egi are handed a folder rather than a file, so
    the setup cell has no way to know what to fetch without a listing.
    """
    src = mne_data_base / folder / rel_dir
    if not src.is_dir():
        print(f"[JupyterLite]   MISSING {folder}/{rel_dir}")
        return
    names, total_mb = [], 0.0
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        # zero-byte members (an .mff carries a couple of lock files) do not
        # survive the artifact upload, so listing them only yields a 404
        if f.stat().st_size == 0:
            continue
        size_mb = f.stat().st_size / 1e6
        if size_mb > LITE_MAX_FILE_MB:
            print(f"[JupyterLite]   SKIPPED {folder}/{rel_dir} ({size_mb:.1f} MB)")
            return
        names.append(str(f.relative_to(src)))
        total_mb += size_mb
    dst = lite_data_base / folder / rel_dir
    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        d = dst / name
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / name, d)
    (dst / "_lite_manifest.txt").write_text("\n".join(names))
    print(
        f"[JupyterLite]   Copied {folder}/{rel_dir} "
        f"({len(names)} files, {total_mb:.1f} MB)"
    )


_lite_copy(
    "MNE-misc-data",
    [
        "xdf/sub-P001_ses-S004_task-Default_run-001_eeg_a2.xdf",
        "movement/simulated_quats.pos",
        "movement/simulated_movement_raw.fif",
        "movement/simulated_stationary_raw.fif",
        "eyetracking/eyelink/px_textpage_ws.asc",
        "eyetracking/eyelink/HREF_textpage_ws.asc",
    ],
)
_lite_copy(
    "MNE-eyelink-data",
    [
        "freeviewing/sub-01_task-freeview_eyetrack.asc",
        "freeviewing/stim/naturalistic.png",
        "eeg-et/sub-01_task-plr_eyetrack.asc",
    ],
)
_lite_copy_tree("MNE-eyelink-data", "eeg-et/sub-01_task-plr_eeg.mff")
_lite_copy_tree("MNE-fNIRS-motor-data", "Participant-1")

# The logging tutorial reads a KIT file that lives inside the package itself,
# under mne/io/kit/tests/. pyproject excludes "/mne/**/tests" from the wheel, so
# it is absent from the browser kernel -- serve it and let the setup cell stage
# it back into the path the tutorial builds.
_kit_src = Path(mne.__file__).parent / "io" / "kit" / "tests" / "data" / "test.sqd"
_kit_dst = lite_data_base / "MNE-kit-testdata" / "test.sqd"
if _kit_src.exists():
    if not _kit_dst.exists():
        _kit_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_kit_src, _kit_dst)
    print(
        f"[JupyterLite]   Copied MNE-kit-testdata/test.sqd "
        f"({_kit_src.stat().st_size / 1e6:.1f} MB)"
    )
else:
    print("[JupyterLite]   MISSING MNE-kit-testdata/test.sqd")

_lite_copy("MNE-phantom-kernel-data", ["phantom_32_100nam_raw.fif"])
_lite_copy("MNE-multimodal-data", ["multimodal_raw.fif"])
_lite_copy("MNE-refmeg-noise-data", ["sample_reference_MEG_noise-raw.fif"])

# somato is deliberately not served: its raw alone is 344 MB and the six
# notebooks that read it are on the exclude list instead.

# Inject the single needed file(s) from extra datasets used by the Epochs and
# decoding examples. Sizes are all within what we already serve
# (sample_audvis_raw.fif is 128.5 MB): kiloword 28.7 MB, erp_core 123.6 MB,
# mtrf speech_data.mat 17.2 MB, eegbci 3x2.6 MB. The CI "Ensure ... data" step
# downloads them so the sources exist here.
for _folder, _ds_files in (
    ("MNE-kiloword-data", ["kword_metadata-epo.fif"]),
    ("MNE-ERP-CORE-data", ["ERP-CORE_Subject-001_Task-Flankers_eeg.fif"]),
    ("mTRF_1.5", ["speech_data.mat"]),
    (
        "MNE-eegbci-data",
        # exactly the runs tools/circleci_download.sh fetches: subject 1 runs
        # 3/6/10/14 and run 3 for subjects 2-4. Notebooks wanting run 1 or 2 are
        # excluded instead, since that data never reaches the CI box.
        [
            "files/eegmmidb/1.0.0/S001/S001R03.edf",
            "files/eegmmidb/1.0.0/S001/S001R06.edf",
            "files/eegmmidb/1.0.0/S001/S001R10.edf",
            "files/eegmmidb/1.0.0/S001/S001R14.edf",
            "files/eegmmidb/1.0.0/S002/S002R03.edf",
            "files/eegmmidb/1.0.0/S003/S003R03.edf",
            "files/eegmmidb/1.0.0/S004/S004R03.edf",
        ],
    ),
):
    _dst_ds = lite_data_base / _folder
    for _ds_file in _ds_files:
        s = _lite_src(_folder, _ds_file)
        d = _dst_ds / _ds_file
        if s is not None:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
            print(f"[JupyterLite]   Copied: {_folder}/{_ds_file}")
        else:
            print(f"[JupyterLite]   MISSING: {_folder}/{_ds_file}")


# Provide the development MNE wheel so JupyterLite installs the current version
# rather than the older release from PyPI. ``doc/sphinxext/build_lite_wheel.py``
# builds it into ``doc/pypi``, where the jupyterlite-pyodide-kernel PipliteAddon
# discovers and indexes it. Running that script before the docs build (in CI or
# locally) means Sphinx reuses the wheel instead of rebuilding it on every
# invocation; if none is present we build it here, so the docs build never
# depends on the pre-step having run.
_lite_wheels = find_wheels() or build_wheel()
sphinx_logger.info(f"[JupyterLite] MNE wheel for the browser kernel: {_lite_wheels}")

sphinx_gallery_conf = {
    "jupyterlite": {
        "use_jupyter_lab": True,
        "jupyterlite_contents": "jupyterlite_contents",
        # named rather than passed: sphinx_gallery_conf has to stay
        # JSON-serializable (see the is_serializable assert below), so
        # sphinx-gallery imports this dotted path itself
        "notebook_modification_function": (
            "jupyterlite_cell_notes.note_unrunnable_cells"
        ),
    },
    "first_notebook_cell": (
        "# 💡 This cell is automatically added to the start of each notebook.\n"
        "# It installs MNE and patches the browser environment for Pyodide.\n"
        "import piplite\n"
        "# Use piplite (not micropip) so the locally-built development MNE wheel\n"
        "# bundled into the JupyterLite build is preferred over the older PyPI\n"
        "# release;\n"
        "# piplite checks the local index first and falls back to PyPI for deps.\n"
        "# keep_going=True lets it install even if Pyodide's bundled\n"
        "# matplotlib/scipy/numpy are older than MNE's declared minimums.\n"
        "await piplite.install(\n"
        "    ['mne', 'scikit-learn', 'joblib', 'pandas', 'seaborn', "
        "'mne-connectivity', 'nibabel', 'pyvista-js', 'pyxdf', 'mffpy', "
        "'python-picard'],\n"
        "    keep_going=True,\n"
        ")\n"
        "\n"
        "import sys\n"
        "import os\n"
        "import io\n"
        "\n"
        "# lzma: try real stdlib first (Pyodide ships it); only mock if absent\n"
        "try:\n"
        "    import lzma\n"
        "except ImportError:\n"
        "    class _LZMAFile:\n"
        "        def __init__(self, *a, **kw): pass\n"
        "        def __enter__(self): return self\n"
        "        def __exit__(self, *a): pass\n"
        "        def write(self, d): pass\n"
        "        def read(self, n=-1): return b''\n"
        "        def close(self): pass\n"
        "    class _MockLZMA:\n"
        "        LZMAError = Exception\n"
        "        LZMAFile = _LZMAFile\n"
        "        FORMAT_XZ = 1\n"
        "        FORMAT_ALONE = 2\n"
        "        def __getattr__(self, name): return object\n"
        "    import sys as _sys\n"
        "    _sys.modules['lzma'] = _MockLZMA()\n"
        "\n"
        "# Mock multiprocessing — missing in Pyodide but imported by joblib\n"
        "from unittest.mock import MagicMock\n"
        "if 'multiprocessing' not in sys.modules:\n"
        "    m = MagicMock()\n"
        "    m.cpu_count.return_value = 1\n"
        "    sys.modules['multiprocessing'] = m\n"
        "    sys.modules['multiprocessing.util'] = m.util\n"
        "    sys.modules['multiprocessing.pool'] = m.pool\n"
        "\n"
        "# Patch requests so pooch can fetch files already on /drive/mne_data.\n"
        "# open_url works for both text and binary in Pyodide >= 0.21.\n"
        "import requests\n"
        "import pyodide\n"
        "orig_send = requests.Session.send\n"
        "def pyodide_send(self, request, **kwargs):\n"
        "    try:\n"
        "        buf = pyodide.http.open_url(request.url)\n"
        "        content = buf.getvalue() if hasattr(buf, 'getvalue') else buf.read()\n"
        "        if isinstance(content, str):\n"
        "            content = content.encode('utf-8')\n"
        "    except Exception as e:\n"
        "        print(f'open_url failed for {request.url}: {e}')\n"
        "        return orig_send(self, request, **kwargs)\n"
        "    response = requests.Response()\n"
        "    response.status_code = 200\n"
        "    response.url = request.url\n"
        "    response.raw = io.BytesIO(content)\n"
        "    return response\n"
        "requests.Session.send = pyodide_send\n"
        "\n"
        "# /drive/ in Pyodide requires Cross-Origin-Isolation headers\n"
        "# (COOP/COEP) which many static servers (e.g. CircleCI artifacts)\n"
        "# do not send. Fetch the data over HTTP into /tmp/mne_data instead\n"
        "# — same-origin, no CORS. The data is served at the docs root\n"
        "# (/mne_data/...) via Sphinx html_extra_path.\n"
        "# Pyodide may run in a web worker (no `window`); `location` exists\n"
        "# in both the main thread and workers, so use it to find the docs\n"
        "# root by splitting on '/lite/'.\n"
        "import pyodide.http as _phttp\n"
        "import js as _js\n"
        "try:\n"
        "    _page = str(_js.location.href)\n"
        "except Exception:\n"
        "    _page = str(_js.window.location.href)\n"
        "_base = _page.split('/lite/')[0] + '/mne_data/'\n"
        "mne_data_path = '/tmp/mne_data'\n"
        "_sample_dir = mne_data_path + '/MNE-sample-data'\n"
        "# Eager 'core': small, commonly-used sample files fetched once at\n"
        "# notebook start. The heavy files (raw / filt raw / ernoise / fwd /\n"
        "# inv / src, ~360 MB total) are intentionally omitted here -- they are\n"
        "# fetched lazily on first read via the reader shims below, so each\n"
        "# notebook only downloads the sample files it actually uses.\n"
        "_sample_files = [\n"
        "    'version.txt',\n"
        "    'MEG/sample/sample_audvis_raw-eve.fif',\n"
        "    'MEG/sample/sample_audvis_filt-0-40_raw-eve.fif',\n"
        "    'MEG/sample/sample_audvis_ecg-proj.fif',\n"
        "    'MEG/sample/sample_audvis-cov.fif',\n"
        "    'MEG/sample/sample_audvis-ave.fif',\n"
        "    'MEG/sample/sample_audvis-no-filter-ave.fif',\n"
        "    'MEG/sample/sample_audvis_raw-trans.fif',\n"
        "    'MEG/sample/sample_audvis-shrunk-cov.fif',\n"
        "    'MEG/sample/sample_audvis-meg-lh.stc',\n"
        "    'MEG/sample/sample_audvis-meg-rh.stc',\n"
        "    'subjects/sample/mri/T1.mgz',\n"
        "    'subjects/sample/surf/rh.pial',\n"
        "    'subjects/sample/surf/lh.pial',\n"
        "    'subjects/sample/surf/rh.white',\n"
        "    'subjects/sample/surf/lh.white',\n"
        "    'subjects/sample/label/lh.aparc.annot',\n"
        "    'subjects/sample/label/rh.aparc.annot',\n"
        "    'SSS/sss_cal_mgh.dat',\n"
        "    'SSS/ct_sparse_mgh.fif',\n"
        "]\n"
        "print('Fetching MNE sample data (once per session)...')\n"
        "for _f in _sample_files:\n"
        "    _dst = _sample_dir + '/' + _f\n"
        "    if os.path.exists(_dst):\n"
        "        continue\n"
        "    _url = _base + 'MNE-sample-data/' + _f\n"
        "    try:\n"
        "        _r = await _phttp.pyfetch(_url)\n"
        "        if _r.status != 200:\n"
        "            print(f'  HTTP {_r.status} for {_url}')\n"
        "            continue\n"
        "        _d = await _r.bytes()\n"
        "        if _d[:4] == b'<!DO' or _d[:5] == b'<html':\n"
        "            print(f'  skipped {_f} (server returned HTML)')\n"
        "            continue\n"
        "        os.makedirs(os.path.dirname(_dst), exist_ok=True)\n"
        "        with open(_dst, 'wb') as _fh:\n"
        "            _fh.write(_d)\n"
        "    except Exception as _e:\n"
        "        print(f'  failed to fetch {_f}: {_e}')\n"
        "os.makedirs(mne_data_path, exist_ok=True)\n"
        "os.environ['MNE_DATA'] = mne_data_path\n"
        "os.environ['MNE_DATASETS_SAMPLE_PATH'] = mne_data_path\n"
        "\n"
        "# Block pooch from attempting large OSF downloads in the browser.\n"
        "# The required files are either pre-injected or unavailable.\n"
        "import pooch\n"
        "orig_pooch_fetch = pooch.Pooch.fetch\n"
        "def pyodide_pooch_fetch(self, fname, processor=None, downloader=None):\n"
        "    url = self.get_url(fname)\n"
        "    if 'osf.io' in url or 'files.osf.io' in url:\n"
        "        raise RuntimeError(\n"
        "            f'Cannot download {fname!r} from OSF in JupyterLite: '\n"
        "            'browser CORS policy and memory limits prevent large '\n"
        "            'dataset downloads. Open this notebook from mne.tools '\n"
        "            'where sample data is pre-bundled, or run it locally.'\n"
        "        )\n"
        "    return orig_pooch_fetch(\n"
        "        self, fname, processor=processor, downloader=downloader\n"
        "    )\n"
        "pooch.Pooch.fetch = pyodide_pooch_fetch\n"
        "\n"
        "# Import MNE and finalize setup.\n"
        "import mne\n"
        "# Pre-create a valid empty config file so MNE never hits a corrupt read.\n"
        "_cfg = mne.get_config_path()\n"
        "os.makedirs(os.path.dirname(_cfg), exist_ok=True)\n"
        "if not os.path.exists(_cfg):\n"
        "    with open(_cfg, 'w') as _f:\n"
        "        _f.write('{}')\n"
        "mne.set_config('MNE_DATA', mne_data_path)\n"
        "for ds in ['SAMPLE', 'TESTING', 'SSVEP', 'EEGBCI', 'SOMATO',\n"
        "           'BRAINSTORM']:\n"
        "    mne.set_config(f'MNE_DATASETS_{ds}_PATH', mne_data_path)\n"
        "\n"
        "# Bypass pooch's archive check: data_path() normally looks for the\n"
        "# .tar.gz archive, not just the extracted folder. Return the folder\n"
        "# directly so pooch never tries to download from OSF. Return a Path\n"
        "# (not a str) since tutorials use the / operator on the result.\n"
        "from pathlib import Path as _Path\n"
        "_sample_path = _Path(_sample_dir)\n"
        "def _lite_sample_data_path(*_a, **_kw):\n"
        "    return _sample_path\n"
        "mne.datasets.sample.data_path = _lite_sample_data_path\n"
        "# Several non-sample datasets are each used by only a couple of\n"
        "# notebooks (kiloword/erp_core for Epochs 30 & 40; mtrf/eegbci for the\n"
        "# decoding examples), so fetch them LAZILY — only when their\n"
        "# data_path()/load_data() is called — to avoid taxing every other\n"
        "# notebook's setup. Pyodide runs in a web worker here, where a\n"
        "# synchronous XHR may set responseType='arraybuffer', letting a sync\n"
        "# data_path() read binary.\n"
        "def _lite_fetch_rel(_rel):\n"
        "    _dst = mne_data_path + '/' + _rel\n"
        "    if not os.path.exists(_dst):\n"
        "        from js import XMLHttpRequest\n"
        "        _xhr = XMLHttpRequest.new()\n"
        "        _xhr.open('GET', _base + _rel, False)\n"
        "        _xhr.responseType = 'arraybuffer'\n"
        "        _xhr.send()\n"
        "        if _xhr.status != 200:\n"
        "            raise FileNotFoundError(\n"
        "                f'Could not fetch {_rel} (HTTP {_xhr.status})'\n"
        "            )\n"
        "        os.makedirs(os.path.dirname(_dst), exist_ok=True)\n"
        "        with open(_dst, 'wb') as _fh:\n"
        "            _fh.write(bytes(_xhr.response.to_py()))\n"
        "    return _dst\n"
        "def _lite_lazy_fetch(_folder, _fname):\n"
        "    _lite_fetch_rel(_folder + '/' + _fname)\n"
        "    return _Path(mne_data_path + '/' + _folder)\n"
        "def _lite_kiloword_data_path(*_a, **_kw):\n"
        "    return _lite_lazy_fetch("
        "'MNE-kiloword-data', 'kword_metadata-epo.fif')\n"
        "mne.datasets.kiloword.data_path = _lite_kiloword_data_path\n"
        "def _lite_erp_core_data_path(*_a, **_kw):\n"
        "    return _lite_lazy_fetch(\n"
        "        'MNE-ERP-CORE-data', "
        "'ERP-CORE_Subject-001_Task-Flankers_eeg.fif'\n"
        "    )\n"
        "mne.datasets.erp_core.data_path = _lite_erp_core_data_path\n"
        "def _lite_mtrf_data_path(*_a, **_kw):\n"
        "    return _lite_lazy_fetch('mTRF_1.5', 'speech_data.mat')\n"
        "mne.datasets.mtrf.data_path = _lite_mtrf_data_path\n"
        "# testing hands back the folder and lets the shimmed readers pull\n"
        "# individual files, so a notebook that wants the EEGLAB recording does\n"
        "# not also drag down the 39 MB movement raw.\n"
        "def _lite_testing_data_path(*_a, **_kw):\n"
        "    return _Path(mne_data_path + '/MNE-testing-data')\n"
        "mne.datasets.testing.data_path = _lite_testing_data_path\n"
        "# Same again for the datasets behind a single example each. Only the\n"
        "# files those examples read are served, and the shimmed readers below\n"
        "# pull them individually.\n"
        "def _lite_folder_data_path(_folder):\n"
        "    def _data_path(*_a, **_kw):\n"
        "        return _Path(mne_data_path + '/' + _folder)\n"
        "    return _data_path\n"
        "for _ds, _folder in (\n"
        "    ('ssvep', 'ssvep-example-data'),\n"
        "    ('misc', 'MNE-misc-data'),\n"
        "    ('eyelink', 'MNE-eyelink-data'),\n"
        "    ('fnirs_motor', 'MNE-fNIRS-motor-data'),\n"
        "    ('refmeg_noise', 'MNE-refmeg-noise-data'),\n"
        "    ('phantom_kernel', 'MNE-phantom-kernel-data'),\n"
        "    ('multimodal', 'MNE-multimodal-data'),\n"
        "):\n"
        "    getattr(mne.datasets, _ds).data_path = _lite_folder_data_path(_folder)\n"
        "def _lite_eegbci_load_data(subject, runs, *_a, **_kw):\n"
        "    _runs = [runs] if isinstance(runs, (int, float)) else list(runs)\n"
        "    _subjects = (\n"
        "        list(subject) if isinstance(subject, (list, tuple))\n"
        "        else [subject]\n"
        "    )\n"
        "    _out = []\n"
        "    for _s in _subjects:\n"
        "        for _r in _runs:\n"
        "            _rel = (\n"
        "                'MNE-eegbci-data/files/eegmmidb/1.0.0/'\n"
        "                f'S{int(_s):03d}/S{int(_s):03d}R{int(_r):02d}.edf'\n"
        "            )\n"
        "            _out.append(_Path(_lite_fetch_rel(_rel)))\n"
        "    return _out\n"
        "mne.datasets.eegbci.load_data = _lite_eegbci_load_data\n"
        "\n"
        "# Some MNE-sample-data files (e.g. the fixed-orientation forward/\n"
        "# inverse used by the point-spread tutorial) aren't in the eager\n"
        "# _sample_files list above because only one or two notebooks need\n"
        "# them. Rather than hand-listing every such file, lazily fetch any\n"
        "# sample-data path the first time read_forward_solution/\n"
        "# read_inverse_operator is asked to open it.\n"
        "def _lite_fetch_if_under_mne_data(fname):\n"
        "    _p = str(fname)\n"
        "    if _p.startswith(mne_data_path + '/'):\n"
        "        _lite_fetch_rel(_p[len(mne_data_path) + 1:])\n"
        "    return fname\n"
        "_orig_read_forward_solution = mne.read_forward_solution\n"
        "def _lite_read_forward_solution(fname, *_a, **_kw):\n"
        "    return _orig_read_forward_solution(\n"
        "        _lite_fetch_if_under_mne_data(fname), *_a, **_kw\n"
        "    )\n"
        "mne.read_forward_solution = _lite_read_forward_solution\n"
        "import mne.minimum_norm as _mne_minv\n"
        "_orig_read_inverse_operator = _mne_minv.read_inverse_operator\n"
        "def _lite_read_inverse_operator(fname, *_a, **_kw):\n"
        "    return _orig_read_inverse_operator(\n"
        "        _lite_fetch_if_under_mne_data(fname), *_a, **_kw\n"
        "    )\n"
        "_mne_minv.read_inverse_operator = _lite_read_inverse_operator\n"
        "mne.minimum_norm.read_inverse_operator = _lite_read_inverse_operator\n"
        "# Lazily fetch the heavy sample raw / source-space files only when a\n"
        "# notebook actually reads them (same pattern as the fwd/inv shims\n"
        "# above), instead of pulling the whole sample set up front.\n"
        "_orig_read_raw_fif = mne.io.read_raw_fif\n"
        "def _lite_read_raw_fif(fname, *_a, **_kw):\n"
        "    return _orig_read_raw_fif(\n"
        "        _lite_fetch_if_under_mne_data(fname), *_a, **_kw\n"
        "    )\n"
        "mne.io.read_raw_fif = _lite_read_raw_fif\n"
        "_orig_read_raw = mne.io.read_raw\n"
        "def _lite_read_raw(fname, *_a, **_kw):\n"
        "    return _orig_read_raw(\n"
        "        _lite_fetch_if_under_mne_data(fname), *_a, **_kw\n"
        "    )\n"
        "mne.io.read_raw = _lite_read_raw\n"
        "_orig_read_source_spaces = mne.read_source_spaces\n"
        "def _lite_read_source_spaces(fname, *_a, **_kw):\n"
        "    return _orig_read_source_spaces(\n"
        "        _lite_fetch_if_under_mne_data(fname), *_a, **_kw\n"
        "    )\n"
        "mne.read_source_spaces = _lite_read_source_spaces\n"
        "# Nearly every MNE reader validates its filename through\n"
        "# _check_fname(must_exist=True) before opening it, so hooking that one\n"
        "# function covers read_info, read_evokeds, read_cov, read_label and the\n"
        "# rest without a wrapper each. Failures stay silent here so MNE still\n"
        "# raises its own, clearer error for a file that genuinely is missing.\n"
        "import mne.utils.check as _mne_check\n"
        "_orig_check_fname = _mne_check._check_fname\n"
        "def _lite_check_fname(fname, overwrite=False, must_exist=False,\n"
        "                      *_a, **_kw):\n"
        "    if must_exist:\n"
        "        try:\n"
        "            _lite_fetch_if_under_mne_data(fname)\n"
        "        except Exception:\n"
        "            pass\n"
        "    return _orig_check_fname(fname, overwrite, must_exist, *_a, **_kw)\n"
        "_mne_check._check_fname = _lite_check_fname\n"
        "# modules that imported it before now hold their own reference; ones\n"
        "# loaded later (mne lazy-loads most of itself) pick up the patch\n"
        "for _m in list(sys.modules.values()):\n"
        "    if (getattr(_m, '__name__', '').startswith('mne')\n"
        "            and getattr(_m, '_check_fname', None) is _orig_check_fname):\n"
        "        _m._check_fname = _lite_check_fname\n"
        "# read_label, read_epochs and read_raw_edf open their file directly\n"
        "# rather than validating it first, so the hook above never sees them\n"
        "_orig_read_label = mne.read_label\n"
        "def _lite_read_label(filename, *_a, **_kw):\n"
        "    return _orig_read_label(\n"
        "        _lite_fetch_if_under_mne_data(filename), *_a, **_kw\n"
        "    )\n"
        "mne.read_label = _lite_read_label\n"
        "_orig_read_epochs = mne.read_epochs\n"
        "def _lite_read_epochs(fname, *_a, **_kw):\n"
        "    return _orig_read_epochs(\n"
        "        _lite_fetch_if_under_mne_data(fname), *_a, **_kw\n"
        "    )\n"
        "mne.read_epochs = _lite_read_epochs\n"
        "_orig_read_raw_edf = mne.io.read_raw_edf\n"
        "def _lite_read_raw_edf(input_fname, *_a, **_kw):\n"
        "    return _orig_read_raw_edf(\n"
        "        _lite_fetch_if_under_mne_data(input_fname), *_a, **_kw\n"
        "    )\n"
        "mne.io.read_raw_edf = _lite_read_raw_edf\n"
        "_orig_read_bem_solution = mne.read_bem_solution\n"
        "def _lite_read_bem_solution(fname, *_a, **_kw):\n"
        "    return _orig_read_bem_solution(\n"
        "        _lite_fetch_if_under_mne_data(fname), *_a, **_kw\n"
        "    )\n"
        "mne.read_bem_solution = _lite_read_bem_solution\n"
        "_orig_read_events = mne.read_events\n"
        "def _lite_read_events(fname, *_a, **_kw):\n"
        "    return _orig_read_events(\n"
        "        _lite_fetch_if_under_mne_data(fname), *_a, **_kw\n"
        "    )\n"
        "mne.read_events = _lite_read_events\n"
        "# an EEGLAB .set keeps its samples in a sibling .fdt, so fetch both\n"
        "_orig_read_raw_eeglab = mne.io.read_raw_eeglab\n"
        "def _lite_read_raw_eeglab(input_fname, *_a, **_kw):\n"
        "    _p = str(input_fname)\n"
        "    if _p.startswith(mne_data_path + '/'):\n"
        "        for _cand in (_p, _p[:-4] + '.fdt'):\n"
        "            try:\n"
        "                _lite_fetch_rel(_cand[len(mne_data_path) + 1:])\n"
        "            except Exception:\n"
        "                pass\n"
        "    return _orig_read_raw_eeglab(input_fname, *_a, **_kw)\n"
        "mne.io.read_raw_eeglab = _lite_read_raw_eeglab\n"
        "# read_raw_nirx and read_raw_egi open a folder, so there is no single\n"
        "# name to fetch; conf.py leaves a listing next to the copy.\n"
        "def _lite_fetch_dir(_rel):\n"
        "    _manifest = _lite_fetch_rel(_rel + '/_lite_manifest.txt')\n"
        "    with open(_manifest) as _fh:\n"
        "        _names = [_n.strip() for _n in _fh if _n.strip()]\n"
        "    for _name in _names:\n"
        "        # one unreachable member must not abandon the rest of the\n"
        "        # recording; the reader complains if it needed that file\n"
        "        try:\n"
        "            _lite_fetch_rel(_rel + '/' + _name)\n"
        "        except Exception as _e:\n"
        "            print('[JupyterLite] skipped ' + _name + ': ' + repr(_e))\n"
        "    return mne_data_path + '/' + _rel\n"
        "def _lite_dir_reader(_orig):\n"
        "    def _read(fname, *_a, **_kw):\n"
        "        _p = str(fname)\n"
        "        if _p.startswith(mne_data_path + '/'):\n"
        "            try:\n"
        "                _lite_fetch_dir(_p[len(mne_data_path) + 1:])\n"
        "            except Exception as _e:\n"
        "                print('[JupyterLite] could not fetch '\n"
        "                      + _p + ': ' + repr(_e))\n"
        "        return _orig(fname, *_a, **_kw)\n"
        "    return _read\n"
        "mne.io.read_raw_nirx = _lite_dir_reader(mne.io.read_raw_nirx)\n"
        "mne.io.read_raw_egi = _lite_dir_reader(mne.io.read_raw_egi)\n"
        "# the logging tutorial reads a KIT file from inside the installed\n"
        "# package; the wheel excludes mne/**/tests, so stage the served copy\n"
        "# into the path the tutorial builds rather than editing the tutorial\n"
        "import shutil as _shutil\n"
        "_orig_read_raw_kit = mne.io.read_raw_kit\n"
        "def _lite_read_raw_kit(input_fname, *_a, **_kw):\n"
        "    _p = str(input_fname)\n"
        "    if _p.endswith('test.sqd') and not os.path.exists(_p):\n"
        "        try:\n"
        "            _staged = _lite_fetch_rel('MNE-kit-testdata/test.sqd')\n"
        "            os.makedirs(os.path.dirname(_p), exist_ok=True)\n"
        "            _shutil.copyfile(_staged, _p)\n"
        "        except Exception as _e:\n"
        "            print('[JupyterLite] could not stage test.sqd: ' + repr(_e))\n"
        "    return _orig_read_raw_kit(input_fname, *_a, **_kw)\n"
        "mne.io.read_raw_kit = _lite_read_raw_kit\n"
        "# a BrainVision .vhdr is a text header pointing at a .eeg and a .vmrk\n"
        "_orig_read_raw_brainvision = mne.io.read_raw_brainvision\n"
        "def _lite_read_raw_brainvision(vhdr_fname, *_a, **_kw):\n"
        "    _p = str(vhdr_fname)\n"
        "    if _p.startswith(mne_data_path + '/'):\n"
        "        _stem = _p[:-5] if _p.endswith('.vhdr') else _p\n"
        "        for _cand in (_p, _stem + '.eeg', _stem + '.vmrk'):\n"
        "            try:\n"
        "                _lite_fetch_rel(_cand[len(mne_data_path) + 1:])\n"
        "            except Exception:\n"
        "                pass\n"
        "    return _orig_read_raw_brainvision(vhdr_fname, *_a, **_kw)\n"
        "mne.io.read_raw_brainvision = _lite_read_raw_brainvision\n"
        "# eyelink .asc recordings are single files\n"
        "_orig_read_raw_eyelink = mne.io.read_raw_eyelink\n"
        "def _lite_read_raw_eyelink(fname, *_a, **_kw):\n"
        "    return _orig_read_raw_eyelink(\n"
        "        _lite_fetch_if_under_mne_data(fname), *_a, **_kw\n"
        "    )\n"
        "mne.io.read_raw_eyelink = _lite_read_raw_eyelink\n"
        "# the heatmap example draws its stimulus straight through pyplot, and\n"
        "# read_xdf goes through pyxdf -- neither is an MNE reader, so shim the\n"
        "# two entry points as well\n"
        "import matplotlib.pyplot as _plt\n"
        "_orig_imread = _plt.imread\n"
        "def _lite_imread(fname, *_a, **_kw):\n"
        "    return _orig_imread(_lite_fetch_if_under_mne_data(fname), *_a, **_kw)\n"
        "_plt.imread = _lite_imread\n"
        "try:\n"
        "    import pyxdf as _pyxdf\n"
        "    _orig_load_xdf = _pyxdf.load_xdf\n"
        "    def _lite_load_xdf(fname, *_a, **_kw):\n"
        "        return _orig_load_xdf(\n"
        "            _lite_fetch_if_under_mne_data(fname), *_a, **_kw\n"
        "        )\n"
        "    _pyxdf.load_xdf = _lite_load_xdf\n"
        "except Exception:\n"
        "    pass\n"
        "import mne.chpi as _mne_chpi\n"
        "_orig_read_head_pos = _mne_chpi.read_head_pos\n"
        "def _lite_read_head_pos(fname, *_a, **_kw):\n"
        "    return _orig_read_head_pos(\n"
        "        _lite_fetch_if_under_mne_data(fname), *_a, **_kw\n"
        "    )\n"
        "_mne_chpi.read_head_pos = _lite_read_head_pos\n"
        "mne.chpi.read_head_pos = _lite_read_head_pos\n"
        "# read_source_estimate is handed the stem of a .stc pair, so fetch\n"
        "# both hemispheres before letting MNE resolve the name itself.\n"
        "_orig_read_source_estimate = mne.read_source_estimate\n"
        "def _lite_read_source_estimate(fname, *_a, **_kw):\n"
        "    _p = str(fname)\n"
        "    if _p.startswith(mne_data_path + '/'):\n"
        "        for _suf in ('', '-lh.stc', '-rh.stc'):\n"
        "            try:\n"
        "                _lite_fetch_rel(_p[len(mne_data_path) + 1:] + _suf)\n"
        "            except Exception:\n"
        "                pass\n"
        "    return _orig_read_source_estimate(fname, *_a, **_kw)\n"
        "mne.read_source_estimate = _lite_read_source_estimate\n"
        "# plot_alignment locates its head surface by probing the filesystem\n"
        "# with os.path.exists before any reader runs, so a reader shim never\n"
        "# fires. Fetch the candidates first and let MNE choose as it normally\n"
        "# would. Several viz modules bind the name at import time, so rebind\n"
        "# it wherever the original landed instead of in one known place.\n"
        "import mne._freesurfer as _mne_fs\n"
        "_orig_get_head_surface = _mne_fs._get_head_surface\n"
        "def _lite_get_head_surface(surf, subject, subjects_dir, bem=None,\n"
        "                           verbose=None):\n"
        "    _sd = str(subjects_dir) if subjects_dir is not None else ''\n"
        "    if subject and _sd.startswith(mne_data_path + '/'):\n"
        "        _rel = _sd[len(mne_data_path) + 1:] + '/' + str(subject)\n"
        "        if surf in ('head-dense', 'seghead'):\n"
        "            _cands = ['bem/' + str(subject) + '-head-dense.fif',\n"
        "                      'surf/lh.seghead']\n"
        "        else:\n"
        "            # same order MNE tries, so the browser picks the same\n"
        "            # surface the rendered docs did\n"
        "            _cands = ['bem/outer_skin.surf',\n"
        "                      'bem/' + str(subject) + '-head.fif']\n"
        "        for _c in _cands:\n"
        "            try:\n"
        "                _lite_fetch_rel(_rel + '/' + _c)\n"
        "            except Exception:\n"
        "                pass\n"
        "    return _orig_get_head_surface(\n"
        "        surf, subject, subjects_dir, bem=bem, verbose=verbose\n"
        "    )\n"
        "_mne_fs._get_head_surface = _lite_get_head_surface\n"
        "# import the 3D module first so the sweep below is guaranteed to see\n"
        "# it; anything imported later picks the patched name up on its own.\n"
        "import mne.viz._3d  # noqa: F401\n"
        "for _m in list(sys.modules.values()):\n"
        "    if (getattr(_m, '__name__', '').startswith('mne')\n"
        "            and getattr(_m, '_get_head_surface', None)\n"
        "            is _orig_get_head_surface):\n"
        "        _m._get_head_surface = _lite_get_head_surface\n"
        "# same story for the skull surfaces, which _check_fname insists\n"
        "# already exist on disk\n"
        "_orig_get_skull_surface = _mne_fs._get_skull_surface\n"
        "def _lite_get_skull_surface(surf, subject, subjects_dir, bem=None,\n"
        "                            verbose=None):\n"
        "    _sd = str(subjects_dir) if subjects_dir is not None else ''\n"
        "    if subject and _sd.startswith(mne_data_path + '/'):\n"
        "        try:\n"
        "            _lite_fetch_rel(\n"
        "                _sd[len(mne_data_path) + 1:] + '/' + str(subject)\n"
        "                + '/bem/' + surf + '_skull.surf'\n"
        "            )\n"
        "        except Exception:\n"
        "            pass\n"
        "    return _orig_get_skull_surface(\n"
        "        surf, subject, subjects_dir, bem=bem, verbose=verbose\n"
        "    )\n"
        "_mne_fs._get_skull_surface = _lite_get_skull_surface\n"
        "for _m in list(sys.modules.values()):\n"
        "    if (getattr(_m, '__name__', '').startswith('mne')\n"
        "            and getattr(_m, '_get_skull_surface', None)\n"
        "            is _orig_get_skull_surface):\n"
        "        _m._get_skull_surface = _lite_get_skull_surface\n"
        "# dig_mri_distances reaches a second, unrelated _get_head_surface, the\n"
        "# one in mne/surface.py: it takes a list of candidate sources and\n"
        "# probes bem/ with os.path.exists and glob, raising if the directory\n"
        "# is absent, so the candidates have to land before it runs.\n"
        "import mne.surface as _mne_surface\n"
        "_orig_surface_head = _mne_surface._get_head_surface\n"
        "def _lite_surface_head_surface(subject, source, subjects_dir,\n"
        "                               on_defects, raise_error=True):\n"
        "    _sd = str(subjects_dir) if subjects_dir is not None else ''\n"
        "    if subject and _sd.startswith(mne_data_path + '/'):\n"
        "        _rel = _sd[len(mne_data_path) + 1:] + '/' + str(subject)\n"
        "        _srcs = [source] if isinstance(source, str) else list(source)\n"
        "        for _s in _srcs:\n"
        "            try:\n"
        "                _lite_fetch_rel(\n"
        "                    _rel + '/bem/' + str(subject) + '-' + _s + '.fif'\n"
        "                )\n"
        "            except Exception:\n"
        "                pass\n"
        "    return _orig_surface_head(\n"
        "        subject, source, subjects_dir, on_defects,\n"
        "        raise_error=raise_error\n"
        "    )\n"
        "_mne_surface._get_head_surface = _lite_surface_head_surface\n"
        "# plot_bem globs bem/*.surf and requires the bem directory to exist,\n"
        "# so pull its three contours (plus the MRI it draws them on) down\n"
        "# first; fetching creates the directory as a side effect.\n"
        "_orig_plot_bem = mne.viz.plot_bem\n"
        "def _lite_plot_bem(subject=None, subjects_dir=None, *_a, **_kw):\n"
        "    _sd = str(subjects_dir) if subjects_dir is not None else ''\n"
        "    if subject and _sd.startswith(mne_data_path + '/'):\n"
        "        _rel = _sd[len(mne_data_path) + 1:] + '/' + str(subject)\n"
        "        _want = ['bem/inner_skull.surf', 'bem/outer_skull.surf',\n"
        "                 'bem/outer_skin.surf',\n"
        "                 'mri/' + str(_kw.get('mri', 'T1.mgz'))]\n"
        "        _bs = _kw.get('brain_surfaces')\n"
        "        if _bs is not None:\n"
        "            _bs = [_bs] if isinstance(_bs, str) else list(_bs)\n"
        "            for _b in _bs:\n"
        "                _want += ['surf/lh.' + _b, 'surf/rh.' + _b]\n"
        "        for _c in _want:\n"
        "            try:\n"
        "                _lite_fetch_rel(_rel + '/' + _c)\n"
        "            except Exception:\n"
        "                pass\n"
        "    return _orig_plot_bem(subject, subjects_dir, *_a, **_kw)\n"
        "mne.viz.plot_bem = _lite_plot_bem\n"
        "\n"
        "# EXPERIMENTAL 3D: MNE's normal Brain/VTK stack can't load in WASM, so\n"
        "# route SourceEstimate.plot() through pyvista-js (vtk.js) instead.\n"
        "# pyvista-js (0.15) has no scalar colormap in its renderer, so we\n"
        "# approximate MNE's Brain look with solid-colored meshes: a two-tone\n"
        "# curvature base (light gyri + dark sulci) plus many thin 'hot' bands\n"
        "# for the activation, on a black background with even scene lighting.\n"
        "# Static, one time point, no time slider yet. Fully guarded — any\n"
        "# failure prints a message so the notebook completes. Returns a stub\n"
        "# 'brain' whose methods (add_foci/add_text/show_view/...) are safe\n"
        "# no-ops, so tutorials that call brain.add_foci(...) after plot() work.\n"
        "class _LiteBrain:\n"
        "    def screenshot(self, *_a, **_kw):\n"
        "        import numpy as _np\n"
        "        return _np.zeros((2, 2, 3), dtype='uint8')\n"
        "    def __getattr__(self, _name):\n"
        "        return lambda *_a, **_kw: None\n"
        "def _lite_stc_plot(self, *_a, **_kw):\n"
        "    try:\n"
        "        import numpy as _np\n"
        "        import nibabel as _nib\n"
        "        from scipy.spatial import cKDTree as _KDTree\n"
        "        from matplotlib import colormaps as _cmaps\n"
        "        import pyvista_js as _pv\n"
        "        _subj = (_kw.get('subject')\n"
        "                 or (_a[0] if _a and isinstance(_a[0], str) else None)\n"
        "                 or 'sample')\n"
        "        _sdir = _kw.get('subjects_dir')\n"
        "        _sdir = (str(_sdir) if _sdir is not None else\n"
        "                 mne_data_path + '/MNE-sample-data/subjects')\n"
        "        # surfaces are fetched relative to the served mne_data root, so\n"
        "        # derive that from subjects_dir rather than assuming sample --\n"
        "        # a dataset may keep its FreeSurfer subjects under its own folder.\n"
        "        _rel_sdir = (_sdir[len(mne_data_path) + 1:]\n"
        "                     if _sdir.startswith(mne_data_path + '/')\n"
        "                     else 'MNE-sample-data/subjects')\n"
        "        _init = _kw.get('initial_time', None)\n"
        "        if _init is None:\n"
        "            _ti = int(_np.argmax(_np.abs(self.data).mean(0)))\n"
        "        else:\n"
        "            _ti = int(_np.argmin(_np.abs(self.times - _init)))\n"
        "        _hot = _cmaps['hot']\n"
        "        _N = 10\n"
        "        def _flat(_t):\n"
        "            return _np.hstack([\n"
        "                _np.full((len(_t), 1), 3, dtype=_np.int64),\n"
        "                _t.astype(_np.int64)]).ravel()\n"
        "        def _sub(_pts, _tris, _mask, _lift=0.0, _cen=None):\n"
        "            _sel = _tris[_mask]\n"
        "            if len(_sel) == 0:\n"
        "                return None\n"
        "            _u, _iv = _np.unique(_sel, return_inverse=True)\n"
        "            _p = _pts[_u]\n"
        "            if _lift and _cen is not None:\n"
        "                _p = _cen + (_p - _cen) * (1.0 + _lift)\n"
        "            return _p, _iv.reshape(-1, 3)\n"
        "        _plotter = _pv.Plotter()\n"
        "        _plotter.background_color = 'black'\n"
        "        # even lighting so the surface isn't black when rotated\n"
        "        for _lp in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),\n"
        "                    (0, -1, 0), (0, 0, 1), (0, 0, -1)):\n"
        "            _plotter.add_light(_pv.Light(\n"
        "                position=(300.0 * _lp[0], 300.0 * _lp[1],\n"
        "                          300.0 * _lp[2]),\n"
        "                focal_point=(0.0, 0.0, 0.0), intensity=0.4))\n"
        "        _nlh = len(self.vertices[0])\n"
        "        _hemis = (('lh', 0, self.vertices[0]),\n"
        "                  ('rh', 1, self.vertices[1]))\n"
        "        for _h, _hi, _vno in _hemis:\n"
        "            if len(_vno) == 0:\n"
        "                continue\n"
        "            _pre = _rel_sdir + '/' + _subj + '/surf/' + _h\n"
        "            _lite_fetch_rel(_pre + '.inflated')\n"
        "            _lite_fetch_rel(_pre + '.curv')\n"
        "            _bpath = _sdir + '/' + _subj + '/surf/' + _h\n"
        "            _rr, _tris = mne.read_surface(_bpath + '.inflated')\n"
        "            _cv = _nib.freesurfer.read_morph_data(_bpath + '.curv')\n"
        "            _hdata = self.data[:_nlh] if _hi == 0 else self.data[_nlh:]\n"
        "            # color each surface vertex from the nearest ACTIVE source\n"
        "            # within a small radius, so single-vertex (point) sources\n"
        "            # show as visible blobs and dense sources fill in as usual\n"
        "            _sv = _hdata[:, _ti].astype(float)\n"
        "            _act = _sv != 0\n"
        "            _scal = _np.zeros(len(_rr))\n"
        "            if _act.any():\n"
        "                _atree = _KDTree(_rr[_vno][_act])\n"
        "                _ad, _ai = _atree.query(_rr)\n"
        "                _scal = _np.where(_ad <= 12.0, _sv[_act][_ai], 0.0)\n"
        "            # offset hemispheres along x so they do not overlap\n"
        "            _off = -60.0 if _h == 'lh' else 60.0\n"
        "            _pts = _np.round(_rr, 2)\n"
        "            _pts[:, 0] = _pts[:, 0] + _off\n"
        "            _cen = _pts.mean(0)\n"
        "            # curvature base: light gyri (curv<0) + dark sulci (curv>=0)\n"
        "            _fc = _cv[_tris].mean(1)\n"
        "            for _cm, _col in (\n"
        "                    (_fc < 0, (0.68, 0.68, 0.68)),\n"
        "                    (_fc >= 0, (0.38, 0.38, 0.38))):\n"
        "                _s = _sub(_pts, _tris, _cm)\n"
        "                if _s is not None:\n"
        "                    _plotter.add_mesh(\n"
        "                        _pv.PolyData(points=_s[0], faces=_flat(_s[1])),\n"
        "                        color=_col, smooth_shading=True)\n"
        "            # activation as a smooth hot gradient in N value bands,\n"
        "            # each lifted 2% off the surface to avoid z-fighting\n"
        "            _fv = _scal[_tris].mean(1)\n"
        "            _p90 = _np.percentile(_scal, 90.0)\n"
        "            _fmax = float(_scal.max())\n"
        "            # keep the background gray: for sparse point sources the\n"
        "            # 90th pct is ~0 (most of the brain is zero), which would\n"
        "            # paint everything, so fall back to a fraction of the max.\n"
        "            _fmin = _p90 if _p90 > _fmax * 0.05 else _fmax * 0.4\n"
        "            if _fmax > _fmin:\n"
        "                _edges = _np.linspace(_fmin, _fmax, _N + 1)\n"
        "                for _i in range(_N):\n"
        "                    if _i < _N - 1:\n"
        "                        _m = (_fv >= _edges[_i]) & (_fv < _edges[_i + 1])\n"
        "                    else:\n"
        "                        _m = _fv >= _edges[_i]\n"
        "                    if int(_m.sum()) == 0:\n"
        "                        continue\n"
        "                    _rgb = _hot(0.25 + 0.41 * (_i / (_N - 1)))\n"
        "                    _col = (float(_rgb[0]), float(_rgb[1]),\n"
        "                            float(_rgb[2]))\n"
        "                    _s = _sub(_pts, _tris, _m, 0.02, _cen)\n"
        "                    if _s is not None:\n"
        "                        _plotter.add_mesh(\n"
        "                            _pv.PolyData(points=_s[0],\n"
        "                                         faces=_flat(_s[1])),\n"
        "                            color=_col, smooth_shading=True)\n"
        "        # Open on the lateral profile (camera along the medial-lateral\n"
        "        # X axis, superior up), like native MNE, instead of vtk.js's\n"
        "        # default anterior/face-on view. Guarded so a missing\n"
        "        # view_vector never costs us the render.\n"
        "        try:\n"
        "            _plotter.view_vector((-1.0, 0.0, 0.0),\n"
        "                                 viewup=(0.0, 0.0, 1.0))\n"
        "        except Exception:\n"
        "            pass\n"
        "        _plotter.show()\n"
        "    except Exception as _e:\n"
        "        print('[JupyterLite] pyvista-js 3D render unavailable: '\n"
        "              + repr(_e))\n"
        "    return _LiteBrain()\n"
        "mne.SourceEstimate.plot = _lite_stc_plot\n"
        "\n"
        "# Pyodide/WASM has no OS threads, so MNE's ProgressBar background\n"
        "# updater thread (used by the ProgressBar context manager, e.g. in\n"
        "# permutation cluster tests) crashes with 'can't start new thread'.\n"
        "# That thread only animates a cosmetic bar — the computation runs on\n"
        "# the main thread and __exit__ writes the final state — so no-op its\n"
        "# start/join. Only affects notebooks that use it; results are unchanged.\n"
        "try:\n"
        "    from mne.utils import progressbar as _mpb\n"
        "    _mpb._UpdateThread.start = lambda self: None\n"
        "    _mpb._UpdateThread.join = lambda self, *_a, **_kw: None\n"
        "except Exception:\n"
        "    pass\n"
        "# tqdm also spawns its own monitor thread, which likewise can't start in\n"
        "# WASM and emits a TqdmMonitorWarning. Setting monitor_interval=0 before\n"
        "# any bar is created skips that thread entirely (bars still display).\n"
        "try:\n"
        "    import tqdm as _tqdm\n"
        "    _tqdm.tqdm.monitor_interval = 0\n"
        "except Exception:\n"
        "    pass\n"
        "\n"
        "# Switch matplotlib to inline so figures render in the notebook.\n"
        "import IPython\n"
        "IPython.get_ipython().run_line_magic('matplotlib', 'inline')\n"
        "import matplotlib.pyplot as plt\n"
        "# Silence the spurious 'FigureCanvasAgg is non-interactive' warning\n"
        "# at its source. MNE's plt_show calls fig.show() (the inline backend\n"
        "# isn't detected as 'agg'), and the inline Agg canvas warns. Patching\n"
        "# viz.utils.plt_show is not enough: other modules did\n"
        "# `from .utils import plt_show` and hold their own reference. Every\n"
        "# path resolves fig.show on the class at call time, so a no-op here\n"
        "# silences it everywhere. Figures still render via the inline backend.\n"
        "import matplotlib.figure as _mfig\n"
        "_mfig.Figure.show = lambda self, *a, **k: None\n"
        "import importlib\n"
        "viz_utils = importlib.import_module('mne.viz.utils')\n"
        "# Also display+close via IPython for paths that call plt_show\n"
        "# directly, so figures render exactly once.\n"
        "def pyodide_plt_show(show=True, fig=None, **kwargs):\n"
        "    if not show:\n"
        "        return\n"
        "    import IPython.display\n"
        "    _f = fig if fig is not None else plt.gcf()\n"
        "    IPython.display.display(_f)\n"
        "    plt.close(_f)\n"
        "viz_utils.plt_show = pyodide_plt_show\n"
        "\n"
        "# EXPERIMENTAL 3D: plot_sparse_source_estimates builds its 3D renderer\n"
        "# BEFORE the time-course figure, so in WASM the whole call dies and the\n"
        "# notebook loses both halves. Rebuild it here: the same glass brain from\n"
        "# the source space and a marker per active dipole via pyvista-js, plus\n"
        "# the matplotlib time courses (which are the quantitative half). Same\n"
        "# approach as the SourceEstimate.plot shim above.\n"
        "def _lite_plot_sparse_source_estimates(\n"
        "        src, stcs, colors=None, linewidth=2, fontsize=18,\n"
        "        bgcolor=(0.05, 0, 0.1), opacity=0.2, brain_color=(0.7,) * 3,\n"
        "        show=True, high_resolution=False, fig_name=None,\n"
        "        fig_number=None, labels=None, modes=('cone', 'sphere'),\n"
        "        scale_factors=(1, 0.6), **kwargs):\n"
        "    import numpy as _np\n"
        "    from itertools import cycle as _cycle\n"
        "    from matplotlib.colors import to_rgb as _to_rgb\n"
        "    if not isinstance(stcs, list):\n"
        "        stcs = [stcs]\n"
        "    _lhp = src[0]['rr']\n"
        "    _pts = _np.r_[_lhp, src[1]['rr']] * 170\n"
        "    _nrm = _np.r_[src[0]['nn'], src[1]['nn']]\n"
        "    # use_tris is the decimated mesh and can be None on some source\n"
        "    # spaces; fall back to the full tris in that case.\n"
        "    _lt = src[0]['tris'] if high_resolution else src[0]['use_tris']\n"
        "    _rt = src[1]['tris'] if high_resolution else src[1]['use_tris']\n"
        "    if _lt is None or _rt is None:\n"
        "        _lt, _rt = src[0]['tris'], src[1]['tris']\n"
        "    _faces = _np.r_[_lt, len(_lhp) + _rt]\n"
        "    _vertnos = [_np.r_[_s.lh_vertno, len(_lhp) + _s.rh_vertno]\n"
        "                for _s in stcs]\n"
        "    _uniq = _np.unique(_np.concatenate(_vertnos).ravel())\n"
        "    # --- time courses -------------------------------------------------\n"
        "    _fig = plt.figure(fig_number, layout='constrained')\n"
        "    _fig.clf()\n"
        "    _ax = _fig.add_subplot(111)\n"
        "    _cyc = _cycle(colors if colors is not None else\n"
        "                  plt.rcParams['axes.prop_cycle'].by_key()['color'])\n"
        "    _marks = []\n"
        "    for _v in _uniq:\n"
        "        _ind = [_k for _k, _vn in enumerate(_vertnos) if _v in _vn]\n"
        "        _c = next(_cyc)\n"
        "        _marks.append((int(_v), _to_rgb(_c), len(_ind) > 1))\n"
        "        for _k in _ind:\n"
        "            _m = _vertnos[_k] == _v\n"
        "            _ax.plot(1e3 * stcs[_k].times,\n"
        "                     1e9 * stcs[_k].data[_m].ravel(),\n"
        "                     c=_c, linewidth=linewidth)\n"
        "    _ax.set_xlabel('Time (ms)', fontsize=fontsize)\n"
        "    _ax.set_ylabel('Source amplitude (nAm)', fontsize=fontsize)\n"
        "    if fig_name is not None:\n"
        "        _ax.set_title(fig_name)\n"
        "    pyodide_plt_show(show)\n"
        "    # --- glass brain + dipole markers ---------------------------------\n"
        "    try:\n"
        "        import pyvista_js as _pv\n"
        "        _plotter = _pv.Plotter()\n"
        "        _plotter.background_color = tuple(\n"
        "            float(min(max(_x, 0.0), 1.0)) for _x in bgcolor)\n"
        "        for _lp in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),\n"
        "                    (0, -1, 0), (0, 0, 1), (0, 0, -1)):\n"
        "            _plotter.add_light(_pv.Light(\n"
        "                position=(300.0 * _lp[0], 300.0 * _lp[1],\n"
        "                          300.0 * _lp[2]),\n"
        "                focal_point=(0.0, 0.0, 0.0), intensity=0.4))\n"
        "        _flat_faces = _np.hstack([\n"
        "            _np.full((len(_faces), 1), 3, dtype=_np.int32),\n"
        "            _faces.astype(_np.int32)]).ravel()\n"
        "        _plotter.add_mesh(\n"
        "            _pv.PolyData(points=_pts.astype(_np.float32),\n"
        "                         faces=_flat_faces),\n"
        "            color=tuple(float(_x) for _x in brain_color),\n"
        "            opacity=float(opacity), smooth_shading=True)\n"
        "        for _v, _col, _common in _marks:\n"
        "            _sf = float(scale_factors[1] if _common\n"
        "                        else scale_factors[0])\n"
        "            _mode = modes[1] if _common else modes[0]\n"
        "            _xyz = tuple(float(_q) for _q in _pts[_v])\n"
        "            if _mode == 'sphere':\n"
        "                _glyph = _pv.Sphere(radius=_sf, center=_xyz)\n"
        "            else:\n"
        "                _glyph = _pv.Cone(\n"
        "                    center=_xyz,\n"
        "                    direction=tuple(float(_q) for _q in _nrm[_v]),\n"
        "                    height=2.0 * _sf, radius=_sf)\n"
        "            _plotter.add_mesh(_glyph, color=_col, smooth_shading=True)\n"
        "        try:\n"
        "            _plotter.view_vector((-1.0, 0.0, 0.0),\n"
        "                                 viewup=(0.0, 0.0, 1.0))\n"
        "        except Exception:\n"
        "            pass\n"
        "        _plotter.show()\n"
        "    except Exception as _e:\n"
        "        print('[JupyterLite] pyvista-js glass brain unavailable: '\n"
        "              + repr(_e))\n"
        "mne.viz.plot_sparse_source_estimates = _lite_plot_sparse_source_estimates\n"
        "\n"
        "# Each MNE plot is rendered once by pyodide_plt_show above (display()).\n"
        "# When a plot call is also a cell's last expression, the method returns\n"
        "# the Figure, which Jupyter echoes a SECOND time as the Out[] result\n"
        "# (the duplicate seen below inline plots). Drop that redundant echo for\n"
        "# Figures (and pure lists of Figures, e.g. ica.plot_properties) so each\n"
        "# plot appears exactly once. Non-figure results (numbers, DataFrames,\n"
        "# reprs) are untouched, and raw matplotlib figures never shown still\n"
        "# render via the inline backend's end-of-cell flush, so nothing hides.\n"
        "# Wrapped in try/except (like the patches below): if anything about\n"
        "# the displayhook is unexpected, silently keep the current behavior\n"
        "# (harmless double render) rather than breaking the setup cell.\n"
        "try:\n"
        "    _lite_dh = type(IPython.get_ipython().displayhook)\n"
        "    if not getattr(_lite_dh, '_lite_no_fig_echo', False):\n"
        "        _lite_dh_call = _lite_dh.__call__\n"
        "        def _lite_displayhook(self, result=None):\n"
        "            if isinstance(result, _mfig.Figure):\n"
        "                result = None\n"
        "            elif (isinstance(result, (list, tuple)) and result\n"
        "                  and all(isinstance(_x, _mfig.Figure) for _x in result)):\n"
        "                result = None\n"
        "            return _lite_dh_call(self, result)\n"
        "        _lite_dh.__call__ = _lite_displayhook\n"
        "        _lite_dh._lite_no_fig_echo = True\n"
        "except Exception:\n"
        "    pass\n"
        "\n"
        "# Real fix (not a warnings filter) for the threadpoolctl Pyodide\n"
        "# RuntimeWarning seen via mne.sys_info(): threadpoolctl 3.6.0 (latest\n"
        "# release) still calls the deprecated Pyodide JsProxy.as_object_map().\n"
        "# Pyodide's own message says to use as_py_json() instead; both yield the\n"
        "# same library filepaths, so we swap the call at its source. This removes\n"
        "# the deprecated API usage entirely, so the warning is never emitted.\n"
        "# The upstream fix is already merged (joblib/threadpoolctl#201) but\n"
        "# unreleased; Pyodide bundles the released 3.6.0 wheel. DROP THIS PATCH\n"
        "# once threadpoolctl 3.7.0 is released and Pyodide bundles it.\n"
        "try:\n"
        "    import os as _os\n"
        "    import threadpoolctl as _tpc\n"
        "    def _find_libraries_pyodide(self):\n"
        "        from pyodide_js._module import LDSO\n"
        "        for _fp in LDSO.loadedLibsByName.as_py_json():\n"
        "            if _os.path.exists(_fp):\n"
        "                self._make_controller_from_path(_fp)\n"
        "    _tpc.ThreadpoolController._find_libraries_pyodide = (\n"
        "        _find_libraries_pyodide\n"
        "    )\n"
        "except Exception:\n"
        "    pass\n" + LITE_RENDERER_CELL
        # Draw MNE's 3D figures with pyvista-js. Appended last so MNE is
        # already imported; see doc/sphinxext/jupyterlite_lite_renderer.py.
    ),
    "doc_module": ("mne",),
    "reference_url": dict(mne=None),
    "examples_dirs": examples_dirs,
    "subsection_order": ExplicitOrder(
        [
            "../examples/io/",
            "../examples/simulation/",
            "../examples/preprocessing/",
            "../examples/visualization/",
            "../examples/time_frequency/",
            "../examples/stats/",
            "../examples/decoding/",
            "../examples/connectivity/",
            "../examples/forward/",
            "../examples/inverse/",
            "../examples/realtime/",
            "../examples/datasets/",
            "../tutorials/intro/",
            "../tutorials/io/",
            "../tutorials/raw/",
            "../tutorials/preprocessing/",
            "../tutorials/epochs/",
            "../tutorials/evoked/",
            "../tutorials/time-freq/",
            "../tutorials/forward/",
            "../tutorials/inverse/",
            "../tutorials/stats-sensor-space/",
            "../tutorials/stats-source-space/",
            "../tutorials/machine-learning/",
            "../tutorials/clinical/",
            "../tutorials/simulation/",
            "../tutorials/sample-datasets/",
            "../tutorials/visualization/",
            "../tutorials/misc/",
        ]
    ),
    "gallery_dirs": gallery_dirs,
    "default_thumb_file": os.path.join("_static", "mne_helmet.png"),
    "backreferences_dir": "generated",
    "plot_gallery": "True",  # Avoid annoying Unicode/bool default warning
    "thumbnail_size": (160, 112),
    "remove_config_comments": True,
    "min_reported_time": 1.0,
    "abort_on_example_error": False,
    "reset_modules": (
        "matplotlib",
        "mne_doc_utils.reset_modules",
    ),  # called w/each script
    "reset_modules_order": "both",
    "image_scrapers": scrapers,
    "show_memory": sys.platform == "linux" and sphinx_gallery_parallel == 1,
    "line_numbers": False,  # messes with style
    "within_subsection_order": "FileNameSortKey",
    "capture_repr": ("_repr_html_",),
    "junit": os.path.join("..", "test-results", "sphinx-gallery", "junit.xml"),
    "matplotlib_animations": True,
    "compress_images": compress_images,
    "filename_pattern": "^((?!sgskip).)*$",
    "exclude_implicit_doc": {
        r"mne\.io\.read_raw_fif",
        r"mne\.io\.Raw",
        r"mne\.Epochs",
        r"mne.datasets.*",
    },
    "show_api_usage": "unused",
    "api_usage_ignore": (
        "("
        ".*__.*__|"  # built-ins
        ".*Base.*|.*Array.*|mne.Vector.*|mne.Mixed.*|mne.Vol.*|"  # inherited
        "mne.coreg.Coregistration.*|"  # GUI
        # common
        ".*utils.*|.*verbose()|.*copy()|.*update()|.*save()|"
        ".*get_data()|"
        # mixins
        ".*add_channels()|.*add_reference_channels()|"
        ".*anonymize()|.*apply_baseline()|.*apply_function()|"
        ".*apply_hilbert()|.*as_type()|.*decimate()|"
        ".*drop()|.*drop_channels()|.*drop_log_stats()|"
        ".*export()|.*get_channel_types()|"
        ".*get_montage()|.*interpolate_bads()|.*next()|"
        ".*pick()|.*pick_channels()|.*pick_types()|"
        ".*plot_sensors()|.*rename_channels()|"
        ".*reorder_channels()|.*savgol_filter()|"
        ".*set_eeg_reference()|.*set_channel_types()|"
        ".*set_meas_date()|.*set_montage()|.*shift_time()|"
        ".*time_as_index()|.*to_data_frame()|"
        # dictionary inherited
        ".*clear()|.*fromkeys()|.*get()|.*items()|"
        ".*keys()|.*pop()|.*popitem()|.*setdefault()|"
        ".*values()|"
        # sklearn inherited
        ".*apply()|.*decision_function()|.*fit()|"
        ".*fit_transform()|.*get_params()|.*predict()|"
        ".*predict_proba()|.*set_params()|.*transform()|"
        # I/O, also related to mixins
        ".*.remove.*|.*.write.*)"
    ),
    "copyfile_regex": r".*index\.rst",  # allow custom index.rst files
    "parallel": sphinx_gallery_parallel,
}
assert is_serializable(sphinx_gallery_conf)

# ---------------------------------------------------------------------------
# Drop the "Open in JupyterLite" launch badge from gallery pages whose
# notebooks cannot run in the browser kernel at all: they need the R runtime
# (rpy2), a compiled package Pyodide does not ship (antio), or multi-GB
# datasets that cannot be bundled/slimmed. sphinx-gallery adds the badge to
# every example unconditionally, so we wrap its badge generator and return an
# empty string for these files. This only removes the badge/link — the
# notebook source is untouched (no in-code guard). Files that merely need data
# bundled, a pure-Python package installed, or pyvista 3D are NOT listed here
# (they are fixable, not impossible).
JUPYTERLITE_EXCLUDE = (
    # Tier 1 — impossible: R runtime / compiled package / huge single dataset
    "examples/stats/r_interop.py",  # rpy2 -> needs the R runtime
    "examples/io/read_impedances.py",  # antio (compiled, not in Pyodide)
    "examples/decoding/decoding_rsa_sgskip.py",  # visual_92_categories ~6 GB
    "examples/decoding/decoding_spoc_CMC.py",  # fieldtrip_cmc ~700 MB
    "examples/decoding/ssd_spatial_filters.py",  # fieldtrip_cmc ~700 MB
    # Tier 2 — multi-GB datasets (brainstorm / spm_face / opm / hf_sef)
    "examples/datasets/brainstorm_data.py",
    "examples/datasets/hf_sef_data.py",
    "examples/datasets/opm_data.py",
    "examples/datasets/spm_faces_dataset.py",
    "examples/preprocessing/movement_detection.py",
    "examples/preprocessing/muscle_detection.py",
    "examples/preprocessing/otp.py",
    "examples/time_frequency/source_power_spectrum_opm.py",
    "examples/visualization/evoked_arrowmap.py",
    "examples/visualization/meg_sensors.py",
    "tutorials/inverse/80_brainstorm_phantom_elekta.py",
    "tutorials/inverse/85_brainstorm_phantom_ctf.py",
    "tutorials/io/60_ctf_bst_auditory.py",
    "tutorials/preprocessing/80_opm_processing.py",
    # Tier 3 — several blockers each, none of them worth clearing on its own
    # the volume inverse is ~178 MB and volume source estimates are not
    # rendered in the browser
    "examples/inverse/compute_mne_inverse_volume.py",
    # needs aseg.mgz and the mixed source space, and calls src.plot(), which
    # is the 3D SourceSpaces view
    "examples/inverse/mixed_source_space_inverse.py",
    # nilearn.datasets.load_mni152_template() downloads a template at runtime,
    # which the browser blocks (CORS); the surrounding try only catches
    # TypeError, so the failure is not survivable
    "tutorials/inverse/20_dipole_fit.py",
    # make_field_map(upsampling=2) subdivides the helmet mesh through VTK, and
    # plot_field needs the interactive viewer that the browser renderer skips
    "examples/visualization/mne_helmet.py",
    # Tier 4 — mne.viz.Brain. The browser renderer draws static meshes; Brain
    # additionally wants dock widgets, a toolbar and a time slider, so these
    # are blocked on the interactive layer rather than on data.
    "examples/visualization/brain.py",
    "examples/visualization/parcellation.py",
    "tutorials/clinical/20_seeg.py",
    "tutorials/forward/10_background_freesurfer.py",
    "tutorials/forward/50_background_freesurfer_mne.py",
    "tutorials/inverse/60_visualize_stc.py",
    "tutorials/io/30_reading_fnirs_data.py",
    "tutorials/preprocessing/70_fnirs_processing.py",
    # Tier 5 — one-off blockers with no browser path
    # plot_field needs the interactive viewer
    "tutorials/evoked/20_visualize_evoked.py",
    # the three-layer BEM solution alone is 237 MB
    "examples/inverse/multi_dipole_model.py",
    # openneuro fetches the recording at runtime, which the browser blocks
    "examples/preprocessing/esg_rm_heart_artefact_pcaobs.py",
    # physionet.org is not CORS-enabled and the dataset is not on the CI box
    "tutorials/clinical/60_sleep.py",
    # the 4D/BTi phantom dataset is not among the ones CI downloads
    "tutorials/inverse/90_phantom_4DBTi.py",
    # needs mne_bids as well as the epilepsy_ecog dataset and 3D sensor views
    "tutorials/clinical/30_ecog.py",
    # Tier 6 — fetch_fsaverage. _manifest_check_download only skips the
    # download when every one of its ~190 manifest entries is already present,
    # so fsaverage cannot be part-bundled, and MNE-sample-data ships no
    # fsaverage/bem at all. The volume forward and inverse these two want are
    # 187 MB and 360 MB on top of that.
    "examples/inverse/morph_volume_stc.py",
    "tutorials/inverse/50_beamformer_lcmv.py",
    "examples/visualization/montage_sgskip.py",
    # same, plus fetch_infant_template downloads a second template
    "tutorials/forward/35_eeg_no_mri.py",
    # snapshot_brain_montage needs a real 3D window to read pixels back from
    "examples/visualization/3d_to_2d.py",
    # the three-layer BEM solution is 237 MB, and T1_electrodes.mgz would pull
    # in the misc dataset's MRI as well
    "tutorials/inverse/70_eeg_mri_coords.py",
    # mne_bids is not installable in the browser kernel
    "tutorials/inverse/95_phantom_KIT.py",
    # Tier 7 — somato. Serving it costs 404 MB (the raw alone is 344 MB) on
    # every docs deploy, which is more than these six pages are worth; the
    # dataset is not copied at all. Restoring them means putting the somato
    # block back in the copy step above.
    "examples/inverse/dics_epochs.py",
    "examples/inverse/dics_source_power.py",
    "examples/inverse/evoked_ers_source_power.py",
    "examples/inverse/multidict_reweighted_tfmxne.py",
    "examples/time_frequency/time_frequency_global_field_power.py",
    "tutorials/time-freq/20_sensors_time_frequency.py",
    # Single recordings well past LITE_MAX_FILE_MB, confirmed against the full
    # build: 379 MB and 251 MB for one example each, so they are skipped by the
    # copy step and the badge would have nothing to load.
    "examples/datasets/kernel_phantom.py",
    "examples/io/elekta_epochs.py",
    # These want EEGBCI runs 1 and 2, which tools/circleci_download.sh never
    # fetches (it takes subject 1 runs 3/6/10/14 and run 3 for subjects 2-4),
    # so the data is not on the machine that builds the docs. eeg_bridging
    # alone would need run 1 for ten subjects.
    "examples/visualization/onionskin.py",
    "examples/preprocessing/muscle_ica.py",
    "examples/preprocessing/eeg_bridging.py",
    # Both Report tutorials build their figures by screenshotting a 3D scene
    # (Report._itv calls backend._take_3d_screenshot), and vtk.js cannot hand a
    # framebuffer back to Python, so those sections would embed blank images.
    # 70_report additionally round-trips a report through HDF5.
    "tutorials/intro/70_report.py",
    "tutorials/preprocessing/14_quality_control_report.py",
)

import sphinx_gallery.gen_rst as _sg_gen_rst  # noqa: E402

_orig_gen_jupyterlite_rst = _sg_gen_rst.gen_jupyterlite_rst


def _lite_badge_filtered(fpath, gallery_conf):
    """Return the JupyterLite badge reST, or "" for excluded notebooks."""
    _p = str(fpath).replace(os.sep, "/")
    if any(_p.endswith(_ex) for _ex in JUPYTERLITE_EXCLUDE):
        return ""
    return _orig_gen_jupyterlite_rst(fpath, gallery_conf)


_sg_gen_rst.gen_jupyterlite_rst = _lite_badge_filtered
# Files were renamed from plot_* with:
# find . -type f -name 'plot_*.py' -exec sh -c 'x="{}"; xn=`basename "${x}"`; git mv "$x" `dirname "${x}"`/${xn:5}' \;  # noqa


def append_attr_meth_examples(app, what, name, obj, options, lines):
    """Append SG examples backreferences to method and attr docstrings."""
    # NumpyDoc nicely embeds method and attribute docstrings for us, but it
    # does not respect the autodoc templates that would otherwise insert
    # the .. include:: lines, so we need to do it.
    # Eventually this could perhaps live in SG.
    if what in ("attribute", "method"):
        size = os.path.getsize(
            os.path.join(
                os.path.dirname(__file__),
                "generated",
                f"{name}.examples",
            )
        )
        if size > 0:
            lines += """
.. _sphx_glr_backreferences_{1}:

.. rubric:: Examples using ``{0}``:

.. minigallery:: {1}

""".format(name.split(".")[-1], name).split("\n")


def fix_sklearn_inherited_docstrings(app, what, name, obj, options, lines):
    """Fix sklearn docstrings because they use autolink and we do not."""
    if (
        name.startswith("mne.decoding.") or name.startswith("mne.preprocessing.Xdawn")
    ) and name.endswith(
        (
            ".get_metadata_routing",
            ".fit",
            ".fit_transform",
            ".set_output",
            ".transform",
        )
    ):
        if ":Parameters:" in lines:
            loc = lines.index(":Parameters:")
        else:
            loc = lines.index(":Returns:")
        lines.insert(loc, "")
        lines.insert(loc, ".. default-role:: autolink")
        lines.insert(loc, "")


# -- Other extension configuration -------------------------------------------

# Consider using http://magjac.com/graphviz-visual-editor for this
graphviz_dot_args = [
    "-Gsep=-0.5",
    "-Gpad=0.5",
    "-Nshape=box",
    "-Nfontsize=20",
    "-Nfontname=Open Sans,Arial",
]
graphviz_output_format = "svg"  # for API usage diagrams
user_agent = "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Mobile Safari/537.36"  # noqa: E501
# Can eventually add linkcheck_request_headers if needed
linkcheck_ignore = [  # will be compiled to regex
    # 403 Client Error: Forbidden
    "https://doi.org/10.1002/",  # onlinelibrary.wiley.com/doi/10.1002/hbm
    "https://doi.org/10.1017/",  # cambridge.org
    "https://doi.org/10.1016/",  # neuroimage
    "https://doi.org/10.1021/",  # pubs.acs.org/doi/abs
    "https://doi.org/10.1063/",  # pubs.aip.org/aip/jap
    "https://doi.org/10.1073/",  # pnas.org
    "https://doi.org/10.1080/",  # www.tandfonline.com
    "https://doi.org/10.1088/",  # www.tandfonline.com
    "https://doi.org/10.1090/",  # ams.org
    "https://doi.org/10.1093/",  # academic.oup.com/sleep/
    "https://doi.org/10.1098/",  # royalsocietypublishing.org
    "https://doi.org/10.1101/",  # www.biorxiv.org
    "https://doi.org/10.1103/",  # journals.aps.org/rmp
    "https://doi.org/10.1111/",  # onlinelibrary.wiley.com/doi/10.1111/psyp
    "https://doi.org/10.1126/",  # www.science.org
    "https://doi.org/10.1137/",  # epubs.siam.org
    "https://doi.org/10.1145/",  # dl.acm.org
    "https://doi.org/10.5281/",  # zenodo.org
    "https://doi.org/10.1155/",  # www.hindawi.com/journals/cin
    "https://doi.org/10.1161/",  # www.ahajournals.org
    "https://doi.org/10.1162/",  # direct.mit.edu/neco/article/
    "https://doi.org/10.1167/",  # jov.arvojournals.org
    "https://doi.org/10.1177/",  # journals.sagepub.com
    "https://doi.org/10.1523/",  # jneurosci.org
    "https://doi.org/10.3109/",  # www.tandfonline.com
    "https://doi.org/10.3390/",  # mdpi.com
    "https://hms.harvard.edu/",  # doc/funding.rst
    "https://stackoverflow.com/questions/21752259/python-why-pickle",  # doc/help/faq
    "https://mitpress.mit.edu/9780262525855",  # works but linkcheck fails to resolve
    "https://zenodo.org",  # doc/help/faq
    "https://blender.org",
    "https://home.alexk101.dev",
    "https://www.mq.edu.au/",
    "https://www.biorxiv.org/content/10.1101/",  # biorxiv.org
    "https://www.researchgate.net/profile/",
    "https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html",
    r"https://scholar.google.com/scholar\?cites=12188330066413208874&as_ylo=2014",
    r"https://scholar.google.com/scholar\?cites=1521584321377182930&as_ylo=2013",
    "https://www.research.chop.edu/imaging",
    "http://prdownloads.sourceforge.net/optipng",
    "https://sourceforge.net/projects/aespa/files/",
    "https://sourceforge.net/projects/ezwinports/files/",
    r"https://.*\.sourceforge\.net/",
    "https://www.cogsci.nl/smathot",
    "https://www.mathworks.com/products/compiler/matlab-runtime.html",
    "https://medicine.umich.edu/dept/khri/ross-maddox-phd",
    "http://blog.kaggle.com/2015/08/12/july-2015-scripts-of-the-week",
    # 500 server error
    "https://openwetware.org/wiki/Beauchamp:FreeSurfer",
    # 503 Server error
    "https://hal.archives-ouvertes.fr/hal-01848442",
    # Read timed out
    "http://www.cs.ucl.ac.uk/staff/d.barber/brml",
    "https://www.cea.fr",
    "http://www.humanconnectome.org/data",
    "https://www.mail-archive.com/freesurfer@nmr.mgh.harvard.edu",
    "https://surfer.nmr.mgh.harvard.edu/fswiki/mri_normalize",
    "https://launchpad.net",
    # Max retries exceeded
    "https://doi.org/10.7488/ds/1556",
    "https://datashare.is.ed.ac.uk/handle/10283",
    "https://imaging.mrc-cbu.cam.ac.uk/imaging/MniTalairach",
    "https://www.nyu.edu/",
    # Too slow
    "https://speakerdeck.com/dengemann/",
    "https://www.dtu.dk/english/service/phonebook/person",
    "https://www.gnu.org/software/make/",
    "https://www.macports.org/",
    "https://hastie.su.domains/CASI",
    # SSL problems sometimes
    "http://ilabs.washington.edu",
    "https://psychophysiology.cpmc.columbia.edu",
    "https://erc.easme-web.eu",
    "https://www.crnl.fr",
    # Spurious failure
    "https://megcore.nih.gov/index.php/Staff",
    # Not rendered by linkcheck builder
    r"ides\.html",
    # Sponsors not rendered properly by linkcheck builder
    "{{inst.url}}",
]
linkcheck_anchors = False  # saves a bit of time
linkcheck_timeout = 15  # some can be quite slow
linkcheck_retries = 3
linkcheck_report_timeouts_as_broken = False

# autodoc / autosummary
autosummary_generate = True
autodoc_default_options = {"inherited-members": None}
# Types are documented (in human-readable numpydoc form) in the docstrings
# themselves, so don't also render the annotations into the signatures.
autodoc_typehints = "none"

# sphinxcontrib-bibtex
bibtex_bibfiles = ["./references.bib"]
bibtex_style = "unsrt"
bibtex_footbibliography_header = ""


# -- Nitpicky ----------------------------------------------------------------

nitpicky = True
show_warning_types = True
nitpick_ignore = [
    ("py:class", "None.  Remove all items from D."),
    (
        "py:class",
        "v, remove specified key and return the corresponding value.",
    ),  # noqa: E501
    ("py:class", "an object providing a view on D's values"),
    ("py:class", "a shallow copy of D"),
    ("py:class", "(k, v), remove and return some (key, value) pair as a"),
    ("py:class", "_FuncT"),  # type hint used in @verbose decorator
    ("py:class", "mne.utils._logging._FuncT"),
    ("py:class", "None.  Remove all items from od."),
]
nitpick_ignore_regex = [
    ("py:class", "a set-like object providing a view on D's (items|keys)"),
    ("py:class", r"None\.  Update D from (dict|mapping)/iterable E and F\."),
    # Classes whose methods we purposefully do not document
    ("py:.*", r"mne\.io\.BaseRaw.*"),  # use mne.io.Raw
    ("py:.*", r"mne\.BaseEpochs.*"),  # use mne.Epochs
    # Type hints for undocumented types
    ("py:.*", r"mne\.io\..*\.Raw.*"),  # RawEDF etc.
    ("py:.*", r"mne\.epochs\.EpochsFIF.*"),
    ("py:.*", r"mne\.io\..*\.Epochs.*"),  # EpochsKIT etc.
    (  # BaseRaw attributes are documented in Raw
        "py:obj",
        "(filename|metadata|proj|times|tmax|tmin|annotations|ch_names"
        "|compensation_grade|duration|filenames|first_samp|first_time"
        "|last_samp|n_times|proj|times|tmax|tmin)",
    ),
]
suppress_warnings = [
    "image.nonlocal_uri",  # we intentionally link outside
]


# -- Sphinx hacks / overrides ------------------------------------------------

versionlabels["versionadded"] = sphinx.locale._("New in v%s")

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
switcher_version_match = "dev" if ".dev" in release else version
html_theme_options = {
    "icon_links": [
        dict(
            name="Discord (office hours)",
            url="https://discord.gg/rKfvxTuATa",
            icon="fa-brands fa-discord fa-fw",
        ),
        dict(
            name="Mastodon",
            url="https://fosstodon.org/@mne",
            icon="fa-brands fa-mastodon fa-fw",
            attributes=dict(rel="me"),
        ),
        dict(
            name="Q&A Forum",
            url="https://mne.discourse.group/",
            icon="fa-brands fa-discourse fa-fw",
        ),
        dict(
            name="Code Repository",
            url="https://github.com/mne-tools/mne-python",
            icon="fa-brands fa-github fa-fw",
        ),
        dict(
            name="Sponsor us on GitHub",
            url="https://github.com/sponsors/mne-tools",
            icon="fa-regular fa-heart fa-fw",
        ),
        dict(
            name="Donate via OpenCollective",
            url="https://opencollective.com/mne-python",
            icon="fa-custom fa-opencollective fa-fw",
        ),
    ],
    "icon_links_label": "External Links",  # for screen reader
    "use_edit_page_button": False,
    "navigation_with_keys": False,
    "show_toc_level": 1,
    "article_header_start": [],  # disable breadcrumbs
    "navbar_end": [
        "theme-switcher",
        "version-switcher",
        "navbar-icon-links",
    ],
    "navbar_align": "left",
    "navbar_persistent": ["search-button"],
    "footer_start": ["copyright"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "analytics": dict(google_analytics_id="G-5TBCPCRB6X"),
    "switcher": {
        "json_url": "https://mne.tools/dev/_static/versions.json",
        "version_match": switcher_version_match,
    },
    "back_to_top_button": False,
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/mne_logo_small.svg"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/favicon.ico"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "style.css",
]
html_js_files = [
    ("js/custom-icons.js", {"defer": "defer"}),
]

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
html_extra_path = [
    "contributing.html",
    "documentation.html",
    "getting_started.html",
    "install_mne_python.html",
    # Serve the pre-bundled JupyterLite sample data at the docs root
    # (e.g. /mne_data/...). The lite setup cell fetches it over HTTP.
    "lite_extra",
]

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    "index": ["sidebar-quicklinks.html"],
}

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False
html_copy_source = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# sponsor and partner logos
with open("_static/sponsors.yml") as fid:
    sponsors_partners = safe_load(fid)
current = sponsors_partners.pop("current")
# sponsors
current_sponsors = list()
former_sponsors = list()
for key, val in sponsors_partners["sponsors"].items():
    if "img" in val:
        val["name"] = key
        (current_sponsors if key in current else former_sponsors).append(val)
    else:
        assert "light" in val and "dark" in val
        for mode in ("light", "dark"):
            (current_sponsors if key in current else former_sponsors).append(
                dict(
                    name=f"{key}{'_dk' if mode == 'dark' else ''}",
                    title=val["title"],
                    img=val[mode],
                    klass=f"only-{mode}",
                )
            )
# institutions
current_institutions = list()
former_institutions = list()
for key, val in sponsors_partners["partner_institutions"].items():
    if "img" in val:
        val["name"] = key
        (current_institutions if key in current else former_institutions).append(val)
    else:
        assert "light" in val and "dark" in val
        for mode in ("light", "dark"):
            (current_institutions if key in current else former_institutions).append(
                dict(
                    name=f"{key}{'_dk' if mode == 'dark' else ''}",
                    title=val["title"],
                    img=val[mode],
                    klass=f"only-{mode}",
                    url=val["url"],
                )
            )
# variables to pass to HTML templating engine
html_context = {
    "default_mode": "auto",
    # next 3 are for the "edit this page" button
    "github_user": "mne-tools",
    "github_repo": "mne-python",
    "github_version": "main",
    "doc_path": "doc",
    "current_sponsors_partners": current,
    "current_sponsors": current_sponsors,
    "former_sponsors": former_sponsors,
    "all_sponsors": [*current_sponsors, *former_sponsors],
    "current_institutions": current_institutions,
    "former_institutions": former_institutions,
    "all_institutions": [*current_institutions, *former_institutions],
    # \u00AD is an optional hyphen (not rendered unless needed)
    # If these are changed, the Makefile should be updated, too
    "carousel": [
        dict(
            title="Source Estimation",
            text="Distributed, sparse, mixed-norm, beam\u00adformers, dipole fitting, and more.",  # noqa E501
            url="auto_tutorials/inverse/index.html",
            img="sphx_glr_30_mne_dspm_loreta_008.gif",
            alt="dSPM",
        ),
        dict(
            title="Machine Learning",
            text="Advanced decoding models including time general\u00adiza\u00adtion.",
            url="auto_tutorials/machine-learning/50_decoding.html",
            img="sphx_glr_50_decoding_006.png",
            alt="Decoding",
        ),
        dict(
            title="Encoding Models",
            text="Receptive field estima\u00adtion with optional smooth\u00adness priors.",  # noqa E501
            url="auto_tutorials/machine-learning/30_strf.html",
            img="sphx_glr_30_strf_001.png",
            alt="STRF",
        ),
        dict(
            title="Statistics",
            text="Parametric and non-parametric, permutation tests and clustering.",
            url="auto_tutorials/stats-source-space/index.html",
            img="sphx_glr_20_cluster_1samp_spatiotemporal_001.png",
            alt="Clusters",
        ),
        dict(
            title="Connectivity",
            text="All-to-all spectral and effective connec\u00adtivity measures.",
            url="https://mne.tools/mne-connectivity/stable/auto_examples/mne_inverse_label_connectivity.html",  # noqa E501
            img="https://mne.tools/mne-connectivity/stable/_images/sphx_glr_mne_inverse_label_connectivity_001.png",  # noqa E501
            alt="Connectivity",
        ),
        dict(
            title="Data Visualization",
            text="Explore your data from multiple perspectives.",
            url="auto_tutorials/evoked/20_visualize_evoked.html",
            img="sphx_glr_20_visualize_evoked_010.png",
            alt="Visualization",
        ),
    ],
}

# Output file base name for HTML help builder.
htmlhelp_basename = "mne-doc"


# -- Options for plot_directive ----------------------------------------------

# Adapted from SciPy
plot_include_source = True
plot_formats = [("png", 96)]
plot_html_show_formats = False
plot_html_show_source_link = False
font_size = 13 * 72 / 96.0  # 13 px
plot_rcparams = {
    "font.size": font_size,
    "axes.titlesize": font_size,
    "axes.labelsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "legend.fontsize": font_size,
    "figure.figsize": (6, 5),
    "figure.subplot.bottom": 0.2,
    "figure.subplot.left": 0.2,
    "figure.subplot.right": 0.9,
    "figure.subplot.top": 0.85,
    "figure.subplot.wspace": 0.4,
    "text.usetex": False,
}


# -- Options for LaTeX output ------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto/manual]).
latex_documents = []

# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = "_static/logo.png"

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
latex_toplevel_sectioning = "part"

# -- Warnings management -----------------------------------------------------
reset_warnings(None, None)

# -- Fontawesome support -----------------------------------------------------
brand_icons = ("apple", "linux", "windows", "discourse", "python")
fixed_width_icons = (
    # homepage:
    "book",
    "code-branch",
    "newspaper",
    "circle-question",
    "quote-left",
    # contrib guide:
    "bug-slash",
    "comment",
    "computer-mouse",
    "hand-sparkles",
    "pencil",
    "text-slash",
    "universal-access",
    "wand-magic-sparkles",
    "discourse",
    "python",
)
other_icons = (
    "hand-paper",
    "question",
    "rocket",
    "server",
    "code",
    "desktop",
    "terminal",
    "cloud-arrow-down",
    "wrench",
    "hourglass-half",
)
icon_class = dict()
for icon in brand_icons + fixed_width_icons + other_icons:
    icon_class[icon] = ("fa-brands",) if icon in brand_icons else ("fa-solid",)
    icon_class[icon] += ("fa-fw",) if icon in fixed_width_icons else ()

rst_prolog = ""
for icon, classes in icon_class.items():
    rst_prolog += f"""
.. |{icon}| raw:: html

    <i class="{" ".join(classes + (f"fa-{icon}",))}"></i>
"""

rst_prolog += """
.. |ensp| unicode:: U+2002 .. EN SPACE

.. include:: /links.inc
.. include:: /changes/names.inc

.. currentmodule:: mne
"""

# -- Dependency info ----------------------------------------------------------

min_py = "3.10"
min_py_minor = "10"
rst_prolog += f"\n.. |min_python_version| replace:: {min_py}\n"

# Static list created 2021/04/13 based on what we needed to redirect,
# since we don't need to add redirects for examples added after this date.
needed_plot_redirects = {
    # tutorials
    "10_epochs_overview.py",
    "10_evoked_overview.py",
    "10_overview.py",
    "10_preprocessing_overview.py",
    "10_raw_overview.py",
    "10_reading_meg_data.py",
    "15_handling_bad_channels.py",
    "20_event_arrays.py",
    "20_events_from_raw.py",
    "20_reading_eeg_data.py",
    "20_rejecting_bad_data.py",
    "20_visualize_epochs.py",
    "20_visualize_evoked.py",
    "30_annotate_raw.py",
    "30_epochs_metadata.py",
    "30_filtering_resampling.py",
    "30_info.py",
    "30_reading_fnirs_data.py",
    "35_artifact_correction_regression.py",
    "40_artifact_correction_ica.py",
    "40_autogenerate_metadata.py",
    "40_sensor_locations.py",
    "40_visualize_raw.py",
    "45_projectors_background.py",
    "50_artifact_correction_ssp.py",
    "50_configure_mne.py",
    "50_epochs_to_data_frame.py",
    "55_setting_eeg_reference.py",
    "59_head_positions.py",
    "60_make_fixed_length_epochs.py",
    "60_maxwell_filtering_sss.py",
    "70_fnirs_processing.py",
    # examples
    "3d_to_2d.py",
    "brainstorm_data.py",
    "channel_epochs_image.py",
    "cluster_stats_evoked.py",
    "compute_csd.py",
    "compute_mne_inverse_epochs_in_label.py",
    "compute_mne_inverse_raw_in_label.py",
    "compute_mne_inverse_volume.py",
    "compute_source_psd_epochs.py",
    "covariance_whitening_dspm.py",
    "custom_inverse_solver.py",
    "decoding_csp_eeg.py",
    "decoding_csp_timefreq.py",
    "decoding_spatio_temporal_source.py",
    "decoding_spoc_CMC.py",
    "decoding_time_generalization_conditions.py",
    "decoding_unsupervised_spatial_filter.py",
    "decoding_xdawn_eeg.py",
    "define_target_events.py",
    "dics_source_power.py",
    "eeg_csd.py",
    "eeg_on_scalp.py",
    "eeglab_head_sphere.py",
    "elekta_epochs.py",
    "ems_filtering.py",
    "eog_artifact_histogram.py",
    "evoked_arrowmap.py",
    "evoked_ers_source_power.py",
    "evoked_topomap.py",
    "evoked_whitening.py",
    "fdr_stats_evoked.py",
    "find_ref_artifacts.py",
    "fnirs_artifact_removal.py",
    "forward_sensitivity_maps.py",
    "gamma_map_inverse.py",
    "hf_sef_data.py",
    "ica_comparison.py",
    "interpolate_bad_channels.py",
    "label_activation_from_stc.py",
    "label_from_stc.py",
    "label_source_activations.py",
    "left_cerebellum_volume_source.py",
    "limo_data.py",
    "linear_model_patterns.py",
    "linear_regression_raw.py",
    "meg_sensors.py",
    "mixed_norm_inverse.py",
    "mixed_source_space_inverse.py",
    "mne_cov_power.py",
    "mne_helmet.py",
    "mne_inverse_coherence_epochs.py",
    "mne_inverse_envelope_correlation.py",
    "mne_inverse_envelope_correlation_volume.py",
    "mne_inverse_psi_visual.py",
    "morph_surface_stc.py",
    "morph_volume_stc.py",
    "movement_compensation.py",
    "movement_detection.py",
    "multidict_reweighted_tfmxne.py",
    "muscle_detection.py",
    "opm_data.py",
    "otp.py",
    "parcellation.py",
    "psf_ctf_label_leakage.py",
    "psf_ctf_vertices.py",
    "psf_ctf_vertices_lcmv.py",
    "publication_figure.py",
    "rap_music.py",
    "trap_music.py",
    "read_inverse.py",
    "read_neo_format.py",
    "read_noise_covariance_matrix.py",
    "read_stc.py",
    "receptive_field_mtrf.py",
    "resolution_metrics.py",
    "resolution_metrics_eegmeg.py",
    "roi_erpimage_by_rt.py",
    "sensor_noise_level.py",
    "sensor_permutation_test.py",
    "sensor_regression.py",
    "shift_evoked.py",
    "simulate_evoked_data.py",
    "simulate_raw_data.py",
    "simulated_raw_data_using_subject_anatomy.py",
    "snr_estimate.py",
    "source_label_time_frequency.py",
    "source_power_spectrum.py",
    "source_power_spectrum_opm.py",
    "source_simulator.py",
    "source_space_morphing.py",
    "source_space_snr.py",
    "source_space_time_frequency.py",
    "ssd_spatial_filters.py",
    "ssp_projs_sensitivity_map.py",
    "temporal_whitening.py",
    "time_frequency_erds.py",
    "time_frequency_global_field_power.py",
    "time_frequency_mixed_norm_inverse.py",
    "time_frequency_simulated.py",
    "topo_compare_conditions.py",
    "topo_customized.py",
    "vector_mne_solution.py",
    "virtual_evoked.py",
    "xdawn_denoising.py",
    "xhemi.py",
}
api_redirects = {
    "connectivity",
    "covariance",
    "creating_from_arrays",
    "datasets",
    "decoding",
    "events",
    "export",
    "file_io",
    "forward",
    "inverse",
    "logging",
    "most_used_classes",
    "mri",
    "preprocessing",
    "python_reference",
    "reading_raw_data",
    "realtime",
    "report",
    "sensor_space",
    "simulation",
    "source_space",
    "statistics",
    "time_frequency",
    "visualization",
}
ex = "auto_examples"
co = "connectivity"
mne_conn = "https://mne.tools/mne-connectivity/stable"
tu = "auto_tutorials"
pr = "preprocessing"
di = "discussions"
sm = "source-modeling"
fw = "forward"
nv = "inverse"
sn = "stats-sensor-space"
sr = "stats-source-space"
sd = "sample-datasets"
ml = "machine-learning"
tf = "time-freq"
si = "simulation"
vi = "visualization"
custom_redirects = {
    # Custom redirects (one HTML path to another, relative to outdir)
    # can be added here as fr->to key->value mappings
    "credit": "credits/credit",
    "funding": "credits/sponsors",
    "install/contributing": "development/contributing",
    "overview/cite": "documentation/cite",
    "overview/get_help": "help/index",
    "overview/people": "credits/leaders",
    "overview/roadmap": "development/roadmap",
    "whats_new": "development/whats_new",
    f"{tu}/evoked/plot_eeg_erp": f"{tu}/evoked/30_eeg_erp",
    f"{tu}/evoked/plot_whitened": f"{tu}/evoked/40_whitened",
    f"{tu}/misc/plot_modifying_data_inplace": f"{tu}/intro/15_inplace",
    f"{tu}/misc/plot_report": f"{tu}/intro/70_report",
    f"{tu}/misc/plot_seeg": f"{tu}/clinical/20_seeg",
    f"{tu}/misc/plot_ecog": f"{tu}/clinical/30_ecog",
    f"{tu}/{ml}/plot_receptive_field": f"{tu}/{ml}/30_strf",
    f"{tu}/{ml}/plot_sensors_decoding": f"{tu}/{ml}/50_decoding",
    f"{tu}/{sm}/plot_background_freesurfer": f"{tu}/{fw}/10_background_freesurfer",
    f"{tu}/{sm}/plot_source_alignment": f"{tu}/{fw}/20_source_alignment",
    f"{tu}/{sm}/plot_forward": f"{tu}/{fw}/30_forward",
    f"{tu}/{sm}/plot_eeg_no_mri": f"{tu}/{fw}/35_eeg_no_mri",
    f"{tu}/{sm}/plot_background_freesurfer_mne": f"{tu}/{fw}/50_background_freesurfer_mne",  # noqa E501
    f"{tu}/{sm}/plot_fix_bem_in_blender": f"{tu}/{fw}/80_fix_bem_in_blender",
    f"{tu}/{sm}/plot_compute_covariance": f"{tu}/{fw}/90_compute_covariance",
    f"{tu}/{sm}/plot_object_source_estimate": f"{tu}/{nv}/10_stc_class",
    f"{tu}/{sm}/plot_dipole_fit": f"{tu}/{nv}/20_dipole_fit",
    f"{tu}/{sm}/plot_mne_dspm_source_localization": f"{tu}/{nv}/30_mne_dspm_loreta",
    f"{tu}/{sm}/plot_dipole_orientations": f"{tu}/{nv}/35_dipole_orientations",
    f"{tu}/{sm}/plot_mne_solutions": f"{tu}/{nv}/40_mne_fixed_free",
    f"{tu}/{sm}/plot_beamformer_lcmv": f"{tu}/{nv}/50_beamformer_lcmv",
    f"{tu}/{sm}/plot_visualize_stc": f"{tu}/{nv}/60_visualize_stc",
    f"{tu}/{sm}/plot_eeg_mri_coords": f"{tu}/{nv}/70_eeg_mri_coords",
    f"{tu}/{sd}/plot_brainstorm_phantom_elekta": f"{tu}/{nv}/80_brainstorm_phantom_elekta",  # noqa E501
    f"{tu}/{sd}/plot_brainstorm_phantom_ctf": f"{tu}/{nv}/85_brainstorm_phantom_ctf",
    f"{tu}/{sd}/plot_phantom_4DBTi": f"{tu}/{nv}/90_phantom_4DBTi",
    f"{tu}/{sd}/plot_brainstorm_auditory": f"{tu}/io/60_ctf_bst_auditory",
    f"{tu}/{sd}/plot_sleep": f"{tu}/clinical/60_sleep",
    f"{tu}/{di}/plot_background_filtering": f"{tu}/{pr}/25_background_filtering",
    f"{tu}/{di}/plot_background_statistics": f"{tu}/{sn}/10_background_stats",
    f"{tu}/{sn}/plot_stats_cluster_erp": f"{tu}/{sn}/20_erp_stats",
    f"{tu}/{sn}/plot_stats_cluster_1samp_test_time_frequency": f"{tu}/{sn}/40_cluster_1samp_time_freq",  # noqa E501
    f"{tu}/{sn}/plot_stats_cluster_time_frequency": f"{tu}/{sn}/50_cluster_between_time_freq",  # noqa E501
    f"{tu}/{sn}/plot_stats_spatio_temporal_cluster_sensors": f"{tu}/{sn}/75_cluster_ftest_spatiotemporal",  # noqa E501
    f"{tu}/{sr}/plot_stats_cluster_spatio_temporal": f"{tu}/{sr}/20_cluster_1samp_spatiotemporal",  # noqa E501
    f"{tu}/{sr}/plot_stats_cluster_spatio_temporal_2samp": f"{tu}/{sr}/30_cluster_ftest_spatiotemporal",  # noqa E501
    f"{tu}/{sr}/plot_stats_cluster_spatio_temporal_repeated_measures_anova": f"{tu}/{sr}/60_cluster_rmANOVA_spatiotemporal",  # noqa E501
    f"{tu}/{sr}/plot_stats_cluster_time_frequency_repeated_measures_anova": f"{tu}/{sn}/70_cluster_rmANOVA_time_freq",  # noqa E501
    f"{tu}/{tf}/plot_sensors_time_frequency": f"{tu}/{tf}/20_sensors_time_frequency",
    f"{tu}/{tf}/plot_ssvep": f"{tu}/{tf}/50_ssvep",
    f"{tu}/{si}/plot_creating_data_structures": f"{tu}/{si}/10_array_objs",
    f"{tu}/{si}/plot_point_spread": f"{tu}/{si}/70_point_spread",
    f"{tu}/{si}/plot_dics": f"{tu}/{si}/80_dics",
    f"{tu}/{tf}/plot_eyetracking": f"{tu}/{pr}/90_eyetracking_data",
    f"{ex}/{co}/mne_inverse_label_connectivity": f"{mne_conn}/{ex}/mne_inverse_label_connectivity",  # noqa E501
    f"{ex}/{co}/cwt_sensor_connectivity": f"{mne_conn}/{ex}/cwt_sensor_connectivity",
    f"{ex}/{co}/mixed_source_space_connectivity": f"{mne_conn}/{ex}/mixed_source_space_connectivity",  # noqa E501
    f"{ex}/{co}/mne_inverse_coherence_epochs": f"{mne_conn}/{ex}/mne_inverse_coherence_epochs",  # noqa E501
    f"{ex}/{co}/mne_inverse_connectivity_spectrum": f"{mne_conn}/{ex}/mne_inverse_connectivity_spectrum",  # noqa E501
    f"{ex}/{co}/mne_inverse_envelope_correlation_volume": f"{mne_conn}/{ex}/mne_inverse_envelope_correlation_volume",  # noqa E501
    f"{ex}/{co}/mne_inverse_envelope_correlation": f"{mne_conn}/{ex}/mne_inverse_envelope_correlation",  # noqa E501
    f"{ex}/{co}/mne_inverse_psi_visual": f"{mne_conn}/{ex}/mne_inverse_psi_visual",
    f"{ex}/{co}/sensor_connectivity": f"{mne_conn}/{ex}/sensor_connectivity",
    f"{ex}/{vi}/publication_figure": f"{tu}/{vi}/10_publication_figure",
    f"{ex}/{vi}/sensor_noise_level": f"{tu}/{pr}/50_artifact_correction_ssp",
}

# Adapted from sphinxcontrib/redirects (BSD-2-Clause)
REDIRECT_TEMPLATE = """\
<!DOCTYPE HTML>
<html lang="en-US">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="1; url={to}">
        <script type="text/javascript">
            window.location.href = "{to}"
        </script>
        <title>Page Redirection</title>
    </head>
    <body>
        If you are not redirected automatically, follow this <a href='{to}'>link</a>.
    </body>
</html>"""


def check_existing_redirect(path):
    """Make sure existing HTML files are redirects, before overwriting."""
    if path.is_file():
        with open(path) as fid:
            for _ in range(8):
                next(fid)
            line = fid.readline()
            if "Page Redirection" not in line:
                raise RuntimeError(
                    "Attempted overwrite of HTML file with a redirect, where the "
                    "original file was not already a redirect."
                )


def _check_valid_builder(app, exception):
    valid_builder = isinstance(app.builder, sphinx.builders.html.StandaloneHTMLBuilder)
    return valid_builder and exception is None


def make_gallery_redirects(app, exception):
    """Make HTML redirects for our sphinx gallery pages."""
    if not _check_valid_builder(app, exception):
        return
    sg_conf = app.config["sphinx_gallery_conf"]
    for src_dir, out_dir in zip(sg_conf["examples_dirs"], sg_conf["gallery_dirs"]):
        root = (Path(app.srcdir) / src_dir).resolve()
        fnames = [
            pyfile.relative_to(root)
            for pyfile in root.rglob(r"**/*.py")
            if pyfile.name in needed_plot_redirects
        ]
        # plot_ redirects
        for fname in fnames:
            dirname = Path(app.outdir) / out_dir / fname.parent
            to_fname = fname.with_suffix(".html").name
            fr_fname = f"plot_{to_fname}"
            to_path = dirname / to_fname
            fr_path = dirname / fr_fname
            assert to_path.is_file(), (fname, to_path)
            with open(fr_path, "w") as fid:
                fid.write(REDIRECT_TEMPLATE.format(to=to_fname))
        sphinx_logger.info(
            f"Added {len(fnames):3d} HTML plot_* redirects for {out_dir}"
        )


def make_api_redirects(app, exception):
    """Make HTML redirects for our API pages."""
    if not _check_valid_builder(app, exception):
        return

    for page in api_redirects:
        fname = f"{page}.html"
        fr_path = Path(app.outdir) / fname
        to_path = Path(app.outdir) / "api" / fname
        # allow overwrite if existing file is just a redirect
        check_existing_redirect(fr_path)
        with open(fr_path, "w") as fid:
            fid.write(REDIRECT_TEMPLATE.format(to=to_path))
    sphinx_logger.info(f"Added {len(api_redirects):3d} HTML API redirects")


def make_custom_redirects(app, exception):
    """Make HTML redirects for miscellaneous pages."""
    if not _check_valid_builder(app, exception):
        return

    for _fr, _to in custom_redirects.items():
        fr = f"{_fr}.html"
        to = f"{_to}.html"
        fr_path = Path(app.outdir) / fr
        check_existing_redirect(fr_path)
        if to.startswith("http"):
            to_path = to
        else:
            to_path = Path(app.outdir) / to
            assert to_path.is_file(), to_path
        # recreate overview folder (only for redirects now)
        os.makedirs(Path(app.outdir) / "overview", exist_ok=True)
        # recreate gallery folders that no longer exist
        defunct_gallery_folders = (
            "misc",
            "discussions",
            "source-modeling",
            "sample-datasets",
            "connectivity",
        )
        parts = fr_path.relative_to(Path(app.outdir)).parts
        if (
            len(parts) > 1  # whats_new violates this
            and parts[1] in defunct_gallery_folders
            and not fr_path.parent.exists()
        ):
            os.makedirs(fr_path.parent, exist_ok=True)
        # write the redirect
        with open(fr_path, "w") as fid:
            fid.write(REDIRECT_TEMPLATE.format(to=to_path))
    sphinx_logger.info(f"Added {len(custom_redirects):3d} HTML custom redirects")


def make_version(app, exception):
    """Make a text file with the git version."""
    if not (
        isinstance(app.builder, sphinx.builders.html.StandaloneHTMLBuilder)
        and exception is None
    ):
        return
    try:
        stdout, _ = run_subprocess(["git", "rev-parse", "HEAD"], verbose=False)
    except Exception as exc:
        sphinx_logger.warning(f"Failed to write _version.txt: {exc}")
        return
    with open(os.path.join(app.outdir, "_version.txt"), "w") as fid:
        fid.write(stdout)
    sphinx_logger.info(f'Added "{stdout.rstrip()}" > _version.txt')


def rstjinja(app, docname, source):
    """Use Jinja to process the sponsors page."""
    # Make sure we're outputting HTML
    if app.builder.format != "html":
        return
    if docname == "credits/sponsors":
        src = source[0]
        rendered = app.builder.templates.render_string(src, app.config.html_context)
        source[0] = rendered


# -- Connect our handlers to the main Sphinx app ---------------------------


def _mark_jupyterlite_parallel_safe(app):
    """Declare jupyterlite_sphinx safe for Sphinx's parallel (-j) read phase.

    The jupyterlite-sphinx version pinned by our JupyterLite/Pyodide stack
    (0.9.3) predates the ``parallel_read_safe`` metadata that newer releases
    declare, so the doc build's ``-j auto`` emits a "does not declare if it is
    safe for parallel reading" warning that ``-W`` turns into a build error.
    Newer jupyterlite-sphinx marks it read-safe; set the same flag here rather
    than bumping the pin (which would drag the whole pinned Pyodide stack
    forward and risk the browser build).
    """
    ext = app.extensions.get("jupyterlite_sphinx")
    if ext is not None and ext.parallel_read_safe is None:
        ext.parallel_read_safe = True


def setup(app):
    """Set up the Sphinx app."""
    app.connect("builder-inited", _mark_jupyterlite_parallel_safe, priority=1)
    app.connect("autodoc-process-docstring", append_attr_meth_examples)
    app.connect("autodoc-process-docstring", fix_sklearn_inherited_docstrings)
    # High prio, will happen before SG
    app.connect("builder-inited", generate_credit_rst, priority=10)
    app.connect("builder-inited", report_scraper.set_dirs, priority=20)
    app.connect("build-finished", make_gallery_redirects)
    app.connect("build-finished", make_api_redirects)
    app.connect("build-finished", make_custom_redirects)
    app.connect("build-finished", make_version)
    app.connect("source-read", rstjinja)
