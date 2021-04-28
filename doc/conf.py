# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import gc
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from distutils.version import LooseVersion

import matplotlib
import sphinx
import sphinx_gallery
from sphinx_gallery.sorting import FileNameSortKey, ExplicitOrder
from numpydoc import docscrape

import mne
from mne.tests.test_docstring_parameters import error_ignores
from mne.utils import (linkcode_resolve, # noqa, analysis:ignore
                       _assert_no_instances, sizeof_fmt)
from mne.viz import Brain  # noqa

if LooseVersion(sphinx_gallery.__version__) < LooseVersion('0.2'):
    raise ImportError('Must have at least version 0.2 of sphinx-gallery, got '
                      f'{sphinx_gallery.__version__}')

matplotlib.use('agg')

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
curdir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(curdir, '..', 'mne')))
sys.path.append(os.path.abspath(os.path.join(curdir, 'sphinxext')))


# -- Project information -----------------------------------------------------

project = 'MNE'
td = datetime.now(tz=timezone.utc)

# We need to triage which date type we use so that incremental builds work
# (Sphinx looks at variable changes and rewrites all files if some change)
copyright = (
    f'2012–{td.year}, MNE Developers. Last updated <time datetime="{td.isoformat()}" class="localized">{td.strftime("%Y-%m-%d %H:%M %Z")}</time>\n'  # noqa: E501
    '<script type="text/javascript">$(function () { $("time.localized").each(function () { var el = $(this); el.text(new Date(el.attr("datetime")).toLocaleString([], {dateStyle: "medium", timeStyle: "long"})); }); } )</script>')  # noqa: E501
if os.getenv('MNE_FULL_DATE', 'false').lower() != 'true':
    copyright = f'2012–{td.year}, MNE Developers. Last updated locally.'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = mne.__version__
# The full version, including alpha/beta/rc tags.
release = version


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '2.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.graphviz',
    'numpydoc',
    'sphinx_gallery.gen_gallery',
    'gen_commands',
    'gh_substitutions',
    'mne_substitutions',
    'gen_names',
    'sphinx_bootstrap_divs',
    'sphinxcontrib.bibtex',
    'sphinx_copybutton',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_includes']

# The suffix of source filenames.
source_suffix = '.rst'

# The main toctree document.
master_doc = 'index'

# List of documents that shouldn't be included in the build.
unused_docs = []

# List of directories, relative to source directory, that shouldn't be searched
# for source files.
exclude_trees = ['_build']

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "py:obj"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'default'

# A list of ignored prefixes for module index sorting.
modindex_common_prefix = ['mne.']


# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/devdocs', None),
    'scipy': ('https://scipy.github.io/devdocs', None),
    'matplotlib': ('https://matplotlib.org', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
    'numba': ('https://numba.pydata.org/numba-doc/latest', None),
    'joblib': ('https://joblib.readthedocs.io/en/latest', None),
    'mayavi': ('http://docs.enthought.com/mayavi/mayavi', None),
    'nibabel': ('https://nipy.org/nibabel', None),
    'nilearn': ('http://nilearn.github.io', None),
    'surfer': ('https://pysurfer.github.io/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
    'statsmodels': ('https://www.statsmodels.org/dev', None),
    'patsy': ('https://patsy.readthedocs.io/en/latest', None),
    'pyvista': ('https://docs.pyvista.org', None),
    'imageio': ('https://imageio.readthedocs.io/en/latest', None),
    'mne_realtime': ('https://mne.tools/mne-realtime', None),
    'picard': ('https://pierreablin.github.io/picard/', None),
    'qdarkstyle': ('https://qdarkstylesheet.readthedocs.io/en/latest', None),
    'eeglabio': ('https://eeglabio.readthedocs.io/en/latest', None)
}


# NumPyDoc configuration -----------------------------------------------------

# XXX This hack defines what extra methods numpydoc will document
docscrape.ClassDoc.extra_public_methods = mne.utils._doc_special_members
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    # Python
    'file-like': ':term:`file-like <python:file object>`',
    # Matplotlib
    'colormap': ':doc:`colormap <matplotlib:tutorials/colors/colormaps>`',
    'color': ':doc:`color <matplotlib:api/colors_api>`',
    'collection': ':doc:`collections <matplotlib:api/collections_api>`',
    'Axes': 'matplotlib.axes.Axes',
    'Figure': 'matplotlib.figure.Figure',
    'Axes3D': 'mpl_toolkits.mplot3d.axes3d.Axes3D',
    'ColorbarBase': 'matplotlib.colorbar.ColorbarBase',
    # Mayavi
    'mayavi.mlab.Figure': 'mayavi.core.api.Scene',
    'mlab.Figure': 'mayavi.core.api.Scene',
    # sklearn
    'LeaveOneOut': 'sklearn.model_selection.LeaveOneOut',
    # joblib
    'joblib.Parallel': 'joblib.Parallel',
    # nibabel
    'Nifti1Image': 'nibabel.nifti1.Nifti1Image',
    'Nifti2Image': 'nibabel.nifti2.Nifti2Image',
    'SpatialImage': 'nibabel.spatialimages.SpatialImage',
    # MNE
    'Label': 'mne.Label', 'Forward': 'mne.Forward', 'Evoked': 'mne.Evoked',
    'Info': 'mne.Info', 'SourceSpaces': 'mne.SourceSpaces',
    'SourceMorph': 'mne.SourceMorph',
    'Epochs': 'mne.Epochs', 'Layout': 'mne.channels.Layout',
    'EvokedArray': 'mne.EvokedArray', 'BiHemiLabel': 'mne.BiHemiLabel',
    'AverageTFR': 'mne.time_frequency.AverageTFR',
    'EpochsTFR': 'mne.time_frequency.EpochsTFR',
    'Raw': 'mne.io.Raw', 'ICA': 'mne.preprocessing.ICA',
    'Covariance': 'mne.Covariance', 'Annotations': 'mne.Annotations',
    'DigMontage': 'mne.channels.DigMontage',
    'VectorSourceEstimate': 'mne.VectorSourceEstimate',
    'VolSourceEstimate': 'mne.VolSourceEstimate',
    'VolVectorSourceEstimate': 'mne.VolVectorSourceEstimate',
    'MixedSourceEstimate': 'mne.MixedSourceEstimate',
    'MixedVectorSourceEstimate': 'mne.MixedVectorSourceEstimate',
    'SourceEstimate': 'mne.SourceEstimate', 'Projection': 'mne.Projection',
    'ConductorModel': 'mne.bem.ConductorModel',
    'Dipole': 'mne.Dipole', 'DipoleFixed': 'mne.DipoleFixed',
    'InverseOperator': 'mne.minimum_norm.InverseOperator',
    'CrossSpectralDensity': 'mne.time_frequency.CrossSpectralDensity',
    'SourceMorph': 'mne.SourceMorph',
    'Xdawn': 'mne.preprocessing.Xdawn',
    'Report': 'mne.Report', 'Forward': 'mne.Forward',
    'TimeDelayingRidge': 'mne.decoding.TimeDelayingRidge',
    'Vectorizer': 'mne.decoding.Vectorizer',
    'UnsupervisedSpatialFilter': 'mne.decoding.UnsupervisedSpatialFilter',
    'TemporalFilter': 'mne.decoding.TemporalFilter',
    'SSD': 'mne.decoding.SSD',
    'Scaler': 'mne.decoding.Scaler', 'SPoC': 'mne.decoding.SPoC',
    'PSDEstimator': 'mne.decoding.PSDEstimator',
    'LinearModel': 'mne.decoding.LinearModel',
    'FilterEstimator': 'mne.decoding.FilterEstimator',
    'EMS': 'mne.decoding.EMS', 'CSP': 'mne.decoding.CSP',
    'Beamformer': 'mne.beamformer.Beamformer',
    'Transform': 'mne.transforms.Transform',
}
numpydoc_xref_ignore = {
    # words
    'instance', 'instances', 'of', 'default', 'shape', 'or',
    'with', 'length', 'pair', 'matplotlib', 'optional', 'kwargs', 'in',
    'dtype', 'object', 'self.verbose',
    # shapes
    'n_vertices', 'n_faces', 'n_channels', 'm', 'n', 'n_events', 'n_colors',
    'n_times', 'obj', 'n_chan', 'n_epochs', 'n_picks', 'n_ch_groups',
    'n_dipoles', 'n_ica_components', 'n_pos', 'n_node_names', 'n_tapers',
    'n_signals', 'n_step', 'n_freqs', 'wsize', 'Tx', 'M', 'N', 'p', 'q',
    'n_observations', 'n_regressors', 'n_cols', 'n_frequencies', 'n_tests',
    'n_samples', 'n_permutations', 'nchan', 'n_points', 'n_features',
    'n_parts', 'n_features_new', 'n_components', 'n_labels', 'n_events_in',
    'n_splits', 'n_scores', 'n_outputs', 'n_trials', 'n_estimators', 'n_tasks',
    'nd_features', 'n_classes', 'n_targets', 'n_slices', 'n_hpi', 'n_fids',
    'n_elp', 'n_pts', 'n_tris', 'n_nodes', 'n_nonzero', 'n_events_out',
    'n_segments', 'n_orient_inv', 'n_orient_fwd', 'n_orient', 'n_dipoles_lcmv',
    'n_dipoles_fwd', 'n_picks_ref', 'n_coords', 'n_meg', 'n_good_meg',
    'n_moments',
    # Undocumented (on purpose)
    'RawKIT', 'RawEximia', 'RawEGI', 'RawEEGLAB', 'RawEDF', 'RawCTF', 'RawBTi',
    'RawBrainVision', 'RawCurry', 'RawNIRX', 'RawGDF', 'RawSNIRF', 'RawBOXY',
    'RawPersyst', 'RawNihon', 'RawNedf',
    # sklearn subclasses
    'mapping', 'to', 'any',
    # unlinkable
    'mayavi.mlab.pipeline.surface',
    'CoregFrame', 'Kit2FiffFrame', 'FiducialsFrame',
    # dipy has resolution problems, wait for them to be solved, e.g.
    # https://github.com/dipy/dipy/issues/2290
    'dipy.align.AffineMap',
    'dipy.align.DiffeomorphicMap',
}
numpydoc_validate = True
numpydoc_validation_checks = {'all'} | set(error_ignores)
numpydoc_validation_exclude = {  # set of regex
    # dict subclasses
    r'\.clear', r'\.get$', r'\.copy$', r'\.fromkeys', r'\.items', r'\.keys',
    r'\.pop', r'\.popitem', r'\.setdefault', r'\.update', r'\.values',
    # list subclasses
    r'\.append', r'\.count', r'\.extend', r'\.index', r'\.insert', r'\.remove',
    r'\.sort',
    # we currently don't document these properly (probably okay)
    r'\.__getitem__', r'\.__contains__', r'\.__hash__', r'\.__mul__',
    r'\.__sub__', r'\.__add__', r'\.__iter__', r'\.__div__', r'\.__neg__',
    # copied from sklearn
    r'mne\.utils\.deprecated',
}


# -- Sphinx-gallery configuration --------------------------------------------

class Resetter(object):
    """Simple class to make the str(obj) static for Sphinx build env hash."""

    def __init__(self):
        self.t0 = time.time()

    def __repr__(self):
        return f'<{self.__class__.__name__}>'

    def __call__(self, gallery_conf, fname):
        import matplotlib.pyplot as plt
        try:
            from pyvista import Plotter  # noqa
        except ImportError:
            Plotter = None  # noqa
        reset_warnings(gallery_conf, fname)
        # in case users have interactive mode turned on in matplotlibrc,
        # turn it off here (otherwise the build can be very slow)
        plt.ioff()
        plt.rcParams['animation.embed_limit'] = 30.
        gc.collect()
        # _assert_no_instances(Brain, 'running')  # calls gc.collect()
        # if Plotter is not None:
        #     _assert_no_instances(Plotter, 'running')
        # This will overwrite some Sphinx printing but it's useful
        # for memory timestamps
        if os.getenv('SG_STAMP_STARTS', '').lower() == 'true':
            import psutil
            process = psutil.Process(os.getpid())
            mem = sizeof_fmt(process.memory_info().rss)
            print(f'{time.time() - self.t0:6.1f} s : {mem}'.ljust(22))


examples_dirs = ['../tutorials', '../examples']
gallery_dirs = ['auto_tutorials', 'auto_examples']
os.environ['_MNE_BUILDING_DOC'] = 'true'
scrapers = ('matplotlib',)
try:
    mlab = mne.utils._import_mlab()
    # Do not pop up any mayavi windows while running the
    # examples. These are very annoying since they steal the focus.
    mlab.options.offscreen = True
    # hack to initialize the Mayavi Engine
    mlab.test_plot3d()
    mlab.close()
except Exception:
    pass
else:
    scrapers += ('mayavi',)
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import pyvista
    pyvista.OFF_SCREEN = False
except Exception:
    pass
else:
    scrapers += ('pyvista',)
if any(x in scrapers for x in ('pyvista', 'mayavi')):
    from traits.api import push_exception_handler
    push_exception_handler(reraise_exceptions=True)
    report_scraper = mne.report._ReportScraper()
    scrapers += (report_scraper,)
else:
    report_scraper = None
if 'pyvista' in scrapers:
    brain_scraper = mne.viz._brain._BrainScraper()
    scrapers = list(scrapers)
    scrapers.insert(scrapers.index('pyvista'), brain_scraper)
    scrapers = tuple(scrapers)

sphinx_gallery_conf = {
    'doc_module': ('mne',),
    'reference_url': dict(mne=None),
    'examples_dirs': examples_dirs,
    'subsection_order': ExplicitOrder(['../examples/io/',
                                       '../examples/simulation/',
                                       '../examples/preprocessing/',
                                       '../examples/visualization/',
                                       '../examples/time_frequency/',
                                       '../examples/stats/',
                                       '../examples/decoding/',
                                       '../examples/connectivity/',
                                       '../examples/forward/',
                                       '../examples/inverse/',
                                       '../examples/realtime/',
                                       '../examples/datasets/',
                                       '../tutorials/intro/',
                                       '../tutorials/io/',
                                       '../tutorials/raw/',
                                       '../tutorials/preprocessing/',
                                       '../tutorials/epochs/',
                                       '../tutorials/evoked/',
                                       '../tutorials/time-freq/',
                                       '../tutorials/forward/',
                                       '../tutorials/inverse/',
                                       '../tutorials/stats-sensor-space/',
                                       '../tutorials/stats-source-space/',
                                       '../tutorials/machine-learning/',
                                       '../tutorials/clinical/',
                                       '../tutorials/simulation/',
                                       '../tutorials/sample-datasets/',
                                       '../tutorials/misc/']),
    'gallery_dirs': gallery_dirs,
    'default_thumb_file': os.path.join('_static', 'mne_helmet.png'),
    'backreferences_dir': 'generated',
    'plot_gallery': 'True',  # Avoid annoying Unicode/bool default warning
    'thumbnail_size': (160, 112),
    'remove_config_comments': True,
    'min_reported_time': 1.,
    'abort_on_example_error': False,
    'reset_modules': ('matplotlib', Resetter()),  # called w/each script
    'image_scrapers': scrapers,
    'show_memory': not sys.platform.startswith('win'),
    'line_numbers': False,  # XXX currently (0.3.dev0) messes with style
    'within_subsection_order': FileNameSortKey,
    'capture_repr': ('_repr_html_',),
    'junit': os.path.join('..', 'test-results', 'sphinx-gallery', 'junit.xml'),
    'matplotlib_animations': True,
    'compress_images': ('images', 'thumbnails'),
    'filename_pattern': '^((?!sgskip).)*$',
}
# Files were renamed from plot_* with:
# find . -type f -name 'plot_*.py' -exec sh -c 'x="{}"; xn=`basename "${x}"`; git mv "$x" `dirname "${x}"`/${xn:5}' \;  # noqa


def append_attr_meth_examples(app, what, name, obj, options, lines):
    """Append SG examples backreferences to method and attr docstrings."""
    # NumpyDoc nicely embeds method and attribute docstrings for us, but it
    # does not respect the autodoc templates that would otherwise insert
    # the .. include:: lines, so we need to do it.
    # Eventually this could perhaps live in SG.
    if what in ('attribute', 'method'):
        size = os.path.getsize(os.path.join(
            os.path.dirname(__file__), 'generated', '%s.examples' % (name,)))
        if size > 0:
            lines += """
.. _sphx_glr_backreferences_{1}:

.. rubric:: Examples using ``{0}``:

.. minigallery:: {1}

""".format(name.split('.')[-1], name).split('\n')


# -- Other extension configuration -------------------------------------------

linkcheck_request_headers = dict(user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36')  # noqa: E501
linkcheck_ignore = [  # will be compiled to regex
    r'https://datashare.is.ed.ac.uk/handle/10283/2189\?show=full',  # noqa Max retries exceeded with url: /handle/10283/2189?show=full (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1123)')))
    'https://doi.org/10.1002/mds.870120629',  # Read timed out.
    'https://doi.org/10.1088/0031-9155/32/1/004',  # noqa Read timed out. (read timeout=15)
    'https://doi.org/10.1088/0031-9155/40/3/001',  # noqa Read timed out. (read timeout=15)
    'https://doi.org/10.1088/0031-9155/51/7/008',  # noqa Read timed out. (read timeout=15)
    'https://doi.org/10.1088/0031-9155/57/7/1937',  # noqa Read timed out. (read timeout=15)
    'https://doi.org/10.1088/0967-3334/22/4/305',  # noqa Read timed out. (read timeout=15)
    'https://doi.org/10.1088/1741-2552/aacfe4',  # noqa Read timed out. (read timeout=15)
    'https://doi.org/10.1093/sleep/18.7.557',  # noqa 403 Client Error: Forbidden for url: https://academic.oup.com/sleep/article-lookup/doi/10.1093/sleep/18.7.557
    'https://doi.org/10.1162/089976699300016719',  # noqa 403 Client Error: Forbidden for url: https://direct.mit.edu/neco/article/11/2/417-441/6242
    'https://doi.org/10.1162/jocn.1993.5.2.162',  # noqa 403 Client Error: Forbidden for url: https://direct.mit.edu/jocn/article/5/2/162-176/3095
    'https://doi.org/10.1162/neco.1995.7.6.1129',  # noqa 403 Client Error: Forbidden for url: https://direct.mit.edu/neco/article/7/6/1129-1159/5909
    'https://doi.org/10.1162/jocn_a_00405',  # noqa 403 Client Error: Forbidden for url: https://direct.mit.edu/jocn/article/25/9/1477-1492/27980
    'https://doi.org/10.1167/15.6.4',  # noqa 403 Client Error: Forbidden for url: https://jov.arvojournals.org/article.aspx?doi=10.1167/15.6.4
    'https://doi.org/10.7488/ds/1556',  # noqa Max retries exceeded with url: /handle/10283/2189 (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1122)')))
    'https://imaging.mrc-cbu.cam.ac.uk/imaging/MniTalairach',  # noqa Max retries exceeded with url: /imaging/MniTalairach (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1122)')))
    'https://www.nyu.edu/',  # noqa Max retries exceeded with url: / (Caused by SSLError(SSLError(1, '[SSL: DH_KEY_TOO_SMALL] dh key too small (_ssl.c:1122)')))
    'https://docs.python.org/3/library/.*',  # noqa ('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))
    'https://hal.archives-ouvertes.fr/hal-01848442.*',  # noqa Sometimes: 503 Server Error: Service Unavailable for url: https://hal.archives-ouvertes.fr/hal-01848442/
]
linkcheck_anchors = False  # saves a bit of time
linkcheck_timeout = 15  # some can be quite slow

# autodoc / autosummary
autosummary_generate = True
autodoc_default_options = {'inherited-members': None}

# sphinxcontrib-bibtex
bibtex_bibfiles = ['./references.bib']
bibtex_style = 'unsrt'
bibtex_footbibliography_header = ''


# -- Nitpicky ----------------------------------------------------------------

nitpicky = True
nitpick_ignore = [
    ("py:class", "None.  Remove all items from D."),
    ("py:class", "a set-like object providing a view on D's items"),
    ("py:class", "a set-like object providing a view on D's keys"),
    ("py:class", "v, remove specified key and return the corresponding value."),  # noqa: E501
    ("py:class", "None.  Update D from dict/iterable E and F."),
    ("py:class", "an object providing a view on D's values"),
    ("py:class", "a shallow copy of D"),
    ("py:class", "(k, v), remove and return some (key, value) pair as a"),
    ("py:class", "_FuncT"),  # type hint used in @verbose decorator
    ("py:class", "mne.utils._logging._FuncT"),
]
for key in ('AcqParserFIF', 'BiHemiLabel', 'Dipole', 'DipoleFixed', 'Label',
            'MixedSourceEstimate', 'MixedVectorSourceEstimate', 'Report',
            'SourceEstimate', 'SourceMorph', 'VectorSourceEstimate',
            'VolSourceEstimate', 'VolVectorSourceEstimate',
            'channels.DigMontage', 'channels.Layout',
            'decoding.CSP', 'decoding.EMS', 'decoding.FilterEstimator',
            'decoding.GeneralizingEstimator', 'decoding.LinearModel',
            'decoding.PSDEstimator', 'decoding.ReceptiveField', 'decoding.SSD',
            'decoding.SPoC', 'decoding.Scaler', 'decoding.SlidingEstimator',
            'decoding.TemporalFilter', 'decoding.TimeDelayingRidge',
            'decoding.TimeFrequency', 'decoding.UnsupervisedSpatialFilter',
            'decoding.Vectorizer',
            'preprocessing.ICA', 'preprocessing.Xdawn',
            'simulation.SourceSimulator',
            'time_frequency.CrossSpectralDensity',
            'utils.deprecated',
            'viz.ClickableImage'):
    nitpick_ignore.append(('py:obj', f'mne.{key}.__hash__'))
suppress_warnings = ['image.nonlocal_uri']  # we intentionally link outside


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'icon_links': [
        dict(name='GitHub',
             url='https://github.com/mne-tools/mne-python',
             icon='fab fa-github-square'),
        dict(name='Twitter',
             url='https://twitter.com/mne_python',
             icon='fab fa-twitter-square'),
        dict(name='Discourse',
             url='https://mne.discourse.group/',
             icon='fab fa-discourse'),
        dict(name='Discord',
             url='https://discord.gg/rKfvxTuATa',
             icon='fab fa-discord')
    ],
    'icon_links_label': 'Quick Links',  # for screen reader
    'use_edit_page_button': False,
    'navigation_with_keys': False,
    'show_toc_level': 1,
    'navbar_end': ['version-switcher', 'navbar-icon-links'],
    'footer_items': ['copyright'],
    'google_analytics_id': 'UA-37225609-1',
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
html_static_path = ['_static']
html_css_files = [
    'style.css',
]

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
html_extra_path = [
    'contributing.html',
    'documentation.html',
    'getting_started.html',
    'install_mne_python.html',
]

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    'index': ['search-field.html', 'sidebar-quicklinks.html'],
}

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False
html_copy_source = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# accommodate different logo shapes (width values in rem)
xs = '2'
sm = '2.5'
md = '3'
lg = '4.5'
xl = '5'
# variables to pass to HTML templating engine
html_context = {
    'build_dev_html': bool(int(os.environ.get('BUILD_DEV_HTML', False))),
    'versions_dropdown': {
        'dev': 'v0.24 (devel)',
        'stable': 'v0.23 (stable)',
        '0.22': 'v0.22',
        '0.21': 'v0.21',
        '0.20': 'v0.20',
        '0.19': 'v0.19',
        '0.18': 'v0.18',
        '0.17': 'v0.17',
        '0.16': 'v0.16',
        '0.15': 'v0.15',
        '0.14': 'v0.14',
        '0.13': 'v0.13',
        '0.12': 'v0.12',
        '0.11': 'v0.11',
    },
    'funders': [
        dict(img='nih.png', size='3', title='National Institutes of Health'),
        dict(img='nsf.png', size='3.5',
             title='US National Science Foundation'),
        dict(img='erc.svg', size='3.5', title='European Research Council'),
        dict(img='doe.svg', size='3', title='US Department of Energy'),
        dict(img='anr.svg', size='4.5',
             title='Agence Nationale de la Recherche'),
        dict(img='cds.png', size='2.25',
             title='Paris-Saclay Center for Data Science'),
        dict(img='google.svg', size='2.25', title='Google'),
        dict(img='amazon.svg', size='2.5', title='Amazon'),
        dict(img='czi.svg', size='2.5', title='Chan Zuckerberg Initiative'),
    ],
    'institutions': [
        dict(name='Massachusetts General Hospital',
             img='MGH.svg',
             url='https://www.massgeneral.org/',
             size=sm),
        dict(name='Athinoula A. Martinos Center for Biomedical Imaging',
             img='Martinos.png',
             url='https://martinos.org/',
             size=md),
        dict(name='Harvard Medical School',
             img='Harvard.png',
             url='https://hms.harvard.edu/',
             size=sm),
        dict(name='Massachusetts Institute of Technology',
             img='MIT.svg',
             url='https://web.mit.edu/',
             size=md),
        dict(name='New York University',
             img='NYU.png',
             url='https://www.nyu.edu/',
             size=xs),
        dict(name='Commissariat à l´énergie atomique et aux énergies alternatives',  # noqa E501
             img='CEA.png',
             url='http://www.cea.fr/',
             size=md),
        dict(name='Aalto-yliopiston perustieteiden korkeakoulu',
             img='Aalto.svg',
             url='https://sci.aalto.fi/',
             size=md),
        dict(name='Télécom ParisTech',
             img='Telecom_Paris_Tech.svg',
             url='https://www.telecom-paris.fr/',
             size=md),
        dict(name='University of Washington',
             img='Washington.png',
             url='https://www.washington.edu/',
             size=md),
        dict(name='Institut du Cerveau et de la Moelle épinière',
             img='ICM.jpg',
             url='https://icm-institute.org/',
             size=md),
        dict(name='Boston University',
             img='BU.svg',
             url='https://www.bu.edu/',
             size=lg),
        dict(name='Institut national de la santé et de la recherche médicale',
             img='Inserm.svg',
             url='https://www.inserm.fr/',
             size=xl),
        dict(name='Forschungszentrum Jülich',
             img='Julich.svg',
             url='https://www.fz-juelich.de/',
             size=xl),
        dict(name='Technische Universität Ilmenau',
             img='Ilmenau.gif',
             url='https://www.tu-ilmenau.de/',
             size=xl),
        dict(name='Berkeley Institute for Data Science',
             img='BIDS.png',
             url='https://bids.berkeley.edu/',
             size=lg),
        dict(name='Institut national de recherche en informatique et en automatique',  # noqa E501
             img='inria.png',
             url='https://www.inria.fr/',
             size=xl),
        dict(name='Aarhus Universitet',
             img='Aarhus.png',
             url='https://www.au.dk/',
             size=xl),
        dict(name='Karl-Franzens-Universität Graz',
             img='Graz.jpg',
             url='https://www.uni-graz.at/',
             size=md),
    ],
    # \u00AD is an optional hyphen (not rendered unless needed)
    'carousel': [
        dict(title='Source Estimation',
             text='Distributed, sparse, mixed-norm, beam\u00ADformers, dipole fitting, and more.',  # noqa E501
             url='auto_tutorials/inverse/30_mne_dspm_loreta.html',
             img='sphx_glr_30_mne_dspm_loreta_008.gif',
             alt='dSPM'),
        dict(title='Machine Learning',
             text='Advanced decoding models including time general\u00ADiza\u00ADtion.',  # noqa E501
             url='auto_tutorials/machine-learning/50_decoding.html',
             img='sphx_glr_50_decoding_006.png',
             alt='Decoding'),
        dict(title='Encoding Models',
             text='Receptive field estima\u00ADtion with optional smooth\u00ADness priors.',  # noqa E501
             url='auto_tutorials/machine-learning/30_strf.html',
             img='sphx_glr_30_strf_001.png',
             alt='STRF'),
        dict(title='Statistics',
             text='Parametric and non-parametric, permutation tests and clustering.',  # noqa E501
             url='auto_tutorials/stats-source-space/20_cluster_1samp_spatiotemporal.html',  # noqa E501
             img='sphx_glr_20_cluster_1samp_spatiotemporal_001.png',
             alt='Clusters'),
        dict(title='Connectivity',
             text='All-to-all spectral and effective connec\u00ADtivity measures.',  # noqa E501
             url='auto_examples/connectivity/mne_inverse_label_connectivity.html',  # noqa E501
             img='sphx_glr_mne_inverse_label_connectivity_001.png',
             alt='Connectivity'),
        dict(title='Data Visualization',
             text='Explore your data from multiple perspectives.',
             url='auto_tutorials/evoked/20_visualize_evoked.py',
             img='sphx_glr_20_visualize_evoked_007.png',
             alt='Visualization'),
    ]
}

# Output file base name for HTML help builder.
htmlhelp_basename = 'mne-doc'


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
latex_toplevel_sectioning = 'part'


# -- Warnings management -----------------------------------------------------

def reset_warnings(gallery_conf, fname):
    """Ensure we are future compatible and ignore silly warnings."""
    # In principle, our examples should produce no warnings.
    # Here we cause warnings to become errors, with a few exceptions.
    # This list should be considered alongside
    # setup.cfg -> [tool:pytest] -> filterwarnings

    # remove tweaks from other module imports or example runs
    warnings.resetwarnings()
    # restrict
    warnings.filterwarnings('error')
    # allow these, but show them
    warnings.filterwarnings('always', '.*non-standard config type: "foo".*')
    warnings.filterwarnings('always', '.*config type: "MNEE_USE_CUUDAA".*')
    warnings.filterwarnings('always', '.*cannot make axes width small.*')
    warnings.filterwarnings('always', '.*Axes that are not compatible.*')
    warnings.filterwarnings('always', '.*FastICA did not converge.*')
    # ECoG BIDS spec violations:
    warnings.filterwarnings('always', '.*Fiducial point nasion not found.*')
    warnings.filterwarnings('always', '.*DigMontage is only a subset of.*')
    warnings.filterwarnings(  # xhemi morph (should probably update sample)
        'always', '.*does not exist, creating it and saving it.*')
    warnings.filterwarnings('default', module='sphinx')  # internal warnings
    warnings.filterwarnings(
        'always', '.*converting a masked element to nan.*')  # matplotlib?
    # allow these warnings, but don't show them
    warnings.filterwarnings(
        'ignore', '.*OpenSSL\\.rand is deprecated.*')
    warnings.filterwarnings('ignore', '.*is currently using agg.*')
    warnings.filterwarnings(  # SciPy-related warning (maybe 1.2.0 will fix it)
        'ignore', '.*the matrix subclass is not the recommended.*')
    warnings.filterwarnings(  # some joblib warning
        'ignore', '.*semaphore_tracker: process died unexpectedly.*')
    warnings.filterwarnings(  # needed until SciPy 1.2.0 is released
        'ignore', '.*will be interpreted as an array index.*', module='scipy')
    for key in ('HasTraits', r'numpy\.testing', 'importlib', r'np\.loads',
                'Using or importing the ABCs from',  # internal modules on 3.7
                r"it will be an error for 'np\.bool_'",  # ndimage
                "DocumenterBridge requires a state object",  # sphinx dev
                "'U' mode is deprecated",  # sphinx io
                r"joblib is deprecated in 0\.21",  # nilearn
                'The usage of `cmp` is deprecated and will',  # sklearn/pytest
                'scipy.* is deprecated and will be removed in',  # dipy
                r'Converting `np\.character` to a dtype is deprecated',  # vtk
                r'sphinx\.util\.smartypants is deprecated',
                'is a deprecated alias for the builtin',  # NumPy
                'the old name will be removed',  # Jinja, via sphinx
                ):
        warnings.filterwarnings(  # deal with other modules having bad imports
            'ignore', message=".*%s.*" % key, category=DeprecationWarning)
    warnings.filterwarnings(  # deal with bootstrap-theme bug
        'ignore', message=".*modify script_files in the theme.*",
        category=Warning)
    warnings.filterwarnings(  # nilearn
        'ignore', message=r'sklearn\.externals\.joblib is deprecated.*',
        category=FutureWarning)
    warnings.filterwarnings(  # nilearn
        'ignore', message=r'The sklearn.* module is.*', category=FutureWarning)
    warnings.filterwarnings(  # nilearn
        'ignore', message=r'Fetchers from the nilea.*', category=FutureWarning)
    warnings.filterwarnings(  # deal with other modules having bad imports
        'ignore', message=".*ufunc size changed.*", category=RuntimeWarning)
    warnings.filterwarnings(  # realtime
        'ignore', message=".*unclosed file.*", category=ResourceWarning)
    warnings.filterwarnings('ignore', message='Exception ignored in.*')
    # allow this ImportWarning, but don't show it
    warnings.filterwarnings(
        'ignore', message="can't resolve package from", category=ImportWarning)
    warnings.filterwarnings(
        'ignore', message='.*mne-realtime.*', category=DeprecationWarning)


reset_warnings(None, None)


# -- Fontawesome support -----------------------------------------------------

# here the "b" and "s" refer to "brand" and "solid" (determines which font file
# to look in). "fw-" prefix indicates fixed width.
icons = {
    'apple': 'b',
    'linux': 'b',
    'windows': 'b',
    'hand-paper': 's',
    'question': 's',
    'quote-left': 's',
    'rocket': 's',
    'server': 's',
    'fw-book': 's',
    'fw-code-branch': 's',
    'fw-newspaper': 's',
    'fw-question-circle': 's',
    'fw-quote-left': 's',
}

prolog = ''
for icon, cls in icons.items():
    fw = ' fa-fw' if icon.startswith('fw-') else ''
    prolog += f'''
.. |{icon}| raw:: html

    <i class="fa{cls} fa-{icon[3:] if fw else icon}{fw}"></i>
'''

# -- website redirects --------------------------------------------------------

# Static list created 2021/04/13 based on what we needed to redirect,
# since we don't need to add redirects for examples added after this date.
needed_plot_redirects = {
    # tutorials
    '10_epochs_overview.py', '10_evoked_overview.py', '10_overview.py',
    '10_preprocessing_overview.py', '10_raw_overview.py',
    '10_reading_meg_data.py', '15_handling_bad_channels.py',
    '20_event_arrays.py', '20_events_from_raw.py', '20_reading_eeg_data.py',
    '20_rejecting_bad_data.py', '20_visualize_epochs.py',
    '20_visualize_evoked.py', '30_annotate_raw.py', '30_epochs_metadata.py',
    '30_filtering_resampling.py', '30_info.py', '30_reading_fnirs_data.py',
    '35_artifact_correction_regression.py', '40_artifact_correction_ica.py',
    '40_autogenerate_metadata.py', '40_sensor_locations.py',
    '40_visualize_raw.py', '45_projectors_background.py',
    '50_artifact_correction_ssp.py', '50_configure_mne.py',
    '50_epochs_to_data_frame.py', '55_setting_eeg_reference.py',
    '59_head_positions.py', '60_make_fixed_length_epochs.py',
    '60_maxwell_filtering_sss.py', '70_fnirs_processing.py',
    # examples
    '3d_to_2d.py', 'brainstorm_data.py', 'channel_epochs_image.py',
    'cluster_stats_evoked.py', 'compute_csd.py',
    'compute_mne_inverse_epochs_in_label.py',
    'compute_mne_inverse_raw_in_label.py', 'compute_mne_inverse_volume.py',
    'compute_source_psd_epochs.py', 'covariance_whitening_dspm.py',
    'custom_inverse_solver.py', 'cwt_sensor_connectivity.py',
    'decoding_csp_eeg.py', 'decoding_csp_timefreq.py',
    'decoding_spatio_temporal_source.py', 'decoding_spoc_CMC.py',
    'decoding_time_generalization_conditions.py',
    'decoding_unsupervised_spatial_filter.py', 'decoding_xdawn_eeg.py',
    'define_target_events.py', 'dics_source_power.py', 'eeg_csd.py',
    'eeg_on_scalp.py', 'eeglab_head_sphere.py', 'elekta_epochs.py',
    'ems_filtering.py', 'eog_artifact_histogram.py', 'evoked_arrowmap.py',
    'evoked_ers_source_power.py', 'evoked_topomap.py', 'evoked_whitening.py',
    'fdr_stats_evoked.py', 'find_ref_artifacts.py',
    'fnirs_artifact_removal.py', 'forward_sensitivity_maps.py',
    'gamma_map_inverse.py', 'hf_sef_data.py', 'ica_comparison.py',
    'interpolate_bad_channels.py', 'label_activation_from_stc.py',
    'label_from_stc.py', 'label_source_activations.py',
    'left_cerebellum_volume_source.py', 'limo_data.py',
    'linear_model_patterns.py', 'linear_regression_raw.py',
    'meg_sensors.py', 'mixed_norm_inverse.py',
    'mixed_source_space_connectivity.py', 'mixed_source_space_inverse.py',
    'mne_cov_power.py', 'mne_helmet.py', 'mne_inverse_coherence_epochs.py',
    'mne_inverse_connectivity_spectrum.py',
    'mne_inverse_envelope_correlation.py',
    'mne_inverse_envelope_correlation_volume.py',
    'mne_inverse_label_connectivity.py', 'mne_inverse_psi_visual.py',
    'morph_surface_stc.py', 'morph_volume_stc.py', 'movement_compensation.py',
    'movement_detection.py', 'multidict_reweighted_tfmxne.py',
    'muscle_detection.py', 'opm_data.py', 'otp.py', 'parcellation.py',
    'psf_ctf_label_leakage.py', 'psf_ctf_vertices.py',
    'psf_ctf_vertices_lcmv.py', 'publication_figure.py', 'rap_music.py',
    'read_inverse.py', 'read_neo_format.py', 'read_noise_covariance_matrix.py',
    'read_stc.py', 'receptive_field_mtrf.py', 'resolution_metrics.py',
    'resolution_metrics_eegmeg.py', 'roi_erpimage_by_rt.py',
    'sensor_connectivity.py', 'sensor_noise_level.py',
    'sensor_permutation_test.py', 'sensor_regression.py',
    'shift_evoked.py', 'simulate_evoked_data.py', 'simulate_raw_data.py',
    'simulated_raw_data_using_subject_anatomy.py', 'snr_estimate.py',
    'source_label_time_frequency.py', 'source_power_spectrum.py',
    'source_power_spectrum_opm.py', 'source_simulator.py',
    'source_space_morphing.py', 'source_space_snr.py',
    'source_space_time_frequency.py', 'ssd_spatial_filters.py',
    'ssp_projs_sensitivity_map.py', 'temporal_whitening.py',
    'time_frequency_erds.py', 'time_frequency_global_field_power.py',
    'time_frequency_mixed_norm_inverse.py', 'time_frequency_simulated.py',
    'topo_compare_conditions.py', 'topo_customized.py',
    'vector_mne_solution.py', 'virtual_evoked.py', 'xdawn_denoising.py',
    'xhemi.py',
}
tu = 'auto_tutorials'
di = 'discussions'
sm = 'source-modeling'
fw = 'forward'
nv = 'inverse'
sn = 'stats-sensor-space'
sr = 'stats-source-space'
sd = 'sample-datasets'
ml = 'machine-learning'
tf = 'time-freq'
si = 'simulation'
custom_redirects = {
    # Custom redirects (one HTML path to another, relative to outdir)
    # can be added here as fr->to key->value mappings
    f'{tu}/evoked/plot_eeg_erp.html': f'{tu}/evoked/30_eeg_erp.html',
    f'{tu}/evoked/plot_whitened.html': f'{tu}/evoked/40_whitened.html',
    f'{tu}/misc/plot_modifying_data_inplace.html': f'{tu}/intro/15_inplace.html',  # noqa E501
    f'{tu}/misc/plot_report.html': f'{tu}/intro/70_report.html',
    f'{tu}/misc/plot_seeg.html': f'{tu}/clinical/20_seeg.html',
    f'{tu}/misc/plot_ecog.html': f'{tu}/clinical/30_ecog.html',
    f'{tu}/{ml}/plot_receptive_field.html': f'{tu}/{ml}/30_strf.html',
    f'{tu}/{ml}/plot_sensors_decoding.html': f'{tu}/{ml}/50_decoding.html',
    f'{tu}/{sm}/plot_background_freesurfer.html': f'{tu}/{fw}/10_background_freesurfer.html',  # noqa E501
    f'{tu}/{sm}/plot_source_alignment.html': f'{tu}/{fw}/20_source_alignment.html',  # noqa E501
    f'{tu}/{sm}/plot_forward.html': f'{tu}/{fw}/30_forward.html',
    f'{tu}/{sm}/plot_eeg_no_mri.html': f'{tu}/{fw}/35_eeg_no_mri.html',
    f'{tu}/{sm}/plot_background_freesurfer_mne.html': f'{tu}/{fw}/50_background_freesurfer_mne.html',  # noqa E501
    f'{tu}/{sm}/plot_fix_bem_in_blender.html': f'{tu}/{fw}/80_fix_bem_in_blender.html',  # noqa E501
    f'{tu}/{sm}/plot_compute_covariance.html': f'{tu}/{fw}/90_compute_covariance.html',  # noqa E501
    f'{tu}/{sm}/plot_object_source_estimate.html': f'{tu}/{nv}/10_stc_class.html',  # noqa E501
    f'{tu}/{sm}/plot_dipole_fit.html': f'{tu}/{nv}/20_dipole_fit.html',
    f'{tu}/{sm}/plot_mne_dspm_source_localization.html': f'{tu}/{nv}/30_mne_dspm_loreta.html',  # noqa E501
    f'{tu}/{sm}/plot_dipole_orientations.html': f'{tu}/{nv}/35_dipole_orientations.html',  # noqa E501
    f'{tu}/{sm}/plot_mne_solutions.html': f'{tu}/{nv}/40_mne_fixed_free.html',
    f'{tu}/{sm}/plot_beamformer_lcmv.html': f'{tu}/{nv}/50_beamformer_lcmv.html',  # noqa E501
    f'{tu}/{sm}/plot_visualize_stc.html': f'{tu}/{nv}/60_visualize_stc.html',
    f'{tu}/{sm}/plot_eeg_mri_coords.html': f'{tu}/{nv}/70_eeg_mri_coords.html',
    f'{tu}/{sd}/plot_brainstorm_phantom_elekta.html': f'{tu}/{nv}/80_brainstorm_phantom_elekta.html',  # noqa E501
    f'{tu}/{sd}/plot_brainstorm_phantom_ctf.html': f'{tu}/{nv}/85_brainstorm_phantom_ctf.html',  # noqa E501
    f'{tu}/{sd}/plot_phantom_4DBTi.html': f'{tu}/{nv}/90_phantom_4DBTi.html',
    f'{tu}/{sd}/plot_brainstorm_auditory.html': f'{tu}/io/60_ctf_bst_auditory.html',  # noqa E501
    f'{tu}/{sd}/plot_sleep.html': f'{tu}/clinical/60_sleep.html',
    f'{tu}/{di}/plot_background_filtering.html': f'{tu}/preprocessing/25_background_filtering.html',  # noqa E501
    f'{tu}/{di}/plot_background_statistics.html': f'{tu}/{sn}/10_background_stats.html',  # noqa E501
    f'{tu}/{sn}/plot_stats_cluster_erp.html': f'{tu}/{sn}/20_erp_stats.html',
    f'{tu}/{sn}/plot_stats_cluster_1samp_test_time_frequency.html': f'{tu}/{sn}/40_cluster_1samp_time_freq.html',  # noqa E501
    f'{tu}/{sn}/plot_stats_cluster_time_frequency.html': f'{tu}/{sn}/50_cluster_between_time_freq.html',  # noqa E501
    f'{tu}/{sn}/plot_stats_spatio_temporal_cluster_sensors.html': f'{tu}/{sn}/75_cluster_ftest_spatiotemporal.html',  # noqa E501
    f'{tu}/{sr}/plot_stats_cluster_spatio_temporal.html': f'{tu}/{sr}/20_cluster_1samp_spatiotemporal.html',  # noqa E501
    f'{tu}/{sr}/plot_stats_cluster_spatio_temporal_2samp.html': f'{tu}/{sr}/30_cluster_ftest_spatiotemporal.html',  # noqa E501
    f'{tu}/{sr}/plot_stats_cluster_spatio_temporal_repeated_measures_anova.html': f'{tu}/{sr}/60_cluster_rmANOVA_spatiotemporal.html',  # noqa E501
    f'{tu}/{sr}/plot_stats_cluster_time_frequency_repeated_measures_anova.html': f'{tu}/{sr}/70_cluster_rmANOVA_time_freq.html',  # noqa E501
    f'{tu}/{tf}/plot_sensors_time_frequency.html': f'{tu}/{tf}/20_sensors_time_frequency.html',  # noqa E501
    f'{tu}/{tf}/plot_ssvep.html': f'{tu}/{tf}/50_ssvep.html',
    f'{tu}/{si}/plot_creating_data_structures.html': f'{tu}/{si}/10_array_objs.html',  # noqa E501
    f'{tu}/{si}/plot_point_spread.html': f'{tu}/{si}/70_point_spread.html',
    f'{tu}/{si}/plot_dics.html': f'{tu}/{si}/80_dics.html',
}


def make_redirects(app, exception):
    """Make HTML redirects."""
    # https://www.sphinx-doc.org/en/master/extdev/appapi.html
    # Adapted from sphinxcontrib/redirects (BSD 2-clause)
    if not isinstance(app.builder, sphinx.builders.html.StandaloneHTMLBuilder):
        return
    logger = sphinx.util.logging.getLogger('mne')
    TEMPLATE = """\
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
</html>"""  # noqa: E501
    sphinx_gallery_conf = app.config['sphinx_gallery_conf']
    for src_dir, out_dir in zip(sphinx_gallery_conf['examples_dirs'],
                                sphinx_gallery_conf['gallery_dirs']):
        root = os.path.abspath(os.path.join(app.srcdir, src_dir))
        fnames = [os.path.join(os.path.relpath(dirpath, root), fname)
                  for dirpath, _, fnames in os.walk(root)
                  for fname in fnames
                  if fname in needed_plot_redirects]
        # plot_ redirects
        for fname in fnames:
            dirname = os.path.join(app.outdir, out_dir, os.path.dirname(fname))
            to_fname = os.path.splitext(os.path.basename(fname))[0] + '.html'
            fr_fname = f'plot_{to_fname}'
            to_path = os.path.join(dirname, to_fname)
            fr_path = os.path.join(dirname, fr_fname)
            assert os.path.isfile(to_path), (fname, to_path)
            with open(fr_path, 'w') as fid:
                fid.write(TEMPLATE.format(to=to_fname))
        logger.info(
            f'Added {len(fnames):3d} HTML plot_* redirects for {out_dir}')
    # custom redirects
    for fr, to in custom_redirects.items():
        to_path = os.path.join(app.outdir, to)
        assert os.path.isfile(to_path), to
        assert to_path.endswith('html'), to_path
        fr_path = os.path.join(app.outdir, fr)
        assert fr_path.endswith('html'), fr_path
        # allow overwrite if existing file is just a redirect
        if os.path.isfile(fr_path):
            with open(fr_path, 'r') as fid:
                for _ in range(8):
                    next(fid)
                line = fid.readline()
                assert 'Page Redirection' in line, line
        # handle folders that no longer exist
        if fr_path.split(os.path.sep)[-2] in (
                'misc', 'discussions', 'source-modeling', 'sample-datasets'):
            os.makedirs(os.path.dirname(fr_path), exist_ok=True)
        # handle links to sibling folders
        path_parts = to.split(os.path.sep)
        path_parts = ['..'] + path_parts[(path_parts.index(tu) + 1):]
        with open(fr_path, 'w') as fid:
            fid.write(TEMPLATE.format(to=os.path.join(*path_parts)))
    logger.info(
        f'Added {len(custom_redirects):3d} HTML custom redirects')


# -- Connect our handlers to the main Sphinx app ---------------------------

def setup(app):
    """Set up the Sphinx app."""
    app.connect('autodoc-process-docstring', append_attr_meth_examples)
    if report_scraper is not None:
        report_scraper.app = app
        app.config.rst_prolog = prolog
        app.connect('builder-inited', report_scraper.copyfiles)
    app.connect('build-finished', make_redirects)
