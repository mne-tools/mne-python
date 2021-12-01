"""Generate self-contained HTML reports from MNE objects."""

# Authors: Alex Gramfort <alexandre.gramfort@inria.fr>
#          Mainak Jas <mainak@neuro.hut.fi>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD-3-Clause

import io
import dataclasses
from dataclasses import dataclass
from typing import Tuple
from collections.abc import Sequence
import base64
from io import BytesIO, StringIO
import contextlib
import os
import os.path as op
from pathlib import Path
import fnmatch
import re
from shutil import copyfile
import time
import warnings
import webbrowser
import html as stdlib_html  # avoid namespace confusion!

import numpy as np

from .. import __version__ as MNE_VERSION
from ..fixes import _compare_version
from .. import (read_evokeds, read_events, read_cov,
                read_source_estimate, read_trans, sys_info,
                Evoked, SourceEstimate, Covariance, Info, Transform)
from ..channels import _get_ch_type
from ..defaults import _handle_default
from ..io import read_raw, read_info, BaseRaw
from ..io._read_raw import supported as extension_reader_map
from ..io.pick import _DATA_CH_TYPES_SPLIT
from ..proj import read_proj
from .._freesurfer import _reorient_image, _mri_orientation
from ..utils import (logger, verbose, get_subjects_dir, warn, _ensure_int,
                     fill_doc, _check_option, _validate_type, _safe_input,
                     _path_like, use_log_level, deprecated, _check_fname,
                     _check_ch_locs)
from ..viz import (plot_events, plot_alignment, plot_cov, plot_projs_topomap,
                   plot_compare_evokeds, set_3d_view, get_3d_backend)
from ..viz.misc import _plot_mri_contours, _get_bem_plotting_surfaces
from ..viz.utils import _ndarray_to_fig, tight_layout
from ..forward import read_forward_solution, Forward
from ..epochs import read_epochs, BaseEpochs
from ..preprocessing.ica import read_ica
from .. import dig_mri_distances
from ..minimum_norm import read_inverse_operator, InverseOperator
from ..parallel import parallel_func, check_n_jobs

from ..externals.tempita import Template
from ..externals.h5io import read_hdf5, write_hdf5

_BEM_VIEWS = ('axial', 'sagittal', 'coronal')


# For raw files, we want to support different suffixes + extensions for all
# supported file formats
SUPPORTED_READ_RAW_EXTENSIONS = tuple(extension_reader_map.keys())
RAW_EXTENSIONS = []
for ext in SUPPORTED_READ_RAW_EXTENSIONS:
    RAW_EXTENSIONS.append(f'raw{ext}')
    if ext not in ('.bdf', '.edf', '.set', '.vhdr'):  # EEG-only formats
        RAW_EXTENSIONS.append(f'meg{ext}')
    RAW_EXTENSIONS.append(f'eeg{ext}')
    RAW_EXTENSIONS.append(f'ieeg{ext}')
    RAW_EXTENSIONS.append(f'nirs{ext}')

# Processed data will always be in (gzipped) FIFF format
VALID_EXTENSIONS = ('sss.fif', 'sss.fif.gz',
                    'eve.fif', 'eve.fif.gz',
                    'cov.fif', 'cov.fif.gz',
                    'proj.fif', 'prof.fif.gz',
                    'trans.fif', 'trans.fif.gz',
                    'fwd.fif', 'fwd.fif.gz',
                    'epo.fif', 'epo.fif.gz',
                    'inv.fif', 'inv.fif.gz',
                    'ave.fif', 'ave.fif.gz',
                    'T1.mgz') + tuple(RAW_EXTENSIONS)
del RAW_EXTENSIONS

CONTENT_ORDER = (
    'raw',
    'events',
    'epochs',
    'ssp-projectors',
    'evoked',
    'covariance',
    'coregistration',
    'bem',
    'forward-solution',
    'inverse-operator',
    'source-estimate'
)

html_include_dir = Path(__file__).parent / 'js_and_css'
template_dir = Path(__file__).parent / 'templates'
JAVASCRIPT = (html_include_dir / 'report.js').read_text(encoding='utf-8')
CSS = (html_include_dir / 'report.sass').read_text(encoding='utf-8')


def _get_ch_types(inst):
    return [ch_type for ch_type in _DATA_CH_TYPES_SPLIT if ch_type in inst]


###############################################################################
# HTML generation


def _html_header_element(*, lang, include, js, css, title, tags, mne_logo_img):
    template_path = template_dir / 'header.html'

    if title is not None:
        title = stdlib_html.escape(title)
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(lang=lang, include=include, js=js, css=css, title=title,
                     tags=tags, mne_logo_img=mne_logo_img)
    return t


def _html_footer_element(*, mne_version, date):
    template_path = template_dir / 'footer.html'
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(mne_version=mne_version, date=date)
    return t


def _html_toc_element(*, content_elements):
    template_path = template_dir / 'toc.html'
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(content_elements=content_elements)
    return t


def _html_raw_element(*, id, repr, psd, butterfly, ssp_projs, title, tags):
    template_path = template_dir / 'raw.html'

    title = stdlib_html.escape(title)
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(id=id, repr=repr, psd=psd, butterfly=butterfly,
                     ssp_projs=ssp_projs, tags=tags, title=title)
    return t


def _html_epochs_element(*, id, repr, erp_imgs, drop_log, psd, ssp_projs,
                         title, tags):
    template_path = template_dir / 'epochs.html'

    title = stdlib_html.escape(title)
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(
        id=id, repr=repr, erp_imgs=erp_imgs, drop_log=drop_log, psd=psd,
        ssp_projs=ssp_projs, tags=tags, title=title
    )
    return t


def _html_evoked_element(*, id, joint, slider, gfp, whitened, ssp_projs, title,
                         tags):
    template_path = template_dir / 'evoked.html'

    title = stdlib_html.escape(title)
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(id=id, joint=joint, slider=slider, gfp=gfp,
                     whitened=whitened, ssp_projs=ssp_projs, tags=tags,
                     title=title)
    return t


def _html_cov_element(*, id, matrix, svd, title, tags):
    template_path = template_dir / 'cov.html'

    title = stdlib_html.escape(title)
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(id=id, matrix=matrix, svd=svd, tags=tags, title=title)
    return t


def _html_forward_sol_element(*, id, repr, sensitivity_maps, title, tags):
    template_path = template_dir / 'forward.html'

    title = stdlib_html.escape(title)
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(id=id, repr=repr, sensitivity_maps=sensitivity_maps,
                     tags=tags, title=title)
    return t


def _html_inverse_operator_element(*, id, repr, source_space, title, tags):
    template_path = template_dir / 'inverse.html'

    title = stdlib_html.escape(title)
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(id=id, repr=repr, source_space=source_space, tags=tags,
                     title=title)
    return t


def _html_ica_element(*, id, repr, overlay, ecg, eog, ecg_scores, eog_scores,
                      properties, topographies, title, tags):
    template_path = template_dir / 'ica.html'

    title = stdlib_html.escape(title)
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(id=id, repr=repr, overlay=overlay, ecg=ecg, eog=eog,
                     ecg_scores=ecg_scores, eog_scores=eog_scores,
                     properties=properties, topographies=topographies,
                     tags=tags, title=title)
    return t


def _html_slider_element(*, id, images, captions, start_idx, image_format,
                         title, tags, klass=''):
    template_path = template_dir / 'slider.html'

    title = stdlib_html.escape(title)
    captions_ = []
    for caption in captions:
        if caption is None:
            caption = ''
        else:
            caption = stdlib_html.escape(caption)
        captions_.append(caption)
    del captions

    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(id=id, images=images, captions=captions_, tags=tags,
                     title=title, start_idx=start_idx,
                     image_format=image_format, klass=klass)
    return t


def _html_image_element(*, id, img, image_format, caption, show, div_klass,
                        img_klass, title, tags):
    template_path = template_dir / 'image.html'

    title = stdlib_html.escape(title)
    if caption is not None:
        caption = stdlib_html.escape(caption)
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(id=id, img=img, caption=caption, tags=tags, title=title,
                     image_format=image_format, div_klass=div_klass,
                     img_klass=img_klass, show=show)
    return t


def _html_code_element(*, id, code, language, title, tags):
    template_path = template_dir / 'code.html'

    title = stdlib_html.escape(title)
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(id=id, code=code, language=language, title=title,
                     tags=tags)
    return t


def _html_element(*, id, div_klass, html, title, tags):
    template_path = template_dir / 'html.html'

    title = stdlib_html.escape(title)
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(id=id, div_klass=div_klass, html=html, title=title,
                     tags=tags)
    return t


@dataclass
class _ContentElement:
    name: str
    name_html_escaped: str
    dom_id: str
    tags: Tuple[str]
    html: str


def _check_tags(tags) -> Tuple[str]:
    # Must be iterable, but not a string
    if (isinstance(tags, str) or not isinstance(tags, (Sequence, np.ndarray))):
        raise TypeError(
            f'tags must be a collection of str, but got {type(tags)} '
            f'instead: {tags}'
        )
    tags = tuple(tags)

    # Check for invalid dtypes
    bad_tags = [tag for tag in tags
                if not isinstance(tag, str)]
    if bad_tags:
        raise TypeError(
            f'tags must be strings, but got the following instead: '
            f'{", ".join([str(tag) for tag in bad_tags])}'
        )

    # Check for invalid characters
    invalid_chars = (' ', '"', '\n')  # we'll probably find more :-)
    bad_tags = []
    for tag in tags:
        for invalid_char in invalid_chars:
            if invalid_char in tag:
                bad_tags.append(tag)
                break
    if bad_tags:
        raise ValueError(
            f'The following tags contained invalid characters: '
            f'{", ".join(repr(tag) for tag in bad_tags)}'
        )

    return tags


###############################################################################
# PLOTTING FUNCTIONS

def _fig_to_img(fig, *, image_format='png', auto_close=True):
    """Plot figure and create a binary image."""
    # fig can be ndarray, mpl Figure, Mayavi Figure
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    if isinstance(fig, np.ndarray):
        fig = _ndarray_to_fig(fig)
    elif not isinstance(fig, Figure):
        from ..viz.backends.renderer import backend, MNE_3D_BACKEND_TESTING
        backend._check_3d_figure(figure=fig)
        if not MNE_3D_BACKEND_TESTING:
            img = backend._take_3d_screenshot(figure=fig)
        else:  # Testing mode
            img = np.zeros((2, 2, 3))

        if auto_close:
            backend._close_3d_figure(figure=fig)
        fig = _ndarray_to_fig(img)

    output = BytesIO()
    logger.debug('Saving figure %s with dpi %s'
                 % (fig.get_size_inches(), fig.get_dpi()))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action='ignore',
            message='.*Axes that are not compatible with tight_layout.*',
            category=UserWarning
        )
        fig.savefig(output, format=image_format, dpi=fig.get_dpi())

    if auto_close:
        plt.close(fig)
    output = output.getvalue()
    return (output.decode('utf-8') if image_format == 'svg' else
            base64.b64encode(output).decode('ascii'))


def _scale_mpl_figure(fig, scale):
    """Magic scaling helper.

    Keeps font size and artist sizes constant
    0.5 : current font - 4pt
    2.0 : current font + 4pt

    This is a heuristic but it seems to work for most cases.
    """
    scale = float(scale)
    fig.set_size_inches(fig.get_size_inches() * scale)
    fig.set_dpi(fig.get_dpi() * scale)
    import matplotlib as mpl
    if scale >= 1:
        sfactor = scale ** 2
    else:
        sfactor = -((1. / scale) ** 2)
    for text in fig.findobj(mpl.text.Text):
        fs = text.get_fontsize()
        new_size = fs + sfactor
        if new_size <= 0:
            raise ValueError('could not rescale matplotlib fonts, consider '
                             'increasing "scale"')
        text.set_fontsize(new_size)

    fig.canvas.draw()


def _get_bem_contour_figs_as_arrays(
    *, sl, n_jobs, mri_fname, surfaces, orientation, src, show,
    show_orientation, width
):
    """Render BEM surface contours on MRI slices.

    Returns
    -------
    list of array
        A list of NumPy arrays that represent the generated Matplotlib figures.
    """
    # Matplotlib <3.2 doesn't work nicely with process-based parallelization
    from matplotlib import __version__ as MPL_VERSION
    if _compare_version(MPL_VERSION, '>=', '3.2'):
        prefer = 'processes'
    else:
        prefer = 'threads'

    use_jobs = min(n_jobs, max(1, len(sl)))
    parallel, p_fun, _ = parallel_func(_plot_mri_contours, use_jobs,
                                       prefer=prefer)
    outs = parallel(
        p_fun(
            slices=s, mri_fname=mri_fname, surfaces=surfaces,
            orientation=orientation, src=src, show=show,
            show_orientation=show_orientation, width=width,
            slices_as_subplots=False
        )
        for s in np.array_split(sl, use_jobs)
    )
    out = list()
    for o in outs:
        out.extend(o)
    return out


def _iterate_trans_views(function, **kwargs):
    """Auxiliary function to iterate over views in trans fig."""
    from ..viz import create_3d_figure
    fig = create_3d_figure((800, 800), bgcolor=(0.5, 0.5, 0.5))
    from ..viz.backends.renderer import backend
    try:
        try:
            return _itv(function, fig, surfaces=['head-dense'], **kwargs)
        except IOError:
            return _itv(function, fig, surfaces=['head'], **kwargs)
    finally:
        backend._close_3d_figure(fig)


def _itv(function, fig, **kwargs):
    from ..viz.backends.renderer import MNE_3D_BACKEND_TESTING, backend
    from ..viz._brain.view import views_dicts
    function(fig=fig, **kwargs)

    views = (
        'frontal', 'lateral', 'medial',
        'axial', 'rostral', 'coronal'
    )

    images = []
    for view in views:
        if not MNE_3D_BACKEND_TESTING:
            set_3d_view(fig, **views_dicts['both'][view])
            backend._check_3d_figure(fig)
            im = backend._take_3d_screenshot(figure=fig)
        else:  # Testing mode
            im = np.zeros((2, 2, 3))
        images.append(im)

    images = np.concatenate(
        [np.concatenate(images[:3], axis=1),
         np.concatenate(images[3:], axis=1)],
        axis=0)

    dists = dig_mri_distances(info=kwargs['info'],
                              trans=kwargs['trans'],
                              subject=kwargs['subject'],
                              subjects_dir=kwargs['subjects_dir'])

    img = _fig_to_img(images, image_format='png')

    caption = (f'Average distance from {len(dists)} digitized points to '
               f'head: {1e3 * np.mean(dists):.2f} mm')

    return img, caption


def _plot_ica_properties_as_arrays(*, ica, inst, picks, n_jobs):
    """Parallelize ICA component properties plotting, and return arrays.

    Returns
    -------
    outs : list of array
        The properties plots as NumPy arrays.
    """
    import matplotlib.pyplot as plt

    if picks is None:
        picks = list(range(ica.n_components_))

    def _plot_one_ica_property(*, ica, inst, pick):
        figs = ica.plot_properties(inst=inst, picks=pick, show=False)
        assert len(figs) == 1
        fig = figs[0]

        with io.BytesIO() as buff:
            fig.savefig(
                buff, format='png', dpi=fig.get_dpi(), pad_inches=0,
            )
            buff.seek(0)
            fig_array = plt.imread(buff, format='png')

        plt.close(fig)
        return fig_array

    use_jobs = min(n_jobs, max(1, len(picks)))
    parallel, p_fun, _ = parallel_func(
        func=_plot_one_ica_property,
        n_jobs=use_jobs
    )
    outs = parallel(
        p_fun(
            ica=ica, inst=inst, pick=pick
        ) for pick in picks
    )
    return outs


###############################################################################
# TOC FUNCTIONS

def _endswith(fname, suffixes):
    """Aux function to test if file name includes the specified suffixes."""
    if isinstance(suffixes, str):
        suffixes = [suffixes]
    for suffix in suffixes:
        for ext in SUPPORTED_READ_RAW_EXTENSIONS:
            if fname.endswith((f'-{suffix}{ext}', f'-{suffix}{ext}',
                               f'_{suffix}{ext}', f'_{suffix}{ext}')):
                return True
    return False


def open_report(fname, **params):
    """Read a saved report or, if it doesn't exist yet, create a new one.

    The returned report can be used as a context manager, in which case any
    changes to the report are saved when exiting the context block.

    Parameters
    ----------
    fname : str
        The file containing the report, stored in the HDF5 format. If the file
        does not exist yet, a new report is created that will be saved to the
        specified file.
    **params : kwargs
        When creating a new report, any named parameters other than ``fname``
        are passed to the ``__init__`` function of the `Report` object. When
        reading an existing report, the parameters are checked with the
        loaded report and an exception is raised when they don't match.

    Returns
    -------
    report : instance of Report
        The report.
    """
    fname = _check_fname(fname=fname, overwrite='read', must_exist=False)
    if op.exists(fname):
        # Check **params with the loaded report
        state = read_hdf5(fname, title='mnepython')
        for param in params.keys():
            if param not in state:
                raise ValueError('The loaded report has no attribute %s' %
                                 param)
            if params[param] != state[param]:
                raise ValueError("Attribute '%s' of loaded report does not "
                                 "match the given parameter." % param)
        report = Report()
        report.__setstate__(state)
    else:
        report = Report(**params)
    # Keep track of the filename in case the Report object is used as a context
    # manager.
    report.fname = fname
    return report


###############################################################################
# HTML scan renderer

mne_logo_path = Path(__file__).parents[1] / 'icons' / 'mne_icon-cropped.png'
mne_logo = base64.b64encode(mne_logo_path.read_bytes()).decode('ascii')


def _check_scale(scale):
    """Ensure valid scale value is passed."""
    if np.isscalar(scale) and scale <= 0:
        raise ValueError('scale must be positive, not %s' % scale)


def _check_image_format(rep, image_format):
    """Ensure fmt is valid."""
    if rep is None or image_format is not None:
        _check_option('image_format', image_format,
                      allowed_values=('png', 'svg', 'gif'))
    else:
        image_format = rep.image_format
    return image_format


@fill_doc
class Report(object):
    r"""Object for rendering HTML.

    Parameters
    ----------
    info_fname : None | str
        Name of the file containing the info dictionary.
    %(subjects_dir)s
    subject : str | None
        Subject name.
    title : str
        Title of the report.
    cov_fname : None | str
        Name of the file containing the noise covariance.
    %(baseline_report)s
        Defaults to ``None``, i.e. no baseline correction.
    image_format : 'png' | 'svg' | 'gif'
        Default image format to use (default is ``'png'``).
        ``'svg'`` uses vector graphics, so fidelity is higher but can increase
        file size and browser image rendering time as well.

        .. versionadded:: 0.15

    raw_psd : bool | dict
        If True, include PSD plots for raw files. Can be False (default) to
        omit, True to plot, or a dict to pass as ``kwargs`` to
        :meth:`mne.io.Raw.plot_psd`.

        .. versionadded:: 0.17
    projs : bool
        Whether to include topographic plots of SSP projectors, if present in
        the data. Defaults to ``False``.

        .. versionadded:: 0.21
    %(verbose)s

    Attributes
    ----------
    info_fname : None | str
        Name of the file containing the info dictionary.
    %(subjects_dir)s
    subject : str | None
        Subject name.
    title : str
        Title of the report.
    cov_fname : None | str
        Name of the file containing the noise covariance.
    %(baseline_report)s
        Defaults to ``None``, i.e. no baseline correction.
    image_format : str
        Default image format to use.

        .. versionadded:: 0.15

    raw_psd : bool | dict
        If True, include PSD plots for raw files. Can be False (default) to
        omit, True to plot, or a dict to pass as ``kwargs`` to
        :meth:`mne.io.Raw.plot_psd`.

        .. versionadded:: 0.17
    projs : bool
        Whether to include topographic plots of SSP projectors, if present in
        the data. Defaults to ``False``.

        .. versionadded:: 0.21
    %(verbose)s
    html : list of str
        Contains items of html-page.
    include : list of str
        Dictionary containing elements included in head.
    fnames : list of str
        List of file names rendered.
    sections : list of str
        List of sections.
    lang : str
        language setting for the HTML file.

    Notes
    -----
    See :ref:`tut-report` for an introduction to using ``mne.Report``.

    .. versionadded:: 0.8.0
    """

    def __init__(self, info_fname=None, subjects_dir=None,
                 subject=None, title=None, cov_fname=None, baseline=None,
                 image_format='png', raw_psd=False, projs=False, verbose=None):
        self.info_fname = str(info_fname) if info_fname is not None else None
        self.cov_fname = str(cov_fname) if cov_fname is not None else None
        self.baseline = baseline
        if subjects_dir is not None:
            subjects_dir = get_subjects_dir(subjects_dir)
        self.subjects_dir = subjects_dir
        self.subject = subject
        self.title = title
        self.image_format = _check_image_format(None, image_format)
        self.projs = projs
        self.verbose = verbose

        self._dom_id = 0
        self._content = []
        self.include = []
        self.lang = 'en-us'  # language setting for the HTML file
        if not isinstance(raw_psd, bool) and not isinstance(raw_psd, dict):
            raise TypeError('raw_psd must be bool or dict, got %s'
                            % (type(raw_psd),))
        self.raw_psd = raw_psd
        self._init_render()  # Initialize the renderer

        self.fname = None  # The name of the saved report
        self.data_path = None

    def __repr__(self):
        """Print useful info about report."""
        s = f'<Report | {len(self._content)} items'
        if self.title is not None:
            s += f' | {self.title}'
        content_element_names = [element.name for element in self._content]
        if len(content_element_names) > 4:
            first_entries = '\n'.join(content_element_names[:2])
            last_entries = '\n'.join(content_element_names[-2:])
            s += f'\n{first_entries}'
            s += '\n ...\n'
            s += last_entries
        elif len(content_element_names) > 0:
            entries = '\n'.join(content_element_names)
            s += f'\n{entries}'
        s += '\n>'
        return s

    def __len__(self):
        """Return the number of files processed by the report.

        Returns
        -------
        n_files : int
            The number of files processed.
        """
        return len(self._content)

    @staticmethod
    def _get_state_params():
        # Which attributes to store in and read from HDF5 files
        return (
            'baseline', 'cov_fname', 'include', '_content', 'image_format',
            'info_fname', '_dom_id', 'raw_psd', 'projs',
            'subjects_dir', 'subject', 'title', 'data_path', 'lang', 'verbose',
            'fname'
        )

    def _get_dom_id(self):
        """Get id of plot."""
        self._dom_id += 1
        return f'global{self._dom_id}'

    def _validate_topomap_kwargs(self, topomap_kwargs):
        _validate_type(topomap_kwargs, (dict, None), 'topomap_kwargs')
        topomap_kwargs = dict() if topomap_kwargs is None else topomap_kwargs
        return topomap_kwargs

    def _validate_input(self, items, captions, tag, comments=None):
        """Validate input."""
        if not isinstance(items, (list, tuple)):
            items = [items]
        if not isinstance(captions, (list, tuple)):
            captions = [captions]
        if not isinstance(comments, (list, tuple)) and comments is not None:
            comments = [comments]
        if comments is not None and len(comments) != len(items):
            raise ValueError(
                f'Number of "comments" and report items must be equal, '
                f'or comments should be None; got '
                f'{len(comments)} and {len(items)}'
            )
        elif captions is not None and len(captions) != len(items):
            raise ValueError(
                f'Number of "captions" and report items must be equal; '
                f'got {len(captions)} and {len(items)}'
            )
        return items, captions, comments

    @property
    def html(self):
        return [element.html for element in self._content]

    @property
    def fnames(self):
        warn(
            message='Report.fnames is deprecated',
            category=DeprecationWarning
        )
        return [element.name for element in self._content]

    @property
    def sections(self):
        warn(
            message='Report.sections is deprecated. Use Reports.tags instead',
            category=DeprecationWarning
        )
        return self.tags

    @property
    def tags(self):
        """All tags currently used in the report."""
        tags = []
        for c in self._content:
            tags.extend(c.tags)

        tags = tuple(sorted(set(tags)))
        return tags

    def add_custom_css(self, css):
        """Add custom CSS to the report.

        Parameters
        ----------
        css : str
            Style definitions to add to the report. The content of this string
            will be embedded between HTML ``<style>`` and ``</style>`` tags.

        Notes
        -----
        .. versionadded:: 0.23
        """
        style = f'\n<style type="text/css">\n{css}\n</style>'
        self.include += style

    def add_custom_js(self, js):
        """Add custom JavaScript to the report.

        Parameters
        ----------
        js : str
            JavaScript code to add to the report. The content of this string
            will be embedded between HTML ``<script>`` and ``</script>`` tags.

        Notes
        -----
        .. versionadded:: 0.23
        """
        script = f'\n<script type="text/javascript">\n{js}\n</script>'
        self.include += script

    @fill_doc
    def add_epochs(self, epochs, title, *, psd=True, projs=None,
                   tags=('epochs',), replace=False, topomap_kwargs=None):
        """Add `~mne.Epochs` to the report.

        Parameters
        ----------
        epochs : path-like | instance of Epochs
            The epochs to add to the report.
        title : str
            The title to add.
        psd : bool | None
            Whether to add PSD plots.
        %(report_projs)s
        %(report_tags)s
        %(report_replace)s
        %(topomap_kwargs)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)

        add_projs = self.projs if projs is None else projs

        htmls = self._render_epochs(
            epochs=epochs,
            add_psd=psd,
            add_projs=add_projs,
            tags=tags,
            image_format=self.image_format,
            topomap_kwargs=topomap_kwargs,
        )
        (repr_html, erp_imgs_html, drop_log_html, psd_html,
         ssp_projs_html) = htmls

        dom_id = self._get_dom_id()
        html = _html_epochs_element(
            repr=repr_html,
            erp_imgs=erp_imgs_html,
            drop_log=drop_log_html,
            psd=psd_html,
            ssp_projs=ssp_projs_html,
            tags=tags,
            title=title,
            id=dom_id,
        )
        self._add_or_replace(
            name=title,
            dom_id=dom_id,
            tags=tags,
            html=html,
            replace=replace
        )

    @fill_doc
    def add_evokeds(self, evokeds, *, titles=None, noise_cov=None, projs=None,
                    n_time_points=None, tags=('evoked',), replace=False,
                    topomap_kwargs=None, n_jobs=1):
        """Add `~mne.Evoked` objects to the report.

        Parameters
        ----------
        evokeds : path-like | instance of Evoked | list of Evoked
            The evoked data to add to the report. Multiple `~mne.Evoked`
            objects – as returned from `mne.read_evokeds` – can be passed as
            a list.
        titles : str | list of str | None
            The titles corresponding to the evoked data. If ``None``, the
            content of ``evoked.comment`` from each evoked will be used as
            title.
        noise_cov : path-like | instance of Covariance | None
            A noise covariance matrix. If provided, will be used to whiten
            the ``evokeds``. If ``None``, will fall back to the ``cov_fname``
            provided upon report creation.
        %(report_projs)s
        n_time_points : int | None
            The number of equidistant time points to render. If ``None``,
            will render each `~mne.Evoked` at 21 time points, unless the data
            contains fewer time points, in which case all will be rendered.
        %(report_tags)s
        %(report_replace)s
        %(topomap_kwargs)s
        %(n_jobs)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        if isinstance(evokeds, Evoked):
            evokeds = [evokeds]
        elif isinstance(evokeds, list):
            pass
        else:
            evoked_fname = evokeds
            logger.debug(f'Evoked: Reading {evoked_fname}')
            evokeds = read_evokeds(evoked_fname, verbose=False)

        if self.baseline is not None:
            evokeds = [e.copy().apply_baseline(self.baseline)
                       for e in evokeds]

        if titles is None:
            titles = [e.comment for e in evokeds]
        elif isinstance(titles, str):
            titles = [titles]

        if len(evokeds) != len(titles):
            raise ValueError(
                f'Number of evoked objects ({len(evokeds)}) must '
                f'match number of captions ({len(titles)})'
            )

        if noise_cov is None:
            noise_cov = self.cov_fname
        if noise_cov is not None and not isinstance(noise_cov, Covariance):
            noise_cov = read_cov(fname=noise_cov)
        tags = _check_tags(tags)

        add_projs = self.projs if projs is None else projs

        for evoked, title in zip(evokeds, titles):
            evoked_htmls = self._render_evoked(
                evoked=evoked,
                noise_cov=noise_cov,
                image_format=self.image_format,
                add_projs=add_projs,
                n_time_points=n_time_points,
                tags=tags,
                topomap_kwargs=topomap_kwargs,
                n_jobs=n_jobs
            )

            (joint_html, slider_html, gfp_html, whitened_html,
             ssp_projs_html) = evoked_htmls

            dom_id = self._get_dom_id()
            html = _html_evoked_element(
                id=dom_id,
                joint=joint_html,
                slider=slider_html,
                gfp=gfp_html,
                whitened=whitened_html,
                ssp_projs=ssp_projs_html,
                title=title,
                tags=tags
            )
            self._add_or_replace(
                dom_id=dom_id,
                name=title,
                tags=tags,
                html=html,
                replace=replace
            )

    @fill_doc
    def add_raw(self, raw, title, *, psd=None, projs=None, butterfly=True,
                tags=('raw',), replace=False, topomap_kwargs=None):
        """Add `~mne.io.Raw` objects to the report.

        Parameters
        ----------
        raw : path-like | instance of Raw
            The data to add to the report.
        title : str
            The title corresponding to the ``raw`` object.
        psd : bool | None
            Whether to add PSD plots. Overrides the ``raw_psd`` parameter
            passed when initializing the `~mne.Report`. If ``None``, use
            ``raw_psd`` from `~mne.Report` creation.
        %(report_projs)s
        butterfly : bool
            Whether to add a butterfly plot of the (decimated) data. Can be
            useful to spot segments marked as "bad" and problematic channels.
        %(report_tags)s
        %(report_replace)s
        %(topomap_kwargs)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)

        if psd is None:
            add_psd = dict() if self.raw_psd is True else self.raw_psd
        elif psd is True:
            add_psd = dict()
        else:
            add_psd = False

        add_projs = self.projs if projs is None else projs

        htmls = self._render_raw(
            raw=raw,
            add_psd=add_psd,
            add_projs=add_projs,
            add_butterfly=butterfly,
            image_format=self.image_format,
            tags=tags,
            topomap_kwargs=topomap_kwargs,
        )
        repr_html, psd_img_html, butterfly_imgs_html, ssp_proj_img_html = htmls
        dom_id = self._get_dom_id()
        html = _html_raw_element(
            repr=repr_html,
            psd=psd_img_html,
            butterfly=butterfly_imgs_html,
            ssp_projs=ssp_proj_img_html,
            tags=tags,
            title=title,
            id=dom_id,
        )
        self._add_or_replace(
            dom_id=dom_id,
            name=title,
            tags=tags,
            html=html,
            replace=replace
        )

    @fill_doc
    def add_stc(self, stc, title, *, subject=None, subjects_dir=None,
                n_time_points=None, tags=('source-estimate',), replace=False,
                stc_plot_kwargs=None):
        """Add a `~mne.SourceEstimate` (STC) to the report.

        Parameters
        ----------
        stc : path-like | instance of SourceEstimate
            The `~mne.SourceEstimate` to add to the report.
        title : str
            The title to add.
        subject : str | None
            The name of the FreeSurfer subject the STC belongs to. The name is
            not stored with the STC data and therefore needs to be specified.
            If ``None``, will use the value of ``subject`` passed on report
            creation.
        subjects_dir : path-like | None
            The FreeSurfer ``SUBJECTS_DIR``.
        n_time_points : int | None
            The number of equidistant time points to render. If ``None``,
            will render ``stc`` at 51 time points, unless the data
            contains fewer time points, in which case all will be rendered.
        %(report_tags)s
        %(report_replace)s
        %(report_stc_plot_kwargs)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)

        html, dom_id = self._render_stc(
            stc=stc,
            title=title,
            tags=tags,
            image_format=self.image_format,
            subject=subject,
            subjects_dir=subjects_dir,
            n_time_points=n_time_points,
            stc_plot_kwargs=stc_plot_kwargs
        )
        self._add_or_replace(
            dom_id=dom_id,
            name=title,
            tags=tags,
            html=html,
            replace=replace
        )

    @fill_doc
    def add_forward(self, forward, title, *, subject=None, subjects_dir=None,
                    tags=('forward-solution',), replace=False):
        """Add a forward solution.

        Parameters
        ----------
        forward : instance of Forward | path-like
            The forward solution to add to the report.
        title : str
            The title corresponding to forward solution.
        subject : str | None
            The name of the FreeSurfer subject ``forward`` belongs to. If
            provided, the sensitivity maps of the forward solution will
            be visualized. If ``None``, will use the value of ``subject``
            passed on report creation. If supplied, also pass ``subjects_dir``.
        subjects_dir : path-like | None
            The FreeSurfer ``SUBJECTS_DIR``.
        %(report_tags)s
        %(report_replace)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)

        html, dom_id = self._render_forward(
            forward=forward, subject=subject, subjects_dir=subjects_dir,
            title=title, image_format=self.image_format, tags=tags
        )
        self._add_or_replace(
            dom_id=dom_id,
            name=title,
            tags=tags,
            html=html,
            replace=replace
        )

    @fill_doc
    def add_inverse_operator(self, inverse_operator, title, *, subject=None,
                             subjects_dir=None, trans=None,
                             tags=('inverse-operator',), replace=False):
        """Add an inverse operator.

        Parameters
        ----------
        inverse_operator : instance of InverseOperator | path-like
            The inverse operator to add to the report.
        title : str
            The title corresponding to the inverse operator object.
        subject : str | None
            The name of the FreeSurfer subject ``inverse_op`` belongs to. If
            provided, the source space the inverse solution is based on will
            be visualized. If ``None``, will use the value of ``subject``
            passed on report creation. If supplied, also pass ``subjects_dir``
            and ``trans``.
        subjects_dir : path-like | None
            The FreeSurfer ``SUBJECTS_DIR``.
        trans : path-like | instance of Transform | None
            The ``head -> MRI`` transformation for ``subject``.
        %(report_tags)s
        %(report_replace)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)

        if ((subject is not None and trans is None) or
                (trans is not None and subject is None)):
            raise ValueError('Please pass subject AND trans, or neither.')

        html, dom_id = self._render_inverse_operator(
            inverse_operator=inverse_operator, subject=subject,
            subjects_dir=subjects_dir, trans=trans, title=title,
            image_format=self.image_format, tags=tags
        )
        self._add_or_replace(
            dom_id=dom_id,
            name=title,
            tags=tags,
            html=html,
            replace=replace
        )

    @fill_doc
    def add_trans(self, trans, *, info, title, subject=None, subjects_dir=None,
                  tags=('coregistration',), replace=False):
        """Add a coregistration visualization to the report.

        Parameters
        ----------
        trans : path-like | instance of Transform
            The ``head -> MRI`` transformation to render.
        info : path-like | instance of Info
            The `~mne.Info` corresponding to ``trans``.
        title : str
            The title to add.
        subject : str | None
            The name of the FreeSurfer subject the ``trans```` belong to. The
            name is not stored with the ``trans`` and therefore needs to be
            specified. If ``None``, will use the value of ``subject`` passed on
            report creation.
        subjects_dir : path-like | None
            The FreeSurfer ``SUBJECTS_DIR``.
        %(report_tags)s
        %(report_replace)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)

        html, dom_id = self._render_trans(
            trans=trans,
            info=info,
            subject=subject,
            subjects_dir=subjects_dir,
            title=title,
            tags=tags
        )
        self._add_or_replace(
            dom_id=dom_id,
            name=title,
            tags=tags,
            html=html,
            replace=replace
        )

    @fill_doc
    def add_covariance(self, cov, *, info, title, tags=('covariance',),
                       replace=False):
        """Add covariance to the report.

        Parameters
        ----------
        cov : path-like | instance of Covariance
            The `~mne.Covariance` to add to the report.
        info : path-like | instance of Info
            The `~mne.Info` corresponding to ``cov``.
        title : str
            The title corresponding to the `~mne.Covariance` object.
        %(report_tags)s
        %(report_replace)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)
        htmls = self._render_cov(
            cov=cov,
            info=info,
            image_format=self.image_format,
            tags=tags
        )
        cov_matrix_html, cov_svd_html = htmls

        dom_id = self._get_dom_id()
        html = _html_cov_element(
            matrix=cov_matrix_html,
            svd=cov_svd_html,
            tags=tags,
            title=title,
            id=dom_id
        )
        self._add_or_replace(
            dom_id=dom_id,
            name=title,
            tags=tags,
            html=html,
            replace=replace
        )

    @fill_doc
    def add_events(self, events, title, *, event_id=None, sfreq, first_samp=0,
                   tags=('events',), replace=False):
        """Add events to the report.

        Parameters
        ----------
        events : path-like | array, shape (n_events, 3)
            An MNE-Python events array.
        title : str
            The title corresponding to the events.
        event_id : dict
            A dictionary mapping event names (keys) to event codes (values).
        sfreq : float
            The sampling frequency used while recording.
        first_samp : int
            The first sample point in the recording. This corresponds to
            ``raw.first_samp`` on files created with Elekta/Neuromag systems.
        %(report_tags)s
        %(report_replace)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)
        html, dom_id = self._render_events(
            events=events,
            event_id=event_id,
            sfreq=sfreq,
            first_samp=first_samp,
            title=title,
            image_format=self.image_format,
            tags=tags
        )
        self._add_or_replace(
            dom_id=dom_id,
            name=title,
            tags=tags,
            html=html,
            replace=replace
        )

    @fill_doc
    def add_projs(self, *, info, projs=None, title, topomap_kwargs=None,
                  tags=('ssp',), replace=False):
        """Render (SSP) projection vectors.

        Parameters
        ----------
        info : instance of Info | path-like
            An `~mne.Info` structure or the path of a file containing one. This
            is required to create the topographic plots.
        projs : iterable of mne.Projection | path-like | None
            The projection vectors to add to the report. Can be the path to a
            file that will be loaded via `mne.read_proj`. If ``None``, the
            projectors are taken from ``info['projs']``.
        title : str
            The title corresponding to the `~mne.Projection` object.
        %(topomap_kwargs)s
        %(report_tags)s
        %(report_replace)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)
        output = self._render_ssp_projs(
            info=info, projs=projs, title=title,
            image_format=self.image_format, tags=tags,
            topomap_kwargs=topomap_kwargs,
        )
        if output is None:
            raise ValueError(
                'The provided data does not contain digitization information. '
                'However, this is required for rendering the SSP projectors.'
            )
        else:
            html, dom_id = output

        self._add_or_replace(
            dom_id=dom_id,
            name=title,
            tags=tags,
            html=html,
            replace=replace
        )

    def _render_ica_overlay(self, *, ica, inst, image_format, tags):
        if isinstance(inst, BaseRaw):
            inst_ = inst
        else:  # Epochs
            inst_ = inst.average()

        fig = ica.plot_overlay(inst=inst_, show=False)
        del inst_
        tight_layout(fig=fig)
        img = _fig_to_img(fig, image_format=image_format)
        dom_id = self._get_dom_id()
        overlay_html = _html_image_element(
            img=img, div_klass='ica', img_klass='ica',
            title='Original and cleaned signal', caption=None, show=True,
            image_format=image_format, id=dom_id, tags=tags
        )

        return overlay_html

    def _render_ica_properties(self, *, ica, picks, inst, n_jobs, image_format,
                               tags):
        ch_type = _get_ch_type(inst=ica.info, ch_type=None)
        if not _check_ch_locs(info=ica.info, ch_type=ch_type):
            ch_type_name = _handle_default("titles")[ch_type]
            warn(f'No {ch_type_name} channel locations found, cannot '
                 f'create ICA properties plots')
            return ''

        figs = _plot_ica_properties_as_arrays(
            ica=ica, inst=inst, picks=picks, n_jobs=n_jobs
        )
        rel_explained_var = (ica.pca_explained_variance_ /
                             ica.pca_explained_variance_.sum())
        cum_explained_var = np.cumsum(rel_explained_var)
        captions = []
        for idx, rel_var, cum_var in zip(
            range(len(figs)),
            rel_explained_var[:len(figs)],
            cum_explained_var[:len(figs)]
        ):
            caption = (
                f'ICA component {idx}. '
                f'Variance explained: {round(100 * rel_var)}%'
            )
            if idx == 0:
                caption += '.'
            else:
                caption += f' ({round(100 * cum_var)}% cumulative).'

            captions.append(caption)

        title = 'ICA component properties'
        # Only render a slider if we have more than 1 component.
        if len(figs) == 1:
            img = _fig_to_img(fig=figs[0], image_format=image_format)
            dom_id = self._get_dom_id()
            properties_html = _html_image_element(
                img=img, div_klass='ica', img_klass='ica',
                title=title, caption=captions[0], show=True,
                image_format=image_format, id=dom_id, tags=tags
            )
        else:
            properties_html, _ = self._render_slider(
                figs=figs, title=title, captions=captions, start_idx=0,
                image_format=image_format, tags=tags
            )

        return properties_html

    def _render_ica_artifact_sources(self, *, ica, inst, artifact_type,
                                     image_format, tags):
        fig = ica.plot_sources(inst=inst, show=False)
        img = _fig_to_img(fig, image_format=image_format)
        dom_id = self._get_dom_id()
        html = _html_image_element(
            img=img, div_klass='ica', img_klass='ica',
            title=f'Original and cleaned {artifact_type} epochs', caption=None,
            show=True, image_format=image_format, id=dom_id, tags=tags
        )
        return html

    def _render_ica_artifact_scores(self, *, ica, scores, artifact_type,
                                    image_format, tags):
        fig = ica.plot_scores(scores=scores, title=None, show=False)
        img = _fig_to_img(fig, image_format=image_format)
        dom_id = self._get_dom_id()
        html = _html_image_element(
            img=img, div_klass='ica', img_klass='ica',
            title=f'Scores for matching {artifact_type} patterns',
            caption=None, show=True, image_format=image_format, id=dom_id,
            tags=tags
        )
        return html

    def _render_ica_components(self, *, ica, picks, image_format, tags):
        ch_type = _get_ch_type(inst=ica.info, ch_type=None)
        if not _check_ch_locs(info=ica.info, ch_type=ch_type):
            ch_type_name = _handle_default("titles")[ch_type]
            warn(f'No {ch_type_name} channel locations found, cannot '
                 f'create ICA component plots')
            return ''

        figs = ica.plot_components(
            picks=picks, title='', colorbar=True, show=False
        )
        if not isinstance(figs, list):
            figs = [figs]

        for fig in figs:
            tight_layout(fig=fig)

        title = 'ICA component topographies'
        if len(figs) == 1:
            img = _fig_to_img(fig=figs[0], image_format=image_format)
            dom_id = self._get_dom_id()
            topographies_html = _html_image_element(
                img=img, div_klass='ica', img_klass='ica',
                title=title, caption=None, show=True,
                image_format=image_format, id=dom_id, tags=tags
            )
        else:
            captions = [None] * len(figs)
            topographies_html, _ = self._render_slider(
                figs=figs, title=title, captions=captions, start_idx=0,
                image_format=image_format, tags=tags
            )

        return topographies_html

    def _render_ica(self, *, ica, inst, picks, ecg_evoked,
                    eog_evoked, ecg_scores, eog_scores, title, image_format,
                    tags, n_jobs):
        if _path_like(ica):
            ica = read_ica(ica)

        if ica.current_fit == 'unfitted':
            raise RuntimeError(
                'ICA must be fitted before it can be added to the report.'
            )

        if inst is None:
            pass  # no-op
        elif _path_like(inst):
            # We cannot know which data type to expect, so let's first try to
            # read a Raw, and if that fails, try to load Epochs
            fname = str(inst)  # could e.g. be a Path!
            raw_kwargs = dict(fname=fname, preload=False)
            if fname.endswith(('.fif', '.fif.gz')):
                raw_kwargs['allow_maxshield'] = True

            try:
                inst = read_raw(**raw_kwargs)
            except ValueError:
                try:
                    inst = read_epochs(fname)
                except ValueError:
                    raise ValueError(
                        f'The specified file, {fname}, does not seem to '
                        f'contain Raw data or Epochs'
                    )
        elif not inst.preload:
            raise RuntimeError(
                'You passed an object to Report.add_ica() via the "inst" '
                'parameter that was not preloaded. Please preload the data  '
                'via the load_data() method'
            )

        if _path_like(ecg_evoked):
            ecg_evoked = read_evokeds(fname=ecg_evoked, condition=0)

        if _path_like(eog_evoked):
            eog_evoked = read_evokeds(fname=eog_evoked, condition=0)

        # Summary table
        dom_id = self._get_dom_id()
        repr_html = _html_element(
            div_klass='ica',
            id=dom_id,
            tags=tags,
            title='Info',
            html=ica._repr_html_()
        )

        # Overlay plot
        if inst:
            overlay_html = self._render_ica_overlay(
                ica=ica, inst=inst, image_format=image_format, tags=tags
            )
        else:
            overlay_html = ''

        # ECG artifact
        if ecg_scores is not None:
            ecg_scores_html = self._render_ica_artifact_scores(
                ica=ica, scores=ecg_scores, artifact_type='ECG',
                image_format=image_format, tags=tags
            )
        else:
            ecg_scores_html = ''

        if ecg_evoked:
            ecg_html = self._render_ica_artifact_sources(
                ica=ica, inst=ecg_evoked, artifact_type='ECG',
                image_format=image_format, tags=tags
            )
        else:
            ecg_html = ''

        # EOG artifact
        if eog_scores is not None:
            eog_scores_html = self._render_ica_artifact_scores(
                ica=ica, scores=eog_scores, artifact_type='EOG',
                image_format=image_format, tags=tags
            )
        else:
            eog_scores_html = ''

        if eog_evoked:
            eog_html = self._render_ica_artifact_sources(
                ica=ica, inst=eog_evoked, artifact_type='EOG',
                image_format=image_format, tags=tags
            )
        else:
            eog_html = ''

        # Component topography plots
        topographies_html = self._render_ica_components(
            ica=ica, picks=picks, image_format=image_format, tags=tags
        )

        # Properties plots
        if inst:
            properties_html = self._render_ica_properties(
                ica=ica, picks=picks, inst=inst, n_jobs=n_jobs,
                image_format=image_format, tags=tags
            )
        else:
            properties_html = ''

        dom_id = self._get_dom_id()
        html = _html_ica_element(
            id=dom_id,
            repr=repr_html,
            overlay=overlay_html,
            ecg=ecg_html,
            eog=eog_html,
            ecg_scores=ecg_scores_html,
            eog_scores=eog_scores_html,
            properties=properties_html,
            topographies=topographies_html,
            title=title,
            tags=tags
        )
        return dom_id, html

    @fill_doc
    def add_ica(
        self, ica, title, *, inst, picks=None, ecg_evoked=None,
        eog_evoked=None, ecg_scores=None, eog_scores=None, n_jobs=1,
        tags=('ica',), replace=False
    ):
        """Add (a fitted) `~mne.preprocessing.ICA` to the report.

        Parameters
        ----------
        ica : path-like | instance of mne.preprocessing.ICA
            The fitted ICA to add.
        title : str
            The title to add.
        inst : path-like | mne.io.Raw | mne.Epochs | None
            The data to use for visualization of the effects of ICA cleaning.
            To only plot the ICA component topographies, explicitly pass
            ``None``.
        %(picks_ica)s  If ``None``, plot all components. This only affects
            the behavior of the component topography and properties plots.
        ecg_evoked, eog_evoked : path-line | mne.Evoked | None
            Evoked signal based on ECG and EOG epochs, respectively. If passed,
            will be used to visualize the effects of artifact rejection.
        ecg_scores, eog_scores : array of float | list of array of float | None
            The scores produced by :meth:`mne.preprocessing.ICA.find_bads_ecg`
            and :meth:`mne.preprocessing.ICA.find_bads_eog`, respectively.
            If passed, will be used to visualize the scoring for each ICA
            component.
        %(n_jobs)s
        %(report_tags)s
        %(report_replace)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)

        dom_id, html = self._render_ica(
            ica=ica, inst=inst, picks=picks,
            ecg_evoked=ecg_evoked, eog_evoked=eog_evoked,
            ecg_scores=ecg_scores, eog_scores=eog_scores,
            title=title, image_format=self.image_format, tags=tags,
            n_jobs=n_jobs
        )
        self._add_or_replace(
            name=title,
            dom_id=dom_id,
            tags=tags,
            html=html,
            replace=replace
        )

    def remove(self, caption=None, section=None, *, title=None, tags=None,
               remove_all=False):
        """Remove elements from the report.

        The element to remove is searched for by its title. Optionally, tags
        may be specified as well to narrow down the search to elements that
        have the supplied tags.

        Parameters
        ----------
        caption : str
            Remove content based on its caption.

            .. deprecated:: 0.24.0
               This parameter is scheduled for removal. Use ``title`` instead.
        section : str | None
            If supplied, restrict the operation to elements within the supplied
            section.

            .. deprecated:: 0.24.0
               This parameter is scheduled for removal. Use ``tags`` instead.
        title : str
            The title of the element(s) to remove.

            .. versionadded:: 0.24.0
        tags : collection of str | None
             If supplied, restrict the operation to elements with the supplied
             tags.

            .. versionadded:: 0.24.0
        remove_all : bool
            Controls the behavior if multiple elements match the search
            criteria. If ``False`` (default) only the element last added to the
            report will be removed. If ``True``, all matches will be removed.

            .. versionadded:: 0.24.0

        Returns
        -------
        removed_index : int | tuple of int | None
            The indices of the elements that were removed, or ``None`` if no
            element matched the search criteria. A tuple will always be
            returned if ``remove_all`` was set to ``True`` and at least one
            element was removed.

            .. versionchanged:: 0.24.0
               Returns tuple if ``remove_all`` is ``True``.
        """
        if caption is not None:
            warn(
                message='The "caption" parameter has been deprecated. Please '
                        'use "title" instead',
                category=DeprecationWarning
            )
            title = caption
        if section is not None:
            warn(
                message='The "section" parameter has been deprecated. Please '
                        'use "tags" instead',
                category=DeprecationWarning
            )
            tags = _clean_tags(section)

        remove_idx = []
        for idx, element in enumerate(self._content):
            if element.name == title:
                if (tags is not None and
                        not all(t in element.tags for t in tags)):
                    continue
                remove_idx.append(idx)

        if not remove_idx:
            remove_idx = None
        elif not remove_all:  # only remove last occurrence
            remove_idx = remove_idx[-1]
            del self._content[remove_idx]
        else:  # remove all occurrences
            remove_idx = tuple(remove_idx)
            self._content = [e for idx, e in enumerate(self._content)
                             if idx not in remove_idx]

        return remove_idx

    def _add_or_replace(self, *, name, dom_id, tags, html, replace=False):
        """Append HTML content report, or replace it if it already exists.

        Parameters
        ----------
        name : str
            The entry under which the content shall be listed in the table of
            contents. If it already exists, the content will be replaced if
            ``replace`` is ``True``
        dom_id : str
            A unique element ``id`` in the DOM.
        tags : tuple of str
            The tags associated with the added element.
        html : str
            The HTML.
        replace : bool
            Whether to replace existing content.
        """
        assert isinstance(html, str)  # otherwise later will break

        new_content = _ContentElement(
            name=name,
            name_html_escaped=stdlib_html.escape(name),
            dom_id=dom_id,
            tags=tags,
            html=html
        )

        existing_names = [element.name for element in self._content]
        if name in existing_names and replace:
            # Find and replace existing content, starting from the last element
            for idx, content_element in enumerate(self._content[::-1]):
                if content_element.name == name:
                    self._content[idx] = new_content
                    return
            raise RuntimeError('This should never happen')
        else:
            # Simply append new content (no replace)
            self._content.append(new_content)

    def _render_code(self, *, code, title, language, tags):
        if isinstance(code, Path):
            code = Path(code).read_text()

        dom_id = self._get_dom_id()
        html = _html_code_element(
            tags=tags,
            title=title,
            id=dom_id,
            code=code,
            language=language
        )
        return html, dom_id

    @fill_doc
    def add_code(self, code, title, *, language='python', tags=('code',),
                 replace=False):
        """Add a code snippet (e.g., an analysis script) to the report.

        Parameters
        ----------
        code : str | pathlib.Path
            The code to add to the report as a string, or the path to a file
            as a `pathlib.Path` object.

            .. note:: Paths must be passed as `pathlib.Path` object, since
                      strings will be treated as literal code.
        title : str
            The title corresponding to the code.
        language : str
            The programming language of ``code``. This will be used for syntax
            highlighting. Can be ``'auto'`` to try to auto-detect the language.
        %(report_tags)s
        %(report_replace)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)
        language = language.lower()
        html, dom_id = self._render_code(
            code=code, title=title, language=language, tags=tags
        )
        self._add_or_replace(
            dom_id=dom_id,
            name=title,
            tags=tags,
            html=html,
            replace=replace
        )

    @fill_doc
    def add_sys_info(self, title, *, tags=('mne-sysinfo',)):
        """Add a MNE-Python system information to the report.

        This is a convenience method that captures the output of
        `mne.sys_info` and adds it to the report.

        Parameters
        ----------
        title : str
            The title to assign.
        %(report_tags)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)

        with contextlib.redirect_stdout(StringIO()) as f:
            sys_info()

        info = f.getvalue()
        self.add_code(code=info, title=title, language='shell', tags=tags)

    @fill_doc
    def add_figure(self, fig, title, *, caption=None, image_format=None,
                   tags=('custom-figure',), replace=False):
        """Add figures to the report.

        Parameters
        ----------
        fig : matplotlib.figure.Figure | mlab.Figure | array | collection of matplotlib.figure.Figure | collection of mlab.Figure | collection of array
            One or more figures to add to the report. All figures must be an
            instance of :class:`matplotlib.figure.Figure`,
            :class:`mayavi.core.api.Scene`, or :class:`numpy.ndarray`. If
            multiple figures are passed, they will be added as "slides"
            that can be navigated using buttons and a slider element.
        title : str
            The title corresponding to the figure(s).
        caption : str | collection of str | None
            The caption(s) to add to the figure(s).
        %(report_image_format)s
        %(report_tags)s
        %(report_replace)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """  # noqa E501
        tags = _check_tags(tags)
        if image_format is None:
            image_format = self.image_format

        if hasattr(fig, '__len__') and not isinstance(fig, np.ndarray):
            figs = tuple(fig)
        else:
            figs = (fig,)

        for fig in figs:
            if _path_like(fig):
                raise TypeError(
                    f'It seems you passed a path to `add_figure`. However, '
                    f'only Matplotlib figures, Mayavi scenes, and NumPy '
                    f'arrays are accepted. You may want to try `add_image` '
                    f'instead. The provided path was: {fig}'
                )
        del fig

        if isinstance(caption, str):
            captions = (caption,)
        elif caption is None and len(figs) == 1:
            captions = [None]
        elif caption is None and len(figs) > 1:
            captions = [f'Figure {i+1}' for i in range(len(figs))]
        else:
            captions = tuple(caption)

        del caption

        assert figs
        if len(figs) == 1:
            img = _fig_to_img(fig=figs[0], image_format=image_format)
            dom_id = self._get_dom_id()
            html = _html_image_element(
                img=img, div_klass='custom-image', img_klass='custom-image',
                title=title, caption=captions[0], show=True,
                image_format=image_format, id=dom_id, tags=tags
            )
        else:
            html, dom_id = self._render_slider(
                figs=figs, title=title, captions=captions, start_idx=0,
                image_format=image_format, tags=tags
            )

        self._add_or_replace(
            dom_id=dom_id,
            name=title,
            tags=tags,
            html=html,
            replace=replace
        )

    @deprecated('Use :meth:`~mne.Report.add_figure` instead')
    @fill_doc
    def add_figs_to_section(self, figs, captions, section='custom',
                            scale=None, image_format=None, comments=None,
                            replace=False, auto_close=True):
        """Append custom user-defined figures.

        Parameters
        ----------
        figs : matplotlib.figure.Figure | mlab.Figure | array | list
            A figure or a list of figures to add to the report. Each figure in
            the list can be an instance of :class:`matplotlib.figure.Figure`,
            :class:`mayavi.core.api.Scene`, or :class:`numpy.ndarray`.
        captions : str | list of str
            A caption or a list of captions to the figures.
        section : str
            Name of the section to place the figures in. If it already
            exists, the figures will be appended to the end of the existing
            section.
        scale : float | None | callable
            Scale the images maintaining the aspect ratio.
            If None, no scaling is applied. If float, scale will determine
            the relative scaling (might not work for scale <= 1 depending on
            font sizes). If function, should take a figure object as input
            parameter. Defaults to None.
        %(report_image_format)s
        comments : None | str | list of str
            A string of text or a list of strings of text to be appended after
            the figure.
        replace : bool
            If ``True``, figures already present that have the same caption
            will be replaced. Defaults to ``False``.
        auto_close : bool
            If True, the plots are closed during the generation of the report.
            Defaults to True.
        """
        figs, captions, comments = self._validate_input(
            figs, captions, section, comments
        )
        image_format = _check_image_format(self, image_format)
        _check_scale(scale)

        if (
            _path_like(figs) or
            (hasattr(figs, '__iter__') and
             any(_path_like(f) for f in figs))
        ):
            raise TypeError(
                'It seems you passed a path to `add_figs_to_section`. '
                'However, only Matplotlib figures, Mayavi scenes, and NumPy '
                'arrays are accepted. You may want to try `add_image` instead.'
            )

        if hasattr(figs, '__len__') and not isinstance(figs, np.ndarray):
            figs = tuple(figs)
        else:
            figs = (figs,)

        if isinstance(captions, str):
            captions = (captions,)
        else:
            captions = tuple(captions)

        if isinstance(comments, str):
            comments = (comments,)
        elif comments is None:
            comments = (None,) * len(figs)
        else:
            comments = tuple(comments)

        if len(figs) != len(captions):
            raise ValueError(
                f'Number of figs ({len(figs)}) must equal number of captions '
                f'({len(captions)})'
            )
        if len(figs) != len(comments):
            raise ValueError(
                f'Number of figs ({len(figs)}) must equal number of comments '
                f'({len(comments)})'
            )

        tags = _clean_tags(section)

        for fig, title, caption in zip(figs, captions, comments):
            if scale is not None:
                _scale_mpl_figure(fig, scale)

            self.add_figure(
                fig=fig, title=title, caption=caption,
                image_format=image_format, tags=tags, replace=replace
            )

    @fill_doc
    def add_image(self, image, title, *, caption=None, tags=('custom-image',),
                  replace=False):
        """Add an image (e.g., PNG or JPEG pictures) to the report.

        Parameters
        ----------
        image : path-like
            The image to add.
        title : str
            Title corresponding to the images.
        caption : str | None
            If not ``None``, the caption to add to the image.
        %(report_tags)s
        %(report_replace)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)
        img_bytes = Path(image).expanduser().read_bytes()
        img_base64 = base64.b64encode(img_bytes).decode('ascii')
        del img_bytes  # Free memory

        img_format = Path(image).suffix.lower()[1:]  # omit leading period
        _check_option('Image format', value=img_format,
                      allowed_values=('png', 'gif', 'svg'))

        dom_id = self._get_dom_id()
        img_html = _html_image_element(
            img=img_base64, div_klass='custom-image',
            img_klass='custom-image', title=title, caption=caption,
            show=True, image_format=img_format, id=dom_id,
            tags=tags
        )
        self._add_or_replace(
            dom_id=dom_id,
            name=title,
            tags=tags,
            html=img_html,
            replace=replace
        )

    @deprecated('Use :meth:`~mne.Report.add_image` instead')
    def add_images_to_section(self, fnames, captions, scale=None,
                              section='custom', comments=None, replace=False):
        """Append custom user-defined images.

        Parameters
        ----------
        fnames : str | list of str
            A filename or a list of filenames from which images are read.
            Images can be PNG, GIF or SVG.
        captions : str | list of str
            A caption or a list of captions to the images.
        scale : float | None
            Scale the images maintaining the aspect ratio.
            Defaults to None. If None, no scaling will be applied.
        section : str
            Name of the section. If section already exists, the images
            will be appended to the end of the section.
        comments : None | str | list of str
            A string of text or a list of strings of text to be appended after
            the image.
        replace : bool
            If ``True``, figures already present that have the same caption
            will be replaced. Defaults to ``False``.
        """
        # Note: using scipy.misc is equivalent because scipy internally
        # imports PIL anyway. It's not possible to redirect image output
        # to binary string using scipy.misc.
        fnames, captions, comments = self._validate_input(fnames, captions,
                                                          section, comments)
        _check_scale(scale)

        if isinstance(fnames, str):
            fnames = (fnames,)
        else:
            fnames = tuple(fnames)

        if isinstance(captions, str):
            captions = (captions,)
        else:
            captions = tuple(captions)

        if isinstance(comments, str):
            comments = (comments,)
        elif comments is None:
            comments = (None,) * len(fnames)
        else:
            comments = tuple(comments)

        if len(fnames) != len(captions):
            raise ValueError(
                f'Number of fnames ({len(fnames)}) must equal number of '
                f'captions ({len(captions)})'
            )
        if len(fnames) != len(comments):
            raise ValueError(
                f'Number of fnames ({len(fnames)}) must equal number of '
                f'comments ({len(comments)})'
            )

        tags = _clean_tags(section)

        for image, title, caption in zip(fnames, captions, comments):
            self.add_image(image=image, title=title, caption=caption,
                           tags=tags, replace=replace)

    @fill_doc
    def add_html(self, html, title, *, tags=('custom-html',), replace=False):
        """Add HTML content to the report.

        Parameters
        ----------
        html : str
            The HTML content to add.
        title : str
            The title corresponding to ``html``.
        %(report_tags)s
        %(report_replace)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)
        dom_id = self._get_dom_id()
        html_element = _html_element(
            id=dom_id, html=html, title=title, tags=tags,
            div_klass='custom-html'
        )
        self._add_or_replace(
            dom_id=dom_id,
            name=title,
            tags=tags,
            html=html_element,
            replace=replace
        )

    @deprecated('Use :meth:`~mne.Report.add_html` instead')
    def add_htmls_to_section(self, htmls, captions, section='custom',
                             replace=False):
        """Append htmls to the report.

        Parameters
        ----------
        htmls : str | list of str
            An html str or a list of html str.
        captions : str | list of str
            A caption or a list of captions to the htmls.
        section : str
            Name of the section. If section already exists, the images
            will be appended to the end of the section.
        replace : bool
            If ``True``, figures already present that have the same caption
            will be replaced. Defaults to ``False``.

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        htmls, captions, _ = self._validate_input(htmls, captions, section)

        if isinstance(htmls, str):
            htmls = (htmls,)
        else:
            htmls = tuple(htmls)

        if isinstance(captions, str):
            captions = (captions,)
        else:
            captions = tuple(captions)

        if len(htmls) != len(captions):
            raise ValueError(
                f'Number of htmls ({len(htmls)}) must equal number of '
                f'captions ({len(captions)})'
            )

        tags = _clean_tags(section)
        for html, title in zip(htmls, captions):
            self.add_html(html=html, title=title, tags=tags, replace=replace)

    @deprecated('Use :meth:`~mne.Report.add_bem` instead')
    @verbose
    def add_bem_to_section(self, subject, caption='BEM', section='bem',
                           decim=2, n_jobs=1, subjects_dir=None,
                           replace=False, width=512, verbose=None):
        """Render a bem slider html str.

        Parameters
        ----------
        subject : str
            Subject name.
        caption : str
            A caption for the BEM.
        section : str
            Name of the section. If it already exists, the BEM
            will be appended to the end of the existing section.
        decim : int
            Use this decimation factor for generating MRI/BEM images
            (since it can be time consuming).
        %(n_jobs)s
        %(subjects_dir)s
        replace : bool
            If ``True``, figures already present that have the same caption
            will be replaced. Defaults to ``False``.
        width : int
            The width of the MRI images (in pixels). Larger values will have
            clearer surface lines, but will create larger HTML files.
            Typically a factor of 2 more than the number of MRI voxels along
            each dimension (typically 512, default) is reasonable.

            .. versionadded:: 0.23
        %(verbose_meth)s

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        tags = _clean_tags(section)
        self.add_bem(
            subject=subject, subjects_dir=subjects_dir, title=caption,
            decim=decim, width=width, n_jobs=n_jobs, tags=tags, replace=replace
        )

    @fill_doc
    def add_bem(self, subject, title, *, subjects_dir=None, decim=2, width=512,
                n_jobs=1, tags=('bem',), replace=False):
        """Render a visualization of the boundary element model (BEM) surfaces.

        Parameters
        ----------
        subject : str
            The FreeSurfer subject name.
        title : str
            The title corresponding to the BEM image.
        %(subjects_dir)s
        decim : int
            Use this decimation factor for generating MRI/BEM images
            (since it can be time consuming).
        width : int
            The width of the MRI images (in pixels). Larger values will have
            clearer surface lines, but will create larger HTML files.
            Typically a factor of 2 more than the number of MRI voxels along
            each dimension (typically 512, default) is reasonable.
        %(n_jobs)s
        %(report_tags)s
        %(report_replace)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)
        width = _ensure_int(width, 'width')
        html = self._render_bem(subject=subject, subjects_dir=subjects_dir,
                                decim=decim, n_jobs=n_jobs, width=width,
                                image_format=self.image_format, tags=tags)

        dom_id = self._get_dom_id()
        html = _html_element(
            div_klass='bem',
            id=dom_id,
            tags=tags,
            title=title,
            html=html,
        )
        self._add_or_replace(
            dom_id=dom_id,
            name=title,
            tags=tags,
            html=html,
            replace=replace
        )

    def _render_slider(self, *, figs, title, captions, start_idx, image_format,
                       tags, klass=''):
        if len(figs) != len(captions):
            raise ValueError(
                f'Number of captions ({len(captions)}) must be equal to the '
                f'number of figures ({len(figs)})'
            )
        images = [_fig_to_img(fig=fig, image_format=image_format)
                  for fig in figs]

        dom_id = self._get_dom_id()
        html = _html_slider_element(
            id=dom_id,
            title=title,
            captions=captions,
            tags=tags,
            images=images,
            image_format=image_format,
            start_idx=start_idx,
            klass=klass
        )

        return html, dom_id

    @deprecated('Use :meth:`~mne.Report.add_figure` instead')
    @fill_doc
    def add_slider_to_section(self, figs, captions=None, section='custom',
                              title='Slider', scale=None, image_format=None,
                              replace=False, auto_close=True):
        """Render a slider of figs to the report.

        Parameters
        ----------
        figs : list of Figure
            Each figure in the list can be an instance of
            :class:`matplotlib.figure.Figure`,
            :class:`mayavi.core.api.Scene`, or :class:`numpy.ndarray`.
        captions : list of str | list of float | None
            A list of captions to the figures. If ``None``, it will default to
            ``Data slice [i]``.
        section : str
            Name of the section. If section already exists, the figures
            will be appended to the end of the section.
        title : str
            The title of the slider.
        scale : float | None | callable
            Scale the images maintaining the aspect ratio.
            If None, no scaling is applied. If float, scale will determine
            the relative scaling (might not work for scale <= 1 depending on
            font sizes). If function, should take a figure object as input
            parameter. Defaults to None.
        %(report_image_format)s
        replace : bool
            If ``True``, figures already present that have the same caption
            will be replaced. Defaults to ``False``.
        auto_close : bool
            If True, the plots are closed during the generation of the report.
            Defaults to True.

            .. versionadded:: 0.23

        Notes
        -----
        .. versionadded:: 0.10.0
        """
        tags = _clean_tags(section)

        if captions is None:
            captions = [f'Figure {i+1} of {len(figs)}'
                        for i in range(len(figs))]

        if image_format is None:
            image_format = self.image_format

        for fig in figs:
            if scale is not None:
                _scale_mpl_figure(fig, scale)

        html, dom_id = self._render_slider(
            figs=figs, title=title, captions=captions, start_idx=0,
            image_format=image_format, tags=tags
        )
        self._add_or_replace(
            dom_id=dom_id,
            name=title,
            tags=tags,
            html=html,
            replace=replace
        )

    ###########################################################################
    # global rendering functions
    @verbose
    def _init_render(self, verbose=None):
        """Initialize the renderer."""
        inc_fnames = [
            'jquery-3.6.0.min.js',
            'bootstrap.bundle.min.js', 'bootstrap.min.css',
            'highlightjs/highlight.min.js',
            'highlightjs/atom-one-dark-reasonable.min.css'
        ]

        include = list()
        for inc_fname in inc_fnames:
            logger.info(f'Embedding : {inc_fname}')
            fname = html_include_dir / inc_fname
            file_content = fname.read_text(encoding='utf-8')

            if inc_fname.endswith('.js'):
                include.append(
                    f'<script type="text/javascript">\n'
                    f'{file_content}\n'
                    f'</script>'
                )
            elif inc_fname.endswith('.css'):
                include.append(
                    f'<style type="text/css">\n'
                    f'{file_content}\n'
                    f'</style>'
                )
        self.include = ''.join(include)

    def _iterate_files(self, *, fnames, cov, sfreq, raw_butterfly,
                       n_time_points_evokeds, n_time_points_stcs, on_error,
                       stc_plot_kwargs, topomap_kwargs):
        """Parallel process in batch mode."""
        assert self.data_path is not None

        for fname in fnames:
            logger.info(
                f"Rendering : {op.join('…' + self.data_path[-20:], fname)}"
            )

            title = Path(fname).name
            try:
                if _endswith(fname, ['raw', 'sss', 'meg', 'nirs']):
                    self.add_raw(
                        raw=fname, title=title, psd=self.raw_psd,
                        projs=self.projs, butterfly=raw_butterfly
                    )
                elif _endswith(fname, 'fwd'):
                    self.add_forward(
                        forward=fname, title=title, subject=self.subject,
                        subjects_dir=self.subjects_dir
                    )
                elif _endswith(fname, 'inv'):
                    # XXX if we pass trans, we can plot the source space, too…
                    self.add_inverse_operator(
                        inverse_operator=fname, title=title
                    )
                elif _endswith(fname, 'ave'):
                    evokeds = read_evokeds(fname)
                    titles = [
                        f'{Path(fname).name}: {e.comment}'
                        for e in evokeds
                    ]
                    self.add_evokeds(
                        evokeds=fname, titles=titles, noise_cov=cov,
                        n_time_points=n_time_points_evokeds,
                        topomap_kwargs=topomap_kwargs
                    )
                elif _endswith(fname, 'eve'):
                    if self.info_fname is not None:
                        sfreq = read_info(self.info_fname)['sfreq']
                    else:
                        sfreq = None
                    self.add_events(events=fname, title=title, sfreq=sfreq)
                elif _endswith(fname, 'epo'):
                    self.add_epochs(epochs=fname, title=title)
                elif _endswith(fname, 'cov') and self.info_fname is not None:
                    self.add_covariance(cov=fname, info=self.info_fname,
                                        title=title)
                elif _endswith(fname, 'proj') and self.info_fname is not None:
                    self.add_projs(info=self.info_fname, projs=fname,
                                   title=title, topomap_kwargs=topomap_kwargs)
                # XXX TODO We could render ICA components here someday
                # elif _endswith(fname, 'ica') and ica:
                #     pass
                elif (_endswith(fname, 'trans') and
                        self.info_fname is not None and
                        self.subjects_dir is not None and
                        self.subject is not None):
                    self.add_trans(
                        trans=fname, info=self.info_fname,
                        subject=self.subject, subjects_dir=self.subjects_dir,
                        title=title
                    )
                elif (fname.endswith('-lh.stc') or
                        fname.endswith('-rh.stc') and
                        self.info_fname is not None and
                        self.subjects_dir is not None and
                        self.subject is not None):
                    self.add_stc(
                        stc=fname, title=title, subject=self.subject,
                        subjects_dir=self.subjects_dir,
                        n_time_points=n_time_points_stcs,
                        stc_plot_kwargs=stc_plot_kwargs
                    )
            except Exception as e:
                if on_error == 'warn':
                    warn(f'Failed to process file {fname}:\n"{e}"')
                elif on_error == 'raise':
                    raise

    @verbose
    def parse_folder(self, data_path, pattern=None, n_jobs=1, mri_decim=2,
                     sort_content=True, sort_sections=None, on_error='warn',
                     image_format=None, render_bem=True, *,
                     n_time_points_evokeds=None, n_time_points_stcs=None,
                     raw_butterfly=True, stc_plot_kwargs=None,
                     topomap_kwargs=None, verbose=None):
        r"""Render all the files in the folder.

        Parameters
        ----------
        data_path : str
            Path to the folder containing data whose HTML report will be
            created.
        pattern : None | str | list of str
            Filename pattern(s) to include in the report.
            For example, ``[\*raw.fif, \*ave.fif]`` will include `~mne.io.Raw`
            as well as `~mne.Evoked` files. If ``None``, include all supported
            file formats.

            .. versionchanged:: 0.23
               Include supported non-FIFF files by default.
        %(n_jobs)s
        mri_decim : int
            Use this decimation factor for generating MRI/BEM images
            (since it can be time consuming).
        sort_content : bool
            If ``True``, sort the content based on tags in the order:
            raw -> events -> epochs -> evoked -> covariance -> coregistration
            -> bem -> forward-solution -> inverse-operator -> source-estimate.

            .. versionadded:: 0.24.0
        sort_sections : bool
            If ``True``, sort the content based on tags in the order:
            raw -> events -> epochs -> evoked -> covariance -> coregistration
            -> bem -> forward-solution -> inverse-operator -> source-estimate.

            .. deprecated:: 0.24.0
               This parameter is scheduled for removal. Use ``sort_content``
               instead.
        on_error : str
            What to do if a file cannot be rendered. Can be 'ignore',
            'warn' (default), or 'raise'.
        %(report_image_format)s

            .. versionadded:: 0.15
        render_bem : bool
            If True (default), try to render the BEM.

            .. versionadded:: 0.16
        n_time_points_evokeds, n_time_points_stcs : int | None
            The number of equidistant time points to render for `~mne.Evoked`
            and `~mne.SourceEstimate` data, respectively. If ``None``,
            will render each `~mne.Evoked` at 21 and each `~mne.SourceEstimate`
            at 51 time points, unless the respective data contains fewer time
            points, in which call all will be rendered.

            .. versionadded:: 0.24.0
        raw_butterfly : bool
            Whether to render butterfly plots for (decimated) `~mne.io.Raw`
            data.

            .. versionadded:: 0.24.0
        %(report_stc_plot_kwargs)s

            .. versionadded:: 0.24.0
        %(topomap_kwargs)s

            .. versionadded:: 0.24.0
        %(verbose_meth)s
        """
        _validate_type(data_path, 'path-like', 'data_path')
        data_path = str(data_path)
        image_format = _check_image_format(self, image_format)
        _check_option('on_error', on_error, ['ignore', 'warn', 'raise'])

        if sort_sections is not None:
            warn(
                message='The "sort_sections" parameter has been deprecated. '
                        'Please use "sort_content" instead',
                category=DeprecationWarning
            )
            sort_content = sort_sections

        n_jobs = check_n_jobs(n_jobs)
        self.data_path = data_path

        if self.title is None:
            self.title = f'MNE Report for {self.data_path[-20:]}'

        if pattern is None:
            pattern = [f'*{ext}' for ext in SUPPORTED_READ_RAW_EXTENSIONS]
        elif not isinstance(pattern, (list, tuple)):
            pattern = [pattern]

        # iterate through the possible patterns
        fnames = list()
        for p in pattern:
            data_path = _check_fname(
                fname=self.data_path, overwrite='read', must_exist=True,
                name='Directory or folder', need_dir=True
            )
            fnames.extend(sorted(_recursive_search(data_path, p)))

        if not fnames and not render_bem:
            raise RuntimeError(f'No matching files found in {self.data_path}')

        fnames_to_remove = []
        for fname in fnames:
            # For split files, only keep the first one.
            if _endswith(fname, ('raw', 'sss', 'meg')):
                kwargs = dict(fname=fname, preload=False)
                if fname.endswith(('.fif', '.fif.gz')):
                    kwargs['allow_maxshield'] = True
                inst = read_raw(**kwargs)

                if len(inst.filenames) > 1:
                    fnames_to_remove.extend(inst.filenames[1:])
            # For STCs, only keep one hemisphere
            elif fname.endswith('-lh.stc') or fname.endswith('-rh.stc'):
                first_hemi_fname = fname
                if first_hemi_fname.endswidth('-lh.stc'):
                    second_hemi_fname = (first_hemi_fname
                                         .replace('-lh.stc', '-rh.stc'))
                else:
                    second_hemi_fname = (first_hemi_fname
                                         .replace('-rh.stc', '-lh.stc'))

                if (second_hemi_fname in fnames and
                        first_hemi_fname not in fnames_to_remove):
                    fnames_to_remove.extend(first_hemi_fname)
            else:
                continue

        fnames_to_remove = list(set(fnames_to_remove))  # Drop duplicates
        for fname in fnames_to_remove:
            if fname in fnames:
                del fnames[fnames.index(fname)]
        del fnames_to_remove

        if self.info_fname is not None:
            info = read_info(self.info_fname, verbose=False)
            sfreq = info['sfreq']
        else:
            # only warn if relevant
            if any(_endswith(fname, 'cov') for fname in fnames):
                warn('`info_fname` not provided. Cannot render '
                     '-cov.fif(.gz) files.')
            if any(_endswith(fname, 'trans') for fname in fnames):
                warn('`info_fname` not provided. Cannot render '
                     '-trans.fif(.gz) files.')
            if any(_endswith(fname, 'proj') for fname in fnames):
                warn('`info_fname` not provided. Cannot render '
                     '-proj.fif(.gz) files.')
            info, sfreq = None, None

        cov = None
        if self.cov_fname is not None:
            cov = read_cov(self.cov_fname)

        # render plots in parallel; check that n_jobs <= # of files
        logger.info(f'Iterating over {len(fnames)} potential files '
                    f'(this may take some ')
        use_jobs = min(n_jobs, max(1, len(fnames)))
        parallel, p_fun, _ = parallel_func(self._iterate_files, use_jobs)
        parallel(
            p_fun(
                fnames=fname, cov=cov, sfreq=sfreq,
                raw_butterfly=raw_butterfly,
                n_time_points_evokeds=n_time_points_evokeds,
                n_time_points_stcs=n_time_points_stcs, on_error=on_error,
                stc_plot_kwargs=stc_plot_kwargs, topomap_kwargs=topomap_kwargs,
            ) for fname in np.array_split(fnames, use_jobs)
        )

        # Render BEM
        if render_bem:
            if self.subjects_dir is not None and self.subject is not None:
                logger.info('Rendering BEM')
                self.add_bem(
                    subject=self.subject, subjects_dir=self.subjects_dir,
                    title='BEM surfaces', decim=mri_decim, n_jobs=n_jobs
                )
            else:
                warn('`subjects_dir` and `subject` not provided. Cannot '
                     'render MRI and -trans.fif(.gz) files.')

        if sort_content:
            self._content = self._sort(
                content=self._content, order=CONTENT_ORDER
            )

    def __getstate__(self):
        """Get the state of the report as a dictionary."""
        state = dict()
        for param_name in self._get_state_params():
            param_val = getattr(self, param_name)

            # Workaround as h5io doesn't support dataclasses
            if param_name == '_content':
                assert all(dataclasses.is_dataclass(val) for val in param_val)
                param_val = [dataclasses.asdict(val) for val in param_val]

            state[param_name] = param_val
        return state

    def __setstate__(self, state):
        """Set the state of the report."""
        for param_name in self._get_state_params():
            param_val = state[param_name]

            # Workaround as h5io doesn't support dataclasses
            if param_name == '_content':
                param_val = [_ContentElement(**val) for val in param_val]
            setattr(self, param_name, param_val)
        return state

    @verbose
    def save(self, fname=None, open_browser=True, overwrite=False,
             sort_content=False, *, verbose=None):
        """Save the report and optionally open it in browser.

        Parameters
        ----------
        fname : path-like | None
            Output filename. If the name ends with ``.h5`` or ``.hdf5``, the
            report is saved in HDF5 format, so it can later be loaded again
            with :func:`open_report`. For any other suffix, the report will be
            saved in HTML format. If ``None`` and :meth:`Report.parse_folder`
            was **not** called, the report is saved as ``report.html`` in the
            current working directory. If ``None`` and
            :meth:`Report.parse_folder` **was** used, the report is saved as
            ``report.html`` inside the ``data_path`` supplied to
            :meth:`Report.parse_folder`.
        open_browser : bool
            Whether to open the rendered HTML report in the default web browser
            after saving. This is ignored when writing an HDF5 file.
        %(overwrite)s
        sort_content : bool
            If ``True``, sort the content based on tags before saving in the
            order:
            raw -> events -> epochs -> evoked -> covariance -> coregistration
            -> bem -> forward-solution -> inverse-operator -> source-estimate.

            .. versionadded:: 0.24.0
        %(verbose_meth)s

        Returns
        -------
        fname : str
            The file name to which the report was saved.
        """
        if fname is None:
            if self.data_path is None:
                self.data_path = os.getcwd()
                warn(f'`data_path` not provided. Using {self.data_path} '
                     f'instead')
            fname = op.join(self.data_path, 'report.html')

        fname = _check_fname(fname, overwrite=overwrite, name=fname)
        fname = op.realpath(fname)  # resolve symlinks

        if sort_content:
            self._content = self._sort(
                content=self._content, order=CONTENT_ORDER
            )

        if not overwrite and op.isfile(fname):
            msg = (f'Report already exists at location {fname}. '
                   f'Overwrite it (y/[n])? ')
            answer = _safe_input(msg, alt='pass overwrite=True')
            if answer.lower() == 'y':
                overwrite = True

        _, ext = op.splitext(fname)
        is_hdf5 = ext.lower() in ['.h5', '.hdf5']

        if overwrite or not op.isfile(fname):
            logger.info(f'Saving report to : {fname}')

            if is_hdf5:
                write_hdf5(fname, self.__getstate__(), overwrite=overwrite,
                           title='mnepython')
            else:
                # Add header, TOC, and footer.
                header_html = _html_header_element(
                    title=self.title, include=self.include, lang=self.lang,
                    tags=self.tags, js=JAVASCRIPT, css=CSS,
                    mne_logo_img=mne_logo
                )

                toc_html = _html_toc_element(content_elements=self._content)

                with warnings.catch_warnings(record=True):
                    warnings.simplefilter('ignore')
                    footer_html = _html_footer_element(
                        mne_version=MNE_VERSION,
                        date=time.strftime("%B %d, %Y")
                    )

                html = [header_html, toc_html, *self.html, footer_html]
                Path(fname).write_text(data=''.join(html), encoding='utf-8')

        building_doc = os.getenv('_MNE_BUILDING_DOC', '').lower() == 'true'
        if open_browser and not is_hdf5 and not building_doc:
            webbrowser.open_new_tab('file://' + fname)

        if self.fname is None:
            self.fname = fname
        return fname

    def __enter__(self):
        """Do nothing when entering the context block."""
        return self

    def __exit__(self, type, value, traceback):
        """Save the report when leaving the context block."""
        if self.fname is not None:
            self.save(self.fname, open_browser=False, overwrite=True)

    @staticmethod
    def _sort(content, order):
        """Reorder content to reflect "natural" ordering."""
        content_unsorted = content.copy()
        content_sorted = []
        content_sorted_idx = []
        del content

        # First arrange content with known tags in the predefined order
        for tag in order:
            for idx, content in enumerate(content_unsorted):
                if tag in content.tags:
                    content_sorted_idx.append(idx)
                    content_sorted.append(content)

        # Now simply append the rest (custom tags)
        content_remaining = [
            content for idx, content in enumerate(content_unsorted)
            if idx not in content_sorted_idx
        ]

        content_sorted = [*content_sorted, *content_remaining]
        return content_sorted

    def _render_one_bem_axis(self, *, mri_fname, surfaces,
                             image_format, orientation, decim=2, n_jobs=1,
                             width=512, tags):
        """Render one axis of bem contours (only PNG)."""
        import nibabel as nib

        nim = nib.load(mri_fname)
        data = _reorient_image(nim)[0]
        axis = _mri_orientation(orientation)[0]
        n_slices = data.shape[axis]

        sl = np.arange(0, n_slices, decim)
        logger.debug(f'Rendering BEM {orientation} with {len(sl)} slices')
        figs = _get_bem_contour_figs_as_arrays(
            sl=sl, n_jobs=n_jobs, mri_fname=mri_fname, surfaces=surfaces,
            orientation=orientation, src=None, show=False,
            show_orientation='always', width=width
        )

        # Render the slider
        captions = [f'Slice index: {i * decim}' for i in range(len(figs))]
        start_idx = int(round(len(figs) / 2))
        html, _ = self._render_slider(
            figs=figs,
            captions=captions,
            title=orientation,
            image_format=image_format,
            start_idx=start_idx,
            tags=tags,
            klass='bem col-md'
        )

        return html

    def _render_raw_butterfly_segments(self, *, raw: BaseRaw, image_format,
                                       tags):
        # Pick 10 1-second time slices
        times = np.linspace(raw.times[0], raw.times[-1], 12)[1:-1]
        figs = []
        for t in times:
            tmin = max(t - 0.5, 0)
            tmax = min(t + 0.5, raw.times[-1])
            duration = tmax - tmin
            fig = raw.plot(butterfly=True, show_scrollbars=False, start=tmin,
                           duration=duration, show=False)
            figs.append(fig)

        captions = [f'Segment {i+1} of {len(figs)}'
                    for i in range(len(figs))]

        html, _ = self._render_slider(
            figs=figs, title='Time series', captions=captions,
            start_idx=0, image_format=image_format, tags=tags
        )

        return html

    def _render_raw(self, *, raw, add_psd, add_projs, add_butterfly,
                    image_format, tags, topomap_kwargs):
        """Render raw."""
        if isinstance(raw, BaseRaw):
            fname = raw.filenames[0]
        else:
            fname = str(raw)  # could e.g. be a Path!
            kwargs = dict(fname=fname, preload=False)
            if fname.endswith(('.fif', '.fif.gz')):
                kwargs['allow_maxshield'] = True
            raw = read_raw(**kwargs)

        # Summary table
        dom_id = self._get_dom_id()
        repr_html = _html_element(
            div_klass='raw',
            id=dom_id,
            tags=tags,
            title='Info',
            html=raw._repr_html_()
        )

        # Butterfly plot
        if add_butterfly:
            butterfly_imgs_html = self._render_raw_butterfly_segments(
                raw=raw, image_format=image_format, tags=tags
            )
        else:
            butterfly_imgs_html = ''

        # PSD
        if isinstance(add_psd, dict):
            dom_id = self._get_dom_id()
            if raw.info['lowpass'] is not None:
                fmax = raw.info['lowpass'] + 15
                # Must not exceed half the sampling frequency
                if fmax > 0.5 * raw.info['sfreq']:
                    fmax = np.inf
            else:
                fmax = np.inf

            fig = raw.plot_psd(fmax=fmax, show=False, **add_psd)
            tight_layout(fig=fig)

            img = _fig_to_img(fig, image_format=image_format)
            psd_img_html = _html_image_element(
                img=img, div_klass='raw', img_klass='raw',
                title='PSD', caption=None, show=True,
                image_format=image_format, id=dom_id, tags=tags
            )
        else:
            psd_img_html = ''

        ssp_projs_html = self._ssp_projs_html(
            add_projs=add_projs, info=raw, image_format=image_format,
            tags=tags, topomap_kwargs=topomap_kwargs)

        return [repr_html, psd_img_html, butterfly_imgs_html, ssp_projs_html]

    def _ssp_projs_html(self, *, add_projs, info, image_format, tags,
                        topomap_kwargs):
        if add_projs:
            output = self._render_ssp_projs(
                info=info, projs=None, title='SSP Projectors',
                image_format=image_format, tags=tags,
                topomap_kwargs=topomap_kwargs,
            )
            if output is None:
                ssp_projs_html = ''
            else:
                ssp_projs_html, _ = output
        else:
            ssp_projs_html = ''
        return ssp_projs_html

    def _render_ssp_projs(self, *, info, projs, title, image_format, tags,
                          topomap_kwargs):
        if isinstance(info, Info):  # no-op
            pass
        elif hasattr(info, 'info'):  # try to get the file name
            if isinstance(info, BaseRaw):
                fname = info.filenames[0]
            # elif isinstance(info, (Evoked, BaseEpochs)):
            #     fname = info.filename
            else:
                fname = ''
            info = info.info
        else:  # read from a file
            fname = info
            info = read_info(fname, verbose=False)

        if projs is None:
            projs = info['projs']
        elif not isinstance(projs, list):
            fname = projs
            projs = read_proj(fname)

        if not projs:  # Abort mission!
            return None

        if not _check_ch_locs(info=info):
            warn('No channel locations found, cannot create projector plots')
            return '', None

        topomap_kwargs = self._validate_topomap_kwargs(topomap_kwargs)
        fig = plot_projs_topomap(
            projs=projs, info=info, colorbar=True, vlim='joint',
            show=False, **topomap_kwargs
        )
        # TODO This seems like a bad idea, better to provide a way to set a
        # desired size in plot_projs_topomap, but that uses prepare_trellis...
        # hard to see how (6, 4) could work in all number-of-projs by
        # number-of-channel-types conditions...
        fig.set_size_inches((6, 4))
        tight_layout(fig=fig)
        img = _fig_to_img(fig=fig, image_format=image_format)

        dom_id = self._get_dom_id()
        html = _html_image_element(
            img=img, div_klass='ssp', img_klass='ssp',
            title=title, caption=None, show=True, image_format=image_format,
            id=dom_id, tags=tags
        )

        return html, dom_id

    def _render_forward(self, *, forward, subject, subjects_dir, title,
                        image_format, tags):
        """Render forward solution."""
        if not isinstance(forward, Forward):
            forward = read_forward_solution(forward)

        subject = self.subject if subject is None else subject
        subjects_dir = (self.subjects_dir if subjects_dir is None
                        else subjects_dir)

        # XXX Todo
        # Render sensitivity maps
        if subject is not None:
            sensitivity_maps_html = ''
        else:
            sensitivity_maps_html = ''

        dom_id = self._get_dom_id()
        html = _html_forward_sol_element(
            id=dom_id,
            repr=forward._repr_html_(),
            sensitivity_maps=sensitivity_maps_html,
            title=title,
            tags=tags
        )
        return html, dom_id

    def _render_inverse_operator(self, *, inverse_operator, subject,
                                 subjects_dir, trans, title, image_format,
                                 tags):
        """Render inverse operator."""
        if not isinstance(inverse_operator, InverseOperator):
            inverse_operator = read_inverse_operator(inverse_operator)

        if trans is not None and not isinstance(trans, Transform):
            trans = read_trans(trans)

        subject = self.subject if subject is None else subject
        subjects_dir = (self.subjects_dir if subjects_dir is None
                        else subjects_dir)

        # XXX Todo Render source space?
        # if subject is not None and trans is not None:
        #     src = inverse_operator['src']

        #     fig = plot_alignment(
        #         subject=subject,
        #         subjects_dir=subjects_dir,
        #         trans=trans,
        #         surfaces='white',
        #         src=src
        #     )
        #     set_3d_view(fig, focalpoint=(0., 0., 0.06))
        #     img = _fig_to_img(fig=fig, image_format=image_format)

        #     dom_id = self._get_dom_id()
        #     src_img_html = _html_image_element(
        #         img=img,
        #         div_klass='inverse-operator source-space',
        #         img_klass='inverse-operator source-space',
        #         title='Source space', caption=None, show=True,
        #         image_format=image_format, id=dom_id,
        #         tags=tags
        #     )
        # else:
        src_img_html = ''

        dom_id = self._get_dom_id()
        html = _html_inverse_operator_element(
            id=dom_id,
            repr=inverse_operator._repr_html_(),
            source_space=src_img_html,
            title=title,
            tags=tags,
        )
        return html, dom_id

    def _render_evoked_joint(self, evoked, ch_types, image_format, tags,
                             topomap_kwargs):
        htmls = []
        for ch_type in ch_types:
            if not _check_ch_locs(info=evoked.info, ch_type=ch_type):
                ch_type_name = _handle_default("titles")[ch_type]
                warn(f'No {ch_type_name} channel locations found, cannot '
                     f'create joint plot')
                continue

            with use_log_level(level=False):
                fig = evoked.copy().pick(ch_type, verbose=False).plot_joint(
                    ts_args=dict(gfp=True),
                    title=None,
                    show=False,
                    topomap_args=topomap_kwargs,
                )

            img = _fig_to_img(fig=fig, image_format=image_format)
            title = f'Time course ({_handle_default("titles")[ch_type]})'
            dom_id = self._get_dom_id()

            htmls.append(
                _html_image_element(
                    img=img,
                    div_klass='evoked evoked-joint',
                    img_klass='evoked evoked-joint',
                    tags=tags,
                    title=title,
                    caption=None,
                    show=True,
                    image_format=image_format,
                    id=dom_id
                )
            )

        html = '\n'.join(htmls)
        return html

    def _plot_one_evoked_topomap_timepoint(
        self, *, evoked, time, ch_types, vmin, vmax, topomap_kwargs
    ):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            1, len(ch_types) * 2,
            gridspec_kw={'width_ratios': [8, 0.5] * len(ch_types)},
            figsize=(4 * len(ch_types), 3.5)
        )
        ch_type_ax_map = dict(
            zip(ch_types,
                [(ax[i], ax[i + 1]) for i in
                    range(0, 2 * len(ch_types) - 1, 2)])
        )

        for ch_type in ch_types:
            evoked.plot_topomap(
                times=[time], ch_type=ch_type,
                vmin=vmin[ch_type], vmax=vmax[ch_type],
                axes=ch_type_ax_map[ch_type], show=False,
                **topomap_kwargs
            )
            ch_type_ax_map[ch_type][0].set_title(ch_type)

        tight_layout(fig=fig)

        with BytesIO() as buff:
            fig.savefig(
                buff, format='png',
                dpi=fig.get_dpi(),
                pad_inches=0
            )
            plt.close(fig)
            buff.seek(0)
            fig_array = plt.imread(buff, format='png')
        return fig_array

    def _render_evoked_topomap_slider(self, *, evoked, ch_types, n_time_points,
                                      image_format, tags, topomap_kwargs,
                                      n_jobs):
        if n_time_points is None:
            n_time_points = min(len(evoked.times), 21)
        elif n_time_points > len(evoked.times):
            raise ValueError(
                f'The requested number of time points ({n_time_points}) '
                f'exceeds the time points in the provided Evoked object '
                f'({len(evoked.times)})'
            )

        if n_time_points == 1:  # only a single time point, pick the first one
            times = [evoked.times[0]]
        else:
            times = np.linspace(
                start=evoked.tmin,
                stop=evoked.tmax,
                num=n_time_points
            )

        t_zero_idx = np.abs(times).argmin()  # index closest to zero

        # global min and max values for each channel type
        scalings = dict(eeg=1e6, grad=1e13, mag=1e15)

        vmax = dict()
        vmin = dict()
        for ch_type in ch_types:
            if not _check_ch_locs(info=evoked.info, ch_type=ch_type):
                ch_type_name = _handle_default("titles")[ch_type]
                warn(f'No {ch_type_name} channel locations found, cannot '
                     f'create topography plots')
                continue

            vmax[ch_type] = (np.abs(evoked.copy()
                                    .pick(ch_type, verbose=False)
                                    .data)
                             .max()) * scalings[ch_type]
            if ch_type == 'grad':
                vmin[ch_type] = 0
            else:
                vmin[ch_type] = -vmax[ch_type]

        if not (vmin and vmax):  # we only had EEG data and no digpoints
            html = ''
            dom_id = None
        else:
            topomap_kwargs = self._validate_topomap_kwargs(topomap_kwargs)

            use_jobs = min(n_jobs, max(1, len(times)))
            parallel, p_fun, _ = parallel_func(
                func=self._plot_one_evoked_topomap_timepoint,
                n_jobs=use_jobs
            )
            fig_arrays = parallel(
                p_fun(
                    evoked=evoked, time=time, ch_types=ch_types,
                    vmin=vmin, vmax=vmax, topomap_kwargs=topomap_kwargs
                ) for time in times
            )

            captions = [f'Time point: {round(t, 3):0.3f} s' for t in times]
            html, dom_id = self._render_slider(
                figs=fig_arrays,
                captions=captions,
                title='Topographies',
                image_format=image_format,
                start_idx=t_zero_idx,
                tags=tags
            )

        return html, dom_id

    def _render_evoked_gfp(self, evoked, ch_types, image_format, tags):
        # Make legend labels shorter by removing the multiplicative factors
        pattern = r'\d\.\d* × '
        label = evoked.comment
        if label is None:
            label = ''
        for match in re.findall(pattern=pattern, string=label):
            label = label.replace(match, '')

        dom_id = self._get_dom_id()

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(ch_types), 1, sharex=True)
        if len(ch_types) == 1:
            ax = [ax]
        for idx, ch_type in enumerate(ch_types):
            plot_compare_evokeds(
                evokeds={
                    label: evoked.copy().pick(ch_type, verbose=False)
                },
                ci=None, truncate_xaxis=False,
                truncate_yaxis=False, legend=False,
                axes=ax[idx], show=False
            )
            ax[idx].set_title(ch_type)

            # Hide x axis label for all but the last subplot
            if idx < len(ch_types) - 1:
                ax[idx].set_xlabel(None)

        tight_layout(fig=fig)
        img = _fig_to_img(fig=fig, image_format=image_format)
        title = 'Global field power'
        html = _html_image_element(
            img=img,
            id=dom_id,
            tags=tags,
            div_klass='evoked evoked-gfp',
            img_klass='evoked evoked-gfp',
            title=title,
            caption=None,
            image_format=image_format,
            show=True
        )

        return html

    def _render_evoked_whitened(self, evoked, *, noise_cov, image_format,
                                tags):
        """Render whitened evoked."""
        dom_id = self._get_dom_id()
        fig = evoked.plot_white(
            noise_cov=noise_cov,
            show=False
        )
        tight_layout(fig=fig)
        img = _fig_to_img(fig=fig, image_format=image_format)
        title = 'Whitened'

        html = _html_image_element(
            img=img, id=dom_id, div_klass='evoked',
            img_klass='evoked evoked-whitened', title=title, caption=None,
            show=True, image_format=image_format, tags=tags
        )
        return html

    def _render_evoked(self, evoked, noise_cov, add_projs, n_time_points,
                       image_format, tags, topomap_kwargs, n_jobs):
        ch_types = _get_ch_types(evoked)
        joint_html = self._render_evoked_joint(
            evoked=evoked, ch_types=ch_types,
            image_format=image_format, tags=tags,
            topomap_kwargs=topomap_kwargs,
        )
        slider_html, _ = self._render_evoked_topomap_slider(
            evoked=evoked, ch_types=ch_types,
            n_time_points=n_time_points,
            image_format=image_format,
            tags=tags, topomap_kwargs=topomap_kwargs,
            n_jobs=n_jobs
        )
        gfp_html = self._render_evoked_gfp(
            evoked=evoked, ch_types=ch_types, image_format=image_format,
            tags=tags
        )

        if noise_cov is not None:
            whitened_html = self._render_evoked_whitened(
                evoked=evoked,
                noise_cov=noise_cov,
                image_format=image_format,
                tags=tags
            )
        else:
            whitened_html = ''

        # SSP projectors
        ssp_projs_html = self._ssp_projs_html(
            add_projs=add_projs, info=evoked, image_format=image_format,
            tags=tags, topomap_kwargs=topomap_kwargs)

        logger.debug('Evoked: done')
        return joint_html, slider_html, gfp_html, whitened_html, ssp_projs_html

    def _render_events(self, events, *, event_id, sfreq, first_samp, title,
                       image_format, tags):
        """Render events."""
        if not isinstance(events, np.ndarray):
            events = read_events(filename=events)

        fig = plot_events(
            events=events,
            event_id=event_id,
            sfreq=sfreq,
            first_samp=first_samp,
            show=False
        )

        img = _fig_to_img(
            fig=fig,
            image_format=image_format,
        )

        dom_id = self._get_dom_id()
        html = _html_image_element(
            img=img,
            id=dom_id,
            div_klass='events',
            img_klass='events',
            tags=tags,
            title=title,
            caption=None,
            show=True,
            image_format=image_format
        )
        return html, dom_id

    def _render_epochs(self, *, epochs, add_psd, add_projs, image_format,
                       tags, topomap_kwargs):
        """Render epochs."""
        if isinstance(epochs, BaseEpochs):
            fname = epochs.filename
        else:
            fname = epochs
            epochs = read_epochs(fname, preload=False)

        # Summary table
        dom_id = self._get_dom_id()
        repr_html = _html_element(
            div_klass='epochs',
            id=dom_id,
            tags=tags,
            title='Info',
            html=epochs._repr_html_()
        )

        # ERP/ERF image(s)
        ch_types = _get_ch_types(epochs)
        erp_img_htmls = []
        epochs.load_data()

        for ch_type in ch_types:
            with use_log_level(level=False):
                figs = epochs.copy().pick(ch_type, verbose=False).plot_image(
                    show=False
                )

            assert len(figs) == 1
            fig = figs[0]
            img = _fig_to_img(fig=fig, image_format=image_format)
            if ch_type in ('mag', 'grad'):
                title_start = 'ERF image'
            else:
                assert 'eeg' in ch_type
                title_start = 'ERP image'

            title = (f'{title_start} '
                     f'({_handle_default("titles")[ch_type]})')
            dom_id = self._get_dom_id()
            erp_img_htmls.append(
                _html_image_element(
                    img=img,
                    div_klass='epochs erp-image',
                    img_klass='epochs erp-image',
                    tags=tags,
                    title=title,
                    caption=None,
                    show=True,
                    image_format=image_format,
                    id=dom_id
                )
            )
        erp_imgs_html = '\n'.join(erp_img_htmls)

        # Drop log
        if epochs._bad_dropped:
            title = 'Drop log'
            dom_id = self._get_dom_id()
            if epochs.drop_log_stats() == 0:  # No drops
                drop_log_img_html = _html_element(
                    html='No epochs exceeded the rejection thresholds. '
                         'Nothing was dropped.',
                    id=dom_id, div_klass='epochs', title=title, tags=tags
                )
            else:
                fig = epochs.plot_drop_log(subject=self.subject, show=False)
                tight_layout(fig=fig)
                img = _fig_to_img(fig=fig, image_format=image_format)
                drop_log_img_html = _html_image_element(
                    img=img, id=dom_id, div_klass='epochs', img_klass='epochs',
                    show=True, image_format=image_format, title=title,
                    caption=None, tags=tags
                )
        else:
            drop_log_img_html = ''

        # PSD
        if add_psd:
            dom_id = self._get_dom_id()
            if epochs.info['lowpass'] is not None:
                fmax = epochs.info['lowpass'] + 15
                # Must not exceed half the sampling frequency
                if fmax > 0.5 * epochs.info['sfreq']:
                    fmax = np.inf
            else:
                fmax = np.inf

            fig = epochs.plot_psd(fmax=fmax, show=False)
            img = _fig_to_img(fig=fig, image_format=image_format)
            psd_img_html = _html_image_element(
                img=img, id=dom_id, div_klass='epochs', img_klass='epochs',
                show=True, image_format=image_format, title='PSD',
                caption=None, tags=tags
            )
        else:
            psd_img_html = ''

        ssp_projs_html = self._ssp_projs_html(
            add_projs=add_projs, info=epochs, image_format=image_format,
            tags=tags, topomap_kwargs=topomap_kwargs
        )

        return (repr_html, erp_imgs_html, drop_log_img_html, psd_img_html,
                ssp_projs_html)

    def _render_cov(self, cov, *, info, image_format, tags):
        """Render covariance matrix & SVD."""
        if not isinstance(cov, Covariance):
            cov = read_cov(cov)
        if not isinstance(info, Info):
            info = read_info(info)

        fig_cov, fig_svd = plot_cov(cov=cov, info=info, show=False,
                                    show_svd=True)
        figs = [fig_cov, fig_svd]
        htmls = []

        titles = (
            'Covariance matrix',
            'Singular values'
        )

        for fig, title in zip(figs, titles):
            dom_id = self._get_dom_id()
            img = _fig_to_img(fig=fig, image_format=image_format)
            html = _html_image_element(
                img=img, id=dom_id, div_klass='covariance',
                img_klass='covariance', title=title, caption=None,
                image_format=image_format, tags=tags, show=True
            )
            htmls.append(html)

        return htmls

    def _render_trans(self, *, trans, info, subject, subjects_dir, title,
                      tags):
        """Render trans (only PNG)."""
        if not isinstance(trans, Transform):
            trans = read_trans(trans)
        if not isinstance(info, Info):
            info = read_info(info)

        kwargs = dict(info=info, trans=trans, subject=subject,
                      subjects_dir=subjects_dir, dig=True,
                      meg=['helmet', 'sensors'], show_axes=True,
                      coord_frame='mri')
        img, caption = _iterate_trans_views(
            function=plot_alignment, **kwargs)

        dom_id = self._get_dom_id()
        html = _html_image_element(
            img=img, id=dom_id, div_klass='trans',
            img_klass='trans', title=title, caption=caption,
            show=True, image_format='png', tags=tags
        )
        return html, dom_id

    def _render_stc(self, *, stc, title, subject, subjects_dir, n_time_points,
                    image_format, tags, stc_plot_kwargs):
        """Render STC."""
        if isinstance(stc, SourceEstimate):
            if subject is None:
                subject = self.subject  # supplied during Report init
                if not subject:
                    subject = stc.subject  # supplied when loading STC
                    if not subject:
                        raise ValueError(
                            'Please specify the subject name, as it  cannot '
                            'be found in stc.subject. You may wish to pass '
                            'the "subject" parameter to read_source_estimate()'
                        )
            else:
                subject = subject
        else:
            fname = stc
            stc = read_source_estimate(fname=fname, subject=subject)

        subjects_dir = (self.subjects_dir if subjects_dir is None
                        else subjects_dir)

        if n_time_points is None:
            n_time_points = min(len(stc.times), 51)
        elif n_time_points > len(stc.times):
            raise ValueError(
                f'The requested number of time points ({n_time_points}) '
                f'exceeds the time points in the provided STC object '
                f'({len(stc.times)})'
            )
        if n_time_points == 1:  # only a single time point, pick the first one
            times = [stc.times[0]]
        else:
            times = np.linspace(
                start=stc.times[0],
                stop=stc.times[-1],
                num=n_time_points
            )
        t_zero_idx = np.abs(times).argmin()  # index of time closest to zero

        # Plot using 3d backend if available, and use Matplotlib
        # otherwise.
        import matplotlib.pyplot as plt
        stc_plot_kwargs = _handle_default(
            'report_stc_plot_kwargs', stc_plot_kwargs
        )
        stc_plot_kwargs.update(subject=subject, subjects_dir=subjects_dir)

        if get_3d_backend() is not None:
            brain = stc.plot(**stc_plot_kwargs)
            brain._renderer.plotter.subplot(0, 0)
            backend_is_3d = True
        else:
            backend_is_3d = False

        figs = []
        for t in times:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action='ignore',
                    message='More than 20 figures have been opened',
                    category=RuntimeWarning)

                if backend_is_3d:
                    brain.set_time(t)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(brain.screenshot(time_viewer=True, mode='rgb'))
                    ax.axis('off')
                    tight_layout(fig=fig)
                    figs.append(fig)
                    plt.close(fig)
                else:
                    fig_lh = plt.figure()
                    fig_rh = plt.figure()

                    brain_lh = stc.plot(
                        views='lat', hemi='lh',
                        initial_time=t,
                        backend='matplotlib',
                        subject=subject,
                        subjects_dir=subjects_dir,
                        figure=fig_lh
                    )
                    brain_rh = stc.plot(
                        views='lat', hemi='rh',
                        initial_time=t,
                        subject=subject,
                        subjects_dir=subjects_dir,
                        backend='matplotlib',
                        figure=fig_rh
                    )
                    tight_layout(fig=fig_lh)  # TODO is this necessary?
                    tight_layout(fig=fig_rh)  # TODO is this necessary?
                    figs.append(brain_lh)
                    figs.append(brain_rh)
                    plt.close(fig_lh)
                    plt.close(fig_rh)

        if backend_is_3d:
            brain.close()
        else:
            brain_lh.close()
            brain_rh.close()

        captions = [f'Time point: {round(t, 3):0.3f} s' for t in times]
        html, dom_id = self._render_slider(
            figs=figs,
            captions=captions,
            title=title,
            image_format=image_format,
            start_idx=t_zero_idx,
            tags=tags
        )
        return html, dom_id

    def _render_bem(self, *, subject, subjects_dir, decim, n_jobs, width=512,
                    image_format, tags):
        """Render mri+bem (only PNG)."""
        if subjects_dir is None:
            subjects_dir = self.subjects_dir
        subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)

        # Get the MRI filename
        mri_fname = op.join(subjects_dir, subject, 'mri', 'T1.mgz')
        if not op.isfile(mri_fname):
            warn(f'MRI file "{mri_fname}" does not exist')

        # Get the BEM surface filenames
        bem_path = op.join(subjects_dir, subject, 'bem')

        surfaces = _get_bem_plotting_surfaces(bem_path)
        if not surfaces:
            warn('No BEM surfaces found, rendering empty MRI')

        htmls = []
        htmls.append('<div class="row">')
        for orientation in _BEM_VIEWS:
            html = self._render_one_bem_axis(
                mri_fname=mri_fname, surfaces=surfaces,
                orientation=orientation, decim=decim, n_jobs=n_jobs,
                width=width, image_format=image_format,
                tags=tags
            )
            htmls.append(html)
        htmls.append('</div>')
        return '\n'.join(htmls)


def _clean_tags(tags):
    if isinstance(tags, str):
        tags = (tags,)

    # Replace any whitespace characters with dashes
    tags_cleaned = tuple(re.sub(r'[\s*]', '-', tag) for tag in tags)
    return tags_cleaned


def _recursive_search(path, pattern):
    """Auxiliary function for recursive_search of the directory."""
    filtered_files = list()
    for dirpath, dirnames, files in os.walk(path):
        for f in fnmatch.filter(files, pattern):
            # only the following file types are supported
            # this ensures equitable distribution of jobs
            if f.endswith(VALID_EXTENSIONS):
                filtered_files.append(op.realpath(op.join(dirpath, f)))

    return filtered_files


###############################################################################
# Scraper for sphinx-gallery

_SCRAPER_TEXT = '''
.. only:: builder_html

    .. container:: row

        .. rubric:: The `HTML document <{0}>`__ written by :meth:`mne.Report.save`:

        .. raw:: html

            <iframe class="sg_report" sandbox="allow-scripts" src="{0}"></iframe>

'''  # noqa: E501
# Adapted from fa-file-code
_FA_FILE_CODE = '<svg class="sg_report" role="img" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 512"><path fill="#dec" d="M149.9 349.1l-.2-.2-32.8-28.9 32.8-28.9c3.6-3.2 4-8.8.8-12.4l-.2-.2-17.4-18.6c-3.4-3.6-9-3.7-12.4-.4l-57.7 54.1c-3.7 3.5-3.7 9.4 0 12.8l57.7 54.1c1.6 1.5 3.8 2.4 6 2.4 2.4 0 4.8-1 6.4-2.8l17.4-18.6c3.3-3.5 3.1-9.1-.4-12.4zm220-251.2L286 14C277 5 264.8-.1 252.1-.1H48C21.5 0 0 21.5 0 48v416c0 26.5 21.5 48 48 48h288c26.5 0 48-21.5 48-48V131.9c0-12.7-5.1-25-14.1-34zM256 51.9l76.1 76.1H256zM336 464H48V48h160v104c0 13.3 10.7 24 24 24h104zM209.6 214c-4.7-1.4-9.5 1.3-10.9 6L144 408.1c-1.4 4.7 1.3 9.6 6 10.9l24.4 7.1c4.7 1.4 9.6-1.4 10.9-6L240 231.9c1.4-4.7-1.3-9.6-6-10.9zm24.5 76.9l.2.2 32.8 28.9-32.8 28.9c-3.6 3.2-4 8.8-.8 12.4l.2.2 17.4 18.6c3.3 3.5 8.9 3.7 12.4.4l57.7-54.1c3.7-3.5 3.7-9.4 0-12.8l-57.7-54.1c-3.5-3.3-9.1-3.2-12.4.4l-17.4 18.6c-3.3 3.5-3.1 9.1.4 12.4z" class=""></path></svg>'  # noqa: E501


class _ReportScraper(object):
    """Scrape Report outputs.

    Only works properly if conf.py is configured properly and the file
    is written to the same directory as the example script.
    """

    def __init__(self):
        self.app = None
        self.files = dict()

    def __repr__(self):
        return '<ReportScraper>'

    def __call__(self, block, block_vars, gallery_conf):
        for report in block_vars['example_globals'].values():
            if (isinstance(report, Report) and
                    report.fname is not None and
                    report.fname.endswith('.html') and
                    gallery_conf['builder_name'] == 'html'):
                # Thumbnail
                image_path_iterator = block_vars['image_path_iterator']
                img_fname = next(image_path_iterator)
                img_fname = img_fname.replace('.png', '.svg')
                with open(img_fname, 'w') as fid:
                    fid.write(_FA_FILE_CODE)
                # copy HTML file
                html_fname = op.basename(report.fname)
                out_dir = op.join(
                    self.app.builder.outdir,
                    op.relpath(op.dirname(block_vars['target_file']),
                               self.app.builder.srcdir))
                os.makedirs(out_dir, exist_ok=True)
                out_fname = op.join(out_dir, html_fname)
                assert op.isfile(report.fname)
                self.files[report.fname] = out_fname
                # embed links/iframe
                data = _SCRAPER_TEXT.format(html_fname)
                return data
        return ''

    def copyfiles(self, *args, **kwargs):
        for key, value in self.files.items():
            copyfile(key, value)
