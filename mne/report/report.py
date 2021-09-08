"""Generate self-contained HTML reports from MNE objects."""

# Authors: Alex Gramfort <alexandre.gramfort@inria.fr>
#          Mainak Jas <mainak@neuro.hut.fi>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD-3-Clause

from typing import TypedDict, Tuple
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
from .. import (read_evokeds, read_events, pick_types, read_cov,
                read_source_estimate, read_trans, sys_info,
                Evoked, SourceEstimate, Covariance, Info, Transform)
from ..io import read_raw, read_info, BaseRaw
from ..io._read_raw import supported as extension_reader_map
from ..proj import read_proj
from .._freesurfer import _reorient_image, _mri_orientation
from ..utils import (logger, verbose, get_subjects_dir, warn, _ensure_int,
                     fill_doc, _check_option, _validate_type, _safe_input,
                     deprecated)
from ..viz import (plot_events, plot_alignment, plot_cov, plot_projs_topomap,
                   plot_compare_evokeds, set_3d_view, get_3d_backend)
from ..viz.misc import _plot_mri_contours, _get_bem_plotting_surfaces
from ..viz.utils import _ndarray_to_fig
from ..forward import read_forward_solution, Forward
from ..epochs import read_epochs, BaseEpochs
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

SECTION_ORDER = ('raw', 'events', 'epochs', 'ssp', 'evoked', 'covariance',
                 'trans', 'mri', 'forward', 'inverse')

html_include_dir = Path(__file__).parent / 'js_and_css'
template_dir = Path(__file__).parent / 'templates'
JAVASCRIPT = (html_include_dir / 'report.js').read_text(encoding='utf-8')
CSS = (html_include_dir / 'report.sass').read_text(encoding='utf-8')

###############################################################################
# HTML generation

def _html_header_element(*, lang, include, js, css, title, tags, mne_logo_img):
    template_path = template_dir / 'header.html'
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(lang=lang, include=include, js=js, css=css, title=title,
                     tags=tags, mne_logo_img=mne_logo_img)
    return t


def _html_footer_element(*, mne_version, date, current_year):
    template_path = template_dir / 'footer.html'
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(mne_version=mne_version, date=date,
                     current_year=current_year)
    return t


def _html_toc_element(*, toc_entries):
    template_path = template_dir / 'toc.html'
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(toc_entries=toc_entries)
    return t


def _html_raw_element(*, id, repr, psd, butterfly, ssp_projs, title, tags):
    template_path = template_dir / 'raw.html'
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(id=id, repr=repr, psd=psd, butterfly=butterfly,
                     ssp_projs=ssp_projs, tags=tags, title=title)
    return t


def _html_epochs_element(*, id, repr, drop_log, psd, ssp_projs, title, tags):
    template_path = template_dir / 'epochs.html'
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(id=id, repr=repr, drop_log=drop_log, psd=psd,
                     ssp_projs=ssp_projs, tags=tags, title=title)
    return t


def _html_evoked_element(*, id, joint, slider, gfp, whitened, ssp_projs, title,
                         tags):
    template_path = template_dir / 'evoked.html'
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(id=id, joint=joint, slider=slider, gfp=gfp,
                     whitened=whitened, ssp_projs=ssp_projs, tags=tags,
                     title=title)
    return t


def _html_cov_element(*, id, matrix, svd, title, tags):
    template_path = template_dir / 'cov.html'
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(id=id, matrix=matrix, svd=svd, tags=tags, title=title)
    return t


def _html_forward_sol_element(*, id, info, sensitivity_maps, title, tags):
    template_path = template_dir / 'forward.html'
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(id=id, info=info, sensitivity_maps=sensitivity_maps,
                     tags=tags, title=title)
    return t


def _html_inverse_op_element(*, id, info, source_space, title, tags):
    template_path = template_dir / 'inverse.html'
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(id=id, info=info, source_space=source_space, tags=tags,
                     title=title)
    return t


def _html_slider_element(*, id, images, captions, start_idx, image_format,
                         title, tags):
    template_path = template_dir / 'slider.html'
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(id=id, images=images, captions=captions, tags=tags,
                     title=title, start_idx=start_idx,
                     image_format=image_format)
    return t


def _html_image_element(*, id, img, image_format, caption, show, div_klass,
                        img_klass, title, tags):
    template_path = template_dir / 'image.html'
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(id=id, img=img, caption=caption, tags=tags, title=title,
                     image_format=image_format, div_klass=div_klass,
                     img_klass=img_klass, show=show)
    return t


def _html_code_element(*, id, code, language, title, tags):
    template_path = template_dir / 'code.html'
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(id=id, code=code, language=language, title=title,
                     tags=tags)
    return t


def _html_element(*, id, div_klass, html, title, tags):
    template_path = template_dir / 'html.html'
    t = Template(template_path.read_text(encoding='utf-8'))
    t = t.substitute(id=id, div_klass=div_klass, html=html, title=title,
                     tags=tags)
    return t


class TocEntry(TypedDict):
    name: str
    dom_target_id: str
    tags: Tuple[str]


###############################################################################
# PLOTTING FUNCTIONS

def _fig_to_img(fig, image_format='png', auto_close=True, **kwargs):
    """Plot figure and create a binary image."""
    # fig can be ndarray, mpl Figure, Mayavi Figure, or callable that produces
    # a mpl Figure
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    if isinstance(fig, np.ndarray):
        fig = _ndarray_to_fig(fig)
    elif callable(fig):
        if auto_close:
            plt.close('all')
        fig = fig(**kwargs)
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
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('ignore')  # incompatible axes
        fig.savefig(output, format=image_format, dpi=fig.get_dpi(),
                    bbox_to_inches='tight')
    plt.close(fig)
    output = output.getvalue()
    return (output.decode('utf-8') if image_format == 'svg' else
            base64.b64encode(output).decode('ascii'))


def _get_mri_contour_figs(sl, n_jobs, **kwargs):
    import matplotlib.pyplot as plt
    plt.close('all')
    use_jobs = min(n_jobs, max(1, len(sl)))
    parallel, p_fun, _ = parallel_func(_plot_mri_contours, use_jobs)
    outs = parallel(p_fun(slices=s, **kwargs)
                    for s in np.array_split(sl, use_jobs))
    out = list()
    for o in outs:
        out.extend(o)
    return out


def _iterate_trans_views(function, **kwargs):
    """Auxiliary function to iterate over views in trans fig."""
    import matplotlib.pyplot as plt
    from ..viz.backends.renderer import MNE_3D_BACKEND_TESTING
    from ..viz._brain.view import views_dicts

    fig = function(**kwargs)

    views = ['frontal', 'lateral', 'medial']
    views += ['axial', 'rostral', 'coronal']

    images = []
    for view in views:
        if not MNE_3D_BACKEND_TESTING:
            from ..viz.backends.renderer import backend
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

    fig2, ax = plt.subplots()
    ax.imshow(images)
    ax.axis('off')
    fig2.tight_layout()

    if not MNE_3D_BACKEND_TESTING:
        backend._close_all()

    img = _fig_to_img(fig2, image_format='png')
    plt.close(fig2)

    caption = (f'Average distance from {len(dists)} digitized points to '
               f'head: {1e3 * np.mean(dists):.2f} mm')

    return img, caption


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
    report._fname = fname
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
                 image_format='png', raw_psd=True, projs=False, verbose=None):
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

        self._initial_id = 0
        self.html = []
        self.include = []
        self.tags = []  # all tags
        self.lang = 'en-us'  # language setting for the HTML file
        # boolean to specify if sections should be ordered in natural
        # order of processing (raw -> events ... -> inverse)
        self._sort_sections = False
        if not isinstance(raw_psd, bool) and not isinstance(raw_psd, dict):
            raise TypeError('raw_psd must be bool or dict, got %s'
                            % (type(raw_psd),))
        self.raw_psd = raw_psd
        self._toc_entries = []
        self._init_render()  # Initialize the renderer

    def __repr__(self):
        """Print useful info about report."""
        s = '<Report | %d items' % len(self._toc_entries)
        if self.title is not None:
            s += ' | %s' % self.title
        fnames = [_get_fname(e['name']) for e in self._toc_entries]
        if len(fnames) > 4:
            s += '\n%s' % '\n'.join(fnames[:2])
            s += '\n ...\n'
            s += '\n'.join(fnames[-2:])
        elif len(fnames) > 0:
            s += '\n%s' % '\n'.join(fnames)
        s += '\n>'
        return s

    def __len__(self):
        """Return the number of files processed by the report.

        Returns
        -------
        n_files : int
            The number of files processed.
        """
        return len(self._toc_entries)

    def _get_id(self):
        """Get id of plot."""
        self._initial_id += 1
        return f'global{self._initial_id}'

    def _validate_input(self, items, captions, tag, comments=None):
        """Validate input."""
        if not isinstance(items, (list, tuple)):
            items = [items]
        if not isinstance(captions, (list, tuple)):
            captions = [captions]
        if not isinstance(comments, (list, tuple)):
            if comments is None:
                comments = [comments] * len(captions)
            else:
                comments = [comments]
        if len(comments) != len(items):
            raise ValueError('Comments and report items must have the same '
                             'length or comments should be None, got %d and %d'
                             % (len(comments), len(items)))
        elif len(captions) != len(items):
            raise ValueError('Captions and report items must have the same '
                             'length, got %d and %d'
                             % (len(captions), len(items)))

        # Book-keeping of section names
        if tag not in self.tags:
            self.tags.append(_clean_tag(tag))

        return items, captions, comments

    @property
    @deprecated
    def fnames(self):
        return [toc_entry['name'] for toc_entry in self._toc_entries]

    @property
    @deprecated(extra='Use Report.tags instead')
    def sections(self):
        return self.tags

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

    def add_epochs(self, epochs, title, *, projs=True, tags=('epochs')):
        """Add `~mne.Epochs` to the report.

        Parameters
        ----------
        epochs : path-like | instance of mne.Epochs
            The epochs to add to the report.
        title : str
            The title to add.
        projs : bool
            Whether to add SSP projector plots, if projectors are present in
            the data.
        tags : collection of str
            Tags to add for later interactive filtering.
        """
        tags = tuple(tags)
        for tag in tags:
            if tag not in self.tags:
                self.tags.append(tag)

        htmls = self._render_epochs(
            epochs=epochs,
            add_ssp_projs=projs,
            tags=tags,
            image_format=self.image_format
        )
        repr_html, drop_log_html, psd_html, ssp_projs_html = htmls

        dom_id = self._get_id()
        html = _html_epochs_element(
            repr=repr_html,
            drop_log=drop_log_html,
            psd=psd_html,
            ssp_projs=ssp_projs_html,
            tags=tags,
            title=title,
            id=dom_id,
        )
        self._add_or_replace(
            toc_entry_name=title,
            dom_id=dom_id,
            tags=tags,
            html=html
        )

    def add_evokeds(self, evokeds, titles, *, baseline=None, noise_cov=None,
                    projs=True, tags=('evoked',)):
        """Add `~mne.Evoked` objects to the report.

        Parameters
        ----------
        evokeds : path-like | instance of mne.Evoked | list of mne.Evoked
            The evoked data to add to the report. Multiple `~mne.Evoked`
            objects – as returned from `mne.read_evokeds` – can be passed as
            a list.
        titles : str | list of str
            The titles corresponding to the evoked data.
        baseline : tuple of float, shape (2,) | None
            Baseline correction to apply. If ``None`` (default), use the
            value set when initializing the report.
        noise_cov : path-like | instance of Covariance | None
            A noise covariance matrix. If provided, will be used to whiten
            the ``evokeds``. If ``None``, will fall back to the ``cov_fname``
            provided upon report creation.
        projs : bool
            Whether to add SSP projector plots, if projectors are present in
            the data.
        tags : collection of str
            Tags to add for later interactive filtering.
        """
        baseline = self.baseline if baseline is None else baseline

        if isinstance(evokeds, Evoked):
            evoked_fname = evokeds.filename
            evokeds = [evokeds]
        elif isinstance(evokeds, list):
            # evoked_fname = evokeds[0].filename
            pass
        else:
            evoked_fname = evokeds
            logger.debug(f'Evoked: Reading {evoked_fname}')
            evokeds = read_evokeds(evoked_fname, verbose=False)

        if baseline is not None:
            [e.apply_baseline(baseline) for e in evokeds]

        if isinstance(titles, str):
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

        tags = tuple(tags)
        for tag in tags:
            if tag not in self.tags:
                self.tags.append(tag)

        for evoked, title in zip(evokeds, titles):
            evoked_htmls = self._render_evoked(
                evoked=evoked,
                noise_cov=noise_cov,
                image_format=self.image_format,
                add_ssp_projs=projs,
                tags=tags
            )

            (joint_html, slider_html, gfp_html, whitened_html,
             ssp_projs_html) = evoked_htmls

            dom_id = self._get_id()
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
                toc_entry_name=title,
                tags=tags,
                html=html,
            )

    def add_raw(self, raw, title, *, psd=True, projs=True, tags=('raw',)):
        """Add `~mne.Raw` objects to the report.

        Parameters
        ----------
        raw : path-like | instance of mne.io.BaseRaw
            The data to add to the report.
        title : str
            The title corresponding to the `~mne.Raw` object.
        psd : bool | None
            Whether to add PSD plots. Overrides the ``raw_psd`` parameter
            passed when initializing the `~mne.Report`. If ``None``, use
            ``raw_psd`` from `~mne.Report` creation.
        projs : bool
            Whether to add SSP projector plots, if projectors are present in
            the data.
        tags : collection of str
            Tags to add for later interactive filtering.
        """
        tags = tuple(tags)
        for tag in tags:
            if tag not in self.tags:
                self.tags.append(tag)

        if psd is None:
            add_psd = dict() if self.raw_psd is True else self.raw_psd
        elif psd is True:
            add_psd = dict()
        else:
            add_psd = False

        htmls  = self._render_raw(
            raw=raw,
            add_psd=add_psd,
            add_ssp_projs=projs,
            image_format=self.image_format,
            tags=tags
        )
        repr_html, psd_img_html, butterfly_img_html, ssp_proj_img_html = htmls
        dom_id = self._get_id()
        html = _html_raw_element(
            repr=repr_html,
            psd=psd_img_html,
            butterfly=butterfly_img_html,
            ssp_projs=ssp_proj_img_html,
            tags=tags,
            title=title,
            id=dom_id,
        )
        self._add_or_replace(
            dom_id=dom_id,
            toc_entry_name=title,
            tags=tags,
            html=html
        )

    def add_stc(self, stc, title, *, subject=None, subjects_dir=None,
                tags=('source-estimate',)):
        """Add a `~mne.SourceEstimate` (STC) to the report.

        Parameters
        ----------
        stc : path-like | instance of mne.SourceEstimate
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
        tags : collection of str
            Tags to add for later interactive filtering.
        """
        tags = tuple(tags)
        for tag in tags:
            if tag not in self.tags:
                self.tags.append(tag)

        html, dom_id = self._render_stc(
            stc=stc,
            title=title,
            tags=tags,
            image_format=self.image_format,
            subject=subject,
            subjects_dir=subjects_dir
        )
        self._add_or_replace(
            dom_id=dom_id,
            toc_entry_name=title,
            tags=tags,
            html=html
        )

    def add_forward(self, forward, title, *, subject=None, subjects_dir=None,
                    tags=('forward-solution',)):
        """Add a forward solution.

        Parameters
        ----------
        forward : instance of mne.Forward | path-like
            The forward solution to add to the report.
        title : str
            The title corresponding to forward solution.
        subject : str | None
            The name of the FreeSurfer subject ``forward`` belongs to. If
            provided, the sensitibity maps of the forward solution will
            be visualized. If ``None``, will use the value of ``subject``
            passed on report creation. If supplied, also pass ``subjects_dir``.
        subjects_dir : path-like | None
            The FreeSurfer ``SUBJECTS_DIR``.
        tags : collection of str
            Tags to add for later interactive filtering.
        """
        tags = tuple(tags)
        for tag in tags:
            if tag not in self.tags:
                self.tags.append(tag)

        html, dom_id = self._render_forward(
            forward=forward, subject=subject, subjects_dir=subjects_dir,
            title=title, image_format=self.image_format, tags=tags
        )
        self._add_or_replace(
            dom_id=dom_id,
            toc_entry_name=title,
            tags=tags,
            html=html
        )

    def add_inverse(self, inverse, title, *, subject=None, subjects_dir=None,
                    trans=None, tags=('inverse-operator',)):
        """Add an inverse operator.

        Parameters
        ----------
        inverse : instance of mne.minimum_norm.InverseOperator | path-like
            The inverse operator to add to the report.
        title : str
            The title corresponding to the inverse operator object.
        subject : str | None
            The name of the FreeSurfer subject ``inverse`` belongs to. If
            provided, the source space the inverse solution is based on will
            be visualized. If ``None``, will use the value of ``subject``
            passed on report creation. If supplied, also pass ``subjects_dir``
            and ``trans``.
        subjects_dir : path-like | None
            The FreeSurfer ``SUBJECTS_DIR``.
        trans : path-like | instance of mne.Transform | None
            The `head -> MRI` transform for ``subject``.
        tags : collection of str
            Tags to add for later interactive filtering.
        """
        tags = tuple(tags)
        for tag in tags:
            if tag not in self.tags:
                self.tags.append(tag)

        if ((subject is not None and trans is None) or
                (trans is not None and subject is None)):
            raise ValueError('Please pass subject AND trans, or neither.')

        html, dom_id = self._render_inverse(
            inverse=inverse, subject=subject, subjects_dir=subjects_dir,
            trans=trans, title=title, image_format=self.image_format, tags=tags
        )
        self._add_or_replace(
            dom_id=dom_id,
            toc_entry_name=title,
            tags=tags,
            html=html
        )

    def add_trans(self, trans, *, info, title, subject=None, subjects_dir=None,
                  tags=('coregistration',)):
        """Add a coregistration visualization to the report.

        Parameters
        ----------
        trans : path-like | instance of mne.Transform
            The `head -> MRI` transform to render.
        info : path-like | instance of mne.Info
            The `~mne.Info` corresponding to ``trans``.
        title : str
            The title to add.
        subject : str | None
            The name of the FreeSurfer subject the ``trans```` belong to. The
            name is not stored with the STC data and therefore needs to be
            specified. If ``None``, will use the value of ``subject`` passed on
            report creation.
        subjects_dir : path-like | None
            The FreeSurfer ``SUBJECTS_DIR``.
        tags : collection of str
            Tags to add for later interactive filtering.
        """
        tags = tuple(tags)
        for tag in tags:
            if tag not in self.tags:
                self.tags.append(tag)

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
            toc_entry_name=title,
            tags=tags,
            html=html
        )

    def add_covariance(self, cov, *, info, title, tags=('covariance',)):
        """Add covariance to the report.

        Parameters
        ----------
        cov : path-like | instance of mne.Covariance
            The `~mne.Covariance` to add to the report.
        info : path-like | instance of mne.Info
            The `~mne.Info` corresponding to ``cov``.
        title : str
            The title corresponding to the `~mne.Covariance` object.
        tags : collection of str
            Tags to add for later interactive filtering.
        """
        tags = tuple(tags)
        for tag in tags:
            if tag not in self.tags:
                self.tags.append(tag)

        htmls = self._render_cov(
            cov=cov,
            info=info,
            image_format=self.image_format,
            tags=tags
        )
        cov_matrix_html, cov_svd_html = htmls

        dom_id = self._get_id()
        html = _html_cov_element(
            matrix=cov_matrix_html,
            svd=cov_svd_html,
            tags=tags,
            title=title,
            id=dom_id
        )
        self._add_or_replace(
            dom_id=dom_id,
            toc_entry_name=title,
            tags=tags,
            html=html
        )

    def add_events(self, events, title, *, event_id=None, sfreq, first_samp=0,
                   tags=('events',)):
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
        tags : collection of str
            Tags to add for later interactive filtering.
        """
        tags = tuple(tags)
        for tag in tags:
            if tag not in self.tags:
                self.tags.append(tag)

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
            toc_entry_name=title,
            tags=tags,
            html=html
        )

    def add_ssp_projs(self, *, info, projs=None, title, tags=('ssp',)):
        """Render SSP projectors.

        Parameters
        ----------
        info : instance of mne.Info | path-like
            An `~mne.Info` structure or the path of a file containing one. This
            is required to create the topographic plots.
        projs : iterable of mne.Projection | path-like | None
            The projection vectors to add to the report. Can be the path to a
            file that will be loaded via `mne.read_proj`. If ``None``, the
            projectors are taken from ``info['projs']``.
        title : str
            The title corresponding to the `~mne.Raw` object.
        tags : collection of str
            Tags to add for later interactive filtering.
        """
        tags = tuple(tags)
        for tag in tags:
            if tag not in self.tags:
                self.tags.append(tag)

        html, dom_id = self._render_ssp_projs(
            info=info, projs=projs, title=title,
            image_format=self.image_format, tags=tags
        )
        self._add_or_replace(
            dom_id=dom_id,
            toc_entry_name=title,
            tags=tags,
            html=html
        )

    def remove(self, caption, section=None):
        """Remove a figure from the report.

        The figure to remove is searched for by its caption. When searching by
        caption, the section label can be specified as well to narrow down the
        search. If multiple figures match the search criteria, the last one
        will be removed.

        Any empty sections will be removed as well.

        Parameters
        ----------
        caption : str
            If set, search for the figure by caption.
        section : str | None
            If set, limit the search to the section with the given label.

        Returns
        -------
        removed_index : int | None
            The integer index of the figure that was removed, or ``None`` if no
            figure matched the search criteria.
        """
        # Construct the search pattern
        pattern = r'^%s-#-.*-#-custom$' % caption

        # Search for figures matching the search pattern, regardless of
        # section
        matches = [i for i, fname_ in enumerate(self.fnames)
                   if re.match(pattern, fname_)]
        if section is not None:
            # Narrow down the search to the given section
            svar = self._sectionvars[section]
            matches = [i for i in matches
                       if self._sectionlabels[i] == svar]
        if len(matches) == 0:
            return None

        # Remove last occurrence
        index = max(matches)

        # Remove the figure
        del self.fnames[index]
        del self._sectionlabels[index]
        del self.html[index]

        # Remove any (now) empty sections.
        # We use a list() to copy the _sectionvars dictionary, since we are
        # removing elements during the loop.
        for section_, sectionlabel_ in list(self._sectionvars.items()):
            if sectionlabel_ not in self._sectionlabels:
                self.tags.remove(section_)
                del self._sectionvars[section_]

        return index

    def _add_or_replace(self, *, toc_entry_name, dom_id, tags, html,
                        replace=False):
        """Append HTML content report, or replace it if it already exists.

        Parameters
        ----------
        toc_entry_name : str
            The entry under which the content shall be listed in the table of
            contents. If it already exists, the content will be replaced.
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

        existing_toc_entry_names = [toc_entry['name']
                                    for toc_entry in self._toc_entries]
        if replace and toc_entry_name in existing_toc_entry_names:
            # Find last occurrence of the figure
            ind = max([i for i, existing in enumerate(existing_toc_entry_names)
                       if existing == toc_entry_name])
            self.html[ind] = html
        else:
            # Append new record
            self.html.append(html)
            toc_entry = TocEntry(
                name=toc_entry_name,
                dom_target_id=dom_id,
                tags=tags
            )
            self._toc_entries.append(toc_entry)

    def _render_code(self, *, code, title, language, tags):
        try:
            if Path(code).exists():
                code = Path(code).read_text()
        except OSError:  # It's most likely a string
            pass

        code = stdlib_html.escape(code)

        dom_id = self._get_id()
        html = _html_code_element(
            tags=tags,
            title=title,
            id=dom_id,
            code=code,
            language=language
        )
        return html, dom_id

    def add_code(self, code, title, *, language='python', tags=('code',)):
        """Add a code snippet (e.g., an analysis script) to the report.

        Parameters
        ----------
        code : path-like | str
            The code to add to the report.
        title : str
            The title corresponding to the code.
        language : str
            The programming language of ``code``. This will be used for syntax
            highlighting. Can be ``'auto'`` to try to auto-detect the language.
        tags : collection of str
            Tags to add for later interactive filtering.
        """
        tags = tuple(tags)
        for tag in tags:
            if tag not in self.tags:
                self.tags.append(tag)

        language = language.lower()
        html, dom_id = self._render_code(
            code=code, title=title, language=language, tags=tags
        )
        self._add_or_replace(
            dom_id=dom_id,
            toc_entry_name=title,
            tags=tags,
            html=html
        )

    def add_sys_info(self, title, *, tags=('mne-sysinfo',)):
        """Add a MNE-Python system information to the report.

        This is a convenience method that captures the output of
        `mne.sys_info` and adds it to the report.

        Parameters
        ----------
        title : str
            The title to assign.
        tags : collection of str
            Tags to add for later interactive filtering.
        """
        tags = tuple(tags)
        for tag in tags:
            if tag not in self.tags:
                self.tags.append(tag)

        with contextlib.redirect_stdout(StringIO()) as f:
            sys_info()

        info = f.getvalue()
        self.add_code(code=info, title=title, language='shell', tags=tags)

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
        if section.lower() not in self.tags:
            self.tags.append(section.lower())
            self._sectionvars[section.lower()] = section.lower()

        figs, captions, comments = self._validate_input(
            figs, captions,
            section.lower(), comments
        )
        image_format = _check_image_format(self, image_format)
        _check_scale(scale)

        htmls = []
        for fig, caption, comment in zip(figs, captions, comments):
            caption = 'custom plot' if caption == '' else caption
            global_id = self._get_id()
            div_klass = self._sectionvars[section.lower()]
            img_klass = self._sectionvars[section.lower()]

            img = _fig_to_img(fig, image_format, scale, auto_close)
            html = image_template.substitute(
                img=img, id=global_id,
                div_klass=div_klass,
                img_klass=img_klass,
                caption=caption,
                show=True,
                image_format=image_format,
                comment=comment
            )
            htmls.append(html)

        global_id = self._get_id()
        html = html_template.substitute(
            div_klass=self._sectionvars[section.lower()],
            id=global_id,
            caption=section,
            html='\n'.join(htmls)
        )
        self._add_or_replace(
            content_id=f'{section}-#-{section.lower()}-#-custom',
            sectionlabel=section.lower(),
            html=html,
            dom_id=global_id,
            replace=replace
        )

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
        if section not in self.tags:
            self.tags.append(section)
            self._sectionvars[section] = section

        fnames, captions, comments = self._validate_input(fnames, captions,
                                                          section, comments)
        _check_scale(scale)

        for fname, caption, comment in zip(fnames, captions, comments):
            caption = 'custom plot' if caption == '' else caption
            sectionvar = self._sectionvars[section]
            global_id = self._get_id()
            div_klass = self._sectionvars[section]
            img_klass = self._sectionvars[section]

            image_format = os.path.splitext(fname)[1][1:]
            image_format = image_format.lower()

            _check_option('image_format', image_format, ['png', 'gif', 'svg'])

            # Convert image to binary string.
            with open(fname, 'rb') as f:
                img = base64.b64encode(f.read()).decode('ascii')
                html = image_template.substitute(img=img, id=global_id,
                                                 image_format=image_format,
                                                 div_klass=div_klass,
                                                 img_klass=img_klass,
                                                 caption=caption,
                                                 width=scale,
                                                 comment=comment,
                                                 show=True)

            self._add_or_replace(
                content_id=f'{caption}-#-{sectionvar}-#-custom',
                sectionvar=sectionvar,
                html=html,
                dom_id=global_id,
                replace=replace
            )

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
        if section not in self.tags:
            self.tags.append(section)
            self._sectionvars[section] = section

        htmls, captions, _ = self._validate_input(htmls, captions, section)
        for html, caption in zip(htmls, captions):
            caption = 'custom plot' if caption == '' else caption
            sectionvar = self._sectionvars[section]
            global_id = self._get_id()
            div_klass = self._sectionvars[section]

            self._add_or_replace(
                content_id=f'{caption}-#-{sectionvar}-#-custom',
                sectionlabel=sectionvar,
                dom_id=global_id,
                html=html_template.substitute(
                    div_klass=div_klass, id=global_id,
                    caption=caption, html=html
                ),
                replace=replace
            )

    @deprecated(extra='Use `Report.add_bem` instead')
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
        if section not in self.tags:
            self.tags.append(section)
            self._sectionvars[section] = section

        width = _ensure_int(width, 'width')
        html = self._render_bem(subject=subject, subjects_dir=subjects_dir,
                                decim=decim, n_jobs=n_jobs, width=width,
                                image_format=self.image_format, tags=tags)

        self._validate_input(html, caption, section)

        global_id = self._get_id()
        html = html_template.substitute(
            html=html,
            id=global_id, div_klass='bem', show=True,
            caption='Boundary Element Model'
        )

        self._add_or_replace(
            content_id=f'{caption}-#-{section}-#-custom',
            sectionlabel=section,
            dom_id=global_id,
            html=html,
            replace=replace
        )

    def add_bem(self, subject, title, *, subjects_dir=None, decim=2, width=512,
                n_jobs=1, tags=('bem',)):
        """Render a visualization of the boundary element model (BEM) surfaces.

        Parameters
        ----------
        subject : str
            The FreeSurfer subject name.
        title : str
            The title corresponding to the `~mne.Raw` object.
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
        tags : collection of str
            Tags to add for later interactive filtering.
        """
        tags = tuple(tags)
        for tag in tags:
            if tag not in self.tags:
                self.tags.append(tag)

        width = _ensure_int(width, 'width')
        html = self._render_bem(subject=subject, subjects_dir=subjects_dir,
                                decim=decim, n_jobs=n_jobs, width=width,
                                image_format=self.image_format, tags=tags)

        dom_id = self._get_id()
        html = _html_element(
            div_klass='bem',
            id=dom_id,
            tags=tags,
            title=title,
            html=html,
        )
        self._add_or_replace(
            dom_id=dom_id,
            toc_entry_name=title,
            tags=tags,
            html=html
        )

    def _render_slider(self, *, figs, title, captions, start_idx, image_format,
                       tags, klass):
        if len(figs) != len(captions):
            raise ValueError('Captions must be the same length as the '
                             'number of slides.')
        images = [_fig_to_img(fig=fig, image_format=image_format)
                  for fig in figs]

        dom_id = self._get_id()
        html = _html_slider_element(
            id=dom_id,
            title=title,
            captions=captions,
            tags=tags,
            images=images,
            image_format=image_format,
            start_idx=start_idx
        )

        return html, dom_id

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
            Must have at least 2 elements.
        captions : list of str | list of float | None
            A list of captions to the figures. If float, a str will be
            constructed as ``%f s``. If None, it will default to
            ``Data slice %d``.
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
        if section not in self.tags:
            self.tags.append(section)
            self._sectionvars[section] = section

        sectionvar = self._sectionvars[section]

        html = self._render_slider(
            figs=figs, captions=captions, section=section, title=title,
            scale=scale, image_format=image_format,
            auto_close=auto_close)

        self._add_or_replace(
            sectionlabel=sectionvar,
            dom_id=global_id,
            html=html,
            replace=replace)

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

    @verbose
    def parse_folder(self, data_path, pattern=None, n_jobs=1, mri_decim=2,
                     sort_sections=True, on_error='warn', image_format=None,
                     render_bem=True, verbose=None):
        r"""Render all the files in the folder.

        Parameters
        ----------
        data_path : str
            Path to the folder containing data whose HTML report will be
            created.
        pattern : None | str | list of str
            Filename pattern(s) to include in the report.
            Example: [\*raw.fif, \*ave.fif] will include Raw as well as Evoked
            files. If ``None``, include all supported file formats.

            .. versionchanged:: 0.23
               Include supported non-FIFF files by default.
        %(n_jobs)s
        mri_decim : int
            Use this decimation factor for generating MRI/BEM images
            (since it can be time consuming).
        sort_sections : bool
            If True, sort sections in the order: raw -> events -> epochs
             -> evoked -> covariance -> trans -> mri -> forward -> inverse.
        on_error : str
            What to do if a file cannot be rendered. Can be 'ignore',
            'warn' (default), or 'raise'.
        %(report_image_format)s

            .. versionadded:: 0.15
        render_bem : bool
            If True (default), try to render the BEM.

            .. versionadded:: 0.16
        %(verbose_meth)s
        """
        _validate_type(data_path, 'path-like', 'data_path')
        data_path = str(data_path)
        image_format = _check_image_format(self, image_format)
        _check_option('on_error', on_error, ['ignore', 'warn', 'raise'])
        self._sort = sort_sections

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
            fnames.extend(sorted(_recursive_search(self.data_path, p)))

        if not fnames and not render_bem:
            raise RuntimeError(f'No matching files found in {self.data_path}')

        # For split files, only keep the first one.
        fnames_to_remove = []
        for fname in fnames:
            if _endswith(fname, ('raw', 'sss', 'meg')):
                kwargs = dict(fname=fname, preload=False)
                if fname.endswith(('.fif', '.fif.gz')):
                    kwargs['allow_maxshield'] = True
                inst = read_raw(**kwargs)
            else:
                continue

            if len(inst.filenames) > 1:
                fnames_to_remove.extend(inst.filenames[1:])

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
        baseline = self.baseline

        # render plots in parallel; check that n_jobs <= # of files
        logger.info('Iterating over %s potential files (this may take some '
                    'time)' % len(fnames))
        use_jobs = min(n_jobs, max(1, len(fnames)))
        parallel, p_fun, _ = parallel_func(_iterate_files, use_jobs)
        parallel(p_fun(self, fname, info, cov, baseline, sfreq, on_error,
                       image_format, self.data_path)
                 for fname in np.array_split(fnames, use_jobs))

        # # combine results from n_jobs discarding plots not rendered
        # self.html = [html for html in sum(htmls, []) if html is not None]
        # self.fnames = [fname for fname in sum(report_fnames, []) if
        #                fname is not None]
        # self._sectionlabels = [slabel for slabel in
        #                        sum(report_sectionlabels, [])
        #                        if slabel is not None]

        # # find unique section labels
        # self.tags = sorted(set(self._sectionlabels))
        # self._sectionvars = dict(zip(self.tags, self.tags))

        # # render mri
        # if render_bem:
        #     if self.subjects_dir is not None and self.subject is not None:
        #         logger.info('Rendering BEM')
        #         self.fnames.append('bem')
        #         self.add_bem_to_section(
        #             self.subject, decim=mri_decim, n_jobs=n_jobs,
        #             subjects_dir=self.subjects_dir)
        #     else:
        #         warn('`subjects_dir` and `subject` not provided. Cannot '
        #              'render MRI and -trans.fif(.gz) files.')

    def _get_state_params(self):
        """Obtain all fields that are in the state dictionary of this object.

        Returns
        -------
        non_opt_params : list of str
            All parameters that must be present in the state dictionary.
        opt_params : list of str
            All parameters that are optionally present in the state dictionary.
        """
        # Note: self._fname is not part of the state
        return (['baseline', 'cov_fname', 'fnames', 'html', 'include',
                 'image_format', 'info_fname', '_initial_id', 'raw_psd',
                 '_sectionlabels', 'sections', '_sectionvars', 'projs',
                 '_sort_sections', 'subjects_dir', 'subject', 'title',
                 'verbose'],
                ['data_path', 'lang', '_sort'])

    def __getstate__(self):
        """Get the state of the report as a dictionary."""
        state = dict()
        non_opt_params, opt_params = self._get_state_params()
        for param in non_opt_params:
            state[param] = getattr(self, param)
        for param in opt_params:
            if hasattr(self, param):
                state[param] = getattr(self, param)
        return state

    def __setstate__(self, state):
        """Set the state of the report."""
        non_opt_params, opt_params = self._get_state_params()
        for param in non_opt_params:
            setattr(self, param, state[param])
        for param in opt_params:
            if param in state:
                setattr(self, param, state[param])
        return state

    @verbose
    def save(self, fname=None, open_browser=True, overwrite=False, *,
             verbose=None):
        """Save the report and optionally open it in browser.

        Parameters
        ----------
        fname : str | None
            File name of the report. If the file name ends in '.h5' or '.hdf5',
            the report is saved in HDF5 format, so it can later be loaded again
            with :func:`open_report`. If the file name ends in anything else,
            the report is rendered to HTML. If ``None``, the report is saved to
            'report.html' in the current working directory.
            Defaults to ``None``.
        open_browser : bool
            When saving to HTML, open the rendered HTML file browser after
            saving if True. Defaults to True.
        %(overwrite)s
        %(verbose_meth)s

        Returns
        -------
        fname : str
            The file name to which the report was saved.
        """
        if fname is None:
            if not hasattr(self, 'data_path'):
                self.data_path = os.getcwd()
                warn('`data_path` not provided. Using %s instead'
                     % self.data_path)
            fname = op.realpath(op.join(self.data_path, 'report.html'))
        else:
            fname = op.realpath(fname)

        if not overwrite and op.isfile(fname):
            msg = ('Report already exists at location %s. '
                   'Overwrite it (y/[n])? '
                   % fname)
            answer = _safe_input(msg, alt='pass overwrite=True')
            if answer.lower() == 'y':
                overwrite = True

        _, ext = op.splitext(fname)
        is_hdf5 = ext.lower() in ['.h5', '.hdf5']

        if overwrite or not op.isfile(fname):
            logger.info('Saving report to location %s' % fname)

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

                toc_html = _html_toc_element(toc_entries=self._toc_entries)

                with warnings.catch_warnings(record=True):
                    warnings.simplefilter('ignore')
                    footer_html = _html_footer_element(
                        mne_version=MNE_VERSION,
                        date=time.strftime("%B %d, %Y"),
                        current_year=time.strftime("%Y")
                    )

                self.html = [header_html, toc_html, *self.html, footer_html]

                # Writing to disk may fail. However, we need to make sure that
                # the TOC and footer are removed regardless, otherwise they
                # will be duplicated when the user attempts to save again.
                try:
                    # Write HTML
                    with open(fname, 'w', encoding='utf-8') as f:
                        f.write(_fix_global_ids(''.join(self.html)))
                finally:
                    self.html.pop(0)
                    self.html.pop(0)
                    self.html.pop()

        building_doc = os.getenv('_MNE_BUILDING_DOC', '').lower() == 'true'
        if open_browser and not is_hdf5 and not building_doc:
            webbrowser.open_new_tab('file://' + fname)

        self.fname = fname
        return fname

    def __enter__(self):
        """Do nothing when entering the context block."""
        return self

    def __exit__(self, type, value, traceback):
        """Save the report when leaving the context block."""
        if self._fname is not None:
            self.save(self._fname, open_browser=False, overwrite=True)

    @staticmethod
    def _gen_caption(prefix, fname, data_path, suffix=''):
        if data_path is not None:
            fname = op.relpath(fname, start=data_path)

        caption = f'{prefix}: {fname} {suffix}'
        return caption.strip()

    # @verbose
    # def _render_toc(self, verbose=None):
    #     """Render the Table of Contents."""
    #     logger.info('Rendering : Table of Contents')

    #     # Reorder self.sections to reflect natural ordering
    #     # if self._sort_sections:
    #     #     sections = list(set(self.tags) & set(SECTION_ORDER))
    #     #     custom = [section for section in self.tags if section
    #     #               not in SECTION_ORDER]
    #     #     order = [sections.index(section) for section in SECTION_ORDER if
    #     #              section in sections]
    #     #     self.tags = np.array(sections)[order].tolist() + custom


    #     toc_html = _html_toc_element(toc_entries=self._toc_entries)
    #     lang = getattr(self, 'lang', 'en-us')
    #     sections = [section if section != 'mri' else 'MRI'
    #                 for section in self.tags]
    #     html_header = _html_header_element(
    #         title=self.title, include=self.include, lang=lang,
    #         tags=sections, js=JAVASCRIPT, css=CSS, mne_logo_img=mne_logo
    #     )
    #     self.html.insert(0, html_header)  # Insert header at position 0
    #     self.html.insert(1, toc_html)  # insert TOC

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
        figs = _get_mri_contour_figs(
            sl=sl, n_jobs=n_jobs, mri_fname=mri_fname, surfaces=surfaces,
            orientation=orientation, img_output=True, src=None, show=False,
            show_orientation=True, width=width
        )

        # Render the slider
        captions = [f'Slice index: {i * decim}' for i in range(len(figs))]
        start_idx = int(round(len(figs) / 2))
        html, slider_id = self._render_slider(
            figs=figs,
            captions=captions,
            title=orientation,
            klass='bem slider',
            image_format=image_format,
            start_idx=start_idx,
            tags=tags
        )
        return html

    def _render_raw(self, *, raw, add_psd, add_ssp_projs, image_format, tags):
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
        dom_id = self._get_id()
        repr_html = _html_element(
            div_klass='raw',
            id=dom_id,
            tags=tags,
            title='Info',
            html=raw._repr_html_()
        )

        # Butterfly plot
        dom_id = self._get_id()
        # Only keep "bad" annotations
        raw_copy = raw.copy()
        annots_to_remove_idx = []
        for idx, annotation in enumerate(raw.annotations):
            if not annotation['description'].lower().startswith('bad'):
                annots_to_remove_idx.append(idx)
        raw_copy.annotations.delete(annots_to_remove_idx)
        fig = raw_copy.plot(
            butterfly=True,
            show_scrollbars=False,
            duration=raw.times[-1],
            decim=10,
            show=False
        )
        fig.tight_layout()
        img = _fig_to_img(fig=fig, image_format=image_format)
        butterfly_img_html = _html_image_element(
            img=img, div_klass='raw', img_klass='raw',
            title='Time course', caption=None, show=True,
            image_format=image_format, id=dom_id, tags=tags
        )
        del raw_copy

        # PSD
        if isinstance(add_psd, dict):
            dom_id = self._get_id()
            if raw.info['lowpass'] is not None:
                fmax = raw.info['lowpass'] + 15
            else:
                fmax = np.inf

            fig = raw.plot_psd(fmax=fmax, show=False, **add_psd)
            fig.tight_layout()
            img = _fig_to_img(fig, image_format=image_format)
            psd_img_html = _html_image_element(
                img=img, div_klass='raw', img_klass='raw',
                title='PSD', caption=None, show=True,
                image_format=image_format, id=dom_id, tags=tags
            )
        else:
            psd_img_html = ''

        # SSP projectors
        if add_ssp_projs:
            ssp_projs_html, _ = self._render_ssp_projs(
                info=raw, projs=None, title='SSP Projectors',
                image_format=image_format, tags=tags
            )
        else:
            ssp_projs_html = ''

        return [repr_html, psd_img_html, butterfly_img_html, ssp_projs_html]

    def _render_ssp_projs(self, *, info, projs, title, image_format, tags):
        if isinstance(info, Info):  # no-op
            pass
        elif hasattr(info, 'info'):  # try to get the file name
            info = info.info
            if isinstance(info, BaseRaw):
                fname = info.filenames[0]
            elif isinstance(info, (Evoked, BaseEpochs)):
                fname = info.filename
            else:
                fname = ''
        else:  # read from a file
            fname = info
            info = read_info(fname, verbose=False)

        if projs is None:
            projs = info['projs']
        elif not isinstance(projs, list):
            fname = projs
            projs = read_proj(fname)

        fig = plot_projs_topomap(
            projs=projs, info=info, colorbar=True, vlim='joint',
            show=False
        )
        fig.set_size_inches((6, 4))
        fig.tight_layout()

        img = _fig_to_img(fig=fig, image_format=image_format)

        dom_id = self._get_id()
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

        repr_string = repr(forward).replace(' | ', '\n ')
        info_string = repr_string + '\n\n' + repr(forward['info'])
        info_html, _  = self._render_code(code=info_string, title='Info',
                                          language='plaintext', tags=tags)

        # Render sensitivity maps
        if subject is not None:

            sensitivity_maps_html = ''  # XXX
        else:
            sensitivity_maps_html = ''

        dom_id = self._get_id()
        html = _html_forward_sol_element(
            id=dom_id,
            info=info_html,
            sensitivity_maps=sensitivity_maps_html,
            title=title,
            tags=tags
        )
        return html, dom_id

    def _render_inverse(self, *, inverse, subject, subjects_dir, trans, title,
                        image_format, tags):
        """Render inverse operator."""
        if not isinstance(inverse, InverseOperator):
            inverse = read_inverse_operator(inverse)

        if trans is not None and not isinstance(trans, Transform):
            trans = read_trans(trans)

        subject = self.subject if subject is None else subject
        subjects_dir = (self.subjects_dir if subjects_dir is None
                        else subjects_dir)

        repr_string = repr(inverse).replace(' | ', '\n ')
        info_string = repr_string + '\n\n' + repr(inverse['info'])
        info_html, _  = self._render_code(code=info_string, title='Info',
                                          language='plaintext', tags=tags)

        # Render source space
        if subject is not None and trans is not None:
            src = inverse['src']

            fig = plot_alignment(
                subject=subject,
                subjects_dir=subjects_dir,
                trans=trans,
                surfaces='white',
                src=src
            )
            set_3d_view(fig, focalpoint=(0., 0., 0.06))
            img = _fig_to_img(fig=fig, image_format=image_format)

            dom_id = self._get_id()
            src_img_html = _html_image_element(
                img=img,
                div_klass='inverse-operator source-space',
                img_klass='inverse-operator source-space',
                title='Source space', caption=None, show=True,
                image_format=image_format, id=dom_id,
                tags=tags
            )
        else:
            src_img_html = ''

        dom_id = self._get_id()
        html = _html_inverse_op_element(
            id=dom_id,
            info=info_html,
            source_space=src_img_html,
            title=title,
            tags=tags,
        )
        return html, dom_id

    def _render_evoked_joint(self, evoked, ch_types, image_format, tags):
        ch_type_to_caption_map = {
            'mag': 'magnetometers',
            'grad': 'gradiometers',
            'eeg': 'EEG'
        }

        htmls = []
        for ch_type in ch_types:
            fig = evoked.copy().pick(ch_type).plot_joint(
                ts_args=dict(gfp=True),
                title=None,
                show=False
            )

            img = _fig_to_img(fig, image_format)
            title = f'Time course ({ch_type_to_caption_map[ch_type]})'
            dom_id = self._get_id()

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

    def _render_evoked_topo_slider(self, *, evoked, ch_types, image_format,
                                   tags):
        import matplotlib.pyplot as plt

        times = np.linspace(start=evoked.tmin, stop=evoked.tmax, num=21)
        t_zero_idx = np.abs(times).argmin()  # index closest to zero

        # global min and max values for each channel type
        scalings = dict(eeg=1e6, grad=1e13, mag=1e15)

        vmax = dict()
        vmin = dict()
        for ch_type in ch_types:
            vmax[ch_type] = (np.abs(evoked.copy()
                                    .pick(ch_type)
                                    .data)
                             .max()) * scalings[ch_type]
            if ch_type == 'grad':
                vmin[ch_type] = 0
            else:
                vmin[ch_type] = -vmax[ch_type]

        figs = []

        for t in times:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action='ignore',
                    message='More than 20 figures have been opened',
                    category=RuntimeWarning
                )

                # topomaps + color bars
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
                        times=[t], ch_type=ch_type,
                        vmin=vmin[ch_type], vmax=vmax[ch_type],
                        axes=ch_type_ax_map[ch_type]
                    )
                    ch_type_ax_map[ch_type][0].set_title(ch_type)
                fig.tight_layout()
                figs.append(fig)

        captions = [f'Time point: {round(t, 3):0.3f} s' for t in times]
        html = self._render_slider(
            figs=figs,
            captions=captions,
            title='Topographies',
            klass='evoked evoked-topo slider',
            image_format=image_format,
            start_idx=t_zero_idx,
            tags=tags
        )

        return html

    def _render_evoked_gfp(self, evoked, ch_types, image_format, tags):
        # Make legend labels shorter by removing the multiplicative factors
        pattern = r'\d\.\d* × '
        label = evoked.comment
        if label is None:
            label = ''
        for match in re.findall(pattern=pattern, string=label):
            label = label.replace(match, '')

        dom_id = self._get_id()

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(ch_types), 1, sharex=True)
        if len(ch_types) == 1:
            ax = [ax]
        for idx, ch_type in enumerate(ch_types):
            plot_compare_evokeds(
                evokeds={label: evoked.copy().pick(ch_type)},
                ci=None, truncate_xaxis=False,
                truncate_yaxis=False, legend='lower right',
                axes=ax[idx], show=False
            )
            ax[idx].set_title(ch_type)

            # Only show legend for the first subplot
            if idx > 0:
                ax[idx].get_legend().remove()

            # Hide x axis label for all but the last subplot
            if idx < len(ch_types) - 1:
                ax[idx].set_xlabel(None)

        fig.tight_layout()
        img = _fig_to_img(fig, image_format)
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
        dom_id = self._get_id()
        fig = evoked.plot_white(
            noise_cov=noise_cov,
            show=False
        )
        fig.tight_layout()
        img = _fig_to_img(fig, image_format=image_format)
        title = 'Whitened'

        html = _html_image_element(
            img=img, id=dom_id, div_klass='evoked',
            img_klass='evoked evoked-whitened', title=title, caption=None,
            show=True, image_format=image_format, tags=tags
        )
        return html

    def _render_evoked(self, evoked, noise_cov, add_ssp_projs, image_format,
                       tags):
        def _get_ch_types(ev):
            has_types = []
            if len(pick_types(ev.info, meg=False, eeg=True)) > 0:
                has_types.append('eeg')
            if len(pick_types(ev.info, meg='grad', eeg=False,
                              ref_meg=False)) > 0:
                has_types.append('grad')
            if len(pick_types(ev.info, meg='mag', eeg=False)) > 0:
                has_types.append('mag')
            return has_types

        ch_types = _get_ch_types(evoked)
        html_joint = self._render_evoked_joint(
            evoked=evoked, ch_types=ch_types,
            image_format=image_format, tags=tags
        )
        html_slider, _ = self._render_evoked_topo_slider(
            evoked=evoked, ch_types=ch_types,
            image_format=image_format,
            tags=tags
        )
        html_gfp = self._render_evoked_gfp(
            evoked=evoked, ch_types=ch_types, image_format=image_format,
            tags=tags
        )

        if noise_cov is not None:
            html_whitened = self._render_evoked_whitened(
                evoked=evoked,
                noise_cov=noise_cov,
                image_format=image_format,
                tags=tags
            )
        else:
            html_whitened = ''

        # SSP projectors
        if add_ssp_projs:
            html_ssp_projs, _ = self._render_ssp_projs(
                info=evoked, projs=None, title='SSP Projectors',
                image_format=image_format, tags=tags
            )
        else:
            html_ssp_projs = ''

        logger.debug('Evoked: done')
        return html_joint, html_slider, html_gfp, html_whitened, html_ssp_projs

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

        dom_id = self._get_id()
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

    def _render_epochs(self, *, epochs, add_ssp_projs, image_format, tags):
        """Render epochs."""
        if isinstance(epochs, BaseEpochs):
            fname = epochs.filename
        else:
            fname = epochs
            epochs = read_epochs(fname)

        # Summary table
        dom_id = self._get_id()
        repr_html = _html_element(
            div_klass='epochs',
            id=dom_id,
            tags=tags,
            title='Info',
            html=epochs._repr_html_()
        )

        # Drop log
        dom_id = self._get_id()
        img = _fig_to_img(epochs.plot_drop_log, image_format,
                          subject=self.subject, show=False)
        drop_log_img_html = _html_image_element(
            img=img, id=dom_id, div_klass='epochs', img_klass='epochs',
            show=True, image_format=image_format, title='Drop log',
            caption=None, tags=tags
        )

        # PSD
        dom_id = self._get_id()
        if epochs.info['lowpass'] is not None:
            fmax = epochs.info['lowpass'] + 15
        else:
            fmax = np.inf

        fig = epochs.plot_psd(fmax=fmax, show=False)
        fig.tight_layout()
        img = _fig_to_img(fig=fig, image_format=image_format)
        psd_img_html = _html_image_element(
            img=img, id=dom_id, div_klass='epochs', img_klass='epochs',
            show=True, image_format=image_format, title='PSD', caption=None,
            tags=tags
        )

        # SSP projectors
        if add_ssp_projs:
            ssp_projs_html, _ = self._render_ssp_projs(
                info=epochs, projs=None, title='SSP Projectors',
                image_format=image_format, tags=tags
            )
        else:
            ssp_projs_html = ''

        return repr_html, drop_log_img_html, psd_img_html, ssp_projs_html

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
            dom_id = self._get_id()
            img = _fig_to_img(fig, image_format)
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
                      meg=['helmet', 'sensors'],
                      coord_frame='mri')
        try:
            img, caption = _iterate_trans_views(
                function=plot_alignment, surfaces=['head-dense'], **kwargs
            )
        except IOError:
            img, caption = _iterate_trans_views(
                function=plot_alignment, surfaces=['head'], **kwargs
            )

        dom_id = self._get_id()
        html = _html_image_element(
            img=img, id=dom_id, div_klass='trans',
            img_klass='trans', title=title, caption=caption,
            show=True, image_format='png', tags=tags
        )
        return html, dom_id

    def _render_stc(self, *, stc, title, subject, subjects_dir, image_format,
                    tags):
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

        n_time_points = min(len(stc.times), 51)
        if n_time_points == 1:  # only a single time point
            times = stc.times
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

        if get_3d_backend() is not None:
            brain = stc.plot(
                views=('lateral', 'medial'),
                hemi='split',
                backend='pyvistaqt',
                time_viewer=True,
                subject=subject,
                subjects_dir=subjects_dir,
                size=(450, 450),
                background='white'
            )
            brain.toggle_interface()
            brain.time_actor.SetVisibility(0)  # don't print the time
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
                    fig.tight_layout()
                    figs.append(fig)
                else:
                    fig_lh = plt.figure()
                    fig_rh = plt.figure()

                    brain_lh = stc.plot(views='lat', hemi='lh',
                                        initial_time=t,
                                        backend='matplotlib',
                                        subject=subject,
                                        subjects_dir=subjects_dir,
                                        figure=fig_lh)
                    brain_rh = stc.plot(views='lat', hemi='rh',
                                        initial_time=t,
                                        subject=subject,
                                        subjects_dir=subjects_dir,
                                        backend='matplotlib',
                                        figure=fig_rh)
                    fig_lh.tight_layout()  # TODO is this necessary?
                    fig_rh.tight_layout()  # TODO is this necessary?
                    figs.append(brain_lh)
                    figs.append(brain_rh)

        captions = [f'Time point: {round(t, 3):0.3f} s' for t in times]
        html, dom_id = self._render_slider(
            figs=figs,
            captions=captions,
            title=title,
            klass='stc slider',
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
        for orientation in _BEM_VIEWS:
            html = self._render_one_bem_axis(
                mri_fname=mri_fname, surfaces=surfaces,
                orientation=orientation, decim=decim, n_jobs=n_jobs,
                width=width, image_format=image_format,
                tags=tags
            )
            htmls.append(html)
        return '\n'.join(htmls)


def _clean_tag(tag):
    # Remove any whitespace characters
    tag_cleaned = re.sub(r'[\s*]', '', tag)
    return tag_cleaned


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


def _fix_global_ids(html):
    """Fix the global_ids after reordering in _render_toc()."""
    html = re.sub(r'id="\d+"', 'id="###"', html)
    global_id = 1
    while len(re.findall('id="###"', html)) > 0:
        html = re.sub('id="###"', 'id="%s"' % global_id, html, count=1)
        global_id += 1
    return html


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
            if (isinstance(report, Report) and hasattr(report, 'fname') and
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


def _iterate_files(
    report: Report,
    fnames,
    cov,
    sfreq,
    on_error,
):
    """Parallel process in batch mode."""
    for fname in fnames:
        logger.info(
            f"Rendering : {op.join('…' + report.data_path[-20:], fname)}"
        )

        title = Path(fname).name
        try:
            if _endswith(fname, ['raw', 'sss', 'meg', 'nirs']):
                report.add_raw(raw=fname, title=title, psd=report.raw_psd)
            elif _endswith(fname, 'fwd'):
                report.add_forward(
                    forward=fname, title=title, subject=report.subject,
                    subjects_dir=report.subjects_dir
                )
            elif _endswith(fname, 'inv'):
                # XXX if we pass trans, we can plot the source space, too…
                report.add_inverse(inverse=fname, title=title)
            elif _endswith(fname, 'ave'):
                evokeds = read_evokeds(fname)
                titles = [
                    f'{Path(fname).name}: {e.comment}'
                    for e in evokeds
                ]
                report.add_evokeds(evokeds=fname, titles=titles, noise_cov=cov)
            elif _endswith(fname, 'eve'):
                if report.info_fname is not None:
                    sfreq = read_info(report.info_fname)['sfreq']
                else:
                    sfreq = None
                report.add_events(events=fname, title=title, sfreq=sfreq)
            elif _endswith(fname, 'epo'):
                report.add_epochs(epochs=fname, title=title)
            elif _endswith(fname, 'cov') and report.info_fname is not None:
                report.add_covariance(cov=fname, info=report.info_fname,
                                      title=title)
            elif _endswith(fname, 'proj') and report.info_fname is not None:
                report.add_ssp_projs(info=report.info_fname, projs=fname,
                                     title=title)
            elif (_endswith(fname, 'trans') and
                  report.info_fname is not None and
                  report.subjects_dir is not None and
                  report.subject is not None):
                report.add_trans(
                    trans=fname, info=report.info_fname,
                    subject=report.subject, subjects_dir=report.subjects_dir,
                    title=title
                )
        except Exception as e:
            if on_error == 'warn':
                warn('Failed to process file %s:\n"%s"' % (fname, e))
            elif on_error == 'raise':
                raise

    return report
