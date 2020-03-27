"""Generate self-contained HTML reports from MNE objects."""

# Authors: Alex Gramfort <alexandre.gramfort@inria.fr>
#          Mainak Jas <mainak@neuro.hut.fi>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import base64
from io import BytesIO
import os
import os.path as op
import fnmatch
import re
import codecs
from shutil import copyfile
import time
from glob import glob
import warnings
import webbrowser
import numpy as np

from . import read_evokeds, read_events, pick_types, read_cov
from .fixes import _get_img_fdata
from .io import read_raw_fif, read_info
from .io.pick import _DATA_CH_TYPES_SPLIT
from .utils import (logger, verbose, get_subjects_dir, warn,
                    fill_doc, _check_option)
from .viz import plot_events, plot_alignment, plot_cov
from .viz._3d import _plot_mri_contours
from .forward import read_forward_solution
from .epochs import read_epochs
from .minimum_norm import read_inverse_operator
from .parallel import parallel_func, check_n_jobs

from .externals.tempita import HTMLTemplate, Template
from .externals.h5io import read_hdf5, write_hdf5

VALID_EXTENSIONS = ['raw.fif', 'raw.fif.gz', 'sss.fif', 'sss.fif.gz',
                    '-eve.fif', '-eve.fif.gz', '-cov.fif', '-cov.fif.gz',
                    '-trans.fif', '-trans.fif.gz', '-fwd.fif', '-fwd.fif.gz',
                    '-epo.fif', '-epo.fif.gz', '-inv.fif', '-inv.fif.gz',
                    '-ave.fif', '-ave.fif.gz', 'T1.mgz', 'meg.fif']
SECTION_ORDER = ['raw', 'events', 'epochs', 'evoked', 'covariance', 'trans',
                 'mri', 'forward', 'inverse']


###############################################################################
# PLOTTING FUNCTIONS

def _ndarray_to_fig(img):
    """Convert to MPL figure, adapted from matplotlib.image.imsave."""
    figsize = np.array(img.shape[:2][::-1]) / 100.
    fig = _figure_agg(dpi=100, figsize=figsize, frameon=False)
    fig.figimage(img)
    return fig


def _fig_to_img(fig, image_format='png', scale=None, **kwargs):
    """Plot figure and create a binary image."""
    # fig can be ndarray, mpl Figure, Mayavi Figure, or callable that produces
    # a mpl Figure
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    if isinstance(fig, np.ndarray):
        fig = _ndarray_to_fig(fig)
    elif callable(fig):
        plt.close('all')
        fig = fig(**kwargs)
    elif not isinstance(fig, Figure):
        from .viz.backends.renderer import backend, MNE_3D_BACKEND_TESTING
        backend._check_3d_figure(figure=fig)
        if not MNE_3D_BACKEND_TESTING:
            img = backend._take_3d_screenshot(figure=fig)
        else:  # Testing mode
            img = np.zeros((2, 2, 3))

        backend._close_3d_figure(figure=fig)
        fig = _ndarray_to_fig(img)

    output = BytesIO()
    if scale is not None:
        _scale_mpl_figure(fig, scale)
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


def _scale_mpl_figure(fig, scale):
    """Magic scaling helper.

    Keeps font-size and artist sizes constant
    0.5 : current font - 4pt
    2.0 : current font + 4pt

    XXX it's unclear why this works, but good to go for most cases
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


def _figs_to_mrislices(sl, n_jobs, **kwargs):
    import matplotlib.pyplot as plt
    plt.close('all')
    use_jobs = min(n_jobs, max(1, len(sl)))
    parallel, p_fun, _ = parallel_func(_plot_mri_contours, use_jobs)
    outs = parallel(p_fun(slices=s, **kwargs)
                    for s in np.array_split(sl, use_jobs))
    for o in outs[1:]:
        outs[0] += o
    return outs[0]


def _iterate_trans_views(function, **kwargs):
    """Auxiliary function to iterate over views in trans fig."""
    import matplotlib.pyplot as plt
    from .viz.backends.renderer import backend, MNE_3D_BACKEND_TESTING

    fig = function(**kwargs)
    backend._check_3d_figure(fig)

    views = [(90, 90), (0, 90), (0, -90)]
    fig2, axes = plt.subplots(1, len(views))
    for view, ax in zip(views, axes):
        backend._set_3d_view(fig, azimuth=view[0], elevation=view[1],
                             focalpoint=None, distance=None)
        if not MNE_3D_BACKEND_TESTING:
            im = backend._take_3d_screenshot(figure=fig)
        else:  # Testing mode
            im = np.zeros((2, 2, 3))
        ax.imshow(im)
        ax.axis('off')

    backend._close_all()
    img = _fig_to_img(fig2, image_format='png')
    return img

###############################################################################
# TOC FUNCTIONS


def _is_bad_fname(fname):
    """Identify bad file naming patterns and highlight them in the TOC."""
    if fname.endswith('(whitened)'):
        fname = fname[:-11]

    if not fname.endswith(tuple(VALID_EXTENSIONS + ['bem', 'custom'])):
        return 'red'
    else:
        return ''


def _get_fname(fname):
    """Get fname without -#-."""
    if '-#-' in fname:
        fname = fname.split('-#-')[0]
    else:
        fname = op.basename(fname)
    fname = ' ... %s' % fname
    return fname


def _get_toc_property(fname):
    """Assign class names to TOC elements to allow toggling with buttons."""
    if fname.endswith(('-eve.fif', '-eve.fif.gz')):
        div_klass = 'events'
        tooltip = fname
        text = op.basename(fname)
    elif fname.endswith(('-ave.fif', '-ave.fif.gz')):
        div_klass = 'evoked'
        tooltip = fname
        text = op.basename(fname)
    elif fname.endswith(('-cov.fif', '-cov.fif.gz')):
        div_klass = 'covariance'
        tooltip = fname
        text = op.basename(fname)
    elif fname.endswith(('raw.fif', 'raw.fif.gz',
                         'sss.fif', 'sss.fif.gz', 'meg.fif')):
        div_klass = 'raw'
        tooltip = fname
        text = op.basename(fname)
    elif fname.endswith(('-trans.fif', '-trans.fif.gz')):
        div_klass = 'trans'
        tooltip = fname
        text = op.basename(fname)
    elif fname.endswith(('-fwd.fif', '-fwd.fif.gz')):
        div_klass = 'forward'
        tooltip = fname
        text = op.basename(fname)
    elif fname.endswith(('-inv.fif', '-inv.fif.gz')):
        div_klass = 'inverse'
        tooltip = fname
        text = op.basename(fname)
    elif fname.endswith(('-epo.fif', '-epo.fif.gz')):
        div_klass = 'epochs'
        tooltip = fname
        text = op.basename(fname)
    elif fname.endswith(('.nii', '.nii.gz', '.mgh', '.mgz')):
        div_klass = 'mri'
        tooltip = 'MRI'
        text = 'MRI'
    elif fname.endswith(('bem')):
        div_klass = 'mri'
        tooltip = 'MRI'
        text = 'MRI'
    elif fname.endswith('(whitened)'):
        div_klass = 'evoked'
        tooltip = fname
        text = op.basename(fname[:-11]) + '(whitened)'
    else:
        div_klass = fname.split('-#-')[1]
        tooltip = fname.split('-#-')[0]
        text = fname.split('-#-')[0]

    return div_klass, tooltip, text


def _iterate_files(report, fnames, info, cov, baseline, sfreq, on_error,
                   image_format):
    """Parallel process in batch mode."""
    htmls, report_fnames, report_sectionlabels = [], [], []

    def _update_html(html, report_fname, report_sectionlabel):
        """Update the lists above."""
        htmls.append(html)
        report_fnames.append(report_fname)
        report_sectionlabels.append(report_sectionlabel)

    for fname in fnames:
        logger.info("Rendering : %s"
                    % op.join('...' + report.data_path[-20:],
                              fname))
        try:
            if fname.endswith(('raw.fif', 'raw.fif.gz',
                               'sss.fif', 'sss.fif.gz', 'meg.fif')):
                html = report._render_raw(fname)
                report_fname = fname
                report_sectionlabel = 'raw'
            elif fname.endswith(('-fwd.fif', '-fwd.fif.gz')):
                html = report._render_forward(fname)
                report_fname = fname
                report_sectionlabel = 'forward'
            elif fname.endswith(('-inv.fif', '-inv.fif.gz')):
                html = report._render_inverse(fname)
                report_fname = fname
                report_sectionlabel = 'inverse'
            elif fname.endswith(('-ave.fif', '-ave.fif.gz')):
                if cov is not None:
                    html = report._render_whitened_evoked(fname, cov, baseline,
                                                          image_format)
                    report_fname = fname + ' (whitened)'
                    report_sectionlabel = 'evoked'
                    _update_html(html, report_fname, report_sectionlabel)

                html = report._render_evoked(fname, baseline, image_format)
                report_fname = fname
                report_sectionlabel = 'evoked'
            elif fname.endswith(('-eve.fif', '-eve.fif.gz')):
                html = report._render_eve(fname, sfreq, image_format)
                report_fname = fname
                report_sectionlabel = 'events'
            elif fname.endswith(('-epo.fif', '-epo.fif.gz')):
                html = report._render_epochs(fname, image_format)
                report_fname = fname
                report_sectionlabel = 'epochs'
            elif (fname.endswith(('-cov.fif', '-cov.fif.gz')) and
                  report.info_fname is not None):
                html = report._render_cov(fname, info, image_format)
                report_fname = fname
                report_sectionlabel = 'covariance'
            elif (fname.endswith(('-trans.fif', '-trans.fif.gz')) and
                  report.info_fname is not None and report.subjects_dir
                  is not None and report.subject is not None):
                html = report._render_trans(fname, report.data_path, info,
                                            report.subject,
                                            report.subjects_dir)
                report_fname = fname
                report_sectionlabel = 'trans'
            else:
                html = None
                report_fname = None
                report_sectionlabel = None
        except Exception as e:
            if on_error == 'warn':
                warn('Failed to process file %s:\n"%s"' % (fname, e))
            elif on_error == 'raise':
                raise
            html = None
            report_fname = None
            report_sectionlabel = None
        _update_html(html, report_fname, report_sectionlabel)

    return htmls, report_fnames, report_sectionlabels


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
# IMAGE FUNCTIONS

def _figure_agg(**kwargs):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    fig = Figure(**kwargs)
    FigureCanvas(fig)
    return fig


def _build_image_png(data, cmap='gray'):
    """Build an image encoded in base64."""
    import matplotlib.pyplot as plt

    figsize = data.shape[::-1]
    if figsize[0] == 1:
        figsize = tuple(figsize[1:])
        data = data[:, :, 0]
    fig = _figure_agg(figsize=figsize, dpi=1.0, frameon=False)
    cmap = getattr(plt.cm, cmap, plt.cm.gray)
    fig.figimage(data, cmap=cmap)
    output = BytesIO()
    fig.savefig(output, dpi=fig.get_dpi(), format='png')
    return base64.b64encode(output.getvalue()).decode('ascii')


def _iterate_sagittal_slices(array, limits=None):
    """Iterate sagittal slices."""
    shape = array.shape[0]
    for ind in range(shape):
        if limits and ind not in limits:
            continue
        yield ind, array[ind, :, :]


def _iterate_axial_slices(array, limits=None):
    """Iterate axial slices."""
    shape = array.shape[1]
    for ind in range(shape):
        if limits and ind not in limits:
            continue
        yield ind, array[:, ind, :]


def _iterate_coronal_slices(array, limits=None):
    """Iterate coronal slices."""
    shape = array.shape[2]
    for ind in range(shape):
        if limits and ind not in limits:
            continue
        yield ind, np.flipud(np.rot90(array[:, :, ind]))


def _iterate_mri_slices(name, ind, global_id, slides_klass, data, cmap):
    """Auxiliary function for parallel processing of mri slices."""
    img_klass = 'slideimg-%s' % name

    caption = u'Slice %s %s' % (name, ind)
    slice_id = '%s-%s-%s' % (name, global_id, ind)
    div_klass = 'span12 %s' % slides_klass
    img = _build_image_png(data, cmap=cmap)
    first = True if ind == 0 else False
    html = _build_html_image(img, slice_id, div_klass, img_klass, caption,
                             first, image_format='png')
    return ind, html


###############################################################################
# HTML functions

def _build_html_image(img, id, div_klass, img_klass, caption=None,
                      show=True, image_format='png'):
    """Build a html image from a slice array."""
    html = []
    add_style = u'' if show else u'style="display: none"'
    html.append(u'<li class="%s" id="%s" %s>' % (div_klass, id, add_style))
    html.append(u'<div class="thumbnail">')
    if image_format == 'png':
        html.append(u'<img class="%s" alt="" style="width:90%%;" '
                    'src="data:image/png;base64,%s">'
                    % (img_klass, img))
    else:
        html.append(u'<div style="text-align:center;" class="%s">%s</div>'
                    % (img_klass, img))
    html.append(u'</div>')
    if caption:
        html.append(u'<h4>%s</h4>' % caption)
    html.append(u'</li>')
    return u'\n'.join(html)


slider_template = HTMLTemplate(u"""
<script>$("#{{slider_id}}").slider({
                       range: "min",
                       /*orientation: "vertical",*/
                       min: {{minvalue}},
                       max: {{maxvalue}},
                       step: {{step}},
                       value: {{startvalue}},
                       create: function(event, ui) {
                       $(".{{klass}}").hide();
                       $("#{{klass}}-{{startvalue}}").show();},
                       stop: function(event, ui) {
                       var list_value = $("#{{slider_id}}").slider("value");
                       $(".{{klass}}").hide();
                       $("#{{klass}}-"+list_value).show();}
                       })</script>
""")

slider_full_template = Template(u"""
<li class="{{div_klass}}" id="{{id}}">
<h4>{{title}}</h4>
<div class="thumbnail">
    <ul><li class="slider">
        <div class="row">
            <div class="col-md-6 col-md-offset-3">
                <div id="{{slider_id}}"></div>
                <ul class="thumbnail">
                    {{image_html}}
                </ul>
                {{html}}
            </div>
        </div>
    </li></ul>
</div>
</li>
""")


def _build_html_slider(slices_range, slides_klass, slider_id,
                       start_value=None):
    """Build an html slider for a given slices range and a slices klass."""
    if start_value is None:
        start_value = slices_range[len(slices_range) // 2]
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('ignore')
        out = slider_template.substitute(
            slider_id=slider_id, klass=slides_klass,
            step=slices_range[1] - slices_range[0],
            minvalue=slices_range[0], maxvalue=slices_range[-1],
            startvalue=start_value)
    return out


###############################################################################
# HTML scan renderer

header_template = Template(u"""
<!DOCTYPE html>
<html lang="{{lang}}">
<head>
{{include}}
<script type="text/javascript">

        var toggle_state = false;
        $(document).on('keydown', function (event) {
            if (event.which == 84){
                if (!toggle_state)
                    $('.has_toggle').trigger('click');
                else if (toggle_state)
                    $('.has_toggle').trigger('click');
            toggle_state = !toggle_state;
            }
        });

        function togglebutton(class_name){
            $(class_name).toggle();

            if ($(class_name + '-btn').hasClass('active'))
                $(class_name + '-btn').removeClass('active');
            else
                $(class_name + '-btn').addClass('active');
        }

        /* Scroll down on click to #id so that caption is not hidden
        by navbar */
        var shiftWindow = function() { scrollBy(0, -60) };
        if (location.hash) shiftWindow();
        window.addEventListener("hashchange", shiftWindow);

        </script>
<style type="text/css">

body {
    line-height: 1.5em;
    font-family: arial, sans-serif;
}

h1 {
    font-size: 30px;
    text-align: center;
}

h4 {
    text-align: center;
}

@link-color:       @brand-primary;
@link-hover-color: darken(@link-color, 15%);

a{
    color: @link-color;
    &:hover {
        color: @link-hover-color;
        text-decoration: underline;
  }
}

li{
    list-style-type:none;
}

#wrapper {
    text-align: left;
    margin: 5em auto;
    width: 700px;
}

#container{
    position: relative;
}

#content{
    margin-left: 22%;
    margin-top: 60px;
    width: 75%;
}

#toc {
  margin-top: navbar-height;
  position: fixed;
  width: 20%;
  height: 90%;
  overflow: auto;
}

#toc li {
    overflow: hidden;
    padding-bottom: 2px;
    margin-left: 20px;
}

#toc span {
    float: left;
    padding: 0 2px 3px 0;
}

div.footer {
    background-color: #C0C0C0;
    color: #000000;
    padding: 3px 8px 3px 0;
    clear: both;
    font-size: 0.8em;
    text-align: right;
}

</style>
</head>
<body>

<nav class="navbar navbar-inverse navbar-fixed-top" role="navigation">
    <div class="container-fluid">
        <div class="navbar-header navbar-left">
            <ul class="nav nav-pills"><li class="active">
                <a class="navbar-btn" data-toggle="collapse"
                data-target="#viewnavbar" href="javascript:void(0)">
                ></a></li></ul>
    </div>
        <h3 class="navbar-text" style="color:white">{{title}}</h3>
        <ul class="nav nav-pills navbar-right" style="margin-top: 7px;"
        id="viewnavbar">

        {{for section in sections}}

        <li class="active {{sectionvars[section]}}-btn">
           <a href="javascript:void(0)"
           onclick="togglebutton('.{{sectionvars[section]}}')"
           class="has_toggle">
    {{section if section != 'mri' else 'MRI'}}
           </a>
        </li>

        {{endfor}}

        </ul>
    </div>
</nav>
""")

footer_template = HTMLTemplate(u"""
</div></body>
<div class="footer">
        &copy; Copyright 2012-{{current_year}}, MNE Developers.
      Created on {{date}}.
      Powered by <a href="http://mne.tools/">MNE.
</div>
</html>
""")

html_template = Template(u"""
<li class="{{div_klass}}" id="{{id}}">
    <h4>{{caption}}</h4>
    <div class="thumbnail">{{html}}</div>
</li>
""")

image_template = Template(u"""

{{default interactive = False}}
{{default width = 50}}
{{default id = False}}
{{default image_format = 'png'}}
{{default scale = None}}
{{default comment = None}}

<li class="{{div_klass}}" {{if id}}id="{{id}}"{{endif}}
{{if not show}}style="display: none"{{endif}}>

{{if caption}}
<h4>{{caption}}</h4>
{{endif}}
<div class="thumbnail">
{{if not interactive}}
    {{if image_format == 'png'}}
        {{if scale is not None}}
            <img alt="" style="width:{{width}}%;"
             src="data:image/png;base64,{{img}}">
        {{else}}
            <img alt=""
             src="data:image/png;base64,{{img}}">
        {{endif}}
    {{elif image_format == 'gif'}}
        {{if scale is not None}}
            <img alt="" style="width:{{width}}%;"
             src="data:image/gif;base64,{{img}}">
        {{else}}
            <img alt=""
             src="data:image/gif;base64,{{img}}">
        {{endif}}
    {{elif image_format == 'svg'}}
        <div style="text-align:center;">
            {{img}}
        </div>
    {{endif}}
    {{if comment is not None}}
        <br><br>
        <div style="text-align:center;">
            <style>
                p.test {word-wrap: break-word;}
            </style>
            <p class="test">
                {{comment}}
            </p>
        </div>
    {{endif}}
{{else}}
    <center>{{interactive}}</center>
{{endif}}
</div>
</li>
""")

repr_template = Template(u"""
<li class="{{div_klass}}" id="{{id}}">
<h4>{{caption}}</h4><hr>
{{repr}}
<hr></li>
""")

raw_template = Template(u"""
<li class="{{div_klass}}" id="{{id}}">
<h4>{{caption}}</h4>
<table class="table table-hover">
    <tr>
        <th>Measurement date</th>
        {{if meas_date is not None}}
        <td>{{meas_date}}</td>
        {{else}}<td>Unknown</td>{{endif}}
    </tr>
    <tr>
        <th>Experimenter</th>
        {{if info['experimenter'] is not None}}
        <td>{{info['experimenter']}}</td>
        {{else}}<td>Unknown</td>{{endif}}
    </tr>
    <tr>
        <th>Digitized points</th>
        {{if info['dig'] is not None}}
        <td>{{len(info['dig'])}} points</td>
        {{else}}
        <td>Not available</td>
        {{endif}}
    </tr>
    <tr>
        <th>Good channels</th>
        <td>{{n_mag}} magnetometer, {{n_grad}} gradiometer,
            and {{n_eeg}} EEG channels</td>
    </tr>
    <tr>
        <th>Bad channels</th>
        {{if info['bads'] is not None}}
        <td>{{', '.join(info['bads'])}}</td>
        {{else}}<td>None</td>{{endif}}
    </tr>
    <tr>
        <th>EOG channels</th>
        <td>{{eog}}</td>
    </tr>
    <tr>
        <th>ECG channels</th>
        <td>{{ecg}}</td>
    <tr>
        <th>Measurement time range</th>
        <td>{{u'%0.2f' % tmin}} to {{u'%0.2f' % tmax}} sec.</td>
    </tr>
    <tr>
        <th>Sampling frequency</th>
        <td>{{u'%0.2f' % info['sfreq']}} Hz</td>
    </tr>
    <tr>
        <th>Highpass</th>
        <td>{{u'%0.2f' % info['highpass']}} Hz</td>
    </tr>
     <tr>
        <th>Lowpass</th>
        <td>{{u'%0.2f' % info['lowpass']}} Hz</td>
    </tr>
</table>
</li>
""")


toc_list = Template(u"""
<li class="{{div_klass}}">
    {{if id}}
        <a href="javascript:void(0)" onclick="window.location.hash={{id}};">
    {{endif}}
<span title="{{tooltip}}" style="color:{{color}}"> {{text}}</span>
{{if id}}</a>{{endif}}
</li>
""")


def _check_scale(scale):
    """Ensure valid scale value is passed."""
    if np.isscalar(scale) and scale <= 0:
        raise ValueError('scale must be positive, not %s' % scale)


def _check_image_format(rep, image_format):
    """Ensure fmt is valid."""
    if rep is None:
        _check_option('image_format', image_format, ['png', 'svg'])
    elif image_format is not None:
        _check_option('image_format', image_format, ['png', 'svg', None])
    else:  # rep is not None and image_format is None
        image_format = rep.image_format
    return image_format


@fill_doc
class Report(object):
    r"""Object for rendering HTML.

    Parameters
    ----------
    info_fname : str
        Name of the file containing the info dictionary.
    %(subjects_dir)s
    subject : str | None
        Subject name.
    title : str
        Title of the report.
    cov_fname : str
        Name of the file containing the noise covariance.
    baseline : None or tuple of length 2 (default (None, 0))
        The time interval to apply baseline correction for evokeds.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal to (None, None) all the time
        interval is used.
        The baseline (a, b) includes both endpoints, i.e. all
        timepoints t such that a <= t <= b.
    image_format : str
        Default image format to use (default is 'png').
        SVG uses vector graphics, so fidelity is higher but can increase
        file size and browser image rendering time as well.

        .. versionadded:: 0.15

    raw_psd : bool | dict
        If True, include PSD plots for raw files. Can be False (default) to
        omit, True to plot, or a dict to pass as ``kwargs`` to
        :meth:`mne.io.Raw.plot_psd`.

        .. versionadded:: 0.17
    %(verbose)s

    Notes
    -----
    See :ref:`tut-report` for an introduction to using ``mne.Report``, and
    :ref:`this example <ex-report>` for an example of customizing the report
    with a slider.

    .. versionadded:: 0.8.0
    """

    def __init__(self, info_fname=None, subjects_dir=None,
                 subject=None, title=None, cov_fname=None, baseline=None,
                 image_format='png', raw_psd=False, verbose=None):
        self.info_fname = info_fname
        self.cov_fname = cov_fname
        self.baseline = baseline
        self.subjects_dir = get_subjects_dir(subjects_dir, raise_error=False)
        self.subject = subject
        self.title = title
        self.image_format = _check_image_format(None, image_format)
        self.verbose = verbose

        self.initial_id = 0
        self.html = []
        self.fnames = []  # List of file names rendered
        self.sections = []  # List of sections
        self.lang = 'en-us'  # language setting for the HTML file
        self._sectionlabels = []  # Section labels
        self._sectionvars = {}  # Section variable names in js
        # boolean to specify if sections should be ordered in natural
        # order of processing (raw -> events ... -> inverse)
        self._sort_sections = False
        if not isinstance(raw_psd, bool) and not isinstance(raw_psd, dict):
            raise TypeError('raw_psd must be bool or dict, got %s'
                            % (type(raw_psd),))
        self.raw_psd = raw_psd
        self._init_render()  # Initialize the renderer

    def __repr__(self):
        """Print useful info about report."""
        s = '<Report | %d items' % len(self.fnames)
        if self.title is not None:
            s += ' | %s' % self.title
        fnames = [_get_fname(f) for f in self.fnames]
        if len(self.fnames) > 4:
            s += '\n%s' % '\n'.join(fnames[:2])
            s += '\n ...\n'
            s += '\n'.join(fnames[-2:])
        elif len(self.fnames) > 0:
            s += '\n%s' % '\n'.join(fnames)
        s += '\n>'
        return s

    def __len__(self):
        """Return the number of items in report."""
        return len(self.fnames)

    def _get_id(self):
        """Get id of plot."""
        self.initial_id += 1
        return self.initial_id

    def _validate_input(self, items, captions, section, comments=None):
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
        if section not in self.sections:
            self.sections.append(section)
            self._sectionvars[section] = _clean_varnames(section)

        return items, captions, comments

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
                self.sections.remove(section_)
                del self._sectionvars[section_]

        return index

    def _add_or_replace(self, fname, sectionlabel, html, replace=False):
        """Append a figure to the report, or replace it if it already exists.

        Parameters
        ----------
        fname : str
            A unique identifier for the figure. If a figure with this
            identifier has already been added, it will be replaced.
        sectionlabel : str
            The section to place the figure in.
        html : str
            The HTML that contains the figure.
        replace : bool
            Existing figures are only replaced if this is set to ``True``.
            Defaults to ``False``.
        """
        assert isinstance(html, str)  # otherwise later will break
        if replace and fname in self.fnames:
            # Find last occurrence of the figure
            ind = max([i for i, existing in enumerate(self.fnames)
                       if existing == fname])
            self.fnames[ind] = fname
            self._sectionlabels[ind] = sectionlabel
            self.html[ind] = html
        else:
            # Append new record
            self.fnames.append(fname)
            self._sectionlabels.append(sectionlabel)
            self.html.append(html)

    def add_figs_to_section(self, figs, captions, section='custom',
                            scale=None, image_format=None, comments=None,
                            replace=False):
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
            Name of the section to place the figure in. If section already
            exists, the figures will be appended to the end of the section.
        scale : float | None | callable
            Scale the images maintaining the aspect ratio.
            If None, no scaling is applied. If float, scale will determine
            the relative scaling (might not work for scale <= 1 depending on
            font sizes). If function, should take a figure object as input
            parameter. Defaults to None.
        image_format : str | None
            The image format to be used for the report, can be 'png' or 'svd'.
            None (default) will use the default specified during Report
            class construction.
        comments : None | str | list of str
            A string of text or a list of strings of text to be appended after
            the figure.
        replace : bool
            If ``True``, figures already present that have the same caption
            will be replaced. Defaults to ``False``.
        """
        figs, captions, comments = self._validate_input(figs, captions,
                                                        section, comments)
        image_format = _check_image_format(self, image_format)
        _check_scale(scale)
        for fig, caption, comment in zip(figs, captions, comments):
            caption = 'custom plot' if caption == '' else caption
            sectionvar = self._sectionvars[section]
            global_id = self._get_id()
            div_klass = self._sectionvars[section]
            img_klass = self._sectionvars[section]

            img = _fig_to_img(fig, image_format, scale)
            html = image_template.substitute(img=img, id=global_id,
                                             div_klass=div_klass,
                                             img_klass=img_klass,
                                             caption=caption,
                                             show=True,
                                             image_format=image_format,
                                             comment=comment)
            self._add_or_replace('%s-#-%s-#-custom' % (caption, sectionvar),
                                 sectionvar, html, replace)

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

            self._add_or_replace('%s-#-%s-#-custom' % (caption, sectionvar),
                                 sectionvar, html, replace)

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
        for html, caption in zip(htmls, captions):
            caption = 'custom plot' if caption == '' else caption
            sectionvar = self._sectionvars[section]
            global_id = self._get_id()
            div_klass = self._sectionvars[section]

            self._add_or_replace(
                '%s-#-%s-#-custom' % (caption, sectionvar), sectionvar,
                html_template.substitute(div_klass=div_klass, id=global_id,
                                         caption=caption, html=html), replace)

    @fill_doc
    def add_bem_to_section(self, subject, caption='BEM', section='bem',
                           decim=2, n_jobs=1, subjects_dir=None,
                           replace=False):
        """Render a bem slider html str.

        Parameters
        ----------
        subject : str
            Subject name.
        caption : str
            A caption for the bem.
        section : str
            Name of the section. If section already exists, the bem
            will be appended to the end of the section.
        decim : int
            Use this decimation factor for generating MRI/BEM images
            (since it can be time consuming).
        %(n_jobs)s
        %(subjects_dir)s
        replace : bool
            If ``True``, figures already present that have the same caption
            will be replaced. Defaults to ``False``.

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        caption = 'custom plot' if caption == '' else caption
        html = self._render_bem(subject=subject, subjects_dir=subjects_dir,
                                decim=decim, n_jobs=n_jobs, section=section,
                                caption=caption)
        html, caption, _ = self._validate_input(html, caption, section)
        sectionvar = self._sectionvars[section]
        # convert list->str
        assert isinstance(html, list)
        html = u''.join(html)
        self._add_or_replace('%s-#-%s-#-custom' % (caption[0], sectionvar),
                             sectionvar, html)

    def add_slider_to_section(self, figs, captions=None, section='custom',
                              title='Slider', scale=None, image_format=None,
                              replace=False):
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
            constructed as `%f s`. If None, it will default to
            `Data slice %d`.
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
        image_format : str | None
            The image format to be used for the report, can be 'png' or 'svd'.
            None (default) will use the default specified during Report
            class construction.
        replace : bool
            If ``True``, figures already present that have the same caption
            will be replaced. Defaults to ``False``.

        Notes
        -----
        .. versionadded:: 0.10.0
        """
        _check_scale(scale)
        image_format = _check_image_format(self, image_format)
        if isinstance(figs[0], list):
            raise NotImplementedError('`add_slider_to_section` '
                                      'can only add one slider at a time.')
        if len(figs) < 2:
            raise ValueError('figs must be at least length 2, got %s'
                             % (len(figs),))
        figs = [figs]
        figs, _, _ = self._validate_input(figs, section, section)
        figs = figs[0]

        sectionvar = self._sectionvars[section]
        global_id = self._get_id()
        name = 'slider'

        html = []
        slides_klass = '%s-%s' % (name, global_id)
        div_klass = 'span12 %s' % slides_klass

        sl = np.arange(0, len(figs))
        slices = []
        img_klass = 'slideimg-%s' % name

        if captions is None:
            captions = ['Data slice %d' % ii for ii in sl]
        elif isinstance(captions, (list, tuple, np.ndarray)):
            if len(figs) != len(captions):
                raise ValueError('Captions must be the same length as the '
                                 'number of slides.')
            if isinstance(captions[0], (float, int)):
                captions = ['%0.3f s' % caption for caption in captions]
        else:
            raise TypeError('Captions must be None or an iterable of '
                            'float, int, str, Got %s' % type(captions))
        for ii, (fig, caption) in enumerate(zip(figs, captions)):
            img = _fig_to_img(fig, image_format, scale)
            slice_id = '%s-%s-%s' % (name, global_id, sl[ii])
            first = True if ii == 0 else False
            slices.append(_build_html_image(img, slice_id, div_klass,
                                            img_klass, caption, first,
                                            image_format=image_format))
        # Render the slider
        slider_id = 'select-%s-%s' % (name, global_id)
        # Render the slices
        image_html = u'\n'.join(slices)
        html.append(_build_html_slider(sl, slides_klass, slider_id,
                                       start_value=0))
        html = '\n'.join(html)

        slider_klass = sectionvar

        self._add_or_replace(
            '%s-#-%s-#-custom' % (title, sectionvar), sectionvar,
            slider_full_template.substitute(id=global_id, title=title,
                                            div_klass=slider_klass,
                                            slider_id=slider_id, html=html,
                                            image_html=image_html))

    ###########################################################################
    # HTML rendering
    def _render_one_axis(self, slices_iter, name, global_id, cmap,
                         n_elements, n_jobs):
        """Render one axis of the array."""
        global_id = global_id or name
        html = []
        html.append(u'<div class="col-xs-6 col-md-4">')
        slides_klass = '%s-%s' % (name, global_id)

        use_jobs = min(n_jobs, max(1, n_elements))
        parallel, p_fun, _ = parallel_func(_iterate_mri_slices, use_jobs)
        r = parallel(p_fun(name, ind, global_id, slides_klass, data, cmap)
                     for ind, data in slices_iter)
        slices_range, slices = zip(*r)

        # Render the slider
        slider_id = 'select-%s-%s' % (name, global_id)
        html.append(u'<div id="%s"></div>' % slider_id)
        html.append(u'<ul class="thumbnail">')
        # Render the slices
        html.append(u'\n'.join(slices))
        html.append(u'</ul>')
        html.append(_build_html_slider(slices_range, slides_klass, slider_id))
        html.append(u'</div>')
        return '\n'.join(html)

    ###########################################################################
    # global rendering functions
    @verbose
    def _init_render(self, verbose=None):
        """Initialize the renderer."""
        inc_fnames = ['jquery.js', 'jquery-ui.min.js',
                      'bootstrap.min.js', 'jquery-ui.min.css',
                      'bootstrap.min.css']

        include = list()
        for inc_fname in inc_fnames:
            logger.info('Embedding : %s' % inc_fname)
            fname = op.join(op.dirname(__file__), 'html', inc_fname)
            with open(fname, 'rb') as fid:
                file_content = fid.read().decode('utf-8')
            if inc_fname.endswith('.js'):
                include.append(u'<script type="text/javascript">' +
                               file_content + u'</script>')
            elif inc_fname.endswith('.css'):
                include.append(u'<style type="text/css">' +
                               file_content + u'</style>')
        self.include = ''.join(include)

    @verbose
    def parse_folder(self, data_path, pattern='*.fif', n_jobs=1, mri_decim=2,
                     sort_sections=True, on_error='warn', image_format=None,
                     render_bem=True, verbose=None):
        r"""Render all the files in the folder.

        Parameters
        ----------
        data_path : str
            Path to the folder containing data whose HTML report will be
            created.
        pattern : str | list of str
            Filename pattern(s) to include in the report.
            Example: [\*raw.fif, \*ave.fif] will include Raw as well as Evoked
            files.
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
        image_format : str | None
            The image format to be used for the report, can be 'png' or 'svd'.
            None (default) will use the default specified during Report
            class construction.

            .. versionadded:: 0.15
        render_bem : bool
            If True (default), try to render the BEM.

            .. versionadded:: 0.16
        %(verbose_meth)s
        """
        image_format = _check_image_format(self, image_format)
        _check_option('on_error', on_error, ['ignore', 'warn', 'raise'])
        self._sort = sort_sections

        n_jobs = check_n_jobs(n_jobs)
        self.data_path = data_path

        if self.title is None:
            self.title = 'MNE Report for ...%s' % self.data_path[-20:]

        if not isinstance(pattern, (list, tuple)):
            pattern = [pattern]

        # iterate through the possible patterns
        fnames = list()
        for p in pattern:
            fnames.extend(sorted(_recursive_search(self.data_path, p)))

        if self.info_fname is not None:
            info = read_info(self.info_fname, verbose=False)
            sfreq = info['sfreq']
        else:
            # only warn if relevant
            if any(fname.endswith(('-cov.fif', '-cov.fif.gz'))
                   for fname in fnames):
                warn('`info_fname` not provided. Cannot render '
                     '-cov.fif(.gz) files.')
            if any(fname.endswith(('-trans.fif', '-trans.fif.gz'))
                   for fname in fnames):
                warn('`info_fname` not provided. Cannot render '
                     '-trans.fif(.gz) files.')
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
        r = parallel(p_fun(self, fname, info, cov, baseline, sfreq, on_error,
                           image_format)
                     for fname in np.array_split(fnames, use_jobs))
        htmls, report_fnames, report_sectionlabels = zip(*r)

        # combine results from n_jobs discarding plots not rendered
        self.html = [html for html in sum(htmls, []) if html is not None]
        self.fnames = [fname for fname in sum(report_fnames, []) if
                       fname is not None]
        self._sectionlabels = [slabel for slabel in
                               sum(report_sectionlabels, [])
                               if slabel is not None]

        # find unique section labels
        self.sections = sorted(set(self._sectionlabels))
        self._sectionvars = dict(zip(self.sections, self.sections))

        # render mri
        if render_bem:
            if self.subjects_dir is not None and self.subject is not None:
                logger.info('Rendering BEM')
                self.html.append(self._render_bem(
                    self.subject, self.subjects_dir, mri_decim, n_jobs))
                self.fnames.append('bem')
                self._sectionlabels.append('mri')
            else:
                warn('`subjects_dir` and `subject` not provided. Cannot '
                     'render MRI and -trans.fif(.gz) files.')

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
                 'image_format', 'info_fname', 'initial_id', 'raw_psd',
                 '_sectionlabels', 'sections', '_sectionvars',
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

    def save(self, fname=None, open_browser=True, overwrite=False):
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
        overwrite : bool
            If True, overwrite report if it already exists. Defaults to False.

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
            answer = input(msg)
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
                self._render_toc()

                # Annotate the HTML with a TOC and footer.
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter('ignore')
                    html = footer_template.substitute(
                        date=time.strftime("%B %d, %Y"),
                        current_year=time.strftime("%Y"))
                self.html.append(html)

                # Writing to disk may fail. However, we need to make sure that
                # the TOC and footer are removed regardless, otherwise they
                # will be duplicated when the user attempts to save again.
                try:
                    # Write HTML
                    with codecs.open(fname, 'w', 'utf-8') as fobj:
                        fobj.write(_fix_global_ids(u''.join(self.html)))
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

    @verbose
    def _render_toc(self, verbose=None):
        """Render the Table of Contents."""
        logger.info('Rendering : Table of Contents')

        html_toc = u'<div id="container">'
        html_toc += u'<div id="toc"><center><h4>CONTENTS</h4></center>'

        global_id = 1

        # Reorder self.sections to reflect natural ordering
        if self._sort_sections:
            sections = list(set(self.sections) & set(SECTION_ORDER))
            custom = [section for section in self.sections if section
                      not in SECTION_ORDER]
            order = [sections.index(section) for section in SECTION_ORDER if
                     section in sections]
            self.sections = np.array(sections)[order].tolist() + custom

        # Sort by section
        html, fnames, sectionlabels = [], [], []
        for section in self.sections:
            logger.info('%s' % section)
            for sectionlabel, this_html, fname in (zip(self._sectionlabels,
                                                   self.html, self.fnames)):
                if self._sectionvars[section] == sectionlabel:
                    html.append(this_html)
                    fnames.append(fname)
                    sectionlabels.append(sectionlabel)
                    logger.info(_get_fname(fname))
                    color = _is_bad_fname(fname)
                    div_klass, tooltip, text = _get_toc_property(fname)

                    # loop through conditions for evoked
                    if fname.endswith(('-ave.fif', '-ave.fif.gz',
                                      '(whitened)')):
                        text = os.path.basename(fname)
                        if fname.endswith('(whitened)'):
                            fname = fname[:-11]
                        # XXX: remove redundant read_evokeds
                        evokeds = read_evokeds(fname, verbose=False)

                        html_toc += toc_list.substitute(
                            div_klass=div_klass, id=None, tooltip=fname,
                            color='#428bca', text=text)

                        html_toc += u'<li class="evoked"><ul>'
                        for ev in evokeds:
                            html_toc += toc_list.substitute(
                                div_klass=div_klass, id=global_id,
                                tooltip=fname, color=color, text=ev.comment)
                            global_id += 1
                        html_toc += u'</ul></li>'

                    elif fname.endswith(tuple(VALID_EXTENSIONS +
                                        ['bem', 'custom'])):
                        html_toc += toc_list.substitute(div_klass=div_klass,
                                                        id=global_id,
                                                        tooltip=tooltip,
                                                        color=color,
                                                        text=text)
                        global_id += 1

        html_toc += u'\n</ul></div>'
        html_toc += u'<div id="content">'

        # The sorted html (according to section)
        self.html = html
        self.fnames = fnames
        self._sectionlabels = sectionlabels

        lang = getattr(self, 'lang', 'en-us')
        html_header = header_template.substitute(
            title=self.title, include=self.include, lang=lang,
            sections=self.sections, sectionvars=self._sectionvars)
        self.html.insert(0, html_header)  # Insert header at position 0
        self.html.insert(1, html_toc)  # insert TOC

    def _render_array(self, array, global_id=None, cmap='gray',
                      limits=None, n_jobs=1):
        """Render mri without bem contours (only PNG)."""
        html = []
        html.append(u'<div class="thumbnail">')
        # Axial
        limits = limits or {}
        axial_limit = limits.get('axial')
        axial_slices_gen = _iterate_axial_slices(array, axial_limit)
        html.append(
            self._render_one_axis(axial_slices_gen, 'axial',
                                  global_id, cmap, array.shape[1], n_jobs))
        # Sagittal
        sagittal_limit = limits.get('sagittal')
        sagittal_slices_gen = _iterate_sagittal_slices(array, sagittal_limit)
        html.append(
            self._render_one_axis(sagittal_slices_gen, 'sagittal',
                                  global_id, cmap, array.shape[1], n_jobs))
        # Coronal
        coronal_limit = limits.get('coronal')
        coronal_slices_gen = _iterate_coronal_slices(array, coronal_limit)
        html.append(
            self._render_one_axis(coronal_slices_gen, 'coronal',
                                  global_id, cmap, array.shape[1], n_jobs))
        # Close section
        html.append(u'</div>')
        return '\n'.join(html)

    def _render_one_bem_axis(self, mri_fname, surf_fnames, global_id,
                             shape, orientation='coronal', decim=2, n_jobs=1):
        """Render one axis of bem contours (only PNG)."""
        orientation_name2axis = dict(sagittal=0, axial=1, coronal=2)
        orientation_axis = orientation_name2axis[orientation]
        n_slices = shape[orientation_axis]
        orig_size = np.roll(shape, orientation_axis)[[1, 2]]

        name = orientation
        html = []
        html.append(u'<div class="col-xs-6 col-md-4">')
        slides_klass = '%s-%s' % (name, global_id)

        sl = np.arange(0, n_slices, decim)
        kwargs = dict(mri_fname=mri_fname, surf_fnames=surf_fnames, show=False,
                      orientation=orientation, img_output=orig_size)
        imgs = _figs_to_mrislices(sl, n_jobs, **kwargs)
        slices = []
        img_klass = 'slideimg-%s' % name
        div_klass = 'span12 %s' % slides_klass
        for ii, img in enumerate(imgs):
            slice_id = '%s-%s-%s' % (name, global_id, sl[ii])
            caption = u'Slice %s %s' % (name, sl[ii])
            first = True if ii == 0 else False
            slices.append(_build_html_image(img, slice_id, div_klass,
                                            img_klass, caption, first,
                                            image_format='png'))

        # Render the slider
        slider_id = 'select-%s-%s' % (name, global_id)
        html.append(u'<div id="%s"></div>' % slider_id)
        html.append(u'<ul class="thumbnail">')
        # Render the slices
        html.append(u'\n'.join(slices))
        html.append(u'</ul>')
        html.append(_build_html_slider(sl, slides_klass, slider_id))
        html.append(u'</div>')
        return '\n'.join(html)

    def _render_image_png(self, image, cmap='gray', n_jobs=1):
        """Render one slice of mri without bem as a PNG."""
        import nibabel as nib

        global_id = self._get_id()

        if 'mri' not in self.sections:
            self.sections.append('mri')
            self._sectionvars['mri'] = 'mri'

        nim = nib.load(image)
        data = _get_img_fdata(nim)
        shape = data.shape
        limits = {'sagittal': range(0, shape[0], 2),
                  'axial': range(0, shape[1], 2),
                  'coronal': range(0, shape[2], 2)}
        name = op.basename(image)
        html = u'<li class="mri" id="%d">\n' % global_id
        html += u'<h4>%s</h4>\n' % name
        html += self._render_array(data, global_id=global_id,
                                   cmap=cmap, limits=limits, n_jobs=n_jobs)
        html += u'</li>\n'
        return html

    def _render_raw(self, raw_fname):
        """Render raw (only text)."""
        import matplotlib.pyplot as plt
        global_id = self._get_id()

        raw = read_raw_fif(raw_fname, allow_maxshield='yes')
        extra = ' (MaxShield on)' if raw.info.get('maxshield', False) else ''
        caption = u'Raw : %s%s' % (raw_fname, extra)

        n_eeg = len(pick_types(raw.info, meg=False, eeg=True))
        n_grad = len(pick_types(raw.info, meg='grad'))
        n_mag = len(pick_types(raw.info, meg='mag'))
        pick_eog = pick_types(raw.info, meg=False, eog=True)
        if len(pick_eog) > 0:
            eog = ', '.join(np.array(raw.info['ch_names'])[pick_eog])
        else:
            eog = 'Not available'
        pick_ecg = pick_types(raw.info, meg=False, ecg=True)
        if len(pick_ecg) > 0:
            ecg = ', '.join(np.array(raw.info['ch_names'])[pick_ecg])
        else:
            ecg = 'Not available'
        meas_date = raw.info['meas_date']
        if meas_date is not None:
            meas_date = meas_date.strftime("%B %d, %Y") + ' GMT'

        html = raw_template.substitute(
            div_klass='raw', id=global_id, caption=caption, info=raw.info,
            meas_date=meas_date, n_eeg=n_eeg, n_grad=n_grad, n_mag=n_mag,
            eog=eog, ecg=ecg, tmin=raw._first_time, tmax=raw._last_time)

        raw_psd = {} if self.raw_psd is True else self.raw_psd
        if isinstance(raw_psd, dict):
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            n_ax = sum(kind in raw for kind in _DATA_CH_TYPES_SPLIT)
            fig, axes = plt.subplots(n_ax, 1, figsize=(6, 1 + 1.5 * n_ax),
                                     dpi=92)
            FigureCanvasAgg(fig)
            img = _fig_to_img(raw.plot_psd, self.image_format,
                              ax=axes, **raw_psd)
            new_html = image_template.substitute(
                img=img, div_klass='raw', img_klass='raw',
                caption='PSD', show=True, image_format=self.image_format)
            html += '\n\n' + new_html
        return html

    def _render_forward(self, fwd_fname):
        """Render forward."""
        div_klass = 'forward'
        caption = u'Forward: %s' % fwd_fname
        fwd = read_forward_solution(fwd_fname)
        repr_fwd = re.sub('>', '', re.sub('<', '', repr(fwd)))
        global_id = self._get_id()
        html = repr_template.substitute(div_klass=div_klass,
                                        id=global_id,
                                        caption=caption,
                                        repr=repr_fwd)
        return html

    def _render_inverse(self, inv_fname):
        """Render inverse."""
        div_klass = 'inverse'
        caption = u'Inverse: %s' % inv_fname
        inv = read_inverse_operator(inv_fname)
        repr_inv = re.sub('>', '', re.sub('<', '', repr(inv)))
        global_id = self._get_id()
        html = repr_template.substitute(div_klass=div_klass,
                                        id=global_id,
                                        caption=caption,
                                        repr=repr_inv)
        return html

    def _render_evoked(self, evoked_fname, baseline, image_format):
        """Render evoked."""
        logger.debug('Evoked: Reading %s' % evoked_fname)
        evokeds = read_evokeds(evoked_fname, baseline=baseline, verbose=False)

        html = []
        for ei, ev in enumerate(evokeds):
            global_id = self._get_id()
            kwargs = dict(show=False)
            logger.debug('Evoked: Plotting instance %s/%s'
                         % (ei + 1, len(evokeds)))
            img = _fig_to_img(ev.plot, image_format, **kwargs)
            caption = u'Evoked : %s (%s)' % (evoked_fname, ev.comment)
            html.append(image_template.substitute(
                img=img, id=global_id, div_klass='evoked',
                img_klass='evoked', caption=caption, show=True,
                image_format=image_format))
            has_types = []
            if len(pick_types(ev.info, meg=False, eeg=True)) > 0:
                has_types.append('eeg')
            if len(pick_types(ev.info, meg='grad', eeg=False)) > 0:
                has_types.append('grad')
            if len(pick_types(ev.info, meg='mag', eeg=False)) > 0:
                has_types.append('mag')
            for ch_type in has_types:
                logger.debug('    Topomap type %s' % ch_type)
                img = _fig_to_img(ev.plot_topomap, image_format,
                                  ch_type=ch_type, **kwargs)
                caption = u'Topomap (ch_type = %s)' % ch_type
                html.append(image_template.substitute(
                    img=img, div_klass='evoked', img_klass='evoked',
                    caption=caption, show=True, image_format=image_format))
        logger.debug('Evoked: done')
        return '\n'.join(html)

    def _render_eve(self, eve_fname, sfreq, image_format):
        """Render events."""
        global_id = self._get_id()
        events = read_events(eve_fname)
        kwargs = dict(events=events, sfreq=sfreq, show=False)
        img = _fig_to_img(plot_events, image_format, **kwargs)
        caption = 'Events : ' + eve_fname
        html = image_template.substitute(
            img=img, id=global_id, div_klass='events', img_klass='events',
            caption=caption, show=True, image_format=image_format)
        return html

    def _render_epochs(self, epo_fname, image_format):
        """Render epochs."""
        global_id = self._get_id()
        epochs = read_epochs(epo_fname)
        kwargs = dict(subject=self.subject, show=False)
        img = _fig_to_img(epochs.plot_drop_log, image_format, **kwargs)
        caption = 'Epochs : ' + epo_fname
        show = True
        html = image_template.substitute(
            img=img, id=global_id, div_klass='epochs', img_klass='epochs',
            caption=caption, show=show, image_format=image_format)
        return html

    def _render_cov(self, cov_fname, info_fname, image_format, show_svd=True):
        """Render cov."""
        global_id = self._get_id()
        cov = read_cov(cov_fname)
        fig, svd = plot_cov(cov, info_fname, show=False, show_svd=show_svd)
        html = []
        figs = [fig]
        captions = ['Covariance : %s (n_samples: %s)' % (cov_fname, cov.nfree)]
        if svd is not None:
            figs.append(svd)
            captions.append('Singular values of the noise covariance')
        for fig, caption in zip(figs, captions):
            img = _fig_to_img(fig, image_format)
            show = True
            html.append(image_template.substitute(
                img=img, id=global_id, div_klass='covariance',
                img_klass='covariance', caption=caption, show=show,
                image_format=image_format))
        return '\n'.join(html)

    def _render_whitened_evoked(self, evoked_fname, noise_cov, baseline,
                                image_format):
        """Render whitened evoked."""
        evokeds = read_evokeds(evoked_fname, verbose=False)
        html = []
        for ev in evokeds:
            ev = read_evokeds(evoked_fname, ev.comment, baseline=baseline,
                              verbose=False)
            global_id = self._get_id()
            kwargs = dict(noise_cov=noise_cov, show=False)
            img = _fig_to_img(ev.plot_white, image_format, **kwargs)

            caption = u'Whitened evoked : %s (%s)' % (evoked_fname, ev.comment)
            show = True
            html.append(image_template.substitute(
                img=img, id=global_id, div_klass='evoked',
                img_klass='evoked', caption=caption, show=show,
                image_format=image_format))
        return '\n'.join(html)

    def _render_trans(self, trans, path, info, subject, subjects_dir):
        """Render trans (only PNG)."""
        kwargs = dict(info=info, trans=trans, subject=subject,
                      subjects_dir=subjects_dir)
        try:
            img = _iterate_trans_views(function=plot_alignment, **kwargs)
        except IOError:
            img = _iterate_trans_views(function=plot_alignment,
                                       surfaces=['head'], **kwargs)

        if img is not None:
            global_id = self._get_id()
            html = image_template.substitute(
                img=img, id=global_id, div_klass='trans',
                img_klass='trans', caption='Trans : ' + trans, width=75,
                show=True, image_format='png')
            return html

    def _render_bem(self, subject, subjects_dir, decim, n_jobs,
                    section='mri', caption='BEM'):
        """Render mri+bem (only PNG)."""
        import nibabel as nib

        subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)

        # Get the MRI filename
        mri_fname = op.join(subjects_dir, subject, 'mri', 'T1.mgz')
        if not op.isfile(mri_fname):
            warn('MRI file "%s" does not exist' % mri_fname)

        # Get the BEM surface filenames
        bem_path = op.join(subjects_dir, subject, 'bem')

        if not op.isdir(bem_path):
            warn('Subject bem directory "%s" does not exist' % bem_path)
            return self._render_image_png(mri_fname, cmap='gray',
                                          n_jobs=n_jobs)

        surf_fnames = []
        for surf_name in ['*inner_skull', '*outer_skull', '*outer_skin']:
            surf_fname = glob(op.join(bem_path, surf_name + '.surf'))
            if len(surf_fname) > 0:
                surf_fnames.append(surf_fname[0])
            else:
                warn('No surface found for %s.' % surf_name)
                continue
        if len(surf_fnames) == 0:
            warn('No surfaces found at all, rendering empty MRI')
            return self._render_image_png(mri_fname, cmap='gray',
                                          n_jobs=n_jobs)
        # XXX : find a better way to get max range of slices
        nim = nib.load(mri_fname)
        data = _get_img_fdata(nim)
        shape = data.shape
        del data  # free up memory

        html = []

        global_id = self._get_id()

        if section == 'mri' and 'mri' not in self.sections:
            self.sections.append('mri')
            self._sectionvars['mri'] = 'mri'

        name = caption

        html += u'<li class="mri" id="%d">\n' % global_id
        html += u'<h4>%s</h4>\n' % name  # all other captions are h4
        html += self._render_one_bem_axis(mri_fname, surf_fnames, global_id,
                                          shape, 'axial', decim, n_jobs)
        html += self._render_one_bem_axis(mri_fname, surf_fnames, global_id,
                                          shape, 'sagittal', decim, n_jobs)
        html += self._render_one_bem_axis(mri_fname, surf_fnames, global_id,
                                          shape, 'coronal', decim, n_jobs)
        html += u'</li>\n'
        return ''.join(html)


def _clean_varnames(s):
    # Remove invalid characters
    s = re.sub('[^0-9a-zA-Z_]', '', s)

    # add report_ at the beginning so that the javascript class names
    # are valid ones
    return 'report_' + s


def _recursive_search(path, pattern):
    """Auxiliary function for recursive_search of the directory."""
    filtered_files = list()
    for dirpath, dirnames, files in os.walk(path):
        for f in fnmatch.filter(files, pattern):
            # only the following file types are supported
            # this ensures equitable distribution of jobs
            if f.endswith(tuple(VALID_EXTENSIONS)):
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
                out_fname = op.join(
                    self.app.builder.outdir,
                    op.relpath(op.dirname(block_vars['target_file']),
                               self.app.builder.srcdir), html_fname)
                self.files[report.fname] = out_fname
                # embed links/iframe
                data = _SCRAPER_TEXT.format(html_fname)
                return data
        return ''

    def copyfiles(self, *args, **kwargs):
        for key, value in self.files.items():
            copyfile(key, value)
