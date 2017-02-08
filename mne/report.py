"""Generate html report from MNE database."""

# Authors: Alex Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Mainak Jas <mainak@neuro.hut.fi>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import os
import os.path as op
import fnmatch
import re
import codecs
import time
from glob import glob
import base64
from datetime import datetime as dt

import numpy as np

from . import read_evokeds, read_events, pick_types, read_cov
from .io import Raw, read_info
from .utils import (_TempDir, logger, verbose, get_subjects_dir, warn,
                    _import_mlab)
from .viz import plot_events, plot_trans, plot_cov
from .viz._3d import _plot_mri_contours
from .forward import read_forward_solution
from .epochs import read_epochs
from .minimum_norm import read_inverse_operator
from .parallel import parallel_func, check_n_jobs

from .externals.tempita import HTMLTemplate, Template
from .externals.six import BytesIO
from .externals.six import moves

VALID_EXTENSIONS = ['raw.fif', 'raw.fif.gz', 'sss.fif', 'sss.fif.gz',
                    '-eve.fif', '-eve.fif.gz', '-cov.fif', '-cov.fif.gz',
                    '-trans.fif', '-trans.fif.gz', '-fwd.fif', '-fwd.fif.gz',
                    '-epo.fif', '-epo.fif.gz', '-inv.fif', '-inv.fif.gz',
                    '-ave.fif', '-ave.fif.gz', 'T1.mgz']
SECTION_ORDER = ['raw', 'events', 'epochs', 'evoked', 'covariance', 'trans',
                 'mri', 'forward', 'inverse']

###############################################################################
# PLOTTING FUNCTIONS


def _fig_to_img(function=None, fig=None, image_format='png',
                scale=None, **kwargs):
    """Wrapper function to plot figure and create a binary image."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    if not isinstance(fig, Figure) and function is None:
        from scipy.misc import imread
        mlab = None
        try:
            mlab = _import_mlab()
        # on some systems importing Mayavi raises SystemExit (!)
        except Exception as e:
            warn('Could not import mayavi (%r). Trying to render'
                 '`mayavi.core.scene.Scene` figure instances'
                 ' will throw an error.' % (e,))
        tempdir = _TempDir()
        temp_fname = op.join(tempdir, 'test')
        if fig.scene is not None:
            fig.scene.save_png(temp_fname)
            img = imread(temp_fname)
            os.remove(temp_fname)
        else:  # Testing mode
            img = np.zeros((2, 2, 3))

        mlab.close(fig)
        fig = plt.figure()
        plt.imshow(img)
        plt.axis('off')

    if function is not None:
        plt.close('all')
        fig = function(**kwargs)
    output = BytesIO()
    if scale is not None:
        _scale_mpl_figure(fig, scale)
    logger.debug('Saving figure %s with dpi %s'
                 % (fig.get_size_inches(), fig.get_dpi()))
    # We don't use bbox_inches='tight' here because it can break
    # newer matplotlib, and should only save a little bit of space
    fig.savefig(output, format=image_format, dpi=fig.get_dpi())
    plt.close(fig)
    output = output.getvalue()
    return (output if image_format == 'svg' else
            base64.b64encode(output).decode('ascii'))


def _scale_mpl_figure(fig, scale):
    """Magic scaling helper.

    Keeps font-size and artist sizes constant
    0.5 : current font - 4pt
    2.0 : current font + 4pt

    XXX it's unclear why this works, but good to go for most cases
    """
    fig.set_size_inches(fig.get_size_inches() * scale)
    fig.set_dpi(fig.get_dpi() * scale)
    import matplotlib as mpl
    if scale >= 1:
        sfactor = scale ** 2
    elif scale < 1:
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
    from scipy.misc import imread
    import matplotlib.pyplot as plt
    import mayavi
    fig = function(**kwargs)

    assert isinstance(fig, mayavi.core.scene.Scene)

    views = [(90, 90), (0, 90), (0, -90)]
    fig2, axes = plt.subplots(1, len(views))
    for view, ax in zip(views, axes):
        mayavi.mlab.view(view[0], view[1])
        # XXX: save_bmp / save_png / ...
        tempdir = _TempDir()
        temp_fname = op.join(tempdir, 'test.png')
        if fig.scene is not None:
            fig.scene.save_png(temp_fname)
            im = imread(temp_fname)
        else:  # Testing mode
            im = np.zeros((2, 2, 3))
        ax.imshow(im)
        ax.axis('off')

    mayavi.mlab.close(fig)
    img = _fig_to_img(fig=fig2)
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
                         'sss.fif', 'sss.fif.gz')):
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


def _iterate_files(report, fnames, info, cov, baseline, sfreq, on_error):
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
                               'sss.fif', 'sss.fif.gz')):
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
                    html = report._render_whitened_evoked(fname, cov, baseline)
                    report_fname = fname + ' (whitened)'
                    report_sectionlabel = 'evoked'
                    _update_html(html, report_fname, report_sectionlabel)

                html = report._render_evoked(fname, baseline)
                report_fname = fname
                report_sectionlabel = 'evoked'
            elif fname.endswith(('-eve.fif', '-eve.fif.gz')):
                html = report._render_eve(fname, sfreq)
                report_fname = fname
                report_sectionlabel = 'events'
            elif fname.endswith(('-epo.fif', '-epo.fif.gz')):
                html = report._render_epochs(fname)
                report_fname = fname
                report_sectionlabel = 'epochs'
            elif (fname.endswith(('-cov.fif', '-cov.fif.gz')) and
                  report.info_fname is not None):
                html = report._render_cov(fname, info)
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

###############################################################################
# IMAGE FUNCTIONS


def _build_image(data, cmap='gray'):
    """Build an image encoded in base64."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    figsize = data.shape[::-1]
    if figsize[0] == 1:
        figsize = tuple(figsize[1:])
        data = data[:, :, 0]
    fig = Figure(figsize=figsize, dpi=1.0, frameon=False)
    FigureCanvas(fig)
    cmap = getattr(plt.cm, cmap, plt.cm.gray)
    fig.figimage(data, cmap=cmap)
    output = BytesIO()
    fig.savefig(output, dpi=1.0, format='png')
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


def _iterate_mri_slices(name, ind, global_id, slides_klass, data, cmap,
                        image_format='png'):
    """Auxiliary function for parallel processing of mri slices."""
    img_klass = 'slideimg-%s' % name

    caption = u'Slice %s %s' % (name, ind)
    slice_id = '%s-%s-%s' % (name, global_id, ind)
    div_klass = 'span12 %s' % slides_klass
    img = _build_image(data, cmap=cmap)
    first = True if ind == 0 else False
    html = _build_html_image(img, slice_id, div_klass,
                             img_klass, caption, first)
    return ind, html


###############################################################################
# HTML functions

def _build_html_image(img, id, div_klass, img_klass, caption=None, show=True):
    """Build a html image from a slice array."""
    html = []
    add_style = u'' if show else u'style="display: none"'
    html.append(u'<li class="%s" id="%s" %s>' % (div_klass, id, add_style))
    html.append(u'<div class="thumbnail">')
    html.append(u'<img class="%s" alt="" style="width:90%%;" '
                'src="data:image/png;base64,%s">'
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
                <ul class="thumbnails">
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
    return slider_template.substitute(slider_id=slider_id,
                                      klass=slides_klass,
                                      step=slices_range[1] - slices_range[0],
                                      minvalue=slices_range[0],
                                      maxvalue=slices_range[-1],
                                      startvalue=start_value)


###############################################################################
# HTML scan renderer

header_template = Template(u"""
<!DOCTYPE html>
<html lang="fr">
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
      Powered by <a href="http://martinos.org/mne">MNE.
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


class Report(object):
    """Object for rendering HTML.

    Parameters
    ----------
    info_fname : str
        Name of the file containing the info dictionary.
    subjects_dir : str | None
        Path to the SUBJECTS_DIR. If None, the path is obtained by using
        the environment variable SUBJECTS_DIR.
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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Notes
    -----
    To toggle the show/hide state of all sections in the html report, press 't'

    .. versionadded:: 0.8.0
    """

    def __init__(self, info_fname=None, subjects_dir=None,
                 subject=None, title=None, cov_fname=None, baseline=None,
                 verbose=None):  # noqa: D102
        self.info_fname = info_fname
        self.cov_fname = cov_fname
        self.baseline = baseline
        self.subjects_dir = get_subjects_dir(subjects_dir, raise_error=False)
        self.subject = subject
        self.title = title
        self.verbose = verbose

        self.initial_id = 0
        self.html = []
        self.fnames = []  # List of file names rendered
        self.sections = []  # List of sections
        self._sectionlabels = []  # Section labels
        self._sectionvars = {}  # Section variable names in js
        # boolean to specify if sections should be ordered in natural
        # order of processing (raw -> events ... -> inverse)
        self._sort_sections = False

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
        """The number of items in report."""
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
                             'length or comments should be None.')
        elif len(captions) != len(items):
            raise ValueError('Captions and report items must have the same '
                             'length.')

        # Book-keeping of section names
        if section not in self.sections:
            self.sections.append(section)
            self._sectionvars[section] = _clean_varnames(section)

        return items, captions, comments

    def _add_figs_to_section(self, figs, captions, section='custom',
                             image_format='png', scale=None, comments=None):
        """Auxiliary method for `add_section` and `add_figs_to_section`."""
        figs, captions, comments = self._validate_input(figs, captions,
                                                        section, comments)
        _check_scale(scale)
        for fig, caption, comment in zip(figs, captions, comments):
            caption = 'custom plot' if caption == '' else caption
            sectionvar = self._sectionvars[section]
            global_id = self._get_id()
            div_klass = self._sectionvars[section]
            img_klass = self._sectionvars[section]

            img = _fig_to_img(fig=fig, scale=scale,
                              image_format=image_format)
            html = image_template.substitute(img=img, id=global_id,
                                             div_klass=div_klass,
                                             img_klass=img_klass,
                                             caption=caption,
                                             show=True,
                                             image_format=image_format,
                                             comment=comment)
            self.fnames.append('%s-#-%s-#-custom' % (caption, sectionvar))
            self._sectionlabels.append(sectionvar)
            self.html.append(html)

    def add_figs_to_section(self, figs, captions, section='custom',
                            scale=None, image_format='png', comments=None):
        """Append custom user-defined figures.

        Parameters
        ----------
        figs : list of figures.
            Each figure in the list can be an instance of
            matplotlib.pyplot.Figure, mayavi.core.scene.Scene,
            or np.ndarray (images read in using scipy.imread).
        captions : list of str
            A list of captions to the figures.
        section : str
            Name of the section. If section already exists, the figures
            will be appended to the end of the section
        scale : float | None | callable
            Scale the images maintaining the aspect ratio.
            If None, no scaling is applied. If float, scale will determine
            the relative scaling (might not work for scale <= 1 depending on
            font sizes). If function, should take a figure object as input
            parameter. Defaults to None.
        image_format : {'png', 'svg'}
            The image format to be used for the report. Defaults to 'png'.
        comments : None | str | list of str
            A string of text or a list of strings of text to be appended after
            the figure.
        """
        return self._add_figs_to_section(figs=figs, captions=captions,
                                         section=section, scale=scale,
                                         image_format=image_format,
                                         comments=comments)

    def add_images_to_section(self, fnames, captions, scale=None,
                              section='custom', comments=None):
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

            if image_format not in ['png', 'gif', 'svg']:
                raise ValueError("Unknown image format. Only 'png', 'gif' or "
                                 "'svg' are supported. Got %s" % image_format)

            # Convert image to binary string.
            output = BytesIO()
            with open(fname, 'rb') as f:
                output.write(f.read())
            img = base64.b64encode(output.getvalue()).decode('ascii')
            html = image_template.substitute(img=img, id=global_id,
                                             image_format=image_format,
                                             div_klass=div_klass,
                                             img_klass=img_klass,
                                             caption=caption,
                                             width=scale,
                                             comment=comment,
                                             show=True)
            self.fnames.append('%s-#-%s-#-custom' % (caption, sectionvar))
            self._sectionlabels.append(sectionvar)
            self.html.append(html)

    def add_htmls_to_section(self, htmls, captions, section='custom'):
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

            self.fnames.append('%s-#-%s-#-custom' % (caption, sectionvar))
            self._sectionlabels.append(sectionvar)
            self.html.append(
                html_template.substitute(div_klass=div_klass, id=global_id,
                                         caption=caption, html=html))

    def add_bem_to_section(self, subject, caption='BEM', section='bem',
                           decim=2, n_jobs=1, subjects_dir=None):
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
        n_jobs : int
          Number of jobs to run in parallel.
        subjects_dir : str | None
            Path to the SUBJECTS_DIR. If None, the path is obtained by using
            the environment variable SUBJECTS_DIR.

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

        self.fnames.append('%s-#-%s-#-custom' % (caption[0], sectionvar))
        self._sectionlabels.append(sectionvar)
        self.html.extend(html)

    def add_slider_to_section(self, figs, captions=None, section='custom',
                              title='Slider', scale=None, image_format='png'):
        """Render a slider of figs to the report.

        Parameters
        ----------
        figs : list of figures.
            Each figure in the list can be an instance of
            matplotlib.pyplot.Figure, mayavi.core.scene.Scene,
            or np.ndarray (images read in using scipy.imread).
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
        image_format : {'png', 'svg'}
            The image format to be used for the report. Defaults to 'png'.

        Notes
        -----
        .. versionadded:: 0.10.0
        """
        _check_scale(scale)
        if not isinstance(figs[0], list):
            figs = [figs]
        else:
            raise NotImplementedError('`add_slider_to_section` '
                                      'can only add one slider at a time.')
        figs, _, _ = self._validate_input(figs, section, section)

        sectionvar = self._sectionvars[section]
        self._sectionlabels.append(sectionvar)
        global_id = self._get_id()
        img_klass = self._sectionvars[section]
        name = 'slider'

        html = []
        slides_klass = '%s-%s' % (name, global_id)
        div_klass = 'span12 %s' % slides_klass

        if isinstance(figs[0], list):
            figs = figs[0]
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
            img = _fig_to_img(fig=fig, scale=scale, image_format=image_format)
            slice_id = '%s-%s-%s' % (name, global_id, sl[ii])
            first = True if ii == 0 else False
            slices.append(_build_html_image(img, slice_id, div_klass,
                          img_klass, caption, first))
        # Render the slider
        slider_id = 'select-%s-%s' % (name, global_id)
        # Render the slices
        image_html = u'\n'.join(slices)
        html.append(_build_html_slider(sl, slides_klass, slider_id,
                                       start_value=0))
        html = '\n'.join(html)

        slider_klass = sectionvar
        self.html.append(
            slider_full_template.substitute(id=global_id, title=title,
                                            div_klass=slider_klass,
                                            slider_id=slider_id, html=html,
                                            image_html=image_html))

        self.fnames.append('%s-#-%s-#-custom' % (section, sectionvar))

    ###########################################################################
    # HTML rendering
    def _render_one_axis(self, slices_iter, name, global_id, cmap,
                         n_elements, n_jobs):
        """Render one axis of the array."""
        global_id = global_id or name
        html = []
        slices, slices_range = [], []
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
        html.append(u'<ul class="thumbnails">')
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
        inc_fnames = ['jquery-1.10.2.min.js', 'jquery-ui.min.js',
                      'bootstrap.min.js', 'jquery-ui.min.css',
                      'bootstrap.min.css']

        include = list()
        for inc_fname in inc_fnames:
            logger.info('Embedding : %s' % inc_fname)
            f = open(op.join(op.dirname(__file__), 'html', inc_fname),
                     'r')
            if inc_fname.endswith('.js'):
                include.append(u'<script type="text/javascript">' +
                               f.read() + u'</script>')
            elif inc_fname.endswith('.css'):
                include.append(u'<style type="text/css">' +
                               f.read() + u'</style>')
            f.close()

        self.include = ''.join(include)

    @verbose
    def parse_folder(self, data_path, pattern='*.fif', n_jobs=1, mri_decim=2,
                     sort_sections=True, on_error='warn', verbose=None):
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
        n_jobs : int
          Number of jobs to run in parallel.
        mri_decim : int
            Use this decimation factor for generating MRI/BEM images
            (since it can be time consuming).
        sort_sections : bool
            If True, sort sections in the order: raw -> events -> epochs
             -> evoked -> covariance -> trans -> mri -> forward -> inverse.
        on_error : str
            What to do if a file cannot be rendered. Can be 'ignore',
            'warn' (default), or 'raise'.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more).
        """
        valid_errors = ['ignore', 'warn', 'raise']
        if on_error not in valid_errors:
            raise ValueError('on_error must be one of %s, not %s'
                             % (valid_errors, on_error))
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
            fnames.extend(_recursive_search(self.data_path, p))

        if self.info_fname is not None:
            info = read_info(self.info_fname)
            sfreq = info['sfreq']
        else:
            warn('`info_fname` not provided. Cannot render -cov.fif(.gz) and '
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
        r = parallel(p_fun(self, fname, info, cov, baseline, sfreq, on_error)
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
        if self.subjects_dir is not None and self.subject is not None:
            logger.info('Rendering BEM')
            self.html.append(self._render_bem(self.subject, self.subjects_dir,
                                              mri_decim, n_jobs))
            self.fnames.append('bem')
            self._sectionlabels.append('mri')
        else:
            warn('`subjects_dir` and `subject` not provided. Cannot render '
                 'MRI and -trans.fif(.gz) files.')

    def save(self, fname=None, open_browser=True, overwrite=False):
        """Save html report and open it in browser.

        Parameters
        ----------
        fname : str
            File name of the report.
        open_browser : bool
            Open html browser after saving if True.
        overwrite : bool
            If True, overwrite report if it already exists.
        """
        if fname is None:
            if not hasattr(self, 'data_path'):
                self.data_path = op.dirname(__file__)
                warn('`data_path` not provided. Using %s instead'
                     % self.data_path)
            fname = op.realpath(op.join(self.data_path, 'report.html'))
        else:
            fname = op.realpath(fname)

        self._render_toc()

        html = footer_template.substitute(date=time.strftime("%B %d, %Y"),
                                          current_year=time.strftime("%Y"))
        self.html.append(html)

        if not overwrite and op.isfile(fname):
            msg = ('Report already exists at location %s. '
                   'Overwrite it (y/[n])? '
                   % fname)
            answer = moves.input(msg)
            if answer.lower() == 'y':
                overwrite = True

        if overwrite or not op.isfile(fname):
            logger.info('Saving report to location %s' % fname)
            fobj = codecs.open(fname, 'w', 'utf-8')
            fobj.write(_fix_global_ids(u''.join(self.html)))
            fobj.close()

            # remove header, TOC and footer to allow more saves
            self.html.pop(0)
            self.html.pop(0)
            self.html.pop()

        if open_browser:
            import webbrowser
            webbrowser.open_new_tab('file://' + fname)

        return fname

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

        html_header = header_template.substitute(title=self.title,
                                                 include=self.include,
                                                 sections=self.sections,
                                                 sectionvars=self._sectionvars)
        self.html.insert(0, html_header)  # Insert header at position 0
        self.html.insert(1, html_toc)  # insert TOC

    def _render_array(self, array, global_id=None, cmap='gray',
                      limits=None, n_jobs=1):
        """Render mri without bem contours."""
        html = []
        html.append(u'<div class="row">')
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
        html.append(u'</div>')
        html.append(u'<div class="row">')
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
        """Render one axis of bem contours."""
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
                          img_klass, caption, first))

        # Render the slider
        slider_id = 'select-%s-%s' % (name, global_id)
        html.append(u'<div id="%s"></div>' % slider_id)
        html.append(u'<ul class="thumbnails">')
        # Render the slices
        html.append(u'\n'.join(slices))
        html.append(u'</ul>')
        html.append(_build_html_slider(sl, slides_klass, slider_id))
        html.append(u'</div>')
        return '\n'.join(html)

    def _render_image(self, image, cmap='gray', n_jobs=1):
        """Render one slice of mri without bem."""
        import nibabel as nib

        global_id = self._get_id()

        if 'mri' not in self.sections:
            self.sections.append('mri')
            self._sectionvars['mri'] = 'mri'

        nim = nib.load(image)
        data = nim.get_data()
        shape = data.shape
        limits = {'sagittal': range(0, shape[0], 2),
                  'axial': range(0, shape[1], 2),
                  'coronal': range(0, shape[2], 2)}
        name = op.basename(image)
        html = u'<li class="mri" id="%d">\n' % global_id
        html += u'<h2>%s</h2>\n' % name
        html += self._render_array(data, global_id=global_id,
                                   cmap=cmap, limits=limits,
                                   n_jobs=n_jobs)
        html += u'</li>\n'
        return html

    def _render_raw(self, raw_fname):
        """Render raw."""
        global_id = self._get_id()
        div_klass = 'raw'
        caption = u'Raw : %s' % raw_fname

        raw = Raw(raw_fname)

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
            meas_date = dt.fromtimestamp(meas_date[0]).strftime("%B %d, %Y")
        tmin = raw.first_samp / raw.info['sfreq']
        tmax = raw.last_samp / raw.info['sfreq']

        html = raw_template.substitute(div_klass=div_klass,
                                       id=global_id,
                                       caption=caption,
                                       info=raw.info,
                                       meas_date=meas_date,
                                       n_eeg=n_eeg, n_grad=n_grad,
                                       n_mag=n_mag, eog=eog,
                                       ecg=ecg, tmin=tmin, tmax=tmax)
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

    def _render_evoked(self, evoked_fname, baseline=None, figsize=None):
        """Render evoked."""
        logger.debug('Evoked: Reading %s' % evoked_fname)
        evokeds = read_evokeds(evoked_fname, baseline=baseline, verbose=False)

        html = []
        for ei, ev in enumerate(evokeds):
            global_id = self._get_id()

            kwargs = dict(show=False)
            logger.debug('Evoked: Plotting instance %s/%s'
                         % (ei + 1, len(evokeds)))
            img = _fig_to_img(ev.plot, **kwargs)

            caption = u'Evoked : %s (%s)' % (evoked_fname, ev.comment)
            div_klass = 'evoked'
            img_klass = 'evoked'
            show = True
            html.append(image_template.substitute(img=img, id=global_id,
                                                  div_klass=div_klass,
                                                  img_klass=img_klass,
                                                  caption=caption,
                                                  show=show))
            has_types = []
            if len(pick_types(ev.info, meg=False, eeg=True)) > 0:
                has_types.append('eeg')
            if len(pick_types(ev.info, meg='grad', eeg=False)) > 0:
                has_types.append('grad')
            if len(pick_types(ev.info, meg='mag', eeg=False)) > 0:
                has_types.append('mag')
            for ch_type in has_types:
                logger.debug('    Topomap type %s' % ch_type)
                img = _fig_to_img(ev.plot_topomap, ch_type=ch_type, **kwargs)
                caption = u'Topomap (ch_type = %s)' % ch_type
                html.append(image_template.substitute(img=img,
                                                      div_klass=div_klass,
                                                      img_klass=img_klass,
                                                      caption=caption,
                                                      show=show))
        logger.debug('Evoked: done')
        return '\n'.join(html)

    def _render_eve(self, eve_fname, sfreq=None):
        """Render events."""
        global_id = self._get_id()
        events = read_events(eve_fname)

        kwargs = dict(events=events, sfreq=sfreq, show=False)
        img = _fig_to_img(plot_events, **kwargs)

        caption = 'Events : ' + eve_fname
        div_klass = 'events'
        img_klass = 'events'
        show = True

        html = image_template.substitute(img=img, id=global_id,
                                         div_klass=div_klass,
                                         img_klass=img_klass,
                                         caption=caption,
                                         show=show)
        return html

    def _render_epochs(self, epo_fname):
        """Render epochs."""
        global_id = self._get_id()

        epochs = read_epochs(epo_fname)
        kwargs = dict(subject=self.subject, show=False)
        img = _fig_to_img(epochs.plot_drop_log, **kwargs)
        caption = 'Epochs : ' + epo_fname
        div_klass = 'epochs'
        img_klass = 'epochs'
        show = True
        html = image_template.substitute(img=img, id=global_id,
                                         div_klass=div_klass,
                                         img_klass=img_klass,
                                         caption=caption,
                                         show=show)
        return html

    def _render_cov(self, cov_fname, info_fname):
        """Render cov."""
        global_id = self._get_id()
        cov = read_cov(cov_fname)
        fig, _ = plot_cov(cov, info_fname, show=False)
        img = _fig_to_img(fig=fig)
        caption = 'Covariance : %s (n_samples: %s)' % (cov_fname, cov.nfree)
        div_klass = 'covariance'
        img_klass = 'covariance'
        show = True
        html = image_template.substitute(img=img, id=global_id,
                                         div_klass=div_klass,
                                         img_klass=img_klass,
                                         caption=caption,
                                         show=show)
        return html

    def _render_whitened_evoked(self, evoked_fname, noise_cov, baseline):
        """Render whitened evoked."""
        global_id = self._get_id()

        evokeds = read_evokeds(evoked_fname, verbose=False)

        html = []
        for ev in evokeds:

            ev = read_evokeds(evoked_fname, ev.comment, baseline=baseline,
                              verbose=False)

            global_id = self._get_id()

            kwargs = dict(noise_cov=noise_cov, show=False)
            img = _fig_to_img(ev.plot_white, **kwargs)

            caption = u'Whitened evoked : %s (%s)' % (evoked_fname, ev.comment)
            div_klass = 'evoked'
            img_klass = 'evoked'
            show = True
            html.append(image_template.substitute(img=img, id=global_id,
                                                  div_klass=div_klass,
                                                  img_klass=img_klass,
                                                  caption=caption,
                                                  show=show))
        return '\n'.join(html)

    def _render_trans(self, trans, path, info, subject,
                      subjects_dir, image_format='png'):
        """Render trans."""
        kwargs = dict(info=info, trans=trans, subject=subject,
                      subjects_dir=subjects_dir)
        try:
            img = _iterate_trans_views(function=plot_trans, **kwargs)
        except IOError:
            img = _iterate_trans_views(function=plot_trans, source='head',
                                       **kwargs)

        if img is not None:
            global_id = self._get_id()
            caption = 'Trans : ' + trans
            div_klass = 'trans'
            img_klass = 'trans'
            show = True
            html = image_template.substitute(img=img, id=global_id,
                                             div_klass=div_klass,
                                             img_klass=img_klass,
                                             caption=caption,
                                             width=75,
                                             show=show)
            return html

    def _render_bem(self, subject, subjects_dir, decim, n_jobs,
                    section='mri', caption='BEM'):
        """Render mri+bem."""
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
            return self._render_image(mri_fname, cmap='gray', n_jobs=n_jobs)

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
            return self._render_image(mri_fname, cmap='gray')
        # XXX : find a better way to get max range of slices
        nim = nib.load(mri_fname)
        data = nim.get_data()
        shape = data.shape
        del data  # free up memory

        html = []

        global_id = self._get_id()

        if section == 'mri' and 'mri' not in self.sections:
            self.sections.append('mri')
            self._sectionvars['mri'] = 'mri'

        name = caption

        html += u'<li class="mri" id="%d">\n' % global_id
        html += u'<h2>%s</h2>\n' % name
        html += u'<div class="row">'
        html += self._render_one_bem_axis(mri_fname, surf_fnames, global_id,
                                          shape, 'axial', decim, n_jobs)
        html += self._render_one_bem_axis(mri_fname, surf_fnames, global_id,
                                          shape, 'sagittal', decim, n_jobs)
        html += u'</div><div class="row">'
        html += self._render_one_bem_axis(mri_fname, surf_fnames, global_id,
                                          shape, 'coronal', decim, n_jobs)
        html += u'</div>'
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
    html = re.sub('id="\d+"', 'id="###"', html)
    global_id = 1
    while len(re.findall('id="###"', html)) > 0:
        html = re.sub('id="###"', 'id="%s"' % global_id, html, count=1)
        global_id += 1
    return html
