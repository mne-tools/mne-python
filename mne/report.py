"""Generate html report from MNE database
"""

# Authors: Alex Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import os
import os.path as op
import fnmatch
import re
import numpy as np
import time
from glob import glob
import warnings
import base64

from . import read_evokeds, read_events, Covariance
from .io import Raw, read_info
from .utils import _TempDir, logger, verbose, get_subjects_dir
from .viz import plot_events, plot_trans, plot_cov
from .viz._3d import _plot_mri_contours
from .forward import read_forward_solution
from .epochs import read_epochs
from .minimum_norm import read_inverse_operator

from .externals.decorator import decorator
from .externals.tempita import HTMLTemplate, Template
from .externals.six import BytesIO
from .externals.six import moves

tempdir = _TempDir()
temp_fname = op.join(tempdir, 'test')

###############################################################################
# PLOTTING FUNCTIONS


@decorator
def _check_report_mode(function, *args, **kwargs):
    """Check whether to actually render or not.

    Parameters
    ----------
    function : function
        Function to be decorated by setting the verbosity level.

    Returns
    -------
    dec : function
        The decorated function
    """

    if 'MNE_REPORT_TESTING' not in os.environ:
        return function(*args, **kwargs)
    else:
        return ''


@_check_report_mode
def _fig_to_img(function=None, fig=None, **kwargs):
    """Wrapper function to plot figure and
       for fig <-> binary image.
    """
    import matplotlib.pyplot as plt

    if function is not None:
        plt.close('all')
        fig = function(**kwargs)

    output = BytesIO()
    fig.savefig(output, format='png', bbox_inches='tight')
    plt.close('all')

    return base64.b64encode(output.getvalue()).decode('ascii')


@_check_report_mode
def _fig_to_mrislice(function, orig_size, **kwargs):
    import matplotlib.pyplot as plt
    from PIL import Image

    plt.close('all')
    fig = _plot_mri_contours(**kwargs)

    fig_size = fig.get_size_inches()
    w, h = orig_size[0], orig_size[1]
    w2, h2 = fig_size[0], fig_size[1]
    fig.set_size_inches([(w2 / w) * w, (w2 / w) * h])
    a = fig.gca()
    a.set_xticks([]), a.set_yticks([])
    plt.xlim(0, h), plt.ylim(w, 0)
    fig.savefig(temp_fname, bbox_inches='tight',
                pad_inches=0, format='png')
    Image.open(temp_fname).resize((w, h)).save(temp_fname,
                                               format='png')
    output = BytesIO()
    Image.open(temp_fname).save(output, format='png')
    return output.getvalue().encode('base64')


@_check_report_mode
def _iterate_trans_views(function, **kwargs):

    from PIL import Image
    import matplotlib.pyplot as plt
    import mayavi

    fig = function(**kwargs)

    if isinstance(fig, mayavi.core.scene.Scene):

        views = [(90, 90), (0, 90), (0, -90)]
        fig2, axes = plt.subplots(1, len(views))
        for view, ax in zip(views, axes):
            mayavi.mlab.view(view[0], view[1])
            # XXX: save_bmp / save_png / ...
            fig.scene.save_bmp(temp_fname)
            im = Image.open(temp_fname)
            ax.imshow(im)
            ax.axis('off')

        img = _fig_to_img(fig=fig2)
        mayavi.mlab.close(all=True)

        return img
    else:
        return None

###############################################################################
# TOC FUNCTIONS


def _is_bad_fname(fname):
    """Auxiliary function for identifying bad file naming patterns
       and highlighting them in red in the TOC.
    """
    if not fname.endswith(('-eve.fif', '-eve.fif.gz',
                           '-ave.fif', '-ave.fif.gz',
                           '-cov.fif', '-cov.fif.gz',
                           '-sol.fif',
                           '-fwd.fif', '-fwd.fif.gz',
                           '-inv.fif', '-inv.fif.gz',
                           '-src.fif',
                           '-trans.fif', '-trans.fif.gz',
                           'raw.fif', 'raw.fif.gz',
                           'sss.fif', 'sss.fif.gz',
                           '-epo.fif', 'T1.mgz',
                           'bem', 'custom')):
        return 'red'
    else:
        return ''


def _get_toc_property(fname):
    """Auxiliary function to assign class names to TOC
       list elements to allow toggling with buttons.
    """
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
    else:
        div_klass = fname.split('-#-')[1]
        tooltip = fname.split('-#-')[0]
        text = fname.split('-#-')[0]

    return div_klass, tooltip, text


###############################################################################
# IMAGE FUNCTIONS


def _build_image(data, cmap='gray'):
    """ Build an image encoded in base64 """

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
    return output.getvalue().encode('base64')


def _iterate_sagittal_slices(array, limits=None):
    """ Iterate sagittal slice """
    shape = array.shape[0]
    for ind in xrange(shape):
        if limits and ind not in limits:
            continue
        yield ind, array[ind, :, :]


def _iterate_axial_slices(array, limits=None):
    """ Iterate axial slice """
    shape = array.shape[1]
    for ind in xrange(shape):
        if limits and ind not in limits:
            continue
        yield ind, array[:, ind, :]


def _iterate_coronal_slices(array, limits=None):
    """ Iterate coronal slice """
    shape = array.shape[2]
    for ind in xrange(shape):
        if limits and ind not in limits:
            continue
        yield ind, np.flipud(np.rot90(array[:, :, ind]))


###############################################################################
# HTML functions

def _build_html_image(img, id, div_klass, img_klass, caption=None, show=True):
    """ Build a html image from a slice array """
    html = []
    add_style = '' if show else 'style="display: none"'
    html.append(u'<li class="%s" id="%s" %s>' % (div_klass, id, add_style))
    html.append(u'<div class="thumbnail">')
    html.append(u'<img class="%s" alt="" style="width:90%%;" '
                'src="data:image/png;base64,%s">'
                % (img_klass, img))
    html.append(u'</div>')
    if caption:
        html.append(u'<h4>%s</h4>' % caption)
    html.append(u'</li>')
    return '\n'.join(html)

slider_template = HTMLTemplate(u"""
<script>$("#{{slider_id}}").slider({
                       range: "min",
                       /*orientation: "vertical",*/
                       min: {{minvalue}},
                       max: {{maxvalue}},
                       step: 2,
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


def _build_html_slider(slices_range, slides_klass, slider_id):
    """ Build an html slider for a given slices range and a slices klass """
    startvalue = (slices_range[0] + slices_range[-1]) / 2 + 1
    return slider_template.substitute(slider_id=slider_id,
                                      klass=slides_klass,
                                      minvalue=slices_range[0],
                                      maxvalue=slices_range[-1],
                                      startvalue=startvalue)


###############################################################################
# HTML scan renderer

header_template = Template(u"""
<!DOCTYPE html>
<html lang="fr">
<head>
{{include}}
<script type="text/javascript">

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
    <div class="container">
        <h3 class="navbar-text" style="color:white">{{title}}</h3>
        <ul class="nav nav-pills navbar-right" style="margin-top: 7px;">

        {{for section in sections}}

        <li class="active {{section}}-btn">
           <a href="javascript:void(0)" onclick="togglebutton('.{{section}}')">
           {{section.capitalize() if section != 'mri' else 'MRI'}}</a>
        </li>

        {{endfor}}

        </ul>
    </div>
</nav>
""")

footer_template = HTMLTemplate(u"""
</div></body>
<div class="footer">
        &copy; Copyright 2012-2013, MNE Developers.
      Created on {{date}}.
      Powered by <a href="http://martinos.org/mne">MNE.
</div>
</html>
""")

image_template = Template(u"""

{{default interactive = False}}
{{default width = 50}}
{{default id = False}}

<li class="{{div_klass}}" {{if id}}id="{{id}}"{{endif}}
{{if not show}}style="display: none"{{endif}}>

{{if caption}}
<h4>{{caption}}</h4>
{{endif}}
<div class="thumbnail">
{{if not interactive}}
    <img alt="" style="width:{{width}}%;"
    src="data:image/png;base64,{{img}}">
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

toc_list = Template(u"""
<li class="{{div_klass}}">
    {{if id}}
        <a href="javascript:void(0)" onclick="window.location.hash={{id}};">
    {{endif}}
<span title="{{tooltip}}" style="color:{{color}}"> {{text}}</span>
{{if id}}</a>{{endif}}
</li>
""")


class Report(object):
    """Object for rendering HTML

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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    """

    def __init__(self, info_fname=None, subjects_dir=None, subject=None,
                 title=None, verbose=None):

        self.info_fname = info_fname
        self.subjects_dir = subjects_dir
        self.subject = subject
        self.title = title
        self.verbose = verbose

        self.initial_id = 0
        self.html = []
        self.fnames = []  # List of file names rendered
        self.sections = []  # List of sections
        self._sectionlabels = []  # Section labels

        self._init_render(verbose=self.verbose)  # Initialize the renderer

    def _get_id(self):
        self.initial_id += 1
        return self.initial_id

    def add_section(self, figs, captions, section='custom'):
        """Append custom user-defined figures.

        Parameters
        ----------
        figs : list of matplotlib.pyplot.Figure
            A list of figures to be included in the report.
        captions : list of str
            A list of captions to the figures.
        section : str
            Name of the section. If section already exists, the figures
            will be appended to the end of the section
        """

        if section not in self.sections:
            self.sections.append(section)

        for fig, caption in zip(figs, captions):
            global_id = self._get_id()
            div_klass = section
            img_klass = section
            img = _fig_to_img(fig=fig)
            html = image_template.substitute(img=img, id=global_id,
                                             div_klass=div_klass,
                                             img_klass=img_klass,
                                             caption=caption,
                                             show=True)
            self.fnames.append('%s-#-%s-#-custom' % (caption, section))
            self._sectionlabels.append(section)
            self.html.append(html)

    ###########################################################################
    # HTML rendering
    def _render_one_axe(self, slices_iter, name, global_id=None, cmap='gray'):
        """Render one axe of the array."""
        global_id = global_id or name
        html = []
        slices, slices_range = [], []
        first = True
        html.append(u'<div class="col-xs-6 col-md-4">')
        slides_klass = '%s-%s' % (name, global_id)
        img_klass = 'slideimg-%s' % name
        for ind, data in slices_iter:
            slices_range.append(ind)
            caption = u'Slice %s %s' % (name, ind)
            slice_id = '%s-%s-%s' % (name, global_id, ind)
            div_klass = 'span12 %s' % slides_klass
            img = _build_image(data, cmap=cmap)
            slices.append(_build_html_image(img, slice_id, div_klass,
                                            img_klass, caption,
                                            first))
            first = False
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
                include.append(u'<script type="text/javascript">'
                               + f.read() + u'</script>')
            elif inc_fname.endswith('.css'):
                include.append(u'<style type="text/css">'
                               + f.read() + u'</style>')
            f.close()

        self.include = ''.join(include)

    @verbose
    def parse_folder(self, data_path, pattern='*.fif', verbose=None):
        """Renders all the files in the folder.

        Parameters
        ----------
        data_path : str
            Path to the folder containing data whose HTML report will be
            created.
        pattern : str
            Filename pattern to include in the report. e.g., -ave.fif will
            include all evoked files.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
        """
        self.data_path = data_path

        if self.title is None:
            self.title = 'MNE Report for ...%s' % self.data_path[-20:]

        folders = []

        fnames = _recursive_search(self.data_path, pattern)

        if self.subjects_dir is not None and self.subject is not None:
            fnames += glob(op.join(self.subjects_dir, self.subject,
                           'mri', 'T1.mgz'))
        else:
            warnings.warn('`subjects_dir` and `subject` not provided.'
                          ' Cannot render MRI and -trans.fif(.gz) files.')

        if self.info_fname is not None:
            info = read_info(self.info_fname)
            sfreq = info['sfreq']
        else:
            warnings.warn('`info_fname` not provided. Cannot render'
                          '-cov.fif(.gz) and -trans.fif(.gz) files.')
            sfreq = None

        for fname in fnames:
            logger.info("Rendering : %s"
                        % op.join('...' + self.data_path[-20:],
                                  fname))
            try:
                if fname.endswith(('.mgz')):
                    self._render_bem(subject=self.subject,
                                     subjects_dir=self.subjects_dir)
                    self.fnames.append('bem')
                    self._sectionlabels.append('mri')
                elif fname.endswith(('raw.fif', 'raw.fif.gz',
                                     'sss.fif', 'sss.fif.gz')):
                    self._render_raw(fname)
                    self.fnames.append(fname)
                    self._sectionlabels.append('raw')
                elif fname.endswith(('-fwd.fif', '-fwd.fif.gz')):
                    self._render_forward(fname)
                    self._sectionlabels.append('forward')
                elif fname.endswith(('-inv.fif', '-inv.fif.gz')):
                    self._render_inverse(fname)
                    self._sectionlabels.append('inverse')
                elif fname.endswith(('-ave.fif', '-ave.fif.gz')):
                    self._render_evoked(fname)
                    self.fnames.append(fname)
                    self._sectionlabels.append('evoked')
                elif fname.endswith(('-eve.fif', '-eve.fif.gz')):
                    self._render_eve(fname, sfreq)
                    self.fnames.append(fname)
                    self._sectionlabels.append('events')
                elif fname.endswith(('-epo.fif', '-epo.fif.gz')):
                    self._render_epochs(fname)
                    self.fnames.append(fname)
                    self._sectionlabels.append('epochs')
                elif (fname.endswith(('-cov.fif', '-cov.fif.gz'))
                      and self.info_fname is not None):
                    self._render_cov(fname, info)
                    self.fnames.append(fname)
                    self._sectionlabels.append('covariance')
                elif (fname.endswith(('-trans.fif', '-trans.fif.gz'))
                      and self.info_fname is not None and self.subjects_dir
                      is not None and self.subject is not None):
                    self._render_trans(fname, self.data_path, info,
                                       self.subject, self.subjects_dir)
                    self.fnames.append(fname)
                    self._sectionlabels.append('trans')
                elif op.isdir(fname):
                    folders.append(fname)
                    logger.info(folders)
            except Exception as e:
                logger.info(e)

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
                warnings.warn('`data_path` not provided. Using %s instead'
                              % self.data_path)
            fname = op.realpath(op.join(self.data_path, 'report.html'))
        else:
            fname = op.realpath(fname)

        self._render_toc(verbose=self.verbose)

        html = footer_template.substitute(date=time.strftime("%B %d, %Y"))
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
            fobj = open(fname, 'w')
            fobj.write(_fix_global_ids(''.join(self.html)))
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

        logger.info('Rendering : Table of Contents')

        html_toc = u'<div id="container">'
        html_toc += u'<div id="toc"><center><h4>CONTENTS</h4></center>'

        global_id = 1

        # Sort by section
        html, fnames = [], []
        for section in self.sections:
            logger.info('%s' % section)
            for sectionlabel, this_html, fname in (zip(self._sectionlabels,
                                                   self.html, self.fnames)):
                if section == sectionlabel:
                    html.append(this_html)
                    fnames.append(fname)
                    logger.info('\t... %s' % fname[-20:])
                    color = _is_bad_fname(fname)
                    div_klass, tooltip, text = _get_toc_property(fname)

                    if fname.endswith(('.nii', '.nii.gz', '.mgh', '.mgz',
                                       'raw.fif', 'raw.fif.gz', 'sss.fif',
                                       'sss.fif.gz', '-eve.fif', '-eve.fif.gz',
                                       '-cov.fif', '-cov.fif.gz', '-trans.fif',
                                       'trans.fif.gz', '-fwd.fif',
                                       '-fwd.fif.gz', '-epo.fif',
                                       '-inv.fif', '-inv.fif.gz',
                                       '-epo.fif.gz', 'bem', 'custom')):
                        html_toc += toc_list.substitute(div_klass=div_klass,
                                                        id=global_id,
                                                        tooltip=tooltip,
                                                        color=color,
                                                        text=text)
                        global_id += 1

                    # loop through conditions for evoked
                    elif fname.endswith(('-ave.fif', '-ave.fif.gz')):
                       # XXX: remove redundant read_evokeds
                        evokeds = read_evokeds(fname, verbose=False)

                        html_toc += toc_list.substitute(div_klass=div_klass,
                                                        id=None, tooltip=fname,
                                                        color='#428bca',
                                                        text=
                                                        os.path.basename(fname)
                                                        )

                        html_toc += u'<li class="evoked"><ul>'
                        for ev in evokeds:
                            html_toc += toc_list.substitute(div_klass=
                                                            div_klass,
                                                            id=global_id,
                                                            tooltip=fname,
                                                            color=color,
                                                            text=ev.comment)
                            global_id += 1
                        html_toc += u'</ul></li>'

        html_toc += u'\n</ul></div>'
        html_toc += u'<div id="content">'

        # The sorted html (according to section)
        self.html = html
        self.fnames = fnames

        html_header = header_template.substitute(title=self.title,
                                                 include=self.include,
                                                 sections=self.sections)
        self.html.insert(0, html_header)  # Insert header at position 0
        self.html.insert(1, html_toc)  # insert TOC

    def _render_array(self, array, global_id=None, cmap='gray',
                      limits=None):
        html = []
        html.append(u'<div class="row">')
        # Axial
        limits = limits or {}
        axial_limit = limits.get('axial')
        axial_slices_gen = _iterate_axial_slices(array, axial_limit)
        html.append(
            self._render_one_axe(axial_slices_gen, 'axial', global_id, cmap))
        # Sagittal
        sagittal_limit = limits.get('sagittal')
        sagittal_slices_gen = _iterate_sagittal_slices(array, sagittal_limit)
        html.append(self._render_one_axe(sagittal_slices_gen, 'sagittal',
                    global_id, cmap))
        html.append(u'</div>')
        html.append(u'<div class="row">')
        # Coronal
        coronal_limit = limits.get('coronal')
        coronal_slices_gen = _iterate_coronal_slices(array, coronal_limit)
        html.append(
            self._render_one_axe(coronal_slices_gen, 'coronal',
                                 global_id, cmap))
        # Close section
        html.append(u'</div>')
        return '\n'.join(html)

    def _render_one_bem_axe(self, mri_fname, surf_fnames, global_id,
                            shape, orientation='coronal'):

        orientation_name2axis = dict(sagittal=0, axial=1, coronal=2)
        orientation_axis = orientation_name2axis[orientation]
        n_slices = shape[orientation_axis]
        orig_size = np.roll(shape, orientation_axis)[[1, 2]]

        name = orientation
        html, img = [], []
        slices, slices_range = [], []
        first = True
        html.append(u'<div class="col-xs-6 col-md-4">')
        slides_klass = '%s-%s' % (name, global_id)
        img_klass = 'slideimg-%s' % name
        for sl in range(0, n_slices, 2):
            logger.info('Rendering BEM contours : orientation = %s, '
                        'slice = %d' % (orientation, sl))
            slices_range.append(sl)
            caption = u'Slice %s %s' % (name, sl)
            slice_id = '%s-%s-%s' % (name, global_id, sl)
            div_klass = 'span12 %s' % slides_klass

            kwargs = dict(mri_fname=mri_fname, surf_fnames=surf_fnames,
                          orientation=orientation, slices=[sl],
                          show=False)
            img = _fig_to_mrislice(function=_plot_mri_contours,
                                   orig_size=orig_size, **kwargs)
            slices.append(_build_html_image(img, slice_id, div_klass,
                                            img_klass, caption,
                                            first))
            first = False

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

    def _render_image(self, image, cmap='gray'):

        import nibabel as nib

        global_id = self._get_id()

        if 'mri' not in self.sections:
            self.sections.append('mri')

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
                                   cmap=cmap, limits=limits)
        html += u'</li>\n'
        self.html.append(html)
        return html

    def _render_raw(self, raw_fname):

        if 'raw' not in self.sections:
            self.sections.append('raw')

        global_id = self._get_id()
        div_klass = 'raw'
        caption = u'Raw : %s' % raw_fname

        raw = Raw(raw_fname)

        repr_raw = re.sub('>', '', re.sub('<', '', repr(raw)))
        repr_info = re.sub('\\n', '\\n</br>',
                           re.sub('>', '',
                                  re.sub('<', '',
                                         repr(raw.info))))

        repr_html = repr_raw + '%s<br/>%s' % (repr_raw, repr_info)

        html = repr_template.substitute(div_klass=div_klass,
                                        id=global_id,
                                        caption=caption,
                                        repr=repr_html)
        self.html.append(html)

    def _render_forward(self, fwd_fname):

        div_klass = 'forward'
        caption = u'Forward: %s' % fwd_fname
        fwd = read_forward_solution(fwd_fname)
        repr_fwd = re.sub('>', '', re.sub('<', '', repr(fwd)))
        global_id = self._get_id()
        html = repr_template.substitute(div_klass=div_klass,
                                        id=global_id,
                                        caption=caption,
                                        repr=repr_fwd)
        self.html.append(html)
        self.fnames.append(fwd_fname)

        if 'forward' not in self.sections:
            self.sections.append('forward')

    def _render_inverse(self, inv_fname):

        div_klass = 'inverse'
        caption = u'Inverse: %s' % inv_fname
        inv = read_inverse_operator(inv_fname)
        repr_inv = re.sub('>', '', re.sub('<', '', repr(inv)))
        global_id = self._get_id()
        html = repr_template.substitute(div_klass=div_klass,
                                        id=global_id,
                                        caption=caption,
                                        repr=repr_inv)
        self.html.append(html)
        self.fnames.append(inv_fname)

        if 'inverse' not in self.sections:
            self.sections.append('inverse')

    def _render_evoked(self, evoked_fname, figsize=None):

        if 'evoked' not in self.sections:
            self.sections.append('evoked')

        evokeds = read_evokeds(evoked_fname, verbose=False)

        html = []
        for ev in evokeds:
            global_id = self._get_id()

            kwargs = dict(show=False)
            img = _fig_to_img(function=ev.plot, **kwargs)

            caption = u'Evoked : %s (%s)' % (evoked_fname, ev.comment)
            div_klass = 'evoked'
            img_klass = 'evoked'
            show = True
            html.append(image_template.substitute(img=img, id=global_id,
                                                  div_klass=div_klass,
                                                  img_klass=img_klass,
                                                  caption=caption,
                                                  show=show))

            for ch_type in ['eeg', 'grad', 'mag']:
                kwargs = dict(ch_type=ch_type, show=False)
                img = _fig_to_img(function=ev.plot_topomap, **kwargs)
                caption = u'Topomap (ch_type = %s)' % ch_type
                html.append(image_template.substitute(img=img,
                                                      div_klass=div_klass,
                                                      img_klass=img_klass,
                                                      caption=caption,
                                                      show=show))

        self.html.append('\n'.join(html))

    def _render_eve(self, eve_fname, sfreq=None):

        if 'events' not in self.sections:
            self.sections.append('events')

        global_id = self._get_id()
        events = read_events(eve_fname)

        kwargs = dict(events=events, sfreq=sfreq, show=False)
        img = _fig_to_img(function=plot_events, **kwargs)

        caption = 'Events : ' + eve_fname
        div_klass = 'events'
        img_klass = 'events'
        show = True

        html = image_template.substitute(img=img, id=global_id,
                                         div_klass=div_klass,
                                         img_klass=img_klass,
                                         caption=caption,
                                         show=show)
        self.html.append(html)

    def _render_epochs(self, epo_fname):

        if 'epochs' not in self.sections:
            self.sections.append('epochs')

        global_id = self._get_id()

        epochs = read_epochs(epo_fname)
        kwargs = dict(subject=self.subject, show=False, return_fig=True)
        img = _fig_to_img(function=epochs.plot_drop_log, **kwargs)
        caption = 'Epochs : ' + epo_fname
        div_klass = 'epochs'
        img_klass = 'epochs'
        show = True
        html = image_template.substitute(img=img, id=global_id,
                                         div_klass=div_klass,
                                         img_klass=img_klass,
                                         caption=caption,
                                         show=show)
        self.html.append(html)

    def _render_cov(self, cov_fname, info_fname):

        if 'covariance' not in self.sections:
            self.sections.append('covariance')

        global_id = self._get_id()
        cov = Covariance(cov_fname)
        fig, _ = plot_cov(cov, info_fname, show=False)

        img = _fig_to_img(fig=fig)
        caption = 'Covariance : ' + cov_fname
        div_klass = 'covariance'
        img_klass = 'covariance'
        show = True
        html = image_template.substitute(img=img, id=global_id,
                                         div_klass=div_klass,
                                         img_klass=img_klass,
                                         caption=caption,
                                         show=show)
        self.html.append(html)

    def _render_trans(self, trans_fname, path, info, subject,
                      subjects_dir):

        kwargs = dict(info=info, trans_fname=trans_fname, subject=subject,
                      subjects_dir=subjects_dir)
        img = _iterate_trans_views(function=plot_trans, **kwargs)

        if img is not None:
            if 'trans' not in self.sections:
                self.sections.append('trans')

            global_id = self._get_id()

            caption = 'Trans : ' + trans_fname
            div_klass = 'trans'
            img_klass = 'trans'
            show = True
            html = image_template.substitute(img=img, id=global_id,
                                             div_klass=div_klass,
                                             img_klass=img_klass,
                                             caption=caption,
                                             width=75,
                                             show=show)
            self.html.append(html)

    def _render_bem(self, subject, subjects_dir):

        import nibabel as nib

        subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)

        # Get the MRI filename
        mri_fname = op.join(subjects_dir, subject, 'mri', 'T1.mgz')
        if not op.isfile(mri_fname):
            warnings.warn('MRI file "%s" does not exist' % mri_fname)

        # Get the BEM surface filenames
        bem_path = op.join(subjects_dir, subject, 'bem')

        if not op.isdir(bem_path):
            warnings.warn('Subject bem directory "%s" does not exist' %
                          bem_path)
            return self._render_image(mri_fname, cmap='gray')

        surf_fnames = []
        for surf_name in ['*inner_skull', '*outer_skull', '*outer_skin']:
            surf_fname = glob(op.join(bem_path, surf_name + '.surf'))
            if len(surf_fname) > 0:
                surf_fname = surf_fname[0]
            else:
                warnings.warn('No surface found for %s.' % surf_name)
                return self._render_image(mri_fname, cmap='gray')
            surf_fnames.append(surf_fname)

        # XXX : find a better way to get max range of slices
        nim = nib.load(mri_fname)
        data = nim.get_data()
        shape = data.shape
        del data  # free up memory

        html = []

        global_id = self._get_id()

        if 'mri' not in self.sections:
            self.sections.append('mri')

        name, caption = 'BEM', 'BEM contours'

        html += u'<li class="mri" id="%d">\n' % global_id
        html += u'<h2>%s</h2>\n' % name
        html += u'<div class="row">'
        html += self._render_one_bem_axe(mri_fname, surf_fnames, global_id,
                                         shape, orientation='axial')
        html += self._render_one_bem_axe(mri_fname, surf_fnames, global_id,
                                         shape, orientation='sagittal')
        html += u'</div><div class="row">'
        html += self._render_one_bem_axe(mri_fname, surf_fnames, global_id,
                                         shape, orientation='coronal')
        html += u'</div>'
        html += u'</li>\n'
        self.html.append(''.join(html))
        return html


def _recursive_search(path, pattern):
    """Auxiliary function for recursive_search of the directory.
    """
    filtered_files = list()
    for dirpath, dirnames, files in os.walk(path):
        for f in fnmatch.filter(files, pattern):
            filtered_files.append(op.realpath(op.join(dirpath, f)))

    return filtered_files


def _fix_global_ids(html):
    """Auxiliary function for fixing the global_ids after reordering in
       _render_toc().
    """
    html = re.sub('id="\d+"', 'id="###"', html)
    global_id = 1
    while len(re.findall('id="###"', html)) > 0:
        html = re.sub('id="###"', 'id="%s"' % global_id, html, count=1)
        global_id += 1
    return html
