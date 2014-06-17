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
import StringIO
import numpy as np
import webbrowser
import time
from PIL import Image
import glob

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import nibabel as nib

import mne
from tempita import HTMLTemplate, Template

tempdir = mne.utils._TempDir()

###############################################################################
# IMAGE FUNCTIONS

def _build_image(data, dpi=1, cmap='gray'):
    """ Build an image encoded in base64 """
    figsize = data.shape[::-1]
    if figsize[0] == 1:
        figsize = tuple(figsize[1:])
        data = data[:, :, 0]
    fig = Figure(figsize=figsize, dpi=dpi, frameon=False)
    canvas = FigureCanvas(fig)
    cmap = getattr(plt.cm, cmap, plt.cm.gray)
    fig.figimage(data, cmap=cmap)
    output = StringIO.StringIO()
    fig.savefig(output, dpi=dpi, format='png')
    return output.getvalue().encode('base64')


def iterate_sagittal_slices(array, limits=None):
    """ Iterate sagittal slice """
    shape = array.shape[0]
    for ind in xrange(shape):
        if limits and ind not in limits:
            continue
        yield ind, np.rot90(array[ind, :, :])


def iterate_coronal_slices(array, limits=None):
    """ Iterate coronal slice """
    shape = array.shape[1]
    for ind in xrange(shape):
        if limits and ind not in limits:
            continue
        yield ind, np.rot90(array[:, ind, :])


def iterate_axial_slices(array, limits=None):
    """ Iterate axial slice """
    shape = array.shape[2]
    for ind in xrange(shape):
        if limits and ind not in limits:
            continue
        yield ind, np.rot90(array[:, :, ind])


###############################################################################
# HTML functions

def build_html_image(img, id, div_klass, img_klass, caption=None, show=True):
    """ Build a html image from a slice array """
    html = []
    add_style = '' if show else 'style="display: none"'
    html.append(u'<li class="%s" id="%s" %s>' % (div_klass, id, add_style))
    html.append(u'<div class="thumbnail">')
    html.append(u'<img class="%s" alt="" style="width:90%%;" src="data:image/png;base64,%s">'
                % (img_klass, img))
    html.append(u'</div>')
    if caption:
        html.append(u'<h4>%s</h4>' % caption)
    html.append(u'</li>')
    return '\n'.join(html)

slider_template = HTMLTemplate(u"""
<div id="{{slider_id}}"></div>
<script>$("#{{slider_id}}").slider({
                       range: "min",
                       /*orientation: "vertical",*/
                       min: {{minvalue}},
                       max: {{maxvalue}},
                       stop: function(event, ui) {
                       var list_value = $("#{{slider_id}}").slider("value");
                       $(".{{klass}}").hide();
                       $("#{{klass}}-"+list_value).show();}
                       })</script>
""")


def build_html_slider(slices_range, slides_klass, slider_id):
    """ Build an html slider for a given slices range and a slices klass """
    return slider_template.substitute(slider_id=slider_id,
                                      klass=slides_klass,
                                      minvalue=slices_range[0],
                                      maxvalue=slices_range[-1])

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

a, a:hover{
    color: #333333;
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

#toc {
  background: #a8a599;
  position: fixed;
  width: 20%;
  height: 100%;
  margin-top: 45px;
  overflow: auto;
}

#toc li {
    overflow: hidden;
    padding-bottom: 2px;
    margin-left: 20px;
}

#toc a {
    padding: 0 0 3px 2px;
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

<h3 class="navbar-text" style="color:white">MNE-Report for {{path}}</h3>

<ul class="nav nav-pills navbar-right" style="margin-top: 7px;">

    <li class="active raw-btn">
        <a href="#" onclick="togglebutton('.raw')">Raw</a>
    </li>
    <li class="active evoked-btn">
        <a href="#"  onclick="togglebutton('.evoked')">Ave</a>
    </li>
    <li class="active covariance-btn">
        <a href="#" onclick="togglebutton('.covariance')">Cov</a>
    </li>
    <li class="active events-btn">
        <a href="#" onclick="togglebutton('.events')">Eve</a>
    </li>
    <li class="active trans-btn">
        <a href="#" onclick="togglebutton('.trans')">Trans</a>
    </li>
    <li class="active slices-images-btn">
        <a href="#" onclick="togglebutton('.slices-images')">Slices</a>
    </li>
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

image_template = HTMLTemplate(u"""
<li class="{{div_klass}}" id="{{id}}" {{if not show}}style="display: none"{{endif}}>
{{if caption}}
<h4>{{caption}}</h4>
{{endif}}
<div class="thumbnail">
<img alt="" style="width:50%;" src="data:image/png;base64,{{img}}">
</div>
</li>
""")


class HTMLScanRenderer(object):
    """Object for rendering HTML"""

    def __init__(self, dpi=1):
        """ Initialize the HTMLScanRenderer.
        cmap is the name of the pyplot color map.
        dpi is the name of the pyplot color map.
        """
        self.dpi = dpi
        self.initial_id = 0

    def get_id(self):
        self.initial_id += 1
        return self.initial_id

    ###########################################################################
    # HTML rendering
    def render_one_axe(self, slices_iter, name, global_id=None, cmap='gray'):
        """ Render one axe of the array """
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
            img = _build_image(data, dpi=self.dpi, cmap=cmap)
            slices.append(build_html_image(img, slice_id, div_klass, img_klass,
                                           caption, first))
            first = False
        # Render the slider
        slider_id = 'select-%s-%s' % (name, global_id)
        html.append(build_html_slider(slices_range, slides_klass, slider_id))
        html.append(u'<ul class="thumbnails">')
        # Render the slices
        html.append(u'\n'.join(slices))
        html.append(u'</ul>')
        html.append(u'</div>')
        return '\n'.join(html)

    ###########################################################################
    # global rendering functions
    def init_render(self, path):
        """ Initialize the renderer
        """
        inc_fnames = ['bootstrap.min.js', 'jquery-1.10.2.min.js',
                      'jquery-ui.js', 'bootstrap.min.css', 'jquery-ui.css']

        include = list()
        for inc_fname in inc_fnames:
            print('Embedding : %s' % inc_fname)
            f = open(op.join(op.dirname(__file__), '..', 'html', inc_fname),
                     'r')
            if inc_fname.endswith('.js'):
                include.append(u'<script type="text/javascript">'
                               + f.read() + u'</script>')
            elif inc_fname.endswith('.css'):
                include.append(u'<style type="text/css">'
                               + f.read() + u'</style>')
            f.close()

        return header_template.substitute(path=path, include=''.join(include))

    def finish_render(self):
        return footer_template.substitute(date=time.strftime("%B %d, %Y"))

    def render_toc(self, fnames):
        html = u'<div id="container">'
        html += u'<div id="toc"><h1>Table of Contents</h1>'

        global_id = 1
        for fname in fnames:

            # identify bad file naming patterns and highlight them
            if not _endswith(fname, ['-eve.fif', '-ave.fif', '-cov.fif',
                                     '-sol.fif', '-fwd.fif', '-inv.fif',
                                     '-src.fif', '-trans.fif', 'raw.fif',
                                     'T1.mgz']):
                color = 'red'
            else:
                color = ''

            # assign class names to allow toggling with buttons
            if _endswith(fname, ['-eve.fif']):
                class_name = 'events'
            elif _endswith(fname, ['-ave.fif']):
                class_name = 'evoked'
            elif _endswith(fname, ['-cov.fif']):
                class_name = 'covariance'
            elif _endswith(fname, ['raw.fif', 'sss.fif']):
                class_name = 'raw'
            elif _endswith(fname, ['-trans.fif']):
                class_name = 'trans'
            elif _endswith(fname, ['.nii', '.nii.gz', '.mgh', '.mgz']):
                class_name = 'slices-images'

            if _endswith(fname, ['.nii', '.nii.gz', '.mgh', '.mgz', 'raw.fif',
                                 'sss.fif', '-eve.fif', '-cov.fif',
                                 '-trans.fif']):
                html += (u'\n\t<li class="%s"><a href="#%d"><span title="%s" '
                         'style="color:%s"> %s </span>'
                         '</a></li>' % (class_name, global_id, fname,
                                        color, os.path.basename(fname)))
                global_id += 1

            # loop through conditions for evoked
            elif _endswith(fname, ['-ave.fif']):
                evokeds = mne.read_evokeds(fname, baseline=(None, 0),
                                           verbose=False)

                html += (u'\n\t<li class="evoked"><span title="%s" '
                         'style="color:%s"> %s </span>'
                         % (fname, color, os.path.basename(fname)))

                html += u'<li class="evoked"><ul>'
                for ev in evokeds:
                    html += (u'\n\t<li class="evoked"><a href="#%d"><span title="%s" style="color:%s"> %s'
                             '</span></a></li>'
                             % (global_id, fname, color, ev.comment))
                    global_id += 1
                html += u'</ul></li>'

        html += (u'\n\t<li><a href="#%d"><span> %s</span></a></li>' %
                 (global_id, 'BEM contours'))

        html += u'\n</ul></div>'

        html += u'<div style="margin-left: 22%;">'

        return html

    def render_array(self, array, global_id=None, cmap='gray', limits=None):
        html = []
        html.append(u'<div class="row">')
        # Axial
        limits = limits or {}
        axial_limit = limits.get('axial')
        axial_slices_gen = iterate_axial_slices(array, axial_limit)
        html.append(
            self.render_one_axe(axial_slices_gen, 'axial', global_id, cmap))
        # Sagittal
        sagittal_limit = limits.get('sagittal')
        sagittal_slices_gen = iterate_sagittal_slices(array, sagittal_limit)
        html.append(
            self.render_one_axe(sagittal_slices_gen, 'sagittal', global_id, cmap))
        html.append(u'</div>')
        html.append(u'<div class="row">')
        # Coronal
        coronal_limit = limits.get('coronal')
        coronal_slices_gen = iterate_coronal_slices(array, coronal_limit)
        html.append(
            self.render_one_axe(coronal_slices_gen, 'coronal', global_id, cmap))
        # Close section
        html.append(u'</div>')
        return '\n'.join(html)

    def render_image(self, image, cmap='gray'):
        global_id = self.get_id()
        nim = nib.load(image)
        data = nim.get_data()
        shape = data.shape
        limits = {'sagittal': range(shape[0] // 3, shape[0] // 2),
                  'axial': range(shape[1] // 3, shape[1] // 2),
                  'coronal': range(shape[2] // 3, shape[2] // 2)}
        name = op.basename(image)
        html = u'<li class="slices-images" id="%d">\n' % global_id
        html += u'<h2>%s</h2>\n' % name
        html += self.render_array(data, global_id=global_id,
                                  cmap=cmap, limits=limits)
        html += u'</li>\n'
        return html

    def render_raw(self, raw_fname):
        global_id = self.get_id()
        raw = mne.io.Raw(raw_fname)

        html = u'<li class="raw" id="%d">' % global_id
        html += u'<h4>Raw : %s</h4>\n<hr>' % raw_fname

        repr_raw = re.sub('<', '', repr(raw))
        repr_raw = re.sub('>', '', repr_raw)

        html += '%s<br/>' % repr_raw

        repr_info = re.sub('<', '', repr(raw.info))
        repr_info = re.sub('>', '', repr_info)

        html += re.sub('\\n', '\\n<br/>', repr_info)

        html += u'<hr></li>'

        return html

    def render_evoked(self, evoked_fname, figsize=None):
        evokeds = mne.read_evokeds(evoked_fname, baseline=(None, 0),
                                   verbose=False)

        html = []
        for ev in evokeds:
            global_id = self.get_id()
            img = _fig_to_img(ev.plot(show=False))
            caption = 'Evoked : ' + evoked_fname + ' (' + ev.comment + ')'
            div_klass = 'evoked'
            img_klass = 'evoked'
            show = True
            html.append(image_template.substitute(img=img, id=global_id,
                                                  div_klass=div_klass,
                                                  img_klass=img_klass,
                                                  caption=caption,
                                                  show=show))

        return '\n'.join(html)

    def render_eve(self, eve_fname, info):
        global_id = self.get_id()
        events = mne.read_events(eve_fname)
        sfreq = info['sfreq']
        ax = mne.viz.plot_events(events, sfreq=sfreq, show=False)  # XXX : weird colors
        img = _fig_to_img(ax.gcf())
        caption = 'Events : ' + eve_fname
        div_klass = 'events'
        img_klass = 'events'
        show = True
        return image_template.substitute(img=img, id=global_id,
                                         div_klass=div_klass,
                                         img_klass=img_klass,
                                         caption=caption,
                                         show=show)

    def render_cov(self, cov_fname):
        global_id = self.get_id()
        cov = mne.Covariance(cov_fname)
        plt.matshow(cov.data)

        img = _fig_to_img(plt.gcf())
        caption = 'Covariance : ' + cov_fname
        div_klass = 'covariance'
        img_klass = 'covariance'
        show = True
        return image_template.substitute(img=img, id=global_id,
                                         div_klass=div_klass,
                                         img_klass=img_klass,
                                         caption=caption,
                                         show=show)

    def render_trans(self, trans_fname, path, info, subject,
                     subjects_dir):

        global_id = self.get_id()

        fig = mne.viz.plot_trans(info, trans_fname=trans_fname,
                                 subject=subject, subjects_dir=subjects_dir)

        fig.scene.save_bmp(tempdir + 'test')  # XXX: save_bmp / save_png / ...
        output = StringIO.StringIO()
        Image.open(tempdir + 'test').save(output, format='bmp')
        img = output.getvalue().encode('base64')

        caption = 'Trans : ' + trans_fname
        div_klass = 'trans'
        img_klass = 'trans'
        show = True
        return image_template.substitute(img=img, id=global_id,
                                         div_klass=div_klass,
                                         img_klass=img_klass,
                                         caption=caption,
                                         show=show)

    def render_bem(self, subject, subjects_dir, orientation='coronal'):

        global_id = self.get_id()

        fig = mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                               orientation=orientation, show=False)
        img = _fig_to_img(fig)

        caption = 'BEM Contours : ' + orientation
        div_klass = 'bem'
        img_klass = 'bem'
        show = True
        return image_template.substitute(img=img, id=global_id,
                                         div_klass=div_klass,
                                         img_klass=img_klass,
                                         caption=caption,
                                         show=show)


def _fig_to_img(fig):
    """Auxiliary function for fig <-> binary image.
    """
    output = StringIO.StringIO()
    fig.savefig(output, format='png')

    return output.getvalue().encode('base64')


def _endswith(fname, extensions):
    for ext in extensions:
        if fname.endswith(ext):
            return True
    return False


def recursive_search(path, pattern):
    """Auxiliary function for recursive_search of the directory.
    """
    filtered_files = list()
    for dirpath, dirnames, files in os.walk(path):
        for f in fnmatch.filter(files, pattern):
            filtered_files.append(op.join(dirpath, f))

    return filtered_files


def render_folder(path, info_fname, subjects_dir, subject):
    renderer = HTMLScanRenderer()
    html = renderer.init_render(path)
    folders = []

    fnames = recursive_search(path, '*.fif')
    fnames += glob.glob(op.join(subjects_dir, subject, 'mri', 'T1.mgz'))

    info = mne.io.read_info(info_fname)

    print('Rendering : Table of Contents')
    html += renderer.render_toc(fnames)

    for fname in fnames:
        print "Rendering : %s" % op.join('...' + path[-20:], fname)
        try:
            if _endswith(fname, ['.nii', '.nii.gz', '.mgh', '.mgz']):
                cmap = 'gray'
                if 'aseg' in fname:
                    cmap = 'spectral'
                html += renderer.render_image(fname, cmap=cmap)
            elif _endswith(fname, ['raw.fif', 'sss.fif']):
                html += renderer.render_raw(fname)
            elif _endswith(fname, ['-ave.fif']):
                html += renderer.render_evoked(fname)
            elif _endswith(fname, ['-eve.fif']):
                html += renderer.render_eve(fname, info)
            elif _endswith(fname, ['-cov.fif']):
                html += renderer.render_cov(fname)
            elif _endswith(fname, ['-trans.fif']):
                html += renderer.render_trans(fname, path, info,
                                              subject, subjects_dir)
            elif op.isdir(fname):
                folders.append(fname)
                print folders
        except Exception, e:
            print e

    html += renderer.render_bem(subject=subject, subjects_dir=subjects_dir,
                                orientation='coronal')

    html += renderer.finish_render()
    report_fname = op.join(path, 'report.html')
    fobj = open(report_fname, 'w')
    fobj.write(html)
    fobj.close()

    webbrowser.open_new_tab(report_fname)

    return report_fname


if __name__ == '__main__':

    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    parser.add_option("-p", "--path", dest="path",
                      help="Path to folder who MNE-Report must be created")
    parser.add_option("-i", "--info-fname", dest="info_fname",
                      help="File from which info dictionary is to be read",
                      metavar="FILE")
    parser.add_option("-d", "--subjects-dir", dest="subjects_dir",
                      help="The subjects directory")
    parser.add_option("-s", "--subject", dest="subject",
                      help="The subject name")

    options, args = parser.parse_args()
    path = options.path
    info_fname = options.info_fname
    subjects_dir = options.subjects_dir
    subject = options.subject

    report_fname = render_folder(path, info_fname, subjects_dir, subject)
