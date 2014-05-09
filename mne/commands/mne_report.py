"""Generate html report from MNE database
"""

# Authors: Alex Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import sys
import os
import os.path as op
import fnmatch
import re
import StringIO
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import nibabel as nib

import mne
from tempita import HTMLTemplate


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


def build_tools(global_id, default_cmap="gray"):
    """ Build tools for slicer """
    html = []
    html.append(u'<div class="col-xs-6 col-md-4">')
    html.append(u'<h4>Color Map</h4>')
    html.append(u'<select>')
    import matplotlib.cm as cm
    html.append(u'<option value="%(c)s">%(c)s</option>' % {'c': default_cmap})
    for cmap in dir(cm):
        if cmap != default_cmap:
            html.append(u'<option value="%(c)s">%(c)s</option>' % {'c': cmap})
    html.append(u'</select>')
    html.append(u'<h4>Coordinates</h4>')
    html.append(u'<dl class="dl-horizontal">')
    html.append(u'<dt>X</dt><dd id="slicer_coord_x"></dd>')
    html.append(u'<dt>Y</dt><dd id="slicer_coord_y"></dd>')
    html.append(u'<dt>Z</dt><dd id="slicer_coord_z"></dd>')
    html.append(u'</dl>')
    html.append(u'</div>')
    return '\n'.join(html)


###############################################################################
# HTML scan renderer

header_template = HTMLTemplate(u"""
<!DOCTYPE html>
<html lang="fr">
<head>
<script type="text/javascript" src="http://code.jquery.com/jquery-1.10.2.min.js"></script>
<script type="text/javascript" src="http://code.jquery.com/ui/1.10.3/jquery-ui.js"></script>
<script src="http://netdna.bootstrapcdn.com/bootstrap/3.0.3/js/bootstrap.min.js"></script>
<link rel="stylesheet" type="text/css" media="all" href="http://code.jquery.com/ui/1.10.3/themes/smoothness/jquery-ui.css"/>
<link rel="stylesheet" href="http://netdna.bootstrapcdn.com/bootstrap/3.0.3/css/bootstrap.min.css">
<link rel="stylesheet" href="http://netdna.bootstrapcdn.com/bootstrap/3.0.3/css/bootstrap-theme.min.css">
<script type="text/javascript">
   function getCoordinates(e, img, ashape, name){
        var x = e.clientX - img.offsetLeft;
        var y = e.clientY - img.offsetTop;
        var maxx = img.width;
        var maxy = img.height;
        var divid = 'slicer_coord';
        if (name=='axial'){
        var c1 = Math.round(ashape[0]*(maxx - x)/maxx);
        var c2 = Math.round(ashape[1]*(maxy - y)/maxy);
        document.getElementById(
            divid + "_x").innerHTML = ashape[0] - c1;
        document.getElementById(divid + "_y").innerHTML = c2;
        };
        if (name=='sagittal'){
        var c1 = Math.round(ashape[1]*(maxx - x)/maxx);
        var c2 = Math.round(ashape[2]*(maxy - y)/maxy);
        document.getElementById(
            divid + "_y").innerHTML = ashape[1] - c1;
        document.getElementById(divid + "_z").innerHTML = c2;
        };
        if (name=='coronal'){
        var c1 = Math.round(ashape[0]*(maxx - x)/maxx);
        var c2 = Math.round(ashape[2]*(maxy - y)/maxy);
        document.getElementById(
            divid + "_x").innerHTML = ashape[0] - c1;
        document.getElementById(divid + "_z").innerHTML = c2;
        };
        }</script>
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

a:hover {
    text-decoration:underline;
    background-color: #eee;
}

#wrapper {
    text-align: left;
    margin: 5em auto;
    width: 700px;
}

#toc {
    list-style: none;
    margin-bottom: 20px;
}

#toc li {
    overflow: hidden;
    padding-bottom: 2px;
}

#toc span {
    display: inline-block;
    background: #fff;
    position: relative;
    bottom: -4px;
}

#toc a {
    float: right;
    padding: 0 0 3px 2px;
}

#toc span {
    float: left;
    padding: 0 2px 3px 0;
}

</style>
</head>
<body>
<br/>
&nbsp;<button type="button" class="btn btn-lg btn-default"
onclick="$('.raw').toggle()">Raw</button>
&nbsp;<button type="button" class="btn btn-lg btn-primary"
onclick="$('.evoked').toggle()">Ave</button>
&nbsp;<button type="button" class="btn btn-lg btn-success"
onclick="$('.covariance').toggle()">Cov</button>
&nbsp;<button type="button" class="btn btn-lg btn-info"
onclick="$('.events').toggle()">Eve</button>
&nbsp;<button type="button" class="btn btn-lg btn-warning"
onclick="$('.slices-images').toggle()">Slices</button>
""")

footer_template = HTMLTemplate(u"""
</body>
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
    def init_render(self):
        """ Initialize the renderer
        """
        return header_template.substitute()

    def finish_render(self):
        return footer_template.substitute()

    def render_toc(self, fnames):

        html = u'<div id="wrapper">'
        html += u'<h1> Table of Contents </h1>\n'
        html += u'<ul id="toc">'

        global_id = 1
        for fname in fnames:

            # identify bad file naming patterns and highlight them
            if not _endswith(fname, ['-eve.fif', '-ave.fif', '-cov.fif',
                                     '-sol.fif', '-fwd.fif', '-inv.fif',
                                     '-src.fif']):
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
            elif _endswith(fname, ['.nii', '.nii.gz', '.mgh', '.mgz']):
                class_name = 'slices-images'

            if _endswith(fname, ['.nii', '.nii.gz', '.mgh', '.mgz', 'raw.fif',
                                 'sss.fif', '-eve.fif', '-cov.fif']):
                html += (u'\n\t<li class="%s"><span title="%s" '
                         'style="color:%s"> %s </span> <a href="#%d">'
                         'report </a> </li>' % (class_name, fname,
                                                color, os.path.basename(fname),
                                                global_id))
                global_id += 1

            # loop through conditions for evoked
            elif _endswith(fname, ['-ave.fif']):
                evokeds = mne.fiff.read_evokeds(fname, baseline=(None, 0))

                html += (u'\n\t<li class="evoked"><span title="%s" '
                         'style="color:%s"> %s </span>'
                         % (fname, color, os.path.basename(fname)))

                html += u'<li class="evoked"><ul>'
                for ev in evokeds:
                    html += (u'\n\t<li class="evoked"><span title="%s"> %s'
                             '</span> <a href="#%d"> report </a> </li>'
                             % (fname, ev.comment, global_id))
                    global_id += 1
                html += u'</ul></li>'

        html += u'\n</ul></div>'

        return html

    def render_array(self, array, cmap='gray', limits=None):
        global_id = self.get_id()
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
        # Tools
        html.append(build_tools(global_id))
        # Close section
        html.append(u'</div>')
        return '\n'.join(html)

    def render_image(self, image, cmap='gray'):
        nim = nib.load(image)
        data = nim.get_data()
        shape = data.shape
        limits = {'sagittal': range(shape[0] // 3, shape[0] // 2),
                  'axial': range(shape[1] // 3, shape[1] // 2),
                  'coronal': range(shape[2] // 3, shape[2] // 2)}
        name = op.basename(image)
        html = u'<ul class="slices-images">\n'
        html += u'<h2>%s</h2>\n' % name
        html += self.render_array(data, cmap=cmap, limits=limits)
        html += u'</ul>\n'
        return html

    def render_raw(self, raw_fname):
        global_id = self.get_id()
        raw = mne.fiff.Raw(raw_fname)

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
        evokeds = mne.fiff.read_evokeds(evoked_fname, baseline=(None, 0))
        html = []
        for ev in evokeds:
            global_id = self.get_id()
            fig = ev.plot(show=False)
            output = StringIO.StringIO()
            # fig.savefig(output, dpi=self.dpi, format='png')
            fig.savefig(output, format='png')
            img = output.getvalue().encode('base64')
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

    def render_eve(self, eve_fname, figsize=None):
        global_id = self.get_id()
        events = mne.read_events(eve_fname)
        sfreq = 1000.  # XXX
        fig = mne.viz.plot_events(events, sfreq=sfreq, show=False)
        output = StringIO.StringIO()
        # fig.savefig(output, dpi=self.dpi, format='png')
        fig.savefig(output, format='png')
        img = output.getvalue().encode('base64')
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
        output = StringIO.StringIO()
        fig = plt.gcf()
        fig.savefig(output, format='png')
        img = output.getvalue().encode('base64')
        caption = 'Covariance : ' + cov_fname
        div_klass = 'covariance'
        img_klass = 'covariance'
        show = True
        return image_template.substitute(img=img, id=global_id,
                                         div_klass=div_klass,
                                         img_klass=img_klass,
                                         caption=caption,
                                         show=show)


def _endswith(fname, extensions):
    for ext in extensions:
        if fname.endswith(ext):
            return True
    return False


def recursive_search(path, pattern):
    filtered_files = list()
    for dirpath, dirnames, files in os.walk(path):
        for f in fnmatch.filter(files, pattern):
            filtered_files.append(op.join(dirpath, f))

    return filtered_files


def render_folder(path):
    renderer = HTMLScanRenderer()
    html = renderer.init_render()
    folders = []

    fnames = recursive_search(path, '*.fif')
    fnames += recursive_search(path, 'T1.mgz')

    print "Rendering Table of Contents"
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
                html += renderer.render_eve(fname)
            elif _endswith(fname, ['-cov.fif']):
                html += renderer.render_cov(fname)
            elif op.isdir(fname):
                folders.append(fname)
                print folders
        except Exception, e:
            print e
    html += renderer.finish_render()
    report_fname = op.join(path, 'report.html')
    fobj = open(report_fname, 'w')
    fobj.write(html)
    fobj.close()
    return report_fname


if __name__ == '__main__':
    path = sys.argv[-1]
    report_fname = render_folder(path)
