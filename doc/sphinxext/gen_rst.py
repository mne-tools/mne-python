"""
Example generation modified from the scikit learn

Generate the rst files for the examples by iterating over the python
example files.

Files that generate images should start with 'plot'

"""
from time import time
import os
import shutil
import traceback
import glob
import sys
from StringIO import StringIO
import cPickle
import re
import inspect

import matplotlib
matplotlib.use('Agg')

import token
import tokenize

MAX_NB_LINES_STDOUT = 20

###############################################################################
# A tee object to redict streams to multiple outputs


class Tee(object):

    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2

    def write(self, data):
        self.file1.write(data)
        self.file2.write(data)

    def flush(self):
        self.file1.flush()
        self.file2.flush()

###############################################################################
rst_template = """

.. _example_%(short_fname)s:

%(docstring)s

**Python source code:** :download:`%(fname)s <%(fname)s>`

.. literalinclude:: %(fname)s
    :lines: %(end_row)s-
    """

plot_rst_template = """

.. _example_%(short_fname)s:

%(docstring)s

%(image_list)s

%(stdout)s

**Python source code:** :download:`%(fname)s <%(fname)s>`

.. literalinclude:: %(fname)s
    :lines: %(end_row)s-

**Total running time of the example:** %(time_elapsed) 4i seconds
    """

# The following strings are used when we have several pictures: we use
# an html div tag that our CSS uses to turn the lists into horizontal
# lists.
HLIST_HEADER = """
.. rst-class:: horizontal

"""

HLIST_IMAGE_TEMPLATE = """
    *

      .. image:: images/%s
            :scale: 47
"""

SINGLE_IMAGE = """
.. image:: images/%s
    :align: center
"""


def extract_docstring(filename):
    """ Extract a module-level docstring, if any
    """
    lines = file(filename).readlines()
    start_row = 0
    if lines[0].startswith('#!'):
        lines.pop(0)
        start_row = 1

    docstring = ''
    first_par = ''
    tokens = tokenize.generate_tokens(iter(lines).next)
    for tok_type, tok_content, _, (erow, _), _ in tokens:
        tok_type = token.tok_name[tok_type]
        if tok_type in ('NEWLINE', 'COMMENT', 'NL', 'INDENT', 'DEDENT'):
            continue
        elif tok_type == 'STRING':
            docstring = eval(tok_content)
            # If the docstring is formatted with several paragraphs, extract
            # the first one:
            paragraphs = '\n'.join(line.rstrip()
                              for line in docstring.split('\n')).split('\n\n')
            if len(paragraphs) > 0:
                first_par = paragraphs[0]
        break
    return docstring, first_par, erow + 1 + start_row


def generate_example_rst(app):
    """ Generate the list of examples, as well as the contents of
        examples.
    """
    root_dir = os.path.join(app.builder.srcdir, 'auto_examples')
    example_dir = os.path.abspath(app.builder.srcdir + '/../../' + 'examples')
    try:
        plot_gallery = eval(app.builder.config.plot_gallery)
    except TypeError:
        plot_gallery = bool(app.builder.config.plot_gallery)
    if not os.path.exists(example_dir):
        os.makedirs(example_dir)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # we create an index.rst with all examples
    fhindex = file(os.path.join(root_dir, 'index.rst'), 'w')
    #Note: The sidebar button has been removed from the examples page for now
    #      due to how it messes up the layout. Will be fixed at a later point
    fhindex.write("""\

.. raw:: html


    <style type="text/css">

    div#sidebarbutton {
        display: none;
    }

    .figure {
        float: left;
        margin: 10px;
        width: auto;
        height: 200px;
        width: 180px;
    }

    .figure img {
        display: inline;
        }

    .figure .caption {
        width: 170px;
        text-align: center !important;
    }
    </style>

Examples
========

.. _examples-index:
""")
    # Here we don't use an os.walk, but we recurse only twice: flat is
    # better than nested.
    generate_dir_rst('.', fhindex, example_dir, root_dir, plot_gallery)
    for dir in sorted(os.listdir(example_dir)):
        if os.path.isdir(os.path.join(example_dir, dir)):
            generate_dir_rst(dir, fhindex, example_dir, root_dir, plot_gallery)
    fhindex.flush()


def generate_dir_rst(dir, fhindex, example_dir, root_dir, plot_gallery):
    """ Generate the rst file for an example directory.
    """
    if not dir == '.':
        target_dir = os.path.join(root_dir, dir)
        src_dir = os.path.join(example_dir, dir)
    else:
        target_dir = root_dir
        src_dir = example_dir
    if not os.path.exists(os.path.join(src_dir, 'README.txt')):
        print 80 * '_'
        print ('Example directory %s does not have a README.txt file'
                        % src_dir)
        print 'Skipping this directory'
        print 80 * '_'
        return
    fhindex.write("""


%s


""" % file(os.path.join(src_dir, 'README.txt')).read())
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    def sort_key(a):
        # put last elements without a plot
        if not a.startswith('plot') and a.endswith('.py'):
            return 'zz' + a
        return a
    for fname in sorted(os.listdir(src_dir), key=sort_key):
        if fname.endswith('py'):
            generate_file_rst(fname, target_dir, src_dir, plot_gallery)
            thumb = os.path.join(dir, 'images', 'thumb', fname[:-3] + '.png')
            link_name = os.path.join(dir, fname).replace(os.path.sep, '_')
            fhindex.write('.. figure:: %s\n' % thumb)
            if link_name.startswith('._'):
                link_name = link_name[2:]
            if dir != '.':
                fhindex.write('   :target: ./%s/%s.html\n\n' % (dir,
                                                               fname[:-3]))
            else:
                fhindex.write('   :target: ./%s.html\n\n' % link_name[:-3])
            fhindex.write("""   :ref:`example_%s`

.. toctree::
   :hidden:

   %s/%s

""" % (link_name, dir, fname[:-3]))
    fhindex.write("""
.. raw:: html

    <div style="clear: both"></div>
    """)  # clear at the end of the section


def generate_file_rst(fname, target_dir, src_dir, plot_gallery):
    """ Generate the rst file for a given example.
    """
    base_image_name = os.path.splitext(fname)[0]
    image_fname = '%s_%%s.png' % base_image_name

    this_template = rst_template
    last_dir = os.path.split(src_dir)[-1]
    # to avoid leading . in file names, and wrong names in links
    if last_dir == '.' or last_dir == 'examples':
        last_dir = ''
    else:
        last_dir += '_'
    short_fname = last_dir + fname
    src_file = os.path.join(src_dir, fname)
    example_file = os.path.join(target_dir, fname)
    shutil.copyfile(src_file, example_file)

    # The following is a list containing all the figure names
    figure_list = []

    image_dir = os.path.join(target_dir, 'images')
    thumb_dir = os.path.join(image_dir, 'thumb')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(thumb_dir):
        os.makedirs(thumb_dir)
    image_path = os.path.join(image_dir, image_fname)
    stdout_path = os.path.join(image_dir,
                               'stdout_%s.txt' % base_image_name)
    time_path = os.path.join(image_dir,
                               'time_%s.txt' % base_image_name)
    thumb_file = os.path.join(thumb_dir, fname[:-3] + '.png')
    time_elapsed = 0
    if plot_gallery:
        # generate the plot as png image if file name
        # starts with plot and if it is more recent than an
        # existing image.
        first_image_file = image_path % 1
        if os.path.exists(stdout_path):
            stdout = open(stdout_path).read()
        else:
            stdout = ''
        if os.path.exists(time_path):
            time_elapsed = float(open(time_path).read())

        if (not os.path.exists(first_image_file) or
                os.stat(first_image_file).st_mtime <=
                                    os.stat(src_file).st_mtime):
            # We need to execute the code
            print 'plotting %s' % fname
            t0 = time()
            import matplotlib.pyplot as plt
            plt.close('all')

            try:
                from mayavi import mlab
            except Exception, e:
                from enthought.mayavi import mlab
            mlab.close(all=True)

            cwd = os.getcwd()
            try:
                # First CD in the original example dir, so that any file
                # created by the example get created in this directory
                orig_stdout = sys.stdout
                os.chdir(os.path.dirname(src_file))
                my_buffer = StringIO()
                my_stdout = Tee(sys.stdout, my_buffer)
                sys.stdout = my_stdout
                my_globals = {'pl': plt}
                execfile(os.path.basename(src_file), my_globals)
                time_elapsed = time() - t0
                sys.stdout = orig_stdout
                my_stdout = my_buffer.getvalue()

                # get functions/objects of our package
                package_code_obj = {}
                for var_name, var in my_globals.iteritems():
                    if not hasattr(var, '__module__'):
                        continue
                    if not isinstance(var.__module__, basestring):
                        continue
                    if var.__module__.startswith('mne'):
                        # get the type as a string with other things stripped
                        tstr = str(type(var))
                        tstr = tstr[tstr.find('\'')\
                               + 1:tstr.rfind('\'')].split('.')[-1]
                        # see if we can import it using a shortened module path
                        module_path = '.'.join(var.__module__.split('.')[:-1])
                        try:
                            exec('from %s import %s' % (module_path, tstr))
                        except ImportError:
                            module_path = var.__module__
                        full_name = module_path + '.' + tstr
                        package_code_obj[var_name] = full_name

                # find package functions
                funregex = re.compile('[\w.]+\(')
                fid = open(src_file, 'rt')
                for line in fid.readlines():
                    if line.startswith('#'):
                        continue
                    for match in funregex.findall(line):
                        fun_name = match[:-1]
                        # find if it is an mne function
                        try:
                            exec('this_fun = %s' % fun_name, my_globals)
                        except Exception as err:
                            print 'extracting function failed'
                            print err
                            continue
                        this_fun = my_globals['this_fun']
                        if not inspect.isfunction(this_fun):
                            continue
                        if not hasattr(this_fun, '__module__'):
                            continue
                        if not isinstance(this_fun.__module__, basestring):
                            continue
                        if not this_fun.__module__.startswith('mne'):
                            continue
                        # ok, we have an mne function :-)
                        # see if we can import it using a shortened module path
                        fun_name_short = fun_name.split('.')[-1]
                        module_path =\
                            '.'.join(this_fun.__module__.split('.')[:-1])
                        try:
                            exec('from %s import %s' % (module_path,
                                                        fun_name_short))
                        except ImportError:
                            module_path = this_fun.__module__
                        full_name = module_path + '.' + fun_name_short
                        package_code_obj[fun_name] = full_name

                fid.close()
                if len(package_code_obj) > 0:
                    # save the dictionary, so we can later add hyperlinks
                    codeobj_fname = example_file[:-3] + '_codeobj.pickle'
                    fid = open(codeobj_fname, 'wb')
                    cPickle.dump(package_code_obj, fid,
                                 cPickle.HIGHEST_PROTOCOL)
                    fid.close()
                if '__doc__' in my_globals:
                    # The __doc__ is often printed in the example, we
                    # don't with to echo it
                    my_stdout = my_stdout.replace(
                                            my_globals['__doc__'],
                                            '')
                my_stdout = my_stdout.strip()
                if my_stdout:
                    output_lines = my_stdout.split('\n')
                    if len(output_lines) > MAX_NB_LINES_STDOUT:
                        output_lines = output_lines[:MAX_NB_LINES_STDOUT]
                        output_lines.append('...')
                    stdout = '**Script output**::\n\n  %s\n\n' % (
                      '\n  '.join(output_lines))
                open(stdout_path, 'w').write(stdout)
                open(time_path, 'w').write('%f' % time_elapsed)
                os.chdir(cwd)

                # In order to save every figure we have two solutions :
                # * iterate from 1 to infinity and call plt.fignum_exists(n)
                #   (this requires the figures to be numbered
                #    incrementally: 1, 2, 3 and not 1, 2, 5)
                # * iterate over [fig_mngr.num for fig_mngr in
                #   matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
                last_fig_num = 0
                for fig_num in (fig_mngr.num for fig_mngr in
                        matplotlib._pylab_helpers.Gcf.get_all_fig_managers()):
                    # Set the fig_num figure as the current figure as we can't
                    # save a figure that's not the current figure.
                    plt.figure(fig_num)
                    # hack to keep black bg
                    facecolor = plt.gcf().get_facecolor()
                    if facecolor == (0.0, 0.0, 0.0, 1.0):
                        plt.savefig(image_path % fig_num, facecolor='black')
                    else:
                        plt.savefig(image_path % fig_num)
                    figure_list.append(image_fname % fig_num)
                    last_fig_num = fig_num

                e = mlab.get_engine()
                for scene in e.scenes:
                    last_fig_num += 1
                    mlab.savefig(image_path % last_fig_num)
                    figure_list.append(image_fname % last_fig_num)
                    mlab.close(scene)

            except:
                print 80 * '_'
                print '%s is not compiling:' % fname
                traceback.print_exc()
                print 80 * '_'
            finally:
                os.chdir(cwd)
                sys.stdout = orig_stdout

            print " - time elapsed : %.2g sec" % time_elapsed
        else:
            figure_list = [f[len(image_dir):]
                            for f in glob.glob(image_path % '[1-9]')]
                            #for f in glob.glob(image_path % '*')]

        # generate thumb file
        this_template = plot_rst_template
        from matplotlib import image
        if os.path.exists(first_image_file):
            image.thumbnail(first_image_file, thumb_file, 0.2)

    if not os.path.exists(thumb_file):
        # create something not to replace the thumbnail
        shutil.copy('source/_images/mne_helmet.png', thumb_file)

    docstring, short_desc, end_row = extract_docstring(example_file)

    # Depending on whether we have one or more figures, we're using a
    # horizontal list or a single rst call to 'image'.
    if len(figure_list) == 1:
        figure_name = figure_list[0]
        image_list = SINGLE_IMAGE % figure_name.lstrip('/')
    else:
        image_list = HLIST_HEADER
        for figure_name in figure_list:
            image_list += HLIST_IMAGE_TEMPLATE % figure_name.lstrip('/')

    f = open(os.path.join(target_dir, fname[:-2] + 'rst'), 'w')
    f.write(this_template % locals())
    f.flush()


def embed_code_links(app, exception):
    """Embed links to documentation into example code"""
    if exception is not None:
        return
    print 'Embedding mne-python links in examples..'
    example_dir = os.path.join(app.builder.srcdir, 'auto_examples')
    html_example_dir = os.path.abspath(app.builder.outdir + '/auto_examples')
    package_generated = glob.glob(app.builder.outdir + '/generated/*.html')
    package_generated = [os.path.split(p)[-1][:-5] for p in package_generated]
    link_pattern = '<a href="%s">%s</a>'
    orig_pattern = '<span class="n">%s</span>'
    period = '<span class="o">.</span>'
    for dirpath, _, filenames in os.walk(html_example_dir):
        for fname in filenames:
            print 'processing: %s' % fname
            subpath = dirpath[len(html_example_dir):]
            pickle_fname = example_dir + '/' + subpath + '/'\
                           + fname[:-5] + '_codeobj.pickle'
            if os.path.exists(pickle_fname):
                # we have a pickle file with the local :-)
                fid = open(pickle_fname, 'rb')
                package_code_obj = cPickle.load(fid)
                fid.close()
                str_repl = {}
                # detect "how deep" we are for relative links
                rel_link = '/'.join(['..'] * len(subpath.split('/')))
                for name, pname in package_code_obj.iteritems():
                    if pname in package_generated:
                        link = (rel_link + '/generated/%s.html' % pname)
                        dest_fname = os.path.join(dirpath, link)
                        if not os.path.exists(dest_fname):
                            continue
                        parts = name.split('.')
                        name_html = orig_pattern % parts[0]
                        for part in parts[1:]:
                            name_html += period + orig_pattern % part
                        str_repl[name_html] = link_pattern % (link, name_html)
                if len(str_repl) > 0:
                    fid = open(os.path.join(dirpath, fname), 'rt')
                    lines_in = fid.readlines()
                    fid.close()
                    fid = open(os.path.join(dirpath, fname), 'wt')
                    for line in lines_in:
                        for name, link in str_repl.iteritems():
                            line = line.replace(name, link)
                        fid.write(line)
                    fid.close()
    print '[done]'


def setup(app):
    app.connect('builder-inited', generate_example_rst)
    app.add_config_value('plot_gallery', True, 'html')

    # embed links after build is finished
    app.connect('build-finished', embed_code_links)

    # Sphinx hack: sphinx copies generated images to the build directory
    #  each time the docs are made.  If the desired image name already
    #  exists, it appends a digit to prevent overwrites.  The problem is,
    #  the directory is never cleared.  This means that each time you build
    #  the docs, the number of images in the directory grows.
    #
    # This question has been asked on the sphinx development list, but there
    #  was no response: http://osdir.com/ml/sphinx-dev/2011-02/msg00123.html
    #
    # The following is a hack that prevents this behavior by clearing the
    #  image build directory each time the docs are built.  If sphinx
    #  changes their layout between versions, this will not work (though
    #  it should probably not cause a crash).  Tested successfully
    #  on Sphinx 1.0.7
    build_image_dir = 'build/html/_images'
    if os.path.exists(build_image_dir):
        filelist = os.listdir(build_image_dir)
        for filename in filelist:
            if filename.endswith('png'):
                os.remove(os.path.join(build_image_dir, filename))
