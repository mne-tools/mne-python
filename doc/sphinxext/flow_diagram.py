# -*- coding: utf-8 -*-

import os
from os import path as op

title = 'mne-python flow diagram'

font_face = 'Arial'
node_size = 12
node_small_size = 9
edge_size = 9
sensor_color = '#7bbeca'
source_color = '#ff6347'

legend = """
<<FONT POINT-SIZE="%s">
<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="4" CELLPADDING="4">
<TR><TD BGCOLOR="%s">    </TD><TD ALIGN="left">
Sensor (M/EEG) space</TD></TR>
<TR><TD BGCOLOR="%s">    </TD><TD ALIGN="left">
Source (brain) space</TD></TR>
</TABLE></FONT>>""" % (edge_size, sensor_color, source_color)
legend = ''.join(legend.split('\n'))

nodes = dict(
    T1='T1',
    flashes='Flash5/30',
    trans='Head-MRI trans',
    recon='Freesurfer surfaces',
    bem='BEM',
    src='Source space\nmne.SourceSpaces',
    cov='Noise covariance\nmne.Covariance',
    fwd='Forward solution\nmne.forward.Forward',
    inv='Inverse operator\nmne.minimum_norm.InverseOperator',
    stc='Source estimate\nmne.SourceEstimate',
    raw='Raw data\nmne.io.Raw',
    epo='Epoched data\nmne.Epochs',
    evo='Averaged data\nmne.Evoked',
    pre='Preprocessed data\nmne.io.Raw',
    legend=legend,
)

sensor_space = ('raw', 'pre', 'epo', 'evo', 'cov')
source_space = ('src', 'stc', 'bem', 'flashes', 'recon', 'T1')

edges = (
    ('T1', 'recon'),
    ('flashes', 'bem'),
    ('recon', 'bem'),
    ('recon', 'src', 'mne.setup_source_space'),
    ('src', 'fwd'),
    ('bem', 'fwd'),
    ('trans', 'fwd', 'mne.make_forward_solution'),
    ('fwd', 'inv'),
    ('cov', 'inv', 'mne.make_inverse_operator'),
    ('inv', 'stc'),
    ('evo', 'stc', 'mne.minimum_norm.apply_inverse'),
    ('raw', 'pre', 'raw.filter\n'
                   'mne.preprocessing.ICA\n'
                   'mne.preprocessing.compute_proj_eog\n'
                   'mne.preprocessing.compute_proj_ecg\n'
                   '...'),
    ('pre', 'epo', 'mne.Epochs'),
    ('epo', 'evo', 'epochs.average'),
    ('epo', 'cov', 'mne.compute_covariance'),
)

subgraphs = (
    [('T1', 'flashes', 'recon', 'bem', 'src'),
     ('<Structural information<BR/><FONT POINT-SIZE="%s"><I>'
      'Freesurfer / MNE-C</I></FONT>>' % node_small_size)],
)


def setup(app):
    app.connect('builder-inited', generate_flow_diagram)
    app.add_config_value('make_flow_diagram', True, 'html')


def setup_module():
    # HACK: Stop nosetests running setup() above
    pass


def generate_flow_diagram(app):
    out_dir = op.join(app.builder.outdir, '_static')
    if not op.isdir(out_dir):
        os.makedirs(out_dir)
    out_fname = op.join(out_dir, 'mne-python_flow.svg')
    make_flow_diagram = app is None or \
        bool(app.builder.config.make_flow_diagram)
    if not make_flow_diagram:
        print('Skipping flow diagram, webpage will have a missing image')
        return

    import pygraphviz as pgv
    g = pgv.AGraph(name=title, directed=True)

    for key, label in nodes.items():
        label = label.split('\n')
        if len(label) > 1:
            label[0] = ('<<FONT POINT-SIZE="%s">' % node_size
                        + label[0] + '</FONT>')
            for li in range(1, len(label)):
                label[li] = ('<FONT POINT-SIZE="%s"><I>' % node_small_size
                             + label[li] + '</I></FONT>')
            label[-1] = label[-1] + '>'
            label = '<BR/>'.join(label)
        else:
            label = label[0]
        g.add_node(key, shape='plaintext', label=label)

    # Create and customize nodes and edges
    for edge in edges:
        g.add_edge(*edge[:2])
        e = g.get_edge(*edge[:2])
        if len(edge) > 2:
            e.attr['label'] = ('<<I>' +
                               '<BR ALIGN="LEFT"/>'.join(edge[2].split('\n')) +
                               '<BR ALIGN="LEFT"/></I>>')
        e.attr['fontsize'] = edge_size

    # Change colors
    for these_nodes, color in zip((sensor_space, source_space),
                                  (sensor_color, source_color)):
        for node in these_nodes:
            g.get_node(node).attr['fillcolor'] = color
            g.get_node(node).attr['style'] = 'filled'

    # Create subgraphs
    for si, subgraph in enumerate(subgraphs):
        g.add_subgraph(subgraph[0], 'cluster%s' % si,
                       label=subgraph[1], color='black')

    # Format (sub)graphs
    for gr in g.subgraphs() + [g]:
        for x in [gr.node_attr, gr.edge_attr]:
            x['fontname'] = font_face
    g.node_attr['shape'] = 'box'

    # A couple of special ones
    for ni, node in enumerate(('fwd', 'inv', 'trans')):
        node = g.get_node(node)
        node.attr['gradientangle'] = 270
        colors = (source_color, sensor_color)
        colors = colors if ni == 0 else colors[::-1]
        node.attr['fillcolor'] = ':'.join(colors)
        node.attr['style'] = 'filled'
    del node
    g.get_node('legend').attr.update(shape='plaintext', margin=0, rank='sink')
    # put legend in same rank/level as inverse
    l = g.add_subgraph(['legend', 'inv'], name='legendy')
    l.graph_attr['rank'] = 'same'

    g.layout('dot')
    g.draw(out_fname, format='svg')
    return g


# This is useful for testing/iterating to see what the result looks like
if __name__ == '__main__':
    from mne.io.constants import Bunch
    out_dir = op.abspath(op.join(op.dirname(__file__), '..', '_build', 'html'))
    app = Bunch(builder=Bunch(outdir=out_dir,
                              config=Bunch(make_flow_diagram=True)))
    g = generate_flow_diagram(app)
