import pygraphviz as pgv

font_face = 'Arial'
node_size = 12
node_small_size = 8
edge_size = 8

nodes = dict(
    T1='T1',
    flashes='Flash5/30',
    trans='Head-MRI trans',
    recon='Freesurfer surfaces',
    bem='BEM',
    src='Source space\nmne.SourceSpaces',
    cov='Noise covariance\nmne.Covariance',
    fwd='Forward solution\nmne.forward.Forward',
    inv='Inverse operator\nmne.minimum_norm.Inverse',
    stc='Source estimate\nmne.SourceEstimate',
    raw='Raw data\nmne.io.Raw',
    epo='Epoched data\nmne.Epochs',
    evo='Averaged data\nmne.Evoked',
    pre='Preprocessed data\nmne.io.Raw',
)

sensor_space = ('raw', 'pre', 'epo', 'evo', 'cov')
sensor_color = '#7bbeca'
source_space = ('src', 'stc', 'bem', 'flashes', 'recon', 'T1')
source_color = '#ff6347'

edges = (
    ('T1', 'recon'),
    ('flashes', 'bem'),
    ('recon', 'bem'),
    ('recon', 'src', 'setup_source_space'),
    ('src', 'fwd'),
    ('bem', 'fwd'),
    ('trans', 'fwd', 'make_forward_solution'),
    ('fwd', 'inv'),
    ('cov', 'inv', 'make_inverse_operator'),
    ('inv', 'stc'),
    ('evo', 'stc', 'apply_inverse'),
    ('raw', 'pre', 'raw.filter\netc.'),
    ('pre', 'epo', 'Epochs'),
    ('epo', 'evo', 'epochs.average'),
    ('epo', 'cov', 'compute_covariance'),
    ('epo', 'stc', 'apply_inverse_epochs'),
)

subgraphs = (
    [('T1', 'flashes', 'recon', 'bem', 'src'), 'Structural information'],
)

g = pgv.AGraph(directed=True)

for key, label in nodes.items():
    label = label.split('\n')
    label[0] = '<<FONT POINT-SIZE="%s">' % node_size + label[0] + '</FONT>'
    for li in range(1, len(label)):
        label[li] = ('<FONT POINT-SIZE="%s"><I>' % node_small_size
                     + label[li] + '</I></FONT>')
    label[-1] = label[-1] + '>'
    label = '<BR/>'.join(label)
    g.add_node(key, shape='plaintext', label=label)

# Create and customize nodes and edges
for edge in edges:
    g.add_edge(*edge[:2])
    e = g.get_edge(*edge[:2])
    if len(edge) > 2:
        e.attr['label'] = edge[2]
    e.attr['fontsize'] = edge_size
g.get_node
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

g.layout('dot')
g.draw('flow_diagram.svg', format='svg')
