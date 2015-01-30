import pygraphviz as pgv

font_face = 'OpenSans'
node_size = 12
edge_size = 8

T1 = 'T1'
flashes = 'Flash5/30'
cov = 'Noise covariance'
trans = 'Head<->MRI trans'
recon = 'Freesurfer surfaces'
bem = 'BEM'
src = 'Source space'
fwd = 'Forward solution'
inv = 'Inverse operator'
stc = 'Source estimate'
raw = 'Raw'
epo = 'Epochs'
evo = 'Evoked'
pre = 'Preprocessed data'

sensor_space = (raw, pre, epo, evo, cov)
sensor_color = 'red'
source_space = (stc,)
source_color = 'green'

edges = (
    (T1, recon),
    (flashes, bem),
    (recon, bem),
    (recon, src, 'setup_source_space'),
    (src, fwd, 'make_forward_solution'),
    (bem, fwd),
    (trans, fwd),
    (fwd, inv, 'make_inverse_operator'),
    (cov, inv),
    (inv, stc),
    (evo, stc, 'apply_inverse'),
    (raw, pre, 'raw.filter'),
    (pre, epo, 'Epochs'),
    (epo, evo, 'epochs.average'),
    (epo, cov, 'compute_covariance'),
    (epo, stc, 'apply_inverse_epochs'),
)

g = pgv.AGraph(directed=True)
for x in (g.node_attr, g.edge_attr):
    x['fontname'] = font_face
    x['fontsize'] = node_size
g.node_attr['shape'] = 'box'

for edge in edges:
    g.add_edge(*edge[:2])
    e = g.get_edge(*edge[:2])
    if len(edge) > 2:
        e.attr['label'] = edge[2]
    e.attr['fontsize'] = edge_size

for these_nodes, color in zip((sensor_space, source_space),
                              (sensor_color, source_color)):
    for node in these_nodes:
        g.get_node(node).attr['color'] = color

g.layout('dot')
g.draw('test.svg', format='svg')
