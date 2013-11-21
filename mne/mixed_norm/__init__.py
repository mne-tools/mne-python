from ..utils import deprecated
from ..inverse_sparse import mxne_inverse

dec = deprecated('Use the function from mne.inverse_sparse')

mixed_norm = dec(mxne_inverse.mixed_norm)
tf_mixed_norm = dec(mxne_inverse.tf_mixed_norm)
