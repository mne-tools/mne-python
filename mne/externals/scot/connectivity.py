# coding=utf-8

# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" Connectivity Analysis """

import numpy as np
import scipy as sp
from scipy.fftpack import fft
from .utils import memoize


def connectivity(measure_names, b, c=None, nfft=512):
    """ calculate connectivity measures.

    Parameters
    ----------
    measure_names : {str, list of str}
        Name(s) of the connectivity measure(s) to calculate. See :class:`Connectivity` for supported measures.
    b : ndarray, shape = [n_channels, n_channels*model_order]
        VAR model coefficients. See :ref:`var-model-coefficients` for details about the arrangement of coefficients.
    c : ndarray, shape = [n_channels, n_channels], optional
        Covariance matrix of the driving noise process. Identity matrix is used if set to None.
    nfft : int, optional
        Number of frequency bins to calculate. Note that these points cover the range between 0 and half the
        sampling rate.

    Returns
    -------
    result : ndarray, shape = [n_channels, n_channels, `nfft`]
        An ndarray of shape [m, m, nfft] is returned if measures is a string. If measures is a list of strings a
        dictionary is returned, where each key is the name of the measure, and the corresponding values are ndarrays
        of shape [m, m, nfft].

    Notes
    -----
    When using this function it is more efficient to get several measures at once than calling the function multiple times.

    Examples
    --------
    >>> c = connectivity(['DTF', 'PDC'], [[0.3, 0.6], [0.0, 0.9]])
    """
    con = Connectivity(b, c, nfft)
    try:
        return getattr(con, measure_names)()
    except TypeError:
        return {m: getattr(con, m)() for m in measure_names}


#noinspection PyPep8Naming
class Connectivity:
    """ Calculation of connectivity measures
    
    This class calculates various spectral connectivity measures from a vector autoregressive (VAR) model.

    Parameters
    ----------
    b : ndarray, shape = [n_channels, n_channels*model_order]
        VAR model coefficients. See :ref:`var-model-coefficients` for details about the arrangement of coefficients.
    c : ndarray, shape = [n_channels, n_channels], optional
        Covariance matrix of the driving noise process. Identity matrix is used if set to None.
    nfft : int, optional
        Number of frequency bins to calculate. Note that these points cover the range between 0 and half the
        sampling rate.

    Methods
    -------
    :func:`A`
       Spectral representation of the VAR coefficients
    :func:`H`
        Transfer function that turns the innovation process into the VAR process
    :func:`S`
        Cross spectral density
    :func:`logS`
        Logarithm of the cross spectral density (S), for convenience.
    :func:`G`
        Inverse cross spectral density
    :func:`logG`
        Logarithm of the inverse cross spectral density
    :func:`PHI`
        Phase angle
    :func:`COH`
        Coherence
    :func:`pCOH`
        Partial coherence
    :func:`PDC`
        Partial directed coherence
    :func:`ffPDC`
        Full frequency partial directed coherence
    :func:`PDCF`
        PDC factor
    :func:`GPDC`
        Generalized partial directed coherence
    :func:`DTF`
        Directed transfer function
    :func:`ffDTF`
        Full frequency directed transfer function
    :func:`dDTF`
        Direct directed transfer function
    :func:`GDTF`
        Generalized directed transfer function

    Notes
    -----
    Connectivity measures are returned by member functions that take no arguments and return a matrix of
    shape [m,m,nfft]. The first dimension is the sink, the second dimension is the source, and the third dimension is
    the frequency.

    A summary of most supported measures can be found in [1]_.

    References
    ----------
    .. [1] M. Billinger et al, “Single-trial connectivity estimation for classification of motor imagery data”,
           *J. Neural Eng.* 10, 2013.
    """

    def __init__(self, b, c=None, nfft=512):
        b = np.asarray(b)
        (m, mp) = b.shape
        p = mp // m
        if m * p != mp:
            raise AttributeError('Second dimension of b must be an integer multiple of the first dimension.')

        if c is None:
            self.c = None
        else:
            self.c = np.atleast_2d(c)

        self.b = np.reshape(b, (m, m, p), 'c')
        self.m = m
        self.p = p
        self.nfft = nfft

    @memoize
    def Cinv(self):
        """ Inverse of the noise covariance
        """
        try:
            return np.linalg.inv(self.c)
        except np.linalg.linalg.LinAlgError:
            print('Warning: non invertible noise covariance matrix c!')
            return np.eye(self.c.shape[0])

    @memoize
    def A(self):
        """ Spectral VAR coefficients

        .. math:: \mathbf{A}(f) = \mathbf{I} - \sum_{k=1}^{p} \mathbf{a}^{(k)} \mathrm{e}^{-2\pi f}
        """
        return fft(np.dstack([np.eye(self.m), -self.b]), self.nfft * 2 - 1)[:, :, :self.nfft]

    @memoize
    def H(self):
        """ VAR transfer function

        .. math:: \mathbf{H}(f) = \mathbf{A}(f)^{-1}
        """
        return _inv3(self.A())

    @memoize
    def S(self):
        """ Cross spectral density

        .. math:: \mathbf{S}(f) = \mathbf{H}(f) \mathbf{C} \mathbf{H}'(f)
        """
        if self.c is None:
            raise RuntimeError('Cross spectral density requires noise covariance matrix c.')
        H = self.H()
        #TODO can we do that more efficiently?
        S = np.empty(H.shape, dtype=H.dtype)
        for f in range(H.shape[2]):
            S[:, :, f] = H[:, :, f].dot(self.c).dot(H[:, :, f].conj().T)
        return S

    @memoize
    def logS(self):
        """ Logarithmic cross spectral density

        .. math:: \mathrm{logS}(f) = \log | \mathbf{S}(f) |
        """
        return np.log10(np.abs(self.S()))

    @memoize
    def absS(self):
        """ Absolute cross spectral density

        .. math:: \mathrm{absS}(f) = | \mathbf{S}(f) |
        """
        return np.abs(self.S())

    @memoize
    def G(self):
        """ Inverse cross spectral density

        .. math:: \mathbf{G}(f) = \mathbf{A}(f) \mathbf{C}^{-1} \mathbf{A}'(f)
        """
        if self.c is None:
            raise RuntimeError('Inverse cross spectral density requires invertible noise covariance matrix c.')
        A = self.A()
        #TODO can we do that more efficiently?
        G = np.einsum('ji..., jk... ->ik...', A.conj(), self.Cinv())
        G = np.einsum('ij..., jk... ->ik...', G, A)
        return G

    @memoize
    def logG(self):
        """ Logarithmic inverse cross spectral density

        .. math:: \mathrm{logG}(f) = \log | \mathbf{G}(f) |
        """
        return np.log10(np.abs(self.G()))

    @memoize
    def COH(self):
        """ Coherence

        .. math:: \mathrm{COH}_{ij}(f) = \\frac{S_{ij}(f)}{\sqrt{S_{ii}(f) S_{jj}(f)}}
        
        References
        ----------
        P. L. Nunez, R. Srinivasan, A. F. Westdorp, R. S. Wijesinghe, D. M. Tucker,
        R. B. Silverstein, P. J. Cadusch. EEG coherency: I: statistics, reference electrode,
        volume conduction, Laplacians, cortical imaging, and interpretation at multiple scales.
        Electroenceph. Clin. Neurophysiol. 103(5): 499-515, 1997.
        """
        S = self.S()
        #TODO can we do that more efficiently?
        return S / np.sqrt(np.einsum('ii..., jj... ->ij...', S, S.conj()))

    @memoize
    def PHI(self):
        """ Phase angle

        Returns the phase angle of complex :func:`S`.
        """
        return np.angle(self.S())

    @memoize
    def pCOH(self):
        """ Partial coherence

        .. math:: \mathrm{pCOH}_{ij}(f) = \\frac{G_{ij}(f)}{\sqrt{G_{ii}(f) G_{jj}(f)}}
        
        References
        ----------
        P. J. Franaszczuk, K. J. Blinowska, M. Kowalczyk. The application of parametric multichannel
        spectral estimates in the study of electrical brain activity. Biol. Cybernetics 51(4): 239-247, 1985.
        """
        G = self.G()
        #TODO can we do that more efficiently?
        return G / np.sqrt(np.einsum('ii..., jj... ->ij...', G, G))

    @memoize
    def PDC(self):
        """ Partial directed coherence

        .. math:: \mathrm{PDC}_{ij}(f) = \\frac{A_{ij}(f)}{\sqrt{A_{:j}'(f) A_{:j}(f)}}
        
        References
        ----------
        L. A. Baccalá, K. Sameshima. Partial directed coherence: a new concept in neural structure
        determination. Biol. Cybernetics 84(6):463-474, 2001.
        """
        A = self.A()
        return np.abs(A / np.sqrt(np.sum(A.conj() * A, axis=0, keepdims=True)))

    @memoize
    def ffPDC(self):
        """ Full frequency partial directed coherence

        .. math:: \mathrm{ffPDC}_{ij}(f) = \\frac{A_{ij}(f)}{\sqrt{\sum_f A_{:j}'(f) A_{:j}(f)}}
        """
        A = self.A()
        return np.abs(A * self.nfft / np.sqrt(np.sum(A.conj() * A, axis=(0, 2), keepdims=True)))

    @memoize
    def PDCF(self):
        """ Partial directed coherence factor

        .. math:: \mathrm{PDCF}_{ij}(f) = \\frac{A_{ij}(f)}{\sqrt{A_{:j}'(f) \mathbf{C}^{-1} A_{:j}(f)}}
        
        References
        ----------
        L. A. Baccalá, K. Sameshima. Partial directed coherence: a new concept in neural structure
        determination. Biol. Cybernetics 84(6):463-474, 2001.
        """
        A = self.A()
        #TODO can we do that more efficiently?
        return np.abs(A / np.sqrt(np.einsum('aj..., ab..., bj... ->j...', A.conj(), self.Cinv(), A)))

    @memoize
    def GPDC(self):
        """ Generalized partial directed coherence

        .. math:: \mathrm{GPDC}_{ij}(f) = \\frac{|A_{ij}(f)|}
            {\sigma_i \sqrt{A_{:j}'(f) \mathrm{diag}(\mathbf{C})^{-1} A_{:j}(f)}}
            
        References
        ----------
        L. Faes, S. Erla, G. Nollo. Measuring Connectivity in Linear Multivariate Processes:
        Definitions, Interpretation, and Practical Analysis. Comput. Math. Meth. Med. 2012:140513, 2012.
        """
        A = self.A()
        return np.abs(A / np.sqrt(np.einsum('aj..., a..., aj..., ii... ->ij...', A.conj(), 1/np.diag(self.c), A, self.c)))

    @memoize
    def DTF(self):
        """ Directed transfer function

        .. math:: \mathrm{DTF}_{ij}(f) = \\frac{H_{ij}(f)}{\sqrt{H_{i:}(f) H_{i:}'(f)}}
        
        References
        ----------
        M. J. Kaminski, K. J. Blinowska. A new method of the description of the information flow
        in the brain structures. Biol. Cybernetics 65(3): 203-210, 1991.
        """
        H = self.H()
        return np.abs(H / np.sqrt(np.sum(H * H.conj(), axis=1, keepdims=True)))

    @memoize
    def ffDTF(self):
        """ Full frequency directed transfer function

        .. math:: \mathrm{ffDTF}_{ij}(f) = \\frac{H_{ij}(f)}{\sqrt{\sum_f H_{i:}(f) H_{i:}'(f)}}
        
        References
        ----------
        A. Korzeniewska, M. Mańczak, M. Kaminski, K. J. Blinowska, S. Kasicki. Determination of
        information flow direction among brain structures by a modified directed transfer 
        function (dDTF) method. J. Neurosci. Meth. 125(1-2): 195-207, 2003.
        """
        H = self.H()
        return np.abs(H * self.nfft / np.sqrt(np.sum(H * H.conj(), axis=(1, 2), keepdims=True)))

    @memoize
    def dDTF(self):
        """" Direct" directed transfer function

        .. math:: \mathrm{dDTF}_{ij}(f) = |\mathrm{pCOH}_{ij}(f)| \mathrm{ffDTF}_{ij}(f)
        
        References
        ----------
        A. Korzeniewska, M. Mańczak, M. Kaminski, K. J. Blinowska, S. Kasicki. Determination of
        information flow direction among brain structures by a modified directed transfer 
        function (dDTF) method. J. Neurosci. Meth. 125(1-2): 195-207, 2003.
        """
        return np.abs(self.pCOH()) * self.ffDTF()

    @memoize
    def GDTF(self):
        """ Generalized directed transfer function

        .. math:: \mathrm{GPDC}_{ij}(f) = \\frac{\sigma_j |H_{ij}(f)|}
            {\sqrt{H_{i:}(f) \mathrm{diag}(\mathbf{C}) H_{i:}'(f)}}
            
        References
        ----------
        L. Faes, S. Erla, G. Nollo. Measuring Connectivity in Linear Multivariate Processes:
        Definitions, Interpretation, and Practical Analysis. Comput. Math. Meth. Med. 2012:140513, 2012.
        """
        H = self.H()
        return np.abs(H / np.sqrt(np.einsum('ia..., aa..., ia..., j... ->ij...', H.conj(), self.c, H, 1/self.c.diagonal())))


def _inv3(x):
    identity = np.eye(x.shape[0])
    return np.array([sp.linalg.solve(a, identity) for a in x.T]).T
