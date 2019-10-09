import math
from copy import deepcopy

import numpy as np
from numpy.linalg import eig
from numpy.linalg import inv
from numpy.linalg import svd
from numpy.linalg import norm

import matplotlib.pyplot as plt

# Remember all of these matrices have M^T = M
def decompose(M, eps = 1e-8):
    lambdas, vs = eig(M)
    n = M.shape[0]
    Os  = [i for i in range(n) if abs(lambdas[i]) < eps]
    pos = [i for i in range(n) if lambdas[i] > eps]
    neg = [i for i in range(n) if lambdas[i] < 0 - eps]
    return lambdas, vs, Os, pos, neg

def nullproject(M, v, eps= 1e-8):
    """
    Not a true projection, but an easy way to go from v to the quadratic kernel of M
    """
    n = M.shape[0]
    lambdas, vs, Os, pos, neg = decompose(M, eps)
    ovs = vs[:, Os]
    ols = lambdas[Os]
    oproj = np.matmul(ovs.T, v)

    if len(pos) == 0 or len(neg) == 0:
        if len(Os) == 0:
            return np.zeros(n)
        else:
            u = np.matmul(ovs, oproj)
            return u/norm(u)
    pvs = vs[:, pos]
    pls = lambdas[pos]
    pproj = np.matmul(pvs.T, v)

    nvs = vs[:, neg]
    nls = lambdas[neg]
    nproj = np.matmul(nvs.T, v)

    pcor = np.sqrt(np.dot(pls, pproj**2))
    ncor = np.sqrt(abs(np.dot(nls, nproj**2)))
    
    pproj2 = pproj/pcor
    nproj2 = nproj/ncor

    u = np.matmul(pvs, pproj2) + np.matmul(nvs, nproj2)
    if len(Os) != 0:
        u += np.matmul(ovs, oproj)
    return u / norm(u)

def convolve(basis, Matrix):
    """
    Given orthonormal basis, computes the action of M on that basis only
    """
    return np.matmul(basis.T, np.matmul(Matrix, basis))

def spanProjection(S):
    """
    Returns a basis and projection matrix onto the span of the columns of S
    """
    A  = np.matmul(S, S.T)
    n = A.shape[0]
    T, v, D = svd(A)
    w  = np.array([float(el > eps) for el in v])
    inds = [i for i in range(n) if w[i] > 0.1]
    B = D[inds].T
    Q  = np.matmul(T,  np.matmul(w*np.eye(5),  D))
    return Q, B

def bothKernels(M, X, eps = 1e-8):
    """
    Returns the projection matrix and a basis that maps to the kernel of both span(X) 
    (where X is column vector orthonormal basis) and span(MX)
    """
    n = M.shape[0]
    kerX = np.eye(n) - np.matmul(X, X.T) 
    S = np.matmul(M, X)
    Q, _ = spanProjection(S)
    kerMx = np.eye(n) - Q
    kernel = np.matmul(kerX, kerMx)
    s, v, d = svd(kernel)
    inds = [i for i in range(n) if v[i] > 0.1]
    B = d[inds].T
    return kernel, B

def expandQuasiClique(M, X, eps = 1e-8, v = None):
    K, B = bothKernels(M, X, eps)
    M2 = convolve(B, M)
    n = K.shape[0]
    m = M2.shape[0]
    if type(v) != type(None):
        v = np.matmul(B.T, v)
    else:
        v = np.random.random(m)
        v = v / np.linalg.norm(v)
    u = nullProject(M, v, eps)
    x = np.matmul(B, u).reshape(n, 1)
    if norm(x) > 0.1:
        x = x / np.linalg.norm(x)
        return concatenate(X, x), x
    else:
        return X, x


     
    






    

