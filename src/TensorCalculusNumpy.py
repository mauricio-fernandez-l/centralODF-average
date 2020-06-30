import numpy as np
import sympy as sym
import os
from . import TensorCalculusSympy as ts

# %% Tensor algebra routines


def flatten(a):
    '''
    flatten(a) returns the tensor a flattened.
    '''
    return np.reshape(a, np.prod(np.shape(a)))


def sp(a, b):
    '''
    sp(a,b) returns the scalar product (full contraction) of tensors of identical
    tensor order a and b.
    '''
    return np.tensordot(a, b, a.ndim)


def nf(a):
    '''
    nf(a) returns the Frobenius norm of the tensor a.
    '''
    return np.sqrt(sp(a, a))


def lm(a, b):
    '''
    lm(a,b) returns the linear map of the tensor b through the tensor a.
    '''
    return np.tensordot(a, b, b.ndim)


def tp(a, b):
    '''
    tp(a,b) returns the tensor product of the tensors a and b.
    '''
    return np.tensordot(a, b, 0)


def tpow(a, n):
    '''
    tpow(a,n) returns the tensor power of the tensor a to the n.
    '''
    if n == 0:
        return 1
    else:
        out = a
        for i in range(1, n):
            out = tp(out, a)
        return out


def rp(a, b):
    '''
    rp(a,b) returns the Rayleigh product of a applied on b.
    '''
    n = b.ndim
    p = [n - 1] + list(range(n - 1))
    c = b
    for i in range(n):
        c = np.transpose(np.tensordot(c, np.transpose(a), 1), p)
    return c


def rpow(a, n):
    '''
    rpow(a,n) returns the Rayleigh power of the tensor a to the n.
    '''
    p = tuple(range(0, int(2 * n - 1), 2)) + tuple(range(1, int(2 * n), 2))
    return np.transpose(tpow(a, n), p)

# %% Rotations


def rotm(n, om):
    '''
    rotm(n,om) returns the rotation around the rotation axis n and rotation
    angle om.
    '''
    n = np.array(n)
    n = n / nf(n)
    n0 = np.cos(om) * I2
    n1 = -np.sin(om) * lm(pt, n)
    n2 = (1 - np.cos(om)) * tp(n, n)
    return n0 + n1 + n2

# %% Isotropic tensors


def Biso(n):
    '''
    Biso(n) returns the the fundamental n-th-order isotropic tensor.
    '''
    if n % 2 == 0:
        return tpow(I2, int(n / 2))
    else:
        return tp(pt, tpow(I2, int((n - 3) / 2)))


def diso(n):
    '''
    diso(n) return the dimension of the space of n-th-order isotropic tensors.
    '''
    a0 = 1
    a1 = 0
    for i in range(2, n + 1):
        a2 = (i - 1) * (2 * a1 + 3 * a0) / (i + 1)
        a0 = a1
        a1 = a2
    return int(a2)


def fict(n):
    '''
    Fundamental isotropic cartesian tensors for even tensor order n.
    '''
    if os.path.isfile('source/data/fict' + str(n) + '.txt'):
        return np.loadtxt('source/data/fict' + str(n) + '.txt', dtype=np.int)
    if n == 4:
        return np.array([
            [0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2]
        ], dtype=np.int)
    else:
        out = []
        for i in range(1, n):
            head = [0, i]
            tail0 = list(set(range(1, n)) - set([i]))
            for previous in fict(n - 2):
                tail = [tail0[p] for p in previous]
                out = out + [head + tail]
        if not os.path.isfile('source/data/fict' + str(n) + '.txt'):
            np.savetxt(
                'source/data/fict' +
                str(n) +
                '.txt',
                np.array(out),
                fmt='%i')
        return out

# %% Harmonic tensors


def dh(a):
    '''
    dh(a) returns the dimension of the space of a-th-order harmonic tensors.
    '''
    return 1 + 2 * a


def genhonb(n):
    '''
    genhonb(n) returns a orthonormal basis for the space of n-th-order
    harmonic tensors as a Numpy array.
    '''
    bs = ts.genhonb(n)
    bn = np.array([
        np.array(sym.Matrix(sym.flatten(b))).astype(np.float64).reshape(*n*[3,])
        for b in bs
        ])
    return bn


def checkhonb(r):
    '''
    checkhonb(r) checks if harmonic bases up to tensor order r are found in
    the data directory.
    '''
    check = True
    rexist = 0
    while check and rexist < r:
        rexist += 1
        check = os.path.isfile('src/data/honb' + str(rexist) + '.txt')
    if check:
        print('Harmonic ONBs already generated up to tensor order %i' % r)
    if not check:
        rexist -= 1
    if rexist < r:
        print('...Harmonic bases up to r=%i exist in directory...' % rexist)
        print(
            '...Generating and saving harmonic bases for r=%i to r=%i' %
            (rexist + 1, r))
        for rr in range(rexist + 1, r + 1):
            print('...Generating harmonic basis für r=%i' % rr)
            b = genhonb(rr)
            b = b.reshape([b.shape[0],-1])
            np.savetxt('src/data/honb' + str(rr) + '.txt', b)

# %% Linear algebra routines


def licqr(a):
    '''
    licqr(a) returns a list with the indices of the linear independent columns
    of the matrix a.
    '''
    _, r = np.linalg.qr(a)
    r = abs(np.diag(r))
    rmax = max(r)
    return [i for i in range(np.shape(r)[0]) if r[i] > rmax * 1e-10]

# %% Module constants


# Identity on vectors
I2 = np.eye(3)

# Permutation tensor
pt = np.zeros((3, 3, 3))
pt[0, 1, 2] = 1
pt[1, 2, 0] = 1
pt[2, 0, 1] = 1
pt[1, 0, 2] = -1
pt[2, 1, 0] = -1
pt[0, 2, 1] = -1
