# %% Import

import numpy as np
from scipy.optimize import minimize

from . import tensor_numpy as tn
from .d_tensors import *

# %% Routines for orientation average and computation of texture coefficients

def oa(odata, a):
    '''
    oa(odata,a) return the orientation average of the tensor a for given orientation
    data odata.
    '''
    # Extract original orientation data
    fori, Qori = odata
    # Compute orientation average
    out = np.zeros_like(a)
    for i in range(len(fori)):
        out += fori[i] * tn.rp(Qori[i], a)
    return out


def tc(odata, r):
    '''
    tc(odata,r) returns a list with all texture coefficients up to tensor order r.
    '''
    # Load onbs of harmonic tensors up to tensor order r
    honb = [np.loadtxt(DATA_FOLDER + 'honb' + str(a) + '.txt')
            for a in range(1, r + 1)]
    honb = [[np.reshape(honb[a - 1][b - 1], a * (3,))
             for b in range(1, tn.dh(a) + 1)] for a in range(1, r + 1)]
    # Compute texture coefficients of original ODF up to relevant tensor order
    # r
    V = [[
        oa(odata, honb[a - 1][b - 1])
        for b in range(1, tn.dh(a) + 1)]
        for a in range(1, r + 1)]
    return V


def tcnorms(V):
    '''
    tcnorms(V) returns a list with all Frobenius norm of the texture coefficients V.
    '''
    r = len(V)
    return np.concatenate(
        [np.array([tn.nf(V[a - 1][b - 1]) for b in range(1, tn.dh(a) + 1)])
         for a in range(1, r + 1)])


def tccen(cendata):
    '''
    tccen(cendata) returns a list with all texture coefficients based on Fourier data of the
    central ODF cendata.
    '''
    # Extract central data
    fL, nL, omL, lL = cendata
    QL = np.array([tn.rotm(n, om) for n, om in zip(nL, omL)])
    r = lL.shape[-1]
    # Load onbs of harmonic tensors up to tensor order r
    honb = [np.loadtxt(DATA_FOLDER + 'honb' + str(a) + '.txt')
            for a in range(1, r + 1)]
    honb = [[np.reshape(honb[a - 1][b - 1], a * (3,))
             for b in range(1, tn.dh(a) + 1)] for a in range(1, r + 1)]
    # Compute and return
    return [[
        np.sum(np.array([f * l * tn.rp(Q, honb[a - 1][b - 1]) for f, l, Q in zip(fL, lL[:, a - 1], QL)]), axis=0)
        for b in range(1, tn.dh(a) + 1)]
        for a in range(1, r + 1)
    ]

# %% Computation of \\epsilon^+ and minimization for convex combination of central ODFs


def epsilonplus(V1, V2):
    '''
    epsilonplus(V1, V2) returns the quantity \\epsilon^+ for given texture
    coefficients V1 of original data and V2 of the model.
    '''
    r = len(V1)
    return np.sum(np.array([
        tn.dh(a) * tn.nf(V1[a - 1][b - 1] - V2[a - 1][b - 1])
        for a in range(1, r + 1)
        for b in range(1, tn.dh(a) + 1)
    ]))


def mineplus(odata, r, ncomponents, npointsLa):
    '''
    mineplus(odata,r,ncomponents,npointsLa) performs the numerical minimization
    of the bound \\epsilon^+ for given orientation data odata, tensor order r,
    number of texture components ncomponents and number of points npointsLa for
    the approximation of the set \\Lambda_r of relevant texture eigenvalues.
    '''
    # Compute texture coefficients of original ODF up to relevant tensor order
    # r
    Vori = tc(odata, r)
    norms = tcnorms(Vori)
    print(
        'Min and max norms of texture coefficients of original ODF: \n\t%2.4f\n\t%2.4f' %
        (norms.min(), norms.max()))

    # Generate points in set Lambda_r^a
    lpoints = np.array([[
        D(a, om) / tn.dh(a)
        for a in range(1, r + 1)]
        for om in np.linspace(0, np.pi, npointsLa)])

    # *******************************************************
    # Setup for numerical minimization

    # Transformation of vetor of variables
    def VectorToQuantities(v):
        fL = v[0:ncomponents]
        nL = np.reshape(v[ncomponents:ncomponents + 3 *
                          ncomponents], (ncomponents, 3))
        omL = v[ncomponents + 3 * ncomponents:ncomponents +
                3 * ncomponents + ncomponents]
        wL = np.reshape(v[5 * ncomponents:], (ncomponents, npointsLa))
        qL = [fL, nL, omL, wL]
        return qL

    def QuantitiesToVector(qL):
        return np.concatenate([q.flatten() for q in qL])

    # Definition of constraints
    def ceq(v):
        '''
        Equality constraints
        '''
        fL, nL, omL, wL = VectorToQuantities(v)
        c1 = np.array([np.sum(fL) - 1])  # sum of all mode concentrations
        # norm of rotation axis vector
        c2 = np.array([tn.nf(n) - 1 for n in nL])
        # sum of weights for convex combination in Lambda_r^a
        c3 = np.array([np.sum(w) - 1 for w in wL])
        return np.concatenate((c1, c2, c3))

    def cineq(v):
        '''
        Inequality (non-negativity) constraints
        '''
        fL, nL, omL, wL = VectorToQuantities(v)
        return np.concatenate((fL, omL, np.pi - omL, wL.flatten()))

    # Function for numerical minimization
    def minf(v):
        fL, nL, omL, wL = VectorToQuantities(v)
        # texture eigenvalues of modes in Lambda_r^a
        lL = np.matmul(wL, lpoints)
        cendata = [fL, nL, omL, lL]
        return epsilonplus(Vori, tccen(cendata))

    # Initial guess and value
    fL0 = np.ones(ncomponents) / ncomponents
    nL0 = np.array([np.array([1, 1, 1]) / np.sqrt(3)
                    for i in range(ncomponents)])
    omL0 = np.linspace(0, np.pi, ncomponents)
    wL0 = np.ones((ncomponents, npointsLa)) / npointsLa
    v0 = QuantitiesToVector([fL0, nL0, omL0, wL0])
    print('Initial value of function (\\epsilon^+): \n\t%2.4f' % minf(v0))

    # Results
    cdef = ({'type': 'eq', 'fun': ceq}, {'type': 'ineq', 'fun': cineq})
    res = minimize(minf, v0, constraints=cdef, options={'disp': True})
    q = VectorToQuantities(res.x)
    q[3] = np.matmul(q[3], lpoints)
    return q
