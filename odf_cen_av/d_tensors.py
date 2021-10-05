# %% Import

import numpy as np
from scipy import integrate

from .permutation_basis import *

# %% Computation of isotropic tensors D_{<2r>\alpha}

def D(a, om):
    '''
    D(a,om) return the value of the Dirichlet kernel for alpha=a and omega=om.
    '''
    if a == 0:
        return 1
    else:
        return 1 + 2 * np.sum([np.cos(k * om) for k in range(1, a + 1)])


def m(om):
    '''
    m(om) returns the value of the function sin(om/2)**2/(2\\pi**2) for om.
    '''
    return np.sin(om / 2)**2 / (2 * np.pi**2)


def Dv_compute(r):
    '''
    Dv_compute(r) returns a matrix containing the isotropic tensors D_{<2r>\alpha} for
    \alpha in {0,1,2,...,r} flattened as column vectors.
    '''
    n = 2 * r
    check = exist(n)
    if check:
        print('...Loading permutation basis')
        print(datetime.datetime.now())
        P = load(n)
        B = tn.Biso(n)
        print('...Constructing basis')
        print(datetime.datetime.now())
        b = np.transpose(np.array([tn.flatten(np.transpose(B, p)) for p in P]))
        print('...Computing ONB')
        print(datetime.datetime.now())
        onb, _ = np.linalg.qr(b)

        def proj(om, i):
            return np.matmul(tn.flatten(
                tn.rpow(tn.rotm([1, 0, 0], om), r)), onb[:, i])
        print('...Computing integrals')
        print(datetime.datetime.now())
        c = np.array([[
            integrate.quad(lambda om: tn.dh(a) * D(a, om) * 4 * np.pi * m(om) * proj(om, i), 0, np.pi)[0]
            for a in range(r + 1)]
            for i in range(tn.diso(n))])
        print('...Done')
        print(datetime.datetime.now())
        return np.matmul(onb, c)


def Dv_list(info=True):
    files = os.listdir(DATA_FOLDER)
    Dvs = np.array([file for file in files if 'Dv' in file])
    rs = [int(Dv.split('Dv')[1].split('.txt')[0]) for Dv in Dvs]
    order = np.argsort(rs)
    paths = [DATA_FOLDER + Dv for Dv in Dvs[order]]
    if info:
        print('List of available Dv:')
        for path in paths:
            print('\t%s' % path)
    return paths


def Dv_exist(r):
    n = 2 * r
    paths = Dv_list(info=False)
    check = any(['Dv' + str(n) in path for path in paths])
    if not check:
        print('...corresponding Dv NOT available.\n\tGenerate and save first.')
        Dv_list(info=True)
    return check


def Dv_load(r, as_tensors=False):
    check = Dv_exist(r)
    n = 2 * r
    if check:
        Dv = np.loadtxt(DATA_FOLDER + 'Dv' + str(n) + '.txt')
        if not as_tensors:
            return Dv
        else:
            return Dv.T.reshape([-1] + (2 * r) * [3, ])