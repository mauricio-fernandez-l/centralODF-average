# %% Import

import datetime, os
import numpy as np

from . import tensor_numpy as tn
from . import tensor_sympy as ts
from .constants import *

# %% Generation of a permutation basis with random permutations

def random(n, r_plus=0.3, info=False):
    '''
    random(n, r_plus) returns a permutation basis for the space of n-th-order
    isotropic tensors.
    '''
    d = tn.diso(n)
    B = tn.Biso(n)
    n_plus = round(r_plus * d + 1)

    P = np.array([np.random.permutation(n) for i in range(d + n_plus)])
    Bmat = np.array([np.reshape(np.transpose(B, p), int(3**n))
                     for p in P], dtype=np.int8)
    r = np.linalg.matrix_rank(Bmat)
    if info:
        print("First try: %i/%i = %.4f" % (r, d, r / d))

    while r < d:
        if info:
            print('Enriching started at:\n\t%s' % datetime.datetime.now())
        Pa = np.array([np.random.permutation(n) for i in range(n_plus)])
        P = np.concatenate((P, Pa), axis=0)
        Bmata = np.array([np.reshape(np.transpose(B, p), int(3**n))
                          for p in Pa], dtype=np.int8)
        Bmat = np.concatenate((Bmat, Bmata), axis=0)
        r = np.linalg.matrix_rank(Bmat)
        if info:
            print("Currently: %i/%i = %.4f" % (r, d, r / d))

    if info:
        print('Extracting linear independent permutations')
    _, uni = np.unique(Bmat, axis=0, return_index=True)
    P = P[uni]
    Bmat = Bmat[uni]

    li = tn.licqr(np.transpose(Bmat))
    return P[li]


def random_det(n, info=False):
    '''
    Incremental routine for the generation of a permutation basis based on
    metric matrix and determinant criterion.
    (Computation of the exact symbolic determinant is the bottle neck)
    '''
    d = tn.diso(n)
    B = tn.Biso(n).astype(int)

    P = [np.arange(n)]
    mv = [[0, 0, tn.sp(B, B)]]

    while len(P) < d:
        c = len(P) + 1
        P2 = P + [np.random.permutation(n)]
        Pc = np.transpose(B, P2[-1]).astype(int)
        mv2 = mv + [[i, c - 1, tn.sp(np.transpose(B, P2[i]), Pc)]
                    for i in range(len(P2))]
        met = np.zeros([c, c]).astype(int)
        for v in mv2:
            met[v[0], v[1]] = v[2]
            met[v[1], v[0]] = v[2]
        if ts.sym.det(ts.sym.Matrix(met)) != 0:
            P = P2
            mv = mv2
        if info:
            print('Currently: %i/%i = %.4f' % (len(P), d, len(P) / d))

    return P


def save_to_data(P):
    n = np.max(P) + 1
    path = DATA_FOLDER + 'pbasis' + str(n) + '.txt'
    np.savetxt(path, P, fmt='%i')
    print('...saved %s' % path)
    return path


def list_available(info=True):
    files = os.listdir(DATA_FOLDER)
    pbases = np.array([file for file in files if 'pbasis' in file])
    ns = [int(pb.split('pbasis')[1].split('.txt')[0]) for pb in pbases]
    order = np.argsort(ns)
    paths = [DATA_FOLDER + pb for pb in pbases[order]]
    if info:
        print('List of available permutation bases:')
        for path in paths:
            print('\t%s' % path)
    return paths


def exist(n):
    paths = list_available(info=False)
    check = any(['pbasis' + str(n) in path for path in paths])
    if not check:
        print(
            '...corresponding permutation basis NOT available.\n\tGenerate and save first.')
        list_available(info=True)
    return check


def load(n):
    check = exist(n)
    if check:
        P = np.loadtxt(DATA_FOLDER + 'pbasis' + str(n) + '.txt', np.int8)
        return P


def check_det(P, info=False):
    n = np.max(P) + 1
    B = tn.Biso(n)
    Bmat = np.array([np.reshape(np.transpose(B, p), int(3**n))
                     for p in P], dtype=np.int8)
    det = np.linalg.det(Bmat@np.transpose(Bmat))
    if info:
        print('...Computed determinant:')
        print(det)
    return det > 0