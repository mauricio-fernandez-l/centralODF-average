#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 15:34:16 2018

@author: fernandez
"""

import numpy as np
import sympy as sym
from scipy import integrate
from itertools import permutations
from scipy.optimize import minimize
import os.path

#%% Routines for demonstration 1

def tp(a,b):
    '''
    tp(a,b) returns the tensor product of the tensors a and b.
    '''
    return np.tensordot(a,b,0)
def tpow(a,n):
    '''
    tpow(a,n) returns the tensor power of the tensor a to the n.
    '''
    if n==0:
        return 1
    else:
        out = a
        for i in range(1,n):
            out = tp(out,a)
        return out
    
def Biso(n):
    '''
    Biso(n) returns the the fundamental n-th-order isotropic tensor. 
    '''
    if n%2 == 0:
        return tpow(I2,int(n/2))
    else:
        return tp(pt,tpow(I2,int((n-3)/2)))
def diso(n):
    '''
    diso(n) return the dimension of the space of n-th-order isotropic tensors.
    '''
    a0 = 1
    a1 = 0
    for i in range(2,n+1):
        a2 = (i-1)*(2*a1+3*a0)/(i+1)
        a0 = a1
        a1 = a2
    return int(a2)

def licqr(a):
    '''
    licqr(a) returns a list with the indices of the linear independent columns
    of the matrix a.
    '''
    _,r = np.linalg.qr(a)
    r = abs(np.diag(r))
    rmax = max(r)
    return [i for i in range(np.shape(r)[0]) if r[i]>rmax*1e-10]

def randompbasis(n,r_plus=0.3):
    '''
    randompbasis(n,r_plus) returns a permutation basis for the space of n-th-order
    isotropic tensors.
    '''
    d = diso(n)
    B = Biso(n)
    n_plus = round(r_plus*d+1)
    
    P = np.array([np.random.permutation(n) for i in range(d+n_plus)])
    Bmat = np.array([np.reshape(np.transpose(B,p),int(3**n)) for p in P],dtype=np.int8)
    r = np.linalg.matrix_rank(Bmat)
    
    while r < d:
        Pa = np.array([np.random.permutation(n) for i in range(n_plus)])
        P = np.concatenate((P,Pa),axis=0)
        Bmata = np.array([np.reshape(np.transpose(B,p),int(3**n)) for p in Pa],dtype=np.int8)
        Bmat = np.concatenate((Bmat,Bmata),axis=0)
        r = np.linalg.matrix_rank(Bmat)
        
    _,uni = np.unique(Bmat,axis=0,return_index=True)
    P = P[uni]
    Bmat = Bmat[uni]
    
    li = licqr(np.transpose(Bmat))
    return P[li]

# Module constants:
# Identity on vectors
I2 = np.eye(3)
# Permutation tensor
pt = np.zeros((3,3,3))
pt[0,1,2] = 1
pt[1,2,0] = 1
pt[2,0,1] = 1
pt[1,0,2] = -1
pt[2,1,0] = -1
pt[0,2,1] = -1

#%% Routines for demonstration 2

def D(a,om):
    '''
    D(a,om) return the value of the Dirichlet kernel for alpha=a and omega=om.
    '''
    if a==0:
        return 1
    else:
        return 1+2*np.sum([np.cos(k*om) for k in range(1,a+1)])
def dh(a):
    '''
    dh(a) returns the dimension of the space of a-th-order harmonic tensors.
    '''
    return 1+2*a
def sp(a,b):
    '''
    sp(a,b) returns the scalar product (full contraction) of tensors of identical
    tensor order a and b.
    '''
    return np.tensordot(a,b,a.ndim)
def nf(a):
    '''
    nf(a) returns the Frobenius norm of the tensor a.
    '''
    return np.sqrt(sp(a,a))
def lm(a,b):
    '''
    lm(a,b) returns the linear map of the tensor b through the tensor a.
    '''
    return np.tensordot(a,b,b.ndim)
def rotm(n,om):
    '''
    rotm(n,om) returns the rotation around the rotation axis n and rotation 
    angle om.
    '''
    n = np.array(n)
    n = n/nf(n)
    n0 = np.cos(om)*I2
    n1 = -np.sin(om)*lm(pt,n)
    n2 = (1-np.cos(om))*tp(n,n)
    return n0+n1+n2
def m(om):
    '''
    m(om) returns the value of the function sin(om/2)**2/(2\pi**2) for om.
    '''
    return np.sin(om/2)**2/(2*np.pi**2)
def flatten(a):
    '''
    flatten(a) returns the tensor a flattened.
    '''
    return np.reshape(a,np.prod(np.shape(a)))
def rpow(a,n):
    '''
    rpow(a,n) returns the Rayleigh power of the tensor a to the n.
    '''
    p = tuple(range(0,int(2*n-1),2))+tuple(range(1,int(2*n),2))
    return np.transpose(tpow(a,n),p)

def computeDv(r):
    '''
    copmuteDv(r) returns a matrix containing the isotropic tensors D_{<2r>\alpha} for
    \alpha in {0,2,...,r} flattened as column vectors.
    '''
    n = 2*r
    P = np.loadtxt('pbasis'+str(n)+'.txt',np.int8)
    B = Biso(n)
    b = np.transpose(np.array([flatten(np.transpose(B,p)) for p in P]))
    onb,_ = np.linalg.qr(b)
    def proj(om,i):
        return np.matmul(flatten(rpow(rotm([1,0,0],om),r)),onb[:,i])
    c = np.array([[
            integrate.quad(lambda om: dh(a)*D(a,om)*4*np.pi*m(om)*proj(om,i),0,np.pi)[0]
        for a in range(r+1)]
        for i in range(diso(n))])
    return np.matmul(onb,c)
    
#%% Routines for demonstration 5
    
def rp(a,b):
    '''
    rp(a,b) returns the Rayleigh product of a applied on b.
    '''
    n = b.ndim
    p = [n-1]+list(range(n-1))
    c = b
    for i in range(n):
        c = np.transpose(np.tensordot(c,np.transpose(a),1),p)
    return c

# Symbolic routines
def vecS(a):
    '''
    vecS(a) returns the symbolic tensor a as a flattened vector.
    '''
    return a.reshape(sym.prod(a.shape),1)
def spS(a,b):
    '''
    spS(a,b) returns the scalar product of the symbolic tensors a and b.
    '''
    return (sym.transpose(vecS(a).tomatrix())*vecS(b).tomatrix())[0,0]
def gentS(s,n):
    '''
    gentS(s,n) returns a symbolic n-th-order tensor based on the symbol s.
    '''
    dims = [3]*n
    return sym.Array(sym.symbols(s+':'+':'.join([str(i) for i in dims])),dims)
def symmetrizeS(a):
    '''
    symmetrizeS(a) returns a symmetrized symbolic tensor based on the symbolic 
    tensor a.
    '''
    p = list(permutations(list(range(a.rank()))))
    for i in p:
        a = a.subs(sym.solve(a - sym.permutedims(a,i)))
    return a
def lmS(a,b):
    '''
    lmS(a,b) returns the linear map of the symbolic tensor b through the symbolic
    tensor a.
    '''
    da = a.shape
    ra = a.rank()
    rb = b.rank()
    d1 = da[0:ra-rb]
    d2 = da[ra-rb:]
    if ra==rb:
        return spS(a,b)
    elif ra-rb==1:
        return vecS(sym.Array(a.reshape(sym.prod(d1),sym.prod(d2)).tomatrix()*b.reshape(sym.prod(d2),1).tomatrix()).reshape(*d1))
    else:
        return sym.Array(a.reshape(sym.prod(d1),sym.prod(d2)).tomatrix()*b.reshape(sym.prod(d2),1).tomatrix()).reshape(*d1)
I2S = sym.Array(sym.eye(3))
def genhS(s,n):
    '''
    genhS(s,n) returns a symbolic harmoni n-th-order tensor based on the symbol s.
    '''
    if n==0:
        return sym.symbols(s)
    elif n==1:
        return gentS(s,1)
    else:
        a = symmetrizeS(gentS(s,n))
        if n==2:
            return a.subs(sym.solve(lmS(a,I2S))[0])
        else:
            return a.subs(sym.solve(lmS(a,I2S)))
def genhbS(n):
    '''
    genhbS(n) returns a basis for the space of n-th-order harmonic tensors.
    '''
    a = genhS('s',n)
    v = sym.Array(list(a.free_symbols))
    return [sym.diff(a,v[i]) for i in range(len(v))]
def orthS(b):
    '''
    orthS(b) returns a orthogonalized basis from the basis b.
    '''
    d = b[0].shape
    G = sym.GramSchmidt([sym.Matrix(a) for a in b],True)
    return [sym.Array(a).reshape(*d) for a in G]
def genhonbS(n):
    '''
    genhonbS(n) returns a orthonormal basis for the space of n-th-order 
    harmonic tensors.
    '''
    return orthS(genhbS(n))
def genhonb(n):
    '''
    genhonb(n) returns a orthonormal basis for the space of n-th-order 
    harmonic tensors as a Numpy array.
    '''
    return np.array(sym.Matrix(genhonbS(n))).astype(np.float64)
def checkhonb(r):
    '''
    checkhonb(r) checks if harmonic bases up to tensor order r are found in 
    the working directory.
    '''
    check = True
    rexist = 0
    while check and rexist<r:
        rexist += 1
        check = os.path.isfile('./honb'+str(rexist)+'.txt')
    if check:
        print('Harmonic ONBs already generated up to tensor order %i' % r)
    if check==False:
        rexist -=1
    if rexist<r:
        print('...Harmonic bases up to r=%i exist in directory...' % rexist)
        print('...Generating and saving harmonic bases for r=%i to r=%i' % (rexist+1,r))
        for rr in range(rexist+1,r+1): 
            print('...Generating harmonic basis für r=%i' % rr)
            np.savetxt('honb'+str(rr)+'.txt',genhonb(rr))
        
def oa(odata,a):
    '''
    oa(odata,a) return the orientation average of the tensor a for given orientation
    data odata.
    '''
    # Extract original orientation data
    fori, Qori = odata
    # Compute orientation average
    out = np.zeros_like(a)
    for i in range(len(fori)):
        out += fori[i]*rp(Qori[i],a)
    return out
def tc(odata,r):
    '''
    tc(odata,r) returns a list with all texture coefficients up to tensor order r.
    '''
    # Load onbs of harmonic tensors up to tensor order r
    honb = [np.loadtxt('honb'+str(a)+'.txt') for a in range(1,r+1)]
    honb = [[np.reshape(honb[a-1][b-1],a*(3,)) for b in range(1,dh(a)+1)] for a in range(1,r+1)]
    # Compute texture coefficients of original ODF up to relevant tensor order r
    V = [[
            oa(odata,honb[a-1][b-1]) 
            for b in range(1,dh(a)+1)] 
            for a in range(1,r+1)]
    return V
def tcnorms(V):
    '''
    tcnorms(V) returns a list with all Frobenius norm of the texture coefficients V.
    '''
    r = len(V)
    return np.concatenate(
            [np.array([nf(V[a-1][b-1]) for b in range(1,dh(a)+1)]) 
            for a in range(1,r+1)])    
def tccen(cendata):
    '''
    tccen(cendata) returns a list with all texture coefficients based on Fourier data of the 
    central ODF cendata.
    '''
    # Extract central data
    fL,nL,omL,lL = cendata
    QL = np.array([rotm(n,om) for n,om in zip(nL,omL)])
    r = lL.shape[-1]
    # Load onbs of harmonic tensors up to tensor order r
    honb = [np.loadtxt('honb'+str(a)+'.txt') for a in range(1,r+1)]
    honb = [[np.reshape(honb[a-1][b-1],a*(3,)) for b in range(1,dh(a)+1)] for a in range(1,r+1)]
    # Compute and return
    return [[
        np.sum(np.array([f*l*rp(Q,honb[a-1][b-1]) for f,l,Q in zip(fL,lL[:,a-1],QL)]),axis=0)
        for b in range(1,dh(a)+1)]
        for a in range(1,r+1)
        ] 
def epsilonplus(V1,V2):
    '''
    epsilonplus(V1,V2) returns the quantity \\epsilon^+ for given texture 
    coefficients V1 of original data and V2 of the model.
    '''
    r = len(V1)
    return np.sum(np.array([
            dh(a)*nf(V1[a-1][b-1]-V2[a-1][b-1]) 
            for a in range(1,r+1) 
            for b in range(1,dh(a)+1)
            ]))

def mineplus(odata,r,ncomponents,npointsLa):
    '''
    mineplus(odata,r,ncomponents,npointsLa) performs the numerical minimization
    of the bound \epsilon^+ for given orientation data odata, tensor order r,
    number of texture components ncomponents and number of points npointsLa for
    the approximation of the set \Lambda_r of relevant texture eigenvalues.
    '''
    # Compute texture coefficients of original ODF up to relevant tensor order r
    Vori = tc(odata,r)
    norms = tcnorms(Vori)
    print('Min and max norms of texture coefficients of original ODF: \n\t%2.4f\n\t%2.4f' % (norms.min(),norms.max()))
        
    # Generate points in set Lambda_r^a 
    lpoints = np.array([[
            D(a,om)/dh(a) 
            for a in range(1,r+1)] 
            for om in np.linspace(0,np.pi,npointsLa)])
    
    #*******************************************************
    # Setup for numerical minimization
    
    # Transformation of vetor of variables
    def VectorToQuantities(v):
        fL = v[0:ncomponents]
        nL = np.reshape(v[ncomponents:ncomponents+3*ncomponents],(ncomponents,3))
        omL = v[ncomponents+3*ncomponents:ncomponents+3*ncomponents+ncomponents]
        wL = np.reshape(v[5*ncomponents:],(ncomponents,npointsLa))
        qL = [fL,nL,omL,wL]
        return qL
    def QuantitiesToVector(qL):
        return np.concatenate([q.flatten() for q in qL])
    
    # Definition of constraints
    def ceq(v):
        '''
        Equality constraints
        '''
        fL,nL,omL,wL = VectorToQuantities(v)
        c1 = np.array([np.sum(fL)-1]) # sum of all mode concentrations
        c2 = np.array([nf(n)-1 for n in nL]) # norm of rotation axis vector
        c3 = np.array([np.sum(w)-1 for w in wL]) # sum of weights for convex combination in Lambda_r^a
        return np.concatenate((c1,c2,c3))
    def cineq(v):
        '''
        Inequality (non-negativity) constraints 
        '''
        fL,nL,omL,wL = VectorToQuantities(v)
        return np.concatenate((fL,omL,np.pi-omL,wL.flatten()))
    
    # Function for numerical minimization
    def minf(v):
        fL,nL,omL,wL = VectorToQuantities(v)
        lL = np.matmul(wL,lpoints) # texture eigenvalues of modes in Lambda_r^a
        cendata = [fL,nL,omL,lL]
        return epsilonplus(Vori,tccen(cendata))
    
    # Initial guess and value
    fL0 = np.ones(ncomponents)/ncomponents
    nL0 = np.array([np.array([1,1,1])/np.sqrt(3) for i in range(ncomponents)])
    omL0 = np.linspace(0,np.pi,ncomponents)
    wL0 = np.ones((ncomponents,npointsLa))/npointsLa
    v0 = QuantitiesToVector([fL0,nL0,omL0,wL0])
    print('Initial value of function (\\epsilon^+): \n\t%2.4f' % minf(v0))
    
    # Results
    cdef = ({'type':'eq','fun':ceq},{'type':'ineq','fun':cineq})
    res = minimize(minf,v0,constraints=cdef,options={'disp':True})
    q = VectorToQuantities(res.x)
    q[3] = np.matmul(q[3],lpoints)
    return q