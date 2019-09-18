#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 15:34:16 2018

@author: fernandez
"""

import numpy as np
from scipy import integrate
from scipy.optimize import minimize

from . import TensorCalculusNumpy as tn
from . import TensorCalculusSympy as ts

import datetime

#%% Generation of a permutation basis with random permutations

def randompbasis(n,r_plus=0.3,info=False):
    '''
    randompbasis(n,r_plus) returns a permutation basis for the space of n-th-order
    isotropic tensors.
    '''
    d = tn.diso(n)
    B = tn.Biso(n)
    n_plus = round(r_plus*d+1)
    
    P = np.array([np.random.permutation(n) for i in range(d+n_plus)])
    Bmat = np.array([np.reshape(np.transpose(B,p),int(3**n)) for p in P],dtype=np.int8)
    r = np.linalg.matrix_rank(Bmat)
    if info: print("First try: %i/%i = %.4f" % (r,d,r/d))
    
    while r < d:
        if info: print('Enriching started at:\n\t%s' % datetime.datetime.now())
        Pa = np.array([np.random.permutation(n) for i in range(n_plus)])
        P = np.concatenate((P,Pa),axis=0)
        Bmata = np.array([np.reshape(np.transpose(B,p),int(3**n)) for p in Pa],dtype=np.int8)
        Bmat = np.concatenate((Bmat,Bmata),axis=0)
        r = np.linalg.matrix_rank(Bmat)
        if info: print("Currently: %i/%i = %.4f" % (r,d,r/d))
        
    if info: print('Extracting linear independent permutations')
    _,uni = np.unique(Bmat,axis=0,return_index=True)
    P = P[uni]
    Bmat = Bmat[uni]
    
    li = tn.licqr(np.transpose(Bmat))
    return P[li]

def randompbasisdet(n,info=False):
    '''
    Incremental routine for the generation of a permutation basis based on 
    metric matrix and determinant criterion.
    (Computation of the exact symbolic determinant is the bottle neck)
    '''
    d = tn.diso(n)
    B = tn.Biso(n).astype(int)
    
    P = [np.arange(n)]
    mv = [[0,0,tn.sp(B,B)]]
    
    while len(P)<d:
        c = len(P)+1
        P2 = P+[np.random.permutation(n)]
        Pc = np.transpose(B,P2[-1]).astype(int)
        mv2 = mv+[[i,c-1,tn.sp(np.transpose(B,P2[i]),Pc)] for i in range(len(P2))]
        met = np.zeros([c,c]).astype(int)
        for v in mv2:
            met[v[0],v[1]] = v[2]
            met[v[1],v[0]] = v[2]
        if ts.sym.det(ts.sym.Matrix(met))!=0:
            P = P2
            mv = mv2
        if info: print('Currently: %i/%i = %.4f' % (len(P),d,len(P)/d))

    return P

#%% Computation of isotropic tensors D_{<2r>\alpha}

def D(a,om):
    '''
    D(a,om) return the value of the Dirichlet kernel for alpha=a and omega=om.
    '''
    if a==0:
        return 1
    else:
        return 1+2*np.sum([np.cos(k*om) for k in range(1,a+1)])
    
def m(om):
    '''
    m(om) returns the value of the function sin(om/2)**2/(2\pi**2) for om.
    '''
    return np.sin(om/2)**2/(2*np.pi**2)

def computeDv(r):
    '''
    copmuteDv(r) returns a matrix containing the isotropic tensors D_{<2r>\alpha} for
    \alpha in {0,1,2,...,r} flattened as column vectors.
    '''
    n = 2*r
    print('...Loading permutation basis')
    print(datetime.datetime.now())
    P = np.loadtxt('source/data/pbasis'+str(n)+'.txt',np.int8)
    B = tn.Biso(n)
    print('...Constructing basis')
    print(datetime.datetime.now())
    b = np.transpose(np.array([tn.flatten(np.transpose(B,p)) for p in P]))
    print('...Computing ONB')
    print(datetime.datetime.now())
    onb,_ = np.linalg.qr(b)
    def proj(om,i):
        return np.matmul(tn.flatten(tn.rpow(tn.rotm([1,0,0],om),r)),onb[:,i])
    print('...Computing integrals')
    print(datetime.datetime.now())
    c = np.array([[
            integrate.quad(lambda om: tn.dh(a)*D(a,om)*4*np.pi*m(om)*proj(om,i),0,np.pi)[0]
        for a in range(r+1)]
        for i in range(tn.diso(n))])
    print('...Done')
    print(datetime.datetime.now())
    return np.matmul(onb,c)
    
#%% Routines for orientation average and computation of texture coefficients
     
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
        out += fori[i]*tn.rp(Qori[i],a)
    return out

def tc(odata,r):
    '''
    tc(odata,r) returns a list with all texture coefficients up to tensor order r.
    '''
    # Load onbs of harmonic tensors up to tensor order r
    honb = [np.loadtxt('source/data/honb'+str(a)+'.txt') for a in range(1,r+1)]
    honb = [[np.reshape(honb[a-1][b-1],a*(3,)) for b in range(1,tn.dh(a)+1)] for a in range(1,r+1)]
    # Compute texture coefficients of original ODF up to relevant tensor order r
    V = [[
            oa(odata,honb[a-1][b-1]) 
            for b in range(1,tn.dh(a)+1)] 
            for a in range(1,r+1)]
    return V

def tcnorms(V):
    '''
    tcnorms(V) returns a list with all Frobenius norm of the texture coefficients V.
    '''
    r = len(V)
    return np.concatenate(
            [np.array([tn.nf(V[a-1][b-1]) for b in range(1,tn.dh(a)+1)]) 
            for a in range(1,r+1)])
    
def tccen(cendata):
    '''
    tccen(cendata) returns a list with all texture coefficients based on Fourier data of the 
    central ODF cendata.
    '''
    # Extract central data
    fL,nL,omL,lL = cendata
    QL = np.array([tn.rotm(n,om) for n,om in zip(nL,omL)])
    r = lL.shape[-1]
    # Load onbs of harmonic tensors up to tensor order r
    honb = [np.loadtxt('source/data/honb'+str(a)+'.txt') for a in range(1,r+1)]
    honb = [[np.reshape(honb[a-1][b-1],a*(3,)) for b in range(1,tn.dh(a)+1)] for a in range(1,r+1)]
    # Compute and return
    return [[
        np.sum(np.array([f*l*tn.rp(Q,honb[a-1][b-1]) for f,l,Q in zip(fL,lL[:,a-1],QL)]),axis=0)
        for b in range(1,tn.dh(a)+1)]
        for a in range(1,r+1)
        ] 
    
#%% Computation of \epsilon^+ and minimization for convex combination of central ODFs    
    
def epsilonplus(V1,V2):
    '''
    epsilonplus(V1,V2) returns the quantity \\epsilon^+ for given texture 
    coefficients V1 of original data and V2 of the model.
    '''
    r = len(V1)
    return np.sum(np.array([
            tn.dh(a)*tn.nf(V1[a-1][b-1]-V2[a-1][b-1]) 
            for a in range(1,r+1) 
            for b in range(1,tn.dh(a)+1)
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
            D(a,om)/tn.dh(a) 
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
        c2 = np.array([tn.nf(n)-1 for n in nL]) # norm of rotation axis vector
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