#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 16:34:56 2018

@author: fernandez
"""

#%% Setup

#from IPython import get_ipython
#get_ipython().magic('clear')

import numpy as np
import datetime
import routines as rou
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

#%% Information
info = '''
Supplementary material corresponding to the work
    On the orientation average based on central orientation density functions for polycrystalline materials
Author:     Dr. Mauricio Fernández 
Email:      mauricio.fernandez.lb@gmail.com
ORCID:      http://orcid.org/0000-0003-1840-1243

The present main file calls for the routines of the file 'routines.py' and reproduces
all examples shown in the above referred work. The author hopes that this script
and the corresponding routines are helpful for the interested readers/users in order 
to reproduce the results on their own machines.
'''
print(info)

#%% Demonstration 1: generate permutation basis for isotropic tensors
print('\n***** Demonstration 1 ******\n')

n = 7
print('Compute permutation basis for n='+str(n))
t1 = datetime.datetime.now()
P = rou.randompbasis(n)
t2 = datetime.datetime.now()
print('Computation time: '+str(t2-t1))
np.savetxt('pbasis'+str(n)+'.txt',P,fmt='%d')

#%% Demonstration 2: compute isotropic tensors $D_{<2r>\\alpha}$
print('\n***** Demonstration 2 ******\n')

# Compute tensors    
r = 2
print('Compute tensors $D_{<2r>\\alpha}$ for r='+str(r))
t1 = datetime.datetime.now()
Dv = rou.computeDv(r)
t2 = datetime.datetime.now()
print('Computation time: '+str(t2-t1))
np.savetxt('Dvec'+str(2*r)+'.txt',Dv)

if r==2:
    info = '''
    Check for r=2 (i.e., n=2*r=4) known analytical results for identity on isotropic 
    2nd-order tensors Iiso, identity on anti-symmetric/skew 2nd-order tensors 
    Ia, and identity on harmonic 2nd-order tensors Ih. Remark: {Iiso,Ia,Ih} form
    a projector representation of the general isotropic fourth-order tensor.
    '''
    print(info)
    Dv = np.loadtxt('Dvec4.txt').T.reshape(3,3,3,3,3)
    IdI = rou.tp(rou.I2,rou.I2)
    Iiso = IdI/3
    I4 = np.transpose(IdI,(0,2,1,3))
    Is = (I4+np.transpose(I4,(0,1,3,2)))/2
    Ia = (I4-np.transpose(I4,(0,1,3,2)))/2
    Ih = Is-Iiso
    print('Deviation of D_{<4>0} from Iiso:\t%.4e' % rou.nf(Dv[0]-Iiso))
    print('Deviation of D_{<4>1} from Ia:\t\t%.4e' % rou.nf(Dv[1]-Ia))
    print('Deviation of D_{<4>2} from Ih:\t\t%.4e' % rou.nf(Dv[2]-Ih))

#%% Demonstration 3: example for r=4
print('\n***** Demonstration 3 ******\n')
print('Example for r=4')

A = np.zeros((3,3,3,3))
A[0,0,0,0] = 1
A[0,1,0,0] = 2
A[0,0,1,2] = 3

r = A.ndim
n = 2*r
Ds = np.loadtxt('Dvec'+str(n)+'.txt')
Ds = [np.reshape(Ds[:,i],n*(3,)) for i in range(r+1)]
Aiso = rou.lm(Ds[0],A)
Aaniso = A-Aiso

print('Norm of A:\n\t%2.4f' % rou.nf(A))
print('Norm of isotropic and anisotropic parts of A\n\t%2.4f\n\t%2.4f' % (rou.nf(Aiso),rou.nf(Aaniso)))
print('Norms of D_{<2r>\\alpha}[A]')
for i in range(r+1): print('\t%2.4f' % rou.nf(rou.lm(Ds[i],A)))

# Minor symmetrized tensor
Am = (A
     +np.transpose(A,axes=(1,0,2,3))
     +np.transpose(A,axes=(0,1,3,2))
     +np.transpose(A,axes=(1,0,3,2))
     )/4
Amiso = rou.lm(Ds[0],Am)
Amaniso = Am-Amiso
print('\nNorm of minor symmetric Am:\n\t%2.4f' % rou.nf(Am))
print('Norm of isotropic and anisotropic parts of Am\n\t%2.4f\n\t%2.4f' % (rou.nf(Amiso),rou.nf(Amaniso)))
print('Norms of D_{<2r>\\alpha}[Am]')
for i in range(r+1): print('\t%2.4f' % rou.nf(rou.lm(Ds[i],Am)))

# Additionally major symmetrized tensor
AM = (Am
     +np.transpose(Am,axes=(2,3,0,1))
     )/2
AMiso = rou.lm(Ds[0],AM)
AManiso = AM-AMiso
print('\nNorm of additionaly major symmetric AM:\n\t%2.4f' % rou.nf(AM))
print('Norm of isotropic and anisotropic parts of AM\n\t%2.4f\n\t%2.4f' % (rou.nf(AMiso),rou.nf(AManiso)))
print('Norms of D_{<2r>\\alpha}[Am]')
for i in range(r+1): print('\t%2.4f' % rou.nf(rou.lm(Ds[i],AM)))


#%% Demonstration 4: example for r=5
print('\n***** Demonstration 4 ******\n')
print('Example for r=5')

# Reference tensor
A = np.zeros((3,3,3,3,3))
A[0,0,0,0,0] = 1
A[0,1,0,2,0] = 2
A[1,0,0,0,0] = 3
A[0,0,1,0,0] = 4

r = A.ndim
n = 2*r
Ds = np.loadtxt('Dvec'+str(n)+'.txt')
Ds = [np.reshape(Ds[:,i],n*(3,)) for i in range(r+1)]
Aiso = rou.lm(Ds[0],A)
Aaniso = A-Aiso

print('Norm of A:\n\t%2.4f' % rou.nf(A))
print('Norm of isotropic and anisotropic parts of A\n\t%2.4f\n\t%2.4f' % (rou.nf(Aiso),rou.nf(Aaniso)))
print('Norms of D_{<2r>\\alpha}[A]')
for i in range(r+1): print('\t%2.4f' % rou.nf(rou.lm(Ds[i],A)))

#%% Demonstration 5: minimization of epsilon^+
print('\n***** Demonstration 5 ******\n')
print('Minimization of $\\epsilon^+$')

# Original orientation data
nfori = 21
fori = 1/nfori*np.ones(nfori)
omori = np.pi*np.linspace(0,1,nfori)
Qori = [rou.rotm([1,0,0],omi) for omi in omori]
odata = [fori,Qori]

# Tensor order
r = 3

# Check existing ONBs for harmonic tensors
rou.checkhonb(r)

# Optimize components for minimization of \\epsilon^+
for m in [2,3,4]:
    print('\n----------------------------------------')
    print('\nNumber of components: %i' % m)
    res = rou.mineplus(odata,r=r,ncomponents=m,npointsLa=4)
    print('\nComponent concentrations')
    for ress in res[0]: print('\t%.4f' % ress)
    print('\nRotation axes (n_1,n_2,n_3) (in rows) of central orientations of corresponding component')
    print(res[1])
    print('\nRotation angles of central orientations of corresponding component')
    print(res[2])
    print('\nTexture eigenvalues (lambda_1,lambda_2,...) (in rows) of corresponding component')
    print(res[3])
    print('Check value of $\\epsilon^+$')
    Vori = rou.tc(odata,r)  
    Vcen = rou.tccen(res)
    print('\t%.4f' % rou.epsilonplus(Vori,Vcen))
    
# Direct evaluation evaluation
print('\n----------------------------------------')

info = ''' 
Depending on the current version of Python, Numpy and Scipy the user might get
different results out of the numerical optimization. Therefore, the user may 
directly input concentrations, variables of central orientations and texture eigenvalues
of interest and evaluate the bound $\\epsilon^+$. This can be done in the present file
based on the following lines of code:
    
    component_concentrations = np.array([0.7572,0.2428])
    rotation_axes = np.array([[1,0,0],[0,-0.7071,0.7071]])
    rotation_angles = np.array([1.5708,3.1416])
    texture_eigenvalues = np.array([[0.7991,0.5179,0.3112],[0,0.2,-0.1429]])
    variables = [component_concentrations,rotation_axes,rotation_angles,texture_eigenvalues]
    texture_coefficients_original = rou.tc(odata,3)
    texture_coefficients_central = rou.tccen(variables)
    eps = rou.epsilonplus(texture_coefficients_original,texture_coefficients_central)
    print('%.4f' % eps)
'''
print(info)

component_concentrations = np.array([0.7572,0.2428])
rotation_axes = np.array([[1,0,0],[0,-0.7071,0.7071]])
rotation_angles = np.array([1.5708,3.1416])
texture_eigenvalues = np.array([[0.7991,0.5179,0.3112],[0,0.2,-0.1429]])
variables = [component_concentrations,rotation_axes,rotation_angles,texture_eigenvalues]
texture_coefficients_original = rou.tc(odata,3)
texture_coefficients_central = rou.tccen(variables)
eps = rou.epsilonplus(texture_coefficients_original,texture_coefficients_central)
print('%.4f' % eps)
