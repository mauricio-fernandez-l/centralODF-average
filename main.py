#%% Information

info = '''
Supplementary material corresponding to the work
    On the orientation average based on central orientation density functions for polycrystalline materials
Author:     Dr. Mauricio Fernández 
Email:      mauricio.fernandez.lb@gmail.com
ORCID:      http://orcid.org/0000-0003-1840-1243

The present main file calls for the routines of the modules in 'source' and reproduces
all examples shown in the above referred work. The author hopes that this script
and the corresponding modules are helpful for the interested readers/users in order 
to reproduce the results on their own machines.

The modules require numpy, sympy, scipy, itertools, os and datetime.
'''
print(info)

#%% Setup

import numpy as np
import datetime

import source.TensorCalculusNumpy as tn
import source.TensorCalculusSympy as ts
import source.CentralODFAverage as cen

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

#%% Demonstration 1: generate permutation basis for isotropic tensors

print('\n**********************************************')
print('**********************************************')
print('***** Demonstration 1 ************************\n')

# Information
info = '''
The following example demonstrates the computation of a permutation 
basis for n = 7.
'''
print(info)

# Computation
n = 7
print('Compute permutation basis for n='+str(n))
t1 = datetime.datetime.now()
P = cen.randompbasis(n)
t2 = datetime.datetime.now()
print('Computation time: '+str(t2-t1))
np.savetxt('source/data/pbasis'+str(n)+'.txt',P,fmt='%d')

#%% Demonstration 2: compute isotropic tensors $D_{<2r>\\alpha}$

print('\n**********************************************')
print('**********************************************')
print('***** Demonstration 2 ************************\n')

# Information
info = '''
The following example demonstrates the computation of the isotropic
tensors D_{<2r>\alpha} for r=2.
'''
print(info)

# Compute tensors    
r = 2
print('Compute tensors $D_{<2r>\\alpha}$ for r='+str(r))
t1 = datetime.datetime.now()
Dv = cen.computeDv(r)
t2 = datetime.datetime.now()
print('Computation time: '+str(t2-t1))
np.savetxt('source/data/Dvec'+str(2*r)+'.txt',Dv)

if r==2:
    info = '''
    Check for r=2 (i.e., n=2*r=4) known analytical results for identity on isotropic 
    2nd-order tensors Iiso, identity on anti-symmetric/skew 2nd-order tensors 
    Ia, and identity on harmonic 2nd-order tensors Ih. Remark: {Iiso,Ia,Ih} form
    a projector representation of the general isotropic fourth-order tensor.
    '''
    print(info)
    Dv = np.loadtxt('source/data/Dvec4.txt').T.reshape(3,3,3,3,3)
    IdI = tn.tp(tn.I2,tn.I2)
    Iiso = IdI/3
    I4 = np.transpose(IdI,(0,2,1,3))
    Is = (I4+np.transpose(I4,(0,1,3,2)))/2
    Ia = (I4-np.transpose(I4,(0,1,3,2)))/2
    Ih = Is-Iiso
    print('Deviation of D_{<4>0} from Iiso:\t%.4e' % tn.nf(Dv[0]-Iiso))
    print('Deviation of D_{<4>1} from Ia:\t\t%.4e' % tn.nf(Dv[1]-Ia))
    print('Deviation of D_{<4>2} from Ih:\t\t%.4e' % tn.nf(Dv[2]-Ih))

#%% Demonstration 3: example for r=4

print('\n**********************************************')
print('**********************************************')
print('***** Demonstration 3 ************************\n')

# Information
info = '''
The following example corresponds to the example for r=4 
in the manuscript, documented section 3.1, equations (68) - (71).
'''
print(info)

# Initial fourth-order tensor
A = np.zeros((3,3,3,3))
A[0,0,0,0] = 1
A[0,1,0,0] = 2
A[0,0,1,2] = 3

r = A.ndim
n = 2*r
Ds = np.loadtxt('source/data/Dvec'+str(n)+'.txt')
Ds = [np.reshape(Ds[:,i],n*(3,)) for i in range(r+1)]
Aiso = tn.lm(Ds[0],A)
Aaniso = A-Aiso

print('Norm of A:\n\t%2.4f' % tn.nf(A))
print('Norm of isotropic and anisotropic parts of A\n\t%2.4f\n\t%2.4f' % (tn.nf(Aiso),tn.nf(Aaniso)))
print('Norms of D_{<2r>\\alpha}[A]')
for i in range(r+1): print('\t%2.4f' % tn.nf(tn.lm(Ds[i],A)))

# Minor symmetrized tensor
Am = (A
     +np.transpose(A,axes=(1,0,2,3))
     +np.transpose(A,axes=(0,1,3,2))
     +np.transpose(A,axes=(1,0,3,2))
     )/4
Amiso = tn.lm(Ds[0],Am)
Amaniso = Am-Amiso
print('\nNorm of minor symmetric Am:\n\t%2.4f' % tn.nf(Am))
print('Norm of isotropic and anisotropic parts of Am\n\t%2.4f\n\t%2.4f' % (tn.nf(Amiso),tn.nf(Amaniso)))
print('Norms of D_{<2r>\\alpha}[Am]')
for i in range(r+1): print('\t%2.4f' % tn.nf(tn.lm(Ds[i],Am)))

# Additionally major symmetrized tensor
AM = (Am
     +np.transpose(Am,axes=(2,3,0,1))
     )/2
AMiso = tn.lm(Ds[0],AM)
AManiso = AM-AMiso
print('\nNorm of additionaly major symmetric AM:\n\t%2.4f' % tn.nf(AM))
print('Norm of isotropic and anisotropic parts of AM\n\t%2.4f\n\t%2.4f' % (tn.nf(AMiso),tn.nf(AManiso)))
print('Norms of D_{<2r>\\alpha}[AM]')
for i in range(r+1): print('\t%2.4f' % tn.nf(tn.lm(Ds[i],AM)))


#%% Demonstration 4: example for r=5

print('\n**********************************************')
print('**********************************************')
print('***** Demonstration 4 ************************\n')

# Information
info = '''
The following example corresponds to the example for r=5 
in the manuscript, documented section 3.1, equations (72) - (73).
'''
print(info)

# Reference tensor
A = np.zeros((3,3,3,3,3))
A[0,0,0,0,0] = 1
A[0,1,0,2,0] = 2
A[1,0,0,0,0] = 3
A[0,0,1,0,0] = 4

r = A.ndim
n = 2*r
Ds = np.loadtxt('source/data/Dvec'+str(n)+'.txt')
Ds = [np.reshape(Ds[:,i],n*(3,)) for i in range(r+1)]
Aiso = tn.lm(Ds[0],A)
Aaniso = A-Aiso

print('Norm of A:\n\t%2.4f' % tn.nf(A))
print('Norm of isotropic and anisotropic parts of A\n\t%2.4f\n\t%2.4f' % (tn.nf(Aiso),tn.nf(Aaniso)))
print('Norms of D_{<2r>\\alpha}[A]')
for i in range(r+1): print('\t%2.4f' % tn.nf(tn.lm(Ds[i],A)))

#%% Demonstration 5: finite elasticity with C4 and C6

print('\n**********************************************')
print('**********************************************')
print('***** Demonstration 5 ************************\n')

# Information
info = '''
The following example corresponds to the finite elasticity 
example in the manuscript documented section 3.2 for aluminum. 
The symbolic generation of the sixth-order cubic tensor takes 
time. It is probably a good idea to go and get some tea or coffee.
'''
print(info)

# C4: 
print('\nGenerating a symbolic C4 tensor...')
print('\t%s' % datetime.datetime.now())
# generate symbolic tensor, symmetrize indices, fulfill cubic group conditions
C = ts.gent('c',4)
print('Solving index symmetries...')
print('\t%s' % datetime.datetime.now())
index_sym = [(1,0,2,3),(2,3,0,1)]
for i in index_sym:
    C = ts.symmetrizeex(C,i)
print('Solving cubic group conditions...')
print('\t%s' % datetime.datetime.now())
for Q in ts.sg_cub:
    C = C.subs(ts.sym.solve(C-ts.rp(Q,C)))
print('Free components')
print(C.free_symbols)
print('\t%s' % datetime.datetime.now())
# Insert material data for aluminum
print('Inserting material data for aluminum...')
print('\t%s' % datetime.datetime.now())
mat_data = {
        C[0,0,0,0]:108
        ,C[1,2,1,2]:33
        ,C[0,0,1,1]:59
        }
C = np.array(C.subs(mat_data)).astype(np.float).reshape(3,3,3,3)
# Load corresponding isotropic tensors and reshape to full tensors
print('Loading corresponding isotropic tensors D_{<2r>\\alpha}...')
print('\t%s' % datetime.datetime.now())
r = C.ndim
n = 2*r
Ds = np.loadtxt('source/data/Dvec'+str(n)+'.txt')
Ds = [np.reshape(Ds[:,i],n*(3,)) for i in range(r+1)]
# Compute linear mapts and display norms
print('Computing norms...')
print('\t%s' % datetime.datetime.now())
for i in range(r+1): print('\t%2.4f' % tn.nf(tn.lm(Ds[i],C)))
print('\t(see equation (82))')
print('\t%s' % datetime.datetime.now())

# C6: 
print('\n\nGenerating a symbolic C6 tensor...')
print('\t%s' % datetime.datetime.now())
# generate symbolic tensor, symmetrize indices, fulfill cubic group conditions
C = ts.gent('c',6)
print('Solving index symmetries...')
print('[This takes a while, 2-5 min. May be, get some coffee :D]')
print('\t%s' % datetime.datetime.now())
index_sym = [(1,0,2,3,4,5),(2,3,0,1,4,5),(0,1,4,5,2,3)]
for i in index_sym:
    C = ts.symmetrizeex(C,i)
print('Solving cubic group conditions...')
print('\t%s' % datetime.datetime.now())
for Q in ts.sg_cub:
    C = C.subs(ts.sym.solve(C-ts.rp(Q,C)))
print('Free components')
print(C.free_symbols)
print('\t%s' % datetime.datetime.now())
# Insert material data for aluminum
print('Inserting material data for aluminum...')
print('\t%s' % datetime.datetime.now())
mat_data = {
        C[0,0,0,0,0,0]:-1100
        ,C[0,0,0,0,1,1]:-371
        ,C[0,0,1,1,2,2]:104
        ,C[0,0,1,2,1,2]:39
        ,C[0,0,0,2,0,2]:-421
        ,C[1,2,0,2,0,1]:-22
        }
C = np.array(C.subs(mat_data)).astype(np.float).reshape(3,3,3,3,3,3)
# Load corresponding isotropic tensors and reshape to full tensors
print('Loading corresponding isotropic tensors D_{<2r>\\alpha}...')
print('\t%s' % datetime.datetime.now())
r = C.ndim
n = 2*r
Ds = np.loadtxt('source/data/Dvec'+str(n)+'.txt')
Ds = [np.reshape(Ds[:,i],n*(3,)) for i in range(r+1)]
# Compute linear mapts and display norms
print('Computing norms...')
print('\t%s' % datetime.datetime.now())
for i in range(r+1): print('\t%2.4f' % tn.nf(tn.lm(Ds[i],C)))
print('\t(see equation (82))')
print('\t%s' % datetime.datetime.now())

#%% Demonstration 6: minimization of epsilon^+

print('\n**********************************************')
print('**********************************************')
print('***** Demonstration 6 ************************\n')

# Information
info = '''
The following example corresponds to the quantitative texture 
analysis example in the manuscript documented section 3.3 
(minimization of \epsilon^+). The numerical optimizations 
take some time. The corresponding restuls can be found in
the manuscript, section 3.3, Table 2.
'''
print(info)

# Original orientation data
nfori = 21
fori = 1/nfori*np.ones(nfori)
omori = np.pi*np.linspace(0,1,nfori)
Qori = [tn.rotm([1,0,0],omi) for omi in omori]
odata = [fori,Qori]

# Tensor order
r = 3

# Check existing ONBs for harmonic tensors
tn.checkhonb(r)

# Optimize components for minimization of \\epsilon^+
for m in [2,3,4]:
    print('\n----------------------------------------')
    print('\nNumber of components: %i' % m)
    res = cen.mineplus(odata,r=r,ncomponents=m,npointsLa=4)
    print('\nComponent concentrations')
    for ress in res[0]: print('\t%.4f' % ress)
    print('\nRotation axes (n_1,n_2,n_3) (in rows) of central orientations of corresponding component')
    print(res[1])
    print('\nRotation angles of central orientations of corresponding component')
    print(res[2])
    print('\nTexture eigenvalues (lambda_1,lambda_2,...) (in rows) of corresponding component')
    print(res[3])
    print('Check value of $\\epsilon^+$')
    Vori = cen.tc(odata,r)  
    Vcen = cen.tccen(res)
    print('\t%.4f' % cen.epsilonplus(Vori,Vcen))
    print('(see Table 2)')
    
# Direct evaluation evaluation
print('\n----------------------------------------')

info = ''' 
Depending on the current version of Python, Numpy and Scipy the user might get
different results out of the numerical optimization. Therefore, the user may 
directly input concentrations, variables of central orientations and texture eigenvalues
of interest and evaluate the bound $\\epsilon^+$. This can be done in the present file
based on the following lines of code in order to, e.g, reproduce the final results
in the case of m=2 components:
    
    component_concentrations = np.array([0.7572,0.2428])
    rotation_axes = np.array([[1,0,0],[0,-0.7071,0.7071]])
    rotation_angles = np.array([1.5708,3.1416])
    texture_eigenvalues = np.array([[0.7991,0.5179,0.3112],[0,0.2,-0.1429]])
    variables = [component_concentrations,rotation_axes,rotation_angles,texture_eigenvalues]
    texture_coefficients_original = cen.tc(odata,3)
    texture_coefficients_central = cen.tccen(variables)
    eps = cen.epsilonplus(texture_coefficients_original,texture_coefficients_central)
    print('%.4f' % eps)
'''
print(info)

component_concentrations = np.array([0.7572,0.2428])
rotation_axes = np.array([[1,0,0],[0,-0.7071,0.7071]])
rotation_angles = np.array([1.5708,3.1416])
texture_eigenvalues = np.array([[0.7991,0.5179,0.3112],[0,0.2,-0.1429]])
variables = [component_concentrations,rotation_axes,rotation_angles,texture_eigenvalues]
texture_coefficients_original = cen.tc(odata,3)
texture_coefficients_central = cen.tccen(variables)
eps = cen.epsilonplus(texture_coefficients_original,texture_coefficients_central)
print('%.4f' % eps)
