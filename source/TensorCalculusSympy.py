import sympy as sym
from itertools import permutations

import numpy as np

#%% Tensor algebra

def vec(a):
    '''
    vec(a) returns the symbolic tensor a as a flattened vector.
    '''
    return a.reshape(sym.prod(a.shape),1)

def sp(a,b):
    '''
    sp(a,b) returns the scalar product of the symbolic tensors a and b.
    '''
    return (sym.transpose(vec(a).tomatrix())*vec(b).tomatrix())[0,0]

def nf(a):
    '''
    nf(a) returns the Frobenius norm of a tensor a.
    '''
    return sym.sqrt(sp(a,a))

def tp(a,b):
    '''
    tp(a,b) is an abbreviation for sym.tensorproduct(a,b).
    '''
    return sym.tensorproduct(a,b)

def tc(a,i):
    '''
    tc(a,i) is an abbreviation for sym.tensorcontract(a,i).
    '''
    return sym.tensorcontraction(a,i)

def lm(a,b):
    '''
    lm(a,b) returns the linear map of the symbolic tensor b through the symbolic
    tensor a.
    '''
    da = a.shape
    ra = a.rank()
    rb = b.rank()
    d1 = da[0:ra-rb]
    d2 = da[ra-rb:]
    if ra==rb:
        return sp(a,b)
    elif ra-rb==1:
        return vec(sym.Array(a.reshape(sym.prod(d1),sym.prod(d2)).tomatrix()*b.reshape(sym.prod(d2),1).tomatrix()).reshape(*d1))
    else:
        return sym.Array(a.reshape(sym.prod(d1),sym.prod(d2)).tomatrix()*b.reshape(sym.prod(d2),1).tomatrix()).reshape(*d1)
    
def rp(Q,a):
    '''
    rp(Q,a) returns the Rayleigh product of the symbolic tensor Q applied to the tensor a.
    '''
    ra = a.rank()
    Qm = Q.tomatrix()
    if ra==1:
        return sym.Array(Qm*sym.Matrix(a))
    else:
        con = (1,ra+2-1)
        temp = tc(tp(Q,a),con)
        for i in range(ra-1):
            temp = tc(tp(Q,temp),con)
        return temp

def symmetrize(a):
    '''
    symmetrize(a) returns a symmetrized symbolic tensor based on the symbolic 
    tensor a.
    '''
    p = list(permutations(list(range(a.rank()))))
    for i in p:
        a = a.subs(sym.solve(a - sym.permutedims(a,i)))
    return a

def symmetrizeex(a,p):
    '''
    symmetrizeex(a,p) returns a symbolic tensor based on the explicit axes permutation p.
    '''
    a = a.subs(sym.solve(a - sym.permutedims(a,p)))
    return a

#%% Generate rotation matrix
    
def rm(n,om):
    '''
    rm(n,om) return a rotation with axis n and angle om.
    '''
    n2 = n/nf(n)
    return sym.cos(om)*I2 - sym.sin(om)*lm(pt,n2) + (1-sym.cos(om))*tp(n2,n2)

#%% Generate symbolic tensors
    
def gent(s,n):
    '''
    gent(s,n) returns a symbolic n-th-order tensor based on the symbol s.
    '''
    dims = [3]*n
    return sym.Array(sym.symbols(s+':'+':'.join([str(i) for i in dims])),dims)

#%% Harmonic tensors

def genh(s,n):
    '''
    genh(s,n) returns a symbolic harmoni n-th-order tensor based on the symbol s.
    '''
    if n==0:
        return sym.symbols(s)
    elif n==1:
        return gent(s,1)
    else:
        a = symmetrize(gent(s,n))
        if n==2:
            return a.subs(sym.solve(lm(a,I2))[0])
        else:
            return a.subs(sym.solve(lm(a,I2)))
        
def genhb(n):
    '''
    genhb(n) returns a basis for the space of n-th-order harmonic tensors.
    '''
    a = genh('s',n)
    v = sym.Array(list(a.free_symbols))
    return [sym.diff(a,v[i]) for i in range(len(v))]

def genhonb(n):
    '''
    genhonb(n) returns a orthonormal basis for the space of n-th-order 
    harmonic tensors.
    '''
    return orth(genhb(n))

#%% Linear algebra routines
    
def orth(b):
    '''
    orth(b) returns a orthogonalized basis from the basis b.
    '''
    d = b[0].shape
    G = sym.GramSchmidt([sym.Matrix(a) for a in b],True)
    return [sym.Array(a).reshape(*d) for a in G]

#%% Module constants

# Identity on vectors
I2 = sym.Array(sym.eye(3))

# Permutation tensor
pt = sym.MutableDenseNDimArray(np.zeros([3,3,3],dtype=np.int))
pt[0,1,2] = 1
pt[1,2,0] = 1
pt[2,0,1] = 1
pt[0,2,1] = -1
pt[2,1,0] = -1
pt[1,0,2] = -1

# Cubic symmetry group
sg_cub = [
        I2
        # faces
        ,rm(sym.Array([1,0,0]),sym.pi*2/4)
        ,rm(sym.Array([1,0,0]),sym.pi*2/4*2)
        ,rm(sym.Array([1,0,0]),sym.pi*2/4*3)
        ,rm(sym.Array([0,1,0]),sym.pi*2/4)
        ,rm(sym.Array([0,1,0]),sym.pi*2/4*2)
        ,rm(sym.Array([0,1,0]),sym.pi*2/4*3)
        ,rm(sym.Array([0,0,1]),sym.pi*2/4)
        ,rm(sym.Array([0,0,1]),sym.pi*2/4*2)
        ,rm(sym.Array([0,0,1]),sym.pi*2/4*3)        
        # edges
        ,rm(sym.Array([1,1,0]),sym.pi)
        ,rm(sym.Array([1,-1,0]),sym.pi)
        ,rm(sym.Array([1,0,1]),sym.pi)
        ,rm(sym.Array([1,0,-1]),sym.pi)
        ,rm(sym.Array([0,1,1]),sym.pi)
        ,rm(sym.Array([0,1,-1]),sym.pi)        
        # diagonals
        ,rm(sym.Array([1,1,1]),sym.pi*2/3)
        ,rm(sym.Array([1,1,1]),sym.pi*2/3*2)
        ,rm(sym.Array([-1,1,1]),sym.pi*2/3)
        ,rm(sym.Array([-1,1,1]),sym.pi*2/3*2)
        ,rm(sym.Array([-1,-1,1]),sym.pi*2/3)
        ,rm(sym.Array([-1,-1,1]),sym.pi*2/3*2)
        ,rm(sym.Array([1,-1,1]),sym.pi*2/3)
        ,rm(sym.Array([1,-1,1]),sym.pi*2/3*2)
        ]