import sympy as sym
from itertools import permutations

#%% Tensor algebra

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

def symmetrizeS(a):
    '''
    symmetrizeS(a) returns a symmetrized symbolic tensor based on the symbolic 
    tensor a.
    '''
    p = list(permutations(list(range(a.rank()))))
    for i in p:
        a = a.subs(sym.solve(a - sym.permutedims(a,i)))
    return a

#%% Generate symbolic tensors
    
def gentS(s,n):
    '''
    gentS(s,n) returns a symbolic n-th-order tensor based on the symbol s.
    '''
    dims = [3]*n
    return sym.Array(sym.symbols(s+':'+':'.join([str(i) for i in dims])),dims)

#%% Harmonic tensors

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

def genhonbS(n):
    '''
    genhonbS(n) returns a orthonormal basis for the space of n-th-order 
    harmonic tensors.
    '''
    return orthS(genhbS(n))

#%% Linear algebra routines
    
def orthS(b):
    '''
    orthS(b) returns a orthogonalized basis from the basis b.
    '''
    d = b[0].shape
    G = sym.GramSchmidt([sym.Matrix(a) for a in b],True)
    return [sym.Array(a).reshape(*d) for a in G]

#%% Module constants
    
I2S = sym.Array(sym.eye(3))