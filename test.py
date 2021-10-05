# %% Import

import numpy as np
import sympy as sym

import odf_cen_av.permutation_basis as pb
import odf_cen_av.d_tensors as dt
import odf_cen_av.tensor_numpy as tn
import odf_cen_av.tensor_sympy as ts
import odf_cen_av.texture as tex

# %%

def test_pb():
    n = 6
    P = pb.random(n)
    assert pb.check_det(P)

def test_dt():
    r = 2
    Dv = dt.Dv_compute(r)
    assert Dv.shape == (81, 3)

def test_elasticity():
    C = ts.gent('c', 4)
    index_sym = [(1, 0, 2, 3), (2, 3, 0, 1)]
    for isym in index_sym:
        C = ts.symmetrizeex(C, isym)
    C = ts.symmetrize_group(C, ts.sg_cub)
    mat_data = {
        C[0, 0, 0, 0]: 108, 
        C[1, 2, 1, 2]: 33, 
        C[0, 0, 1, 1]: 59
    }
    C = C.subs(mat_data)
    C = np.array(sym.flatten(C)).astype(float).reshape(4*[3,])
    r = C.ndim
    Ds = dt.Dv_load(r, as_tensors=True)
    norm = tn.nf(tn.lm(Ds[0], C))
    assert 261.91 < norm and norm < 262 

def test_texture():
    nfori = 21
    fori = 1 / nfori * np.ones(nfori)
    omori = np.pi * np.linspace(0, 1, nfori)
    Qori = [tn.rotm([1, 0, 0], omi) for omi in omori]
    odata = [fori, Qori]

    component_concentrations = np.array([
        0.3070,
        0.3861,
        0.3069
    ])
    rotation_axes = np.array([
        [1,0,0],
        [1,0,0],
        [1,0,0]
    ])
    rotation_angles = np.array([
        2.7404, 
        1.5706,
        0.4009
    ])
    texture_eigenvalues = np.array([
        [0.9967, 0.9921, 0.9887], 
        [0.9484, 0.9674, 0.9534],
        [0.9967, 0.9921, 0.9887]
    ])

    variables = [
        component_concentrations,
        rotation_axes,
        rotation_angles,
        texture_eigenvalues
    ]
    texture_coefficients_original = tex.tc(odata, 3)
    texture_coefficients_central = tex.tccen(variables)
    eps = tex.epsilonplus(
        texture_coefficients_original,
        texture_coefficients_central
    )
    assert 0.557 < eps and eps < 0.558