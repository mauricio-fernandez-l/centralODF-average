# centralODF-averages

by Mauricio Fernández.

Contact: mauricio.fernandez.lb@gmail.com

Content:

* [Description](#Description)
* [Source code](#Source-code)
* [Demonstration notebooks](#Demonstration-notebooks)
* [Dependencies](#Dependencies)

## Description

Python 3.5+ package and demonstration files for

**[1] "On the orientation average based on central orientation density functions for polycrystalline materials"**

by Mauricio Fernández (publication in Journal of Elasticity <https://link.springer.com/article/10.1007/s10659-019-09754-8>).

## Source code

The Python 3.5+ source code is organized in modules in the package `odf_cen_av`.

## Demonstration notebooks

The first two demonstration notebooks show how permutation bases for isotropic tensor spaces can be generated and how the isotropic D tensors in [1] may be computed.

* [01 Generation of a permutation basis](01_permutation_basis.ipynb)
* [02 D tensors](02_D_tensors.ipynb)

The remaining three demonstration notebooks correspond to the examples discussed in [1] in section 3.

* [03 Materials design](03_materials_design.ipynb): see pubished work [1], section 3.1
* [04 Higher order elasticity](04_higher_order_elasticity.ipynb): see pubished work [1], section 3.2
* [05 Texture](03_materials_design.ipynb): see pubished work [1], section 3.3

## Dependencies

The package depends on `numpy`, `scipy` and `sympy` and has been tested with Python 3.5/3.6/3.7/3.8/3.9. Users may want to consider installing [miniconda](https://docs.conda.io/en/latest/miniconda.html), create an environment with

```shell
conda create -n odf python=3.9 numpy scipy sympy
```

and activate the environment `odf`

```shell
conda activate odf
```

in order to use the package. Additionally, users may want to install the `jupyter` packages

```shell
conda install jupyter
```

in order to open the demonstration notebooks. The package routines may be tested with `pytest`

```shell
conda install pytest
```

with the test script `test.py`

```shell
python test.py
```
