# Kronecker Group Elastic-Net

Method for solving a linear regression problem subject to group LASSO and ridge penalisation when the model has a Kronecker structure. 

Let $A\in \mathbb{R}^{p\times q}$, $D\in \mathbb{R}^{n\times k}$, $y \in \mathbb{R^{pn}}$, $\lambda, \eta_i > 0$ and $\alpha \in [0, 1]$,
$$
	\underset{\theta}{\min} \; \frac{1}{2}\|\sum_{i \in \Gamma} Z_i \theta_i - y\|_2^2 + \lambda \left[(1-\alpha) \frac{1}{2} \|\theta\|_2^2 + \alpha \sum_{i\in \Gamma} \eta_i \|\theta_i\|_2\right]
$$ where $Z = D \otimes A$ and $\Gamma$ is a partition of $\{0,\dots,qk-1\}$ with the following constraints for each i-group: $\exists r_i, a_i, b_i \in \mathbb{N}$ such as,
$$
    i \;\text{div}\; q = \{r_i,\dots,r_i\} \\
    i \;\text{mod}\; q = \{a_i,a_i+1,\dots,b_i-2, b_i-1\} 
$$ where $\text{div}$ is the integer division and $\text{mod}$ is the module operator. The [kron_groupper](https://github.com/sfcaracciolo/kron_groupper) lib has been developed to handle this constraints. 

The solver consists of an accelerated proximal gradient method and a block coordinate descent algorithm coded in NumPy (underscored version `_kapg` & `_kbcd`) and Cython (noscored version `kapg` & `kbcd`). The Cython version uses scipy's BLAS wrappers to speed up vector/matrix operations.

## Install

Clone the repo and compile the cython source through 
```bash
cythonize -i src\kron_solvers\solvers.pyx
```
Or try with pip
```bash
pip install git+https://github.com/sfcaracciolo/kron_solvers.git
```

## Test
Run the following tests in order to check the solvers
```bash
python tests\apg.py
python tests\bcd.py
```
