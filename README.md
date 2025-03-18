# InterDependence Scores (IDS) 

Code to compute IDS between sets of variables.  Supports both numpy only users and PyTorch users.  

## Prerequisites 

Python version >= 3.7.

## Installation 

Clone the repo and run the following command to install ids:

```bash
pip install -e .
```

## Example usage

The following two use cases are supported.  

1. Given a data matrix, X, of size n x d (n samples, d variables), one can compute IDS between all pairs of variables using: 

```python
ids.compute_IDS(X)
```

2.  Given data matrices, X and Y, of size n x d and n x c (n samples, d variables in X and c variables in Y), one can compute IDS between all d variables in X and all c variables of Y using: 

```python
ids.compute_IDS(X, Y)
```

Notebooks demonstrating various use cases of IDS are provided in examples.

## Parameters
The ```compute_IDS``` function supports the following parameters: 
1. ```X``` - a matrix of size n samples by d variables
2. ```Y``` (optional, default=None) - a matrix of size n samples x c variables
3. ```num_terms``` (optional, default=6) - number of terms used in the Taylor series approximation of the Gaussian kernel 
4. ```p_norm``` (optional, default='max') - 'max' means using IDS-max, integer 1 means using IDS-1, integer 2 means using IDS-2 
5. ```p_val``` (optional, default=False) - boolean indicating whether to return p-values or not
6. ```num_tests``` (optional, default=100) - number of permutation tests to run for computing p-values
7. ```bandwidth_term``` (optional, default=1/2) - constant multiplier in exponent of Gaussian kernel