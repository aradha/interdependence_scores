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

1. Given a data matrix, $X$, of size $n \times d$ ($n$ samples, $d$ variables), one can compute IDS between all $d^2$ pairs of variables using: 

```python
ids.compute_IDS(X)
```

2.  