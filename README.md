Kernel Conditional Moment Constraints for Confounding Robust Inference
======================================================================
[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/kstoneriv3/confounding_robust_inferenc/LICENSE)
[![Documentation](http://github.com/kstoneriv3/confounding_robust_inference/actions/workflows/docs.yaml/badge.svg)](https://github.com/kstoneriv3/confounding_robust_inference/actions/workflows/docs.yaml)



This repository contains source code to replicate the results of the above paper.
The core part of the code is provided as a python package `cri` and the codes for the other parts of the experiments can be found in the examples folder as jupyter notebooks.


Installation
------------

To install the package `kcmc` with `pip`, move to the directory of this file and use
```bash
pip intall kcmc
```
or 
```bash
pip install -e kcmc
```
if you want to edit the installed package.



Documentation
-------------

TODO


Development
-----------
For formatting, testing, and building documentation, we use `make` and `tox`.
If you would like to do these in your local environment, run
```bash
make
```
Alternatively, you can conduct these in an isolated environment using tox as 
```bash
tox
```
