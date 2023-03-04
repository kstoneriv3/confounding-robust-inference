Kernel Conditional Moment Constraints for Confounding Robust Inference
======================================================================
[![Build Status](http://github.com/kstoneriv3/cri/workflows/build/badge.svg?event=push)](https://github.com/kstoneriv3/cri/actions/workflows/build.yml)
![PyPI - downloads](https://img.shields.io/pypi/dm/confounding-robust-inference.svg?label=Pypi%20downloads)


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
