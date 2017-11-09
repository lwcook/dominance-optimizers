Integrated testing with Travis-CI: [![Build Status](https://travis-ci.org/lwcook/dominance-optimizers.svg?branch=master)](https://travis-ci.org/lwcook/dominance-optimizers.svg?branch=master)

Multi-objective optimization algorithms modified to use multiple dominance criteria

This python module implements versions of a genetic algorithm and a tabu search that implement different dominance criteria.
This is achieved by defining an object that overrides the less than comparison operator to use in the optimizations.

**Documenation**
Documentation (built using sphinx), is available at https://www-edc.cam.ac.uk/aerotools/dominanceoptimizers/documentation

In the examples folder are scripts to recreate results from the publication:
Cook, L. W. and Jarrett, J. P. "Using Stochastic Dominance in Multi-Objective Optimizers for Aerospace Design Under Uncertainty".

**Installation Instructions**

1) Download this repository as a .zip file and extract the contents, clone this repository, or dowload this repository as a .tar.gz file from https://github.com/lwcook/dominance-optimizers/archive/0.4.tar.gz and extract the contents.

2) Move to the location of the package, and from the root directory (i.e. PATH/TO/EXTRACT/dominance-optimizers-x.y), install the package with "python setup.py install".
Please ensure you have the appropriate permissions for your machine (e.g. by using "sudo" in linux based systems).

**Rquirements**

The package is built for python 2.7 and requires the numpy module (>=1.12.1)
