### Fair Max-Entropy Distributions

This repository contains code for constructing and evaluating unbiased maximum-entropy distributions from biased datasets.

The goal of constructing such distributions is to debias data from real-world sources. 
In this repository, examples of evaluation with respect to Compas recidivism data and Adult income data are given.
The package requirements are provided in the requirements.txt file.

### Examples

The files FairMaxEnt-expts-1.ipynb and FairMaxEnt-expts-2.ipynb provide a detailed examples for the usage of the max-entropy framework to debias real-world datasets.

The evaluation with respect to Compas dataset is divided into two parts.
- One for small version of Compas dataset (with 144 elements in domain)
- One for large version of Compas dataset (with 1.4 x 10^11 elements in domain)

The file FairMaxEnt-expts-1.ipynb provides evaluation code for the first and the file FairMaxEnt-expts-2.ipynb provides evaluation code for the second.

For Adult dataset (504 elements in domain), just replace the dataset and domain loading function in FairMaxEnt-expts-1.ipynb file.

The folder *FairMaxEnt* contains the backend code for constructing max-entropy distributions for any given set of parameters, and can be used for more general applications as well.

### References

*Data preprocessing to mitigate bias: A maximum entropy based approach* <br>
L. Elisa Celis, Vijay Keswani, Nisheeth K. Vishnoi <br>
International Conference on Machine Learning, ICML 2020

**Please cite the corresponding paper when using the code**
