This folder contains the code to replicate Karlstrom et al.(2004) which was created as an exam project

## Dynamic Programming - July 3rd 2022

Zip file for our exam: **What Are The Effects of Sweden’s Pension System on Retirement Behaviour**

### Authors:

**Matias Bjørn Frydensberg Hall (PKT593)** and **Thomas Theodor Kjølbye (XTB358)**

## This folder contains

1. ver5.py The code for solving our model with 4 states.
2. ver6.py The code for solving our model with all 5 states.
3. estimate.py The code for loading data and estimating parameters with log-likelihood.
4. [Results.ipynb](Results.ipynb) Notebook containing the results from our solution and estimation.
5. [Data.ipynb](Data.ipynb) Notebook containing data cleaning and firststep regressions. Also has marital transition calculations and hazard rates for data.

### Data(folder)

1. surv.txt Survival probabilities
2. sparadata.csv Data for estimation

## Packages used

* scipy
* numpy
* pandas
* os
* statsmodels
* matplotlib
* patsy
