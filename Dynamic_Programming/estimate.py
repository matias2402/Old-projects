import math
import time
from scipy import interpolate
from scipy import optimize
import pandas as pd
import os
import numpy as np

def read_data(model, ver=1):
    file_directory = os.getcwd() + '\\Data\\' + 'sparadata.xls' 
    data = pd.read_excel(file_directory, skipfooter = 3) # Remove last 3 rows that are NA 
    
    # Clean data for structural estimation. Filter all individuals who retire before the age of 60
    data["invalid"] = ((data["ret"] == 1) & (data["age"] < 60))
    
    # Filter condition
    filter_cond = data[(data['invalid'] == True).groupby(data['id']).transform('any')] 
    
    # Filter
    keys = list(filter_cond.columns.values)
    i1 = data.set_index(keys).index
    i2 = filter_cond.set_index(keys).index
    data = data[~i1.isin(i2)]
    
    # Make data compatible with the solution of the model with all 5 state variables
    if ver == 6:
        data["ret_age"] = data["id"].map(data[data["ret"] == 1].set_index("id")["age"])
        data["ret_age"] = data.groupby(["id"])["age"].max()
        data = data.dropna()

    # Discretise wage and AP from grids and create in data

    w_grid = model.w_grid
    ap_grid = model.ap_grid
    data["w_index"] = np.digitize(data["income"], w_grid)-1
    data["ap_index"] = np.digitize(data["atp"], ap_grid)-1

    # create age index in data
    data["age_index"] = data["age"]-50

    return data


def ll(theta,model,data,pnames):

    # Unpack data
    w = data.w_index
    ap = data.ap_index
    age = data.age_index
    m = data.married
    d = data.ret

    # Update values
    model.theta = theta
    #model=updatepar(model,pnames,theta)

    # print theta to follow along
    print(f'{model.theta[0]:.4f}    {model.theta[1]:.4f}    {model.theta[2]:.4f}    {model.theta[3]:.4f}\
        {model.theta[4]:.4f}')

    # Solve model
    sol = model.bellman()

    # Choice probability for working for each observation
    pr = sol.ccp[age,w,ap,m]

    # Truncate ends of pr to prevent loglik -inf
    pr[pr<10e-5] += 10e-5
    pr[pr>1-10e-5] -= 10e-5

    
    # Calculate the loglik, print and return the mean/sum times minus 1 for minimizing
    loglik = np.log(pr*(1-d) + (1-pr)*d)
    print(np.mean(loglik))
    return np.mean(-loglik)

def estimate(model,data,theta0=[0,0,0,0,0]):
    
    # Names for updating values
    pnames = ['alpha','theta1','theta2','theta3','theta4']
    #theta0 = [model.alpha,model.theta1, model.theta2,model.theta3,model.theta4] Set initial guess to karlstrom values
    res = optimize.minimize(ll,theta0, args = (model,data,pnames), method = 'BFGS', tol = 1e-6)
    
    # Save result
    model=updatepar(model,pnames,res.x)
    theta_hat = res.x
    loglik = ll(theta_hat,model,data,pnames)

    return theta_hat, loglik, model


def updatepar(par,parnames, parvals):
    # Function for updating values in a class(tyv stjålet fra Bertels kode)

    for i,parname in enumerate(parnames):
        if i<2:
            parval = parvals[i]
            setattr(par,parname,parval)
        else:
            list_val = [None]*(parvals.size-2) 
            for j,parval in enumerate(parvals[2:]):
                list_val[j]=parval
            setattr(par,parname,list_val)
    return par