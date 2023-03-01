from scipy.stats import norm
from sklearn.linear_model import Lasso
import numpy as np

def BRT(X_tilde,y):
    (N,p)=X_tilde.shape
    sigma = np.std(y)
    c = 1.1
    alpha = 0.05

    penalty_BRT= (sigma * c)/np.sqrt(N)*norm.ppf(1-alpha/(2*p)) # on normalised data since sum of squares is =1, 
    #NB div by 2 because of python definition of 

    return penalty_BRT

def BCCH(X_tilde,y):
    (N,p)=X_tilde.shape
    sigma = np.std(y)
    c = 1.1
    alpha = 0.05

    yXscale = (np.max((X_tilde.T ** 2) @ ((y-np.mean(y)) ** 2) / N)) ** 0.5
    lambda_pilot = c*norm.ppf(1-alpha/(2*p))*yXscale/np.sqrt(N)

    # Pilot estimates
    coef_pilot = Lasso(alpha=lambda_pilot).fit(X_tilde,y).coef_
    coef_intercept = Lasso(alpha=lambda_pilot).fit(X_tilde,y).intercept_
    pred = (coef_intercept + X_tilde@coef_pilot)
    pred = Lasso(alpha=lambda_pilot).fit(X_tilde,y).predict(X_tilde)

    # Updated penalty
    res = y - pred
    resXscale = (np.max((X_tilde.T ** 2) @ (res ** 2) / N)) ** 0.5
    lambda_bcch = c*norm.ppf(1-alpha/(2*p))*resXscale/np.sqrt(N)

    penalty_BCCH= c*norm.ppf(1-alpha/(2*p))*resXscale/np.sqrt(N)

    return penalty_BCCH

def standardize(X):
    X_mean = np.mean(X,axis=0)
    X_std = np.std(X,axis=0)
    X_stan=(X-X_mean)/X_std
    return X_stan

def PDL_ols(resdz,resyxz,d):
    denom = np.sum(resdz*d)
    num = np.sum(resdz*resyxz)
    return num/denom

def PDL_CI(resdz,resyzz,PDL):
    # Variance

    N = resyzz.shape[0]
    num = np.sum(resdz**2*resyzz**2)/N
    denom = (np.sum(resdz**2)/N)**2
    sigma2_PDL = num/denom

    # Confidence interval
    q=norm.ppf(1-0.025)
    se_PDL = np.sqrt(sigma2_PDL)
    CI_PDL=(((PDL-q*se_PDL/np.sqrt(N)).round(2),(PDL+q*se_PDL/np.sqrt(N)).round(2)))
    
    return se_PDL, CI_PDL