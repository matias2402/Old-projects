#Import relevant packages
from scipy import optimize
import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt

# Utility function
def inter_utility(C_1, par):
    """
    Intertemporal consumer utility function in two periods

    Args:

        C_1 (float): consumption in period 1
        par: simplenamespace containing relevant parameters
            T_1 (float): lump-sum tax in period 1
            T_2 (float): lump-sum tax in period 2
            Y_L1 (float): labour income in period 1
            Y_L2 (float): labour income in period 2
            V_1 (float): initial endowment
            phi (float): degree of impatience
            r (float): rental rate

    Returns:
    
        (float): total utility
    """
    return np.log(C_1) + np.log((1+par.r)*(par.V_1 + par.Y_L1 - par.T_1 - C_1) + par.Y_L2 - par.T_2)/(1+par.phi)

# Utility optimise function
def u_optimise(par):
    """
    Optimises utility using the inter_utility function

     Args:

     C_1 (float): consumption in period 1
        par: simplenamespace containing relevant parameters
            T_1 (float): lump-sum tax in period 1
            T_2 (float): lump-sum tax in period 2
            Y_L1 (float): labour income in period 1
            Y_L2 (float): labour income in period 2
            V_1 (float): initial endowment
            phi (float): degree of impatience
            r (float): rental rate

    Returns:
    
        C_1star (float): optimal consumption in period 1
        C_2star (float): optimal consumption in period 2
        U_star (float): utility in optimum
    """
    def objective(C_1, par):
        return -inter_utility(C_1, par)
    
    #Creating bounds for optimization
    lower = 0
    upper = par.V_1 + par.Y_L1 - par.T_1 + (par.Y_L2 - par.T_2)/(1 + par.r)

    #Running the optimization function
    res = optimize.minimize_scalar(objective, method ='bounded', bounds = (lower,upper), args = (par))

    # Get optimal C_1, using monotonicity to find optimal C_2, then using u_func to find utility in optimum
    C_1star = res.x
    C_2star = (1+par.r)*(par.V_1 + par.Y_L1 - par.T_1 - C_1star) + par.Y_L2 - par.T_2
    U_star = np.log(C_1star) + (np.log(C_2star)/(1+par.phi))
    return C_1star, C_2star, U_star


#Array/container function
def array(a, b, N):
    """
    Creates array using linspace and 3 empty containers

     Args:

     a (float): starting value in linspace
     b (float): end value in linspace
     N (integer): number of objects in array and containers

    Returns:
    
        array1 (numpy.ndarray): array containing N objects with values between a and b
        container1 (numpy.ndarray): empty container
        container2 (numpy.ndarray): empty container
        container3 (numpy.ndarray): empty container
    """
    array1 = np.linspace(a, b, N)
    container1 = np.zeros(N)
    container2 = np.zeros(N)
    container3 = np.zeros(N)

    return array1, container1, container2, container3


#Create and optimise over an array function(doesn't work)
#def o_array(a, b, N, ar_var, par):
    """
    Creates array using linspace and 3 empty containers then optimises the 
    intertemporal utility function looping over the array

     Args:

     a (float): starting value in linspace
     b (float): end value in linspace
     N (integer): number of objects in array and containers

    Returns:
    
        array1 (numpy.ndarray): array containing N objects with values between a and b
        container1 (numpy.ndarray): empty container
        container2 (numpy.ndarray): empty container
        container3 (numpy.ndarray): empty container
    """
    array1 = np.linspace(a, b, N)
    container1 = np.zeros(N)
    container2 = np.zeros(N)
    container3 = np.zeros(N)

    par2 = copy.copy(par)

    for i in range(N):
        par2.ar_var = array1[i]
        container1[i], container2[i], container3[i] = u_optimise(par2)

    return container1, container2, container3, array1



# Plot function
def two_figures(x_left, y_left, title_left, xlabel_left, ylabel_left, x_right, y_right, title_right, xlabel_right, ylabel_right, grid=True):
    """ 
    Plots two aligned figures. 
    
    Args: should be self explanatory...

    Returns: Two figures in 2D
    """
    # a. initialise figure
    fig = plt.figure(figsize=(10,4))# figsize is in inches...

    # b. left plot
    ax_left = fig.add_subplot(1,2,1)
    ax_left.plot(x_left,y_left)

    ax_left.set_title(title_left)
    ax_left.set_xlabel(xlabel_left)
    ax_left.set_ylabel(ylabel_left)
    ax_left.grid(grid)

    # c. right plot
    ax_right = fig.add_subplot(1,2,2)

    ax_right.plot(x_right, y_right)

    ax_right.set_title(title_right)
    ax_right.set_xlabel(xlabel_right)
    ax_right.set_ylabel(ylabel_right)
    ax_right.grid(grid)

def one_figure(x, y, title, xlabel, ylabel, grid=True):
    """ 
    Plots one figure. 
    
    Args: should be self explanatory...

    Returns: One figures in 2D
    """
    # a. initialise figure
    fig = plt.figure(figsize=(10,4))# figsize is in inches...

    # b. create plot
    ax = fig.add_subplot(1,2,1)
    ax.plot(x,y)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(grid)

# Functions determining consumption in period two depending on the state
def c2_good(C_1, alpha, par):
    return alpha*(1+par.r_rf)*(par.V_1 + par.Y_L1 - par.T_1 - C_1) + (1-alpha)*(1+par.r_rb)*(par.V_1 + par.Y_L1 - par.T_1 - C_1) + par.Y_L2 - par.T_2
def c2_bad(C_1,alpha, par):
    return alpha*(1+par.r_rf)*(par.V_1 + par.Y_L1 - par.T_1 - C_1)+ par.Y_L2 - par.T_2


# Utility function with risk
def inter_utility_ext(C_1, alpha, par):
    """
    Intertemporal consumer utility function in two periods now including risk
    
    Args:
        C_1 (float): consumption in period 1
        par: simplenamespace containing relevant parameters
            T_1 (float): lump-sum tax in period 1
            T_2 (float): lump-sum tax in period 2
            Y_L1 (float): labour income in period 1
            Y_L2 (float): labour income in period 2
            V_1 (float): initial endowment
            phi (float): degree of impatience
            r_rf (float): risk free rate
            r_rb (float): risk bearing rate
            alpha (float): the percentage of savings going towards the risk free asset
            beta (float): the chance of being in a good state
    Returns:
    
        (float): total utility
    """

    # Calculate the good and bad states 
    u2_good = np.log(c2_good(C_1, alpha,par))
    u2_bad = np.log(c2_bad(C_1, alpha,par))

    return np.log(C_1) + (par.beta * u2_good + (1-par.beta)*u2_bad)/(1+par.phi)

#Utility optimse function with risk
def u_optimise_ext(par):
    """
    Optimises utility using the inter_utility_ext function-

     Args:
     C_1 (float): consumption in period 1
        par: simplenamespace containing relevant parameters
            T_1 (float): lump-sum tax in period 1
            T_2 (float): lump-sum tax in period 2
            Y_L1 (float): labour income in period 1
            Y_L2 (float): labour income in period 2
            V_1 (float): initial endowment
            phi (float): degree of impatience
            r_rf (float): risk free rate
            r_rb (float): risk bearing rate
            alpha (float): the percentage of savings going towards the risk free asset
            beta (float): the chance of being in a good state
    Returns:
    
        C_1star (float): optimal consumption in period 1
        C_2star_good (float): consumption in period 2 in the good state when optimising for both states
        C_2star_bad (float): consumption in period 2 in the bad state when optimising for both states
        alpha_star (float): optimal percentage of savings going towards the risk free asset
        U_star (float): utility in optimum
    """
    def objective(x,par):
        return -inter_utility_ext(x[0],x[1], par)
    
    #Creating bounds for optimization
    lower = 0
    upper = min(par.V_1 + par.Y_L1 - par.T_1 + (par.Y_L2 - par.T_2)/(par.alpha*(1+par.r_rf)+(1-par.alpha)*(1+par.r_rb)), \
                par.V_1 + par.Y_L1 - par.T_1 + (par.Y_L2 - par.T_2)/(par.alpha*(1+par.r_rf)))

    #Running the optimization function
    res = optimize.minimize(objective,x0 =[1,par.alpha], bounds = ((lower,upper),(0,1)), args = (par))

    # Get optimal C_1, using monotonicity to find optimal C_2, then using u_func to find utility in optimum
    C_1star, alpha_star = res.x
    C_2star_good = c2_good(C_1star, alpha_star,par)
    C_2star_bad = c2_bad(C_1star, alpha_star,par)
    U_star = np.log(C_1star) + (par.beta*np.log(C_2star_good)+(1-par.beta)*np.log(C_2star_bad) ) /(1+par.phi)
    return C_1star, C_2star_good, C_2star_bad, alpha_star, U_star




