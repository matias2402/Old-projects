# Import relevant packages
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# Utility function
def u_func(h, c, par):
    """
    Cobb-Douglas utility function for consumption and housing quality

    Args:

        h (float): housing quality and equal to housing price
        c (float): other consumption
        par: simplenamespace containing relevant parameters
            phi (float): C-D weights
            epsilon (float): public housing assement factor
            r (float): mortgage interest
            tau_g (float): base housing tax
            tau_p (float): progressive housing tax 
            p_bar (float): cutoff price
            m (float): cash-on-hand
    Returns:
    
        (float): utility
    """
    return c**(1-par.phi)*h**par.phi

# Optimize function
def u_optimize(par):
    """
    Optimises u_func with respect to housing quality and finds housing quality and consumption at the optimum

     Args:

        h (float): housing quality and equal to housing price
        par: simplenamespace containing relevant parameters
            phi (float): C-D weights
            epsilon (float): public housing assement factor
            r (float): mortgage interest
            tau_g (float): base housing tax
            tau_p (float): progressive housing tax 
            p_bar (float): cutoff price
            m (float): cash-on-hand

    Local variables:

        p_thilde (float): public housing assement price
        tax (float): interest rates and tax paid as a function of housing quality
        c (float): other consumption
    Returns:
    
        h_star (float): optimal housing quality
        c_star (float): optimal consumption
        u_star (float): utility in optimum
    """
    def objective(h, par):
        # Use monotonicity to find c as a function of h
        p_thilde = h * par.epsilon
        tax = par.r * h + par.tau_g * p_thilde + par.tau_p * max(p_thilde-par.p_bar, 0)
        c = par.m - tax
        return -u_func(h, c, par)
    
    res = optimize.minimize_scalar(objective, method ='brent', args = (par))

    # Get optimal h, using monotonicity to find optimal c, then using u_func to find utility in optimum
    h_star = res.x
    p_thilde = h_star * par.epsilon
    tax = par.r * h_star + par.tau_g * p_thilde + par.tau_p * max(p_thilde-par.p_bar, 0)
    c_star = par.m - tax
    u_star = u_func(h_star, c_star, par)
    return h_star, c_star, u_star

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

# Tax revenue function
def tax_total(par):
    """ 
    Finds total tax burden in a log normal distributed population
    
    Args:
       
        par: simplenamespace containing relevant parameters
            phi (float): C-D weights
            epsilon (float): public housing assement factor
            r (float): mortgage interest
            tau_g (float): base housing tax
            tau_p (float): progressive housing tax 
            p_bar (float): cutoff price
            m (float): cash-on-hand
            seed (int): seed number for random draws
            mu (float): mean value for the distribution
            sigma (float): standard deviation for the distribution

    Local variables:

        h_cit (float): housing quality choice of one citizen in the population
        c_cit (float): other consumption choice of one citizen in the population
        u_cit (float): utility for one citizen in the population given chice of h and c

    Returns:
    
        T (float): total tax burden
    """
    # Set seed and tax = 0
    np.random.seed(par.seed)
    T = 0
    # Loop through every citizen in the population and calculate optimal choices
    # and tax given those choices
    for i in range(par.pop):
        par.m = np.random.lognormal(par.mu, par.sigma)
        h_cit, c_cit, u_cit = u_optimize(par)
        T += par.tau_g*(h_cit*par.epsilon) + par.tau_p*max((par.epsilon*h_cit)-par.p_bar, 0)
    return T

# Base tax percentage function
def base_tax_pct(par):
    """ 
    Finds optimal base tax percentage for tax reform given the tax burden before the reform
    Uses root optimisation
    
    Args:

        tau_g (float): base tax level
        par: simplenamespace containing relevant parameters
            phi (float): C-D weights
            epsilon (float): public housing assement factor
            r (float): mortgage interest
            tau_g (float): base housing tax
            tau_p (float): progressive housing tax 
            p_bar (float): cutoff price
            m (float): cash-on-hand
            seed (int): seed number for random draws
            mu (float): mean value for the distribution
            sigma (float): standard deviation for the distribution
            T_goal (float): level of tax burden the policy maker wants to hit

    Returns:
    
        tau (float): base tax percentage
    """
    def obj(tau_g, par):
        par.tau_g = tau_g
        return tax_total(par) - par.T_goal
    sol = optimize.root(obj, 0.01, args=(par))
    tau = float(sol.x)
    return tau