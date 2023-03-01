import numpy as np
import os
import pandas as pd
import math
import time
from scipy import interpolate

def gauss_hermite(n):

    # a. calculations
    i = np.arange(1,n)
    a = np.sqrt(i/2)
    CM = np.diag(a,1) + np.diag(a,-1)
    L,V = np.linalg.eig(CM)
    I = L.argsort()
    V = V[:,I].T

    # b. nodes and weights
    x = L[I]
    w = np.sqrt(math.pi)*V[:,0]**2

    return x,w


class karlstrom():
    
    "Class for solving the model in karlstrom et al. 2004"
    
    def __init__(self,**kwargs): # Init konstruerer data til vores instance. Så vores instance tager **kwargs argument, når vi laver det
        self.setup(**kwargs)
        
    def setup(self, **kwargs):
        
        # Space Parameters
        self.t_min = 50
        self.t_max = 102
        self.t_size = self.t_max - self.t_min
        self.w_min = 1
        self.w_max = 1000
        self.w_n = 50
        self.ap_min = 1e-8
        self.ap_max = 6.5
        self.ap_n = 10
        self.ra_min = 60
        self.ra_max = 70
        

        # Other paramters
        self.f_work = 0.55
        self.f_retire = 1
        self.num_shocks = 5
        self.forced_retire = 70
        self.forced_work = 60

        # Pension parameters
        self.BA = 38600/1000
        self.unmarried = 0.96
        self.married = 0.785
        self.m_bonus_pct = np.array([self.unmarried,self.married])
        self.supplement_pct = 0.555
        self.age_punish = (0.005*12)
        self.age_reward = (0.007*12)
        self.um = [1-0.0136, 0.0136]
        self.m = [0.0126,1-0.0126]
        self.married_trans = np.array([self.um,self.m])

        # Structural Parameters
        self.beta = 0.97

        self.theta = [0.90,0.85,0.40,65.4,0.007]
        self.alpha = 0.90
        self.theta1 = 0.85
        self.theta2 = 0.40
        self.theta3 = 65.4
        self.theta4 = 0.007

        #self.p65 = 0.97
        
        # First step parameters
        self.alpha1 = 0.2376 #1.0386
        self.alpha2 = 0.8868
        self.alpha3 = 0.0133
        self.alpha4 = -0.0001
        self.sigma_zeta = 0.0269 #0.0429  # 
        self.gamma1 = -0.0428
        self.gamma2 = 0.9733
        self.gamma3 = 0.0040
        self.gamma4 = -4.162e-5
        self.sigma_eta = 0.0003

        
        # Update baseline parameters using keywords
        for key,val in kwargs.items(): 
            setattr(self,key,val) 
            
        # Create grid
        self.create_grid()
        
        # Read survival data
        self.read_surv()

        # Create shocks
        self.quad()
       
    def read_surv(self):
        file_directory = os.getcwd() + '\\Data\\' + 'surv.xls' 
        data = pd.read_excel(file_directory, skipfooter = 7, skiprows = range(1,21) ) # Remove ages above 101 and below 50
        self.surv = data["Conditional survival probability"]

    def create_grid(self):
        # Discretise states into grids
        self.w_grid = np.linspace(self.w_min,self.w_max,self.w_n) # n gridpoints for w fra 1 til 1000
        self.t_grid = np.arange(self.t_min,self.t_max+1) # Age grid 50 - 102
        self.ra_grid = np.arange(self.ra_min,self.ra_max+1) # Retirement age grid (60-70)
        self.m_grid = np.array([0,1])
        self.ap_grid = np.linspace(self.ap_min,self.ap_max,self.ap_n)
        #self.compute_ap()
    
    #def compute_ap(self):
        #ap_grid = np.linspace(self.ap_min,self.ap_max,self.ap_n) # Grid over AP
        #self.ap_grid = np.zeros([self.ap_n, np.size(self.t_grid)])
        #self.ap_grid[:,0] = ap_grid
        #for it in range(1, np.size(self.t_grid)-1):
        #    t = t = self.t_grid[it]
        #    self.ap_grid[:,it] = np.maximum(1e-8,np.minimum(np.exp(self.gamma1 + self.gamma2*np.log(self.ap_grid[:,it-1]) + self.gamma3*t + 
        #                        self.gamma4*t*t + self.sigma_eta*0.5), self.ap_max))

    def quad(self):
        # Calculate wage shocks and probabilities
        x,w = gauss_hermite(self.num_shocks)
        self.zeta = np.exp(self.sigma_zeta*np.sqrt(2)*x)
        self.zeta_w = w/np.sqrt(np.pi)
   
    def utility_work(self,w,t):
        # Python can't handle theta for high t's but it goes to one as t increases
        theta = np.exp(self.theta[1]) + np.exp(self.theta[2])*\
                ((np.exp((t-self.theta[3])/self.theta[4]))/(1+np.exp((t-self.theta[3])/self.theta[4])))
        if np.isnan(theta) == True:
            theta = 1
        u_work = self.theta[0] * np.log(w) + theta * np.log(self.f_work)
        return u_work
    
    def utility_retire(self,b,t):
        # Python can't handle theta for high t's but it goes to one as t increases
        theta = np.exp(self.theta[1]) + np.exp(self.theta[2])*\
                ((np.exp((t-self.theta[3])/self.theta[4]))/(1+np.exp((t-self.theta[3])/self.theta[4])))
        if np.isnan(theta) == True:
            theta = 1
        u_retire = self.theta[0] * np.log(b) + theta * np.log(self.f_retire)
        return u_retire
    
    def pension(self,ap,ra=65,m=1):
        # Compute benefit
        if ra < 65:
            bonus = self.age_punish*(ra-65)
        elif ra > 65:
            bonus = self.age_reward*(ra-65)
        else:
            bonus = 0
        ba = self.BA*(self.m_bonus_pct[m])*(1+bonus)
        supplement = ba*self.supplement_pct
        atp = 0.6*ap*ba
        b = np.maximum(ba + supplement, ba + atp)
        return b
    
    def bellman(self):
        # Create time instance
        t0 = time.time()

        # Create class for solution grids
        class sol: pass
        sol.vr = np.zeros([np.size(self.t_grid),self.ap_n, np.size(self.m_grid)])
        sol.vw = np.zeros([np.size(self.t_grid),self.w_n,self.ap_n, np.size(self.m_grid)])
        sol.ccp = np.zeros([np.size(self.t_grid),self.w_n,self.ap_n,np.size(self.m_grid)])
        sol.grid_W = np.zeros([self.w_n, np.size(self.t_grid)])
        sol.grid_ap = np.linspace(self.ap_min,self.ap_max,self.ap_n)

        # Loop over periods
        for it in range(np.size(self.t_grid)-1,-1,-1): # from period 52 (column 51) until period 0, backwards
            t = self.t_grid[it]
            # Create max grid over wage
            #W_max = np.exp(self.alpha1 + self.alpha2*np.log(self.w_max) +
            #        self.alpha3*t + self.alpha4*t*t + t*max(self.zeta))
            grid_W = np.linspace(self.w_min, self.w_max, self.w_n)
            sol.grid_W[:, it] = grid_W

            # Loop over AP
            for iap, ap in enumerate(sol.grid_ap):

                    
                # Compute benefits for both marital status
                b_um = self.pension(ap=ap, m=0)
                b_m = self.pension(ap=ap, m=1)
                
                # Last period
                if t == 102:

                    # Compute value of retiring for both marital status in last period
                    u_retire_um = self.utility_retire(t=self.t_max,b=b_um)
                    u_retire_m = self.utility_retire(t=self.t_max,b=b_m)

                    sol.vr[np.size(self.t_grid)-1,iap,0] = u_retire_um
                    sol.vr[np.size(self.t_grid)-1,iap,1] = u_retire_m
                    sol.vw[np.size(self.t_grid)-1,:,iap,:] = 0
                    continue
            
                # Forced retirement at age 70
                if t >= self.forced_retire:
                    # Compute value of retiring
                    u_retire_um = self.utility_retire(t=t,b=b_um)
                    u_retire_m = self.utility_retire(t=t,b=b_m)
                    # EV for retire for both marrital statuses over 70
                    sol.vr[it,iap,0] = u_retire_um + self.surv[it]*self.beta*(self.married_trans[0,0]
                        *np.log(0 + np.exp(sol.vr[it+1,iap,0]))+self.married_trans[0,1]
                        *np.log(0 + np.exp(sol.vr[it+1,iap,1])))
                    sol.vr[it,iap,1] = u_retire_m + self.surv[it]*self.beta*(self.married_trans[1,0]
                        *np.log(0 + np.exp(sol.vr[it+1,iap,0]))+self.married_trans[1,1]
                        *np.log(0 + np.exp(sol.vr[it+1,iap,1])))

                    # EV for work for both marrital statuses over 70
                    sol.vw[it,:,iap,0] = 0 + self.surv[it]*self.beta*(self.married_trans[0,0]*
                        np.log(np.exp(sol.vw[it+1,:,iap,0])+np.exp(sol.vr[it+1,iap,0]))+
                        self.married_trans[0,1]*np.log(np.exp(sol.vw[it+1,:,iap,1])+np.exp(sol.vr[it+1,iap,1])))
                    sol.vw[it,:,iap,1] = 0 + self.surv[it]*self.beta*(self.married_trans[1,0]*
                        np.log(np.exp(sol.vw[it+1,:,iap,0])+np.exp(sol.vr[it+1,iap,0]))+
                        self.married_trans[1,1]*np.log(np.exp(sol.vw[it+1,:,iap,1])+np.exp(sol.vr[it+1,iap,1])))
                    continue
                
                # Cannot retire before age 60
                if t < self.forced_work:
                    # Retirement is an absorbing state
                    sol.vr[it,iap,0] = 0 + self.surv[it]*self.beta*(self.married_trans[0,0]
                        *np.log(0 + np.exp(sol.vr[it+1,iap,0]))+self.married_trans[0,1]
                        *np.log(0 + np.exp(sol.vr[it+1,iap,1])))
                    sol.vr[it,iap,1] = 0 + self.surv[it]*self.beta*(self.married_trans[1,0]
                        *np.log(0 + np.exp(sol.vr[it+1,iap,0]))+self.married_trans[1,1]
                        *np.log(0 + np.exp(sol.vr[it+1,iap,1])))

                # When age allows for both retiring and working
                if self.forced_work < t and t < self.forced_retire:
                    u_retire_um = self.utility_retire(t=t,b=b_um)
                    u_retire_m = self.utility_retire(t=t,b=b_m)
                    # Retirement is an absorbing state
                    sol.vr[it,iap,0] = u_retire_um + self.surv[it]*self.beta*(self.married_trans[0,0]
                            *np.log(0 + np.exp(sol.vr[it+1,iap,0]))+self.married_trans[0,1]
                            *np.log(0 + np.exp(sol.vr[it+1,iap,1])))
                    sol.vr[it,iap,1] = u_retire_m + self.surv[it]*self.beta*(self.married_trans[1,0]
                            *np.log(0 + np.exp(sol.vr[it+1,iap,0]))+self.married_trans[1,1]
                            *np.log(0 + np.exp(sol.vr[it+1,iap,1])))


            # Compute ap next period outside wage loop
                ap_next = np.maximum(1e-8,np.minimum(np.exp(self.gamma1 + self.gamma2*np.log(ap) + self.gamma3*t + 
                            self.gamma4*t*t + self.sigma_eta*0.5), self.ap_max))
                # Loop over wages to solve for working
                for iw, w in enumerate(grid_W):
                    EV_next_work_um = 0
                    EV_next_work_m = 0
                    logw_noshock = self.alpha1 + self.alpha2*np.log(w) + \
                                self.alpha3*t + self.alpha4*t*t
                    # Given that we work EV of retiring changes
                    EV_next_retire_if_work_um = np.interp(ap_next, sol.grid_ap[:],sol.vr[it+1,:,0])
                    EV_next_retire_if_work_m = np.interp(ap_next, sol.grid_ap[:],sol.vr[it+1,:,1])
                    # Create function taking ap and w and interpolates to vw
                    f_um = interpolate.interp2d(sol.grid_ap[:],sol.grid_W[:, it+1],sol.vw[it+1,:,:,0])
                    f_m = interpolate.interp2d(sol.grid_ap[:],sol.grid_W[:, it+1],sol.vw[it+1,:,:,1])
                    for s in range(self.num_shocks):
                        #Loop over shocks
                        w_next = np.exp(logw_noshock + self.zeta[s])
                        EV_next_work_um += self.zeta_w[s] * f_um(ap_next,w_next)
                        EV_next_work_m += self.zeta_w[s] * f_m(ap_next,w_next)
                
                    # Calculate logsum 
                    logsum_um = np.log(np.exp(EV_next_retire_if_work_um) + np.exp(EV_next_work_um))
                    logsum_m = np.log(np.exp(EV_next_retire_if_work_m) + np.exp(EV_next_work_m))
                    
                    #Update values
                    sol.vw[it,iw,iap,0] = self.utility_work(w=w, t=t) + self.surv[it]*self.beta*(
                        self.married_trans[0,0]*logsum_um + self.married_trans[0,1]*logsum_m)
                    sol.vw[it,iw,iap,1] = self.utility_work(w=w, t=t) + self.surv[it]*self.beta*(
                        self.married_trans[1,0]*logsum_um + self.married_trans[1,1]*logsum_m)
                    # Conditional Choice Prob.
                    sol.ccp[it,iw,iap,0] = np.exp(sol.vw[it,iw,iap,0])/(np.exp(sol.vw[it,iw,iap,0])\
                                        + np.exp(sol.vr[it,iap,0]))
                    sol.ccp[it,iw,iap,1] = np.exp(sol.vw[it,iw,iap,1])/(np.exp(sol.vw[it,iw,iap,1])\
                                        + np.exp(sol.vr[it,iap,1]))
        t1 = time.time()
        elapsed = t1-t0
        print(f'model solved in {elapsed} seconds')
        return sol