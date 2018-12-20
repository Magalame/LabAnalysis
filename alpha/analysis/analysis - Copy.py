# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 09:51:27 2018

@author: noudy
"""

from chn import Chn
import matplotlib.pyplot as plt
import numpy as np
import spinmob as s
import os
import scipy.integrate as integrate

DATAPATH = r"C:\Users\noudy\Documents\Cours\PHYS439\Exp2\data"
DPCAL = DATAPATH+r"\calib"
DPHAF = DATAPATH+r"\halflife"
DPHAF2 = DATAPATH+r"\halflife2"

def loadRuns(path):
    spectr = []
    live = []
    counter = 1
    for file in os.listdir(path):
        
        chnloaded = Chn(path+"\\"+file)
        
        spectr.append(chnloaded.spectrum)
        live.append(chnloaded.live_time*counter)
        
        counter += 1
        
            
    return np.asarray(live), np.asarray(spectr)

def getTotSpectr(data):
    spectr = data[1]
    return np.array(spectr).sum(axis=0)
    
def getActivity(data):
    (live, spectr) = data
    spectr = spectr[:,274:]
    return live, np.array(spectr).sum(axis=1)

def activity(t,l1,l2,N0,N2):
    return (N0*l1*l2*(np.exp(-l1*(t))-np.exp(-l2*(t))))/(l2-l1)+ l2*N2*np.exp(-l2*t)

def halflife():
    data = loadRuns(DPHAF2)
    live,act = getActivity(data)
    
    fit = s.data.fitter()
    fit.set_data(xdata=live, ydata=act)
    fit.set_functions(f=activity,p='l1=0.000023,l2=0.00037,N0=173000000,N2')
    fit.set(plot_guess=False)
    fit.fit()
    
    print(fit)
    
def totSpectr():
    live,spectr = loadRuns(DPHAF2)
    totspectr = getTotSpectr((live,spectr))[274:2015]
    
    plt.plot(totspectr)
    
    fit = s.data.fitter()
    fit.set_data(xdata=range(274,len(totspectr)+274), ydata=totspectr,eydata=0.01)
    fit.set_functions('a1*exp(-(x-x1)**2/w1**2)+a2*exp(-(x-x2)**2/w2**2)+a3*exp(-(x-x3)**2/w3**2)+a4*exp(-(x-x4)**2/w4**2)+a5*exp(-(x-x5)**2/w5**2)+b', 
                'a1=14615, b=0, x1=1950, w1=3, x2=1335, w2=4., a2=6114,a3=2444,x3=1345,w3=4.54,a4=169,x4=1272,w4=4,a5=128,x5=1236,w5=4')
    

    
    fit.set(
            plot_guess=False,
            xlabel='Channel number',
            ylabel='Number of count')
    
    fit.fit()