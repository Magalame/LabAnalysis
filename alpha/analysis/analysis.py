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
DPHAF = DATAPATH+r"\halflife" #1h
DPHAF2 = DATAPATH+r"\halflife2" #20m
DPHAF3 = DATAPATH+r"\halflife3" #5h

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
    #1 => 24
    #2 => 12
    #3 => 38
    data = loadRuns(DPHAF)
    live,act = getActivity(data)
    
    fit = s.data.fitter()
    fit.set_data(xdata=live, ydata=act,eydata=24)
    fit.set_functions(f=activity,p='l1=0.000023,l2=0.00037,N0=173000000,N2')
    fit.set(plot_guess=False,
            xlabel="Time (s)",
            ylabel="Number of count")
    
    fit.fit()
    
    #s.plot.tweaks.ubertidy(keep_axis_labels=True)
    s.plot.tweaks.trim()
    s.plot.tweaks.legend()
    print(fit)
    
from numpy import pi, exp, real
from scipy.special import wofz, erf
ROOT2 = 2.0**0.5 # Code speedup
 
def erfcx(x):
    """
    Scaled complementary error function.
    """
    return exp(x**2)*(1-erf(x))
 
def em_gaussian(x, a, sigma, tau, x0):
    """
    Returns an exponentially modified Gaussian (a convolution of an exponential
    cutoff at x=0 and Gaussian) having standard deviation sigma and exponential 
    decay length tau. This function is normalized to have unity area.
     
    Parameters
    ----------
    x:
        Distance from the center of the peak.
    sigma:
        Standard deviation of Gaussian ~ exp(-x**2/(2*sigma**2))
    tau:
        Length scale of exponential ~ exp(x/tau). Positive tau skews the peak 
        to higher values and negative tau skews to lower values.
    """
    t = abs(tau)
    s = sigma
    
#    try:
#        assert a/t*exp(-0.5*( (x-x0)/s)**2)*erfcx((s/t - (x-x0)/s)*0.5**0.5) != np.nan
#    except:
#        print(x, a, sigma, tau, x0)
     
    if tau >= 0: 
        res = a/t*exp(-0.5*( (x-x0)/s)**2)*erfcx((s/t - (x-x0)/s)*0.5**0.5)
    else:        
        res = a/t*exp(-0.5*(-(x-x0)/s)**2)*erfcx((s/t + (x-x0)/s)*0.5**0.5)
    

    
    res[res == np.inf] = 0


    
    return np.nan_to_num(res)
    
#def fitfunc(x,a1,s1,t1,x1,a2,s2,t2,x2,a3,s3,t3,x3,a4,s4,t4,x4,a5,s5,t5,x5,b):
#    
#    return em_gaussian(x,a1,s1,t1,x1)+em_gaussian(x,a2,s2,t2,x2)+em_gaussian(x,a3,s3,t3,x3)+em_gaussian(x,a4,s4,t4,x4)+em_gaussian(x,a5,s5,t5,x5)+b

def gaussian(x,a,sigma,x0):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def fitfunc(x,a1,s1,t1,x1,a2,s2,t2,x2,a3,s3,t3,x3,a4,s4,t4,x4,a5,s5,t5,x5,b):
    
    
    return em_gaussian(x,a1,s1,t1,x1)+em_gaussian(x,a2,s2,t2,x2)+em_gaussian(x,a3,s3,t3,x3)+em_gaussian(x,a4,s4,t4,x4)+em_gaussian(x,a5,s5,t5,x5)+b

def E(n):
    
    return 1e3*(n+20.8)/222.9
    

#FIT RESULTS (reduced chi^2 = 30.04 +/- 0.24, 36 DOF)
#  a5         = 78140.0 +/- 110.0
#  s5         = 8.352 +/- 0.03
#  t5         = -11.61 +/- 0.067
#  x5         = 8845.291 +/- 0.035

def totSpectr():
    live,spectr = loadRuns(DPHAF)
    debut = 1200
    fin = 2000
    totspectr = getTotSpectr((live,spectr))[debut:fin]
#    [274:2015]
#    plt.plot(totspectr)
    err = totspectr*12
    
    x = E(np.asarray(range(debut,len(totspectr)+debut)))
    
    fit = s.data.fitter()
    fit.set_data(xdata=x, ydata=totspectr,eydata=24)
#    fit.set_functions(f=fitfunc, p='a1=33300,s1=8.744,t1=-12.23,x1=6093.152,a2=12300,s2=10.639,t2=-2.154,x2=6124.87,a3=592,s3=11,t3=-3.2,x3=5801.8,a4=280,s4=9.9,t4=-1.51,x4=5633.6,a5=77242,s5=10,t5=-10,x5=8840,b=12.15')
#                'a1=14615, b=0, x1=1950, s1=3, x2=1335, s2=4., a2=6114,a3=2444,x3=1345,w3=4.54,a4=169,x4=1272,w4=4,a5=128,x5=1236,w5=4')
#p='a1=33400,s1=8.71,t1=-12.36,x1=6090.21,a2=12160,s2=10.5,t2=-2.1,x2=6124,a3=632,s3=11.8,t3=-2.7,x3=5800,a4=286,s4=8.2,t4=-1.3,x4=5635,b=11'    
    fit.set_functions(f=fitfunc, p='a1=33300,s1=8.744,t1=-12.23,x1=6093.152,a2=12300,s2=9,t2=-2.154,x2=6130,a3=800,s3=11,t3=-3.2,x3=5801.8,a4=1000,s4=9.9,t4=-1.51,x4=5640,a5=78140.0,s5=8,t5=-11,x5=8845,b=12.15')
    
    fit.set(xmin=E(debut),xmax=E(fin),
            plot_guess=False,
            xlabel='Energy (keV)',
            ylabel='Number of count')
    
    fit.fit()
    
#    plt.figure()
#    plt.plot(x,em_gaussian(x,*fit.results[0][:4]))
#    plt.plot(x,em_gaussian(x,*fit.results[0][4:8]))
#    plt.plot(x,em_gaussian(x,*fit.results[0][8:12]))
#    plt.plot(x,em_gaussian(x,*fit.results[0][12:16]))
#    plt.plot(x,em_gaussian(x,*fit.results[0][16:20]))
    
    
    tot = np.trapz(totspectr,x)
    
    tot = np.trapz(em_gaussian(x,*fit.results[0][:4]),x) + np.trapz(em_gaussian(x,*fit.results[0][4:8]),x)+np.trapz(em_gaussian(x,*fit.results[0][8:12]),x)+np.trapz(em_gaussian(x,*fit.results[0][12:16]),x)+np.trapz(em_gaussian(x,*fit.results[0][16:20]),x)
    
    print("Prop at",fit.results[0][3],"keV :",np.trapz(em_gaussian(x,*fit.results[0][:4]),x)/tot)
    print("Prop at",fit.results[0][7],"keV :",np.trapz(em_gaussian(x,*fit.results[0][4:8]),x)/tot)
    print("Prop at",fit.results[0][11],"keV :",np.trapz(em_gaussian(x,*fit.results[0][8:12]),x)/tot)
    print("Prop at",fit.results[0][15],"keV :",np.trapz(em_gaussian(x,*fit.results[0][12:16]),x)/tot)
    print("Prop at",fit.results[0][19],"keV :",np.trapz(em_gaussian(x,*fit.results[0][16:20]),x)/tot)
    
    
    
    return fit