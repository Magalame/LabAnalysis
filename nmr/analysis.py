# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 17:08:55 2018

@author: noudy
"""

import numpy as np
import matplotlib.pyplot as plt
#import spinmob as s


DPATH=r"C:\Users\noudy\Documents\Cours\PHYS439\Exp3\SCOPE_DATA_INT"

T2 =r"C:\Users\noudy\Documents\Cours\PHYS439\Exp3\light oil t2"

def findm(zipped,value):
    ts = []
    Vs = []
    for t,V in zipped:
        if V >= value:
            ts.append(t)
            Vs.append(V)
            print(t,V)
    return ts,Vs


def load(path):
    return np.loadtxt(path,skiprows=2,delimiter=",")

def computeT2(t,M0,T2):
    return M0*np.exp(-t/T2)

def fitT2(ts,Vs,err):
    
    f = s.data.fitter()
    f.set_data(xdata=ts, ydata=Vs, eydata=err)
    f.set_functions(f='M0*exp(-t/T2)',p='M0,T2')
    
    f.fit()
    
def plotf(file):
    out = load(T2+file)
    t,V = out.transpose()
    plt.plot(t,V)