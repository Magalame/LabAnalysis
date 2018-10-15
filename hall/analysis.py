# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 09:15:20 2018

@author: noudy
"""
import matplotlib.pyplot as plt
import spinmob as s
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.cm as cm

import os

arr1 = [-5.603,-5.587,-5.571,-5.555,-5.539,-5.523,-5.506,-5.489,-5.473,-5.456,
        -5.439,-5.421,-5.404,-5.387,-5.369,-5.351,-5.334,-5.316,-5.297,-5.279,
        -5.261,-5.242,-5.224,-5.205,-5.186,-5.167,-5.148,-5.128,-5.109,-5.089,
        -5.070,-5.050,-5.030,-5.010,-4.989,-4.969,-4.949,-4.928,-4.907,-4.886,
        -4.865,-4.844,-4.823,-4.802,-4.780,-4.759,-4.737,-4.715,-4.693,-4.671,
        -4.648,-4.626,-4.604,-4.581,-4.558,-4.535,-4.512,-4.489,-4.466,-4.443,
        -4.419,-4.395,-4.372,-4.348,-4.324,-4.300,-4.275,-4.251,-4.226,-4.202,
        -4.177,-4.152,-4.127,-4.102,-4.077,-4.052,-4.026,-4.000,-3.975,-3.949,
        -3.923,-3.897,-3.871,-3.844,-3.818,-3.791,-3.765,-3.738,-3.711,-3.684,
        -3.657,-3.629,-3.602,-3.574,-3.546,-3.519,-3.491,-3.463,-3.435,-3.407,
        -3.379,-3.350,-3.322,-3.293,-3.264,-3.235,-3.206,-3.177,-3.148,-3.118,
        -3.089,-3.059,-3.030,-3.000,-2.970,-2.940,-2.910,-2.879,-2.849,-2.818,
        -2.788,-2.757,-2.726,-2.695,-2.664,-2.633,-2.602,-2.571,-2.539,-2.507,
        -2.476,-2.444,-2.412,-2.380,-2.348,-2.316,-2.283,-2.251,-2.218,-2.186,
        -2.153,-2.120,-2.087,-2.054,-2.021,-1.987,-1.954,-1.920,-1.887,-1.853,
        -1.819,-1.785,-1.751,-1.717,-1.683,-1.648,-1.614,-1.579,-1.545,-1.510,
        -1.475,-1.440,-1.405,-1.370,-1.335,-1.299,-1.264,-1.228,-1.192,-1.157,
        -1.121,-1.085,-1.049,-1.013,-0.976,-0.940,-0.904,-0.867,-0.830,-0.794,
        -0.757,-0.720,-0.683,-0.646,-0.608,-0.571,-0.534,-0.496,-0.459,-0.421,
        -0.383,-0.345,-0.307,-0.269,-0.231,-0.193,-0.154,-0.116,-0.077,-0.039]

arr2 =[0.000,0.039,0.078,0.117,0.156,0.195,0.234,0.273,0.312,0.352,
       0.391,0.431,0.470,0.510,0.549,0.589,0.629,0.669,0.709,0.749,
       0.790,0.830,0.870,0.911,0.951,0.992,1.033,1.074,1.114,1.155,
       1.196,1.238,1.279,1.320,1.362,1.403,1.445,1.486,1.528,1.570,
       1.612,1.654,1.696,1.738,1.780,1.823,1.865,1.908,1.950,1.993,
       2.036,2.079,2.122,2.165,2.208,2.251,2.294,2.338,2.381,2.425,
       2.468,2.512,2.556,2.600,2.643,2.687,2.732,2.776,2.820,2.864,
       2.909,2.953,2.998,3.043,3.087,3.132,3.177,3.222,3.267,3.312,
       3.358,3.403,3.448,3.494,3.539,3.585,3.631,3.677,3.722,3.768,
       3.814,3.860,3.907,3.953,3.999,4.046,4.092,4.138,4.185,4.232,
       4.279,4.325,4.372,4.419,4.466,4.513,4.561,4.608,4.655,4.702,4.750]

voltages = arr1 + arr2
temp = range(-200,111)
errors = np.maximum(1,[0.015*(-i) if i < 0 else 0.0075*i for i in temp])
voltToCel = interp1d(np.array(voltages)*(1e-3), temp)
#def func(x,A,C):
#    return A*np.exp(x)+C

#fit = s.data.fitter()
##fit(plot_guess=False)
##fit.set_functions(f='a*log(c*x+b)', p='a=50,b=6.2,c')
#fit.set_functions(f='a0+a1*x+a2*x**2+a3*x**3+a4*x**4', p='a0=0.645412,a1= 25501.1,a2=-992146,a3')
#fit.set_data(xdata=np.array(voltages)*(1e-3), ydata=temp, eydata=errors)
#fit.fit()


PATH = r"C:\Users\noudy\Documents\Cours\PHYS439\Exp1\data\\"
DATAPATH = r"C:\Users\noudy\Documents\Cours\PHYS439\Exp1\data\tempdep\\"
DATAPATH2 = r"C:\Users\noudy\Documents\Cours\PHYS439\Exp1\data\halleffect\\"
DATAPATH3 = r"C:\Users\noudy\Documents\Cours\PHYS439\Exp1\data\halleffect2\\"


#def Terr(x):
#    return np.max(1,0.015*(-x) if x < 0 else 0.0075*x)

def fuse(path):
    tmp = []
    for file in os.listdir(path):
        tmp.append(np.loadtxt(path+file,skiprows=1,delimiter=","))
        
    return np.concatenate(tmp,axis=0)

def fuseOld(listnames,path):
    tmp = []
    for file in listnames:
        tmp.append(np.loadtxt(path+file,skiprows=1,delimiter=","))
        
    return np.concatenate(tmp,axis=0)

def fuseData(listNames):
    return fuseOld(listNames,DATAPATH)

def fuseData2(listNames):
    return fuseOld(listNames,DATAPATH2)

def fuseData3(listNames):
    return fuseOld(listNames,DATAPATH3)


def getVx(data):
    return data[:,4],data[:,5],0.0002,0.0003
#    np.std([data[:,4],data[:,5]],axis=0)
#    return data[:,5],0.0001e-3

def getVy(data):
    return data[:,0],data[:,1],0.00003,0.00008

def getKelvin(data):
    celsius = voltToCel(data[:,7])
    kelvins = celsius +273.15
    err = np.maximum(1,[0.015*(-i) if i < 0 else 0.0075*i for i in celsius])
    return kelvins,err

def plotTempVsVx(data):
    kelvins,err = getKelvin(data)

    Vx,err = getVx(data)
    
    plt.figure()
    plt.scatter(kelvins,vx,color='red')
    
    plt.xlabel("Temperature (°C)")
    plt.ylabel("$V_x$ (V)")

def plotTempVsSigma(data):
    
    kelvins,errK = getKelvin(data)

    Vx,errVx = getVx(data)
    
    sigma,dsigma = Vtos(Vx,errVx)
    
    plt.figure()
    
    plt.errorbar(kelvins,sigma, xerr=errK,yerr=dsigma,fmt='o', ecolor='grey')
    
    plt.xlabel("Temperature (K)")
    plt.ylabel("$\sigma$")
    
def fitTempvsSigmaHot(data):
    
    kelvins,errK = getKelvin(data)

    Vx,errVx = getVx(data)
    
    sigma,dsigma = Vtos(Vx,errVx)
    
    fit = s.data.fitter()
#    fit(plot_guess=False)
    fit(xlabel="Temperature (K)")
    fit(ylabel="$\sigma\quad(S/m)$")
    # E should be around 1.073458e-19
    #a*exp(E/(2*1.38064852e-23*x))
    fit.set_functions(f='c+a*exp(-E/(2*1.38064852e-23*(x)))', p='a=10000000,E=1.073458e-19,c')

    start = int(0.70*len(kelvins))
    
    assert len(dsigma) == len(errK)
    
    fit.set_data(xdata=kelvins[start:], ydata=sigma[start:],eydata=dsigma[start:])
    
    fit.fit()
    
    
    return fit

def fitTempvsSigmaCold(data):
    
    kelvins,errK = getKelvin(data)

    Vx,errVx = getVx(data)
    
    sigma,dsigma = Vtos(Vx,errVx)
    
    fit = s.data.fitter()
    fit(plot_guess=False)
    fit(xlabel="Temperature (K)")
    fit(ylabel="$\sigma\quad(S/m)$")
    # E should be around 1.073458e-19
    #a*exp(E/(2*1.38064852e-23*x))
    fit.set_functions(f='a*x**(-3/2)+c', p='a=507500,c')

    
    assert len(dsigma) == len(errK)
    
    fit.set_data(xdata=kelvins, ydata=sigma,eydata = dsigma)
    
    fit.fit()
    
    
    return fit
out = []
def fitTempvsSigma(data):
    global out
    kelvins,errK = getKelvin(data)

    Vxlow,Vxhigh,errVxlow,errVxhigh = getVx(data)
    
#    out = kelvins,Vx
    
    sigma1,dsigma1 = Vtos(Vxlow,errVxlow,chn="V5")
    
    sigma2,dsigma2= Vtos(Vxhigh,errVxhigh,chn="V6")
    
    #fit from lower voltage
    
    fit = s.data.fitter()
    fit(plot_guess=False)
    fit(xlabel="Temperature (K)")
    fit(ylabel="$\sigma\quad(S/m)$") 
    fit.set_functions(f='b*x**(-3/2)+a*exp(-E/(2*8.6173303e-5*x))+c', p='a=20000000000,E=0.6,b=775600,c')
    fit.set_data(xdata=kelvins, ydata=sigma1,eydata=dsigma1)
    fit.fit()
    
#    #from higher
#
#    fit2 = s.data.fitter()
#    fit2(plot_guess=False)
#    fit2(xlabel="Temperature (K)")
#    fit2(ylabel="$\sigma\quad(S/m)$") 
#    fit2.set_functions(f='b*x**(-3/2)+a*exp(-E/(2*8.6173303e-5*x))+c', p='a=20000000000,E=0.6,b=775600,c')
#    fit2.set_data(xdata=kelvins[500:], ydata=sigma2[500:],eydata=dsigma2[500:])
#    fit2.fit() 
#    
#    return fit,fit2

def Vtos(V,dV,chn="V5"):
    
    
    
    h = 1.01e-3
    dh = 0.02e-3
    
    w = 2.79e-3
    dw = 0.02e-3
    
#    L = 9.87e-3
    if chn == "V5":
        L = 4.29e-3
    elif chn == "V6":
        L = 3.49e-3
    
    dL = 0.02e-3
    
    I = 1e-3 #1mA
    
    A = h*w
    dA = A*np.sqrt((dh/h)**2+(dw/w)**2)
    
    dVA = V*A*np.sqrt((dV/V)**2+(dA/A)**2)
    
    dIL = I*dL
    
    sigma = I*L/(V*A)
    
    dsigma = sigma*np.sqrt((dIL/(I*L))**2+(dVA/(V*A))**2)
    
    
    return sigma,dsigma



def analysis():
    data = fuseData([
                     "2018-09-13-09_24-Voltage-Readings-Data.csv",
                     "2018-09-13-09_34-Voltage-Readings-Data.csv",
                     "2018-09-13-09_47-Voltage-Readings-Data.csv",
                     "2018-09-13-09_57-Voltage-Readings-Data.csv",
                     "2018-09-13-10_07-Voltage-Readings-Data.csv",
                     "2018-09-13-10_17-Voltage-Readings-Data.csv",
                     "2018-09-13-10_26-Voltage-Readings-Data.csv",
                     "2018-09-13-10_36-Voltage-Readings-Data.csv",
                     "2018-09-13-10_46-Voltage-Readings-Data.csv",
                      "2018-09-18-09_38-Voltage-Readings-Data.csv",
                      "2018-09-18-09_48-Voltage-Readings-Data.csv",
                      "2018-09-18-09_58-Voltage-Readings-Data.csv",
                      "2018-09-18-10_08-Voltage-Readings-Data.csv",
                      "2018-09-18-10_17-Voltage-Readings-Data.csv",
                      "2018-09-18-10_27-Voltage-Readings-Data.csv",
                      "2018-09-18-10_37-Voltage-Readings-Data.csv",
                      "2018-09-18-10_47-Voltage-Readings-Data.csv",
                      "2018-09-18-10_56-Voltage-Readings-Data.csv",
                      "2018-09-18-11_06-Voltage-Readings-Data.csv",
                      "2018-09-18-11_16-Voltage-Readings-Data.csv"
                      ])
    
#    data = fuse(DATAPATH)
#    plotTempVsSigma(data)
#    out = fitTempvsSigma(data)
#    return out
#    plotTempVsSigma(data)
    fitTempvsSigma(data)
    return data

def analysisHall():
#    data = fuseData2(["2018-09-20-09_49-Voltage-Readings-Data.csv",
#                     "2018-09-20-09_59-Voltage-Readings-Data.csv",
#                     "2018-09-20-10_09-Voltage-Readings-Data.csv",
#                     "2018-09-20-10_18-Voltage-Readings-Data.csv",
#                     "2018-09-20-10_28-Voltage-Readings-Data.csv",
#                     "2018-09-20-10_38-Voltage-Readings-Data.csv",
#                     "2018-09-20-10_48-Voltage-Readings-Data.csv",
#                     "2018-09-20-10_57-Voltage-Readings-Data.csv",
#                     "2018-09-20-11_07-Voltage-Readings-Data.csv"])
#    
#    data2 = fuseData3(["2018-09-25-10_14-Voltage-Readings-Data.csv",
#                      "2018-09-25-10_23-Voltage-Readings-Data.csv",
#                      "2018-09-25-10_33-Voltage-Readings-Data.csv",
#                      "2018-09-25-10_43-Voltage-Readings-Data.csv",
#                      "2018-09-25-10_53-Voltage-Readings-Data.csv",
#                      "2018-09-25-11_03-Voltage-Readings-Data.csv",
#                      "2018-09-25-11_13-Voltage-Readings-Data.csv",
#                      "2018-09-25-11_22-Voltage-Readings-Data.csv",
#                      "2018-09-25-11_32-Voltage-Readings-Data.csv",
#                      "2018-09-25-11_42-Voltage-Readings-Data.csv",
#                      "2018-09-25-11_52-Voltage-Readings-Data.csv"])
    
    data = fuse(PATH+"halleffect\\")
    data2 = fuse(PATH+"halleffect2\\")
    data3 = fuse(PATH+"halleffect3\\")
    
    temp,Vytop1,Vytop2 = binning2(data3,data2,0)
    Vytop = Vytop1-Vytop2

    temp,Vydown1,Vydown2 = binning2(data3,data2,1)
    Vydown = Vydown1-Vydown2
    
    temp = np.array(list(temp[:46])+list(temp[48:]))
    
    Vytop = np.array(list(Vytop[:46])+list(Vytop[48:]))
    
    Vydown = np.array(list(Vydown[:46])+list(Vydown[48:]))
    
#    plt.figure()
#    plt.scatter(temp,Vytop)
#    plt.scatter(temp,Vydown)

    temp,Vxtop1,Vxtop2 = binning2(data3,data2,4)
    Vxtop = np.mean([Vxtop1,Vxtop2],axis=0)
    
    temp,Vxdown1,Vxdown2 = binning2(data3,data2,5)
    Vxdown = np.mean([Vxdown1,Vxdown2],axis=0)
    
    temp = np.array(list(temp[:46])+list(temp[48:]))
    
    Vxtop = np.array(list(Vxtop[:46])+list(Vxtop[48:]))
    
    Vxdown = np.array(list(Vxdown[:46])+list(Vxdown[48:]))
    
    #-----------sub(Vylow,Vxlow),sub(Vylow,Vxhigh),sub(Vyhigh,Vxlow),sub(Vyhigh,Vxhigh)

    errK = np.maximum(1,[0.015*(-i) if i < 0 else 0.0075*i for i in temp]) #errors on temperature

    
#    (angle1,err1),(angle2,err2),(angle3,err3),(angle4,err4) = getHallAngle(Vytop,Vxtop,Vydown,Vxdown)
#    
#    
#    
#    fig, ax = plt.subplots()
#        
#    blue = mpatches.Patch(color='blue', label='$\phi(V_1,V_5)$')
#    orange = mpatches.Patch(color='orange', label='$\phi(V_1,V_6)$')
#    red = mpatches.Patch(color='red', label='$\phi(V_2,V_5)$')
#    green = mpatches.Patch(color='green', label='$\phi(V_2,V_6)$')
#    
#    plt.legend(handles=[blue,orange,red,green])
#        
#    temp1,angle1,err1,errK1 = trimData([temp,angle1,err1,errK],scale=1)
#    temp2,angle2,err2,errK2 = trimData([temp,angle2,err2,errK],scale=1)
#    temp3,angle3,err3,errK3 = trimData([temp,angle3,err3,errK],scale=1)
#    temp4,angle4,err4,errK4 = trimData([temp,angle4,err4,errK],scale=1)
#    
#    ax.errorbar(temp1,angle1, xerr=errK1,yerr=err1,fmt='o', ecolor='grey',color='blue')
#    ax.errorbar(temp2,angle2, xerr=errK2,yerr=err2,fmt='o', ecolor='grey',color='orange')
#    ax.errorbar(temp3,angle3, xerr=errK3,yerr=err3,fmt='o', ecolor='grey',color='red')
#    ax.errorbar(temp4,angle4, xerr=errK4,yerr=err4,fmt='o', ecolor='grey',color='green')
#    
#    ax.set_xlabel("Temperature (K)")
#    ax.set_ylabel("Hall angle")
#    
#    plt.figure()
#    
#    plt.scatter(temp1,angle1/angle2)
#    plt.scatter(temp3,angle3/angle4)
    
    #----------
    
    fig, ax = plt.subplots()
    
    custom_lines = [Line2D([0], [0], color='blue', lw=4),Line2D([0], [0], color='red', lw=4)]
    ax.legend(custom_lines, ['$R_H$ from $V_1$', '$R_H$ from $V_2$'])
    
    (coeflow,errlow),(coefhigh,errhigh) = getHallcoef(Vytop,Vydown)
    
    templow,coeflow,errlow,errKlow = trimData([temp,coeflow,errlow,errK])
    temphigh,coefhigh,errhigh,errKhigh = trimData([temp,coefhigh,errhigh,errK])
    
    
    ax.errorbar(templow,coeflow,xerr=errKlow,yerr=errlow,fmt='o', ecolor='grey',color='blue')
    ax.errorbar(temphigh,coefhigh,xerr=errKhigh,yerr=errhigh,fmt='o', ecolor='grey',color='red')
    
    plt.xlabel("Temperature (K)")
    plt.ylabel("Hall coefficient $(m^3C^{-1})$")
    
    return templow,coeflow,errlow
#    
#    #------------------
    
    
    
#    
##    
#    mobilitytoptop = hallMobility(Vytop,Vxtop,"V5")
#    mobilitytopdown = hallMobility(Vytop,Vxdown,"V6")
#    mobilitydowndown = hallMobility(Vydown,Vxdown,"V6")
#    mobilitydowntop = hallMobility(Vydown,Vxtop,"V5")
#    
#    temptoptop,mobilitytoptop = trimData([temp,mobilitytoptop],scale=1)
#    temptopdown,mobilitytopdown = trimData([temp,mobilitytopdown],scale=1)
#    tempdowndown,mobilitydowndown = trimData([temp,mobilitydowndown],scale=1)
#    tempdowntop,mobilitydowntop = trimData([temp,mobilitydowntop],scale=1)
#    
#    
##    
#    plt.figure()
#    
#    blue = mpatches.Patch(color='blue', label='$\mu(V_1,V_5)$')
#    orange = mpatches.Patch(color='orange', label='$\mu(V_1,V_6)$')
#    red = mpatches.Patch(color='red', label='$\mu(V_2,V_6)$')
#    green = mpatches.Patch(color='green', label='$\mu(V_2,V_5)$')
#    
#    plt.legend(handles=[blue,orange,red,green])
#    
#    
#    plt.scatter(temptoptop,mobilitytoptop)
#    plt.scatter(temptopdown,mobilitytopdown)
#    plt.scatter(tempdowndown,mobilitydowndown)
#    plt.scatter(tempdowntop,mobilitydowntop)
#    plt.scatter(temp,coefhigh*Vtos(Vxtop,0.0002)[0])
##    
#    plt.xlabel("Temperature ($K$)")
#    plt.ylabel("Mobility ($m^2 V^{−1} s^{−1}$)")
#    
#    plt.figure()
#    
#    blue = mpatches.Patch(color='blue', label='$\dfrac{\mu(V_1,V_5)}{\mu(V_1,V_6)}$')
#    red = mpatches.Patch(color='red', label='$\dfrac{\mu(V_2,V_5)}{\mu(V_2,V_6)}$')
#    
#    plt.legend(handles=[blue,red])
#    
#    
#    plt.scatter(temptoptop,mobilitytoptop/mobilitytopdown)
#    plt.scatter(tempdowntop,mobilitydowntop/mobilitydowndown,color='red')
##    
#    plt.xlabel("Temperature ($K$)")
#    plt.ylabel("Mobility (ration")
    
#    f = s.data.fitter()
#    f(xlabel="Temperature (K)")
#    f(ylabel="$Mobility$") 
#    f.set_functions(['a*x**(-3/2)+b','c*x**(-3/2)+d','e*x**(-3/2)+f','g*x**(-3/2)+h'], p='a,b,c,d,e,f,g,h')
#    trim = -20
#    f.set_data([temptoptop[:trim],temptopdown[:trim],tempdowndown[:trim],tempdowntop[:trim]], [mobilitytoptop[:trim],mobilitytopdown[:trim],mobilitydowndown[:trim],mobilitydowntop[:trim]])
#    f.fit()
    
#    trim = -20
#    
#    f = s.data.fitter()
#    f(xlabel="Temperature (K)")
#    f(ylabel="$\mu$ ($m^2 V^{−1} s^{−1}$)") 
#    f.set_functions(['a*x**(-3/2)+b'], p='a,b')
#    f.set_data(temptoptop[:trim],mobilitytoptop[:trim])
#    f.fit()
    
#    plt.figure()
#    plt.scatter(kelvins,data[:,0])
#    plt.scatter(kelvins,data[:,1])
#    
#    plt.xlabel("Temperature")
#    plt.ylabel("Hall voltages")
    
#    return data,data2,data3
    return Vytop,Vxtop,Vydown,Vxdown

def getHallAngle(Vylow,Vxlow,Vyhigh,Vxhigh):
    
    def sub(Vy,Vx):
        
        dV = 0.0003
        
        Z = Vy/Vx
        
        return Z,Z*np.sqrt((dV/Vy)**2+(dV/Vx)**2)
        
    
    return sub(Vylow,Vxlow),sub(Vylow,Vxhigh),sub(Vyhigh,Vxlow),sub(Vyhigh,Vxhigh)

def getHallcoef(Vytop,Vydown):
    
    h = 1.01e-3
    dh = 0.02e-3
    
    def sub(Vy):
        Z = Vy*h/(0.4973*1e-3)
        
        return Z, Z*np.sqrt((dh/h)**2+(0.0003/Vy)**2)
    
    coeftop = sub(Vytop)
    coefdown = sub(Vydown)
    
    
    
    return coeftop,coefdown

def binning(data,index):#organizes unsorted data in a histogram form
    
    kelvins,errK = getKelvin(data)
    
    dT = kelvins.max() - kelvins.min()
    
    temp = np.arange(kelvins.min(),kelvins.max(),2)
    
    N = len(temp)
    
    out = [[] for i in range(N)]
    
    for i,k in enumerate(kelvins):
    
        out[int((k-kelvins.min())//2)].append(data[i,index])
        
    for i in range(len(out)):
        out[i] = np.asarray(out[i])
        
    return temp,out
    
def binning2(data,data2,index):
    
    kelvins,errK = getKelvin(data)
    
    kelvins2,errK2 = getKelvin(data2)
    
    maxk = min(kelvins.max(),kelvins2.max())
    
    mink = max(kelvins.min(),kelvins2.min())
    
    temp = np.arange(mink,maxk,2)
    
    N = len(temp)
    
    out1 = subbinning(N,kelvins,data, index)
    out2 = subbinning(N,kelvins2,data2, index)
        
    return temp,out1,out2
    

def subbinning(N,kelvins,data, index):

    out = [[] for i in range(N)]
    
    for i,k in enumerate(kelvins):
    
        if int((k-kelvins.min())//2) < N:
            out[int((k-kelvins.min())//2)].append(data[i,index])
    
    
    for i in range(len(out)):
        out[i] = np.mean(out[i])
        
    return np.asarray(out)

def hallMobility(Vy,Vx,chn):
    if chn == "V5":
        l = 4.29e-3
    elif chn == "V6":
        l = 3.49e-3
    w = 4e-3
    B = 0.4973
    return (Vy*l)/(Vx*B*w)

def trimData(data,scale=10):#data = [T,x,err]
    
    
    data = np.transpose(data)
    
    out = []
    
    out.append(data[0])
    
    for i in range(len(data)-2):
        
        if abs(data[i][1]-data[i+1][1]) > scale*abs(data[i][1]-data[i+2][1]) and abs(data[i+2][1]-data[i+1][1]) > scale*abs(data[i][1]-data[i+2][1]):
            pass
        else:
            out.append(data[i+1])
            
    out.append(data[-1])
    
    return np.transpose(out)