# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 10:29:40 2018

@author: noudy
"""
import numpy as np
import matplotlib.pyplot as plt 
import spinmob as s 
import matplotlib.lines as mlines

T1_lo = [
[0.12,7.39,7.12,7.43,7.32,7.24,7.11,7.23,7.36,7.27,7.07],
[0.13,7.80,7.48,7.42,7.84,7.59,8.32,7.35,7.42,7.43,7.72],
[0.14,8.25,7.91,7.84,8.01,8.41,8.24,8.01,8.02,8.33,8.32],
[0.15,8.66,8.56,8.64,8.58,8.89,8.72,8.29,8.40,8.72,8.13],
[0.16,9.05,8.69,8.97,8.49,8.96,8.76,8.97,8.62,8.60,8.79],
[0.17,8.76,9.20,9.28,9.20,9.20,9.10,9.60,9.11,9.05,8.96],
[0.18,9.22,9.02,9.51,9.38,9.44,9.44,9.75,9.00,9.45,9.17],
[0.19,9.60,9.31,9.61,9.68,9.44,9.68,9.35,9.92,9.39,9.76],
[0.20,9.31,9.92,9.77,9.92,9.59,9.84,9.85,9.55,9.50,9.84],
[0.21,10.1,9.81,9.98,9.84,10.0,9.84,9.93,9.71,10.0,9.92],
[0.22,10.3,10.2,10.0,9.99,9.78,10.4,9.97,10.2,10.1,9.89]]

T1_lo = np.asarray(T1_lo)

def computeT1(T1_lo):
    taus = T1_lo[:,0]
    values = np.mean(T1_lo[:,1:],axis=1)
    err = np.std(T1_lo[:,1:],axis=1)/np.sqrt(10)

    plt.figure()
    plt.errorbar(taus,values,yerr=err,linestyle="",marker="o")

    f = s.data.fitter()
    f.set(plot_guess=False)
    f.set_data(xdata=taus, ydata=values,eydata=err)
    f.set_functions(f='M0*(1-2*exp(-x/T1))',p='M0=13,T1=0.1')
    f.set(xlabel="Time (ms)",ylabel="Peak height (V)")
    f.fit()
    
    return f.results[0][1]

#fitting for the systematics

tau = [0.22,0.13,0.04]
#7,3.6
V7_36 = [10.02,7.76,4.36]
err7_36 = [0.06,0.07,0.06]
#7.10,3.6
V71_36 = [10.03,7.85,4.78]
err71_36 = [0.03,0.06,0.02]
#
V72_36 = [10.0, 7.81,5.07]
err72_36 = [0.05,0.06,0.06]
#
V73_36 = [9.93,7.79,5.56]
err73_36 = [0.07,0.05,0.03]
#
V74_36 = [9.95,7.75,5.97]
err74_36 = [0.07,0.04,0.03]
#
V72_34 = [9.88,7.68,5.11]
err72_34 = [0.04,0.05,0.03]
#
V72_35 = [9.8,7.75,5.06]
err72_35 = [0.1,0.04,0.06]
#
V72_37 = [10.01,7.82,5.13]
err72_37 = [0.04,0.03,0.03]
#
V72_38 = [10.00,7.83,5.10]
err72_38 = [0.04,0.04,0.04]

plt.errorbar(tau,V7_36,yerr=err7_36,linestyle="",fmt="o",ecolor="grey",color="blue")
plt.errorbar(tau,V71_36,yerr=err71_36,linestyle="",fmt="o",ecolor="grey",color="orange")
plt.errorbar(tau,V72_36,yerr=err72_36,linestyle="",fmt="o",ecolor="grey",color="green")
plt.errorbar(tau,V73_36,yerr=err73_36,linestyle="",fmt="o",ecolor="grey",color="red")
plt.errorbar(tau,V74_36,yerr=err74_36,linestyle="",fmt="o",ecolor="grey",color="purple")

plt.xlabel("$ \\tau $ ($\mu s$)")

plt.ylabel("Height ($V$)")


blue = mlines.Line2D([], [], color='blue', marker='o',linestyle='', label='$A_{len} = 7 \mu s$, $B_{len} = 3.6 \mu s$')
orange = mlines.Line2D([], [], color='orange', marker='o',linestyle='', label='$A_{len} = 7.1 \mu s$, $B_{len} = 3.6 \mu s$')
green = mlines.Line2D([], [], color='green', marker='o',linestyle='', label='$A_{len} = 7.2 \mu s$, $B_{len} = 3.6 \mu s$')
red = mlines.Line2D([], [], color='red', marker='o',linestyle='', label='$A_{len} = 7.3 \mu s$, $B_{len} = 3.6 \mu s$')
purple = mlines.Line2D([], [], color='purple', marker='o',linestyle='', label='$A_{len} = 7.4 \mu s$, $B_{len} = 3.6 \mu s$')

plt.legend(handles=[blue,orange,green,red,purple][::-1])

plt.figure()

plt.errorbar(tau,V72_34,yerr=err72_34,linestyle="",fmt="o",ecolor="grey",color="blue")
plt.errorbar(tau,V72_35,yerr=err72_35,linestyle="",fmt="o",ecolor="grey",color="orange")
plt.errorbar(tau,V72_36,yerr=err72_36,linestyle="",fmt="o",ecolor="grey",color="green")
plt.errorbar(tau,V72_35,yerr=err72_37,linestyle="",fmt="o",ecolor="grey",color="red")
plt.errorbar(tau,V72_37,yerr=err72_38,linestyle="",fmt="o",ecolor="grey",color="purple")

plt.xlabel("$ \\tau $ ($\mu s$)")

plt.ylabel("Height ($V$)")


blue = mlines.Line2D([], [], color='blue', marker='o',linestyle='', label='$A_{len} = 7.2 \mu s$, $B_{len} = 3.4 \mu s$')
orange = mlines.Line2D([], [], color='orange', marker='o',linestyle='', label='$A_{len} = 7.2 \mu s$, $B_{len} = 3.5 \mu s$')
green = mlines.Line2D([], [], color='green', marker='o',linestyle='', label='$A_{len} = 7.2 \mu s$, $B_{len} = 3.6 \mu s$')
red = mlines.Line2D([], [], color='red', marker='o',linestyle='', label='$A_{len} = 7.2 \mu s$, $B_{len} = 3.7 \mu s$')
purple = mlines.Line2D([], [], color='purple', marker='o',linestyle='', label='$A_{len} = 7.2 \mu s$, $B_{len} = 3.8 \mu s$')

plt.legend(handles=[blue,orange,green,red,purple][::-1])


t = [[40.14,40.14,40.21],[60.2,60.2,60.28],[80.27,80.27,80.34],[100.42,100.42,100.42],[120.48,120.48,120.56],[140.55,140.63,140.63],[160.69,160.69,160.69],[180.76,180.76,180.84]]
V = [[8.36,8.27,8.18],[6.25,6.07,5.98],[4.93,4.84,4.75],[3.87,3.70,3.61],[3.08,2.90,2.90],[2.55,2.46,2.46],[2.11,2.11,1.85],[1.94,1.76,1.58]]

tA360B710,_ = np.mean(np.asarray(t)/1000,axis=1),np.std(np.asarray(t)/1000,axis=1)
VA360B710,errA360B710 = np.mean(V,axis=1),np.std(V,axis=1)/np.sqrt(3)

t = [[40.14,40.21,40.21],[60.28,60.28,60.2],[80.35,80.35,80.27],[100.42,100.42,100.42],[120.48,120.56,120.56],[140.63,140.63,140.63],[160.69,160.69,160.69]]
V = [[8.36,8.36,8.09],[6.07,5.98,5.98],[4.22,4.84,4.93],[3.78,3.78,3.61],[2.99,3.08,3.08],[2.55,2.46,2.38],[2.29,2.2,2.2]]


tA360B730,_ = np.mean(np.asarray(t)/1000,axis=1),np.std(np.asarray(t)/1000,axis=1)
VA360B730, errA360B730 = np.mean(V,axis=1),np.std(V,axis=1)/np.sqrt(3)

#tA350B720 = meant
#VA350B720 = meanV
#errA350B720 = stdV

#tA370B720 = meant
#VA370B720 = meanV
#errA370B720 = stdV

#tA350B700 = meant
#VA350B700 = meanV
#errA350B700 = stdV
plt.figure()
plt.errorbar(meant,meanV,yerr=stdV,linestyle="",fmt="o",ecolor="grey",color="blue",alpha=0.5)

plt.errorbar(tA360B710,VA360B710,yerr=errA360B710,linestyle="",fmt="v",ecolor="grey",color="red",alpha=0.5)

plt.errorbar(tA360B730,VA360B730,yerr=errA360B730,linestyle="",fmt="s",ecolor="grey",color="green",alpha=0.5)

plt.errorbar(tA350B720,VA350B720,yerr=errA350B720,linestyle="",fmt="P",ecolor="grey",color="orange",alpha=0.5)

plt.errorbar(tA370B720,VA370B720,yerr=errA370B720,linestyle="",fmt="*",ecolor="grey",color="purple",alpha=0.5)

plt.errorbar(tA350B700,VA350B700,yerr=errA350B700,linestyle="",fmt="d",ecolor="grey",color="brown",alpha=0.5)

blue = mlines.Line2D([], [], color='blue', marker='o',linestyle='', label='$A_{len} = 7.2 \mu s$, $B_{len} = 3.6 \mu s$')
red = mlines.Line2D([], [], color='red', marker='v',linestyle='', label='$A_{len} = 7.1 \mu s$, $B_{len} = 3.6 \mu s$')
green = mlines.Line2D([], [], color='green', marker='s',linestyle='', label='$A_{len} = 7.3 \mu s$, $B_{len} = 3.6 \mu s$')
orange = mlines.Line2D([], [], color='orange', marker='P',linestyle='', label='$A_{len} = 7.2 \mu s$, $B_{len} = 3.5 \mu s$')
purple = mlines.Line2D([], [], color='purple', marker='*',linestyle='', label='$A_{len} = 7.2 \mu s$, $B_{len} = 3.7 \mu s$')
brown = mlines.Line2D([], [], color='brown', marker='d',linestyle='', label='$A_{len} = 7 \mu s$, $B_{len} = 3.5 \mu s$')

plt.xlabel("Time (s)")
plt.ylabel("Height (V)")

plt.legend(handles=[blue,orange,green,red,purple,brown])
