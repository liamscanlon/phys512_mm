# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 20:25:50 2019
A glorified connect the dots :)
@author: liams
"""

import numpy as np
from matplotlib import pyplot as plt


"""
#this is how I got the data but you probs dont have the ~~magic~~ of spinmob installed so I'll
#hard code it like some kind of fool since getting stuff from files is a pain and is different for different OSs

import spinmob as sm 
d =sm.data.load()
T = d[0]
V = d[1]
slope = d[2]/1000
dT_dV = 1/slope
"""


#temperature data in kelvin
T = np.array([  1.4 ,   1.5 ,   1.6 ,   1.7 ,   1.8 ,   1.9 ,   2.  ,   2.1 ,
         2.2 ,   2.3 ,   2.4 ,   2.5 ,   2.6 ,   2.7 ,   2.8 ,   2.9 ,
         3.  ,   3.1 ,   3.2 ,   3.3 ,   3.4 ,   3.5 ,   3.6 ,   3.7 ,
         3.8 ,   3.9 ,   4.  ,   4.2 ,   4.4 ,   4.6 ,   4.8 ,   5.  ,
         5.2 ,   5.4 ,   5.6 ,   5.8 ,   6.  ,   6.5 ,   7.  ,   7.5 ,
         8.  ,   8.5 ,   9.  ,   9.5 ,  10.  ,  10.5 ,  11.  ,  11.5 ,
        12.  ,  12.5 ,  13.  ,  13.5 ,  14.  ,  14.5 ,  15.  ,  15.5 ,
        16.  ,  16.5 ,  17.  ,  17.5 ,  18.  ,  18.5 ,  19.  ,  19.5 ,
        20.  ,  21.  ,  22.  ,  23.  ,  24.  ,  25.  ,  26.  ,  27.  ,
        28.  ,  29.  ,  30.  ,  31.  ,  32.  ,  33.  ,  34.  ,  35.  ,
        36.  ,  37.  ,  38.  ,  39.  ,  40.  ,  42.  ,  44.  ,  46.  ,
        48.  ,  50.  ,  52.  ,  54.  ,  56.  ,  58.  ,  60.  ,  65.  ,
        70.  ,  75.  ,  77.35,  80.  ,  85.  ,  90.  , 100.  , 110.  ,
       120.  , 130.  , 140.  , 150.  , 160.  , 170.  , 180.  , 190.  ,
       200.  , 210.  , 220.  , 230.  , 240.  , 250.  , 260.  , 270.  ,
       273.  , 280.  , 290.  , 300.  , 310.  , 320.  , 330.  , 340.  ,
       350.  , 360.  , 370.  , 380.  , 390.  , 400.  , 410.  , 420.  ,
       430.  , 440.  , 450.  , 460.  , 470.  , 480.  , 490.  , 500.  ])

#voltage data in volts
V = np.array([1.64429 , 1.64299 , 1.64157 , 1.64003 , 1.63837 , 1.6366  ,
       1.63472 , 1.63274 , 1.63067 , 1.62852 , 1.62629 , 1.624   ,
       1.62166 , 1.61928 , 1.61687 , 1.61445 , 1.612   , 1.60951 ,
       1.60697 , 1.60438 , 1.60173 , 1.59902 , 1.59626 , 1.59344 ,
       1.59057 , 1.58764 , 1.58465 , 1.57848 , 1.57202 , 1.56533 ,
       1.55845 , 1.55145 , 1.54436 , 1.53721 , 1.53    , 1.52273 ,
       1.51541 , 1.49698 , 1.47868 , 1.46086 , 1.44374 , 1.42747 ,
       1.41207 , 1.39751 , 1.38373 , 1.37065 , 1.3582  , 1.34632 ,
       1.33499 , 1.32416 , 1.31381 , 1.3039  , 1.29439 , 1.28526 ,
       1.27645 , 1.26794 , 1.25967 , 1.25161 , 1.24372 , 1.23596 ,
       1.2283  , 1.2207  , 1.21311 , 1.20548 , 1.197748, 1.181548,
       1.162797, 1.140817, 1.125923, 1.119448, 1.115658, 1.11281 ,
       1.110421, 1.108261, 1.106244, 1.104324, 1.102476, 1.100681,
       1.09893 , 1.097216, 1.095534, 1.093878, 1.092244, 1.090627,
       1.089024, 1.085842, 1.082669, 1.079492, 1.076303, 1.073099,
       1.069881, 1.06665 , 1.063403, 1.060141, 1.056862, 1.048584,
       1.040183, 1.031651, 1.027594, 1.022984, 1.014181, 1.005244,
       0.986974, 0.968209, 0.949   , 0.92939 , 0.909416, 0.889114,
       0.868518, 0.847659, 0.82656 , 0.805242, 0.78372 , 0.762007,
       0.740115, 0.718054, 0.695834, 0.673462, 0.650949, 0.628302,
       0.621141, 0.605528, 0.582637, 0.559639, 0.536542, 0.513361,
       0.490106, 0.46676 , 0.443371, 0.41996 , 0.396503, 0.373002,
       0.349453, 0.325839, 0.302161, 0.278416, 0.254592, 0.230697,
       0.206758, 0.182832, 0.15901 , 0.13548 , 0.112553, 0.090681])


#dV/dT data in mV/k
dV_dTmv = np.array([-12.5 , -13.6 , -14.8 , -16.  , -17.1 , -18.3 , -19.3 , -20.3 ,
       -21.1 , -21.9 , -22.6 , -23.2 , -23.6 , -24.  , -24.2 , -24.4 ,
       -24.7 , -25.1 , -25.6 , -26.2 , -26.8 , -27.4 , -27.9 , -28.4 ,
       -29.  , -29.6 , -30.2 , -31.6 , -32.9 , -34.  , -34.7 , -35.2 ,
       -35.6 , -35.9 , -36.2 , -36.5 , -36.7 , -36.9 , -36.2 , -35.  ,
       -33.4 , -31.7 , -29.9 , -28.3 , -26.8 , -25.5 , -24.3 , -23.2 ,
       -22.1 , -21.2 , -20.3 , -19.4 , -18.6 , -17.9 , -17.3 , -16.8 ,
       -16.3 , -15.9 , -15.6 , -15.4 , -15.3 , -15.2 , -15.2 , -15.3 ,
       -15.6 , -17.  , -21.1 , -20.8 ,  -9.42,  -4.6 ,  -3.19,  -2.58,
        -2.25,  -2.08,  -1.96,  -1.88,  -1.82,  -1.77,  -1.73,  -1.7 ,
        -1.69,  -1.64,  -1.62,  -1.61,  -1.6 ,  -1.59,  -1.59,  -1.59,
        -1.6 ,  -1.61,  -1.61,  -1.62,  -1.63,  -1.64,  -1.64,  -1.67,
        -1.69,  -1.72,  -1.73,  -1.75,  -1.77,  -1.8 ,  -1.85,  -1.9 ,
        -1.94,  -1.98,  -2.01,  -2.05,  -2.07,  -2.1 ,  -2.12,  -2.14,
        -2.16,  -2.18,  -2.2 ,  -2.21,  -2.23,  -2.24,  -2.26,  -2.27,
        -2.28,  -2.28,  -2.29,  -2.3 ,  -2.31,  -2.32,  -2.33,  -2.34,
        -2.34,  -2.34,  -2.35,  -2.35,  -2.36,  -2.36,  -2.37,  -2.38,
        -2.39,  -2.39,  -2.39,  -2.39,  -2.37,  -2.33,  -2.25,  -2.12])
#convert units to V/k
dV_dT = dV_dTmv/1e3
#inverts to get dT/dV since we wint to get a temp from a voltage
#this data set doesnt have zeros in it so this is fine 
dT_dV = 1/dV_dT

##^^^ well that's awful to look at^^^



#takes an array "xs" and values (int,float,etc.) and an numpy array "xn"
#returns the index of the xs values that is cloestest to xn 
def find_nearest_index(xs, xn):
    i = (np.abs(xs - xn)).argmin()
    return i

"""
takes arrays of known x valus (xs) and corsponding y values (ys) and slopes ie dy/dx (slopes) 
and a x value (xn) for which you are trying to interpolate the corsponding y (yn)
returns interpolated value (yn)
This is optimized to "take an arbitrary voltage and interpolate to return a temperature" so if you are planing on ploting this
there are faster ways of going about this 
"""
def myCubicInterpolator(xs,ys,slopes,xn):
    #quick checks if xs values are in increasing order if
    #reverses order if order of arrays if xs values are in decreasing order
    if xs[0]>xs[-1]:
        xs = np.flip(xs)
        ys = np.flip(ys)
        slopes =np.flip(slopes)
    #makes sure xn is in range of xs if not in range returns 0 and gives message telling u to pick better xn
    if (xn<xs[0] or xn>xs[-1]):
        print("please choose value in given range of["+str(xs[0])+"," + str(xs[-1]) +"]")
        return 0
    #calls helper function to find index of xs closest to xn
    i_near = find_nearest_index(xs, xn)
    #assigns i_a and i_b to be the indeces of the nearest xs values above and below xn 
    #or if xn is equal to an xs values returns the corsponding ys value
    if xs[i_near]==xn:
        return ys[i_near]
    if xs[i_near]<xn:
        i_b,i_a =i_near,i_near+1
    if xs[i_near]>xn:
        i_b,i_a =i_near-1,i_near
    #meat of the interpolation process google "Cubic Hermite spline" if you really want to know what is going on
    g = (xn-xs[i_b])/(xs[i_a]-xs[i_b])
    h00 = 2*g**3 -3*g**2 +1
    h10 = g**3 -2*g**2 +g
    h01 = -2*g**3 +3*g**2
    h11 = g**3 - g**2
    yn = h00*ys[i_b]+ h10*(xs[i_a]-xs[i_b])*(slopes[i_b]) +h01*ys[i_a]+h11*(xs[i_a]-xs[i_b])*(slopes[i_a])
    #returns the interpolated yn value for your given xn and a rough error
    return yn

    
#This makes an array (T_terp) of interpolated T values from 5000 evenally spaced V's in V datas range and plots over points in table    
T_terp = np.array([])
testVs = np.linspace(V[0],V[-1],2000)
i=0
for x in testVs:
    i=i+1
#    terp =myCubicInterpolator(V,T,dT_dV,x)
#    T_terp =np.append(T_terp,terp)
    T_terp =np.append(T_terp,myCubicInterpolator(V,T,dT_dV,x))
plt.xlabel('Voltage (V)')
plt.ylabel('Temperature (k)')
plt.plot(testVs,T_terp,label='all data interp')
plt.plot(V,T,'*',label = 'known values')

#guess error
#so the plan here is to compare known even indexed voltages to interpolated voltages from the odd indexed T and V knoen values

V_o = V[1::2]
V_e = V[0::2]
T_o = T[1::2]
T_e = T[0::2]
dT_dV_o = dT_dV[1::2]


terp = 0 
T_terp_e = np.array([])
for x in V_e[1:]:
    terp =myCubicInterpolator(V_o,T_o,dT_dV_o,x)
    T_terp_e =np.append(T_terp_e,terp)
err_e = np.mean(np.abs(T_terp_e-T_e[1:]))
plt.plot(V_e[1:],T_terp_e,"v",label = 'interp even indexed from odd indexed')
plt.plot(V_e[1:],T_e[1:],".",label = 'even indexed')
plt.legend()
#since we know tha the error depends on distance between points to the 4 
#we assume that the average distance between points is doubled when we split into odd and even so to get erorr of the interpolation from the whole set we scale by (1/2)^4
err_all = err_e*(0.5)**4
print('My best guess for error on the diode temp voltage Interpolation is {}k'.format(err_all))
  
