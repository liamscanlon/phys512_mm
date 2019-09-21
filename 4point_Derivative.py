# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 20:09:09 2019

@author: liams
"""



#4 point d/dx
import numpy as np
from matplotlib import pyplot as plt




def der4p(func,d,x):
    a = func(x+d)
    aa = func(x+2*d)
    b = func(x-d)
    bb = func(x-2*d) 
    t1 = (1/(12.0*d))
    t2 = 8*(np.add(a,-1*b))
    t3 = -1*np.add(aa,-1*bb)
    df = t1*np.add(t2,t3)
    return df


def exp2(x):
    y = np.exp(0.01*x)
    return y
print('cats')
#
x = np.linspace(0,10,100)
NDexp1 = der4p(np.exp,10**(-16/5),x)
NDexp2 = der4p(exp2,10**(-16/5),x)

#plt.plot(x,NDexp1,".",label = "numerical  d/dx(e^x)")
#plt.plot(x,np.exp(x),"*",label = "True  d/dx(e^x)")


#plt.plot(x,NDexp2,".",label = "numerical  d/dx(e^(0.01x))")
#plt.plot(x,0.01*np.exp(0.01*x),"*",label = "True d/dx(e^(0.01x))")
#

plt.plot(x,np.abs(NDexp1-np.exp(x)), label = 'error from e^x')

plt.plot(x,np.abs(NDexp2-0.01*np.exp(0.01*x)), label = 'error from e^0.01x')
plt.legend()



import numpy as np
from matplotlib import pyplot as plt




def der4p(func,d,x):
    a = func(x+d)
    aa = func(x+2*d)
    b = func(x-d)
    bb = func(x-2*d) 
    t1 = (1/(12.0*d))
    t2 = 8*(np.add(a,-1*b))
    t3 = -1*np.add(aa,-1*bb)
    df = t1*np.add(t2,t3)
    return df


def exp2(x):
    y = np.exp(0.01*x)
    return y
print('cats')
#
x = np.linspace(0,10,100)
NDexp1 = der4p(np.exp,10**(-16/5),x)
NDexp2 = der4p(exp2,10**(-16/5),x)

#plt.plot(x,NDexp1,".",label = "numerical  d/dx(e^x)")
#plt.plot(x,np.exp(x),"*",label = "True  d/dx(e^x)")


#plt.plot(x,NDexp2,".",label = "numerical  d/dx(e^(0.01x))")
#plt.plot(x,0.01*np.exp(0.01*x),"*",label = "True d/dx(e^(0.01x))")

plt.plot(x,np.abs(NDexp1-np.exp(x)), label = 'error from e^x')
plt.plot(x,np.abs(NDexp2-0.01*np.exp(0.01*x)), label = 'error from e^0.01x')

print('looks alright tho I think I might have made a calculation error somewhere but Im not willing go go back now')
