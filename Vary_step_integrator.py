# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 18:40:08 2019

@author: liams
"""

import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as plt

def linear(x):
    return x

def sqr(x):
    return x**2
    
def crazypoly(x):
    return 5*x**4 + 3*x**3 -x**2 +58*x +55





#################################################################################################################
#################################################################################################################
#NOT MY CODE 
#used to compare results
#you can tell because I have the sense to add more than 1 comment in my code :)
def simple_integrate(fun,a,b,tol):
    x=np.linspace(a,b,5)
    y=fun(x)
    neval=len(x) #let's keep track of function evaluations
    f1=(y[0]+4*y[2]+y[4])/6.0*(b-a)
    f2=(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12.0*(b-a)
    myerr=np.abs(f2-f1)
    if (myerr<tol):
        return (16.0*f2-f1)/15.0,myerr,neval
    else:
        mid=0.5*(b+a)
        f_left,err_left,neval_left=simple_integrate(fun,a,mid,tol/2.0)
        f_right,err_right,neval_right=simple_integrate(fun,mid,b,tol/2.0)
        neval=neval+neval_left+neval_right
        f=f_left+f_right
        err=err_left+err_right
        return f,err,neval
#################################################################################################################
#################################################################################################################



"""
takes: a function "func" and integrates between 2 values  "a" to "b"  using a recursive variable set size method
tolerance "toler" which determins precision to an extent 
You can also set to not warn you if it goes over count limit and gives up on recusion

returns 3 values: evaluated integral, aproximate error, number of function calls

**users should not enter anything for kwargs starting with "__" unless u are trying to break the code
** ->I know the __ doesn't do anything in a function but it works as an indicator to not touch them
 -> also I know starting variables with Caps is not the most pythonic way of doing thing but Im over it :) 
"""
def var_step_integrate(func,a,b,toler,__oldy =[],__counter = 0,warnOverCounter = True):
    #set the number of levels of recursion you can go before it gives up and tells you that u broke it
    #would not put this pass 100 unless u want to brink your computer for some time
    #if you go past the count cut off it will just return the best it has ignoring the required tolerance
    countcutoff= 10

    #N is the number of segments the function is cut into with the range of (a,b)
    #don't change N it will just break the code its just there for ease of reading
    N = 4
    dx = (b-a)/N #dx is width of segments
    #first time: calls function at all 5 points at the ends of segments 
    if __counter == 0:
        x = np.linspace(a,b,N+1)
        y = func(x)
        I1=(y[0]+4*y[2]+y[4])*(2*dx/3) #coarse numerical intigral using 3 points and simpson's rule
        nFuncEval = len(x) #counts number of function calls
    
    #inside recursion
    else:
        #now we use function call values from above layer of recursion for the coarse integral
        I1=(__oldy[0]+4*__oldy[1]+__oldy[2])*(2*dx/3)
        #here we make an array with the old function calls and the new func calls we need to make
        y = np.array([__oldy[0],func(a+dx),__oldy[1],func(b-dx),__oldy[-1]])
        #number of new function calls made is 2 within recursion
        nFuncEval = 2
        
    #fine Simpson's intigration with 5 points 
    I2=(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])*(dx/3)
    #y error comes from the difference between coarse and fine intigrals  
    y_err = np.abs(I1-I2)
    #adds 1 to recursion lvl counter
    __counter = __counter+1 
    
    #if error is within tolerance returns best integral with the error and the number of function calls 
    if y_err <toler:
        #combines both integrals to give best result
        I = (16.0*I2-I1)/15.0
        return I,y_err,nFuncEval
    
    if (__counter > countcutoff):
        if (warnOverCounter==True):
            print('well now you have gone too deep and broke it. this is why we can\'t have nice things')
            print('counter > counter cut off: counter cut off = {}'.format(countcutoff)) 
        I = (16.0*I2-I1)/15.0
        return I,y_err,nFuncEval
    #starts recursion when not in tolerance 
    else:
        #splits integral range in half and does the whole thing over again on left and right half
        #also transfers the already called func points to the next level of recusion
        midP = (a+b)/2
        I_L,y_err_L,nFuncEval_L = var_step_integrate(func,a,midP,toler/2,__oldy = y[:3], __counter = __counter,warnOverCounter= warnOverCounter)
        I_R,y_err_R,nFuncEval_R = var_step_integrate(func,midP,b,toler/2,__oldy = y[2:], __counter = __counter,warnOverCounter = warnOverCounter)
        #keeps track of the error and number of function calls from lower lvls of recusion
        y_err  = y_err_L +y_err_R
        nFuncEval = nFuncEval_L+nFuncEval_R
        I = I_L+I_R
        return I,y_err,nFuncEval
    
#compare function calls
myI = var_step_integrate(crazypoly,0,20,1e-3)
badI = simple_integrate(crazypoly,0,20,1e-3)
myCallCount = myI[2]
badCallCount = badI[2]
countdiff = badCallCount - myCallCount
print("For 5th order polynomial my integrator has {} function calls other integrator has {} function calls".format(myCallCount,badCallCount))
print("golly thats {} more function calls".format(countdiff))
      
    
#these are the E feild from a charged ring that we will integrate over    
def dE(theta,R,z,E0):
    dE = E0*np.sin(theta)*(z - R*np.cos(theta))/(((z-R*np.cos(theta))**2 + (R*np.sin(theta))**2)**(3/2.0))
    return dE
#fancy lambda calc defined dE so we can enter parameters and still have function input in var step integrator
def dE_setPara(R,z,E0):
    return lambda theta:  E0*np.sin(theta)*(z - R*np.cos(theta))/(((z-R*np.cos(theta))**2 + (R*np.sin(theta))**2)**(3/2.0))




#set radius R, distance from center z and E0 = k(Q_total)/2, k depends on your units and unit convensions so we will set it to one 
R = 1 
z = 1.5
E0 = 1



"""now comparing quad to my integrator
I split this into 2 for loops because my integrator is a little unpredictable with 
at R = z but it would be faster to put them in the same loop
"""
z = np.linspace(0,5,501) #array of z values 
Ezq = []
#errq = [] 
for x in z:
    Ezq.append(integrate.quad(dE,0,np.pi,args=(R,x,E0))[0])
    #errq.append(integrate.quad(dE,0,np.pi,args=(R,x,E0))[1]) #list of errors 


EzVstep = []
errVstep = [] 
for x in z:
    EzVstep.append(var_step_integrate(dE_setPara(R,x,E0),0,np.pi,1e-3,warnOverCounter=False)[0])
    #errVstep.append(var_step_integrate(dE_setPerm(R,x,E0),0,np.pi,1e-3)[1]) #list of errrors

plt.xlabel('distance fron center of spherical shell (z)')
plt.ylabel('Electric feild feild')
#plots the integral from "scipy.integrate.quad" over values in z 
plt.plot(z,Ezq,'*',label = 'quad' )
#plt.plot(z,errq,'*')# quad error plot    
#plots the integral my "var_step_integrate"  over values in z 
plt.plot(z,EzVstep,'.',label = 'my intergrator')
#plt.plot(z,errVstep,'.')#my var_step error plot
plt.legend()

print('\nplot shows E integral evaluated over z range from {} to {} at with R = {} and E0 = {}'.format(np.min(z),np.max(z),R,E0))
print('\n As you can see my integrator can\'t compute E(z=R) output at E(z=R) = {}'.format(EzVstep[100]))
print('There is a sigularity here but the clever people behind scipy.integrate have found a way to deal with that \n')