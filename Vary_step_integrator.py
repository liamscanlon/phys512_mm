# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 18:40:08 2019

@author: liams
"""

import numpy as np
import sys
import scipy.integrate as integrate
from matplotlib import pyplot as plt

def linear(x):
    return x

def sqr(x):
    return x**2
    
def crazypoly(x):
    return 5*x**4 + 3*x**3 -x**2 +58*x +55

def indefcrazypoly(x):
    I = x**5 + (3.0/4)*(x**4) - (1/3.0)*(x**3) + (58/2.0)*x**2 + 55*x
    return I
def defin_crazpoly(a,b):
    I = indefcrazypoly(b) - indefcrazypoly(a)
    return I 




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

returns 3 values: evaluated integral, aproximate error, number of function calls

**users should not enter anything for kwargs starting with "__" unless u are trying to break the code
** ->I know the __ doesn't do anything in a function but it works as an indicator to not touch them
 -> also I know starting variables with Caps is not the most pythonic way of doing thing but Im over it :) 
"""
def var_step_integrate(func,a,b,toler,__oldy =[],__counter = 0,warnOverCounter = True):
    #set the number of levels of recursion you can go before it gives up and tells you that u broke it
    #would not put this pass 100 unless u want to brink your computer for some time
    countcutoff= 10
    #throws Exception if recusion goes too deep

    #N is the number of segments the function is cut into with the range of (a,b)
    #don't change N it will just break the code its just there for ease of reading
    N = 4
    #dx is width of segments
    dx = (b-a)/N
    #first time: calls function at all 5 points at the ends of segments 
    if __counter == 0:
        x = np.linspace(a,b,N+1)
        y = func(x)
        #coarse numerical intigral using 3 points and simpson's rule
        I1=(y[0]+4*y[2]+y[4])*(2*dx/3)
        #number of function calls
        nFuncEval = len(x)
    
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
            print('counter>counter cut off: counter cut off = {}'.format(countcutoff)) 
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
    
    
#print( var_step_integrate(crazypoly,0,1,1e-10))
#print(integrate.quadrature(crazypoly,0,1))
#print(simple_integrate(crazypoly,0,1,1e-10))
#print(defin_crazpoly(0,1))
        
    
    
def dE(theta,R,z,E0):
    dE = E0*np.sin(theta)*(z - R*np.cos(theta))/(((z-R*np.cos(theta))**2 + (R*np.sin(theta))**2)**(3/2.0))
    return dE

def dE_setPerm(R,z,E0):
    return lambda theta:  E0*np.sin(theta)*(z - R*np.cos(theta))/(((z-R*np.cos(theta))**2 + (R*np.sin(theta))**2)**(3/2.0))




#set radius R, distance from center z and E0 = k(Q_total)/2 <-- depends on your units and unit convensions
R = 1 
z = 1.5
E0 = 1


#quad = integrate.quad(dE,0,np.pi,args=(R,z,E0))
#print(quad)
#myInte = var_step_integrate(dE_setPerm(R,z,E0),0,np.pi,1e-3)
#print(myInte)

z = np.linspace(0,5,501) #array of z values 
Ezq = []
#errq = [] 
for x in z:
    Ezq.append(integrate.quad(dE,0,np.pi,args=(R,x,E0))[0])
    #errq.append(integrate.quad(dE,0,np.pi,args=(R,x,E0))[1]) #list of errors 


#plots the integral from "scipy.integrate.quad" over values in z 
plt.plot(z,Ezq,'*')
#plt.plot(z,errq,'*')# quad error plot

EzVstep = []
errVstep = [] 
for x in z:
    EzVstep.append(var_step_integrate(dE_setPerm(R,x,E0),0,np.pi,1e-3,warnOverCounter=False)[0])
    #errVstep.append(var_step_integrate(dE_setPerm(R,x,E0),0,np.pi,1e-3)[1]) #list of errrors
    
    
#plots the integral my "var_step_integrate"  over values in z 
plt.plot(z,EzVstep,'.') #plots 
#plt.plot(z,errVstep,'.')#my var_step error plot


