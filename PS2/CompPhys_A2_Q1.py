# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:33:38 2019

@author: liams
"""

import numpy as np 
import matplotlib.pyplot as plt  


"""
normally I would comment more within functions but since none of the code  in-class or in the tutorals is 
properly commented it seems like I have already done more than I need to so I'll leave like this  
"""



"""
finds Chebyshev polynomial of with intail order "order". fit parameters for a function "func" number of 
function evaluation points is set by n_points, range of the function over which 
you want to fit is set by a "scale" array with the 0th element being the lower bound 
and the 1st element being the upper bound (by default this is set to [-1,1] ). 
Accuracy of the fit is set by the "acc" parameter coefficient less than this cut off will be truncated 
minimizing the order of the fitting polynial within the allowed accuracy cut off.

returns the Chebyshev polynomial coefficient of the least squares fit. 
Note: if scale is set to values other than the default the x values will need to be rescaled properly (as is done with functions below)
also returns the scale array if returnscale is set to True.

"""
def chebModeler(func,order,n_points,scale=[-1,1],acc = 1E-6,returnscale = False):
    A,x  = genChebMatrix(n_points, order)
    if (scale!=[-1,1]):
        x_scaled = scaler(x,scale) 
    else: 
        x_scaled = x
    d = func(x_scaled)
    U,S,Vt = np.linalg.svd(A, full_matrices=False)
    V = np.transpose(Vt)
    m = V@np.diag(np.reciprocal(S))@np.transpose(U)@d
    m = m[ (np.abs(m) >= acc)]
    if returnscale == True:
        return m,scale
    return m
 
    


"""
returns point(s) from fit coeffients m (from chebModler function)
for point(s) given by xn 
fit range given by "scale" array
"""    
def getChebPoint(xn,m,scale=[-1,1]):    
    yn = genChebMatrix_givePoints(descaler(xn,scale),len(m)-1)@m
    return yn


"""
takes x values from the standard chebshev range (i.e [-1,1]) and rescales to desires range
given by "scale" array 
"""
def scaler(x,scale):
    a = scale[0]
    b = scale[1]
    x_scaled=((b-a)/2)*x + (a+b)/2
    return x_scaled

"""
takes x values in "scale" range and rescales to the schebshev range
"""
def descaler(x,scale):
    a = scale[0]
    b = scale[1]
    x_descaled = 2/(b-a)*x -(a+b)/(b-a) 
    return x_descaled

"""
Generates an n by m matrix of Chebyshev polynomals evaluated at values between -1 and 1
M_ij = T_j(x_i)

where T_j is the jth order chebyshev polynomial 
and x_i's are equally spaced values between -1 and 1 

number of rows= number of x_i points polynomials are evaluated at = "n_points"
number of columns = highest order of polymials equaluated = "order"
"""
def genChebMatrix(n_points, order):
    mtrx = np.zeros([n_points, order+1])
    x_points = np.linspace(-1,1,n_points)
    mtrx[:,0] = 1
    if order==0:
        return mtrx,x_points
    mtrx[:,1] = x_points
    for n in range(1,order):
        mtrx[:,n+1] = 2*x_points*mtrx[:,n]-mtrx[ :,n-1]
    return mtrx,x_points



"""similar to genChebMatrix() but x_i values are set with the sx array"""
def genChebMatrix_givePoints(xs, order):
    if (type(xs)==float) or (type(xs)==int):
        mtrx = np.zeros([1, order+1])
    else:
        mtrx = np.zeros([len(xs), order+1])
    mtrx[:,0] = 1
    if order==0:
        return mtrx,xs
    mtrx[:,1] = xs
    for n in range(1,order):
        mtrx[:,n+1] = 2*xs*mtrx[:,n]-mtrx[ :,n-1]
    return mtrx


"""
With fit coeffients m (from chebModler function) 
returns xs: "n_point" # of points within the fitting range "scale" 
and ys: the corsponding points from the fit
"""
def chebModlePoints(m,n_points,scale):
    cheb_polyarray,xs = genChebMatrix(n_points, len(m)-1)
    ys = cheb_polyarray@m
    xs = scaler(xs,scale)
    return xs,ys



"""calculates root mean squared of elements in an array"""
def rms(y):
    vrms = np.sqrt(np.mean(y**2))
    return vrms



#Question 1
n_funcEval = 50
order = 20
scale = [0.5,1]
n_plotPoints = 50
xn = 0.7



#calculates the best fit parameters as Question 1 asks
m = chebModeler(np.log2,order,n_funcEval,scale)

#checks that fit point and true vale are close
yn = getChebPoint(xn,m,scale)
print('at x={} cheb fit gives {} and true value is {}'.format(xn,yn,np.log2(xn)))

#xs is the same points used to generate the fit, ys are the points from best fit
xs,ys= chebModlePoints(m,n_plotPoints,scale)


#calculates residuals of the chebyshev fit
eys_cheb = np.log2(xs)-ys
#finds largest deviation from the true value
emax_cheb = np.max(np.abs(eys_cheb))

#does normal polynomial fit
x_points = np.linspace(scale[0],scale[1],n_funcEval)
coefs = np.polyfit(x_points,np.log2(x_points),len(m)-1)
p = np.poly1d(coefs)

#calculates residuals and max error
eys_poly = np.log2(xs)-p(xs)
emax_poly = np.max(np.abs(eys_poly))

#plots resuduals
plt.plot(x_points,eys_cheb,'*',label = 'Chebyshev poly fit residuals')
plt.plot(x_points,eys_poly,'.',label = 'normal poly fit residuals')
plt.legend()


print('chebyshev max error = {} \normal polynomial max error = {}'.format(emax_cheb,emax_poly))
print('Chebyshev has lower max error')
print('chebyshev rms error = {} \normal polynomial rms error = {}'.format(rms(eys_cheb),rms(eys_poly)))
print('but normal poly has lower rms error')

"""
models log base 2 of a value or array of value "x" fits  can set number of function values to fit with with N-funcEval 
by defualt it is set to 55.
and yes I know I spelled model 
"""
def log2model_fullRange(x,n_funcEval=55):
    mantis,expo = np.frexp(x)
    scale = [np.min(mantis),np.max(mantis)]
    m = chebModeler(np.log2,18,n_funcEval,scale)
    A = genChebMatrix_givePoints(descaler(mantis,scale),len(m)-1)
    y = A@m + expo 
    return y


"""this plots fits vs true value but you dont really see much difference so its better to look at residuals"""
#plt.plot(xs,ys,label = 'my model')
#plt.plot(xs,np.log2(xs),label = 'true values')
#plt.plot(xs ,p(xs),label = 'normal poly fit')
#plt.legend()


"""
this shows off how to use the log2modle_fullRange() function I want just the residuals plot to pop up so im going to 
comment out the ploting part but feel free to test uncomment to test it out. 
also I ploted the error
"""
#x = np.linspace(1,100,200)
#y = log2model_fullRange(x,20)
#plt.plot(x,y)
#plt.plot(x,(np.log2(x)-y))
