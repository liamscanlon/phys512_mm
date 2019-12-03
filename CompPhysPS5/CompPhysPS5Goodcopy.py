# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 14:47:51 2019

@author: Liam Scanlon
"""

import numpy as np 
from matplotlib import pyplot as plt
from scipy.ndimage.filters import convolve as con
import time




def circle2Mask(r,n,a,mask):
    m = mask
    dn = a/n
    for i in range(n):
        for j in range(n):
            if((i*dn-0.5*a)**2 +(j*dn-0.5*a)**2 )<=(r)**2:
                m[i,j]=True


def circle2Mask_p2(r,n,a,mask):
    m = mask
    dn = a/n
    for i in range(n):
        for j in range(n):
            if((i*dn-0.5*a)**2 +(j*dn-0.5*a)**2 )<=(r+2*dn)**2:
                m[i,j]=True

#addes True values to a mask on the edges
def edge2mask(mask):
    mask[0,:] = True
    mask[:,-1] = True
    mask[-1,:] = True
    mask[:,0] = True

"""
sumNN = "sum of nearest neighbour"
takes n by n array V 
returns new array where each point [i,j] corisponds to the sum 
of the neast neighbours of V[i,j]

I tried doing this in many dumb ways before settling on this all seem to give 
the same result this one is a little cleaner imho
"""
def sumNN(V):
    k = np.array([[0,1,0],[1,0,1],[0,1,0]])
    return con(V,k)

"""
this is a glofied divide by 4 function to the sumNN() function but 
for clearity of code and since this comes up so much in the prolem and 
so I dont get confused with factors of 4 everywhere 
I gave it its own name"""
def smoother(V):
    return (1/4.0)*sumNN(V)

#laplace operator
def D2(V):
    k = -np.array([[0,1,0],[1,-4,1],[0,1,0]])
    return con(V,k)


def relaxMethod(n,r,a,nsteps= 0,maskFunc = circle2Mask,dvmin = 0.0001,showplot = False):
    if nsteps ==0:
        nsteps = 5*n
    #mask for the circle at const. V
    cmask = np.zeros([n,n],dtype=bool)
    #mask for edges
    emask = np.zeros([n,n],dtype=bool)
    
    maskFunc(r,n,a,cmask)
    edge2mask(emask)
    bc = np.zeros([n,n])
    bc[cmask] = 1
    bc[emask] = 0
    V = bc.copy()
    for i in range(nsteps):
        V_new = smoother(V)
        V_new[emask],V_new[cmask] = 0, 1
        dv = np.sum(np.abs(V_new - V))
        V = V_new
        if showplot == True:
            plt.clf();
            plt.imshow(V)
            plt.colorbar()
            plt.pause(0.0001)
        if dv < dvmin:
            print("converged at interation"+ str(i))
            break
    rho = -D2(V[1:-1,1:-1])
    return bc,V,rho


""""
ConjGrad TIME!!
"""


#Ax helper function for conjgrad
def Ax(V,cmask,emask):
    v = V.copy()
    v[cmask] = 0
    temp = smoother(v)
    temp[emask] = 0
    ax = temp-V
    return ax

#conjGrad method
def conjGradMethod(n,r,a,nsteps = 0, residcutoff = 0.00000001,showplot=False):
    if nsteps == 0:
        nsteps = 2*n
    cmask = np.zeros([n,n],dtype = bool)
    emask = np.zeros([n,n],dtype = bool)
    circle2Mask(r,n,a,cmask)
    edge2mask(emask)
    bc = np.zeros([n,n])
    bc[cmask] = 1 
    V = 0*bc 
    b = -smoother(bc)
    b[emask] = 0 
    resid = b - Ax(V,cmask,emask)
    p = resid.copy()
    for k in range(n):
        Ap = Ax(p,cmask,emask)
        rtr=np.sum(resid*resid)
        if rtr <= residcutoff:
            print('CG convergence after {} interations'.format(k))
            break
        alpha=rtr/np.sum(Ap*p)
        V=V+alpha*p
        resid_new=resid-alpha*Ap
        beta=np.sum(resid_new*resid_new)/rtr
        p=resid_new+beta*p
        resid=resid_new
        if showplot == True:
            plt.clf();
            plt.imshow(resid)
            plt.colorbar()
            plt.pause(0.001)
    rho = -D2(V[1:-1,1:-1])
    return bc,V,rho


"""
Now with varrying resolution
"""
#takes an intial guess for V and does the ConjGrad method
def conjGrad_V0guess(n,r,a,V0,nsteps = 0, residcutoff = 0.00000001,maskFunc = circle2Mask ,showplot=False ):
    if nsteps == 0:
        nsteps = 10*n
    cmask = np.zeros([n,n],dtype = bool)
    emask = np.zeros([n,n],dtype = bool)
    maskFunc(r,n,a,cmask)
    edge2mask(emask)
    bc = np.zeros([n,n])
    bc[cmask] = 1 
    V = V0.copy()
    b = -smoother(bc)
    b[emask] = 0 
    resid = b - Ax(V,cmask,emask)
   # plt.imshow(Ax(V,mask,emask))
    p = resid.copy()
    for k in range(nsteps):
        Ap = Ax(p,cmask,emask)
        rtr=np.sum(resid*resid)
        if rtr <= residcutoff:
            print('CG convergence after {} interations'.format(k))
            break
        alpha=rtr/np.sum(Ap*p)

        #should add a condition to stop loop if alpha is small enough
        V=V+alpha*p
        resid_new=resid-alpha*Ap
        beta=np.sum(resid_new*resid_new)/rtr
        p=resid_new+beta*p
        resid=resid_new
        if showplot == True:
            plt.clf();
            plt.imshow(resid)
            plt.colorbar()
            plt.pause(0.001)
    return V


# this essentually splits every pixel 4 pixels with the same value
def doubleGridRes(lr):
    ni =lr.shape[0]
    nf = 2*ni
    hr = 5*np.ones([nf,nf])
    for i in range(ni):
        for j in range(ni):
            hr[2*i:2*(i+1),2*j:2*(j+1)] = lr[i,j]
    return hr
"""
n0 is the initial resolution nres is the number of times you want to double n0 maskfunc let you put different shaped conductor in the box
with a bool mask of the shape
"""
def conjGrad_varRes(n0 ,nres,r,a, maskFunc = circle2Mask ,showplot = False ,residcutoff = 0.000001):
    V = np.zeros([n0,n0])
    n = n0
    for i in range(nres):
        print('the current resolution is {} by {}'.format(n,n))
        V =  conjGrad_V0guess(n,r,a,V,maskFunc=maskFunc,showplot = showplot,residcutoff = residcutoff)
        if i==nres-1:
            break
        V = doubleGridRes(np.transpose(V))
        n = 2*n
    rho = -D2(V[1:-1,1:-1])
    return V , rho
""""
Time to lumpify the circle 
"""

#makes mask a lumpy circle
def lumpifyCirc(r,n,a,mask,lumpFactor=0.2):
    m = mask
    dn = a/n
    for i in range(n):
        for j in range(n):
            if((i*dn-0.5*a -r)**2 +(j*dn-0.5*a)**2 )<=(r*lumpFactor)**2:
                m[i,j]=True
#makes mask for the lump to be added to circle mask
def lumpCircle2mask(r,n,a,mask):
    circle2Mask(r,n,a,mask)
    lumpifyCirc(r,n,a,mask)
    
#calculates the gradaint and the magnitude of the grad. usewd for the electric feild calculations
def grad(V):
    kx = (1/2)*np.array([[0,0,0],[1,0,-1],[0,0,0]])
    ky = (1/2)*np.array([[0,1,0],[0,0,0],[0,-1,0]])
    Ex = con(V,kx)
    Ey = con(V,ky)
    Emag = np.sqrt(Ex**2+Ey**2)
    return Emag, np.array([Ex,Ey])




"""
5)
Heating Wall in Box


As with most physics we can invoke symmetry and make everything easier
only need to look along the the center since solution along x is indepent
of y

trying to solve:
    d^2T(x,t)/dx^2  =a * dT(x,t)
    to make this discrete:
    dT/dt -> (T(t+dt) -T(t))/dt
    d^2T(x)/dx^2 -> (T(x+dx) + T(x-dx) - 2*T(x))/dx^2
    can rearange a bit:
        (T(t+dt) -T(t))/dt = a* (T(x+dx) + T(x-dx) - 2*T(x))/dx
        (T(t+dt) -T(t))= (a*dt/dx^2) (T(x+dx) + T(x-dx) - 2*T(x))
        T(t+dt) = T(t) + (a*dt/dx^2) (T(x+dx) + T(x-dx) - 2*T(x))
        T(x,t+dt) = T(x,t) + (a*dt/dx^2) (T(x+dx,t) + T(x-dx,t) - 2*T(x,t))
    
    
    Temp of wall is linear with time: 
        T_wall(t) = T(0,t) = g * t
    



"""



#dx is size x units, dt is size of time steps, n_*step's are the number of x or t steps k related to thermal conductivity, g is rate the T increases
def heatSolver(dx,dt,n_tstep,n_xstep,k,g):
    x = np.arange(0,n_xstep*dx,dx)#set up the x points
    t = np.arange(0,n_tstep*dt,dt)#set up times 
    T = np.zeros([n_xstep,n_tstep])#this will hold the Temp at a given x anf t
    for i in range(n_tstep-1):
        for j in range(n_xstep-1):
            T[0,i] = g*t[i]
            T[j,i+1] =T[j,i] + (k*dt/dx**2)*(T[j+1,i]+T[j-1,i]-2*T[j,i])
        
    return x,t,T


#physical unit stuff 
a = 100#sidelength in physical units (e.g. m, mm , km)
r = 20#radius in physical units 
#"grid" unit stuff
#side length in grid units
n = 2**9# this is a power of 2 because it makes easier to compare the var res part easier since it can only deal with doubled resolutions. 




"""
I have all the code for the calculations commented because it seems like everyone is working in vastly differnet 
environments so please just uncomment as needed I have plots in the folder which Im sure you have seen by now so it should not 
be that needed
"""


"""
1) getting linear charge dencity for relaxation method:
since relaxation method takes so long I will use a pretty low grid resolution
"""
##setting relaxation method grid size:
#nr= 120
#bc_r, V_r, rho_r = relaxMethod(nr,r,a,showplot=True,dvmin = 0.00001)
#
#chargemask =  np.zeros([nr-2,nr-2],dtype =bool)
#circle2Mask(r+ a/nr , nr,a , chargemask)
#lamb = np.sum(rho_r[chargemask])
#print(lamb)
#
#
#
#plt.figure();plt.plot((a/nr)*(np.arange(len(V_r[nr//2,:]))+1),V_r[nr//2,:])
#x = np.arange(a/2+r,a,30/36)
#xx = np.arange(0,a/2-r,30/36)
#V_theory = (lamb/(2*np.pi))*np.log(xx)+1
#plt.plot(x,V_theory)
#





"""
2) ConjGrad 
"""
#t = time.time()
#bc_r, V_r, rho_r = relaxMethod(nr,r,a,dvmin = 0.00001)
#print('realaxation method takes {}s for {} by {} with residualcutoff of {} '.format(time.time()-t, nr,nr,0.00001) )
#t = time.time()
#bc_cj, V_cj, rho_cj = conjGradMethod(nr,r,a,residcutoff= 0.00001)
#print('ConjGrad method takes {}s for {} by {} with residualcutoff of {} '.format(time.time()-t, nr,nr,0.00001) )
#
#""""
#


"""
3) variable resolution ConjGrad
"""
#
#t = time.time()
#bc_cj, V_cj, rho_cj = conjGradMethod(n,r,a,residcutoff = 0.00001)
#print('standard CG method takes {}s for {} by {} with residual cutoff of {} '.format(time.time()-t, n,n,0.00001) )
#t = time.time()
#V_cj_var, rho_cj_var = conjGrad_varRes( 1 , 9, r , a , residcutoff= 0.00001)
#print(' variable res ConjGrad method takes {}s for {} by {} with residual cutoff of {} '.format(time.time()-t, n,n,0.00001) )
#




"""
4) lumpy wire
"""

#V_lump, rho_lump = conjGrad_varRes( 1 , 9, r , a , residcutoff= 0.00001)
#V_nolump, rho_nolump = conjGrad_varRes( 1 , 9, r , a ,maskFunc=lumpCircle2mask, residcutoff= 0.00001)
#
#Emag_lump = grad(V_lump)[0]
#Emag_nolump = grad(V_nolump)[0]
#
#
#plt.figure();plt.imshow(Emag_lump);plt.colorbar()
#plt.figure();plt.imshow(Emag_nolump);plt.colorbar()
#
#
#Emax_lump = np.max(Emag_lump)
#Emax_nolump = np.max(Emag_nolump)
#print("max electric feild on smooth wire = {}".format(Emax_nolump))
#print("max electric feild on lumpy wire = {}".format(Emax_lump))
#


"""
5) for real this time: 
heat equation stuff

"""

#
##these need some tuning to get it to work
##the 2 regimes  will relate to steady state condition when dT/dt pretty much zeros
#dt = 0.00005 #time step size
#dx = 0.0005#
#n_tstep=2000
#n_xstep = 50
#k = 0.001#constant
#g = 0.01
#
#
#
#x,t,T = heatSolver(dx,dt,n_tstep,n_xstep,k,g)
#plt.figure();plt.imshow(np.transpose(T),aspect = 'auto');plt.colorbar()


