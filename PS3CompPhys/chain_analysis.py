# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:22:41 2019

@author: liams
"""

import numpy as np
import camb
from matplotlib import pyplot as plt
import time


def get_spectrum(pars,lmax=2000):
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[2:1201,0]    #gives points that match the data we have
    return tt

p0=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
data  = wmap[:,1]
poles = wmap[:,0]
err   = wmap[:,2]


chain1 = np.loadtxt('long_params_2_18_22_12_scale[0point6   0point525 0point6   0point45  0point525 0point525].txt') , np.loadtxt('long_chisqs_2_18_22_12_scale[0point6   0point525 0point6   0point45  0point525 0point525].txt')
chain2 = np.loadtxt('c_params_2_19_33_13_scale[0point54   0point468  0point54   0point387  0point477  0point4725].txt') , np.loadtxt('c_chisqs_2_19_33_13_scale[0point54   0point468  0point54   0point387  0point477  0point4725].txt')
chain3 = np.loadtxt('d_params_2_21_58_16_scale[0point6   0point52  0point6   0point27  0point525 0point525].txt') , np.loadtxt('d_chisqs_2_21_58_16_scale[0point6   0point52  0point6   0point27  0point525 0point525].txt')
chain4 = np.loadtxt('e_params_3_0_49_7_scale[0point54  0point48  0point54  0point37  0point477 0point473].txt') , np.loadtxt('e_chisqs_3_0_49_7_scale[0point54  0point48  0point54  0point37  0point477 0point473].txt')
chain5 = np.loadtxt('f_params_3_14_26_7_scale0point5.txt') , np.loadtxt('f_chisqs_3_14_26_7_scale0point5.txt')
chain6 = np.loadtxt('g_params_3_15_33_6_scale0point35.txt') , np.loadtxt('g_chisqs_3_15_33_6_scale0point35.txt')
chain7 = np.loadtxt('h_params_3_15_37_32_scale0point35.txt') , np.loadtxt('h_chisqs_3_15_37_32_scale0point35.txt')




"""
The idea here is to weight a chain you already have with the new known tau with error
first you updata the chi^2 using the new tau with error 
if the updated chi^2 is better you keep it if not you do the same probability thing we did for the normal mcmc chain
"""
def update_chisqs(chainp3s,chain_chis):
    p3 =  0.0544
    ep3 = 0.0073
    newchis = chain_chis + ((chainp3s-p3)/ep3)**2
    return newchis


def updatechain(newchis,oldchis):
    keep = [True] * len(newchis)
    keep = np.asanyarray(keep)
    for i in range(1,len(newchis)):
        dchi = newchis[i]-oldchis[i]
        if dchi>0:
            prob = np.exp(-0.5*dchi)
            if np.random.rand()>prob:
                keep[i] = False
    return keep


oldchis = chain7[1]
newchis = update_chisqs(chain7[0][:,3],oldchis)
keeparray = updatechain(newchis,oldchis)
newchain   = np.delete(chain7[0],np.where(keeparray == False),axis=0)

#here is the fit with the updated chain also the 
newfit = get_spectrum(np.mean(newchain,axis=0))          
newfir_chisq = np.sum(((data-newfit)/err)**2)
newfit_ps = np.mean(newchain,axis=0)
newfit_perr = np.std(newchain,axis=0)

##this plots the chain paramas divided by the starting value

plt.plot(chain7[0][:,0]/(chain7[0][0,0]),'.',label = 'H0 normed')
plt.plot(chain7[0][:,1]/(chain7[0][0,1]),'*',label = 'wb normed')
plt.plot(chain7[0][:,2]/(chain7[0][0,2]),'^',label = 'wc normed')
plt.plot(chain7[0][:,3]/(chain7[0][0,3]),'<',label = 'T normed')
plt.plot(chain7[0][:,4]/(chain7[0][0,4]),'>',label = 'As normed')
plt.plot(chain7[0][:,5]/(chain7[0][0,5]),'o',label = 'ns normed')
plt.legend()
