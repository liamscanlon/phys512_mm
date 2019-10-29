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


chain1 = np.loadtxt('bettertry_params_26_18_52_58_scale0point016.txt') , np.loadtxt('bettertry_chisqs_26_18_52_58_scale0point016.txt')
chain2 = np.loadtxt('bigsteps_params_27_0_58_19_scale0point1.txt') , np.loadtxt('bigsteps_chisqs_27_0_58_19_scale0point1.txt')
chain3 = np.loadtxt('bigsteps_params_27_1_22_34_scale0point1.txt') , np.loadtxt('bigsteps_chisqs_27_1_11_53_scale0point1.txt')
chain4 = np.loadtxt('bigsteps_params_27_1_11_53_scale0point1.txt') , np.loadtxt('bigsteps_params_27_1_11_53_scale0point1.txt')





checkp = 2

plt.plot(chain1[0][:,checkp])
plt.plot(chain2[0][:,checkp])
plt.plot(chain3[0][:,checkp])
plt.plot(chain4[0][:,checkp])




for i in range(6):
    plt.plot(chain1[0][:,i]/chain1[0][0,i],label = str(i))
plt.legend()