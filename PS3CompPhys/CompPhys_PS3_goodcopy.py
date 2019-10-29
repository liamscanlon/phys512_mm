# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:34:00 2019

@author: liams
"""

import numpy as np
import camb
from matplotlib import pyplot as plt
import time


p0=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
data  = wmap[:,1]
poles = wmap[:,0]
err   = wmap[:,2]
 
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
#does what it says in the name
def calc_chisq(data,fit,err):
    chisq = np.sum(((data-fit)/err)**2)
    return chisq


#calcualtes the grad
def calc_J(p,d, n_points = 1199, func = get_spectrum):
    J = np.zeros([n_points,len(d)])
    for i in range(len(d)):
        np.put(p,i,p[i]+d[i])
        up = func(p)
        np.put(p,i,p[i]-2*d[i])
        down = func(p)
        dy_dpi = (up-down)/(2*d[i])
        J[:,i] = dy_dpi
    return J
#inelegant hack to hold a parameter: ih is the index of the parameter you want to hold ph is the value you want to hold it at 
def holdp_get_spectrum(ih=3,ph=0.05):
    holdp_spec = lambda p: get_spectrum(np.insert(p,ih,ph))
    return holdp_spec

        
def lm_fitter(data, err, p0,func = get_spectrum,max_runs=100):
    p = p0
    w = np.matrix(np.eye(len(err))*1/err**2)
    fit = func(p0)
    chisq = calc_chisq(data,fit,err)
    p = p0
    print('starting params gives chi^2 = {}'.format(chisq))
    #start with small lambda
    lamb = 0.1
    for i in range(max_runs):
        #standard LM code I tried doing this with normal np arrays since the matrix operations are being 
        #phased ou tin python but kept running into errors so back to the soon to be unsupported matrices
        J = np.matrix(calc_J(p,p/100,func=func))
        r = np.matrix((data- fit)).transpose()
        #covar matrix
        inCv = J.transpose() *w* J
        lhs = inCv + lamb*np.eye(inCv[0].size)*np.diag(inCv)
        rhs = J.transpose()*w*r
        dp  = np.linalg.inv(lhs) * rhs
        dp  = np.squeeze(np.array(dp))
        fit_new = func(p+dp)
        chisq_new  = calc_chisq(data,fit_new,err)
        d_chisq = chisq_new - chisq
        if d_chisq >= 0:
            lamb = 10*lamb
        else:
            lamb = lamb*0.2
            if d_chisq > -0.01:
                print('Convergence with params = {} \n chi^2 = {}'.format(p,chisq))
                break
        fit = fit_new
        p = p + dp
        chisq = chisq_new
        print('new params are: {} \n with chi^2 = {} \n change in chi^2 = {} \n lambda = {} '.format(p,chisq,d_chisq,lamb))
    return p, chisq


#starting params with the help param omitted 
p1=np.asarray([65,0.02,0.1,2e-9,0.96])


#ih=3
#ph=0.05
#fit_ps , chi_sq = lm_fitter(data,err,p1,func=holdp_get_spectrum(ih=ih,ph=ph),max_runs=100)
#fit_ps = np.insert(fit_ps,ih,ph)
#fit = get_spectrum(fit_ps)
#w = np.matrix(np.eye(len(err))*1/err**2)
#J = np.matrix(calc_J(fit_ps,p0/100))
##computes covarience matrix
#Cv = np.linalg.inv(J.transpose() *w* J)
#Cv = np.asarray(Cv)
#np.savetxt('CoVarM.txt',Cv)
##in theory this is the errors on the fit 
#p_errs = np.diag(Cv)
##plots fit over data
##plt.xlabel('$l$')
##plt.plot(poles,data,label = 'data')
##plt.plot(poles,fit , label = 'fit')
##plt.legend()
#np.savetxt('CoVarM.txt',Cv)

#generates random step for mcmc for given covar matrix
def rand_step(Cv):
    chosky = np.linalg.cholesky(Cv)
    #here I hacked it to have a lower step size for tau because there seems to be a local 
    #min at tau lower that we are looking for and I dont feel like wasting more time running chains
    return np.asarray(np.dot(chosky,np.random.randn(Cv.shape[0])))#*np.array([1,1,1,1/6,1,1]))
    #return [6.5e-01, 2.0e-04, 1.0e-03, 5.0e-04, 2.0e-11, 9.6e-03]*np.random.randn(Cv.shape[0])/10


def mcmc(data,err,p0,Cv,n_steps=100,step_scale= 0.02,chainname='Chain_',saveAfter = 50):
    #t is has the local time info for naming saved chains 
    t = time.localtime()
    #makes sure the covar matrix is an array
    Cv = np.asarray(Cv)
    #get the number of parameters we are trying to fit
    n_pars = p0.size
    #set up array to hold params from the chain
    pars = np.zeros([n_steps,n_pars])
    #first entry is the starting guess
    pars[0,:] = p0
    #set up array to hold the chisq values
    chisqs = np.zeros([n_steps])
    chisq_now = calc_chisq(data,get_spectrum(p0),err)
    #sets the starting params and chisq for the loop from p0 fit
    chisqs[0] = chisq_now
    p_now     = p0
    failcount = 0 
    
    #loop for MCMC
    for i in range(1,n_steps):
        print('step # = {} out of {}'.format(i,n_steps))
        #calls the random step function and adds to current params 
        #after being scaled by scaling factor to get the new "test' params
        p_test     = p_now + step_scale*rand_step(Cv)
        #prints the current test params
        #print('test p = {}'.format(p_test))
        #catches any exptions from the camb and sets that guess to be a "bad" step if it can't
        #compute a fit for the given test params
        try:
            chisq_test = calc_chisq(data,get_spectrum(p_test),err)
            d_chisq    = chisq_test-chisq_now
        except:
            print('error step not accepted')
            d_chisq = 1E6
            accept_test = False
        print('chi^2 diff = {}'.format(d_chisq))
        #throws out any guess that have tau less that 0
        if p_test[3] < 0:
            accept_test = False
        else:
            #if chi^2 is improved (goes down) from current params step is set to be accepted
            if d_chisq<0:
                accept_test = True
            else:
                #if chi^2 is not imporved (goes up) from current params probability of step 
                #being accepted is determined by boltzmann stats
                prob = np.exp(-0.5*d_chisq)
                if np.random.rand()<prob:
                    accept_test = True
                else:
                    #counts how many failed steps
                    #if there are too many and youre late in the chain will stop the chain 
                    #because the step size is probs to big
                    accept_test = False
                    failcount = failcount + 1
                    if (failcount/i > 0.87 ) and i>n_steps*(0.667):
                        print('step size probs too big...')
                        #changes the name of the soon to be saved chain 
                        #to indicate that the chain was cut short
                        chainname = chainname+'cutShort'
                        break
        #prints the acceptance status of current test step
        print('step accepted = {}'.format(accept_test))
        print('current accept rate = {}  look for 0.25'.format(1 - failcount/(i)))
        #sets the current step to the test step if it is accepted
        if accept_test==True: 
            p_now     = p_test
            chisq_now = chisq_test
        #adds accepted param  to params array
        pars[i,:] = p_now
        #^^^ like this but for the chisq
        chisqs[i] = chisq_now
        #saves after some number of interations (in case somthing crashes) 
        if i % (saveAfter) == 0:
            print('****interm save point****')
            #saves the params and chisq arrays to a txt file 
            np.savetxt(genFileName(t,step_scale,chainname=chainname, pORc='p'),pars[:i,:])
            np.savetxt(genFileName(t,step_scale,chainname=chainname,pORc='c'),chisqs[:i])
            #plt.plot(pars[:i,:])
        print('*************************************')
    #saves the chains param's and chisqs
    print('done, saving to files...')        
    np.savetxt(genFileName(t,step_scale,chainname=chainname,pORc='p'),pars)
    np.savetxt(genFileName(t,step_scale,chainname=chainname,pORc='c'),chisqs)
    #returns the chain's params and chisqs
    acceptrate = 1- failcount/(n_steps-1)
    print('final accept rate  = {} \n ideal accept rate is 0.25'.format(acceptrate))
    return pars,chisqs


#Generates file name for saves params and chisqs from MCMC chains with the date,time,name of the chain and step size
def genFileName(t,step_scale,chainname = '',  pORc = 'p'):
    if pORc == 'p':
        fn = chainname +'_params_{}_{}_{}_{}_scale{}.txt'.format(t[2],t[3],t[4],t[5],str(step_scale).replace('.','point'))
    elif pORc == 'c':
        fn =  chainname+'_chisqs_{}_{}_{}_{}_scale{}.txt'.format(t[2],t[3],t[4],t[5],str(step_scale).replace('.','point'))
    else:
        fn =  chainname+'_Unknown_{}_{}_{}_{}_scale{}.txt'.format(t[2],t[3],t[4],t[5],str(step_scale).replace('.','point'))
    return fn

#loads the covar matrix so  from lm fit so I dont need to do it every time
fit_CoVarM = np.loadtxt('CoVarM.txt')

#
lastchain = np.loadtxt('bettertry_params_26_18_52_58_scale0point016.txt')
lm_fit_p = lastchain[0,:]

chainname = 'verybigsteps'
step_scale = 0.12
n_steps = 500
saveAfter = 50


mcmcpars , mcmcchisqs = mcmc(data,err,p0=lm_fit_p,Cv=fit_CoVarM,n_steps=n_steps,step_scale= step_scale,chainname=chainname,saveAfter=saveAfter)


#plt.plot(mcmcpars[:,0],mcmcpars[:,2],'*')

mcmcpars2 , mcmcchisqs2 = mcmc(data,err,p0=mcmcpars[-1,:],Cv=fit_CoVarM,n_steps=n_steps,step_scale= step_scale,chainname=chainname,saveAfter=saveAfter)



mcmcpars3 , mcmcchisqs3 = mcmc(data,err,p0=p0,Cv=fit_CoVarM,n_steps=n_steps,step_scale= step_scale,chainname=chainname,saveAfter=saveAfter)
mcmcpars4 , mcmcchisqs4 = mcmc(data,err,p0=mcmcpars[-1,:],Cv=fit_CoVarM,n_steps=n_steps,step_scale= step_scale,chainname=chainname,saveAfter=saveAfter)

mcmcpars5 , mcmcchisqs5 = mcmc(data,err,p0=p0,Cv=fit_CoVarM,n_steps=n_steps,step_scale= step_scale,chainname=chainname,saveAfter=saveAfter)

mcmcpars6 , mcmcchisqs6 = mcmc(data,err,p0=mcmcpars5[25,:],Cv=fit_CoVarM,n_steps=n_steps,step_scale= step_scale,chainname=chainname,saveAfter=saveAfter)
mcmcpars6 , mcmcchisqs6 = mcmc(data,err,p0=mcmcpars5[25,:],Cv=fit_CoVarM,n_steps=n_steps,step_scale= step_scale,chainname=chainname,saveAfter=saveAfter)

mcmcpars7 , mcmcchisqs7 = mcmc(data,err,p0=lm_fit_p,Cv=fit_CoVarM,n_steps=n_steps,step_scale= step_scale,chainname=chainname,saveAfter=saveAfter)

plt.plot(np.abs(np.fft.fft(mcmcpars4[:,0])),label = 'H0 normed')




plt.plot(mcmcpars5[:,0]/(mcmcpars4[0,0]),label = 'H0 normed')
plt.plot(mcmcpars5[:,1]/(mcmcpars4[0,1]),label = 'wb normed')
plt.plot(mcmcpars5[:,2]/(mcmcpars4[0,2]),label = 'wc normed')
plt.plot(mcmcpars5[:,3]/(mcmcpars4[0,3]),label = 'T normed')
plt.plot(mcmcpars5[:,4]/(mcmcpars4[0,4]),label = 'As normed')
plt.plot(mcmcpars5[:,5]/(mcmcpars4[0,5]),label = 'ns normed')
plt.legend()





pmcmcfit4=np.array([np.mean(mcmcpars4[:,0]),np.mean(mcmcpars4[:,1]),np.mean(mcmcpars4[:,1]),np.mean(mcmcpars4[:,2]),np.mean(mcmcpars4[:,3]),np.mean(mcmcpars4[:,4]),np.mean(mcmcpars4[:,5])])
#np.savetxt(genFileName(time.localtime(),step_scale,chainname=chainname,pORc='p'),mcmcpars)



#chosky = np.linalg.cholesky(Covar)
#print(np.asarray(np.dot(chosky,np.random.randn(Covar.shape[0]))))

