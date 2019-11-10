# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:34:00 2019

@author: liams
"""




"""
I really cant express how much I hate everything about this
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
        newp = p+dp
       
#        if newp[3] <= 0:
#            lamb = 10*lamb
#            continue
        fit_new = func(newp)
        chisq_new  = calc_chisq(data,fit_new,err)
        d_chisq = chisq_new - chisq
        if d_chisq >= 0:
#            if lamb < 5E4:
#                lamb = 1
            lamb = 10*lamb
        else:
            lamb = lamb*0.2
            if d_chisq > -0.09:
                print('Convergence with params = {} \n chi^2 = {}'.format(newp,chisq_new))
                fit = fit_new
                p = newp
                chisq = chisq_new
                break
        fit = fit_new
        p = newp
        chisq = chisq_new
        print('new params are: {} \n with chi^2 = {} \n change in chi^2 = {} \n lambda = {} '.format(p,chisq,d_chisq,lamb))
    return p, chisq


#starting params with the held param omitted 
p1=np.asarray([65,0.02,0.1,2e-9,0.96])


ih=3
ph=0.05
fit_ps_holdT , chi_sq_holdT = lm_fitter(data,err,p1,func=holdp_get_spectrum(ih=ih,ph=ph),max_runs=100)
fit_ps_holdT = np.insert(fit_ps_holdT,ih,ph)
lmFit_holdT  = get_spectrum(fit_ps_holdT)
w = np.matrix(np.eye(len(err))*1/err**2)
J_holdT = np.matrix(calc_J(fit_ps_holdT,p0/200))
#computes covarience matrix
Cv_holdT = np.linalg.inv(J_holdT.transpose() *w* J_holdT)
Cv_holdT = np.asarray(Cv_holdT)
p_errs_holdT = np.diag(Cv_holdT)
print('held tau fit give parameters of \n  {} \n with errors of {} \n chi^2 = {}'.format(fit_ps_holdT,p_errs_holdT,chi_sq_holdT))



fit_ps , chi_sq = lm_fitter(data,err,fit_ps_holdT,func=get_spectrum,max_runs=100)

fit = get_spectrum(fit_ps)
J = np.matrix(calc_J(fit_ps,p0/100))
#computes covarience matrix
Cv = np.linalg.inv(J.transpose() *w* J)
Cv = np.asarray(Cv)
#in theory this is the errors on the fit 
p_errs = np.diag(Cv)
print('free params fit give parameters of \n {} \n with errors of {} \n chi^2 = {}'.format(fit_ps,p_errs,chi_sq))


plt.plot(poles,data,label = 'data')
plt.plot(poles,fit,label='LM fit free params')
plt.plot(poles,lmFit_holdT,label='LM fit tau held')
plt.legend()
plt.xlabel('l (pole index)')
plt.ylabel('CMB power spec') 



#generates random step for mcmc for given covar matrix
def rand_step(Cv):
    chosky = np.linalg.cholesky(Cv)
    return np.asarray(np.dot(chosky,np.random.randn(Cv.shape[0])))



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
        t1 = time.time()
        print('step # = {} out of {}'.format(i,n_steps))
        #calls the random step function and adds to current params 
        #after being scaled by scaling factor to get the new "test' params
        p_test     = p_now + step_scale*rand_step(Cv)
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
            failcount = failcount + 1
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
        #prints the acceptance status of current test step
        print('step accepted = {}'.format(accept_test))
        print('current accept rate = {}  look for 0.25'.format(1 - failcount/(i)))
        #sets the current step to the test step if it is accepted
        if accept_test==True: 
            p_now     = p_test
            chisq_now = chisq_test
            print('chi^2=' + str(chisq_now))
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
        print('time for step = {} , projected time = {}'.format(time.time()-t1, (time.time()-t1)*(n_steps-i) ))
        print('*************************************')
    #saves the chains param's and chisqs
    print('done, saving to files...')        
    np.savetxt(genFileName(t,step_scale,chainname=chainname,pORc='p'),pars)
    np.savetxt(genFileName(t,step_scale,chainname=chainname,pORc='c'),chisqs)
    #returns the chain's params and chisqs
    acceptrate = 1- failcount/(n_steps)
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
#fit_CoVarM = np.loadtxt('CoVarM.txt')

#
#lastchain = np.loadtxt('bettertry_params_26_18_52_58_scale0point016.txt')
lm_fit_p = np.array([6.71878134e+01, 2.18484550e-02, 1.11019110e-01, 4.95000000e-02,1.98489097e-09, 9.40835003e-01])
newp0 = np.array([7.02832046e+01, 2.29083013e-02, 1.12626487e-01, 5e-02,2.05178292e-09, 9.73583264e-01])

Covary = np.array([[ 1.28610417e+01,  2.33347574e-03, -2.32944232e-02,
         3.83994796e-01,  1.42762577e-09,  7.93630402e-02],
       [ 2.33347574e-03,  6.73477492e-07, -3.26707341e-06,
         8.96715058e-05,  3.51707810e-13,  1.93549067e-05],
       [-2.32944232e-02, -3.26707341e-06,  4.91097294e-05,
        -6.55470216e-04, -2.34918455e-12, -1.27324790e-04],
       [ 3.83994796e-01,  8.96715058e-05, -6.55470216e-04,
         2.07195622e-02,  8.05521974e-11,  3.11973351e-03],
       [ 1.42762577e-09,  3.51707810e-13, -2.34918455e-12,
         8.05521974e-11,  3.14803359e-19,  1.20935324e-11],
       [ 7.93630402e-02,  1.93549067e-05, -1.27324790e-04,
         3.11973351e-03,  1.20935324e-11,  6.48668930e-04]])

chain = mcmc(data,err,newp0,Cv,n_steps=100,step_scale= 0.5,name = 'newcovar',saveAfter = 50)
     
    
p3 =  0.0544
ep3 = 0.0073
newp0[3] = p3
fixedtauchain = mcmc(data,err,newp0,Cv,n_steps=100,name ='fixedtau',step_scale= 0.5*np.array([1,1,1,0,1,1]),saveAfter = 50)

Cvnew = Cv.copy()
Cvnew[3,3] = ep3
othertauchain = mcmc(data,err,newp0,Cv,n_steps=100,name = 'weirdtau' ,step_scale= 0.5,saveAfter = 50)
