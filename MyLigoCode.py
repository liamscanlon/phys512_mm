import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob

from scipy import signal as  sig 
from scipy.signal import butter, filtfilt , tukey

#from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
#you are going to need to pip install mvgavg  for this one to work
from mvgavg  import mvgavg 
interp1d

plt.ion()

def read_template(filename):
    dataFile=h5py.File(filename,'r')
    template=dataFile['template']
    th=template[0]
    tl=template[1]
    return th,tl
def read_file(filename):
    dataFile=h5py.File(filename,'r')
    dqInfo = dataFile['quality']['simple']
    #qmask=dqInfo['DQmask'][...]

    meta=dataFile['meta']
    #gpsStart=meta['GPSstart'].value
    #print meta.keys()
    utc=meta['UTCstart'].value
    duration=meta['Duration'].value
    strain=dataFile['strain']['Strain'].value
    dt=(1.0*duration)/len(strain)

    dataFile.close()
    return strain,dt,utc



#fnames=glob.glob("[HL]-*.hdf5")
#fname=fnames[0]
fnameHV2='H-H1_LOSC_4_V2-1126259446-32.hdf5'
print ('reading file ',fnameHV2)
strainHV2,dtHV2,utcH1V2=read_file(fnameHV2)

fnameHV1='H-H1_LOSC_4_V1-1126259446-32.hdf5'
print ('reading file ',fnameHV1)
strainHV1,dtHV1,utcHV1=read_file(fnameHV1)



fnameL='L-L1_LOSC_4_V1-1126259446-32.hdf5'
print ('reading file ',fnameL)
strainL,dtL,utcL1=read_file(fnameL)


windows = ['boxcar', 'triang','blackman','hamming','hann', 'bartlett','flattop', 'parzen', 'bohman', 'blackmanharris', 'nuttall','barthann']


#Calculates the power spectral dencity of the data
#uses the fancy pants welch method from scipy.signal
#PSD[g(t)] = (FT[g(t)])^2/(df) where df is the size of frequency bin 
def calc_psd(strain,dt, window= 'blackman'):
    fs = 4/dt # I think this is to make the 
    frqs , psd = sig.welch(strain,1/dt,window= window,nperseg=fs)
    return frqs , psd

frqsHV2 , psdHV2 = calc_psd(strainHV2,dtHV2)
frqsHV1 , psdHV1 = calc_psd(strainHV1,dtHV1)
frqsL , psdL = calc_psd(strainL,dtL)

#this averages points with there neighbours to smooth then since that 
#changes the frequencies it outputs a intperlation function 
#so you can use whatever frequecies you need 
def interp_smooth_psd(psd,frqs,ravg):
    psd_smoo , frqs_smoo  = mvgavg(psd,ravg) , mvgavg(frqs,ravg)
    psd_intp_smoo = interp1d(frqs_smoo,psd_smoo, bounds_error= False, fill_value='extrapolate')
    #psd_intp_smoo = CubicSpline(frqs_smoo,psd_smoo)
    return psd_intp_smoo
#whitens data
def whiten(strain,psd_func,dt):
    FTstrain = np.fft.rfft(strain)
    FTfreqs  = np.fft.rfftfreq(strain.size,dt)
    # dont think I need thois for the rfft   FTfreqs  = np.abs(FTfreqs)
    asp      = np.sqrt(psd_func(FTfreqs))
    FT_w     = (FTstrain/asp)*(np.sqrt(dt/2))
    strain_w = np.fft.irfft(FT_w)
    return np.real(strain_w)
    
#adds band pass filter with corners at f1 and f2
def bp_filter(strain,f1,f2,dt ):
    bb, ab = butter(4, [f1*2*dt, f2*2*dt], btype='band')
    strain_filt = filtfilt(bb,ab,strain)
    return strain_filt
    
def match_filter(strain,psd_func,template,dt):
    w =  sig.get_window(('gaussian',strain.size*0.22),strain.size)
    frqs = np.fft.rfftfreq(strain.size,dt)
    Nspec =  psd_func(frqs)
    FTstrain = np.fft.rfft(w*strain)
    FTtemplate = np.fft.rfft(w*strain) 
    Nspec =  psd_func(frqs)
    mFiltFT = np.conjugate(FTtemplate)*FTstrain/Nspec
    mFilt = np.fft.irfft(mFiltFT)
    return mFilt


    
    
    
#looking at different windows
plt.figure()
for i in  range(len(windows)):
    frqs , psd = calc_psd(strainHV2,dtHV2,window=windows[i])
    plt.loglog(frqs,psd,label = str(windows[i]))
plt.legend()
plt.grid()


#average with neighbouring points
#want to cut the noise a bit but keep the peaks because they are real and due to instrument
#compare the averaging radius
plt.figure()
for n in [1,2,3,4,5,6]:
    plt.loglog(mvgavg(frqsHV1,n),np.sqrt(mvgavg(psdHV1,n)),label = str(n))
plt.legend()
plt.grid()
plt.loglog(frqsHV1,np.sqrt(psdHV1),'g',label = 'H1 V1')




"""
Next steps:

* Do the file management stuff
*implement a match filter function

"""




#plt.loglog(frqsHV2,np.sqrt(psdHV2),'b',label = 'H1 V2')
#plt.loglog(frqsL,np.sqrt(psdL),'r',label = 'L1 V1')
#plt.axis([10, 2E3,1E-24,1E-17])


#
#FtH = np.fft.fft(strainH*np.hamming(strainH.size))
#freqH = np.fft.fftfreq(strainH.size,d = dtH)
#
#FtL= np.fft.fft(strainL*np.hamming(strainL.size))
#freqL = np.fft.fftfreq(strainL.size,d = dtL)
#
#
#
#
#dfH = freqH[1] - freqH[0] 
#PSH = np.abs(FtH)**2/dfH
#dfL = freqL[1] - freqL[0] 
#PSL = np.abs(FtL)**2/dfL
#
#
#plt.loglog(freqH,np.sqrt(PSH),'.')
#plt.loglog(freqH,np.sqrt(PSL),'.')
#plt.axis([10, 2E3,np.min(np.sqrt(PSH)),np.max(np.sqrt(PSH))],)

#spec,nu=measure_ps(strain,do_win=True,dt=dt,osamp=16)
#strain_white=noise_filter(strain,numpy.sqrt(spec),nu,nu_max=1600.,taper=5000)

#th_white=noise_filter(th,numpy.sqrt(spec),nu,nu_max=1600.,taper=5000)
#tl_white=noise_filter(tl,numpy.sqrt(spec),nu,nu_max=1600.,taper=5000)


#matched_filt_h=numpy.fft.irfft(numpy.fft.rfft(strain_white)*numpy.conj(numpy.fft.rfft(th_white)))
#matched_filt_l=numpy.fft.irfft(numpy.fft.rfft(strain_white)*numpy.conj(numpy.fft.rfft(tl_white)))




#copied from bash from class
# strain2=np.append(strain,np.flipud(strain[1:-1]))
# tobs=len(strain)*dt
# k_true=np.arange(len(myft))*dnu
