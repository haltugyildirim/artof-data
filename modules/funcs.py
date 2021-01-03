import h5py
import xarray as xr
import numpy as np
from numpy import pi, fft
import math
import scipy.signal as sci
from scipy.optimize import curve_fit
import mclahe as mc
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#Energy correction values

#this from a paper DOI:10.1103/PhysRevB.93.155426
dirac_point_origin_energy=0.15

#with eye from data.. will write a piece of func to determine it.
dirac_point_data_energy=1.65

dif_dp=dirac_point_origin_energy-dirac_point_data_energy

def importdelaydata(keys,scan):
    
    subkeys=list(scan.keys())
    
    n=len(subkeys)-1
    #empty list to append the DataArrays created by conversion matrices.
    a=[]
    #Empty Dataset to append the elements in 'a' list.
    data = xr.Dataset()
    #We will not going to use the dimension factors and dimension offset to calculate the  
    positions_raw=np.array(scan.get(subkeys[n]))
    #positions_raw=np.delete(positions_raw, 0)
    steps=int((positions_raw.size+1)/n)
        
    #The Data Acquisition program sometimes decrements the size of the first repetition randomly.
    #This is for checking.
    daq_program_error=np.array(scan.get(subkeys[1])).size-np.array(scan.get(subkeys[0])).size
    
    if daq_program_error==0:
        print('There is no omitting problem! Strange...')
    #If programs crashes before the repetitions are done, than it is not saving the positions array for delay
    #points.
    crash_error=len(np.array(scan.get(subkeys[len(subkeys)-1])).shape)
    
    if crash_error!=1:
        print('Be careful! Delay points are not exactly real as the dimension factors are not written on the acquisition program right. The true values are hold inside the position array but data miss the position array due to crash!')
        #because the hdf5 file saved as con0,con1,con10 !!! 
        crashed_conversion='conversion%d' % n
        subkeys.remove(crashed_conversion)
    n=len(subkeys)-1 

    for i in range(len(subkeys)-1):

        scanimage=np.array(scan.get(subkeys[i]))
        image_obj=scan.get(subkeys[i])
        dim_fac=np.array(image_obj.attrs['DIMENSION_FACTORS'])
        dim_off=np.array(image_obj.attrs['DIMENSION_OFFSETS'])
        dim_labels=np.array(image_obj.attrs['DIMENSION_LABELS'])
        dim_units=np.array(image_obj.attrs['DIMENSION_UNITS'])

        # in fs
        #multiplication with -1 is needed to flip the delaypoints to real values.
        if crash_error==1: 
            positions=-1*positions_raw[i*steps:(i+1)*steps]
        else:
            positions=-1*(np.arange(0,scanimage.shape[0])*scanimage.shape[0]/(scanimage.shape[0]-1)*dim_fac[0]-dim_off[0])

        #This if structure is because of a fault in the data acquisition program.
        #This program decrement the array size by one value in the first
        #repetition. For example if you have a delay scan with 101 steps, 4 rep.
        #First conversion0 will be in size 100 and following will be 101. This
        #reduces the other size to 100 as well.
        if daq_program_error!=0:
            # in fs
            #multiplication with -1 is needed to flip the delaypoints to real values.
            if crash_error==1: 
                positions=-1*positions_raw[i*steps:(i+1)*steps-1]
            else:
                positions=-1*(np.arange(0,scanimage.shape[0])*scanimage.shape[0]/(scanimage.shape[0]-1)*dim_fac[0]-dim_off[0])


            if i > 0:
                #this rolling is to put the last delaypoin to first delaypoint.
                #reason can be seen in the delaypoint omitted notebook
                scanimage=np.roll(scanimage, 1, axis=0)
                #deleting the last delaypoint as it only exists in the first conversion matrix
                scanimage=np.delete(scanimage, 0, axis=0)
                
        # in eV
        energies=(np.arange(0,scanimage.shape[1])*scanimage.shape[1]/(scanimage.shape[1]-1)*dim_fac[1]-dim_off[1])
        # in deg
        angles_y=(np.arange(0,scanimage.shape[2])*scanimage.shape[2]/(scanimage.shape[2]-1)*dim_fac[2]-dim_off[2])
        # in deg(this is the manipulator moving angle)
        angles_x=(np.arange(0,scanimage.shape[3])*scanimage.shape[3]/(scanimage.shape[3]-1)*dim_fac[3]-dim_off[3])

        a.append(xr.DataArray(scanimage, dims=['delay', 'energy', 'y', 'x'], 
                                              coords={'delay':positions, 'energy': energies, 'y': angles_y,'x': angles_x}, 
                                              name="con%d" % i))
        data["con%d" % i] = a[i]
        data.attrs['Sample Name'] = keys
        data.attrs['Type'] = "Delay Scan with %d repetitions" % n
    datasetsum=data["con0"]
    for i in range(n-1):
        datasetsum=datasetsum+data["con%d" % (i+1)]
    data["datasetsum"] = datasetsum
    return data

def dataimport(name,grouphome_path,method='none',encor='corrected'):
    """
    This function is to load data from THEMIS chamber measurements. It loads data from 
    hdf5 files, sorts it and returns an indexed matrix in DataArray format using xarray
    package. This matrix can be either in 4d or 3d, a delay scan or image respectively.
    If it contains several repetitions of a delay scan, it returns a DataSet with
    all repeated measurements as DataArray inside the DataFrame and a separate DataArray
    with all repetition matrices summed up to use for better statistics. If the masurement
    is a an image, it returns a DataArray with 3d matrix. For more information about DataArray
    and DataSet format refer to documentation of xarray package.
    """
    #takes the name of the file as the file path
    #'r' is for only reading the measurement data
    g = h5py.File(grouphome_path+name,'r')
    #"every hdf5 file has a something called Group. Groups are the container mechanism 
    #by which HDF5 files are organized. From a Python perspective, they operate somewhat 
    #like dictionaries. In this case the “keys” are the names of group members, and the 
    #“values” are the members themselves (Group and Dataset[subkeys]) objects."
    #-from docs of h5py library, modified.
    keys=list(g.keys())
    scan=g.get(keys[-1])
    #Subkeys correspond to our datasets. Every subkey has a conversion matrix inside 
    #and it is a list. If it is an image it only contains a single entry.
    #If it is a delay scan, additionaly it contains an array with position values of
    #the delay scan which is not necessary to use as this positions can be calculated
    #from the metadata of the 4-dimensional matrix conversion. If delay scan measurement
    #contains several repetitions, then there exists multiple entries. For example, if
    # there are 4 repetitions done, then subkeys length will be 5;1 position array +
    #4 conversion matrix.
    subkeys=list(scan.keys())
    #Every subkey contains also metedata associated with the matrix. There are 6 metadata
    #that contained in conversion matrix; dimension factors, dimension offsets, dimension
    #labels, dimension units, SPECS version and time zero. We only import the first four
    #metadata and use first two to calculate the label array of energies, angles and if
    #it is delay scan, positions.
    scanimage=np.array(scan.get(subkeys[0]))
    #Multidimensional Contrast Limited Adaptive Histogram Equalization (MCLAHE) for enhancing features
    #https://github.com/mpes-kit/mclahe
    if method=='mc':
        scanimage = mc.mclahe(scanimage)
    image_obj=scan.get(subkeys[0])
    dim_fac=np.array(image_obj.attrs['DIMENSION_FACTORS'])
    dim_off=np.array(image_obj.attrs['DIMENSION_OFFSETS'])
    dim_labels=np.array(image_obj.attrs['DIMENSION_LABELS'])
    dim_units=np.array(image_obj.attrs['DIMENSION_UNITS'])
    if dim_fac.size == 3:
        # in deg
        angles_y=(np.arange(0,scanimage.shape[1])*scanimage.shape[1]/(scanimage.shape[1]-1)*dim_fac[1]-dim_off[1]) 
        # in deg(this is the manipulator moving angle)
        angles_x=(np.arange(0,scanimage.shape[2])*scanimage.shape[2]/(scanimage.shape[2]-1)*dim_fac[2]-dim_off[2]) 
        # in eV
        energies=(np.arange(0,scanimage.shape[0])*scanimage.shape[0]/(scanimage.shape[0]-1)*dim_fac[0]-dim_off[0]) 
        data = xr.DataArray(scanimage, dims=['energy', 'y', 'x'], coords={'energy': energies, 'y': angles_y,'x': angles_x})
        data.attrs['Sample Name'] = keys
        data.attrs['Type'] = 'Image'
        if encor!='none':
            #this is energy correction, referenced to dirac point.
            data.coords['energy_corr'] = data.energy + dif_dp
            data = data.swap_dims({'energy': 'energy_corr'})
    else:
        data=importdelaydata(keys,scan)        
    return(data)

############################################################################################
############################################################################################

#upcoming block of functions are related to cross-correlation fit
#to flip the functions if desired, replace t[i] variable with (-t[i]+2*t[0]+(t[-1]-t[0]))

def convolutedfitfunc(t, C, A, sigma, tau, t_0):
    """
    This is the fit function for the cross correlation.
    #t is an the time array. If you supply the function with a float or integer, it might
     cause an error. Try to give an array, even for single numbers
    #C is the offset in y-axis
    #A is amplitude
    #sigma is the sigma-value of Gaussian, where we extract the FWHM of cross-correlation.
    #tau is the time constant of exponential decay, bigger values corresponds to shorter
     life time.
    #t_0 is time-zero where time is zero, a singularity where nothing happens.
     No, this is lie, cake is a lie.
    #This function is calculated analytically with Mathematica, one can see the process in
     'gaussian_exponential_decay_convolution_analytical.nb' inside cross-correlation folder.
     More detail about it can be learned inside cross-correlation Jupyter notebook.
    """
    f=np.zeros(t.size)
    for i in range(t.size):
        val = C-A*np.exp(sigma**2/(2*tau**2)-(t[i]-t_0)/tau)*(-1+math.erf((sigma**2-tau*(t[i]-t_0))/(np.sqrt(2)*sigma*tau)))
        f[i] = val
    return f

def gaussian(t, sigma, t_0):
    """
    This is a standart gaussian.
    """
    #C is the offset
    #t_0 is t zero
    f=np.zeros(t.size)
    for i in range(t.size):
        val = np.exp(-(t[i])**2/(2*sigma**2))
        f[i] = val
    return f

def expdecay(t, tau, t_0):
    """
    This is an exponential decay with a Heaviside function.
    """
    #C is the offset
    #t_0 is t zero
    f=np.zeros(t.size)
    for i in range(t.size):
        val = (t[i]>=t_0)*np.exp((t_0-t[i])/tau)
        f[i] = val
    return f

def gedc(t, C, A, sigma, tau, t_0):
    """
    This is fit function that does the convolution inside of it with scipy package convolution.
    One can sometimes see a kind of a 'bump' in the end of the exponential decay. This is caused
    from the fact that, if you don't centralize your exponential function in the middle of your
    
    """
    #C is the offset
    #t_0 is t zero
    f=np.zeros(t.size)
    g=np.zeros(t.size)
    for i in range(t.size):
        expdecayv = (t[i]>=t_0)*np.exp((t_0-t[i])/tau)
        gaussianv = np.exp(-(t_0-t[i])**2/(2*sigma**2))
        f[i] = expdecayv
        g[i] = gaussianv
    return C-A*sci.convolve(f,g,mode='same')

############################################################################################
############################################################################################

#upcoming block of functions are used for convoluting the exp and gaussian with fourier transform method.

def ft_eq(t, X, omega = False, zero_centred = True, absolute = False):
    """
    Returns f, Fx, both same length as t and x
    If omega is set to true, then omega is returned instead of f
    If zero_centred, then 0 Hz is shifted to the centre of the spectrum (Default)
    If absolute is true, abs(Fx) is returned instead of Fx
    """
    N = len(X)
    dt = (t[-1]-t[0])/(N-1)
    F_X = fft.fft(X)*dt # correct the FT by multiplying with dt 
    
    # Creating the omega axis - by default 0Hz is at the first entry of the array:
    nu = fft.fftshift(fft.fftfreq(N,dt))
    
    # shifting 0 Hz to the centre of the spectrum
    if zero_centred:
        F_X = fft.fftshift(F_X)
    else:
        nu -= nu[0]
        
    if omega:
        nu *= 2*pi
        
    if absolute:
        F_X = abs(F_X)
        
    return nu, F_X


def ift_eq(nu, F_X, omega = False, was_zero_centred = True):
    """
    Returns t, x, both same length as f and Fx
    If omega is set to true, you can input omega instead of f. Then f = f/(2*pi) internally.
    If your spectrum was not zero-centred, set was_zero_centred = False
    Warning: Usually you would want to feed Fx into this fuction, not abs(Fx)!
    """
   
    N = len(F_X)
    dnu = (nu[-1]-nu[0])/(N-1)
    
    if omega:
        dnu /= 2*pi
    
    t0 = fft.fftfreq(N, dnu)
    t = fft.fftshift(t0)
    t -= t[0]
    
    # ifft has a prefactor of 1/N so we need to correct for it by multipling with N*df
    if was_zero_centred == 0:    
        X = fft.ifft(F_X)*dnu*N
    if was_zero_centred == 1:    
        X = fft.ifft(fft.fftshift(F_X))*dnu*N
    
    return t, X

def fourierfitfunc(t, C, A, sigma, tau, t_0):
    """
    This is the fit function using fourier method.
    """
    gaussianv=gaussian(t, sigma, t_0)
    expdecayv=expdecay(t, tau, t_0)
    nug, gaussianfourier=ft_eq(t,gaussianv)
    nue, expfourier=ft_eq(t,expdecayv)
    tgg=expfourier*gaussianfourier
    t1,convv=ift_eq(nug,tgg)
    return C-A*convv

############################################################################################
############################################################################################

#upcoming functions are to find the FWHM of a function(mostly gaussian) without using sigma.
def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]

############################################################################################
############################################################################################
#Functions below are used for cross-correlation fitting.

def xc_fit(name,grouphome_path,slice_down,slice_up,sigma,tau,FWHM_pump,mode='corrected',method='none',crashset='none',encor='corrected'):
    """
    Quite reasonably written and pretty working well automatic cross correlation function.
    """

    if crashset!='none':
        dataset=crashset
    else:
        dataset=dataimport(name,grouphome_path,method=method)
    
    taken_value='delay'
    
    if encor!='none':
        #this is energy correction, referenced to dirac point.
        dataset.coords['energy_corr'] = dataset.energy + dif_dp
        dataset = dataset.swap_dims({'energy': 'energy_corr'})
    
    
    #if the mode is 'corrected', it will map delay scan values to a new array where fitted
    #t_0 value is zero.
    if mode=='corrected':
        parameter_space=xc_fit_rep(dataset,slice_down,slice_up,sigma,tau,taken_value,encor)

        #changing the 
        dataset.coords['delay_corr'] = dataset.delay - parameter_space[1]
        dataset = dataset.swap_dims({'delay': 'delay_corr'})
        taken_value='delay_corr'
        t_0=-1*parameter_space[1]

    parameter_space_1=xc_fit_rep(dataset,slice_down,slice_up,sigma,tau,taken_value,encor)
    
    datasetsum=parameter_space_1[2]
    delayvaluesasarrays=parameter_space_1[3]
    countvaluesasarray=parameter_space_1[4]
    t=parameter_space_1[5]
    f_ig=parameter_space_1[6]
    f_fit=parameter_space_1[7]
    tau=parameter_space_1[8]
    
    sigma=parameter_space_1[0]
    
    if mode!='corrected':
        t_0=-1*parameter_space_1[1]
    
    FWHM_xc=2*np.sqrt(2*np.log(2))*sigma
    FWHM_probe=np.sqrt(FWHM_xc**2-FWHM_pump**2)
    
    #print values
    print('$sigma=$', sigma)
    print('$tau=$', tau)
    print('$t_{0}=$', t_0)
    print('$FWHM_{xc}=$', FWHM_xc)
    print('$FWHM_{probe}=$', FWHM_probe)
    
    #plotting
    gs = gridspec.GridSpec(2, 2)

    plt.figure()
    ax = plt.subplot(gs[0, :]) # row 0, span all columns
    datasetsum.sum(dim=('y','x')).T.plot()
    plt.axhline(y=slice_down, color='r',linestyle='--')
    plt.axhline(y=slice_up, color='r',linestyle='--')
    plt.xlabel('delay (fs)')
    plt.ylabel('$E-E_F (eV)$')

    ax = plt.subplot(gs[1, 0]) # row 1, col 0
    plt.plot(t,f_ig,label='fit with initial guess')
    plt.plot(delayvaluesasarrays,countvaluesasarray,label='data')
    plt.xlabel('delay (fs)')
    plt.ylabel('counts')
    plt.legend()

    ax = plt.subplot(gs[1, 1]) # row 1, col 1
    plt.plot(t,f_fit,label='fit')
    plt.plot(delayvaluesasarrays,countvaluesasarray,label='data')
    plt.xlabel('delay (fs)')
    plt.ylabel('counts')
    
    plt.suptitle(name)
    plt.tight_layout()
    plt.legend()
    
    return sigma, FWHM_xc, FWHM_probe, datasetsum, slice_down, slice_up, delayvaluesasarrays, countvaluesasarray, t, f_ig, f_fit, tau, dataset, t_0, encor

def xc_fit_rep(dataset,slice_down,slice_up,sigma,tau,taken_value,encor):
    """
    This function called inside the xc_fit function. It first takes the repetation value n in order to add the datasets up
    to each other for better statistics. Use 1.25 and 1.5 for slice_down and slice_up if you don't know what to use.
    """
    datasetsum=dataset.datasetsum
    
    if encor!='none':
        selectedxc=datasetsum.sum(dim=('y','x')).sel(energy_corr=slice(slice_down,slice_up)).sum(dim='energy_corr')
    else:
        selectedxc=datasetsum.sum(dim=('y','x')).sel(energy=slice(slice_down,slice_up)).sum(dim='energy')
    
    
    countvaluesasarray=selectedxc.values
    delayvaluesasarrays=selectedxc.coords[taken_value].values

    #initial values for fitting C, A, offset, sigma, tao, tzero in order. To fit the guess to the same 
    #window with the data, change tzero value to an appropriate one.
    p0=[np.amin(countvaluesasarray),np.amax(countvaluesasarray)/1.6,sigma,tau,delayvaluesasarrays[np.where(countvaluesasarray == np.amax(countvaluesasarray))][0]]
    
    #delay stage can be in 'negative' positions. To prevent 
    if delayvaluesasarrays[delayvaluesasarrays.size-1]>delayvaluesasarrays[0]:
        t = np.arange(delayvaluesasarrays[0],delayvaluesasarrays[delayvaluesasarrays.size-1], 10)
    else:
        t = np.arange(delayvaluesasarrays[delayvaluesasarrays.size-1],delayvaluesasarrays[0], 10)

    f_ig = convolutedfitfunc(t,*p0)
    popt, pcov = curve_fit(convolutedfitfunc, delayvaluesasarrays, countvaluesasarray,p0)
    f_fit = convolutedfitfunc(t,*popt)
    sigma=popt[2]
    tau=popt[3]
    t_zero_value=popt[4]
    return sigma, t_zero_value, datasetsum, delayvaluesasarrays, countvaluesasarray, t, f_ig, f_fit, tau

def ds_time_b(t_0,start,stop,steps,rep,expt):
    """
    Calculate how much time it will take delayscan to finish and print the real values that delay stage takes.
    """
    real_start=t_0-start
    real_stop=t_0-stop
    delayscant=(steps*rep*expt)/(60*60)
    print('delay scan will take',str(datetime.timedelta(hours=delayscant)),'hours to complete')
    print('delay scan region is from',real_start,' fs to',real_stop, 'fs')
    print('Step size:',(stop-start)/steps)
    print('Exposure time per image:',rep*expt)
    return