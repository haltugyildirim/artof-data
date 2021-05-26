import h5py
import xarray as xr
import numpy as np
from numpy import pi, fft
import math
import scipy.signal as sci
from scipy.optimize import curve_fit
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# DIRAC_POINT_ORIGIN_ENERGY
# For Sb2Te3 project this value is 0.15, it is taken from the paper DOI:10.1103/PhysRevB.93.155426
# As similar to encor_var, this value is also physical system specific.
# If you want to define the energy correction in respect to fermi energy, make this value 0 and just use 'encor_val'
# with accessing it via 'funcs.DIRAC_POINT_ORIGIN_ENERGY = 0.15' inside your code/notebook.
# for converted data(image or delay) in dataimport_conv, for angle-image in dataimport, for angle-delay in xc_fit.
DIRAC_POINT_ORIGIN_ENERGY = 0.15

#def dataimport_ang(name, path, encor='corrected', encor_var=1.65):
#dataimport_conv(name, path, exp_type, outer_correction=0.15, x_off=0, y_off=0, delay_cor='none', encor_var=1.65, x_corr=0, y_corr=0):

def dataimport(name, path, data_type, exp_type, encor='corrected', delay_cor='none', outer_correction=0.15, x_off=0, y_off=0, encor_var=1.65, x_corr=0, y_corr=0):
    """
    Data import function for delay stage experiments and images. It loads data from
    hdf5 files, sorts it and returns an indexed matrix in DataArray format using xarray
    package. This matrix is a 4d object, a delay scan or image respectively. In both cases it returns a DataArray
    with 4d matrix. For more information about DataArray and DataSet format refer to documentation of xarray package.
    Returns a DataArray object.

    For angle-converted data:
    If it contains several repetitions of a delay scan, it returns a DataSet with
    all repeated measurements as DataArray inside the DataFrame and a separate DataArray
    with all repetition matrices summed up to use for better statistics. If the masurement
    is a an image, it returns a DataArray with 3d matrix.

    name                        -> name of the data

    path                        -> path to data

    data_type                   -> 'momentum' or 'angle'. for 'momentum' and 'angle' converted data, the import
    prodecure is different and should be chosen accordingly.

    exp_type                    -> 'image' or 'delay'. This can be either a delay experiment or an image.
    Both of the objects are 4D but the image has only a single value on 1 dimension and delay experiment
    has the number of values as much as the experiments step size.

    encor                       -> 'corrected' or none. When corrected selected the energy will be shifted by the global
    variable 'dif_dp', only for image type of data. For the delay data, energy correction is done inside the function
    'xc_fit' together with delay correction. Only called for angle-converted data.

    outer_correction            -> for masking the errors on the outer part of the converted data with
    creating a paraboloid mask.

    x_off                       -> x_off=0(optional) x Offset for outer correction

    y_off                       -> y_off=0(optional) y Offset for outer correction

    delay_cor                   -> delay_cor='none'(optional) If the corresponding
    data is just an image, than one can correct the delay point to physical value rather than what
    the program is giving from the arbitrary value from delay stage.

    encor_var -> This value should be selected by the user for appropriate physical system.
    For the case of Sb2Te3 in energy conversion range of 1 to 3eV, this value is approximately ~1.65eV.
    If you want to define the energy correction in respect to fermi energy, make DIRAC_POINT_ORIGIN_ENERGY
    inside 'funcs.py' value 0 and just use 'encor_val'. If you are doing experiments in another topological
    insulators, change the DIRAC_POINT_ORIGIN_ENERGY to appropriate for that system.
    for converted data(image or delay) in dataimport_conv, for angle-image in dataimport, for angle-delay in xc_fit.

    x_corr                      -> Correction for kx values for indexing. This is useful, if the sample is
    not perfectly aligned in front of the detector. As the detector angle is \pm15 degrees and angle offset is
    less than 5 degrees most of the time, together with small angle approximation, there is no need to introduce
    this correction during the angle to momentum conversion.

    y_corr                      -> Correction for ky values for indexing.
    """
    if data_type == 'momentum':
        data = dataimport_conv(name, path, exp_type, outer_correction, x_off, y_off, delay_cor, encor_var, x_corr, y_corr)
    elif data_type == 'angle':
        data = dataimport_ang(name, path, encor, encor_var, delay_cor)
    return data

def dataimport_conv(name, path, exp_type, outer_correction=0.15, x_off=0, y_off=0, delay_cor='none', encor_var=1.65, x_corr=0, y_corr=0):
    """
    This function is called inside the dataimport function for converted data.
    """

    # hdf5 file reading
    g = h5py.File(path+name, 'r')

    # image matrix
    keys = list(g.keys())
    scan = g.get(keys[-1])
    scanimage = np.array(scan)

    # reading metadata for index values
    e_min = scan.attrs['E_min']
    e_max = scan.attrs['E_max']
    delay_pos = np.squeeze(scan.attrs['DelayList'])
    k_min = scan.attrs['Winkel_min']
    k_max = scan.attrs['Winkel_max']
    k_size = scan.attrs['AnzahlWinkel']
    e_size = scan.attrs['AnzahlEnergien']
    dim_fac_k = (k_max-k_min)/k_size
    dim_fac_e = (e_max-e_min)/e_size

    # delay position
    if exp_type == 'image':
        if delay_cor != 'none':
            delay = np.array([delay_cor])
        else:
            delay = np.array([delay_pos])
    else:
        delay = -1*delay_pos

    # in k_parallel
    k_values = (np.arange(
        0, scanimage.shape[2])*scanimage.shape[2]/(scanimage.shape[2]-1)*dim_fac_k+k_min)
    # in eV
    energies = (np.arange(
        0, scanimage.shape[1])*scanimage.shape[1]/(scanimage.shape[1]-1)*dim_fac_e+e_min)

    # in order to bin the data, energy and k arrays needs to have an even shape. If they are odd(in the case of k values
    # they are always odd as the symmetry of the conversion requires it.)
    if np.size(scanimage, 1) % 2 != 0:
        scanimage = scanimage[:, :-1, :, :]
        energies = energies[:-1]
    if np.size(scanimage, 2) % 2 != 0:
        scanimage = scanimage[:, :, :-1, :-1]
        k_values = k_values[:-1]

    # creating DataArray object
    if exp_type == 'image':
        data = xr.DataArray(scanimage, dims=['Delay', 'energy', 'x_ncorr', 'y_ncorr'], coords={
                        'Delay': delay, 'energy': energies, 'x_ncorr': k_values, 'y_ncorr': k_values})
    else:
        data = xr.DataArray(scanimage, dims=['delay', 'energy', 'x_ncorr', 'y_ncorr'], coords={
                        'delay': delay, 'energy': energies, 'x_ncorr': k_values, 'y_ncorr': k_values})

    # energy correction calculation
    dif_dp = DIRAC_POINT_ORIGIN_ENERGY - encor_var

    # corrections for energy, kx and ky
    data.coords['Energy'] = data.energy + dif_dp
    data = data.swap_dims({'energy': 'Energy'})

    data.coords['x'] = (data.x_ncorr / 10) + x_corr
    data = data.swap_dims({'x_ncorr': 'x'})

    data.coords['y'] = (data.y_ncorr / 10) + y_corr
    data = data.swap_dims({'y_ncorr': 'y'})

    # This function is to get rid of the outer rings caused by the conversion.
    # it masks the data with a paraboloid.
    data = data.where(np.sqrt((data.x_ncorr+x_off)**2 +
                      (data.y_ncorr+y_off)**2) < data.energy-outer_correction, 0)

    # this values can be used in further calculations.
    data.attrs['x_corr'] = x_corr

    data.attrs['y_corr'] = y_corr

    data.attrs['dif_dp'] = dif_dp

    # IMPORTANT! This tranpose is not happening in the data anymore, in order to have the chance the align the incoming
    # photon direction from the lower part of the energy-cut image data(x-y plane).

    # compared to angle-data, x and y is inverted in converted-data, but this can cause misunderstandings, thus we
    # transpose between x and y dimensions.
    #data = data.transpose('delay', 'Energy', 'y', 'x')

    return data


def importdelaydata(keys, scan):
    """
    This function is called inside the dataimport function to import delayscan data. Contrary to converted datasets,
    angle data has only 4 dimensions when the character of data is delay, otherwise(image), it is 3. Related to data
    that is directly saved under as *.h5 file, there are many problems related to data saving, which is not observed
    under the converted data. This program deals with most them, so it is very system specific part of the code.
    Returns a DataArray object.
    """
    subkeys = list(scan.keys())

    n = len(subkeys)-1
    # empty list to append the DataArrays created by conversion matrices.
    a = []
    # Empty Dataset to append the elements in 'a' list.
    data = xr.Dataset()
    # We will not going to use the dimension factors and dimension offset to calculate the
    positions_raw = np.array(scan.get(subkeys[n]))
    #positions_raw=np.delete(positions_raw, 0)
    steps = int((positions_raw.size+1)/n)

    # The Data Acquisition program sometimes decrements the size of the first repetition randomly.
    # This is for checking.
    daq_program_error = np.array(
        scan.get(subkeys[1])).size-np.array(scan.get(subkeys[0])).size

    if daq_program_error == 0:
        print('There is no decrementation of the first repetition.')
    # If programs crashes before the repetitions are done, than it is not saving the positions array for delay
    # points.
    crash_error = len(np.array(scan.get(subkeys[len(subkeys)-1])).shape)

    if crash_error != 1:
        print('Be careful! Delay points are not exactly real as the dimension factors are not written on the acquisition program right. The true values are hold inside the position array but data miss the position array due to crash!')

        # because the hdf5 file saved as con0,con1,con10 one can not simply take out the last array from hdf5 file
        # but needs to find the latest array.
        crashed_conversion = 'conversion%d' % n
        subkeys.remove(crashed_conversion)
    n = len(subkeys)-1

    for i in range(len(subkeys)-1):

        scanimage = np.array(scan.get(subkeys[i]))
        image_obj = scan.get(subkeys[i])
        dim_fac = np.array(image_obj.attrs['DIMENSION_FACTORS'])
        dim_off = np.array(image_obj.attrs['DIMENSION_OFFSETS'])
        dim_labels = np.array(image_obj.attrs['DIMENSION_LABELS'])
        dim_units = np.array(image_obj.attrs['DIMENSION_UNITS'])

        # in fs
        # multiplication with -1 is needed to flip the delaypoints to real values.
        if crash_error == 1:
            positions = -1*positions_raw[i*steps:(i+1)*steps]
        else:
            positions = -1*(np.arange(0, scanimage.shape[0])*scanimage.shape[0]/(
                scanimage.shape[0]-1)*dim_fac[0]-dim_off[0])

        # This if structure is because of a fault in the data acquisition program.
        # This program decrement the array size by one value in the first
        # repetition. For example if you have a delay scan with 101 steps, 4 rep.
        # First conversion0 will be in size 100 and following will be 101. This
        # reduces the other size to 100 as well.
        if daq_program_error != 0:
            # in fs
            # multiplication with -1 is needed to flip the delaypoints to real values.
            if crash_error == 1:
                positions = -1*positions_raw[i*steps:(i+1)*steps-1]
            else:
                positions = -1*(np.arange(0, scanimage.shape[0])*scanimage.shape[0]/(
                    scanimage.shape[0]-1)*dim_fac[0]-dim_off[0])

            if i > 0:
                # this rolling is to put the last delaypoin to first delaypoint.
                # reason can be seen in the delaypoint omitted notebook
                scanimage = np.roll(scanimage, 1, axis=0)
                # deleting the last delaypoint as it only exists in the first conversion matrix
                scanimage = np.delete(scanimage, 0, axis=0)

        # in eV
        energies = (np.arange(
            0, scanimage.shape[1])*scanimage.shape[1]/(scanimage.shape[1]-1)*dim_fac[1]-dim_off[1])
        # in deg
        angles_y = (np.arange(
            0, scanimage.shape[2])*scanimage.shape[2]/(scanimage.shape[2]-1)*dim_fac[2]-dim_off[2])
        # in deg(this is the manipulator moving angle)
        angles_x = (np.arange(
            0, scanimage.shape[3])*scanimage.shape[3]/(scanimage.shape[3]-1)*dim_fac[3]-dim_off[3])

        a.append(xr.DataArray(scanimage, dims=['delay', 'energy', 'y', 'x'],
                              coords={'delay': positions, 'energy': energies,
                                      'y': angles_y, 'x': angles_x},
                              name="con%d" % i))
        data["con%d" % i] = a[i]
        data.attrs['Sample Name'] = keys
        data.attrs['Type'] = "Delay Scan with %d repetitions" % n
    datasetsum = data["con0"]
    for i in range(n-1):
        datasetsum = datasetsum+data["con%d" % (i+1)]
    data["datasetsum"] = datasetsum
    return data


def dataimport_ang(name, path, encor='corrected', encor_var=1.65, delay_cor='none'):
    """
    This function is called inside the dataimport function for converted data to load non-converted
    data(angle-converted) from measurements.
    """
    # takes the name of the file as the file path
    # 'r' is for only reading the measurement data
    g = h5py.File(path+name, 'r')
    # "every hdf5 file has a something called Group. Groups are the container mechanism
    # by which HDF5 files are organized. From a Python perspective, they operate somewhat
    # like dictionaries. In this case the “keys” are the names of group members, and the
    # “values” are the members themselves (Group and Dataset[subkeys]) objects."
    # -from docs of h5py library, modified.
    keys = list(g.keys())
    scan = g.get(keys[-1])
    # Subkeys correspond to our datasets. Every subkey has a conversion matrix inside
    # and it is a list. If it is an image it only contains a single entry.
    # If it is a delay scan, additionaly it contains an array with position values of
    # the delay scan which is not necessary to use as this positions can be calculated
    # from the metadata of the 4-dimensional matrix conversion. If delay scan measurement
    # contains several repetitions, then there exists multiple entries. For example, if
    # there are 4 repetitions done, then subkeys length will be 5;1 position array +
    # 4 conversion matrix.
    subkeys = list(scan.keys())
    # Every subkey contains also metedata associated with the matrix. There are 6 metadata
    # that contained in conversion matrix; dimension factors, dimension offsets, dimension
    # labels, dimension units, SPECS version and time zero. We only import the first four
    # metadata and use first two to calculate the label array of energies, angles and if
    # it is delay scan, positions.
    scanimage = np.array(scan.get(subkeys[0]))

    image_obj = scan.get(subkeys[0])
    dim_fac = np.array(image_obj.attrs['DIMENSION_FACTORS'])
    dim_off = np.array(image_obj.attrs['DIMENSION_OFFSETS'])
    dim_labels = np.array(image_obj.attrs['DIMENSION_LABELS'])
    dim_units = np.array(image_obj.attrs['DIMENSION_UNITS'])

    # This value is an approximate value for the angle-resolved mode. For momentum-converted data, value is given
    # as a default argument in the dataimport_conv function and should be changed by the user for appropriate physical system
    # or sample.
    dif_dp = DIRAC_POINT_ORIGIN_ENERGY-encor_var

    if dim_fac.size == 3:
        # dataimport for 3d(image) files.
        # in deg
        angles_y = (np.arange(
            0, scanimage.shape[1])*scanimage.shape[1]/(scanimage.shape[1]-1)*dim_fac[1]-dim_off[1])
        # in deg(this is the manipulator moving angle)
        angles_x = (np.arange(
            0, scanimage.shape[2])*scanimage.shape[2]/(scanimage.shape[2]-1)*dim_fac[2]-dim_off[2])
        # in eV
        energies = (np.arange(
            0, scanimage.shape[0])*scanimage.shape[0]/(scanimage.shape[0]-1)*dim_fac[0]-dim_off[0])

        #if no delay_cor given number '99999' is an arbitrary number for delay.
        if delay_cor != 'none':
            delay = np.array([delay_cor])
        else:
            delay = np.array([99999])

        data = xr.DataArray(scanimage, dims=['energy', 'y', 'x'], coords={
                            'energy': energies, 'y': angles_y, 'x': angles_x})

        data = data.expand_dims({'Delay':delay})

        data.attrs['Sample Name'] = keys
        data.attrs['Type'] = 'Image'
        if encor != 'none':
            # this is energy correction, referenced to dirac point.
            data.coords['Energy'] = data.energy + dif_dp
            data = data.swap_dims({'energy': 'Energy'})
    else:
        # dataimport for 4d(delay) files.
        data = importdelaydata(keys, scan)

    return data

############################################################################################
############################################################################################

# upcoming block of functions are related to cross-correlation fit
# to flip the functions if desired, replace t[i] variable with (-t[i]+2*t[0]+(t[-1]-t[0]))


def convolutedfitfunc(t, C, A, sigma, tau, t_0):
    """
    This is the fit function for the cross correlation.
    Returns fitted function array.

    t     -> the time array. If you supply the function with a float or integer, it might
     cause an error. Try to give an array, even for single numbers

    C     -> offset in y-axis

    A     -> amplitude

    sigma -> sigma-value of Gaussian, where we extract the FWHM of cross-correlation.

    tau   -> time constant of exponential decay, bigger values corresponds to shorter
    life time.

    t_0   -> time-zero where time is zero, a singularity where nothing happens.


    This function is calculated analytically with Mathematica, one can see the process in
     'gaussian_exponential_decay_convolution_analytical.nb' file.
     More detail about it can be learned inside cross-correlation Jupyter notebook.
    """
    f = np.zeros(t.size)
    for i in range(t.size):
        val = C-A*np.exp(sigma**2/(2*tau**2)-(t[i]-t_0)/tau)*(-1+math.erf(
            (sigma**2-tau*(t[i]-t_0))/(np.sqrt(2)*sigma*tau)))
        f[i] = val
    return f


def gaussian(t, sigma, t_0):
    """
    This is a standart gaussian.
    Returns gaussian array.

    t     -> the time array. If you supply the function with a float or integer, it might
    cause an error. Try to give an array, even for single numbers

    sigma -> sigma-value of Gaussian, where we extract the FWHM of cross-correlation.

    t_0   -> time-zero where time is zero, a singularity where nothing happens.
    """
    # C is the offset
    # t_0 is t zero
    f = np.zeros(t.size)
    for i in range(t.size):
        val = np.exp(-(t[i])**2/(2*sigma**2))
        f[i] = val
    return f


def expdecay(t, tau, t_0):
    """
    This is an exponential decay with a Heaviside function.
    Returns exponential array

    t   -> the time array. If you supply the function with a float or integer, it might
    cause an error. Try to give an array, even for single numbers

    tau -> time constant of exponential decay, bigger values corresponds to shorter
    life time.

    t_0 -> time-zero where time is zero, a singularity where nothing happens.
    """
    f = np.zeros(t.size)
    for i in range(t.size):
        val = (t[i] >= t_0)*np.exp((t_0-t[i])/tau)
        f[i] = val
    return f


def gedc(t, C, A, sigma, tau, t_0):
    """
    This is fit function that does the convolution inside of it with scipy package convolution.
    One can sometimes see a kind of a 'bump' in the end of the exponential decay. This is caused
    from the fact that, if you don't centralize your exponential function in the middle of your t,
    it will only convolute in this range.
    Returns convoluted gauusian * exponential array

    t     -> the time array. If you supply the function with a float or integer, it might
     cause an error. Try to give an array, even for single numbers

    C     -> offset in y-axis

    A     -> amplitude

    sigma -> sigma-value of Gaussian, where we extract the FWHM of cross-correlation.

    tau   -> time constant of exponential decay, bigger values corresponds to shorter
    life time.

    t_0   -> time-zero where time is zero, a singularity where nothing happens.
    """
    # C is the offset
    # t_0 is t zero
    f = np.zeros(t.size)
    g = np.zeros(t.size)
    for i in range(t.size):
        expdecayv = (t[i] >= t_0)*np.exp((t_0-t[i])/tau)
        gaussianv = np.exp(-(t_0-t[i])**2/(2*sigma**2))
        f[i] = expdecayv
        g[i] = gaussianv
    return C-A*sci.convolve(f, g, mode='same')


def fourierfitfunc(t, C, A, sigma, tau, t_0):
    """
    This is the fit function similar 'gedc' function. But instead of using the conventional 'scipy' package
    for convolution, this function takes the Fourier transform of a gaussian and exponential, multiplies
    them and then inverse Fourier transform is taken to find the convoluted function. Similar to 'gedc',
    in here also we can see the 'bump'. This can be solved in the same way as centralizing the exponential
    function.
    Returns convoluted gauusian * exponential array

    t     -> the time array. If you supply the function with a float or integer, it might
     cause an error. Try to give an array, even for single numbers

    C     -> offset in y-axis

    A     -> amplitude

    sigma -> sigma-value of Gaussian, where we extract the FWHM of cross-correlation.

    tau   -> time constant of exponential decay, bigger values corresponds to shorter
    life time.

    t_0   -> time-zero where time is zero, a singularity where nothing happens.
    """
    gaussianv = gaussian(t, sigma, t_0)
    expdecayv = expdecay(t, tau, t_0)
    nug, gaussianfourier = ft_eq(t, gaussianv)
    nue, expfourier = ft_eq(t, expdecayv)
    tgg = expfourier*gaussianfourier
    t1, convv = ift_eq(nug, tgg)
    return C-A*convv

############################################################################################
############################################################################################

# upcoming block of functions are for fourier transform used inside fourierfitfunc and polar fourier analysis


def ft_eq(t, X, omega=False, zero_centred=True, absolute=False):
    """
    Implemented from the course material; Signal Analysis in Physics: from Fourier transformation and
    sampling to lock-in amplifier, Summer Semester 2020, FU Berlin, Credit: Tobias Kampfrath & Alexander Chekhov
    Generates a specified periodic waveform

    Returns f, Fx, both same length as t and x
    If omega is set to true, then omega is returned instead of f
    If zero_centred, then 0 Hz is shifted to the centre of the spectrum (Default)
    If absolute is true, abs(Fx) is returned instead of Fx
    """
    N = len(X)
    dt = (t[-1]-t[0])/(N-1)
    F_X = fft.fft(X)*dt  # correct the FT by multiplying with dt

    # Creating the omega axis - by default 0Hz is at the first entry of the array:
    nu = fft.fftshift(fft.fftfreq(N, dt))

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


def ift_eq(nu, F_X, omega=False, was_zero_centred=True):
    """
    Implemented from the course material; Signal Analysis in Physics: from Fourier transformation and
    sampling to lock-in amplifier, Summer Semester 2020, FU Berlin, Credit: Tobias Kampfrath & Alexander Chekhov
    Generates a specified periodic waveform

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

############################################################################################
############################################################################################

# upcoming functions are to find the FWHM of a function(mostly gaussian) without using sigma.


def lin_interp(x, y, i, half):
    """
    This function is called inside the half_max_x. Implemented from:
    https://stackoverflow.com/questions/49100778/fwhm-calculation-using-python
    """
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))


def half_max_x(x, y):
    """
    Find the FWHM of a symmetric function. Implemented from:
    https://stackoverflow.com/questions/49100778/fwhm-calculation-using-python
    """
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]

############################################################################################
############################################################################################
# Functions below are used for cross-correlation fitting.


def xc_fit(name, path, slice_down, slice_up, sigma, tau, FWHM_pump, mode='corrected', crashset='none', encor='corrected', encor_var = 1.65, plot_xc = True):
    """
    Cross correlation function to correct from arbitrary delay values of delay stage to real delay values. It first
    returns a plot of the energy-time, fit function guess and fit function itself, together with printed values of
    sigma, tau, t_0, FWHM_xc, FWHM_probe. After, initiating the experimental setup, one can do an experiment of a
    delay scan together with this function to determine t_0 for further experiments. Also, if one optimize the cross
    correlation of the beam, FWHM_xc can be used to check the full width half maximum of the overlapped beam.
    Second; several parameters of fit that is listed below after given parameters. More detailed explanation with data
    is under the Jupyter Notebook 'cross_correlation_functionipynb'.

    given parameters:

    name               -> name of the data

    path               -> path to data

    slice_down         -> sliced lower energy to select which part of the energy will be added for cross-correlation fitting.
    Higer values are recommended in order to suppress the direct population of lower bands, so the cross-correlation
    can be more accurate.

    slice_up           -> sliced upper energy to select which part of the energy will be added for cross-correlation fitting.
    Higher values are recommended in order to suppress the direct population of lower bands, so the cross-correlation
    can be more accurate.

    sigma              -> initial guess for the sigma value should be given by the user.

    tau                -> initial guess for the tau value should be given by the user.

    FWHM_pump          -> To calculate the the full width half maximum of the probe, FWHM of the pump should be given. This value
    is measured with an Auto-correlator. An arbitrary value can be given, if only FWHM of the cross-correlation needed.

    mode               -> 'corrected' or 'none' (optional). It will map delay scan values to a new array where fitted t_0 value is
    zero.

    crashset           -> crashset='none' or any value('converted') (optional). if 'none', this function will also import the data.
    Initially designed to combine two data but will be deprecated soon.

    encor              -> encor='corrected' or none (optional). If corrected selected the energy will be shifted by the global
    variable 'dif_dp', only for image type of data. For the delay data, energy correction is done inside the function 'xc_fit'
    together with delay correction.
    This value should be selected by the user for appropriate physical system. For the case
    of Sb2Te3 in energy conversion range of 1 to 3eV, this value is approximately ~1.65eV.
    For angle data, this value can not be given as an optional argument as the converted data and should be change
    inside 'funcs.py'.

    encor_var -> This value should be selected by the user for appropriate physical system.
    For the case of Sb2Te3 in energy conversion range of 1 to 3eV, this value is approximately ~1.65eV.
    If you want to define the energy correction in respect to fermi energy, make DIRAC_POINT_ORIGIN_ENERGY
    inside 'funcs.py' value 0 and just use 'encor_val'. If you are doing experiments in another topological
    insulators, change the DIRAC_POINT_ORIGIN_ENERGY to appropriate for that system.
    for converted data(image or delay) in dataimport_conv, for angle-image in dataimport, for angle-delay in xc_fit.

    plot_xc             -> False or True. If True, it will plot the xc_fits and print the sigma, FWHM etc. values.


    returns:

    sigma               -> fit sigma value

    FWHM_xc             -> full width half maximum of the cross-correlation

    FWHM_probe          -> full width half maximum of the cross-correlation, calculated from FWHM of pump.

    datasetsum          -> DataArray object that sums up all the DataArrays inside of the Dataset. For further plotting,
    and analysis of the delay experiment, this is most useful object.

    slice_down          -> return the user given slice_down

    slice_up            -> return the user given slice_up

    delayvaluesasarrays -> initially same with t

    countvaluesasarray  -> initially same with f_fit

    t                   -> time array. corrected if mode is selected 'corrected'.

    f_ig                -> initially guessed convoluted function

    f_fit               -> fitted convoluted function

    tau                 -> fit tau value

    dataset             -> dataset object with all the DataArrays inside together with corrected delay if mode is 'corrected'.

    t_0                 -> real time zero value.

    encor               -> user given value.
    """

    dif_dp = DIRAC_POINT_ORIGIN_ENERGY - encor_var

    if crashset != 'none':
        dataset = name
    else:
        dataset = dataimport(name, path, 'angle', 'delay')

    taken_value = 'delay'

    if encor != 'converted':
        # this is energy correction, referenced to dirac point.
        dataset.coords['Energy'] = dataset.energy + dif_dp
        dataset = dataset.swap_dims({'energy': 'Energy'})

    # if the mode is 'corrected', it will map delay scan values to a new array where fitted
    # t_0 value is zero.
    if mode == 'corrected':
        parameter_space = xc_fit_rep(
            dataset, slice_down, slice_up, sigma, tau, taken_value, encor)

        # changing the
        dataset.coords['Delay'] = dataset.delay - parameter_space[1]
        dataset = dataset.swap_dims({'delay': 'Delay'})
        taken_value = 'Delay'
        t_0 = -1*parameter_space[1]

    parameter_space_1 = xc_fit_rep(
        dataset, slice_down, slice_up, sigma, tau, taken_value, encor)

    datasetsum = parameter_space_1[2]
    delayvaluesasarrays = parameter_space_1[3]
    countvaluesasarray = parameter_space_1[4]
    t = parameter_space_1[5]
    f_ig = parameter_space_1[6]
    f_fit = parameter_space_1[7]
    tau = parameter_space_1[8]

    sigma = parameter_space_1[0]

    if mode != 'corrected':
        t_0 = -1*parameter_space_1[1]

    FWHM_xc = 2*np.sqrt(2*np.log(2))*sigma
    FWHM_probe = np.sqrt(FWHM_xc**2-FWHM_pump**2)

    if plot_xc == True :
        # print values
        print('$sigma=$', sigma)
        print('$tau=$', tau)
        print('$t_{0}=$', t_0)
        print('$FWHM_{xc}=$', FWHM_xc)
        print('$FWHM_{probe}=$', FWHM_probe)

        # plotting
        gs = gridspec.GridSpec(2, 2)

        plt.figure()
        ax = plt.subplot(gs[0, :])  # row 0, span all columns
        datasetsum.sum(dim=('y', 'x')).T.plot()
        plt.axhline(y=slice_down, color='r', linestyle='--')
        plt.axhline(y=slice_up, color='r', linestyle='--')
        plt.xlabel('delay (fs)')
        plt.ylabel('$E-E_F (eV)$')

        ax = plt.subplot(gs[1, 0])  # row 1, col 0
        plt.plot(t, f_ig, label='fit with initial guess')
        plt.plot(delayvaluesasarrays, countvaluesasarray, label='data')
        plt.xlabel('delay (fs)')
        plt.ylabel('counts')
        plt.legend()

        ax = plt.subplot(gs[1, 1])  # row 1, col 1
        plt.plot(t, f_fit, label='fit')
        plt.plot(delayvaluesasarrays, countvaluesasarray, label='data')
        plt.xlabel('delay (fs)')
        plt.ylabel('counts')

        if encor == 'converted' and \
           crashset != 'none':
            plt.suptitle('delay cross-correlation')
        else:
            plt.suptitle(name)

        plt.tight_layout()
        plt.legend()

    return sigma, FWHM_xc, FWHM_probe, datasetsum, slice_down, slice_up, delayvaluesasarrays, countvaluesasarray, t, f_ig, f_fit, tau, dataset, t_0, encor


def xc_fit_rep(dataset, slice_down, slice_up, sigma, tau, taken_value, encor):
    """
    This function called inside the xc_fit function. It first takes the repetation value n in order to add the datasets up
    to each other for better statistics. Use 1.25 and 1.5 for slice_down and slice_up for the case
    of Sb2Te3 in energy conversion range of 1 to 3eV, corrected with the value 1.65eV.
    """

    if encor == 'converted':
        datasetsum = dataset
    else:
        datasetsum = dataset.datasetsum

    if encor != 'none':
        selectedxc = datasetsum.sum(dim=('y', 'x')).sel(
            Energy=slice(slice_down, slice_up)).sum(dim='Energy')
    else:
        selectedxc = datasetsum.sum(dim=('y', 'x')).sel(
            energy=slice(slice_down, slice_up)).sum(dim='energy')

    countvaluesasarray = selectedxc.values
    delayvaluesasarrays = selectedxc.coords[taken_value].values

    # initial values for fitting C, A, offset, sigma, tao, tzero in order. To fit the guess to the same
    # window with the data, change tzero value to an appropriate one.
    p0 = [np.amin(countvaluesasarray), np.amax(countvaluesasarray)/1.6, sigma, tau,
          delayvaluesasarrays[np.where(countvaluesasarray == np.amax(countvaluesasarray))][0]]

    # delay stage can be in 'negative' positions. To prevent
    if delayvaluesasarrays[delayvaluesasarrays.size-1] > delayvaluesasarrays[0]:
        t = np.arange(
            delayvaluesasarrays[0], delayvaluesasarrays[delayvaluesasarrays.size-1], 10)
    else:
        t = np.arange(
            delayvaluesasarrays[delayvaluesasarrays.size-1], delayvaluesasarrays[0], 10)

    f_ig = convolutedfitfunc(t, *p0)
    popt, pcov = curve_fit(
        convolutedfitfunc, delayvaluesasarrays, countvaluesasarray, p0)
    f_fit = convolutedfitfunc(t, *popt)
    sigma = popt[2]
    tau = popt[3]
    t_zero_value = popt[4]
    return sigma, t_zero_value, datasetsum, delayvaluesasarrays, countvaluesasarray, t, f_ig, f_fit, tau


def mask_1fold(shape_0, shape_1, centre, radius_high, radius_low, angle_range_1):
    """
    Return a boolean mask for semi-circle type arcs with a non-central radii. Modified from:
    https://stackoverflow.com/questions/18352973/mask-a-circular-sector-in-a-numpy-array

    To use it with the DataArray object, apply it as DataArray.where(mask_1fold).
    Warning: it returns a numpy array of the object.

    shape_0       -> x size of the dataArray

    shape_1       -> y size of the dataArray

    centre        -> center of the dataArray, given as (y,x).

    radius_high   -> outer radius value for arcs

    radius_low    -> inner radius value for arcs

    angle_range_1 -> angle range in degrees referring to unit circle for the arc.
    """

    x, y = np.ogrid[:shape_0, :shape_1]
    cx, cy = centre

    tmin_1, tmax_1 = np.deg2rad(angle_range_1)

    # ensure stop angle > start angle
    if tmax_1 < tmin_1:
        tmax_1 += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta_1 = np.arctan2(x-cx, y-cy) - tmin_1

    # wrap angles between 0 and 2*pi
    theta_1 %= (2*np.pi)

    # circular mask
    circmask_high = r2 <= radius_high*radius_high
    circmask_low = r2 <= radius_low*radius_low
    circmask = circmask_high ^ circmask_low

    # angular mask
    anglemask_1 = theta_1 <= (tmax_1-tmin_1)

    return circmask*anglemask_1


def mask_3fold(shape_0, shape_1, centre, radius_high, radius_low, angle_range_1, angle_range_2, angle_range_3):
    """
    Return a boolean mask for a triac arcs with a non-central radii. Modified from:
    https://stackoverflow.com/questions/18352973/mask-a-circular-sector-in-a-numpy-array

    To use it with the DataArray object, apply it as DataArray.where(mask_3fold).
    Warning: it returns a numpy array of the object.

    shape_0       -> x size of the dataArray

    shape_1       -> y size of the dataArray

    centre        -> center of the dataArray, given as (y,x).

    radius_high   -> outer radius value for arcs

    radius_low    -> inner radius value for arcs

    angle_range_1 -> angle range in degrees referring to unit circle for 1st arc.

    angle_range_2 -> angle range in degrees referring to unit circle for 2nd arc.

    angle_range_3 -> angle range in degrees referring to unit circle for 3rd arc.
    """

    x, y = np.ogrid[:shape_0, :shape_1]
    cx, cy = centre

    tmin_1, tmax_1 = np.deg2rad(angle_range_1)

    tmin_2, tmax_2 = np.deg2rad(angle_range_2)

    tmin_3, tmax_3 = np.deg2rad(angle_range_3)

    # ensure stop angle > start angle
    if tmax_1 < tmin_1:
        tmax_1 += 2*np.pi

    if tmax_2 < tmin_2:
        tmax_2 += 2*np.pi

    if tmax_3 < tmin_3:
        tmax_3 += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta_1 = np.arctan2(x-cx, y-cy) - tmin_1

    theta_2 = np.arctan2(x-cx, y-cy) - tmin_2

    theta_3 = np.arctan2(x-cx, y-cy) - tmin_3

    # wrap angles between 0 and 2*pi
    theta_1 %= (2*np.pi)

    theta_2 %= (2*np.pi)

    theta_3 %= (2*np.pi)

    # circular mask
    circmask_high = r2 <= radius_high*radius_high
    circmask_low = r2 <= radius_low*radius_low
    circmask = circmask_high ^ circmask_low

    # angular mask
    anglemask_1 = theta_1 <= (tmax_1-tmin_1)

    anglemask_2 = theta_2 <= (tmax_2-tmin_2)

    anglemask_3 = theta_3 <= (tmax_3-tmin_3)

    return circmask*anglemask_1 + circmask*anglemask_2 + circmask*anglemask_3


def mask_3fold_sym(shape_0, shape_1, centre, radius_high, radius_low, start_angle, wide):
    """
    Return a boolean mask for a symmetric triac arcs with a non-central radii. Modified from:
    https://stackoverflow.com/questions/18352973/mask-a-circular-sector-in-a-numpy-array

    To use it with the DataArray object, apply it as DataArray.where(mask_3fold_sym).
    Warning: it returns a numpy array of the object.

    shape_0     -> x size of the dataArray

    shape_1     -> y size of the dataArray

    centre      -> center of the dataArray, given as (y,x).

    radius_high -> outer radius value for arcs

    radius_low  -> inner radius value for arcs

    start_angle -> start angle referring to unit circle

    wide        -> \pm wide will be added to start_angle and other 2 arcs.
    """

    angle_range_1 = (start_angle - wide, start_angle + wide)

    angle_range_2 = (start_angle - wide + 120, start_angle + wide + 120)

    angle_range_3 = (start_angle - wide + 240, start_angle + wide + 240)

    x, y = np.ogrid[:shape_0, :shape_1]
    cx, cy = centre

    tmin_1, tmax_1 = np.deg2rad(angle_range_1)

    tmin_2, tmax_2 = np.deg2rad(angle_range_2)

    tmin_3, tmax_3 = np.deg2rad(angle_range_3)

    # ensure stop angle > start angle
    if tmax_1 < tmin_1:
        tmax_1 += 2*np.pi

    if tmax_2 < tmin_2:
        tmax_2 += 2*np.pi

    if tmax_3 < tmin_3:
        tmax_3 += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta_1 = np.arctan2(x-cx, y-cy) - tmin_1

    theta_2 = np.arctan2(x-cx, y-cy) - tmin_2

    theta_3 = np.arctan2(x-cx, y-cy) - tmin_3

    # wrap angles between 0 and 2*pi
    theta_1 %= (2*np.pi)

    theta_2 %= (2*np.pi)

    theta_3 %= (2*np.pi)

    # circular mask
    circmask_high = r2 <= radius_high*radius_high
    circmask_low = r2 <= radius_low*radius_low
    circmask = circmask_high ^ circmask_low

    # angular mask
    anglemask_1 = theta_1 <= (tmax_1-tmin_1)

    anglemask_2 = theta_2 <= (tmax_2-tmin_2)

    anglemask_3 = theta_3 <= (tmax_3-tmin_3)

    return circmask*anglemask_1 + circmask*anglemask_2 + circmask*anglemask_3


def mask_6fold_sym(shape_0, shape_1, centre, radius_high, radius_low, start_angle, wide):
    """
    Return a boolean mask for six symmetric arcs with a non-central radii. Modified from:
    https://stackoverflow.com/questions/18352973/mask-a-circular-sector-in-a-numpy-array

    To use it with the DataArray object, apply it as DataArray.where(mask_6fold_sym).
    Warning: it returns a numpy array of the object.

    shape_0     -> x size of the dataArray

    shape_1     -> y size of the dataArray

    centre      -> center of the dataArray, given as (y,x).

    radius_high -> outer radius value for arcs

    radius_low  -> inner radius value for arcs

    start_angle -> start angle referring to unit circle

    wide        -> \pm wide will be added to start_angle and other 5 arcs.
    """

    angle_range_1 = (start_angle - wide, start_angle + wide)

    angle_range_2 = (start_angle - wide + 60, start_angle + wide + 60)

    angle_range_3 = (start_angle - wide + 120, start_angle + wide + 120)

    angle_range_4 = (start_angle - wide + 180, start_angle + wide + 180)

    angle_range_5 = (start_angle - wide + 240, start_angle + wide + 240)

    angle_range_6 = (start_angle - wide + 300, start_angle + wide + 300)

    x, y = np.ogrid[:shape_0, :shape_1]
    cx, cy = centre

    tmin_1, tmax_1 = np.deg2rad(angle_range_1)

    tmin_2, tmax_2 = np.deg2rad(angle_range_2)

    tmin_3, tmax_3 = np.deg2rad(angle_range_3)

    tmin_4, tmax_4 = np.deg2rad(angle_range_4)

    tmin_5, tmax_5 = np.deg2rad(angle_range_5)

    tmin_6, tmax_6 = np.deg2rad(angle_range_6)

    # ensure stop angle > start angle
    if tmax_1 < tmin_1:
        tmax_1 += 2*np.pi

    if tmax_2 < tmin_2:
        tmax_2 += 2*np.pi

    if tmax_3 < tmin_3:
        tmax_3 += 2*np.pi

    if tmax_4 < tmin_4:
        tmax_4 += 2*np.pi

    if tmax_5 < tmin_5:
        tmax_5 += 2*np.pi

    if tmax_6 < tmin_6:
        tmax_6 += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)

    theta_1 = np.arctan2(x-cx, y-cy) - tmin_1

    theta_2 = np.arctan2(x-cx, y-cy) - tmin_2

    theta_3 = np.arctan2(x-cx, y-cy) - tmin_3

    theta_4 = np.arctan2(x-cx, y-cy) - tmin_4

    theta_5 = np.arctan2(x-cx, y-cy) - tmin_5

    theta_6 = np.arctan2(x-cx, y-cy) - tmin_6

    # wrap angles between 0 and 2*pi
    theta_1 %= (2*np.pi)

    theta_2 %= (2*np.pi)

    theta_3 %= (2*np.pi)

    theta_4 %= (2*np.pi)

    theta_5 %= (2*np.pi)

    theta_6 %= (2*np.pi)

    # circular mask
    circmask_high = r2 <= radius_high*radius_high
    circmask_low = r2 <= radius_low*radius_low
    circmask = circmask_high ^ circmask_low

    # angular mask
    anglemask_1 = theta_1 <= (tmax_1-tmin_1)

    anglemask_2 = theta_2 <= (tmax_2-tmin_2)

    anglemask_3 = theta_3 <= (tmax_3-tmin_3)

    anglemask_4 = theta_4 <= (tmax_4-tmin_4)

    anglemask_5 = theta_5 <= (tmax_5-tmin_5)

    anglemask_6 = theta_6 <= (tmax_6-tmin_6)

    return circmask*anglemask_1 + circmask*anglemask_2 + circmask*anglemask_3 + circmask*anglemask_4 + circmask*anglemask_5 + circmask*anglemask_6


def sig_waveform(t, T, shape, width_rel, t0_rel=0.5):
    """
    Implemented from the course material; Signal Analysis in Physics: from Fourier transformation and
    sampling to lock-in amplifier, Summer Semester 2020, FU Berlin, Credit: Tobias Kampfrath & Alexander Chekhov
    Generates a specified periodic waveform

    Arguments:
    t         -> Time array. Should be equidistant with spacing dt.
    T         -> Period length. Should be an integer multiple of dt.
    shape     -> 'square', 'sawtooth', 'triangle', 'sine', 'delta', 'gauss' / 'normal', 'cosine'
    width_rel -> Relative with of the 'shape' element, rest is zero. 0<=width_rel<=1
    t0_rel    -> Center time of the 'shape' element relative to
                 interval [0, T]. Default value is 0.5

    The waveform maximum is normalized to 1
    """
    # shift the first time array element to be zero
    t = t - t[0]

    # exploit periodicity: Map t axis on interval [0,1)-t0_rel
    x = np.mod(t/T, 1) - t0_rel

    # find the minimal increment on [0,1)-t0_rel interval
    dx = abs(min(x[:-1]-x[1:]))

    h = (abs(x) <= width_rel/2)*1.  # =1 if |x|<width_rel/2, =0 otherwise

    if shape == 'square':
        return h

    elif shape == 'sawtooth':
        return h*(x/width_rel+0.5)

    elif shape == 'triangle':
        return h*(1-abs(x)*2/width_rel)

    elif shape == 'sine':
        return h*sin(x*2*pi/width_rel)

    elif shape == 'delta':
        return abs(x - x[abs(x).argmin()]) < dx/2

    elif shape == 'gauss' or shape == 'normal':
        return exp(-(x/width_rel*2.335)**2/2)

    elif shape == 'cosine':
        return h*cos(x*2*pi/width_rel)

    elif shape == 'singlesideexp':
        return (x >= 0)*exp(-x/width_rel)  # Do not use heaviside here

    elif shape == 'doublesideexp':
        return exp(-abs(x)/width_rel)

    else:
        raise ValueError('Invalid Shape!')


def binning(data, bin_number_en, bin_number_xy):
    """
    Binning the data to lower bin numbers. Returns binned data.

    bin_number_en -> bin number for energy.

    bin_number_xy -> bin numbers for x,y or kx, ky.
    """

    # binning energy values
    # binning to lower values of original bin, have to be integer multiplier of the original bin value.
    # Want to make it a function if i find a way to put variables inside xarray functions.
    if data.Energy.size < bin_number_en or \
            data.x.size < bin_number_xy:

        print('Please choose a lower value than binning of the current data!')
        data_binned = data

    else:

        rebin = bin_number_en
        rbmp = int(data['Energy'].size/rebin)
        data_binned = data.coarsen(Energy=rbmp).sum()

        # binning x-y values
        # binning to lower values of original bin, have to be integer multiplier of the original bin value.
        # Want to make it a function if i find a way to put variables inside xarray functions.
        rebin = bin_number_xy
        rbmp = int(data['x'].size/rebin)
        data_binned = data_binned.coarsen(x=rbmp).sum().coarsen(y=rbmp).sum()

    return data_binned


def find_nearest_index(array, value):
    """
    Implemented from:
    https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def normalize_data(data):
    """
    Mapping the data to values between 0 and 1. Useful for comparison between different datasets.
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def data_av(data_left, data_right, min_plot_num_en=0.1, max_plot_num_en=0.7):
    """
    This normalizes the higher dataset's count rate to lower dataset's count rate in selected regions of energy.

    data_left       -> left circular polarized data

    data_right      -> left circular polarized data

    min_plot_num_en -> lower energy value, which the normalization starts

    max_plot_num_en -> higher energy value, which the normalization ends
    """
    if data_right.sel(Energy=slice(min_plot_num_en, max_plot_num_en)).sum() > data_left.sel(Energy=slice(min_plot_num_en, max_plot_num_en)).sum():
        data_left = data_left*(data_right.sel(Energy=slice(min_plot_num_en, max_plot_num_en)
                                              ).sum()/data_left.sel(Energy=slice(min_plot_num_en, max_plot_num_en)).sum())
    else:
        data_right = data_right*(data_left.sel(Energy=slice(min_plot_num_en, max_plot_num_en)).sum(
        )/data_right.sel(Energy=slice(min_plot_num_en, max_plot_num_en)).sum())
    return data_left, data_right

def ds_time_b(t_0, start, stop, steps, rep, expt):
    """
    Calculate how much time it will take delayscan to finish and print the real values that delay stage takes.
    Returns a print of the calculated values.

    t_0   -> time zero of the current sample

    start -> start value in fs

    stop  -> stop value in fs

    steps -> steps inside delay scan experiment

    rep   -> how much repetition will be done between start and stop values.

    expt  -> exposure time for a single step.
    """
    real_start = t_0-start
    real_stop = t_0-stop
    delayscant = (steps*rep*expt)/(60*60)
    print('delay scan will take', str(
        datetime.timedelta(hours=delayscant)), 'hours to complete')
    print('delay scan region is from', real_start, ' fs to', real_stop, 'fs')
    print('Step size:', (stop-start)/steps)
    print('Exposure time per image:', rep*expt)
    return
