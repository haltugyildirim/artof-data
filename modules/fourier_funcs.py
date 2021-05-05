import h5py
import numpy as np
import xarray as xr
import matplotlib as mp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pytexit import py2tex
import matplotlib.gridspec as gridspec
from scipy.constants import m_e,hbar,e

#fourier analysis specific functions
import abel
import cv2
import os

from .funcs import find_nearest_index, mask_6fold_sym, ft_eq, sig_waveform, ift_eq
from .plt_funcs import cdad_shc_just_func, cdad_svc_just_func

def data_selection_polar_fourier(right_data,left_data,min_plot_num_en,max_plot_num_en,data_type,mask,outer_rad,cone_wide = 30,inner_rad = 0):
    """
    Data selection for the polar fourier analysis.
    cone_wide ->  cone wide should always be 30 as this uses the 6fold masking function(under module/funcs),
                  30 * 2 * 60 = 360 to create a whole circle for mask.
    inner_rad ->  Even though one can give a higher inner_rad for this mask, during the selection of polar
    reprojected data this can be done more practically.
    """
    #If the dataset is delay, than this part is for selection of the corresponding part of data. Otherwise
    #(image or dichroism) this justget rid of the extra coordinate of delay, that only carries a single value
    if data_type == 'dichroism':
        data_in = cdad_shc_just_func(right_data,left_data,min_plot_num_en,max_plot_num_en,av='off').sum(dim='delay')
    elif data_type == 'delay':
        data_in = right_data.sel(delay_corr = delay_input, method="nearest")
    else:
        data_in = right_data.sum(dim='delay')

    #center of the data from k points. k_x and k_y center should be
    x_off = find_nearest_index(data_in.y, 0)
    y_off = find_nearest_index(data_in.x, 0)
    origin = (y_off,x_off)

    #selection for which data type is used, for dichroism energy selection is done above for others it is done here.
    #also this section is for selection of either applying a circular mask for non-centered sample to avoid artefacts
    #during polar reprojection.
    if data_type == 'dichroism':
        if mask == 'circular':
            datasel_xr = data_in.where(mask_6fold_sym(
                data_in.x.size,data_in.y.size,origin,inner_rad,outer_rad,0,cone_wide),0)
        else:
            datasel_xr = data_in
    else:
        if mask == 'circular':
            datasel_xr = data_in.sel(energy_corr=slice(
                min_plot_num_en,max_plot_num_en)).sum(dim='energy_corr').where(
                mask_6fold_sym(data_in.x.size,data_in.y.size,origin,inner_rad,outer_rad,0,cone_wide),0)
        else:
            datasel_xr = data_in.sel(energy_corr=slice(min_plot_num_en,max_plot_num_en)).sum(dim='energy_corr')

    return datasel_xr, origin

def polar_reprojection_dataarray(datasel_xr, origin):
    """
    Polar reprojection of an DataArray object from Cartesian coordinates.
    """
    #dataArray image to numpy array
    datasel = datasel_xr.values

    #From which angle the Abelian transformation for Cartesian to Polar reprojection will start. The abel
    #package always starts from 'top'(corresponding to 90 degrees in unit circle.) But during the reconstruction
    #after the Fourier transform, this creates problems as it introduce a phase difference of 90 degrees which
    #original data and reconstructed image. So angle_offset should be 0(in Unit circle), if one wants to achieve the filtered
    #reconstructed image of the Fourier components.
    angle_offset = 0

    #Projecting cartesian to polar with Abelian transformation using abel package
    PolarImage_pr, r_grid, theta_grid = abel.tools.polar.reproject_image_into_polar(datasel,origin=origin)

    #polar reprojection start angle
    PolarImage = np.roll(PolarImage_pr,int(((-1 * angle_offset+90) * PolarImage_pr.shape[1])/360))

    #phi and r coordinates for projected DataArray object
    phi_coord = np.arange(0,360,360/PolarImage[0,:].size)
    r_coord = np.arange(0,PolarImage[:,0].size)

    #DataArray creation from projected image
    data_polar = xr.DataArray(PolarImage, dims=['r','phi'], coords={'r': r_coord,'phi': phi_coord})

    return data_polar

def find_fourier(data_fourier,filter_width = 0.13,filter_small_comp = 'none', filter_zero_comp = 'none'):
    """
    This function finds the Fourier components and filters the higher frequencies
    with given filter.
    filter_width -> After the Fourier transform a ⎍ square waveform centered around zero is applied
    to get rid of higher Fourier components that corresponds to noise. User can check axs[0,3] plot
    to decide the width of the filter. If set to values higher than 0.5, filter become obsolete.
    """
    data_size = data_fourier.size

    # specify x-array and width of the response function
    x = np.arange(0, data_size)

    #initialize a zero-valued fourier transform of smeared image
    F_L = np.zeros(data_size,dtype='cfloat')

    #calculate fourier transform of data_fourier
    nu,F_L = ft_eq(x,data_fourier)

    #filter of the Fourier transform values that absolute value of Fourier transform is smaller than 1,
    #which corresponds to noise.
    if filter_small_comp != 'none':
        F_L[np.abs(F_L) < 1] = 0

    #0th components is the offset. One can get rid of to
    if filter_zero_comp != 'none':
        F_L[zero_comp_indx] = 0

    #filter higher frequencies. This will get rid of the values higher than given filter_width.
    #For good dichroic data that doesn't have strong asymmetries this step might be not
    #necessary as it is highly probable that components bigger than 3 are already zero.
    #But there can be cases that 4th(4fold), 5th(6fold) and 6Th(8th fold) are presence.
    square_sig_01 = sig_waveform(nu, 1, 'square', filter_width, t0_rel = 0.5)

    #application of filter to Fourier transform of the data_fourier
    F_L_filt = square_sig_01*F_L

    return nu, F_L, F_L_filt, square_sig_01

def find_phase_deg(F_L):
    """
    This function finds the phases of the Fourier components in range of [0,2pi) for
    a polar mapping.
    """
    data_size = F_L.size

    zero_comp_indx = int(data_size/2)

    #corresponding phase values in -pi,pi range
    phase_neg = np.angle(F_L)

    #filter of the phase values that absolute value of Fourier transform is smaller than 1,
    #which corresponds to noise.
    phase_neg[np.abs(F_L) < 1] = 0

    #finding the real 'readable' phase is trickier than it seems.
    phase = np.zeros(data_size)
    for i in range(data_size):
        indx = np.arange(-1*zero_comp_indx,zero_comp_indx)
        if indx[i] == 0:
            phase[i] = 0
        else:
            if phase_neg[i] < 0:
                phase[i] = np.rad2deg(abs(phase_neg[i] / indx[i]))
            if phase_neg[i] > 0:
                phase[i] = np.rad2deg(abs((np.deg2rad(360) - phase_neg[i]) / indx[i]))
    return phase


def find_component_transforms(nu,F_L_filt):
    """
    This function finds the inverse transform of the filtered data. additionally, it finds
    the  inverse of the first, second, third components and dual convolution of these
    components.
    """
    #inverse fourier transform after the
    x,data_fourier_filt=ift_eq(nu,F_L_filt)

    data_size = F_L_filt.size

    zero_comp_indx = int(data_size/2)

    #first component
    F_L_1 = np.copy(F_L_filt)

    F_L_1[zero_comp_indx] = 0

    F_L_1[zero_comp_indx+2] = 0
    F_L_1[zero_comp_indx-2] = 0

    F_L_1[zero_comp_indx+3] = 0
    F_L_1[zero_comp_indx-3] = 0

    x,data_fourier_1=ift_eq(nu,F_L_1)

    #second component
    F_L_2 = np.copy(F_L_filt)

    F_L_2[zero_comp_indx] = 0

    F_L_2[zero_comp_indx+1] = 0
    F_L_2[zero_comp_indx-1] = 0

    F_L_2[zero_comp_indx+3] = 0
    F_L_2[zero_comp_indx-3] = 0

    x,data_fourier_2=ift_eq(nu,F_L_2)

    #third component
    F_L_3 = np.copy(F_L_filt)

    F_L_3[zero_comp_indx] = 0

    F_L_3[zero_comp_indx+1] = 0
    F_L_3[zero_comp_indx-1] = 0

    F_L_3[zero_comp_indx+2] = 0
    F_L_3[zero_comp_indx-2] = 0

    x,data_fourier_3=ift_eq(nu,F_L_3)

    #1st and 2nd component together
    F_L_12 = np.copy(F_L_filt)

    F_L_12[zero_comp_indx+3] = 0
    F_L_12[zero_comp_indx-3] = 0

    x,data_fourier_12=ift_eq(nu,F_L_12)

    #2nd and 3rd component together
    F_L_23 = np.copy(F_L_filt)

    F_L_23[zero_comp_indx+1] = 0
    F_L_23[zero_comp_indx-1] = 0

    x,data_fourier_23=ift_eq(nu,F_L_23)

    #1st and 3rd component together
    F_L_13 = np.copy(F_L_filt)

    F_L_13[zero_comp_indx+2] = 0
    F_L_13[zero_comp_indx-2] = 0

    x,data_fourier_13=ift_eq(nu,F_L_13)

    df_filt = data_fourier_filt.real
    df_1    = data_fourier_1.real
    df_2    = data_fourier_2.real
    df_3    = data_fourier_3.real
    df_12   = data_fourier_12.real
    df_23   = data_fourier_23.real
    df_13   = data_fourier_13.real

    return df_filt, df_1, df_2, df_3, df_12, df_23, df_13


def rotate_mapping(phase_initial):
    """
    'rotates' the phase values inside array. if the frequency is an odd number
    it will add 1/2 of the frequency to phase angle if it is bigger than frequency/2,
    otherwise it will substract it.
    """
    phase_rotated = np.zeros(phase_initial.size)

    for i in range(phase_initial.size):
        if (i+1) % 2 == 0:
            phase_rotated[i] = phase_initial[i]
        else:
            if phase_initial[i] < (360 / (2*(i+1))):
                phase_rotated[i] = ((360 / (2*(i+1))) + phase_initial[i])
            else:
                phase_rotated[i] = (phase_initial[i] - (360 / (2*(i+1))))
    return phase_rotated

def polar_fourier_plot_func(right_data, left_data, min_plot_num_en, max_plot_num_en, data_type, datasel_xr, data_polar, phase, amplitude,  amplitude_filt, square_sig_01, data_fourier_all, r_down,r_up, rad_circ):

    zero_comp_indx = int(right_data.x.size/2)
    phi_coord = data_polar.phi.values
    #Plotting
    #fig, axs = plt.subplots(4,2, figsize=(7,12))
    fig, axs = plt.subplots(2,5, figsize=(30,11))

    #plotting the frequency x values as the number of maxima per signal
    nu_max_cyc = np.arange(int(-1*zero_comp_indx), zero_comp_indx)

    cdad_svc_just_func(right_data,left_data,0.25,0.6, -0.01, 0.01, 'y',av='off').plot(ax=axs[0,0])
    axs[0,0].axhline(y=min_plot_num_en ,color = 'r', linestyle = '--')
    axs[0,0].axhline(y=max_plot_num_en ,color = 'r', linestyle = '--')
    axs[0,0].set_xlabel('$k_x$')
    axs[0,0].set_ylabel('$E-E_F (eV)$')

    right_data.sel(energy_corr=slice(min_plot_num_en, max_plot_num_en)).sum(dim='energy_corr').plot(ax=axs[1,0])
    axs[1,0].set_xlabel('$k_x$')
    axs[1,0].set_ylabel('$k_y$')
    #to determine the radius of the presented circle. it needs an if conditions because of the circular masks
    #radius is either depended on x or y, this is realted to sample's center in respect to detector.
    #y_circ_rad = right_data.y[int(find_nearest_index(right_data.y,0)+outer_rad)].values
    #x_circ_rad = right_data.x[int(find_nearest_index(right_data.x,0)+outer_rad)].values
    #if y_circ_rad < x_circ_rad:
    #    rad_circ = y_circ_rad
    #else:
    #    rad_circ = x_circ_rad
    #x_position_0=find_nearest_index(right_data.x,0)
    #y_position_0=find_nearest_index(right_data.y,0)
    #rad_circ = 0.11 #right_data.x[x_position_0 - int(outer_rad * np.cos(np.arctan2(x_position_0,y_position_0)))]
    circle = plt.Circle((0, 0), rad_circ, color='r', linestyle = '--', fill=False)
    axs[1,0].add_patch(circle)

    datasel_xr.plot(ax=axs[0,1])
    axs[0,1].set_title('Cartesian')
    axs[0,1].set_xlabel('$k_x$')
    axs[0,1].set_ylabel('$k_y$')
    axs[0,1].axvline(x = 0 ,color = 'r', linestyle = '--')
    axs[0,1].axhline(y = 0 ,color = 'r', linestyle = '--')

    if data_type == 'dichroism':
        data_polar.plot(ax=axs[1,1])
    else:
        data_polar.plot(vmin=0, ax=axs[1,1])
    axs[1,1].set_title('Polar')
    axs[1,1].set_xlabel('$\phi (\degree)$')
    axs[1,1].set_ylabel('r (a.u)')

    if data_type == 'dichroism':
        data_polar.sel(r=slice(r_down,r_up)).plot(ax=axs[0,2])
    else:
        data_polar.sel(r=slice(r_down,r_up)).plot(vmin=0, ax=axs[0,2])
    axs[0,2].set_xlabel('$\phi (\degree)$')
    axs[0,2].set_ylabel('r (a.u)')

    data_polar.sel(r=slice(r_down,r_up)).sum(dim='r').plot(ax=axs[1,2])
    axs[1,2].set_xlabel('$\phi (\degree)$')
    axs[1,2].set_ylabel('counts (a.u)')

    markerline, stemlines, baseline = axs[0,3].stem(nu_max_cyc,amplitude, linefmt='--')
    plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
    plt.setp(stemlines, 'linestyle', 'dotted')
    axs[0,3].set_xlim((0,10))
    axs[0,3].set_ylabel('Amplitude (a.u)')
    axs[0,3].set_xlabel('\# maxima /cycle')
    axs[0,3].set_xticks([0,1,2,3,4,5,6,7,8,9,10])

    markerline, stemlines, baseline = axs[1,3].stem(
        nu_max_cyc,phase, 'g', markerfmt='go', linefmt='--')
    plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
    plt.setp(stemlines, 'linestyle', 'dotted')
    axs[1,3].set_xlim((0,10))
    axs[1,3].set_xticks([0,1,2,3,4,5,6,7,8,9,10])
    axs[1,3].set_ylabel('Phase ($\degree$)')
    axs[1,3].set_xlabel('\# maxima /cycle')

    markerline, stemlines, baseline = axs[0,4].stem(
        nu_max_cyc,amplitude_filt, linefmt='--',label='filtered')
    plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
    plt.setp(stemlines, 'linestyle', 'dotted')
    axs[0,4].plot(nu_max_cyc,square_sig_01,label='filter')
    axs[0,4].set_ylabel('Amplitude (a.u)')
    axs[0,4].set_xlabel('\# maxima /cycle')
    axs[0,4].set_xticks([0,1,2,3,4,5])
    axs[0,4].set_xlim((0,5))
    axs[0,4].legend()

    data_polar.sel(r=slice(r_down,r_up)).sum(dim='r').plot(ax=axs[1,4], label='raw')
    axs[1,4].plot(phi_coord,data_fourier_all[0],label='filtered')
    axs[1,4].plot(phi_coord,data_fourier_all[6],label='1 \& 3')
    axs[1,4].plot(phi_coord,data_fourier_all[1],label='1st')
    axs[1,4].plot(phi_coord,data_fourier_all[2],label='2nd')
    axs[1,4].plot(phi_coord,data_fourier_all[3],label='3rd')
    axs[1,4].axhline(y=0 ,color = 'black', linestyle = '--')
    axs[1,4].set_xlabel('$\phi (\degree)$')
    axs[1,4].set_ylabel('counts (a.u)')
    axs[1,4].legend()

    plt.tight_layout()
    return
