import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.constants import m_e, hbar, e
from .funcs import data_av, binning


SMALL_SIZE = 10
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def cdad_shc_just_func(data_right_binned, data_left_binned, min_plot_num_en, max_plot_num_en, av='on'):
    """
    Circular Dichroism Selected Horizontal Cut.
    Returns an energy cut for both direction for selected other direction.
    """

    # This averages the lower dataset's count rate to higher dataset's count rate.
    if av == 'on':
        data_left_binned, data_right_binned = data_av(
            data_left_binned, data_right_binned)

    if min_plot_num_en == max_plot_num_en:
        data_rightsel = data_right_binned.sel(
            Energy=min_plot_num_en, method="nearest")
        data_leftsel = data_left_binned.sel(
            Energy=min_plot_num_en, method="nearest")
    else:
        data_rightsel = data_right_binned.sel(Energy=slice(
            min_plot_num_en, max_plot_num_en)).sum(dim='Energy')
        data_leftsel = data_left_binned.sel(Energy=slice(
            min_plot_num_en, max_plot_num_en)).sum(dim='Energy')

    cdad_left_right = (data_rightsel-data_leftsel)/(data_rightsel+data_leftsel)
    return cdad_left_right


def cdad_svc_just_func(data_right_binned, data_left_binned, min_plot_num_en, max_plot_num_en,
                       min_plot_num_deg, max_plot_num_deg, dim, av='on'):
    """
    Circular Dichroism Selected Vertical Cut Function to be used inside cdad_dc_comparison_full.
    Returns an Energy-Angle/Momentum cut for both direction for selected other direction.
    """
    # This averages the lower dataset's count rate to higher dataset's count rate.
    if av == 'on':
        data_left_binned, data_right_binned = data_av(
            data_left_binned, data_right_binned)

    if dim == 'x':

        if max_plot_num_deg == min_plot_num_deg:
            data_rightsel = data_right_binned.sel(
                    Energy=slice(min_plot_num_en, max_plot_num_en)).sel(
                y=max_plot_num_deg, method="nearest")
            data_leftsel = data_left_binned.sel(
                    Energy=slice(min_plot_num_en, max_plot_num_en)).sel(
                y=max_plot_num_deg, method="nearest")
        else:
            data_rightsel = data_right_binned.sel(
                    Energy=slice(min_plot_num_en, max_plot_num_en)).sel(
                y=slice(min_plot_num_deg, max_plot_num_deg)).sum(dim='y')
            data_leftsel = data_left_binned.sel(
                    Energy=slice(min_plot_num_en, max_plot_num_en)).sel(
                y=slice(min_plot_num_deg, max_plot_num_deg)).sum(dim='y')
        cdad_left_right = (data_rightsel-data_leftsel) / \
            (data_rightsel+data_leftsel)

    else:

        if max_plot_num_deg == min_plot_num_deg:
            data_rightsel = data_right_binned.sel(
                    Energy=slice(min_plot_num_en, max_plot_num_en)).sel(
                x=max_plot_num_deg, method="nearest")
            data_leftsel = data_left_binned.sel(
                    Energy=slice(min_plot_num_en, max_plot_num_en)).sel(
                x=max_plot_num_deg, method="nearest")
        else:
            data_rightsel = data_right_binned.sel(
                    Energy=slice(min_plot_num_en, max_plot_num_en)).sel(
                x=slice(min_plot_num_deg, max_plot_num_deg)).sum(dim='x')
            data_leftsel = data_left_binned.sel(
                    Energy=slice(min_plot_num_en, max_plot_num_en)).sel(
                x=slice(min_plot_num_deg, max_plot_num_deg)).sum(dim='x')
        cdad_left_right = (data_rightsel-data_leftsel) / \
            (data_rightsel+data_leftsel)
    return cdad_left_right


def cdad_plot(data_right, data_left, length_subplot_in, min_plot_num_en, max_plot_num_en,
              min_plot_num_deg, max_plot_num_deg, bin_number_en,bin_number_xy, vmax_val='none', sing_val='none', av='on'):
    """
    Circular dichroism plots. To get single values in the range that user gave,
    give single_val another value than 'none'. vmax of the graphs also can be changed vmax_val.
    """

    data_right_binned = binning(data_right, bin_number_en, bin_number_xy)
    data_left_binned = binning(data_left, bin_number_en, bin_number_xy)

    print('count rate for right handed is;', data_right_binned.sum().values)
    print('count rate for left handed is;', data_left_binned.sum().values)

    # This averages the lower dataset's count rate to higher dataset's count rate.
    if av == 'on':
        data_left_binned, data_right_binned = data_av(
            data_left_binned, data_right_binned)
        print('count rate for normalized right handed is;',
              data_right_binned.sum().values)
        print('count rate for normalized left handed is;',
              data_left_binned.sum().values)

    # plots every nthimage
    grid_4fig = gridspec.GridSpec(ncols=2, nrows=length_subplot_in)
    fig = plt.figure(figsize=(10, length_subplot_in*4))
    length_subplot = 2*length_subplot_in

    for i in range(length_subplot):
        ax = fig.add_subplot(
            grid_4fig[np.floor_divide(i, 2), np.remainder(i, 2)])

        if i % 2 == 0:
            i = i/2
            z_factor = (max_plot_num_deg-min_plot_num_deg)/(length_subplot_in)
            z = i*z_factor+min_plot_num_deg

            if sing_val != 'none':
                data_rightsel = data_right_binned.sel(y=z, method="nearest")
                data_leftsel = data_left_binned.sel(y=z, method="nearest")
            else:
                data_rightsel = data_right_binned.sel(
                    y=slice(z, z+z_factor)).sum(dim='y')
                data_leftsel = data_left_binned.sel(
                    y=slice(z, z+z_factor)).sum(dim='y')

            cdad_left_right = (data_rightsel-data_leftsel) / \
                (data_rightsel+data_leftsel)

            if vmax_val == 'none':
                cdad_left_right.plot()
            else:
                cdad_left_right.plot(vmax=vmax_val)
            plt.ylim(min_plot_num_en, max_plot_num_en)
            plt.tight_layout()
            plt.xlabel('$ \Phi_x$')
            plt.ylabel('$E-E_F (eV)$')
            if sing_val != 'none':
                plt.title('at %5.2f$\degree$' % z)
            else:
                plt.title('%5.2f$\degree$ to%5.2f$\degree$' % (z, z+z_factor))

        else:
            i = (i-1)/2
            z_factor = (max_plot_num_deg-min_plot_num_deg)/(length_subplot_in)
            z = i*z_factor+min_plot_num_deg

            if sing_val != 'none':
                data_rightsel = data_right_binned.sel(x=z, method="nearest")
                data_leftsel = data_left_binned.sel(x=z, method="nearest")
            else:
                data_rightsel = data_right_binned.sel(
                    x=slice(z, z+z_factor)).sum(dim='x')
                data_leftsel = data_left_binned.sel(
                    x=slice(z, z+z_factor)).sum(dim='x')

            cdad_left_right = (data_rightsel-data_leftsel) / \
                (data_rightsel+data_leftsel)

            if vmax_val == 'none':
                cdad_left_right.plot()
            else:
                cdad_left_right.plot(vmax=vmax_val)
            plt.ylim(min_plot_num_en, max_plot_num_en)
            plt.tight_layout()
            plt.xlabel('$ \Theta_y$')
            plt.ylabel('$E-E_F (eV)$')
            if sing_val != 'none':
                plt.title('at %5.2f$\degree$' % z)
            else:
                plt.title('%5.2f$\degree$ to%5.2f$\degree$' % (z, z+z_factor))

    # plots every nthimage
    grid_4fig = gridspec.GridSpec(ncols=2, nrows=length_subplot_in)
    fig = plt.figure(figsize=(10, length_subplot_in*4))

    for i in range(length_subplot_in):
        ax = fig.add_subplot(
            grid_4fig[np.floor_divide(i, 2), np.remainder(i, 2)])

        z_factor = (max_plot_num_en-min_plot_num_en)/length_subplot_in
        z = i*z_factor+min_plot_num_en

        if sing_val != 'none':
            data_rightsel = data_right_binned.sel(
                Energy=z, method="nearest")
            data_leftsel = data_left_binned.sel(
                Energy=z, method="nearest")
        else:
            data_rightsel = data_right_binned.sel(
                Energy=slice(z, z+z_factor)).sum(dim='Energy')
            data_leftsel = data_left_binned.sel(
                Energy=slice(z, z+z_factor)).sum(dim='Energy')

        cdad_left_right = (data_rightsel-data_leftsel) / \
            (data_rightsel+data_leftsel)

        if vmax_val == 'none':
            cdad_left_right.plot()
        else:
            cdad_left_right.plot(vmax=vmax_val)
        plt.tight_layout()
        plt.xlabel('$ \Phi_x$')
        plt.ylabel('$ \Theta_y$')
        if sing_val != 'none':
            plt.title('at %5.2feV' % z)
        else:
            plt.title('%5.2feV to%5.2feV' % (z, z+z_factor))
    return


def cdad_plot_svc(data_right_binned, data_left_binned, min_plot_num_en, max_plot_num_en,
                  min_plot_num_deg_y, max_plot_num_deg_y, min_plot_num_deg_x, max_plot_num_deg_x, vmax_val='none', av='on'):
    """
    Circular Dichroism Selected Vertical Cut.
    Returns an Energy-Angle cut for both direction for selected other direction.
    """
    # This averages the lower dataset's count rate to higher dataset's count rate.
    if av == 'on':
        data_left_binned, data_right_binned = data_av(
            data_left_binned, data_right_binned)

    length_subplot_in = 1

    # plots every nthimage
    grid_4fig = gridspec.GridSpec(ncols=2, nrows=length_subplot_in)
    fig = plt.figure(figsize=(10, length_subplot_in*4))
    length_subplot = 2*length_subplot_in

    for i in range(length_subplot):
        ax = fig.add_subplot(
            grid_4fig[np.floor_divide(i, 2), np.remainder(i, 2)])

        if i % 2 == 0:
            i = i/2
            z_factor = (max_plot_num_deg_y-min_plot_num_deg_y) / \
                (length_subplot_in)
            z = i*z_factor+min_plot_num_deg_y

            if max_plot_num_deg_y == min_plot_num_deg_y:
                data_rightsel = data_right_binned.sel(y=z, method="nearest")
                data_leftsel = data_left_binned.sel(y=z, method="nearest")
            else:
                data_rightsel = data_right_binned.sel(
                    y=slice(z, z+z_factor)).sum(dim='y')
                data_leftsel = data_left_binned.sel(
                    y=slice(z, z+z_factor)).sum(dim='y')
            cdad_left_right = (data_rightsel-data_leftsel) / \
                (data_rightsel+data_leftsel)

            if vmax_val == 'none':
                cdad_left_right.plot()
            else:
                cdad_left_right.plot(vmax=vmax_val)

            plt.ylim(min_plot_num_en, max_plot_num_en)
            plt.tight_layout()
            plt.xlabel('$ \Phi_x$')
            plt.ylabel('$E-E_F (eV)$')
            if max_plot_num_deg_y == min_plot_num_deg_y:
                plt.title('at %5.2f$\degree$' % z)
            else:
                plt.title('%5.2f$\degree$ to%5.2f$\degree$' % (z, z+z_factor))

        else:
            i = (i-1)/2
            z_factor = (max_plot_num_deg_x-min_plot_num_deg_x) / \
                (length_subplot_in)
            z = i*z_factor+min_plot_num_deg_x

            if max_plot_num_deg_x == min_plot_num_deg_x:
                data_rightsel = data_right_binned.sel(x=z, method="nearest")
                data_leftsel = data_left_binned.sel(x=z, method="nearest")
            else:
                data_rightsel = data_right_binned.sel(
                    x=slice(z, z+z_factor)).sum(dim='x')
                data_leftsel = data_left_binned.sel(
                    x=slice(z, z+z_factor)).sum(dim='x')
            cdad_left_right = (data_rightsel-data_leftsel) / \
                (data_rightsel+data_leftsel)

            if vmax_val == 'none':
                cdad_left_right.plot()
            else:
                cdad_left_right.plot(vmax=vmax_val)

            plt.ylim(min_plot_num_en, max_plot_num_en)
            plt.tight_layout()
            plt.xlabel('$ \Theta_y$')
            plt.ylabel('$E-E_F (eV)$')
            if max_plot_num_deg_x == min_plot_num_deg_x:
                plt.title('at %5.2f$\degree$' % z)
            else:
                plt.title('%5.2f$\degree$ to%5.2f$\degree$' % (z, z+z_factor))
    return


def cdad_shc(data_right_binned, data_left_binned, min_plot_num_en, max_plot_num_en, vmax_val='none', av='on'):
    """
    Circular Dichroism Selected Horizontal Cut.
    Returns an energy cut for both direction for selected other direction.
    """

    # This averages the lower dataset's count rate to higher dataset's count rate.
    if av == 'on':
        data_left_binned, data_right_binned = data_av(
            data_left_binned, data_right_binned)

    if min_plot_num_en == max_plot_num_en:
        data_rightsel = data_right_binned.sel(
            Energy=min_plot_num_en, method="nearest")
        data_leftsel = data_left_binned.sel(
            Energy=min_plot_num_en, method="nearest")
    else:
        data_rightsel = data_right_binned.sel(Energy=slice(
            min_plot_num_en, max_plot_num_en)).sum(dim='Energy')
        data_leftsel = data_left_binned.sel(Energy=slice(
            min_plot_num_en, max_plot_num_en)).sum(dim='Energy')

    cdad_left_right = (data_rightsel-data_leftsel)/(data_rightsel+data_leftsel)

    if vmax_val == 'none':
        cdad_left_right.plot(aspect=1.25, size=5)
    else:
        cdad_left_right.plot(aspect=1.25, size=5, vmax=vmax_val)
    plt.tight_layout()
    plt.xlabel('$ \Phi_x$')
    plt.ylabel('$ \Theta_y$')
    if min_plot_num_en == max_plot_num_en:
        plt.title('at %5.2feV' % min_plot_num_en)
    else:
        plt.title('%5.2feV to%5.2feV' % (min_plot_num_en, max_plot_num_en))
    return


def cdad_dc_comparison(data_right, data_left, x_cut, y_cut, min_plot_num_updc,
                       max_plot_num_updc, min_plot_num_dc, max_plot_num_dc, min_plot_num_lowdc,
                       max_plot_num_lowdc, vmax_val='none', enlim_down=0, enlim_up=0.8):

    fig10 = plt.figure(constrained_layout=True, figsize=(15, 12))
    gs0 = fig10.add_gridspec(1, 3)

    gs00 = gs0[0].subgridspec(2, 1)

    gs01 = gs0[1].subgridspec(3, 1)

    gs02 = gs0[2].subgridspec(3, 1)

    fig10.add_subplot(gs00[0, 0])
    data_right.sel(x=x_cut, method="nearest").plot()
    plt.axhline(min_plot_num_updc, color='r', linestyle='--', linewidth=2.5)
    plt.axhline(max_plot_num_updc, color='r', linestyle='--', linewidth=2.5)

    plt.axhline(min_plot_num_dc, color='magenta',
                linestyle='--', linewidth=2.5)
    plt.axhline(max_plot_num_dc, color='magenta',
                linestyle='--', linewidth=2.5)

    plt.axhline(min_plot_num_lowdc, color='orange',
                linestyle='--', linewidth=2.5)
    plt.axhline(max_plot_num_lowdc, color='orange',
                linestyle='--', linewidth=2.5)

    plt.ylim(enlim_down, enlim_up)
    plt.xlabel('$ \Theta_y$')
    plt.ylabel('$E-E_F (eV)$')

    fig10.add_subplot(gs00[1, 0])
    data_right.sel(y=y_cut, method="nearest").plot()
    plt.axhline(min_plot_num_updc, color='r', linestyle='--', linewidth=2.5)
    plt.axhline(max_plot_num_updc, color='r', linestyle='--', linewidth=2.5)

    plt.axhline(min_plot_num_dc, color='magenta',
                linestyle='--', linewidth=2.5)
    plt.axhline(max_plot_num_dc, color='magenta',
                linestyle='--', linewidth=2.5)

    plt.axhline(min_plot_num_lowdc, color='orange',
                linestyle='--', linewidth=2.5)
    plt.axhline(max_plot_num_lowdc, color='orange',
                linestyle='--', linewidth=2.5)
    plt.ylim(enlim_down, enlim_up)
    plt.xlabel('$ \Phi_x$')
    plt.ylabel('$E-E_F (eV)$')

    fig10.add_subplot(gs01[0, 0])
    cdad_left_right = cdad_shc_just_func(
        data_right, data_left, min_plot_num_updc, max_plot_num_updc)

    if vmax_val == 'none':
        cdad_left_right.plot()
    else:
        cdad_left_right.plot(vmax=vmax_val)
    plt.xlabel('$ \Phi_x$')
    plt.ylabel('$ \Theta_y$')
    if min_plot_num_updc == max_plot_num_updc:
        plt.title('at %5.2feV' % min_plot_num_updc)
    else:
        plt.title('%5.2feV to%5.2feV' % (min_plot_num_updc, max_plot_num_updc))

    fig10.add_subplot(gs01[1, 0])
    cdad_left_right = cdad_shc_just_func(
        data_right, data_left, min_plot_num_dc, max_plot_num_dc)
    if vmax_val == 'none':
        cdad_left_right.plot()
    else:
        cdad_left_right.plot(vmax=vmax_val)
    plt.xlabel('$ \Phi_x$')
    plt.ylabel('$ \Theta_y$')
    if min_plot_num_dc == max_plot_num_dc:
        plt.title('at %5.2feV' % min_plot_num_dc)
    else:
        plt.title('%5.2feV to%5.2feV' % (min_plot_num_dc, max_plot_num_dc))

    fig10.add_subplot(gs01[2, 0])
    cdad_left_right = cdad_shc_just_func(
        data_right, data_left, min_plot_num_lowdc, max_plot_num_lowdc)
    if vmax_val == 'none':
        cdad_left_right.plot()
    else:
        cdad_left_right.plot(vmax=vmax_val)
    plt.xlabel('$ \Phi_x$')
    plt.ylabel('$ \Theta_y$')
    if min_plot_num_lowdc == max_plot_num_lowdc:
        plt.title('at %5.2feV' % min_plot_num_lowdc)
    else:
        plt.title('%5.2feV to%5.2feV' %
                  (min_plot_num_lowdc, max_plot_num_lowdc))

    fig10.add_subplot(gs02[0, 0])
    data_right.sel(Energy=slice(min_plot_num_updc, max_plot_num_updc)).sum(
        dim='Energy').plot()
    plt.axhline(y_cut, color='r', linestyle='--', linewidth=2.5)
    plt.axvline(x_cut, color='orange', linestyle='--', linewidth=2.5)
    plt.xlabel('$ \Phi_x$')
    plt.ylabel('$ \Theta_y$')
    if min_plot_num_updc == max_plot_num_updc:
        plt.title('at %5.2feV' % min_plot_num_updc)
    else:
        plt.title('%5.2feV to%5.2feV' % (min_plot_num_updc, max_plot_num_updc))

    fig10.add_subplot(gs02[1, 0])
    data_right.sel(Energy=slice(min_plot_num_dc, max_plot_num_dc)).sum(
        dim='Energy').plot()
    plt.xlabel('$ \Phi_x$')
    plt.ylabel('$ \Theta_y$')
    if min_plot_num_dc == max_plot_num_dc:
        plt.title('at %5.2feV' % min_plot_num_dc)
    else:
        plt.title('%5.2feV to%5.2feV' % (min_plot_num_dc, max_plot_num_dc))

    fig10.add_subplot(gs02[2, 0])
    data_right.sel(Energy=slice(min_plot_num_lowdc, max_plot_num_lowdc)).sum(
        dim='Energy').plot()
    plt.xlabel('$ \Phi_x$')
    plt.ylabel('$ \Theta_y$')
    if min_plot_num_lowdc == max_plot_num_lowdc:
        plt.title('at %5.2feV' % min_plot_num_lowdc)
    else:
        plt.title('%5.2feV to%5.2feV' %
                  (min_plot_num_lowdc, max_plot_num_lowdc))
    return


def cdad_dc_comparison_full(data_right, data_left, min_plot_num_deg_x, max_plot_num_deg_x,
                            min_plot_num_deg_y, max_plot_num_deg_y, min_plot_num_updc,
                            max_plot_num_updc, min_plot_num_dc, max_plot_num_dc, min_plot_num_lowdc,
                            max_plot_num_lowdc, vmax_val='none', enlim_down=0, enlim_up=0.8):

    fig10 = plt.figure(constrained_layout=True, figsize=(15, 10))
    gs0 = fig10.add_gridspec(1, 4)

    gs00 = gs0[0].subgridspec(2, 1)

    gs01 = gs0[1].subgridspec(2, 1)

    gs02 = gs0[2].subgridspec(3, 1)

    gs03 = gs0[3].subgridspec(3, 1)

    fig10.add_subplot(gs00[0, 0])
    if max_plot_num_deg_x == min_plot_num_deg_x:
        data_right.sel(x=min_plot_num_deg_x, method="nearest").plot()
    else:
        data_right.sel(x=slice(min_plot_num_deg_x, max_plot_num_deg_x)).sum(
            dim='x').plot()

    plt.axhline(min_plot_num_updc, color='r', linestyle='--', linewidth=2.5)
    plt.axhline(max_plot_num_updc, color='r', linestyle='--', linewidth=2.5)

    plt.axhline(min_plot_num_dc, color='magenta',
                linestyle='--', linewidth=2.5)
    plt.axhline(max_plot_num_dc, color='magenta',
                linestyle='--', linewidth=2.5)

    plt.axhline(min_plot_num_lowdc, color='orange',
                linestyle='--', linewidth=2.5)
    plt.axhline(max_plot_num_lowdc, color='orange',
                linestyle='--', linewidth=2.5)

    plt.ylim(enlim_down, enlim_up)

    plt.xlabel('$ \Theta_y$')
    plt.ylabel('$E-E_F (eV)$')
    if max_plot_num_deg_x == min_plot_num_deg_x:
        plt.title('at %5.2f$\degree$' % min_plot_num_deg_x)
    else:
        plt.title('%5.2f$\degree$ to%5.2f$\degree$' %
                  (min_plot_num_deg_x, max_plot_num_deg_x))

    fig10.add_subplot(gs00[1, 0])
    if max_plot_num_deg_y == min_plot_num_deg_y:
        data_right.sel(y=min_plot_num_deg_y, method="nearest").plot()
    else:
        data_right.sel(y=slice(min_plot_num_deg_y, max_plot_num_deg_y)).sum(
            dim='y').plot()

    plt.axhline(min_plot_num_updc, color='r', linestyle='--', linewidth=2.5)
    plt.axhline(max_plot_num_updc, color='r', linestyle='--', linewidth=2.5)

    plt.axhline(min_plot_num_dc, color='magenta',
                linestyle='--', linewidth=2.5)
    plt.axhline(max_plot_num_dc, color='magenta',
                linestyle='--', linewidth=2.5)

    plt.axhline(min_plot_num_lowdc, color='orange',
                linestyle='--', linewidth=2.5)
    plt.axhline(max_plot_num_lowdc, color='orange',
                linestyle='--', linewidth=2.5)

    plt.ylim(enlim_down, enlim_up)

    plt.xlabel('$ \Phi_x$')
    plt.ylabel('$E-E_F (eV)$')
    if max_plot_num_deg_y == min_plot_num_deg_y:
        plt.title('at %5.2f$\degree$' % min_plot_num_deg_y)
    else:
        plt.title('%5.2f$\degree$ to%5.2f$\degree$' %
                  (min_plot_num_deg_y, max_plot_num_deg_y))

    fig10.add_subplot(gs01[0, 0])
    cdad_left_right = cdad_svc_just_func(data_right, data_left, enlim_down, enlim_up,
                                         min_plot_num_deg_x, max_plot_num_deg_x, 'y', av='on')
    if vmax_val == 'none':
        cdad_left_right.plot()
    else:
        cdad_left_right.plot(vmax=vmax_val)

    plt.ylim(enlim_down, enlim_up)
    plt.xlabel('$ \Theta_y$')
    plt.ylabel('$E-E_F (eV)$')
    if max_plot_num_deg_x == min_plot_num_deg_x:
        plt.title('at %5.2f$\degree$' % min_plot_num_deg_x)
    else:
        plt.title('%5.2f$\degree$ to%5.2f$\degree$' %
                  (min_plot_num_deg_x, max_plot_num_deg_x))

    fig10.add_subplot(gs01[1, 0])
    cdad_left_right = cdad_svc_just_func(data_right, data_left, enlim_down, enlim_up,
                                         min_plot_num_deg_x, max_plot_num_deg_x, 'x', av='on')
    if vmax_val == 'none':
        cdad_left_right.plot()
    else:
        cdad_left_right.plot(vmax=vmax_val)

    plt.ylim(enlim_down, enlim_up)
    plt.xlabel('$ \Phi_x$')
    plt.ylabel('$E-E_F (eV)$')
    if max_plot_num_deg_y == min_plot_num_deg_y:
        plt.title('at %5.2f$\degree$' % max_plot_num_deg_y)
    else:
        plt.title('%5.2f$\degree$ to%5.2f$\degree$' %
                  (min_plot_num_deg_y, max_plot_num_deg_y))

    fig10.add_subplot(gs02[0, 0])
    cdad_left_right = cdad_shc_just_func(
        data_right, data_left, min_plot_num_updc, max_plot_num_updc)

    if vmax_val == 'none':
        cdad_left_right.plot()
    else:
        cdad_left_right.plot(vmax=vmax_val)
    plt.xlabel('$ \Phi_x$')
    plt.ylabel('$ \Theta_y$')
    if min_plot_num_updc == max_plot_num_updc:
        plt.title('at %5.2feV' % min_plot_num_updc)
    else:
        plt.title('%5.2feV to%5.2feV' % (min_plot_num_updc, max_plot_num_updc))

    fig10.add_subplot(gs02[1, 0])
    cdad_left_right = cdad_shc_just_func(
        data_right, data_left, min_plot_num_dc, max_plot_num_dc)
    if vmax_val == 'none':
        cdad_left_right.plot()
    else:
        cdad_left_right.plot(vmax=vmax_val)
    plt.xlabel('$ \Phi_x$')
    plt.ylabel('$ \Theta_y$')
    if min_plot_num_dc == max_plot_num_dc:
        plt.title('at %5.2feV' % min_plot_num_dc)
    else:
        plt.title('%5.2feV to%5.2feV' % (min_plot_num_dc, max_plot_num_dc))

    fig10.add_subplot(gs02[2, 0])
    cdad_left_right = cdad_shc_just_func(
        data_right, data_left, min_plot_num_lowdc, max_plot_num_lowdc)
    if vmax_val == 'none':
        cdad_left_right.plot()
    else:
        cdad_left_right.plot(vmax=vmax_val)
    plt.xlabel('$ \Phi_x$')
    plt.ylabel('$ \Theta_y$')
    if min_plot_num_lowdc == max_plot_num_lowdc:
        plt.title('at %5.2feV' % min_plot_num_lowdc)
    else:
        plt.title('%5.2feV to%5.2feV' %
                  (min_plot_num_lowdc, max_plot_num_lowdc))

    fig10.add_subplot(gs03[0, 0])
    if min_plot_num_updc == max_plot_num_updc:
        data_right.sel(Energy=min_plot_num_updc, method="nearest").plot()
    else:
        data_right.sel(Energy=slice(min_plot_num_updc, max_plot_num_updc)).sum(
            dim='Energy').plot()

    plt.axhline(max_plot_num_deg_y, color='r', linestyle='--', linewidth=2.5)
    plt.axhline(min_plot_num_deg_y, color='r', linestyle='--', linewidth=2.5)
    plt.axvline(max_plot_num_deg_x, color='orange',
                linestyle='--', linewidth=2.5)
    plt.axvline(min_plot_num_deg_x, color='orange',
                linestyle='--', linewidth=2.5)

    plt.xlabel('$ \Phi_x$')
    plt.ylabel('$ \Theta_y$')
    if min_plot_num_updc == max_plot_num_updc:
        plt.title('at %5.2feV' % min_plot_num_updc)
    else:
        plt.title('%5.2feV to%5.2feV' % (min_plot_num_updc, max_plot_num_updc))

    fig10.add_subplot(gs03[1, 0])
    if min_plot_num_dc == max_plot_num_dc:
        data_right.sel(Energy=min_plot_num_dc, method="nearest").plot()
    else:
        data_right.sel(Energy=slice(min_plot_num_dc, max_plot_num_dc)).sum(
            dim='Energy').plot()

    plt.xlabel('$ \Phi_x$')
    plt.ylabel('$ \Theta_y$')
    if min_plot_num_dc == max_plot_num_dc:
        plt.title('at %5.2feV' % min_plot_num_dc)
    else:
        plt.title('%5.2feV to%5.2feV' % (min_plot_num_dc, max_plot_num_dc))

    fig10.add_subplot(gs03[2, 0])
    if min_plot_num_lowdc == max_plot_num_lowdc:
        data_right.sel(Energy=min_plot_num_lowdc, method="nearest").plot()
    else:
        data_right.sel(Energy=slice(min_plot_num_lowdc, max_plot_num_lowdc)).sum(
            dim='Energy').plot()

    plt.xlabel('$ \Phi_x$')
    plt.ylabel('$ \Theta_y$')
    if min_plot_num_lowdc == max_plot_num_lowdc:
        plt.title('at %5.2feV' % min_plot_num_lowdc)
    else:
        plt.title('%5.2feV to%5.2feV' %
                  (min_plot_num_lowdc, max_plot_num_lowdc))
    return


def image_plot(data_binned, length_subplot_in, min_plot_num_en, max_plot_num_en,
               min_plot_num_deg, max_plot_num_deg, vmax_val='none', sing_val='none', conv_mode='k'):
    """
    Plotting function for images in same fashion as cdad ones.
    """

    print('count rate for data is;', data_binned.sum().values)

    # plots every nthimage
    grid_4fig = gridspec.GridSpec(ncols=2, nrows=length_subplot_in)
    fig = plt.figure(figsize=(10, length_subplot_in*4))
    length_subplot = 2*length_subplot_in

    for i in range(length_subplot):
        ax = fig.add_subplot(
            grid_4fig[np.floor_divide(i, 2), np.remainder(i, 2)])

        if i % 2 == 0:
            i = i/2
            z_factor = (max_plot_num_deg-min_plot_num_deg)/(length_subplot_in)
            z = i*z_factor+min_plot_num_deg

            if sing_val != 'none':
                datasel = data_binned.sel(Energy=slice(
                    min_plot_num_en, max_plot_num_en)).sel(y=z, method="nearest")
            else:
                datasel = data_binned.sel(Energy=slice(
                    min_plot_num_en, max_plot_num_en)).sel(y=slice(z, z+z_factor)).sum(dim='y')

            if vmax_val == 'none':
                datasel.plot(vmin=0)
            else:
                datasel.plot(vmin=0, vmax=vmax_val)

            plt.tight_layout()

            if conv_mode == 'k':
                plt.xlabel('$k_{\parallel x}(\AA^{-1})$')
                if sing_val != 'none':
                    plt.title('at %5.2f $\AA^{-1}$' % z)
                else:
                    plt.title(
                        '%5.2f $\AA^{-1}$ to%5.2f $\AA^{-1}$' % (z, z+z_factor))
            else:
                plt.xlabel('$ \Phi_x$')
                if sing_val != 'none':
                    plt.title('at %5.2f$\degree$' % z)
                else:
                    plt.title('%5.2f$\degree$ to%5.2f$\degree$' %
                              (z, z+z_factor))

            plt.ylabel('$E-E_F (eV)$')

        else:
            i = (i-1)/2
            z_factor = (max_plot_num_deg-min_plot_num_deg)/(length_subplot_in)
            z = i*z_factor+min_plot_num_deg

            if sing_val != 'none':
                datasel = data_binned.sel(Energy=slice(
                    min_plot_num_en, max_plot_num_en)).sel(x=z, method="nearest")
            else:
                datasel = data_binned.sel(Energy=slice(
                    min_plot_num_en, max_plot_num_en)).sel(x=slice(z, z+z_factor)).sum(dim='x')

            if vmax_val == 'none':
                datasel.plot(vmin=0)
            else:
                datasel.plot(vmin=0, vmax=vmax_val)

            plt.tight_layout()

            if conv_mode == 'k':
                plt.xlabel('$k_{\parallel y}(\AA^{-1})$')
                if sing_val != 'none':
                    plt.title('at %5.2f $\AA^{-1}$' % z)
                else:
                    plt.title(
                        '%5.2f $\AA^{-1}$ to%5.2f $\AA^{-1}$' % (z, z+z_factor))
            else:
                plt.xlabel('$ \Theta_y$')
                if sing_val != 'none':
                    plt.title('at %5.2f$\degree$' % z)
                else:
                    plt.title('%5.2f$\degree$ to%5.2f$\degree$' %
                              (z, z+z_factor))

            plt.ylabel('$E-E_F (eV)$')

    # plots every nthimage
    grid_4fig = gridspec.GridSpec(ncols=2, nrows=length_subplot_in)
    fig = plt.figure(figsize=(10, length_subplot_in*4))

    for i in range(length_subplot_in):
        ax = fig.add_subplot(
            grid_4fig[np.floor_divide(i, 2), np.remainder(i, 2)])

        z_factor = (max_plot_num_en-min_plot_num_en)/length_subplot_in
        z = i*z_factor+min_plot_num_en

        if sing_val != 'none':
            datasel = data_binned.sel(Energy=z, method="nearest")
        else:
            datasel = data_binned.sel(Energy=slice(
                z, z+z_factor)).sum(dim='Energy')

        if vmax_val == 'none':
            datasel.plot(vmin=0)
        else:
            datasel.plot(vmin=0, vmax=vmax_val)

        plt.tight_layout()

        if conv_mode == 'k':
            if data_binned.dims[2] == 'y':
                plt.ylabel('$k_{\parallel y}(\AA^{-1})$')
                plt.ylabel('$k_{\parallel x}(\AA^{-1})$')
                #plt.ylabel('$k_{\parallel %s}(\AA^{-1})$' % data_binned.dims[3])
                #plt.xlabel('$k_{\parallel %s}(\AA^{-1})$' % data_binned.dims[2])
            else:
                plt.ylabel('$k_{\parallel x}(\AA^{-1})$')
                plt.ylabel('$k_{\parallel y}(\AA^{-1})$')
                #plt.ylabel('$k_{\parallel %s}(\AA^{-1})$' % data_binned.dims[3])
                #plt.xlabel('$k_{\parallel %s}(\AA^{-1})$' % data_binned.dims[2])
        else:
            if data_binned.dims[1] == 'y':
                plt.ylabel('$ \Theta_%s$' % data_binned.dims[1])
                plt.xlabel('$ \Phi_%s$' % data_binned.dims[2])
            else:
                plt.ylabel('$ \Phi_%s$' % data_binned.dims[1])
                plt.xlabel('$ \Theta_%s$' % data_binned.dims[2])

        if sing_val != 'none':
            plt.title('at %5.2feV' % z)
        else:
            plt.title('%5.2feV to%5.2feV' % (z, z+z_factor))
    return


def image_svc(data_binned, min_plot_num_en, max_plot_num_en,
              min_plot_num_deg_y, max_plot_num_deg_y, min_plot_num_deg_x, max_plot_num_deg_x, vmax_val='none'):
    """
    Selected Vertical Cut Plotting function for images in same fashion as cdad ones.
    Returns an Energy-Angle cut for both direction for selected other direction.
    """
    length_subplot_in = 1

    # plots every nthimage
    grid_4fig = gridspec.GridSpec(ncols=2, nrows=length_subplot_in)
    fig = plt.figure(figsize=(10, length_subplot_in*4))
    length_subplot = 2*length_subplot_in

    for i in range(length_subplot):
        ax = fig.add_subplot(
            grid_4fig[np.floor_divide(i, 2), np.remainder(i, 2)])

        if i % 2 == 0:
            i = i/2
            z_factor = (max_plot_num_deg_y-min_plot_num_deg_y) / \
                (length_subplot_in)
            z = i*z_factor+min_plot_num_deg_y

            if max_plot_num_deg_y == min_plot_num_deg_y:
                datasel = data_binned.sel(Energy=slice(
                    min_plot_num_en, max_plot_num_en)).sel(y=z, method="nearest")
            else:
                datasel = data_binned.sel(Energy=slice(
                    min_plot_num_en, max_plot_num_en)).sel(y=slice(z, z+z_factor)).sum(dim='y')

            if vmax_val == 'none':
                datasel.plot(vmin=0)
            else:
                datasel.plot(vmin=0, vmax=vmax_val)

            plt.tight_layout()
            plt.xlabel('$ \Phi_x$')
            plt.ylabel('$E-E_F (eV)$')
            if max_plot_num_deg_y == min_plot_num_deg_y:
                plt.title('at %5.2f$\degree$' % z)
            else:
                plt.title('%5.2f$\degree$ to%5.2f$\degree$' % (z, z+z_factor))

        else:
            i = (i-1)/2
            z_factor = (max_plot_num_deg_x-min_plot_num_deg_x) / \
                (length_subplot_in)
            z = i*z_factor+min_plot_num_deg_x

            if max_plot_num_deg_x == min_plot_num_deg_x:
                datasel = data_binned.sel(Energy=slice(
                    min_plot_num_en, max_plot_num_en)).sel(x=z, method="nearest")
            else:
                datasel = data_binned.sel(Energy=slice(
                    min_plot_num_en, max_plot_num_en)).sel(x=slice(z, z+z_factor)).sum(dim='x')

            if vmax_val == 'none':
                datasel.plot(vmin=0)
            else:
                datasel.plot(vmin=0, vmax=vmax_val)

            plt.tight_layout()
            plt.xlabel('$ \Theta_y$')
            plt.ylabel('$E-E_F (eV)$')
            if max_plot_num_deg_x == min_plot_num_deg_x:
                plt.title('at %5.2f$\degree$' % z)
            else:
                plt.title('%5.2f$\degree$ to%5.2f$\degree$' % (z, z+z_factor))
    return


def image_shc(data_binned, min_plot_num_en, max_plot_num_en, vmax_val='none'):
    """
    Selected Horizontal Cut Plotting function for images in same fashion as cdad ones.
    Returns an energy cut for both direction for selected other direction.
    """

    if min_plot_num_en == max_plot_num_en:
        datasel = data_binned.sel(
            Energy=min_plot_num_en, method="nearest")
    else:
        datasel = data_binned.sel(Energy=slice(
            min_plot_num_en, max_plot_num_en)).sum(dim='Energy')

    if vmax_val == 'none':
        datasel.plot(aspect=1.25, size=5, vmin=0)
    else:
        datasel.plot(aspect=1.25, size=5, vmin=0, vmax=vmax_val)
    plt.tight_layout()
    plt.xlabel('$ \Phi_x$')
    plt.ylabel('$ \Theta_y$')
    if min_plot_num_en == max_plot_num_en:
        plt.title('at %5.2feV' % min_plot_num_en)
    else:
        plt.title('%5.2feV to%5.2feV' % (min_plot_num_en, max_plot_num_en))
    return


def onelinecut(data, energy_slice_down, energy_slice_up, line_cut_x, line_cut_y, central_angle_x=0, central_angle_y=0):

    if energy_slice_down == energy_slice_up:
        selecteddata_x = data.sel(Energy=energy_slice_down, method="nearest").sel(
            y=line_cut_x, method="nearest").values
        selecteddata_y = data.sel(Energy=energy_slice_down, method="nearest").sel(
            x=line_cut_y, method="nearest").values
    else:
        selecteddata_x = data.sel(Energy=slice(energy_slice_down, energy_slice_up)).sum(
            dim='Energy').sel(y=line_cut_x, method="nearest").values
        selecteddata_y = data.sel(Energy=slice(energy_slice_down, energy_slice_up)).sum(
            dim='Energy').sel(x=line_cut_y, method="nearest").values
    x_data = data.coords['x'].values-central_angle_x
    y_data = data.coords['y'].values-central_angle_y
    en_data = data.coords['Energy'].values
    energy_val = (energy_slice_up+energy_slice_down)/2
    k_x = np.sqrt(2*m_e*energy_val*e)*(np.sin(x_data*np.pi/180)/hbar)
    k_y = np.sqrt(2*m_e*energy_val*e)*(np.sin(y_data*np.pi/180)/hbar)

    fig10 = plt.figure(constrained_layout=True, figsize=(10, 12))
    gs0 = fig10.add_gridspec(1, 2)

    gs00 = gs0[0].subgridspec(4, 1)

    gs01 = gs0[1].subgridspec(3, 1)

    fig10.add_subplot(gs00[0, 0])
    plt.plot(k_y/1e10, selecteddata_y, 'orange')
    plt.xlabel('$k_{\parallel y}(\AA^{-1})$')
    plt.ylabel('counts')

    fig10.add_subplot(gs00[1, 0])
    plt.plot(k_x/1e10, selecteddata_x, 'r')
    plt.xlabel('$k_{\parallel x}(\AA^{-1})$')
    plt.ylabel('counts')

    fig10.add_subplot(gs00[2, 0])
    plt.plot(y_data, selecteddata_y, 'g')
    plt.vlines(central_angle_y, 0, np.amax(selecteddata_y),
               colors='magenta', linestyles='dashed')
    plt.xlabel('$ \Theta_y$')
    plt.ylabel('counts')

    fig10.add_subplot(gs00[3, 0])
    plt.plot(x_data, selecteddata_x, 'g')
    plt.vlines(central_angle_x, 0, np.amax(selecteddata_x),
               colors='magenta', linestyles='dashed')
    plt.xlabel('$ \Phi_x$')
    plt.ylabel('counts')

    fig10.add_subplot(gs01[0, 0])

    if energy_slice_down == energy_slice_up:
        data.sel(Energy=energy_slice_down, method="nearest").plot()
    else:
        data.sel(Energy=slice(energy_slice_down, energy_slice_up)).sum(
            dim='Energy').plot()
    plt.title('Energy Cut')

    plt.axhline(line_cut_x, color='r', linestyle='--')
    plt.axvline(line_cut_y, color='orange', linestyle='--')
    plt.xlabel('$ \Phi_x$')
    plt.ylabel('$ \Theta_y$')

    fig10.add_subplot(gs01[1, 0])
    data.sel(y=line_cut_x, method="nearest").plot()
    plt.axhline(y=energy_slice_down, color='r', linestyle='--')
    plt.axhline(y=energy_slice_up, color='r', linestyle='--')
    plt.ylim(0, 0.8)
    plt.xlabel('$ \Phi_x$')
    plt.ylabel('$E-E_F (eV)$')

    fig10.add_subplot(gs01[2, 0])
    data.sel(x=line_cut_y, method="nearest").plot()
    plt.axhline(y=energy_slice_down, color='r', linestyle='--')
    plt.axhline(y=energy_slice_up, color='r', linestyle='--')
    plt.ylim(0, 0.8)
    plt.xlabel('$ \Theta_y$')
    plt.ylabel('$E-E_F (eV)$')
    return


def comparison_plot(data_right,data_left,min_plot_num_en,max_plot_num_en, x = 0, y = 0, dirac_energy = 0.15, single_plot = 'on'):
    """
    This for checking the newly converted data's correction values and orientation.
    """
    fig, axs = plt.subplots(2,2, figsize=(12,10))

    if single_plot != 'none':
        data_right.sum(dim=('x','y')).plot(ax=axs[0,0],label='$ 0  \degree \circlearrowright$')
        data_left.sum(dim= ('x','y')).plot(ax=axs[0,0],label='$ 90 \degree \circlearrowleft$')
        axs[0,0].set_xlabel('$E-E_F (eV)$')
        axs[0,0].set_ylabel('counts (a.u.)')
        axs[0,0].legend()
        axs[0,0].set_title(label = '')
    else:
        data_right.sum(dim=('x','y')).plot(ax=axs[0,0],label='data')
        axs[0,0].set_xlabel('$E-E_F (eV)$')
        axs[0,0].set_ylabel('counts (a.u.)')
        axs[0,0].legend()
        axs[0,0].set_title(label = '')

    data_right.sel(Energy=slice(
        min_plot_num_en, max_plot_num_en)).sum(dim='Energy').plot(ax=axs[1,0])
    axs[1,0].axvline(x = 0 ,color = 'r', linestyle = '--')
    axs[1,0].axhline(y = 0 ,color = 'r', linestyle = '--')
    axs[1,0].set_title(label = '')
    axs[1,0].set_xlabel('$k_{ x}(\AA^{-1})$')
    axs[1,0].set_ylabel('$k_{ y}(\AA^{-1})$')

    data_right.sel(Energy=slice(0.08, 0.5)).sel(y=y, method="nearest").plot(ax=axs[0,1])
    axs[0,1].axvline(x = 0 ,color = 'r', linestyle = '--')
    axs[0,1].axhline(y = dirac_energy ,color = 'r', linestyle = '--')
    axs[0,1].axhline(y = min_plot_num_en ,color = 'g', linestyle = '--')
    axs[0,1].axhline(y = max_plot_num_en ,color = 'g', linestyle = '--')
    axs[0,1].set_title(label = '')
    axs[0,1].set_ylabel('$E-E_F (eV)$')
    axs[0,1].set_xlabel('$k_{ y}(\AA^{-1})$')

    data_right.sel(Energy=slice(0.08, 0.5)).sel(x=x, method="nearest").plot(ax=axs[1,1])
    axs[1,1].axvline(x = 0 ,color = 'r', linestyle = '--')
    axs[1,1].axhline(y = dirac_energy ,color = 'r', linestyle = '--')
    axs[1,1].set_title(label = '')
    axs[1,1].set_ylabel('$E-E_F (eV)$')
    axs[1,1].set_xlabel('$k_{ x}(\AA^{-1})$')


    plt.tight_layout()

    return
