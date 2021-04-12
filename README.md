# Angle-resolved Time-of-Flight Data Import & Analysis Module

<p align="center">
<img src="https://github.com/haltugyildirim/ARTOF-Data-Analysis/blob/main/images/drawing_plain_blue.svg" width="500" class="center" alt="logo"/>
    <br/>
</p>

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Last Commit](https://img.shields.io/github/last-commit/haltugyildirim/ARTOF-Data-Analysis)

This project is focused on reading, plotting and analyzing data from Angle-Resolved Time of Flight(ARTOF) experiments. ARTOF is an offshoot of ARPES systems where instead of using hemispherical analyzers, a time-of-flight tube coupled with a delay line detector is used. This gives an advantage to map band structure on both dimensions without any need to change the angle of the sample.

Data is imported as a hdf5 file, common in ARPES experiments. The character of data is mostly 4 dimensional; delaypoints(time), energy(time-of-flight), y(angle perpendicular to plane of incidence), x(angle parallel to plane of incidence). An incredible package; xarray is used for labeling this 4 dimensional matrix. Selection, addition and algebraic manipulation of data is done using the functions from xarray package.

Cross-correlation fitting; a method to find the length of ultrashort laser-pulses from delaypoints, is done with curve_fit from scipy using a convoluted version of an exponential decay and a gaussian. As far as the cross correlation is concerned, I am satisfied with curve_fit but for future in-band fittings, I would like to switch lmfit package.

All the plotting is either done with build-in function from xarray or matplotlib. An experimental implementation of the MCLAHE(https://github.com/mpes-kit/mclahe) package is used for enhancing the image data, although as the binning is generally low is not so useful until now. I would like to implement more of the packages from MPES-kit.

Future directions;

    angle to momentum conversion
    3D plotting
    Momentum microscopy implementation

As the data acquisition software is very problematic and old, many error comes during the experiment; crashing of the software, data omitting during saving etc. So this code in here also deals with this system specific problems. So if you want to use this piece of code in here, please consider asking me beforehand.
