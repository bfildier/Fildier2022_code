#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute spectral fit kappa(nu) in the rotational and vibration-rotation band of water vapor.

Created on Fri Jul 29 13:53:49 2022

@author: bfildier
"""

import xarray as xr
import numpy as np
import pandas as pd
import datetime
import pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import netCDF4
from scipy.interpolate import interp1d
from scipy.stats import linregress
import pickle

projectname = 'Fildier2022_analysis'
# workdir = os.path.dirname(os.path.realpath(__file__))
workdir = '/Users/bfildier/Code/analyses/EUREC4A/Fildier2022_analysis/scripts'
rootdir = os.path.dirname(workdir)
while os.path.basename(rootdir) != projectname:
    rootdir = os.path.dirname(rootdir)
repodir = rootdir
figdir = os.path.join(repodir,'figures','analytical_models')
resultdir = os.path.join(repodir,'results','idealized_calculations')
scriptsubdir = 'gas_optics'

os.makedirs(os.path.join(figdir,scriptsubdir),exist_ok=True)


#-- load CKDMIP data
dataDir = pathlib.Path("/Users/bfildier/Data/radiation/gas_optics_ckdmip/")
h2o_base   = 'ckdmip_idealized_lw_spectra_h2o_constant'
h2o_nocont = h2o_base.replace("h2o_constant", "h2o-no-continuum_constant")

# load
wv_key = 'k'# 10^-7 kg/kg; log 10 values are np.linspace(-7, -1.5, num=12)
hdf5file_nocont = os.path.join(dataDir,'{}-{}.h5'.format(h2o_nocont,   wv_key))
h2o_data_nocont = xr.open_dataset(hdf5file_nocont, engine='h5netcdf')
data = h2o_data_nocont

# Take the whole matrix and get the whole matrix for kappa
def convert2kappa(data,s_temp,s_pres,s_pres_m1):
    """Convert optical depth to extinction"""
    
    M_d = 0.02897 # kg/mol
    M_w = 0.01802 # kg/mol
    g = 9.81
    
    tau = data.optical_depth[s_temp,s_pres]
    N_nu = tau.shape[-1]
    mole_fraction = np.repeat(np.expand_dims(data.mole_fraction_fl[s_temp,s_pres],axis=2),N_nu,axis=2)
    pres_plus = np.repeat(np.expand_dims(data.pressure_hl[s_temp,s_pres],axis=2),N_nu,axis=2)
    pres_minus = np.repeat(np.expand_dims(data.pressure_hl[s_temp,s_pres_m1],axis=2),N_nu,axis=2)
    
    kappa = g*M_d*tau / (M_w*mole_fraction*(pres_plus-pres_minus))
    
    return kappa

def interpolateKappa(temp_ref,pres_ref):

    pres_small_1d = pres_small[0]

    # first interpolate kappa on pressure, for each nu and temp
    f_interp_kappa_on_pres = interp1d(pres_small_1d,kappa,axis=1)
    kappa_interp_on_pres = f_interp_kappa_on_pres(pres_ref)
    # print(kappa_interp_on_pres.shape)

    # do the same for temperature
    f_interp_temp_on_pres = interp1d(pres_small_1d,temp_small,axis=1)
    temp_small_interp_on_pres = f_interp_temp_on_pres(pres_ref)
    # print(temp_small_interp_on_pres.shape)

    # use the interpolated kappa and interpolated temp, to interpolate one onto the other and get the final kappa array
    f_interp_kappa_on_temp = interp1d(temp_small_interp_on_pres,kappa_interp_on_pres,axis=0)
    kappa_interp_on_pres_temp = f_interp_kappa_on_temp(temp_ref)
    # print(kappa_interp_on_pres_temp.shape)

    return kappa_interp_on_pres_temp


#-- interpolate extinction coefficients on reference temperature and pressure 

# reference values
pres_ref = 80000 # Pa
temp_ref = 290 # K

# select wavenumber range
wn_min = 200 # cm-1
wn_max = 1450 # cm-1

i_min = np.where(data.wavenumber >= wn_min)[0][0]
i_max = np.where(data.wavenumber >= wn_max)[0][0]

slice_wn = slice(i_min,i_max)

wn_range = data.wavenumber.sel(wavenumber=slice(wn_min,wn_max))

# we see that pressure is linearly decreasing for all profiles:
# plt.imshow(data.pressure_hl)
# but not temperature:
# plt.imshow(data.temperature_hl)
#... so we first have to interpolate in pressure and then in temperature

# look for where they are, approximately
i_pres = np.where(data.pressure_fl.data[0] <= pres_ref)[0][-1]
i_temp = np.where(data.temperature_fl.data[:,i_pres] <= temp_ref)[0][-1]

pres_ref_approx = data.pressure_fl.data[0][i_pres]
temp_ref_approx = data.temperature_fl.data[i_temp][i_pres]

print('size array',data.pressure_fl.shape)
print('approximate reference pressure:',i_pres,pres_ref_approx)
print('approximate reference temperature:',i_temp,temp_ref_approx)

# extract values +-2 data points of index selected
slice_temp = slice(i_temp-2,i_temp+3)
slice_pres = slice(i_pres-2,i_pres+3)
slice_pres_m1 = slice(i_pres-3,i_pres+2)

pres_small = data.pressure_fl.data[slice_temp,slice_pres]
temp_small = data.temperature_fl.data[slice_temp,slice_pres]

# convert optical depths to kappa
kappa = convert2kappa(data,slice_temp,slice_pres,slice_pres_m1)

# interpolate kappa onto reference temperature and pressure
kappa_interp_290_800 = interpolateKappa(290,80000)

# select wavenumber range
kappa_interp_290_800_range = kappa_interp_290_800[slice_wn]



#-- linear interpolation of the rotational and vibration-rotation branch

i_max_rot = np.where(wn_range >= 1000)[0][0]
slice_fit_rot = slice(0,i_max_rot)
slice_fit_vr = slice(i_max_rot,None)

nu_rot = wn_min
nu_vr = wn_max

def fit(kappa,slice_fit,nu_0):

    slope, intercept, r, p, se = linregress(wn_range[slice_fit]-nu_0,np.log(kappa[slice_fit]))
    l_0 = -1/slope
    kappa_0 = np.exp(intercept)
    kappa_fit = kappa_0 * np.exp(-(wn_range[slice_fit]-nu_0)/l_0)
    
    return kappa_0,l_0,kappa_fit


# rotational band
print('rotational band')
kappa_rot_290_800,l_rot_290_800,kappa_rot_fit_290_800 = fit(kappa_interp_290_800_range,slice_fit_rot,nu_rot)

# v-r band
print()
print('v-r band')
print(kappa_interp_290_800_range)
print(slice_fit_vr)
print(wn_range[slice_fit_vr])
kappa_vr_290_800,l_vr_290_800,kappa_vr_fit_290_800 = fit(kappa_interp_290_800_range,slice_fit_vr,nu_vr)

##-- Save results
output_dict = {'kappa':kappa_interp_290_800_range,
               'nu':wn_range,
               'kappa_fit_rot':kappa_rot_fit_290_800,
               's_fit_rot':slice_fit_rot,
               'kappa_rot':kappa_rot_290_800,
               'l_rot':l_rot_290_800,
               'nu_rot':nu_rot,
               'kappa_fit_vr':kappa_vr_fit_290_800,
               's_fit_vr':slice_fit_vr,
               'kappa_vr':kappa_vr_290_800,
               'l_vr':l_vr_290_800,
               'nu_vr':nu_vr}

out_file = os.path.join(resultdir,'kappa_fit.pickle')
pickle.dump(output_dict,open(out_file,'wb'))

