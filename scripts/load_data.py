#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 16:18:21 2022

@author: bfildier
"""

import scipy.io
import sys, os, glob
import numpy as np
import xarray as xr
from datetime import datetime as dt
from datetime import timedelta, timezone
import pytz
import pickle
import matplotlib.image as mpimg


from radiativefeatures import *
from radiativescaling import *
# from thermodynamics import *
from conditionalstats import *
from matrixoperators import *
from thermoConstants import *
from thermoFunctions import *



#-- load EUREC4A metadata

exec(open(os.path.join(workdir,"load_EUREC4A_metadata.py")).read())


#%%    ###--- Load data ---###

# Profiles
radprf = xr.open_dataset(os.path.join(inputdir,'rad_profiles_CF.nc'))
# choose profiles for that day that start at bottom
data_all = radprf.where(radprf.z_min<=50,drop=True)

z = data_all.alt.values/1e3 # km
pres = np.nanmean(data_all.pressure.data,axis=dim_t)/100 # hPa

rad_features_all = {}
rad_scaling_all = {}
ref_dist_all = {}
cond_dist_all = {}

# initalize
for cond_varid in cond_varids:
     cond_dist_all[cond_varid] = {}

for day in days:

    #-- Radiative features
    features_filename = 'rad_features.pickle'
    print('loading %s'%features_filename)
    # load
    features_path = os.path.join(resultdir,day,features_filename)
    f = pickle.load(open(features_path,'rb'))
    # store
    rad_features_all[day] = f
    
    #-- Radiative scaling
    rad_scaling_filename = 'rad_scaling.pickle'
    print('loading %s'%rad_scaling_filename)
    rad_scaling_path = os.path.join(resultdir,day,rad_scaling_filename)
    rs = pickle.load(open(rad_scaling_path,'rb'))
    # store
    rad_scaling_all[day] = rs
    
    #-- Reference PW distribution
    ref_filename = 'dist_%s.pickle'%(ref_varid)
    print('load reference %s distribution'%ref_varid)
    ref_dist_path = os.path.join(resultdir,day,ref_filename)
    ref_dist = pickle.load(open(ref_dist_path,'rb'))
    # save in current environment
    ref_dist_all[day] = ref_dist
    
    #-- Conditional distributions
    for cond_varid in cond_varids:
        
        # load
        cond_filename = 'cdist_%s_on_%s.pickle'%(cond_varid,ref_varid)
        print('loading %s'%cond_filename)
        cond_dist_path = os.path.join(resultdir,day,cond_filename)
        cond_dist = pickle.load(open(cond_dist_path,'rb'))
        # save in current environment
        cond_dist_all[cond_varid][day] = cond_dist
    
# Bounds
ref_var_min = ref_dist_all['20200122'].bins[0]
ref_var_max = ref_dist_all['20200122'].bins[-1]


# load GOES images
date = dt(2020,1,26)
indir_goes_images = '/Users/bfildier/Data/satellite/GOES/images/%s'%date.strftime('%Y_%m_%d')
image_vis_files = glob.glob(os.path.join(indir_goes_images,'*C02*00.jpg'))
image_vis_files.sort()
# in the visible channel
images_vis = []
for i in range(len(image_vis_files)):
    images_vis.append(mpimg.imread(image_vis_files[i]))


# load Caroline's data for rad circulation
c_inputdir = os.path.join(repodir,'input','MullerBony2015')
c_muller = scipy.io.loadmat(os.path.join(c_inputdir,'Qrad_pwbinnedvariables_ir90_t40_a50_nbins64.mat'))

# load moist intrusion data
# with piecewise linear fit and removed intrusions
rad_file_MI_20200213 = os.path.join(radinputdir,'rad_profiles_moist_intrusions_20200213.nc')
radprf_MI_20200213 = xr.open_dataset(rad_file_MI_20200213)

# with piecewise linear fit and removed intrusions
rad_file_MI_20200213lower = os.path.join(radinputdir,'rad_profiles_moist_intrusions_20200213lower.nc')
radprf_MI_20200213lower = xr.open_dataset(rad_file_MI_20200213lower)

# # with rectangular intrusions
# rad_file_RMI_20200213 = os.path.join(radinputdir,'rad_profiles_rectangular_moist_intrusions.nc')
# radprf_RMI_20200213 = xr.open_dataset(rad_file_RMI_20200213)

# All automated intrusions fixing kappa to its lower tropospheric value
rad_file_MI_20200213lower_fix_k = os.path.join(radinputdir,'rad_profiles_moist_intrusions_20200213lower_fix_k.nc')
radprf_MI_20200213lower_fix_k = xr.open_dataset(rad_file_MI_20200213lower_fix_k)

# warming
rad_file_warming = os.path.join(radinputdir,'rad_profiles_stepRH20200126_idealized_warming.nc')
radprf_warming = xr.open_dataset(rad_file_warming)

# load info on all moist intrusions
mi_file = os.path.join(repodir,'results','idealized_calculations','observed_moist_intrusions','moist_intrusions.pickle')
moist_intrusions = pickle.load(open(mi_file,'rb'))

# kappa(nu) data and fit
kappa_file = os.path.join(repodir,'results','idealized_calculations','kappa_fit.pickle')
kappa_data= pickle.load(open(kappa_file,'rb'))

# Iorg data from Hauke
iorg_file = os.path.join(repodir,'input','iorg','GOES16_IR_nc_Iorg_EUREC4A_10-20_-58--48.nc')
iorg_data = xr.open_dataset(iorg_file)

