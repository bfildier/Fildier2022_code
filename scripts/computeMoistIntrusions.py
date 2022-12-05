#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute moist intrusions, all intrusion days

Created on Tue Aug  9 15:18:17 2022

@author: bfildier
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
import os,sys,glob
from math import *
import xarray as xr
import pickle
import pytz
from datetime import datetime as dt
from datetime import timedelta, timezone
from scipy import optimize
from scipy.io import loadmat
from scipy.stats import linregress



# Load own modules
projectname = 'Fildier2022_analysis'
workdir = '/Users/bfildier/Code/analyses/EUREC4A/Fildier2022_analysis/scripts'
rootdir = os.path.dirname(workdir)
while os.path.basename(rootdir) != projectname:
    rootdir = os.path.dirname(rootdir)
repodir = rootdir
moduledir = os.path.join(repodir,'functions')
resultdir = os.path.join(repodir,'results','idealized_calculations')
figdir = os.path.join(repodir,'figures','idealized_calculations')
#inputdir = '/Users/bfildier/Dropbox/Data/EUREC4A/sondes_radiative_profiles/'
inputdir = os.path.join(repodir,'input')
resultinputdir = os.path.join(repodir,'results','radiative_features')
radinputdir = os.path.join(repodir,'input')
scriptsubdir = 'observed_moist_intrusions'

os.makedirs(os.path.join(figdir,scriptsubdir),exist_ok=True)
os.makedirs(os.path.join(resultdir,scriptsubdir),exist_ok=True)


# current environment
thismodule = sys.modules[__name__]

##-- Own modules
sys.path.insert(0,moduledir)
print("Own modules available:", [os.path.splitext(os.path.basename(x))[0]
                                 for x in glob.glob(os.path.join(moduledir,'*.py'))])

#- Parameters & constants
from matrixoperators import *

mo = MatrixOperators()


#-- load EUREC4A metadata

exec(open(os.path.join(workdir,"load_EUREC4A_metadata.py")).read())




#--- load data

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
    features_path = os.path.join(resultinputdir,day,features_filename)
    f = pickle.load(open(features_path,'rb'))
    # store
    rad_features_all[day] = f

    #-- Radiative scaling
    rad_scaling_filename = 'rad_scaling.pickle'
    print('loading %s'%rad_scaling_filename)
    rad_scaling_path = os.path.join(resultinputdir,day,rad_scaling_filename)
    rs = pickle.load(open(rad_scaling_path,'rb'))
    # store
    rad_scaling_all[day] = rs

    #-- Reference PW distribution
    ref_filename = 'dist_%s.pickle'%(ref_varid)
    print('load reference %s distribution'%ref_varid)
    ref_dist_path = os.path.join(resultinputdir,day,ref_filename)
    ref_dist = pickle.load(open(ref_dist_path,'rb'))
    # save in current environment
    ref_dist_all[day] = ref_dist

    #-- Conditional distributions
    for cond_varid in cond_varids:

        # load
        cond_filename = 'cdist_%s_on_%s.pickle'%(cond_varid,ref_varid)
        print('loading %s'%cond_filename)
        cond_dist_path = os.path.join(resultinputdir,day,cond_filename)
        cond_dist = pickle.load(open(cond_dist_path,'rb'))
        # save in current environment
        cond_dist_all[cond_varid][day] = cond_dist
        
        
#%% Functions to get profiles and compute piecewise-linear fits

def getProfiles(rad_features, data_day, z_min, z_max):
    
    #- Mask
    # |qrad| > 5 K/day
    qrad_peak = np.absolute(rad_features.qrad_lw_peak)
    keep_large = qrad_peak > 5 # K/day
    # in box
    lon_day = data_day.longitude[:,50]
    lat_day = data_day.latitude[:,50]
    keep_box = np.logical_and(lon_day < lon_box[1], lat_day >= lat_box[0])
    # high-level peak
    keep_high =  np.logical_and(rad_features.z_net_peak < z_max, # m
                                rad_features.z_net_peak > z_min)
    # combined
    k = np.logical_and(np.logical_and(keep_large,keep_box),keep_high)
    
    # temperature
    temp = data_day.temperature.values[k,:]
    # relative humidity
    rh = data_day.relative_humidity.values[k,:]
    # specific humidity 
    qv = data_day.specific_humidity.values[k,:]
    # lw cooling
    qradlw = rad_features.qrad_lw_smooth[k,:]
    
    return temp, qv, rh, qradlw


def piecewise_linear(z:np.array,z_breaks:list,rh_breaks:list):
    """
    Define piecewise linear RH shape with constant value at top and bottom.

    Args:
        z (np.array): z coordinate
        z_breaks (list): z values of break points
        rh_breaks (list): rh values of break points

    Returns:
        np.array: piecewize rh
        
    """
    
    N_breaks = len(z_breaks)
    
    cond_list = [z <= z_breaks[0]]+\
                [np.logical_and(z > z_breaks[i-1],z <= z_breaks[i]) for i in range(1,N_breaks)]+\
                [z > z_breaks[N_breaks-1]]
    def make_piece(k):
        def f(z):
            return rh_breaks[k-1]+(rh_breaks[k]-rh_breaks[k-1])/(z_breaks[k]-z_breaks[k-1])*(z-z_breaks[k-1])
        return f 
    func_list = [lambda z: rh_breaks[0]]+\
                [make_piece(k) for k in range(1,N_breaks)]+\
                [lambda z: rh_breaks[N_breaks-1]]
                
    return np.piecewise(z,cond_list,func_list)

def piecewise_fit(z:np.array,rh:np.array,z_breaks_0:list,rh_breaks_0:list):    
    """
    Compute piecewise-linear fit of RH(z).

    Args:
        z (np.array): z coordinate
        rh (np.array): rh profile
        z_breaks_0 (list): initial z values of break points
        rh_breaks_0 (list): initial rh values of break points

    Returns:
        z_breaks (list): fitted z values of break points
        rh_breaks (list): fitted rh values of break points
        rh_id (np.array): piecewize rh fit

    """

    N_breaks = len(z_breaks_0)

    def piecewise_fun(z,*p):
        return piecewise_linear(z,p[0:N_breaks],p[N_breaks:2*N_breaks])

    mask = ~np.isnan(z) & ~np.isnan(rh)

    p , e = optimize.curve_fit(piecewise_fun, z[mask], rh[mask],p0=z_breaks_0+rh_breaks_0)

    rh_id = piecewise_linear(z,p[0:N_breaks],p[N_breaks:2*N_breaks])
    rh_breaks = list(p[N_breaks:2*N_breaks])
    z_breaks = list(p[0:N_breaks])
    
    return z_breaks,rh_breaks,rh_id


def computeWPaboveZ(qv,pres,p_top):
    """Calculates the integrated water path above each level.

    Arguments:
        - qv: specific humidity in kg/kg, Nz-vector
        - pres: pressure coordinate in hPa, Nz vector
        - p_top: pressure of upper integration level

    returns:
        - wp_z: water path above each level, Nz-vector"""

    Np = qv.shape[0]
    wp_z = np.full(Np,np.nan)

    p_increasing = np.diff(pres)[0] > 0
    
    if p_increasing:
        
        i_p_top = np.where(pres >= p_top)[0][0]
        
        for i_p in range(i_p_top,Np):
        # self.wp_z[:,i_z] = self.mo.pressureIntegral(arr=data.specific_humidity[:,i_z:],pres=pres[i_z:],p_levmin=pres[i_z],p_levmax=pres[-1],z_axis=z_axis)

            arr = qv
            p = pres
            p0 = p_top
            p1 = p[i_p]
            i_w = i_p
            
            wp_z[i_w] = mo.pressureIntegral(arr=arr,pres=p,p0=p0,p1=p1)

    else:
        
        i_p_top = np.where(pres >= p_top)[0][-1]

        for i_p in range(i_p_top):
            
            arr = np.flip(qv)
            p = np.flip(pres)
            p0 = p_top
            p1 = pres[i_p]
            i_w = i_p

            wp_z[i_w] = mo.pressureIntegral(arr=arr,pres=p,p0=p0,p1=p1)

    return wp_z



#%% Calculation

days_intrusions = '20200213', '20200213', '20200211', '20200209', '20200209', '20200128'
z_min_intrusions = 5000, 4000, 4000, 3500, 5500, 4000
z_max_intrusions = 9000, 5000, 6000, 5500, 8500, 6000
z_breaks_0_all = [1.8,2,4,5,6.5,7], [1.8,2,4,5], [2,3.5,4.5,5], [2,2.5,4.5,5], [2,2.5,5,7], [2,2.5,4.5,5] 
rh_breaks_0_all = [0.8,0.1,0.7,0.1,0.7,0.1], [0.8,0.1,0.7,0.1], [0.8,0.3,0.7,0.05], [0.75,0.1,0.25,0.05], [0.75,0.1,0.25,0.05], [0.75,0.1,0.25,0.05] 

moist_intrusions = {}

for day, z_min, z_max, z_breaks_0, rh_breaks_0 \
in zip(days_intrusions,z_min_intrusions,z_max_intrusions,z_breaks_0_all,rh_breaks_0_all):
    
    print('-- day',day)
    daylab = day
    if day == '20200213':
        if z_max > 6000:
            daylab = '20200213, upper'
        else:
            daylab = '20200213, lower'
    moist_intrusions[daylab] = {'profiles':{},
                             'fit':{},
                             'stats':{}}
    
    date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
    data_day = data_all.sel(launch_time=day)
    rad_features = rad_features_all[day]
    
    heights_label = r'%s; %1.1f $< z_p <$%1.1f km'%(day, z_min/1e3,z_max/1e3)
    temp, qv, rh, qradlw = getProfiles(rad_features, data_day, z_min, z_max)
    qvstar = qv/rh
    
    # compute interquartile range and median of all profiles types
    temp_Q1, temp_med, temp_Q3 = np.nanpercentile(temp,25,axis=0), np.nanpercentile(temp,50,axis=0), np.nanpercentile(temp,75,axis=0)
    rh_Q1, rh_med, rh_Q3 = np.nanpercentile(rh,25,axis=0), np.nanpercentile(rh,50,axis=0), np.nanpercentile(rh,75,axis=0)
    qradlw_Q1, qradlw_med, qradlw_Q3 = np.nanpercentile(qradlw,25,axis=0), np.nanpercentile(qradlw,50,axis=0), np.nanpercentile(qradlw,75,axis=0)
    qvstar_Q1, qvstar_med, qvstar_Q3 = np.nanpercentile(qvstar,25,axis=0), np.nanpercentile(qvstar,50,axis=0), np.nanpercentile(qvstar,75,axis=0)
    
    # save
    for varid in ['temp','qvstar','rh','qradlw']:

        moist_intrusions[daylab]['profiles'][varid] = {}
        
        for quartile in ['Q1','med','Q3']:
            
            moist_intrusions[daylab]['profiles'][varid][quartile] = getattr(thismodule,'%s_%s'%(varid,quartile))
        
    moist_intrusions[daylab]['profiles']['z'] = z
    moist_intrusions[daylab]['profiles']['pres'] = pres
        
        
    #-- Fits
    # piecewise-linear fit
    z_breaks_id,rh_breaks_id,rh_id = piecewise_fit(z,rh_med,z_breaks_0,rh_breaks_0)

    # remove (upper) intrusion
    rh_breaks_remint = rh_breaks_id.copy()
    rh_breaks_remint[-2] = rh_breaks_id[-3]
    z_breaks_remint = z_breaks_id.copy()
    z_breaks_remint[-2] = z_breaks_id[-3]
    rh_remint = piecewise_linear(z,z_breaks_remint,rh_breaks_remint)

    # Linear fit T(z)
    mask = ~np.isnan(z) & ~np.isnan(temp_med)
    slope, intercept, r, p, se = linregress(z[mask],temp_med[mask])
    print('T(z) = %2.1f z + %3.1f, r = %1.2f'%(slope,intercept,r))
    temp_remint = temp_id = slope*z + intercept

    # power fit qvstar(p)
    mask = ~np.isnan(pres) & ~np.isnan(qvstar_med)
    slope, intercept, r, p, se = linregress(np.log(pres[mask]),np.log(qvstar_med[mask]))
    print('ln(qvstar) = %2.1f ln(p) + %3.1f, r = %1.2f'%(slope,intercept,r))
    print('alpha = %1.2f, qvstar(1000hPa) = %1.2f'%(slope,slope*np.log(1000)+intercept))
    qvstar_remint = qvstar_id = np.exp(slope*np.log(pres) + intercept)

    # save
    for varid in ['z_breaks','rh_breaks','rh','temp','qvstar']:
        for suffix in ['id','remint']:
            varname = '%s_%s'%(varid,suffix)
            moist_intrusions[daylab]['fit'][varname] = getattr(thismodule,varname)

    #-- intrusion statistics
    # intrusion anomaly
    rh_delta_int = rh_id - rh_remint
    
    # intrusion water path
    qvstar = saturationSpecificHumidity(temp_med,pres*100)
    qv_int = rh_delta_int*qvstar
    not_nan = ~np.isnan(qv_int)
    p_top = 300 # hPa
    W_cumul = computeWPaboveZ(qv_int[not_nan],pres[not_nan],p_top)
    W_int = W_cumul[0]
    # center of mass (level)
    where_W_below_half = W_cumul < W_int/2
    p_int_center = pres[not_nan][where_W_below_half][0]
    z_int_center = z[not_nan][where_W_below_half][0]
    # bottom of intrusion
    i_int_bottom = np.where(rh_delta_int>0)[0][0]
    p_int_bottom = pres[i_int_bottom]
    z_int_bottom = z[i_int_bottom]
    # top of intrusion
    i_int_top = np.where(rh_delta_int>0)[0][-1]
    p_int_top = pres[i_int_top]
    z_int_top = z[i_int_top]
    
    # save
    for varid in ['W_int','p_int_center','z_int_center','i_int_top','p_int_top','z_int_top','i_int_bottom','p_int_bottom','z_int_bottom']:
        moist_intrusions[daylab]['stats'][varid] = getattr(thismodule,varid)
    
    print('intrusion top height: %3fhPa, %1.2fkm'%(p_int_top,z_int_top))
    print('intrusion center height: %3fhPa, %1.2fkm'%(p_int_center,z_int_center))
    print('intrusion bottom height: %3fhPa, %1.2fkm'%(p_int_bottom,z_int_bottom))
    print('intrusion mass: %2.2fmm'%W_int)
    
    

#-- save all variables

import pickle

save_path = os.path.join(resultdir,scriptsubdir,'moist_intrusions.pickle')
pickle.dump(moist_intrusions,open(save_path,'wb'))
