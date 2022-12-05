#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure S5

Created on Wed Aug  3 10:18:25 2022

@author: bfildier
"""



##-- modules

# general
import scipy.io
import sys, os, glob
import numpy as np
import xarray as xr
from datetime import datetime as dt
from datetime import timedelta, timezone
import pytz
import pickle
import argparse

# stats
from scipy.stats import gaussian_kde
from scipy.stats import linregress
from scipy import optimize

# images
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import Circle
from PIL import Image
import matplotlib.image as mpimg
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

##-- directories

# workdir = os.path.dirname(os.path.realpath(__file__))
workdir = '/Users/bfildier/Code/analyses/EUREC4A/EUREC4A_organization/scripts'
repodir = os.path.dirname(workdir)
moduledir = os.path.join(repodir,'functions')
resultdir = os.path.join(repodir,'results','radiative_features')
figdir = os.path.join(repodir,'figures','paper')
inputdir = '/Users/bfildier/Dropbox/Data/EUREC4A/sondes_radiative_profiles/'
radinputdir = os.path.join(repodir,'input')
imagedir = os.path.join(repodir,'figures','snapshots','with_HALO_circle')
scriptsubdir = 'Fildier2021'

# Load own module
projectname = 'EUREC4A_organization'
thismodule = sys.modules[__name__]

## Own modules
sys.path.insert(0,moduledir)
print("Own modules available:", [os.path.splitext(os.path.basename(x))[0]
                                 for x in glob.glob(os.path.join(moduledir,'*.py'))])

from radiativefeatures import *
from radiativescaling import *
# from thermodynamics import *
from conditionalstats import *
from matrixoperators import *
from thermoConstants import *
from thermoFunctions import *

mo = MatrixOperators()

##--- local functions

def defineSimDirectories():
    """Create specific subdirectories"""
        
    # create output directory if not there
    os.makedirs(os.path.join(figdir),exist_ok=True)
    
    
if __name__ == "__main__":
    
    # arguments
    parser = argparse.ArgumentParser(description="Draw paper figures from all precomputed data")
    parser.add_argument('--overwrite',type=bool,nargs='?',default=False)

    # output directory
    defineSimDirectories()
    
    ##-- Load all data
    
    exec(open(os.path.join(workdir,"load_data.py")).read())
    

#%% Figure 5 -- observed moist intrusions

i_fig = 5

z_breaks_0_all = [1.8,2,4,5,6.5,7], [1.8,2,4,5], [2,3.5,4.5,5], [2,2.5,4.5,5], [2,2.5,5,7], [2,2.5,4.5,5] 
rh_breaks_0_all = [0.8,0.1,0.7,0.1,0.7,0.1], [0.8,0.1,0.7,0.1], [0.8,0.3,0.7,0.05], [0.75,0.1,0.25,0.05], [0.75,0.1,0.25,0.05], [0.75,0.1,0.25,0.05] 
colors = 'orange','blue','green','red','purple','brown'

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
    # lw cooling
    qradlw = rad_features.qrad_lw_smooth[k,:]
    
    return temp, rh, qradlw


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



fig,axs = plt.subplots(ncols=3,figsize=(11,4.5))
plt.rc('legend',fontsize=7)
plt.rc('legend',labelspacing=0.07)

for day, z_min, z_max, z_breaks_0, rh_breaks_0,col \
in zip(days_intrusions,z_min_intrusions,z_max_intrusions,z_breaks_0_all,rh_breaks_0_all,colors):
    
    date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
    data_day = data_all.sel(launch_time=day)
    rad_features = rad_features_all[day]
    
    # heights_label = r'%2d/%2d; %1.1f $< z^\star <$%1.1f km'%(date.month,date.day, z_min,z_max)
    heights_label = r'%2d/%2d; $z^\star \in $ [%1.1f,%1.1f] km'%(date.month,date.day, z_min,z_max)
    temp, rh, qradlw = getProfiles(rad_features, data_day, z_min*1e3, z_max*1e3)
    
    # compute interquartile range and median of all profiles types
    temp_Q1, temp_med, temp_Q3 = np.nanpercentile(temp,25,axis=0), np.nanpercentile(temp,50,axis=0), np.nanpercentile(temp,75,axis=0)
    rh_Q1, rh_med, rh_Q3 = np.nanpercentile(rh,25,axis=0), np.nanpercentile(rh,50,axis=0), np.nanpercentile(rh,75,axis=0)
    qradlw_Q1, qradlw_med, qradlw_Q3 = np.nanpercentile(qradlw,25,axis=0), np.nanpercentile(qradlw,50,axis=0), np.nanpercentile(qradlw,75,axis=0)
    
    #-- everything more quantitative about intrusions
    # piecewise-linear fit
    z_breaks_id,rh_breaks_id,rh_id = piecewise_fit(z,rh_med,z_breaks_0,rh_breaks_0)
    
    # remove intrusion
    rh_breaks_remint = rh_breaks_id.copy()
    rh_breaks_remint[-2] = rh_breaks_id[-3]
    z_breaks_remint = z_breaks_id.copy()
    z_breaks_remint[-2] = z_breaks_id[-3]
    rh_remint = piecewise_linear(z,z_breaks_remint,rh_breaks_remint)
    
    # intrusion anomaly
    rh_delta_int = rh_id - rh_remint
    
    # intrusion water path
    qvstar = saturationSpecificHumidity(temp_med,pres*100)
    qv_int = rh_delta_int*qvstar
    not_nan = ~np.isnan(qv_int)
#     p_notnan = pres[not_nan][0]
    p_top = 300 # hPa
    W_cumul = computeWPaboveZ(qv_int[not_nan],pres[not_nan],p_top)
    W_int = W_cumul[0]
    # center of mass (level)
    where_W_below_half = W_cumul < W_int/2
    p_center = pres[not_nan][where_W_below_half][0]
    z_center = z[not_nan][where_W_below_half][0]
    
    print('intrusion height: %3fhPa, %1.2fkm'%(p_center,z_center))
    print('intrusion mass: %2.2fmm'%W_int)
    
    # full RH
    axs[0].plot(rh_med*100,z,linewidth=1,alpha=1,label=heights_label,c=col)
    axs[0].fill_betweenx(z,rh_Q1*100,rh_Q3*100,alpha=0.1,facecolor=col)
    # Rh in intrusion
    axs[1].plot(rh_delta_int*100,z,linewidth=1,alpha=1,label='h=%1.1fkm, W=%1.2fmm'%(z_center,W_int),linestyle='-',c=col)
    # qrad
    axs[2].plot(qradlw_med,z,linewidth=1,alpha=1,c=col)
    axs[2].fill_betweenx(z,qradlw_Q1,qradlw_Q3,alpha=0.1,facecolor=col)
    
axs[0].set_xlabel(r'Relative humidity $\varphi$ (%)')
axs[1].set_xlabel(r'Intrusion $\Delta \varphi$ (%)')
axs[2].set_xlabel('LW cooling (K/day)')
axs[0].set_ylabel('z (km)')

axs[0].set_title('Observed moist intrusions')
axs[1].set_title('Fitted anomalous water')
axs[2].set_title('Longwave cooling')

axs[0].legend(loc='upper right',fontsize=9)
axs[1].legend(loc='upper right',fontsize=9)
# axs[0].legend(labelspacing = 0.5)

#--- Add panel labeling
pan_labs = '(a)','(b)','(c)'
for ax,pan_lab in zip(axs,pan_labs):
    t = ax.text(0.03,0.92,pan_lab,transform=ax.transAxes,fontsize=14)
    t.set_bbox(dict(facecolor='w',alpha=0.8,edgecolor='none'))

#--- save
plt.savefig(os.path.join(figdir,'Figure%d.pdf'%i_fig),bbox_inches='tight')
# plt.savefig(os.path.join(figdir,'Figure%d.png'%i_fig),dpi=300,bbox_inches='tight')