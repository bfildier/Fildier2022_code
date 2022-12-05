#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute and store features of radiative cooling data and sonde data during 
the EUREC4A campaign.

Created on Tue Feb  2 10:51:58 2021

@author: bfildier
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import xarray as xr
import pandas as pd
import pytz
from datetime import datetime as dt
from datetime import timedelta, timezone
import sys,os,glob
import IPython
import argparse
import pickle
from matplotlib import cm
# import matplotlib.image as mpimg
from math import ceil

# geodesic distances and displacements
import geopy
import geopy.distance
# map display
# import cartopy.crs as ccrs

# ## Graphical parameters
# plt.style.use(os.path.join(matplotlib.get_configdir(),'stylelib/presentation.mplstyle'))

##-- directories and modules

workdir = os.path.dirname(os.path.realpath(__file__))
repodir = os.path.dirname(workdir)
moduledir = os.path.join(repodir,'functions')
subdirname = 'radiative_features'
resultdir = os.path.join(repodir,'results',subdirname)
figdir = os.path.join(repodir,'figures',subdirname)
#inputdir = '/Users/bfildier/Dropbox/Data/EUREC4A/sondes_radiative_profiles/'
inputdir = os.path.join(repodir,'input')

# Load own module
projectname = 'EUREC4A_organization'
# while os.path.basename(repodir) != projectname:
#     repodir = os.path.dirname(repodir)
thismodule = sys.modules[__name__]

## Own modules

sys.path.insert(0,moduledir)
print("Own modules available:", [os.path.splitext(os.path.basename(x))[0]
                                 for x in glob.glob(os.path.join(moduledir,'*.py'))])

from radiativefeatures import *
# from thermodynamics import *
from conditionalstats import *
from matrixoperators import *

##--- local functions

def defineSimDirectories(day):
        
    # create output directory if not there
    os.makedirs(os.path.join(resultdir,day),exist_ok=True)


#%% main

if __name__ == "__main__":
    
    # arguments
    parser = argparse.ArgumentParser(description="Compute and store features from radiative profile data")
    parser.add_argument('--day', default='20200126',help="day of analysis")
    parser.add_argument('--overwrite',type=bool,nargs='?',default=False)
    
    args = parser.parse_args()
    day = args.day
    
    # day = '20200126'
    date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
    ref_varid = 'PW'
    ref_var_min = 15 # mm
    ref_var_max = 55 # mm
    nbins = int((ref_var_max-ref_var_min)/2)
    
    cond_varids = 'pressure','q_rad','q_rad_sw','q_rad_lw','specific_humidity','u_norm','temperature'
    cond_varids_cap = 'P','QRAD','QRADSW','QRADLW','QV','UNORM','T'
    
    # cond_varids = 'pressure',
    # cond_varids_cap = 'P',
    
    defineSimDirectories(day)
    
    ##-- import data
    # load radiative profiles
    radprf = xr.open_dataset(os.path.join(inputdir,'rad_profiles_CF.nc'))
    
    # choose profiles for that day that start at bottom
    data_all = radprf.where(radprf.z_min<=50,drop=True)
    data_day = data_all.sel(launch_time=day)
    times = np.array([pytz.utc.localize(dt.strptime(str(d)[:19],'%Y-%m-%dT%H:%M:%S')) for d in data_day.launch_time.values])
    
    dim_t,dim_z = 0,1
    
    # compute all PW values
    mo = MatrixOperators()
    
    qv = data_day.specific_humidity.data # kg(w)/kg(a)
    temp = data_day.temperature.data # K
    pres = data_day.pressure.data/1e2 # hPa
    pres_mean = np.nanmean(pres,axis=dim_t) # hPa
    z = data_day.alt.data # m
    launch_time = data_day.launch_time.values
    # p_levmin = 1010 # hPa
    # p_levmax = 200 # hPa
    # PW_all = mo.pressureIntegral(QV_all,pres,p_levmin=p_levmin,p_levmax=p_levmax,z_axis=dim_z)
    # PW_min = np.nanmin(PW_all)
    # PW_max = np.nanmax(PW_all)
    
    ##-- compute radiative features
    print("- compute radiative features")
    
    # Initialize
    # f = FeaturesFromXarray()
    f = Features(pres_mean,z,launch_time)
    # Find peaks in net Q_rad
    f.computePeaks(data_day.q_rad.data,which='net')
    # Find peaks in LW Q_rad
    f.computePeaks(data_day.q_rad_lw.data,which='lw')
    # Find peaks in SW Q_rad
    f.computePeaks(data_day.q_rad_sw.data,which='sw')
    # Compute PW
    f.computePW(qv,temp,pres,z)
    # Compute water path above z
    f.computeWPaboveZ(qv,pres_mean,z_axis=dim_z)
    # Compute water path above z
    f.computeSaturatedWPaboveZ(temp,pres_mean,z_axis=dim_z)
    # Compute PW truncated at qrad peak
    f.computePW(qv,temp,pres,z,i_z_max=f.i_net_peak,attr_name='pw_below_net_qrad_peak')
    # Compute PW truncated at lw qrad peak
    f.computePW(qv,temp,pres,z,i_z_max=f.i_lw_peak,attr_name='pw_below_lw_qrad_peak')
    
    # Save
    features_filename = 'rad_features.pickle'
    print('saving %s'%features_filename)
    features_path = os.path.join(resultdir,day,features_filename)
    pickle.dump(f,open(features_path,'wb'))
    
    ##-- compute conditional statistics
    
    #-- reference PW distribution
    #- if reference distribution exists, load it
    ref_filename = 'dist_%s.pickle'%(ref_varid)
    ref_dist_path = os.path.join(resultdir,day,ref_filename)
    ref_dist_exists = len(glob.glob(ref_dist_path)) > 0


    if ref_dist_exists and not args.overwrite:
    
        print('-load existing reference %s distribution'%ref_varid)
        ref_dist = pickle.load(open(ref_dist_path,'rb'))
    
    else:
    
        print("- compute reference %s distribution"%ref_varid)
        # fix range to the total range for each time slice
        var = f.pw
        # compute the distribution
        ref_dist = Distribution(name=ref_varid,nppb=1,nlb=nbins)
        ref_dist.computeDistribution(var,vmin=ref_var_min,vmax=ref_var_max)
        ref_dist.storeSamplePoints(var,method='shuffle_mask',
                                         sizemax=50)
        # save reference distribution
        print('saving %s'%ref_filename)
        pickle.dump(ref_dist,open(ref_dist_path,'wb'))
    
    #-- conditional distributions
    print("- compute conditional distributions")
    
    for cond_varid,cond_varid_cap in zip(cond_varids,cond_varids_cap):

        cond_filename = 'cdist_%s_on_%s.pickle'%(cond_varid_cap,ref_varid)
        cond_dist_path = os.path.join(resultdir,day,cond_filename)
        cond_dist_exists = len(glob.glob(cond_dist_path)) > 0
        
        if not cond_dist_exists or args.overwrite:
            print('. for %s'%cond_varid)
            
            if cond_varid == 'u_norm':
                cond_var = np.sqrt(np.power(np.swapaxes(getattr(data_day,'u_wind').data,0,1),2),\
                                   np.power(np.swapaxes(getattr(data_day,'v_wind').data,0,1),2))
            else:
                cond_var = np.swapaxes(getattr(data_day,cond_varid).data,0,1)
            # initialize
            cond_dist = ConditionalDistribution(name=cond_varid_cap,is3D=True,on=ref_dist)
            # compute
            cond_dist.computeConditionalMeanAndVariance(cond_var)
            # save conditional distribution
            print('saving %s'%cond_filename)
            pickle.dump(cond_dist,open(cond_dist_path,'wb'))
            
            # keep in local environment
            setattr(thismodule,"cdist_%s_on_%s"%(cond_varid_cap,ref_varid),cond_dist)
        
    
    # ##-- check with plots
    # cmap=plt.cm.viridis_r

    # fig,ax = plt.subplots(figsize=(6,5))
    
    # array = cdist_q_rad_lw_on_PW.cond_mean
    # h = ax.imshow(array,
    #           aspect=5,
    #           origin='lower',
    #           extent=[ref_var_min,ref_var_max,f.z[0]/1000,f.z[-1]/1000],
    #           cmap=cmap)
    
    # ax.set_xlabel('PW (mm)')
    # ax.set_ylabel('z (km)')
    # ax.set_title('Longwave cooling on %s'%date.strftime("%Y-%m-%d"))
    
    # # colorbar
    # # plt.colorbar(h)
    # norm = matplotlib.colors.Normalize(vmin=np.nanmin(array), vmax=np.nanmax(array))
    # cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
    #              ax=ax,shrink=0.9,pad=0.06)
    # cb.set_label(r'Longwave $Q_{rad}$ (K/day)')

    
    sys.exit(0)
    
