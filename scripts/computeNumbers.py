#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:46:19 2022

Numbers in PNAS main 2022

@author: bfildier
"""


##-- modules

import scipy.io
import sys, os, glob
import numpy as np
import xarray as xr
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import Circle
from PIL import Image
from datetime import datetime as dt
from datetime import timedelta, timezone
import pytz
import matplotlib.image as mpimg
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pickle
from scipy.stats import gaussian_kde
from scipy.stats import linregress
from scipy import optimize
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

##-- directories

# workdir = os.path.dirname(os.path.realpath(__file__))
workdir = '/Users/bfildier/Code/analyses/EUREC4A/EUREC4A_organization/scripts'
repodir = os.path.dirname(workdir)
moduledir = os.path.join(repodir,'functions')
resultdir = os.path.join(repodir,'results','radiative_features')
figdir = os.path.join(repodir,'figures','paper')
#inputdir = '/Users/bfildier/Dropbox/Data/EUREC4A/sondes_radiative_profiles/'
inputdir = os.path.join(repodir,'input')
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

mo = MatrixOperators()

##--- local functions

def defineSimDirectories():
    """Create specific subdirectories"""
        
    # create output directory if not there
    os.makedirs(os.path.join(figdir),exist_ok=True)


if __name__ == "__main__":
    
    # arguments
    parser = argparse.ArgumentParser(description="Compute paper numbers from all precomputed data")
    parser.add_argument('--overwrite',type=bool,nargs='?',default=False)

    # output directory
    defineSimDirectories()
    
    ##-- Load all data
    
    exec(open(os.path.join(workdir,"load_data.py")).read())


#%%   Rerecence wavenumbers

    print('-- compute reference wavenumbers --')
    print()
    
    T_ref = 290 # K
    W_ref = 3 # mm
    
    print('choose reference temperature T = %3.1fK'%T_ref)
    print('choose reference water path W = %3.1fmm'%W_ref)
    print()
    
    print("> compute reference wavenumber ")   
    kappa_ref = 1/W_ref # mm-1
    
    rs = rad_scaling_all['20200202']
    nu_ref_rot = rs.nu(kappa_ref,'rot')
    nu_ref_vr = rs.nu(kappa_ref,'vr')
    
    print('reference wavenumber in rotational band: nu = %3.1f cm-1'%(nu_ref_rot/1e2))
    print('reference wavenumber in vibration-rotation band: nu = %3.1f cm-1'%(nu_ref_vr/1e2))
    print()
    
    print("> Planck function at both reference wavenumbers")
    
    piB_ref_rot = pi*rs.planck(nu_ref_rot,T_ref)
    piB_ref_vr = pi*rs.planck(nu_ref_vr,T_ref)  
    
    print('reference Planck term in rotational band: piB = %3.4f J.s-1.sr-1.m-2.cm'%(piB_ref_rot*1e2))
    print('reference Planck term in vibration-rotation band: piB = %3.4f J.s-1.sr-1.m-2.cm'%(piB_ref_vr*1e2))
    
    
#%%  Alpha


    #-- Analytical approximation


    # show temperature profiles
    day = '20200126'
    date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
    data_day = data_all.sel(launch_time=day)
    f = rad_features_all[day]
    
    # colors
    var_col = f.pw
    norm = matplotlib.colors.Normalize(vmin=var_col.min(), vmax=var_col.max())
    cmap = plt.cm.nipy_spectral
    cmap = plt.cm.RdYlBu
    cols = cmap(norm(var_col))
    
    # N data
    Ns = data_day.dims['launch_time']
    
    # Exploratory figure for lapse rate
    
    fig,ax = plt.subplots()
    for i_s in range(Ns):
        ax.plot(data_day.temperature[i_s],data_day.alt,c=cols[i_s],linewidth=0.5,alpha=0.5)
    
    s_fit_FT = slice(200,600)
    s_fit_BL = slice(0,160)
    
    for suff in '_FT','_BL':
        
        s_fit = getattr(thismodule,'s_fit%s'%suff)
        s_dry = f.pw < 30 # mmm
        temp_mean = np.nanmean((data_day.temperature)[s_dry],axis=0)
        not_nan = ~np.isnan(temp_mean)
        z_fit = data_day.alt[not_nan][s_fit]
        # regress
        slope, intercept, r, p, se = scipy.stats.linregress(z_fit,temp_mean[not_nan][s_fit])
        # show
        ax.plot(slope*z_fit+intercept,z_fit,'k')
    
        #!- analytical alpha
        Gamma = -slope
        T_ref = 290
        alpha_an = L_v*Gamma/gg/T_ref * R_d/R_v - 1
        print('alpha_analytical%s ='%suff,alpha_an)
        
    ax.set_xlabel('T (K)')
    ax.set_ylabel('z (km)')
    

#%% Inversion


Ns = rad_scaling_all[day].rad_features.pw.size
fig,ax = plt.subplots()
# Ns = data_all.temperature.shape[0]

for i_s in range(Ns):

    theta = data_day.temperature[i_s] * (1e5/data_day.pressure[i_s])**(R_d/c_pd)

    ax.plot(theta,data_day.pressure[i_s]/100,c = cols[i_s],alpha=0.2)

ax.invert_yaxis()
ax.set_ylabel('p (hPa)')
ax.set_xlabel(r'Potential temperature $\theta$ (K)')



#%% Water paths vs RH

# alpha_qvstar = 2.3
# qvstar_0 = 0.02
# qvstar_power = qvstar_0 * np.power(pres_fit/pres_fit[-1],alpha_qvstar)

def waterPath(qvstar_surf,pres,pres_jump,rh_min,rh_max,alpha,i_surf=-1):
    """Water path from top of atmosphere, in mm
    
    - qv_star_surf: surface saturated specific humidity (kg/kg)
    - pres: reference pressure array (hPa)
    - pres_jump: level of RH jump (hPa)
    - rh_max: lower-tropospheric RH
    - rh_min: upper-tropospheric RH
    - alpha: power exponent
    - i_surf: index of surface layer in array (default is -1, last element)
    """
    
    hPa_to_Pa = 100
    rho_w = 1e3 # kg/m3
    m_to_mm = 1e3
    
    # init
    W = np.full(pres.shape,np.nan)
    # constant
    A = qvstar_surf/(pres[i_surf]*hPa_to_Pa)**alpha/gg/(1+alpha)
    print(A)
    # lower troposphere
    lowert = pres >= pres_jump
    W[lowert] = A*(rh_max*(pres[lowert]*hPa_to_Pa)**(alpha+1)-(rh_max-rh_min)*(pres_jump*hPa_to_Pa)**(alpha+1))
    # upper troposphere
    uppert = pres < pres_jump
    W[uppert] = A*rh_min*(pres[uppert]*hPa_to_Pa)**(alpha+1)
    
    return W/rho_w*m_to_mm

qvstar_0 = 0.02
pres_fit = np.linspace(0,1000,1001)
pres_jump = 800 # hPa
rh_min = 1
rh_max = 1
alpha_qvstar = 2.3

W_prof = waterPath(qvstar_0,pres_fit,pres_jump,rh_min,rh_max,alpha_qvstar)
i_jump = np.where(pres_fit >= pres_jump)[0][0]
W_FT = W_prof[i_jump]

print('Free tropospheric water path at saturation (qvstar integral) =',W_FT)
print('with uniform RH_t = 1%, W =',W_FT/100)
print('with uniform RH_t = 5%, W =',W_FT*0.05)
print('with uniform RH_t = 50%, W =',W_FT*0.5)
print('with uniform RH_t = 80%, W =',W_FT*0.8)
