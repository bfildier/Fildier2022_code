#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure S4

Created on Wed Jul 13 14:33:02 2022

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
    
    
#%%    Functions for masks and Boundary layer integral vs. PW

    def createMask(day):
        
        # remove surface peaks
        z_peak = rad_features_all[day].z_lw_peak/1e3 # km
        remove_sfc = z_peak > 0.5 # km
        
        # keep large peaks only
        qrad_peak = np.absolute(rad_features_all[day].qrad_lw_peak)
        keep_large = np.logical_and(remove_sfc,qrad_peak > 2.5) # K/day
        
        # keep soundings in domain of interest
        lon_day = data_all.sel(launch_time=day).longitude.data[:,50]
        lat_day = data_all.sel(launch_time=day).latitude.data[:,50]
        keep_box = np.logical_and(lon_day < lon_box[1], lat_day >= lat_box[0])
        
        # merge all
        keep = np.logical_and(keep_large,keep_box)
        
        return keep

    def createMaskIntrusions(day):
        
        z_peak = rad_features_all[day].z_lw_peak/1e3 # km
        
        # if day in days_high_peaks:
        if day in days_intrusions:
            
            # subset without intrusions
            i_d_all = np.where(np.array(days_intrusions) == day)[0]
            
            k_high = False
            
            for i_d in i_d_all:
                
                k_high = np.logical_or(k_high,np.logical_and(z_peak <= z_max_intrusions[i_d],
                                        z_peak > z_min_intrusions[i_d]))
            
            k_low = np.logical_not(k_high)
        
            return k_low
        
        else:
            
            return np.full(z_peak.shape,True)
        
    def createMaskClouds(day,rh_cloud = 0.95):
        
        data_day = data_all.sel(launch_time=day)
        # select moist-convecting levels
        isAboveLCL = data_day.alt.data > 700
        # select levels above 95% RH
        isLevelSaturated = data_day.relative_humidity.data > rh_cloud
        # is there one level meeting both criteria
        iscloud_day = np.any(np.logical_and(isAboveLCL,isLevelSaturated),axis=1)
    
        return iscloud_day
    
    def createMaskDry(day,pw_max = 30):
        
        pw = rad_features_all[day].pw
        
        mask_dry = pw <= pw_max
        
        return mask_dry


    def combineMasks(all_masks,all_show):
        
        mask_combined = True
        
        for mask,show in zip(all_masks,all_show):
        
            if show is not None:
                
                # add either 'mask' or its complement, depending on 'show'
                mask_combined *= (show * mask + np.logical_not(show) * ~mask)
        
        return mask_combined
    
    
    def computeQradIntegral(day,p_surf):

        rs = rad_scaling_all[day]
        f = rad_scaling_all[day].rad_features
        # k_bottom = np.where(np.logical_not(np.isnan(f.wp_z[0])))[0][0]
        
        #- approximated BL cooling
        Ns = rad_scaling_all[day].rad_features.pw.size
        qrad_int = np.full(Ns,np.nan)
        
        for i_s in range(Ns):
            i_peak = f.i_lw_peak[i_s]
            p_peak = f.pres_lw_peak[i_s]
            
                        
            # compute integral cooling

            s_notnans = slice(12,None)
            qrad_lw_not_nans = np.flipud(rad_scaling_all[day].rad_features.qrad_lw_smooth[i_s][s_notnans])
            pres_not_nans = np.flipud(f.pres[s_notnans])
            
            qrad_int[i_s] = f.mo.pressureIntegral(arr=qrad_lw_not_nans,pres=pres_not_nans,p0=p_peak,p1=p_surf,z_axis=0) / \
                                    f.mo.pressureIntegral(arr=np.ones(qrad_lw_not_nans.shape),pres=pres_not_nans,p0=p_peak,p1=p_surf,z_axis=0)

        return qrad_int

    def computeQradPeak(day):
        
        qrad_peak = rad_features_all[day].qrad_lw_peak
        
        return qrad_peak
        
    def computeZQrad(day):
        
        z_peak = rad_features_all[day].z_lw_peak/1e3 # km
        
        return z_peak
    
    
#%% Figure S4

i_fig = 4

##--- compute data

z_peak_all = {}
qrad_peak_all = {}
qrad_int_all = {}
mask_geometry = {}
mask_no_intrusions = {}
mask_clouds = {}
mask_dry = {}

p_surf = 950 # hPa

##-- Compute

for day in days:
            
    # store
    z_peak_all[day] = computeZQrad(day)
    qrad_peak_all[day] = computeQradPeak(day)
    qrad_int_all[day] = computeQradIntegral(day,p_surf)
    
    ## masks
    # geometry
    mask_geometry[day] = createMask(day)
    # intrusions
    mask_no_intrusions[day] = createMaskIntrusions(day)
    # clouds
    mask_clouds[day] = createMaskClouds(day)
    # dry columns
    mask_dry[day] = createMaskDry(day)
    

fig, axs = plt.subplots(nrows=3,figsize=(4,11))


#--- (a) scatter of peak heights as a function of PW

ax = axs[0]

for day,pat,conf in zip(days,name_pattern,confidence_pattern):
    
    pw = rad_features_all[day].pw
    z_peak = z_peak_all[day]
    qrad_peak = qrad_peak_all[day]
    
    # circle diameter proportional to size of peak
    s = np.absolute(qrad_peak)*2
    c = col_pattern[pat]
    
    all_masks = [mask_geometry[day],mask_no_intrusions[day],mask_clouds[day],mask_dry[day]]
    show_noint_nocld = [True,True,False,None]
    show_noint_cld = [True,True,True,None]
    show_int = [True,False,None,None]
    
    for suff in 'noint_nocld','noint_cld','int':
        show_which = getattr(thismodule,'show_%s'%suff)
        setattr(thismodule,'mask_%s'%suff,combineMasks(all_masks,show_which))
    
    # intrusions
    ax.scatter(pw[mask_int],z_peak[mask_int],s=s[mask_int],c='grey',edgecolor='none',alpha=0.2)
    
    # cloudy sondes, no intrusion
    ax.scatter(pw[mask_noint_cld],z_peak[mask_noint_cld],s=s[mask_noint_cld],c='none',edgecolor=c,linewidth=0.5,alpha=0.2)
    
    # clear-sky sondes, no intrusion
    ax.scatter(pw[mask_noint_nocld],z_peak[mask_noint_nocld],s=s[mask_noint_nocld],c=c,edgecolor='none',alpha=0.3)

ax.set_xlabel('PW (mm)')
ax.set_ylabel(r'Peak height (km)',labelpad=20)
ax.set_ylim((-0.3,8.1))


#--- (b) scatter of peak magnitudes as a function of PW

ax = axs[1]


for day,pat,conf in zip(days,name_pattern,confidence_pattern):
    
    pw = rad_features_all[day].pw
    z_peak = z_peak_all[day]
    qrad_peak = qrad_peak_all[day]
    Ns = pw.size
    
    # circle diameter proportional to size of peak
    s = 10*np.ones(Ns)
    c = col_pattern[pat]
    
    all_masks = [mask_geometry[day],mask_no_intrusions[day],mask_clouds[day],mask_dry[day]]
    show_noint_nocld = [True,True,False,None]
    show_noint_cld = [True,True,True,None]
    show_int = [True,False,None,None]
    
    for suff in 'noint_nocld','noint_cld','int':
        show_which = getattr(thismodule,'show_%s'%suff)
        setattr(thismodule,'mask_%s'%suff,combineMasks(all_masks,show_which))
    
    # intrusions
    ax.scatter(pw[mask_int],qrad_peak[mask_int],s=s[mask_int],c='grey',edgecolor='none',alpha=0.2)
    
    # cloudy sondes, no intrusion
    ax.scatter(pw[mask_noint_cld],qrad_peak[mask_noint_cld],s=s[mask_noint_cld],c='none',edgecolor=c,linewidth=0.5,alpha=0.2)
    
    # clear-sky sondes, no intrusion
    ax.scatter(pw[mask_noint_nocld],qrad_peak[mask_noint_nocld],s=s[mask_noint_nocld],c=c,edgecolor='none',alpha=0.3)
    

ax.set_xlabel('PW (mm)')
ax.set_ylabel(r'Peak magnitude (K/day)')
ax.set_ylim((-14.1,-2.3))


#--- (c) scatter of integral cooling magnitudes as a function of PW

ax = axs[2]


for day,pat,conf in zip(days,name_pattern,confidence_pattern):
    
    pw = rad_features_all[day].pw
    z_peak = z_peak_all[day]
    qrad_peak = qrad_peak_all[day]
    qrad_int = qrad_int_all[day]
    print(np.sum(np.isnan(qrad_int)))
    
    # circle diameter proportional to size of peak
    s = np.absolute(qrad_peak)*2
    c = col_pattern[pat]
    
    all_masks = [mask_geometry[day],mask_no_intrusions[day],mask_clouds[day],mask_dry[day]]
    show_noint_nocld = [True,True,False,None]
    show_noint_cld = [True,True,True,None]
    show_int = [True,False,None,None]
    
    for suff in 'noint_nocld','noint_cld','int':
        show_which = getattr(thismodule,'show_%s'%suff)
        setattr(thismodule,'mask_%s'%suff,combineMasks(all_masks,show_which))
    
    # intrusions
    ax.scatter(pw[mask_int],qrad_int[mask_int],s=s[mask_int],c='grey',edgecolor='none',alpha=0.2)
    
    # cloudy sondes, no intrusion
    ax.scatter(pw[mask_noint_cld],qrad_int[mask_noint_cld],s=s[mask_noint_cld],c='none',edgecolor=c,linewidth=0.5,alpha=0.2)
    
    # clear-sky sondes, no intrusion
    ax.scatter(pw[mask_noint_nocld],qrad_int[mask_noint_nocld],s=s[mask_noint_nocld],c=c,edgecolor='none',alpha=0.3)


ax.set_xlabel('PW (mm)')
ax.set_ylabel(r'Integral cooling (K/day)')
ax.set_ylim((-5.4,-1.8))


## Fully-manual legend QRAD
x_box = 0.65
y_box = 0.2
rect = mpatches.Rectangle((x_box+0.02,y_box+0.015), width=0.3, height=0.13,edgecolor='grey', facecolor="none",linewidth=0.5,alpha=0.5, transform=ax.transAxes)
ax.add_patch(rect)
for qpeak,y in zip([5,10],[0.04,0.09]):
    
    # s = ((np.absolute(qint)-2.5)**2)*150/sqrt(20)
    s = np.absolute(qpeak)*2/sqrt(3)
    
    circle = mlines.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor='r', markersize=s)
    ax.scatter(x_box+0.06,y_box+y+0.01,s=s,c='k',edgecolor='none',transform=ax.transAxes)
    ax.text(x_box+0.08,y_box+y,s=r'$\left\vert H^*\right\vert >%d$ K/day'%qpeak,fontsize=7,transform=ax.transAxes)

## legend pattern
for pat in col_pattern.keys():
    print(pat)
    lab = pat
    if pat == '':
        lab = 'Moist intrusions'
    setattr(thismodule,"h_%s"%pat,mpatches.Patch(color=col_pattern[pat],linewidth=0,alpha=0.6,label=lab))
h_cld = mpatches.Patch(color='w',edgecolor='k',linewidth=1,alpha=1,label='cloudy')
ax.legend(loc='lower right',handles=[h_Fish,h_Flower,h_Gravel,h_Sugar,h_,h_cld],ncol=2,fontsize=7)


for ax in axs:
    
    ax.set_xlim(17,48)

#--- Add panel labeling
pan_labs = '(a)','(b)','(c)'
pan_cols = 'k','k','k'
for ax,pan_lab,pan_col in zip(axs,pan_labs,pan_cols):
    t = ax.text(0.02,0.98,pan_lab,c=pan_col,ha='left',va='top',
            transform=ax.transAxes,fontsize=16)
    t.set_bbox(dict(facecolor='w', alpha=0.5, edgecolor='none'))

#--- save
plt.savefig(os.path.join(figdir,'FigureS%d.pdf'%i_fig),bbox_inches='tight')
# plt.savefig(os.path.join(figdir,'FigureS%d.png'%i_fig),dpi=300,bbox_inches='tight')