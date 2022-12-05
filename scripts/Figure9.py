#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 9

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
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

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

    def createMaskNoIntrusions(day):
        
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

    def computeZQrad(day):
        
        z_peak = rad_features_all[day].z_lw_peak/1e3 # km
        
        return z_peak
    
    def computePhiT(day):
        
        f = rad_scaling_all[day].rad_features
        Ns = rad_scaling_all[day].rad_features.pw.size
        
        CRHabove_peak = np.full(Ns,np.nan)
        
        for i_s in range(Ns):
            
            i_peak = f.i_lw_peak[i_s]
            
            # CRH above = W(p)/Wsat(p)
            CRHabove = f.wp_z[i_s]/f.wpsat_z[i_s]
            # store at peak height
            CRHabove_peak[i_s] = CRHabove[i_peak]
            
        return CRHabove_peak
    
    
#%% RH_t vs. cloud fraction


    # Figure layout
    fig = plt.figure(figsize=(5,7.5))
    
    gs = GridSpec(5, 2, width_ratios=[1,1], height_ratios=[1,1,0.05,1,1.5],hspace=0,wspace=0.1)
    ax1 = fig.add_subplot(gs[0],projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(gs[1],projection=ccrs.PlateCarree())
    ax3 = fig.add_subplot(gs[2],projection=ccrs.PlateCarree())
    ax4 = fig.add_subplot(gs[3],projection=ccrs.PlateCarree())
    ax5 = fig.add_subplot(gs[3:5,:])
    
    #--- (a-d) maps of patterns
    
    axs = ax1,ax2,ax3,ax4
    
    names_4patterns = 'Fish','Flower','Gravel','Sugar'
    days_4patterns = '20200122','20200202','20200205','20200209'

    for ax,name_p,day_p in zip(axs,names_4patterns,days_4patterns):
        
        image = Image.open(os.path.join(workdir,'../images/patterns/PNG','GOES16__%s_1400.png'%day_p))
        ax.imshow(np.asarray(image),extent=[*lon_box,*lat_box],origin='upper')
        # HALO circle
        circ = Circle((lon_center,
                        lat_center),
                      r_circle, linewidth=0.8,
                      ec='w',
                      fill=False)
        ax.add_patch(circ)
        # Barbados island
        res = '50m'
        land = cfeature.NaturalEarthFeature('physical', 'land', \
                                            scale=res, edgecolor='k', \
                                            facecolor=cfeature.COLORS['land'])
        ax.add_feature(land, facecolor='beige')
        
        #- add pattern name
        # ax.text(0.98,0.03,name_p,c='w',ha='right',transform=ax.transAxes,fontsize=14)
        #- add date
        pat_name = day_p[:4]+'-'+day_p[4:6]+'-'+day_p[6:8]+'\n'+name_p
        ax.text(0.98,0.98,pat_name,c='w',ha='right',va='top',transform=ax.transAxes,fontsize=10)
        # ax.outline_patch.set_visible(False)
        
        rect = mpatches.Rectangle((0,0), width=1, height=1,edgecolor=col_pattern[name_p], facecolor="none",linewidth=3,alpha=1, transform=ax.transAxes)
        ax.add_patch(rect)
    
    
    #--- (e) scatter of peak heights as a function of PW
    
    ax = ax5

    s = {}
    phi_t_all = {}
    qrad_int_all = {}
    mask_geometry = {}
    mask_no_intrusions = {}
    mask_clouds = {}
    mask_dry = {}
    
    p_surf = 950 # hPa
    
    ##-- Compute
    
    for day in days:
                
        # store
        phi_t_all[day] = computePhiT(day)
        # phi_t_all[day] = computePhiT(day,4000)
        qrad_int_all[day] = computeQradIntegral(day,p_surf)
        
        ## masks
        # geometry
        mask_geometry[day] = createMask(day)
        # intrusions
        mask_no_intrusions[day] = createMaskNoIntrusions(day)
        # clouds
        mask_clouds[day] = createMaskClouds(day)
        # dry columns
        mask_dry[day] = createMaskDry(day)
    
    ##-- Display
    
    for day,pat in zip(days,name_pattern):

        # mask intrusions here
        all_masks = [mask_geometry[day],mask_no_intrusions[day],mask_clouds[day],mask_dry[day]]
        all_show = [True,True,False,None]
        k = combineMasks(all_masks,all_show)
        
        print(' . day ',day,',',pat)
        print(np.sum(k))

        if  np.sum(k) <= 5:
            print(' --> pass')
            continue

        # x values
        phi_t_mean = np.nanmean(phi_t_all[day][k]) * 100
        # y values
        cloud_frac_mean = np.mean(iorg_data.cloud_fraction.sel(time=day)) * 100
        # marker sizes
        qrad_int_mean = np.nanmean(qrad_int_all[day][k])
        # circles
        pw = rad_features_all[day].pw
        Ndry = np.sum(pw[k] < 30)
        
        print(Ndry, qrad_int_mean)

        # z = ((np.absolute(qrad_int_mean)-2.5)**2)*150
        z = ((np.absolute(qrad_int_mean)-2.5)*120)
        c = col_pattern[pat]

        # show date
        yoffset = -1
        xoffset = 0.3
        if day == '20200205':
            yoffset = -2.5
        if day in ['20200126','20200128']:
            xoffset = 0.5
          
        # egde color based on number of remaining soundings in dry region
        ec = 'none'
        if Ndry >= 20:
            ec = 'k'
            
        # show
        ax.scatter(phi_t_mean,cloud_frac_mean,s=z,color=c,edgecolor=ec,linewidth=1.2,alpha=0.6)
        
        # show date
        date = dt.strptime(day,'%Y%m%d').strftime('%Y-%m-%d')
        ax.text(phi_t_mean+xoffset,cloud_frac_mean+yoffset,date,c='dimgrey',ha='left',va='bottom',rotation='horizontal',fontsize=8)
        
    ax.set_xlabel(r'Mean relative humidity above peak, $\varphi_t$ (%)')
    ax.set_ylabel(r'Cloud fraction (%)')

    ax.set_xlim((6,19))
    ax.set_ylim((0,35))
    
    ## Fully-manual legend QRAD
    rect = mpatches.Rectangle((0.02,0.015), width=0.3, height=0.13,edgecolor='grey', facecolor="none",linewidth=0.5,alpha=0.5, transform=ax.transAxes)
    ax.add_patch(rect)
    for qint,y in zip([3,4],[0.04,0.09]):
        
        # s = ((np.absolute(qint)-2.5)**2)*150/sqrt(20)
        s = ((np.absolute(qint)-2.5)*120)
        
        circle = mlines.Line2D([0], [0], marker='o', color='w',
                        markerfacecolor='r', markersize=s)
        ax.scatter(0.06,y+0.01,s=s,c='k',edgecolor='none',transform=ax.transAxes)
        ax.text(0.1,y+0.005,s=r'$\left\vert \int Q\right\vert >%d$ K/day'%qint,fontsize=7,transform=ax.transAxes)

    ## legend pattern
    for pat in col_pattern.keys():
        print(pat)
        lab = pat
        if pat == '':
            lab = 'Unclassified'
        setattr(thismodule,"h_%s"%pat,mpatches.Patch(color=col_pattern[pat],linewidth=0,alpha=0.6,label=lab))
    
    # handles = [h_Fish,h_Flower,h_Gravel,h_Sugar,h_]
    handles = [h_Fish,h_Flower,h_Gravel,h_Sugar]
    ax.legend(loc='upper right',handles=handles,ncol=1,fontsize=7)

    #--- Add panel labeling
    pan_labs = '(a)','(b)','(c)','(d)','(e)'
    pan_cols = 'w','w','w','w','k'
    axs = ax1,ax2,ax3,ax4,ax5
    for ax,pan_lab,pan_col in zip(axs,pan_labs,pan_cols):
        ax.text(0.02,0.98,pan_lab,c=pan_col,ha='left',va='top',
                transform=ax.transAxes,fontsize=12)
    
    #--- save
    plt.savefig(os.path.join(figdir,'Figure9.pdf'),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,'Figure9.png'),bbox_inches='tight',dpi=300)