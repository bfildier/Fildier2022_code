#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4

Created on Thu Jun 16 16:05:54 2022

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
        
        if day in days_high_peaks:
            
            # subset without intrusions
            i_d = np.where(np.array(days_high_peaks) == day)[0][0]
            k_high = np.logical_and(z_peak <= z_max_all[i_d],
                                    z_peak > z_min_all[i_d])
            k_low = np.logical_not(k_high)
        
            return k_low
        
        else:
            
            return np.full(z_peak.shape,True)
        
        
    def createMaskClouds(day,rh_cloud = 0.95):
        
        data_day = data_all.sel(launch_time=day)
        iscloud_day = np.any(data_day.relative_humidity > rh_cloud,axis=1).data
    
        return iscloud_day

    def combineMasks(mask_geom,mask_noint,mask_nocloud,show_geom,show_noint,show_nocloud):
        
        mask_combined = True
        
        for mask,show in zip([mask_geom,mask_noint,mask_nocloud],[show_geom,show_noint,show_nocloud]):
        
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
    
    
#%% Draw Figure 4

    i_fig = 4

    label_jump = '^\dagger'
    m_to_cm = 1e2
    day_to_seconds = 86400
    hPa_to_Pa = 1e2
    
    days2show = days

    def computeBeta(pres,pres_jump,rh_min,rh_max,alpha,i_surf=-1):
        """beta exponent
        
        Arguments:
        - pres: reference pressure array (hPa)
        - pres_jump: level of RH jump (hPa)
        - rh_max: lower-tropospheric RH
        - rh_min: upper-tropospheric RH
        - alpha: power exponent
        """
        
        hPa_to_Pa = 100 
    
        # init
        beta = np.full(pres.shape,np.nan)
        # lower troposphere
        lowert = pres >= pres_jump
        beta[lowert] = (alpha+1)/(1 - (1-rh_min/rh_max)*(pres_jump/pres[lowert])**(alpha+1))
        # upper troposphere
        uppert = pres < pres_jump
        beta[uppert] = alpha+1
        
        return beta

    def scatterDensity(ax,x,y,s,alpha):
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        
        return ax.scatter(x,y,c=z,s=s,alpha=0.4)
        
    # def createMask(day):
        
    #     # remove surface peaks
    #     z_peak = rad_features_all[day].z_lw_peak/1e3 # km
    #     remove_sfc = z_peak > 0.5 # km
        
    #     # keep large peaks only
    #     qrad_peak = np.absolute(rad_features_all[day].qrad_lw_peak)
    #     keep_large = np.logical_and(remove_sfc,qrad_peak > 2.5) # K/day
        
    #     # keep soundings in domain of interest
    #     lon_day = data_all.sel(launch_time=day).longitude.data[:,50]
    #     lat_day = data_all.sel(launch_time=day).latitude.data[:,50]
    #     keep_box = np.logical_and(lon_day < lon_box[1], lat_day >= lat_box[0])
        
    #     # merge all
    #     keep = np.logical_and(keep_large,keep_box)
        
    #     return keep

    # def createMaskIntrusions(day):
        
    #     z_peak = rad_features_all[day].z_lw_peak/1e3 # km
        
    #     if day in days_high_peaks:
                
    #         # subset without intrusions
    #         i_d = np.where(np.array(days_high_peaks) == day)[0][0]
    #         k_high = np.logical_and(z_peak <= z_max_all[i_d],
    #                                 z_peak > z_min_all[i_d])
    #         k_low = np.logical_not(k_high)
        
    #         return k_low
        
    #     else:
            
    #         return np.full(z_peak.shape,True)
        


##---- Start Figure


    fig,axs_2D = plt.subplots(ncols=2,nrows=2,figsize=(9,9))
    axs = axs_2D.flatten()
    plt.subplots_adjust(hspace=0.3,wspace=0.3)
    
#-- (a) schematic
    ax = axs[0]
    
    day = '20200126'
    date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
    data_day = data_all.sel(launch_time=day)
    f = rad_features_all[day]
    
    pres_data = data_day.pressure/1e2 # hPa
    pres_fit = np.linspace(0,1000,1001)
    
    # colors
    var_col = f.pw
    norm = matplotlib.colors.Normalize(vmin=var_col.min(), vmax=var_col.max())
    cmap = plt.cm.nipy_spectral
    cmap = plt.cm.RdYlBu
    cols = cmap(norm(var_col))

    for i_s in range(Ns):
        ax.plot(data_day.q_rad_lw[i_s],pres_data[i_s],c=cols[i_s],linewidth=0.5,alpha=0.3)
    
    # idealized beta
    pres_jump = 825
    rh_min = 0.05
    rh_max = 0.8
    alpha_qvstar = 2.3
    piB_star = 0.0054
    delta_nu = 160 # cm-1
    spec_int_approx = piB_star * delta_nu*m_to_cm/e
    beta_id = computeBeta(pres_fit,pres_jump,rh_min,rh_max,alpha_qvstar)
    H_id = -gg/c_pd*(beta_id/(pres_jump*hPa_to_Pa))*spec_int_approx*day_to_seconds
    
    ax.plot(H_id,pres_fit,'k',linewidth=2,linestyle="--")

    
    ax.set_ylim((490,1010))
    ax.invert_yaxis()
    ax.set_ylabel('p (hPa)')
    
    # ax.set_yscale('log')
    ax.set_xlim((-20.1,0.6))
    ax.set_xlabel(r'Full profile $H$(p) (K/day)')
    ax.set_title(r'Cooling profile approximation')
    
    # colorbar density
    axins1 = inset_axes(ax,
                    width="50%",  # width = 70% of parent_bbox width
                    height="2%",  # height : 5%
                    loc='lower left')
    var_col = f.pw
    norm = matplotlib.colors.Normalize(vmin=var_col.min(), vmax=var_col.max())
    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),cax=axins1, orientation='horizontal')
    axins1.xaxis.set_ticks_position("top")
    axins1.tick_params(axis='x', labelsize=9)
    cb.set_label('PW (mm)',labelpad=-35)

    
#-- (b) peak height, using the approximation for the full profile, showing all profiles
    ax = axs[1]
    
    x = []
    y = []
    s = []
    mask_all = {}
    
    for day in days2show:
    # for day in '20200126',:

        # qrad peak height
        pres_qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak
        x.append(pres_qrad_peak)        
        # proxy peak heights
        pres_beta_peak = rad_scaling_all[day].rad_features.beta_peaks.pres_beta_peak
        y.append(pres_beta_peak)
        
        s.append(np.absolute(rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak))
        
        ## mask
        # geometry
        keep = createMask(day)
        # intrusions
        remove_instrusions = createMaskIntrusions(day)
        # merge
        mask_all[day] = np.logical_and(keep,remove_instrusions)

    # peaks
    m = np.hstack([mask_all[day] for day in days2show])
    m_inv = np.logical_not(m)
    x,y,s = np.hstack(x),np.hstack(y),np.hstack(s)

    h = scatterDensity(ax,x,y,s,alpha=0.4)

    # 1:1 line
    ax.plot([910,360],[910,360],'k',linewidth=0.5,alpha=0.5)
    
    # colorbar density
    axins1 = inset_axes(ax,
                    width="50%",  # width = 70% of parent_bbox width
                    height="2%",  # height : 5%
                    loc='lower right')
    cb = fig.colorbar(h, cax=axins1, orientation="horizontal")
    axins1.xaxis.set_ticks_position("top")
    axins1.tick_params(axis='x', labelsize=9)
    cb.set_label('Gaussian kernel \ndensity',labelpad=-45)

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel('Measured cooling peak level (hPa)')
    ax.set_ylabel(r'$\beta$ peak level (hPa)')
    ax.set_title(r'Peak height as $\beta$ maximum')
    # square figure limits
    ylim = ax.get_ylim()
    ax.set_xlim(ylim)

    
#-- (c) peak magnitude, using the simplified scaling (RH step function and 1 wavenumber), showing all profiles
    ax = axs[2]
    
    H_peak_all = {}
    qrad_peak_all = {}
    s = []
    mask_all = {}
    
    for day in days2show:
        
        rs = rad_scaling_all[day]
        f = rad_scaling_all[day].rad_features
        k_bottom = np.where(np.logical_not(np.isnan(f.wp_z[0])))[0][0] 
        
        #- approximated peak
        # H_peak = rad_scaling_all[day].scaling_magnitude_lw_peak*1e8
        Ns = rad_scaling_all[day].rad_features.pw.size
        H_peak = np.full(Ns,np.nan)
        for i_s in range(Ns):
            i_peak = f.i_lw_peak[i_s]
            p_peak = f.pres_lw_peak[i_s]
            
            # CRH above = W(p)/Wsat(p)
            CRHabove = f.wp_z[i_s]/f.wpsat_z[i_s]
            # CRH below = (W_s-W(p))/(Wsat_s-Wsat(p))
            CRHbelow = (f.wp_z[i_s,k_bottom]-f.wp_z[i_s])/(f.wpsat_z[i_s,k_bottom]-f.wpsat_z[i_s])
            # approximation of spectral integral
            piB_star = 0.0054
            delta_nu = 160 # cm-1
            spec_int_approx = piB_star * delta_nu*m_to_cm/e
            # constants
            alpha = alpha_qvstar
            C = -gg/c_pd * (1+alpha)
            
            # without effect of the inversion
            H_peak[i_s] = C/(p_peak*hPa_to_Pa) * CRHbelow[i_peak]/CRHabove[i_peak] * spec_int_approx * day_to_seconds
            
            # # Try including inversion factor
            # deltaT_inv = 6
            # inv_factor = exp(-0.07*deltaT_inv)

            # # with effect of the inversion
            # H_peak[i_s] = C/(p_peak*hPa_to_Pa) * inv_factor * CRHbelow[i_peak]/CRHabove[i_peak] * spec_int_approx * day_to_seconds
                
        
        
        H_peak_all[day] = H_peak
    
        #- true peak
        qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak
        qrad_peak_all[day] = qrad_peak
    
        s.append(0.005*np.absolute(rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak))
        
        ## mask
        # geometry
        keep = createMask(day)
        # intrusions
        remove_instrusions = createMaskIntrusions(day)
        # merge
        mask_all[day] = np.logical_and(keep,remove_instrusions)
    
    # plot
    m = np.hstack([mask_all[day] for day in days2show])
    m_inv = np.logical_not(m)
    x = np.hstack([qrad_peak_all[day] for day in days2show])
    y = np.hstack([H_peak_all[day] for day in days2show])
    s = np.hstack(s)
    
    ax.scatter(x[m_inv],y[m_inv],s[m_inv],'k',alpha=0.1)
    # scatterDensity(ax,x,y,s,alpha=0.5)
    h = scatterDensity(ax,x[m],y[m],s[m],alpha=0.5)

    # # 1:1 line
    # x_ex = np.array([-18,-2])
    # ax.plot(x_ex,x_ex,'k:',alpha=0.4,label='1 : 1')
    
    ax.set_xlabel('Measured cooling peak magnitude (K/day)')
    ax.set_ylabel('Estimate (K/day)')
    ax.set_title(r'Peak magnitude as eq. (13)')
    # square figure limits
    xlim = (-25.4,-1.6)
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    
    #- linear fit
    xmin,xmax = np.min(x), np.max(x) 
    xrange = xmax-xmin
    a_fit , pcov = optimize.curve_fit(lambda x,a:a*x, x[m], y[m],p0=1)
    x_fit = np.linspace(xmin-xrange/40,xmax+xrange/40)
    y_fit = a_fit*x_fit
    # y_fit = 1*x_fit
    cov = np.cov(x[m],y[m])
    r_fit = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    
    # show
    ax.plot(x_fit,y_fit,'k:')

    # # 1:1 line
    # ax.plot([-21,0],[-21,0],'k')

    # write numbers
    t = ax.text(0.05,0.05,'slope = %1.2f \n r = %1.2f'%(a_fit,r_fit),transform=ax.transAxes)
    t.set_bbox(dict(facecolor='w',alpha=0.8,edgecolor='none'))
    
    # colorbar density
    axins1 = inset_axes(ax,
                    width="2%",  # width = 70% of parent_bbox width
                    height="50%",  # height : 5%
                    loc='center left')
    cb = fig.colorbar(h, cax=axins1, orientation="vertical")
    axins1.yaxis.set_ticks_position("right")
    axins1.tick_params(axis='y', labelsize=9)
    cb.set_label('Gaussian kernel density',labelpad=5)
        
 #-- (d) integral of BL cooling, using the simplified scaling (RH step function and 1 wavenumber), showing all profiles
    ax = axs[3]
    
    H_int_all = {}
    qrad_int_all = {}
    s = {}
    mask_all = {}
   
    for day in days2show:
        
        rs = rad_scaling_all[day]
        f = rad_scaling_all[day].rad_features
        k_bottom = np.where(np.logical_not(np.isnan(f.wp_z[0])))[0][0]
        
        #- approximated BL cooling
        Ns = rad_scaling_all[day].rad_features.pw.size
        H_int = np.full(Ns,np.nan)
        qrad_int = np.full(Ns,np.nan)
        
        for i_s in range(Ns):
            i_peak = f.i_lw_peak[i_s]
            p_peak = f.pres_lw_peak[i_s]
            
            #- estimate of BL total cooling
            
            # CRH above = W(p)/Wsat(p)
            CRHabove = f.wp_z[i_s]/f.wpsat_z[i_s]
            # CRH below = (W_s-W(p))/(Wsat_s-Wsat(p))
            CRHbelow = (f.wp_z[i_s,k_bottom]-f.wp_z[i_s])/(f.wpsat_z[i_s,k_bottom]-f.wpsat_z[i_s])
            # approximation of spectral integral
            piB_star = 0.0054
            delta_nu = 160 # cm-1
            spec_int_approx = piB_star * delta_nu*m_to_cm/e
            # constants
            p_surf = 1000 # hPa
            alpha = alpha_qvstar
            C = -gg/c_pd
            
            # without the effect of the inversion
            H_int[i_s] = C/((p_surf-p_peak)*hPa_to_Pa) * \
                            spec_int_approx * day_to_seconds * \
                            np.log( 1 + CRHbelow[i_peak]/CRHabove[i_peak] * ( (p_surf/p_peak)**(1+alpha) - 1))

            # # Try including inversion factor
            # deltaT_inv = 6
            # inv_factor = exp(-0.07*deltaT_inv)
            # # with the effect of the inversion
            # H_int[i_s] = C/((p_surf-p_peak)*hPa_to_Pa) * \
            #                 spec_int_approx * day_to_seconds * \
            #                 np.log( 1 + inv_factor * CRHbelow[i_peak]/CRHabove[i_peak] * ( (p_surf/p_peak)**(1+alpha) - 1))
    
            #- true total BL cooling
            
            s_notnans = slice(12,None)
            qrad_lw_not_nans = np.flipud(rad_scaling_all[day].rad_features.qrad_lw_smooth[i_s][s_notnans])
            pres_not_nans = np.flipud(f.pres[s_notnans])
            
            qrad_int[i_s] = f.mo.pressureIntegral(arr=qrad_lw_not_nans,pres=pres_not_nans,p0=p_peak,p1=p_surf,z_axis=0) / \
                                    f.mo.pressureIntegral(arr=np.ones(qrad_lw_not_nans.shape),pres=pres_not_nans,p0=p_peak,p1=p_surf,z_axis=0)

        H_int_all[day] = H_int
        qrad_int_all[day] = qrad_int
        s[day] = 0.005*np.absolute(rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak)
            
        ## mask
        # geometry
        keep = createMask(day)
        # intrusions
        remove_instrusions = createMaskIntrusions(day)
        # merge
        mask_all[day] = np.logical_and(keep,remove_instrusions)
        
    # plot
    m = np.hstack([mask_all[day] for day in days2show])
    m_inv = np.logical_not(m)
    x = np.hstack([qrad_int_all[day] for day in days2show])
    y = np.hstack([H_int_all[day] for day in days2show])
    s = np.hstack([s[day] for day in days2show])
    
    ax.scatter(x[m_inv],y[m_inv],s[m_inv],'k',alpha=0.1)
    # scatterDensity(ax,x,y,s,alpha=0.5)
    h = scatterDensity(ax,x[m],y[m],s[m],alpha=0.5)
    
    #- linear fit
    a_fit , pcov = optimize.curve_fit(lambda x,a:a*x, x[m], y[m],p0=1)
    x_fit = np.linspace(-5,0)
    y_fit = a_fit*x_fit
    # y_fit = 1*x_fit
    cov = np.cov(x[m],y[m])
    r_fit = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    
    # show
    ax.plot(x_fit,y_fit,'k:')
    
    # # 1:1 line
    # ax.plot([-6,0],[-6,0],'k')
    
    # ax.set_xlabel('True LW $Q_{rad}$ (K/day)')
    # ax.set_ylabel('Estimate (K/day)')
    # ax.set_title(r'Magnitude as $-\frac{g}{c_p} \frac{1}{p%s} (1+\alpha) \frac{CRH_s}{CRH_t} B_{\nu^\star} \frac{\Delta \nu}{e}$'%label_jump)
    # square figure limits
    xlim = (-5.1,-1.4)
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    
    # write numbers
    t = ax.text(0.05,0.05,'slope = %1.2f \n r = %1.2f'%(a_fit,r_fit),transform=ax.transAxes)
    t.set_bbox(dict(facecolor='w',alpha=0.8,edgecolor='none'))
    
    # colorbar density
    axins1 = inset_axes(ax,
                    width="50%",  # width = 70% of parent_bbox width
                    height="2%",  # height : 5%
                    loc='lower right')
    cb = fig.colorbar(h, cax=axins1, orientation="horizontal")
    axins1.xaxis.set_ticks_position("top")
    axins1.tick_params(axis='x', labelsize=9)
    cb.set_label('Gaussian kernel density',labelpad=-37)
    
    # labels
    ax.set_xlabel(r'Measured integral cooling (K/day)')
    ax.set_ylabel(r'Estimate (K/day)')
    ax.set_title(r'Average cooling as eq. (14)')
    
    
    #--- Add panel labeling
    pan_labs = '(a)','(b)','(c)','(d)'
    for ax,pan_lab in zip(axs,pan_labs):
        t = ax.text(0.03,0.92,pan_lab,transform=ax.transAxes,fontsize=14)
        t.set_bbox(dict(facecolor='w',alpha=0.8,edgecolor='none'))
    
    #--- save
    plt.savefig(os.path.join(figdir,'Figure%d.pdf'%i_fig),bbox_inches='tight')    
    
