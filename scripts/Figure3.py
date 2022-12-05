#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 3

Created on Mon Jun 20 09:33:29 2022

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
    
#%% Draw Figure 3

    i_fig = 3

    m_to_cm = 1e2
        
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
    
    hide_id_prof = False
    
    day = '20200126'
    date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
    data_day = data_all.sel(launch_time=day)
    f = rad_features_all[day]
    
    pres_data = data_day.pressure/1e2 # hPa
    pres_fit = np.linspace(0,1000,1001)
    
    def updateBounds(ax,x_left,x_right,y_bot,y_top):
        """Save boundaries for legend"""
        
        x,y,w,h = ax.get_position().bounds
        x_left = np.nanmin(np.array([x,x_left]))
        x_right = np.nanmax(np.array([x+w,x_right]))
        y_bot = np.nanmin(np.array([y,y_bot]))
        y_top = np.nanmax(np.array([y+h,y_top]))
        
        return x_left,x_right,y_bot,y_top
    
    x_left = np.nan
    x_right = np.nan
    y_bot = np.nan
    y_top = np.nan
    
    #-- start figure
    
    fig,axs = plt.subplots(ncols=3,nrows=3,figsize=(10,12))
    
    Ns = data_day.dims['launch_time']
    
    # colors
    var_col = f.pw
    norm = matplotlib.colors.Normalize(vmin=var_col.min(), vmax=var_col.max())
    cmap = plt.cm.nipy_spectral
    cmap = plt.cm.RdYlBu
    cols = cmap(norm(var_col))
    
    #---- RH
    ax = axs[0,0]
    
    for i_s in range(Ns):
        ax.plot(data_day.relative_humidity[i_s]*100,pres_data[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
    
    ax.set_xlim((-3,103))
    ax.set_xlabel(r'Relative humidity $\varphi$ (%)')
    
    if not hide_id_prof:
        
        # stepfunction RH
        pres_jump = 825
        rh_min = 0.05
        rh_max = 0.8
        rh_step = piecewise_linear(pres_fit,[pres_jump,pres_jump],[rh_min,rh_max])
        ax.plot(rh_step*100,pres_fit,'k',linewidth=2,linestyle="--")
    
    x_left,x_right,y_bot,y_top = updateBounds(ax,x_left,x_right,y_bot,y_top)
    
    #---- qvstar
    ax = axs[0,1]
    for i_s in range(Ns):
        ax.plot(data_day.specific_humidity[i_s]/data_day.relative_humidity[i_s],pres_data[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
    
    # ax.set_xlim((270,302))
    ax.set_xlabel(r'Saturated specific hymidity $q_v^\star$ (kg/kg)')
    
    if not hide_id_prof:
        
        # power qvstar
        # alpha_qvstar = 2.5
        alpha_qvstar = 2.3
        qvstar_0 = 0.02
        qvstar_power = qvstar_0 * np.power(pres_fit/pres_fit[-1],alpha_qvstar)
        ax.plot(qvstar_power,pres_fit,'k',linewidth=2,linestyle="--")
    
    
    x_left,x_right,y_bot,y_top = updateBounds(ax,x_left,x_right,y_bot,y_top)
    
    #---- W
    ax = axs[0,2]
    for i_s in range(Ns):
        # ax.plot(f.wp_z[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
        ax.plot(rad_scaling_all[day].rad_features.wp_z[i_s],pres_data[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
    
    ax.set_xscale('log')
    ax.set_xlim((3e-1,100))
    # ax.set_xlabel(r'$W(p) \equiv \int_z^{TOA} q_v \frac{dp}{g}$ (mm) $\propto \tau$',labelpad=-1)
    ax.set_xlabel(r'Water path from top W (mm)',labelpad=-1)
    
    if not hide_id_prof:
        
        # idealized water path
        W_fit = waterPath(qvstar_0,pres_fit,pres_jump,rh_min,rh_max,alpha_qvstar)
        ax.plot(W_fit,pres_fit,'k',linewidth=2,linestyle="--")
    
    x_left,x_right,y_bot,y_top = updateBounds(ax,x_left,x_right,y_bot,y_top)
    
    #---- B_nu(T(p))
    ax = axs[1,0]
    
    nu_star_rot = 554 # cm-1
    nu_star_rot_m_m1 = nu_star_rot*1e2 # m-1
    B_rot = rad_scaling_all[day].planck(nu_star_rot_m_m1,data_day.temperature)*1e2 # cm-1
    nu_star_vr = 1329 # cm-1
    nu_star_vr_m_m1 = nu_star_vr*1e2 # m-1
    B_vr = rad_scaling_all[day].planck(nu_star_vr_m_m1,data_day.temperature)*1e2 # cm-1
    
    # B = B_rot + B_vr
    B = B_rot
    
    for i_s in range(Ns):
        ax.plot(pi*B[i_s],pres_data[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
    
    ax.set_xlim((0.33,0.47))
    # ax.set_xlim((0.33,0.62))
    ax.set_xlabel(r'Planck $\pi B(\tilde{\nu}^+_1,T)$ (W.m$^{-2}$.cm)')
    # ax.set_xlabel(r'Planck $\pi \tilde{B}(T)$ (W.m$^{-2}$.cm)')
    
    x_left,x_right,y_bot,y_top = updateBounds(ax,x_left,x_right,y_bot,y_top)
    
    #---- phi_nu_star = tau*e(-tau)
    ax = axs[1,1]
    
    W_star = 3 # mm
    # kappa_star = 1/W_star # m2/kg, or mm-1
    kappa_star = 0.3
    tau_star = kappa_star*f.wp_z 
    phi = tau_star*np.exp(-tau_star)
    
    for i_s in range(Ns):
        ax.plot(phi[i_s],pres_data[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
    
    # ax.set_yscale('log')
    # ax.set_xlim((-0.051,0.001))
    ax.set_xlabel(r'Weighting function $\phi(\tilde{\nu}^+_1,W(p))$')
    
    x_left,x_right,y_bot,y_top = updateBounds(ax,x_left,x_right,y_bot,y_top)
    
    #---- integral
    ax = axs[1,2]
    
    #-- first, compute profiles    
    nu_array_inv_m = np.linspace(20000,90000) # m-1
    dnu_array = np.diff(nu_array_inv_m) # m-1
    N_nu = len(nu_array_inv_m)
    N_s = B.shape[0]
    N_z = B.shape[1]
    
    rs = rad_scaling_all[day]
    integral = rs.spectral_integral_rot + rs.spectral_integral_vr
    
    x_left,x_right,y_bot,y_top = updateBounds(ax,x_left,x_right,y_bot,y_top)
    
    #-- show
    for i_s in range(Ns):
        ax.plot(integral[i_s],pres_data[i_s],c=cols[i_s],linewidth=0.5,alpha=0.3)
    
    ax.set_xlabel(r'$\int \pi B(\tilde{\nu},T)\phi(\tilde{\nu},W) d{\tilde{\nu}}$ (W.m$^2$)')
    
    if not hide_id_prof:
        
        # estimate pi B delta nu / e
        piB_star = 0.0054
        delta_nu = 160 # cm-1
        int_id = piB_star * delta_nu*m_to_cm / e
        ax.axvline(x=int_id,c='k',linewidth=2,linestyle="--")
    
    x_left,x_right,y_bot,y_top = updateBounds(ax,x_left,x_right,y_bot,y_top)
    
    #---- beta/p
    ax = axs[2,0]
    display_factor = 1e2
    
    for i_s in range(Ns):
        ax.plot(display_factor*(rad_scaling_all[day].beta/pres_data)[i_s],pres_data[i_s],c=cols[i_s],linewidth=0.5,alpha=0.3)
    
    # ax.set_yscale('log')
    ax.set_xlim((-0.1,5.1))
    ax.set_xlabel(r'$\beta/p$ ($\times 10^{4}$ Pa$^{-1}$)')
    
    if not hide_id_prof:
        
        # idealized beta
        beta_id = computeBeta(pres_fit,pres_jump,rh_min,rh_max,alpha_qvstar)
        ax.plot(display_factor*beta_id/pres_fit,pres_fit,'k',linewidth=2,linestyle="--")
    
    x_left,x_right,y_bot,y_top = updateBounds(ax,x_left,x_right,y_bot,y_top)
    
    #---- full H estimate
    ax = axs[2,1]
    
    g = 9.81 # m/s
    c_p = 1000 # J/kg
    day_to_seconds = 86400
    H_est = -g/c_p*(rad_scaling_all[day].beta/pres_data/100)*integral*day_to_seconds
    
    for i_s in range(Ns):
        ax.plot(H_est[i_s],pres_data[i_s],c=cols[i_s],linewidth=0.5,alpha=0.3)
    
    ax.set_xlim((-15.1,0.6))
    ax.set_xlabel(r'$-\frac{g}{c_p} \frac{\beta}{p} \int \pi B_{\tilde{\nu}}(T)\phi_{\tilde{\nu}} d{\tilde{\nu}}$ (K/day)')
    
    x_left,x_right,y_bot,y_top = updateBounds(ax,x_left,x_right,y_bot,y_top)
    
    #---- actual H
    ax = axs[2,2]
    
    for i_s in range(Ns):
        ax.plot(data_day.q_rad_lw[i_s],pres_data[i_s],c=cols[i_s],linewidth=0.5,alpha=0.3)
    
    # ax.set_yscale('log')
    ax.set_xlim((-15.1,0.6))
    ax.set_xlabel(r'$H$ (K/day)')
    
    x_left,x_right,y_bot,y_top = updateBounds(ax,x_left,x_right,y_bot,y_top)
    
    #- all y axes
    for ax in axs.flatten():
        ax.set_ylim((490,1010))
        ax.invert_yaxis()
    for ax in axs[:,0]:
        ax.set_ylabel('p (hPa)')
        
    fs = 14
    #- label hypotheses
    axs[1,0].text(0.5, 0.05, r'H$_1$', horizontalalignment='center',
         verticalalignment='center', transform=axs[1,0].transAxes,fontsize=fs)
    axs[1,1].text(0.5, 0.05, r'H$_2$', horizontalalignment='center',
         verticalalignment='center', transform=axs[1,1].transAxes,fontsize=fs)
    axs[1,2].text(0.5, 0.05, r'H$_1$ & H$_2$', horizontalalignment='right',
         verticalalignment='center', transform=axs[1,2].transAxes,fontsize=fs)
    axs[2,0].text(0.5, 0.05, r'H$_3$', horizontalalignment='center',
         verticalalignment='center', transform=axs[2,0].transAxes,fontsize=fs)
    #- label appproximation and target
    axs[2,1].text(0.5, 0.05, r'(approximation)', horizontalalignment='center',
         verticalalignment='center', transform=axs[2,1].transAxes,fontsize=fs-2)
    axs[2,2].text(0.5, 0.05, r'(target)', horizontalalignment='center',
         verticalalignment='center', transform=axs[2,2].transAxes,fontsize=fs-2)
    
    #- Color bar
    dy = (y_top-y_bot)/80
    cax = plt.axes([x_left,y_bot-8*dy,x_right-x_left,dy])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax, orientation='horizontal')
    cbar.set_label('PW (mm)',fontsize=fs)
    
    #--- Add panel labeling
    pan_labs = ["(%s)"%chr(i) for i in range(ord('a'), ord('i')+1)]
    for ax,pan_lab in zip(axs.flatten(),pan_labs):
        t = ax.text(0.04,0.91,pan_lab,transform=ax.transAxes,fontsize=14)
        t.set_bbox(dict(facecolor='w',alpha=0.8,edgecolor='none'))
    
    #-- Save
    # plt.savefig(os.path.join(figdir,'FigureS%d%s.pdf'%(i_fig,'_noidprof'*hide_id_prof)),bbox_inches='tight')
    # plt.savefig(os.path.join(figdir,'FigureS%d%s.png'%(i_fig,'_noidprof'*hide_id_prof)),dpi=300,bbox_inches='tight')
    plt.savefig(os.path.join(figdir,'FigureS%d%s.jpg'%(i_fig,'_noidprof'*hide_id_prof)),dpi=300,bbox_inches='tight')    






