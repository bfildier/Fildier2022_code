#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 11:22:06 2021

@author: bfildier
"""


import numpy as np
import xarray as xr
from math import pi,e

# own modules
from matrixoperators import *


class RadiativeScaling():
    """"Stores each step in the calculation of the radiative scaling, for each profile 
    (for a given day, or whichever set of profiles provided as input)."""
    
    # Planck's constant
    h = 6.626e-34 # J.s
    # Stefan-Boltzmann's constant
    kB = 1.381e-23 # J.K-1
    # speed of light
    c = 2.998e8 # m.s-1
    # gravity
    g = 9.81 # m2.s-1
    # specifit heat of air
    c_p = 1000 # J.K-1.kg-1
    
    # radiative fit parameters - rotational band
    k_rot = 131 # m2/kg
    nu_rot = 200 # cm-1
    l_rot = 59.2 # cm-1
    # radiative fit parameters - v-r band
    k_vr = 4.6 # m2/kg
    nu_vr = 1450 # cm-1
    l_vr = 46 # cm-1
    
    ##-- Constructors
    
    def __init__(self,pres,temp,qv,rad_features):
        """
        Arguments:
        - pres: in Pa, Ns x Nz matrix
        - temp: in K, Ns x Nz matrix
        - qv: in kg/kg, Ns x Nz matrix
        - rad_features: a Features object, computed with the same dataset
        """
        
        self.pres = pres # Pa
        self.temp = temp # K
        self.qv = qv # kg/kg
        self.rad_features = rad_features
        self.mo = MatrixOperators()
        
    def __str__(self):
        """Override string function to print attributes
        """
        str_out = ''
        for a in dir(self):
            if '__' not in a:
                a_str = str(getattr(self,a))
                if 'array' in str(getattr(self,a).__class__):
                    str_out = str_out+("%s : %s\n"%(a,str(getattr(self,a).__class__)))
                elif 'method' not in a_str:
                    str_out = str_out+("%s = %s\n"%(a,a_str))
                
        return str_out

    ##-- Methods

    def planck(self,nu,temp):
        """Planck's function
        
        Arguments:
            - nu: wavenumber in m-1
            - temp: temperature in K
            
        Output in J.s-1.sr-1.m-2.m
        """
        
        planck = 2*self.h*nu**3*self.c**2/(np.exp(self.h*self.c*nu/self.kB/temp)-1) # J.s-1.sr-1.m-2.m
        
        return planck
    
    def kappa(self,nu,band='rot'):
        """Extinction coefficient kappa from wavenumber nu, 
        analytical expression from Jeevanjee&Fueglistaler (2020b)
        
        Arguments:
            - nu wavenumber in m-1
        Returns:
            - kappa=f(nu): extinction coefficient in m2/kg = mm-1"""
        
        # rotational band (JF2020)
        # k_rot = 127 # m2/kg
        # nu_rot = 150 # cm-1
        # l_rot = 56 # cm-1
        
        # k_rot = 135 # m2/kg
        # nu_rot = 200 # cm-1
        # l_rot = 60.1 # cm-1
                
        kappa_rot = self.k_rot*np.exp(-(nu/1e2-self.nu_rot)/self.l_rot) # m2/kg, or mm-1
        
        # # vr band (JF2020)
        # k_vr = 3.8 # m2/kg
        # nu_vr = 1450 # cm-1
        # l_vr = 40 # cm-1
        
        # k_vr = 4.8 # m2/kg
        # nu_vr = 1450 # cm-1
        # l_vr = 47 # cm-1
        
        kappa_vr = self.k_vr*np.exp(-(self.nu_vr-nu/1e2)/self.l_vr) # m2/kg, or mm-1
        
        if band == 'rot':
            return kappa_rot
        elif band == 'vr':
            return kappa_vr
    
    def nu(self,kappa,band='rot'):
        """Wavenumber as a function of extinction coefficient kappa,
        analytical expression from Jeevanjee&Fueglistaler (2020b)
        
        Arguments:
            - kappa extinction coefficient in m2/kg = mm-1
        returns:
            - nu=f(kappa): wavenumber in m-1"""
        
        # rotational band        
        if band == 'rot':
            
            # (JF2020)
            # k_rot = 127 # m2/kg
            # nu_rot = 150 # cm-1
            # l_rot = 56 # cm-1
            
            # k_rot = 135 # m2/kg
            # nu_rot = 200 # cm-1
            # l_rot = 60.1 # cm-1
    
            return 1e2*(self.nu_rot+self.l_rot*np.log(self.k_rot/kappa)) # m-1
        
        # vr band
        elif band == 'vr':
            
            # (JF2020)
            # k_vr = 3.8 # m2/kg
            # nu_vr = 1450 # cm-1
            # l_vr = 40 # cm-1
            
            # k_vr = 4.8 # m2/kg
            # nu_vr = 1450 # cm-1
            # l_vr = 47 # cm-1
            
            return 1e2*(self.nu_vr-self.l_vr*np.log(self.k_vr/kappa)) # m-1
    
    def phi(self,kappa,W):
        """Weighting function"""
        
        return kappa * W * np.exp(-kappa*W)
 
    
    #-- terms for all scalings
    
    def computeBeta(self):
        """Compute beta = d(ln W)/d(ln p) and store intermediate values."""
        
        self.ln_W = np.log(self.rad_features.wp_z)
        self.ln_p = np.log(self.pres) # Pa
        self.ln_p_mean = np.nanmean(self.ln_p,axis=0)
        self.beta,_ = self.mo.deriv(self.ln_W,self.ln_p_mean,axis=1)
        self.beta_over_p = self.beta/self.pres # Pa-1
    
    def computeSpectralIntegral(self,band='rot'):
        """Spectral integral \int pi B(T(p)) phi(tau(p)) d\nu"""

        # wavenumber structure
        if band == 'rot':
            self.nu_array = np.linspace(20000,100000) # m-1
        elif band == 'vr':
            self.nu_array = np.linspace(100000,150000) # m-1
        self.dnu_array = np.diff(self.nu_array) # m-1
        N_nu = len(self.nu_array)
        N_s = self.pres.shape[0]
        N_z = self.pres.shape[1]
        
        # stores int_nu(pi*B*phi) in dimensions (N_s,N_z)
        spectral_integral = np.full((N_s,N_z),np.nan)
        
        for i_s in range(N_s):
            
            # stores pi*B*phi*d_nu in (N_nu,N_s) at i_s, before integration
            integrand_nu = np.full((N_nu-1,N_z),np.nan)
            
            for i_nu in range(N_nu-1):
                
                nu_inv_m = self.nu_array[i_nu]
                # Planck
                B_nu = self.planck(nu_inv_m,self.temp[i_s]) # W.sr-1.m-2.m
                # phi
                W_s = self.rad_features.wp_z[i_s] # mm
                kappa_nu = self.kappa(nu_inv_m,band=band) # m2/kg, or mm-1
                phi_nu = self.phi(kappa_nu,W_s)
                # product
                integrand_nu[i_nu] = pi*B_nu*phi_nu*self.dnu_array[i_nu]
                # print(integrand_nu[i_nu])
            
            spectral_integral[i_s] = np.nansum(integrand_nu,axis=0)
    
        # save as attribute
        setattr(self,'spectral_integral_%s'%band,spectral_integral)
    
    #-- scaling for radiative profile
    
    def computeScalingProfileWithIntegral(self):
        """Analytical scaling including the spectral integral:
            H(p) ≈ - g/c_p * beta(p)/p * \int_\nu \pi B_\nu(T(p)) \phi_nu(p) d\nu
            
        Stores the result."""
        
        day_to_seconds = 86400
        
        # rotational band only
        self.scaling_profile_rot = -self.g/self.c_p*(self.beta/self.pres)*self.spectral_integral_rot*day_to_seconds
        # vr band only
        self.scaling_profile_vr = -self.g/self.c_p*(self.beta/self.pres)*self.spectral_integral_vr*day_to_seconds
        # both band
        self.scaling_profile = self.scaling_profile_rot + self.scaling_profile_vr

    #-- scaling for radiative peak height
    
    def computeBetaPeaks(self,n_smooth = 15):
        """Calculate peak beta, reusing rad_features code."""

        if not hasattr(self,'beta'):
            self.computeBeta()

        # compute beta peaks
        self.rad_features.computePeaks(self.beta,n_smooth_0=n_smooth,below_p=350,
                                       f_peak='nanargmax',varid='beta')
        self.rad_features.computePeaks(self.beta_over_p,n_smooth_0=n_smooth,below_p=350,
                                       f_peak='nanargmax',varid='beta_over_p')
        self.rad_features.computePeaks(self.scaling_profile,n_smooth_0=n_smooth,
                                       f_peak='nanargmin',varid='scaling_profile')

    ##-- Just if I want to be fancy, implement the analytic criterion
    #
    # def scalingPeakHeight(self):
    #     """Approximation for the height of radiative cooling peak, with the criterion
    #     1/q^2 dq/dp = 1/(gW)"""
        
    #     # smoothing filter for qv
        
    #     # left-hand side

    
    #-- scaling for radiative peak magnitude
    
    def computeScalingPeakMagnitude(self,B_star=None):
        """Analytical aprpoximation of Qrad peak magnitude, using reference wavenumber nu_star:
            H^\star = -g\c_p * beta^\star/p^\star * \pi B_\nu(T^\star) * \Delta \nu / e
        Stores the result.
        
        Arguments:
            - B_star: prescribed Planck function, in J.s-1.sr-1.m-2.m (default: None)
        """

        # delta_nu = 73.1 # cm-1
        delta_nu = 160 # cm-1
        m_to_cm = 1e2
        day_to_seconds = 86400

        N_s = self.pres.shape[0]
        
        # first save B_star argument value
        B_star_prescribed = B_star

        which_peak = 'beta_peak', 'lw_peak'
        if B_star_prescribed is None:
            suffix = ''
        else:
            suffix = '_B'+''.join(("%2.2f"%(B_star_prescribed*100)).split("."))
        terms = "scaling_magnitude","nu_star","kappa_star","B_star"

        # init
        for peak_name in which_peak:
            for prefix in terms:

                if prefix not in ['nu_star','kappa_star'] or B_star_prescribed is None:
                    setattr(self,"%s_%s%s"%(prefix,peak_name,suffix),np.full(N_s,np.nan))

        # compute for each profile
        for i_s in range(N_s):
            
            def computeHpeak(beta_star,i_star,B_star_prescribed=None):
                """B_star in SI units: J.s-1.sr-1.m-2.m. Good reference value
                is 0.004 (max at 300K is about 0.0045)"""
                
                p_star = self.pres[i_s,i_star]
                
                if B_star_prescribed is None: # Careful to go meet this condition not only once, by not reinitializing B_star from its value given in argument
                    temp_star = self.temp[i_s,i_star]
                    W_star = self.rad_features.wp_z[i_s,i_star]
                    kappa_star = 1/W_star
                    nu_star = self.nu(kappa_star)
                    B_star = self.planck(nu_star, temp_star)
                else:
                    B_star = B_star_prescribed
                    nu_star = np.nan 
                    kappa_star = np.nan
                
                sc_magnitude = - self.g/self.c_p * beta_star/p_star * pi * B_star*m_to_cm * delta_nu/e * day_to_seconds
                
                return sc_magnitude, nu_star, kappa_star, B_star
            
            for peak_name in which_peak:
                
                if peak_name == 'beta_peak':
                    #- scaled magnitude at beta peak
                    beta_star = self.rad_features.beta_peak[i_s]
                    i_star = self.rad_features.i_beta_peak[i_s]

                elif peak_name == 'lw_peak':
                    #- scaled magnitude at lw peak
                    i_star = self.rad_features.lw_peaks.i_lw_peak[i_s]
                    beta_star = self.rad_features.beta_smooth[i_s,i_star]

                # compute
                scaling_magnitude, nu_star, kappa_star, B_star = computeHpeak(beta_star,i_star,B_star_prescribed)
                # save
                for prefix in terms:
                    
                    if 'nu' in prefix or 'kappa' in prefix:
                        if np.isnan(nu_star) or np.isnan(kappa_star): # B_star prescribed
                            continue
                    
                    var = locals()[prefix]
                    obj = getattr(self,"%s_%s%s"%(prefix,peak_name,suffix))
                    obj[i_s] = var
                    
            # self.scaling_magnitude_beta_peak[i_s] = scaling_magnitude
            # self.nu_star_beta_peak[i_s] = nu_star
            # self.kappa_star_beta_peak[i_s] = kappa_star
            # self.B_star_beta_peak[i_s] = B_star
            

            
    #-- Wrap up all in one method
    
    def computeScaling(self):
        """Compute radiative scaling for profile and peak"""
        
        # prerequisites
        self.computeBeta()
        self.computeSpectralIntegral(band='rot')
        self.computeSpectralIntegral(band='vr')
        # profile
        self.computeScalingProfileWithIntegral()
        # height
        self.computeBetaPeaks()
        # magnitude
        self.computeScalingPeakMagnitude()
        self.computeScalingPeakMagnitude(B_star = 0.0054) # fix B_star
        
        
        





