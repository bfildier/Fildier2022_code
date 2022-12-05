
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Features of radiative profiles

@author: bfildier
"""

import numpy as np
import xarray as xr
import os

# workdir = os.path.dirname(os.path.realpath(__file__))
# moduledir = os.path.join()
# sys.path.insert(0,moduledir)
from matrixoperators import *
import thermoFunctions as tf


class Features():
    """Finds and stores characteristics of the peak radiative cooling"""
    
    def __init__(self,pres,z,launch_time=None,dz_smooth=0.15):
        """Class constructor
        
        Arguments:
            - pres: pressure in hPa
            - z: height in km
            - dz_smooth: filter width (default, 0.150 km)"""

        self.pres = pres        
        self.z = z
        self.launch_time = launch_time
        self.dz_smooth = dz_smooth
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

    # def computePeaks(self,array,z,which='net',varid='qrad'):
    #     """Copy computeQradPeaks"""
        
    #     self.computeQradPeaks(array,z,which='net',varid='qrad')

    def computePeaks(self,array,n_smooth_0=15,below_p=100,f_peak='nanargmin',which='net',varid='qrad'):
        """Arguments:
            - array: Ns x Nz matrix
            - n_smooth_0: number of levels to apply filter (x 10hPa)
            - below_p: level below which to look for peak (in hPa)
            - f_peak: numpy method to find maximum or minimum
        
        Stores:
            - array values smoothed in the vertical
            - peak height
            - peak pressure
            - index of peak height
            - peak magnitude"""

        if varid == 'qrad':
            lab_long = '%s_%s'%(varid,which)
            lab_short = which
        elif varid in ['beta','beta_over_p','scaling_profile']:
            lab_long = lab_short = varid
        
        if hasattr(self,'%s_peak'%(lab_long)):
            print("pass: '%s_peak is already computed"%(lab_long))
            pass
        
        Ns,Nz = array.shape
        # dz = np.nanmean(np.diff(self.z))
        # n_smooth = self.dz_smooth/dz
        
        # define
        setattr(self,'i_%s_peak'%lab_short,np.nan*np.zeros((Ns,),dtype=int))
        setattr(self,'z_%s_peak'%lab_short,np.nan*np.zeros((Ns,)))
        setattr(self,'pres_%s_peak'%lab_short,np.nan*np.zeros((Ns,)))
        setattr(self,"%s_peak"%(lab_long),np.nan*np.zeros((Ns,)))
        setattr(self,"%s_smooth"%(lab_long),np.nan*np.zeros((Ns,Nz)))
        
        for i_s in range(Ns):
            
            data_i = array[i_s]
            if np.all(np.isnan(data_i)): # if profile is nan, go to next
                continue
                
            i, var_i, var_s = self.findPeak(data_i,n_smooth_0=n_smooth_0,below_p=below_p,f_peak=f_peak,return_all=True)
            
            getattr(self,'i_%s_peak'%lab_short)[i_s] = i
            getattr(self,'z_%s_peak'%lab_short)[i_s] = self.z[i]
            getattr(self,'pres_%s_peak'%lab_short)[i_s] = self.pres[i]
            getattr(self,'%s_peak'%lab_long)[i_s] = var_i
            getattr(self,'%s_smooth'%lab_long)[i_s,:] = var_s
        
        # convert to int (needed again, for some reason..)
        setattr(self,'i_%s_peak'%lab_short,np.array(getattr(self,'i_%s_peak'%lab_short),dtype=int))
        
        setattr(self,'%s_peaks'%lab_short,xr.Dataset({"z":(["zlay"],self.z),\
                                 "i_%s_peak"%lab_short:(["PW"],getattr(self,'i_%s_peak'%lab_short)),\
                                 "z_%s_peak"%lab_short:(["PW"],getattr(self,'z_%s_peak'%lab_short)),\
                                 "pres_%s_peak"%lab_short:(["PW"],getattr(self,'pres_%s_peak'%lab_short)),\
                                 "%s_peak"%lab_long:(["PW"],getattr(self,'%s_peak'%lab_long)),\
                                 "%s_smooth"%lab_long:(["PW","z"],getattr(self,'%s_smooth'%lab_long))}))

    def findPeak(self,values,n_smooth_0=15,below_p=100,f_peak='nanargmin',return_all=False):
        """Returns index and value of radiative cooling peak.
        
        Arguments:
        - values: 1D numpy array
        - n_smooth: width of smoothing window (number of points)
        - below_p: pressure level below which to look for peak, in hPa
        - f_peak: numpy method to determine peak (default is 'nanargmin')
        - return_all: boolean
        
        Returns:
        - """
        
        # apply moving window to data, with size n_smooth_0
        val_smooth_0 = np.convolve(values,np.repeat([1/n_smooth_0],n_smooth_0),
                                 'same')
        # find levels to discard
        mask_above_p = self.pres < below_p
        # replace above reference p level with nans
        val_smooth_0[mask_above_p] = np.nan
        # find index of peak
        ind = getattr(np,f_peak)(val_smooth_0)
        
        # returns
        if return_all:
            return ind, val_smooth_0[ind], val_smooth_0
        else:
            return ind, val_smooth_0[ind]
        
    def computePW(self,qv,temp,pres,z,i_z_max=-1,attr_name='pw'):
        """Compute and store precipitable water
        
        Arguments:
        - qv,temp,pres: 2D numpy arrays, p in hPa
        - z: 1D numpy array"""

        hPa_to_Pa = 1e2
        
        # density
        R = 287 # J/kg/K
        rho = pres*hPa_to_Pa/(R*temp)
        self.rho = rho
        # dz
        dz = np.diff(np.convolve(z,[0.5,0.5],mode='valid'))
        dz = np.append(np.append([dz[0]],dz),[dz[-1]])
        dz_3D = np.repeat(dz[np.newaxis,:],temp.shape[0],axis=0)
        # PW
        pw_layers = qv*rho*dz_3D
        
        if i_z_max.__class__ is int:
            setattr(self,attr_name,np.nansum(pw_layers[:,:i_z_max],axis=1))
        elif i_z_max.__class__ is np.ndarray:
            # truncate at height
            n_pw = pw_layers.shape[0]
            pw = np.nan*np.zeros((n_pw,))
            for i_pw in range(n_pw):
                i_z = i_z_max[i_pw]
                pw[i_pw] = np.nansum(pw_layers[i_pw,:i_z])
            setattr(self,attr_name,pw)
            
    def computeWPaboveZ(self,qv,pres,z_axis=1):
        """Calculates the integrated water path above each level.
        
        Arguments:
            - qv: specific humidity in kg/kg, Ns x Nz matrix
            - pres: pressure coordinate in hPa, Nz vector
            
        Stores:
            - self.wp_z: water path above each level, Ns x Nz matrix"""

        Ns,Np = qv.shape
        self.wp_z = np.full((Ns,Np),np.nan)
        
        
        # for i_z in range(Nz-2):

        #     # test order
        #     if np.diff(pres)[0] > 0: # then p (z) is in increasing (decreasing) order
        #         slice_z = slice(None,i_z)
        #         i_0 = 0
        #         i_1 = i_z
        #     else:
        #         slice_z = slice(i_z,None)
        #         i_0 = i_z
        #         i_1 = -1
            
        #     # compute integral
        #     # print(pres[slice_z].shape, qv[:,slice_z].shape)
        #     self.wp_z[:,i_z] = self.mo.pressureIntegral(arr=qv,pres=pres,p0=pres[i_0],p1=pres[i_1],z_axis=z_axis)
        
        ##-- new version below, mo.pressureIntegral only works if pressure is increasing with index
        
        p_increasing = np.diff(pres)[0] > 0
    
        for i_p in range(Np-2):
            # self.wp_z[:,i_z] = self.mo.pressureIntegral(arr=data.specific_humidity[:,i_z:],pres=pres[i_z:],p_levmin=pres[i_z],p_levmax=pres[-1],z_axis=z_axis)
        
            if p_increasing:
                
                arr = qv
                p = pres
                p0 = p[0]
                p1 = p[i_p]
                i_w = i_p

            else:

                arr = np.flip(qv,axis=z_axis)
                p = np.flip(pres)
                p0 = p[0]
                p1 = p[i_p]
                i_w = Np-1-i_p
                
            self.wp_z[:,i_w] = self.mo.pressureIntegral(arr=arr,pres=p,p0=p0,p1=p1,z_axis=z_axis)

    def computeSaturatedWPaboveZ(self,temp,pres,z_axis=1):
        """Calculates the integrated water path of saturation specific humidity above each level.
        
        Arguments:
            - qv: specific humidity in kg/kg, Ns x Nz matrix
            - pres: pressure coordinate in hPa, Nz vector
            
        Stores:
            - self.wpsat_z: water path above each level, Ns x Nz matrix"""

        hPa_to_Pa = 1e2
        
        Ns,Np = temp.shape
        self.wpsat_z = np.full((Ns,Np),np.nan)
        
        p_increasing = np.diff(pres)[0] > 0
    
        for i_p in range(Np-2):
            # self.wp_z[:,i_z] = self.mo.pressureIntegral(arr=data.specific_humidity[:,i_z:],pres=pres[i_z:],p_levmin=pres[i_z],p_levmax=pres[-1],z_axis=z_axis)
        
            if p_increasing:
                
                p = pres
                arr = tf.saturationSpecificHumidity(temp,p*hPa_to_Pa)
                p0 = p[0]
                p1 = p[i_p]
                i_w = i_p
                
            else:    

                p = np.flip(pres)
                arr = tf.saturationSpecificHumidity(np.flip(temp,axis=z_axis),p*hPa_to_Pa)
                p0 = p[0]
                p1 = p[i_p]
                i_w = Np-1-i_p
                
            self.wpsat_z[:,i_w] = self.mo.pressureIntegral(arr=arr,pres=p,p0=p0,p1=p1,z_axis=z_axis)

    # def computeRH(self,qv,temp,pres,z_axis=1):
    #     """Calculates the integrated water path of saturation specific humidity above each level.
        
    #     Arguments:
    #         - qv: specific humidity in kg/kg, Ns x Nz matrix
    #         - temp: temperature in K, Ns x Nz matrix
    #         - pres: pressure coordinate in hPa, Nz vector
            
    #     Stores:
    #         - self.wpsat_z: water path above each level, Ns x Nz matrix"""

    #     hPa_to_Pa = 1e2
        
    #     qvsat = tf.saturationSpecificHumidity(temp,pres*hPa_to_Pa)
    #     qv = 
        
        

class RadiativeFeaturesFromXarray():
    """Finds and stores characteristics of the peak radiative cooling"""
    
    def __init__(self,dz_smooth=150):
        """Class constructor
        
        Arguments:
            - dz_smooth: filter width (default, 150m)"""
        
        self.dz_smooth = dz_smooth
        self.qrad_peak = None
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


    def computeQradPeaks(self,data,which='net'):
        
        if self.qrad_peak is not None:
            print("Abort: qrad_peak is already computed")
            pass
        
        self.launch_time = data.launch_time.values
        self.z = data.alt.values
        # dz = np.nanmean(np.diff(self.z))
        # n_smooth = self.dz_smooth/dz
        
        # define
        setattr(self,'i_%s_peak'%which,np.nan*np.zeros((self.launch_time.size,),dtype=int))
        setattr(self,'z_%s_peak'%which,np.nan*np.zeros((self.launch_time.size,)))
        setattr(self,"qrad_%s_peak"%which,np.nan*np.zeros((self.launch_time.size,)))
        setattr(self,"qrad_%s_smooth"%which,np.nan*np.zeros((self.launch_time.size,self.z.size)))
        
        for i_lt in range(data.dims['launch_time']):
            
            if which == 'net':
                data_i = data.q_rad.values[i_lt]
            else:
                data_i = getattr(data,'q_rad_%s'%which).values[i_lt]
            i, qrad_i, qrad_s = self.findPeak(data_i,return_all=True)
            
            getattr(self,'i_%s_peak'%which)[i_lt] = i
            getattr(self,'z_%s_peak'%which)[i_lt] = self.z[i]
            getattr(self,'qrad_%s_peak'%which)[i_lt] = qrad_i
            getattr(self,'qrad_%s_smooth'%which)[i_lt,:] = qrad_s
            
        # convert to int (again..)
        setattr(self,'i_%s_peak'%which,np.array(getattr(self,'i_%s_peak'%which),dtype=int))
        
        setattr(self,'%s_peaks'%which,xr.Dataset({"launch_time":(["launch_time"], self.launch_time),\
                                 "z":(["zlay"],self.z),\
                                 "longitude":(["launch_time","zlay"],data.longitude.values),\
                                 "latitude":(["launch_time","zlay"],data.latitude.values),\
                                 "i_%s_peak"%which:(["launch_time"],getattr(self,'i_%s_peak'%which)),\
                                 "z_%s_peak"%which:(["launch_time"],getattr(self,'z_%s_peak'%which)),\
                                 "qrad_%s_peak"%which:(["launch_time"],getattr(self,'qrad_%s_peak'%which)),\
                                 "qrad_%s_smooth"%which:(["launch_time","zlay"],getattr(self,'qrad_%s_smooth'%which))}))

    def findPeak(self,values,n_smooth_0=15,return_all=False):
        """Returns index and value of radiative cooling peak.
        
        Arguments:
        - values: numpy array
        - n_smooth: width of smoothing window (number of points)
        - return_all: boolean
        
        Returns:
        - """
        
        val_smooth_0 = np.convolve(values,np.repeat([1/n_smooth_0],n_smooth_0),
                                 'same')

        ind = np.nanargmin(val_smooth_0)

        if return_all:
            return ind, val_smooth_0[ind], val_smooth_0
        else:
            return ind, val_smooth_0[ind]
        
    def computePW(self,data,i_z_max=-1,attr_name='pw'):
        """Compute and store precipitable water
        
        Arguments:
        - data: xarray"""

        # qv
        qv = data.specific_humidity.values
        # density
        t_lay = data.temperature.values
        R = 287 # J/kg/K
        p_lay = data.pressure.values
        rho_lay = p_lay/(R*t_lay)
        self.rho = rho_lay
        # dz
        dz = np.diff(data.alt_edges)
        dz = np.append(np.append([dz[0]],dz),[dz[-1]])
        dz_3D = np.repeat(dz[np.newaxis,:],t_lay.shape[0],axis=0)
        # PW
        pw_layers = qv*rho_lay*dz_3D
        
        if i_z_max.__class__ is int:
            setattr(self,attr_name,np.nansum(pw_layers[:,:i_z_max],axis=1))
        elif i_z_max.__class__ is np.ndarray:
            # truncate at height
            n_pw = pw_layers.shape[0]
            pw = np.nan*np.zeros((n_pw,))
            for i_pw in range(n_pw):
                i_z = i_z_max[i_pw]
                pw[i_pw] = np.nansum(pw_layers[i_pw,:i_z])
            setattr(self,attr_name,pw)
            
    def computeWPaboveZ(self,data,pres,z_axis=1):

        Ns,Np = data.specific_humidity.shape
        self.wp_z = np.full((Ns,Np),np.nan)
        
        p_increasing = np.diff(pres[0])[0] > 0
    
        for i_p in range(Np-2):
            # self.wp_z[:,i_z] = self.mo.pressureIntegral(arr=data.specific_humidity[:,i_z:],pres=pres[i_z:],p_levmin=pres[i_z],p_levmax=pres[-1],z_axis=z_axis)
        
            if p_increasing:
                
                arr = data.specific_humidity
                p = pres
                p0 = p[0]
                p1 = p[i_p]
                
            else:    

                arr = np.flip(data.specific_humidity,axis=z_axis)
                p = np.flip(pres)
                p0 = p[0]
                p_1 = p[i_p]
                
        self.wp_z[:,i_z] = self.mo.pressureIntegral(arr=arr,pres=p,p0=p0,p1=p1,z_axis=z_axis)
    
    def computeSaturatedWPaboveZ(self,data,pres,z_axis=1):

        Ns,Nz = data.specific_humidity.shape
        self.wpsat_z = np.full((Ns,Nz),np.nan)        
        
        for i_s in range(Ns):
            
            temp = data.temperature.values[i_s]
            qv_star = tf.saturationSpecificHumidity(temp,pres)

            for i_z in range(Nz-2):
                self.wpsat_z[i_s,i_z] = self.mo.pressureIntegral(arr=qv_star,pres=pres,p0=pres[0],p1=pres[i_z],z_axis=z_axis)
    
        
    
