"""Module thermoFunctions

Functions to compute thermodynamic properties of the atmosphere. Adiabatic lapse
rates and temperature profiles, saturation specific humidity, 
"""


#---- Modules ----#
from math import *
import numpy as np
import numpy.ma as ma
from scipy.optimize import leastsq

#---- Own modules ----#
from thermoConstants import *

#---- Functions ----#

## Latent heat of vaporisation for water
def latentHeatWater(temp):

    """Input in K. Output in J/kg. Valid between 248K and 313K."""

    T = temp-273.15

    return (2500.8 - 2.36*T + 0.0016*(T**2) - 0.00006*(T**3))*1000

## Virtual coefficient
def virtualCoef(shum):
    return (eps+shum)/(eps*(1+shum))

## Air density from values of T, p and q
def airDensity(temp,pres,shum):

    """Use ideal gas law and virtual effect in linearized form
    Arguments:
        - temperature (K), pressure (Pa) and specific humidity (kg/kg) values 
        as numpy.ndarray's or dask.array's
    Returns:
        - density (kg/m3) values in the same format and type"""

    cn = np
    rho_dry = (pres)/(R_d*temp)
    virtual_coef = virtualCoef(shum)
    rho_moist = (rho_dry)/(virtual_coef)

    return rho_moist

## Saturation vapor pressure from Goff-Gratch (1984)
def saturationVaporPressure(temp):

    """Argument: Temperature (K) as a numpy.ndarray or dask.array
    Returns: saturation vapor pressure (Pa) in the same format."""

    T_0 = 273.15
    cn = np
    
    def qvstar_numpy(temp):

        whereAreNans = np.isnan(temp)
        temp_wo_Nans = temp.copy()
        temp_wo_Nans[whereAreNans] = 0.
        # Initialize
        e_sat = np.zeros(temp.shape)
        e_sat[whereAreNans] = np.nan
        #!!! T > 0C
        overliquid = (temp_wo_Nans > T_0)
        ## Buck
        e_sat_overliquid = 611.21*np.exp(np.multiply(18.678-(temp-T_0)/234.5,
                                                      np.divide((temp-T_0),257.14+(temp-T_0))))
        # ## Goff Gratch equation  for liquid water below 0ºC
        # e_sat_overliquid = np.power(10,-7.90298*(373.16/temp-1)
        #             + 5.02808*np.log10(373.16/temp) 
        #             - 1.3816e-7*(np.power(10,11.344*(1-temp/373.16)) -1) 
        #            + 8.1328e-3*(np.power(10,-3.49149*(373.16/temp-1)) -1) 
        #            + np.log10(1013.246)) 
        e_sat[overliquid] = e_sat_overliquid[overliquid]
        #!!! T < 0C 
        overice = (temp_wo_Nans < T_0)
        # ## Buck
        # e_sat_overice = 611.15*np.exp(np.multiply(23.036-(temp-T_0)/333.7,
        #                                            np.divide((temp-T_0),279.82+(temp-T_0))))
        ## Goff Gratch equation over ice 
        e_sat_overice =  100*np.power(10,-9.09718*(273.16/temp - 1) 
                    - 3.56654*np.log10(273.16/temp) 
                    + 0.876793*(1 - temp/ 273.16) 
                    + np.log10(6.1071))
        e_sat[overice] = e_sat_overice[overice]

        return e_sat       # in Pa

    if cn is np:
        return qvstar_numpy(temp)
#    elif temp.__class__ == da.core.Array:
#        return da.map_blocks(qvstar_numpy,temp,dtype=np.float64)
    elif 'float' in str(temp.__class__):
        if temp > T_0:
            return 611.21*np.exp((18.678-(temp-T_0)/234.5)*(temp-T_0)/(257.14+(temp-T_0)))
        else:
            return 611.15*np.exp((23.036-(temp-T_0)/333.7)*(temp-T_0)/(279.82+(temp-T_0)))
    else:
        print("[Error in thermoFunctions.saturationVaporPressure] Unvalid data type:", type(temp))
        return

## Compute the saturation specific humidity based on the expressions by Buck
def saturationSpecificHumidity(temp,pres):

    """Convert from estimate of saturation vapor pressure to saturation specific
    humidity using the approximate equation qvsat ~ epsilon"""

    e_sat = saturationVaporPressure(temp)
    qvstar = (e_sat/R_v)/(pres/R_d)

    return qvstar


## Potential temperature
def potentialTemperature(temp,pres,p0=100000):

    return np.multiply(temp,np.power(pres/p0,-R_d/c_pd))

## Equivalent theta_e, exponential approximation
def equivPotentialTemperature(temp,pres,spechum=None,relhum=None,p0=100000):
    with np.errstate(invalid='ignore'):
        if spechum is None and relhum is not None:
            spechum = relhum*saturationSpecificHumidity(temp,pres)
        return potentialTemperature(temp,pres,p0)*(np.exp(L_v/c_pd*np.divide(spechum,temp)))

## Saturation equivalent theta_e, exponential approximation
def saturationEquivPotentialTemperature(temp,pres,p0=100000):
    with np.errstate(invalid='ignore'):
        return potentialTemperature(temp,pres,p0)*(np.exp(np.divide(L_v/(c_pd),temp)*eps*
                                                          np.divide(saturationVaporPressure(temp),pres)))




## Dry-adiabatic lapse rate
def dryAdiabaticLapseRate(temp,pres,spechum):

    """In pressure coordinates: Gamma_d/(rho*g) (K/Pa)."""

    cn = np
    dryGAmma_zCoord = gg/c_pd   # K/m
    rho = airDensity(temp,pres,spechum) # kg/m3
    dryGAmma_pCoord = (dryGAmma_zCoord)/(rho*gg)  # K/Pa

    return dryGAmma_pCoord

## Parametric analytic approximation to the moist adiabat
def moistAdiabatParametric(pres,p_ref,t_ref,parameter=1,latent_heat=L_v):

    # return t_ref/(1-R_v*t_ref/L_v*parameter*np.log(pres/p_ref))
    return t_ref/(1-R_v*t_ref/latent_heat*parameter*np.log(pres/p_ref))

## Optimize parameter from the qvstar profile
def parameterMoistAdiabat(pres,p_ref,t_ref,qvstar):
    
    """Compute the parameter in the analytic approximation for the moist adiabat,
    using least squares on the resulting profile in saturation specific humidity."""

    def fun(p):
        t_profile_of_p = moistAdiabatParametric(pres,p_ref,t_ref,parameter=p)
        qvs_of_p = saturationSpecificHumidity(t_profile_of_p,pres)
        diff_p = qvstar - qvs_of_p
        diff_p[np.isnan(diff_p)] = 0
        return diff_p

    return leastsq(fun,1)[0][0]

## Multiplicative factor to convert dry adiabat into moist adiabat
def moistAdiabatFactor(temp,pres):

    """Derived from the conservation of saturated moist static energy, the 
    Clausius-Clapeyron formula and the hydrostatic equation. Ignores conversion
    to ice and graupel and condensate loading."""

    cn = np
    inshape = temp.shape
    qvstar = saturationSpecificHumidity(temp,pres)
    coef_m = (1+(L_v*qvstar)/(R_d*temp))/(1+(L_v**2.*qvstar)/(c_pd*R_v*(temp**2.)))

    return coef_m

## Moist adiabatic lapse rate from condensation over liquid
def moistAdiabaticLapseRateSimple(temp,pres,spechum):

    """Returns the value of the moist adiabatic lapse rate as derived in textbooks
    from the conservation of liquid moist static energy. Convert on pressure
    coordinate by assuming hydrostaticity (K/Pa)."""

    cn = np
    Gamma_d_pCoord = dryAdiabaticLapseRate(temp,pres,spechum)  # K/Pa
    coef_m = moistAdiabatFactor(temp,pres)          # Unitless
    Gamma_m_pCoord = (Gamma_d_pCoord*coef_m) # K/Pa

    return Gamma_m_pCoord   

## Moist adiabatic temperature profile on sigma-levels from values of 
## surface temperature, and atmospheric pressure and specific humidity
def moistAdiabatSimple(surftemp,pres,spechum=None,relhum=None,levdim=0,startind=0,reverse=False):

    """Vertically integrate the analytic expression for the moist adiabatic 
    lapse rate from surface values (K).
    Arguments:
        - Ts (K, dimensions: [Nt,1,Nlat,Nlon])
        - p (Pa, dimensions: [Nt,Nlev,Nlat,Nlon])
        - q (kg/kg, dimensions: [Nt,Nlev,Nlat,Nlon])
    Usage with dask:
        Make sure the vertical coordinate is in dimension 0
        Make sure the chunks are the same
        Make sure there is only one chunk in the vertical
        Execute da.map_blocks(moistAdiabatSimple,Ts,p,q)
    Careful:
        When using this function with da.map_blocks, make sure the vertical
        coordinate is not subdivided into chunks; otherwise the temperature profile 
        will show zigzags.
        This function works with the vertical coordinate in first dimension. Other
        configurations haven't been tested.
    """

    if levdim != 0:
        if spechum is not None:
            temp = moistAdiabatSimple(surftemp,
                                  np.moveaxis(preslevdim,0),
                                  spechum=np.moveaxis(spechum,levdim,0),
                                  levdim=0,
                                  startind=startind,
                                  reverse=reverse)
            return np.moveaxis(temp,0,levdim)
        elif relhum is not None:
            temp = moistAdiabatSimple(surftemp,
                                  np.moveaxis(pres,levdim,0),
                                  relhum=np.moveaxis(relhum,levdim,0),
                                  levdim=0,
                                  startind=startind,
                                  reverse=reverse)
            return np.moveaxis(temp,0,levdim)

    cn = np

    if reverse:
        if spechum is not None:
            return cn.flipud(moistAdiabatSimple(surftemp,
                                            cn.flipud(pres),
                                            spechum=cn.flipud(spechum),
                                            levdim=levdim,
                                            startind=startind))
        elif relhum is not None:
            return cn.flipud(moistAdiabatSimple(surftemp,
                                            cn.flipud(pres),
                                            relhum=cn.flipud(relhum),
                                            levdim=levdim,
                                            startind=startind))

    if startind != 0:
        
        print("This piece of the code for moist adiabats needs to be rewritten. Elevated index chosen arbitrarily.")
        
    p_shape = pres.shape
    ndims = len(pres.shape)
    ind_low = [slice(None)]*ndims
    ind_low[levdim] = 0
    ind_high = ind_low.copy()
    Nlev = p_shape[levdim]

    temp = np.nan*np.zeros(p_shape)
    temp[ind_low] = surftemp
    
    if not np.isnan(surftemp):

        for k in range(1,Nlev):
            ind_low[levdim] = k-1
            ind_high[levdim] = k
            if spechum is None:
                spechum_ind_low = saturationSpecificHumidity(temp[ind_low],
                    pres[ind_low]) * relhum[ind_low]
            else:
                spechum_ind_low = spechum[ind_low]
            dTdp = moistAdiabaticLapseRateSimple(temp[ind_low],
                                                 pres[ind_low],
                                                 spechum=spechum_ind_low)
            dp = cn.subtract(pres[ind_high],pres[ind_low])
            temp[ind_high] = cn.add(temp[ind_low],cn.multiply(dTdp,dp))

    return temp



