"""
Python module to collect useful matrix operations in atmospheric science 
(derivatives, filtering, integrals, divergence, rotationals, etc.).

Author: B. Fildier, 2021
"""

import numpy as np
from scipy.ndimage import gaussian_filter,gaussian_filter1d


class MatrixOperators():

    def __str__(self):
        """Override string function to print attributes
        """
        # str_out = '-- Attributes --'
        method_names = []
        str_out = ''
        for a in dir(self):
            if '__' not in a:
                a_str = str(getattr(self,a))
                if 'method' not in a_str:
                    str_out = str_out+("%s : %s\n"%(a,a_str))
                else:
                    method_names.append(a)
        print('-- Methods --')
        for m in method_names:
            print(m)
        return str_out
    
    def duplicate(self,arr,dims,ref_axis=1):
        """Extent 1D numpy array to ND array corresponding to shape dims, 
        where the original data is the ref_axis dimension.
        dims[ref_axis] = 1 if no duplicate along that axis."""
    
        for i_dim in range(ref_axis+1,len(dims)):
            arr = arr[...,np.newaxis]
    
        return np.tile(arr,dims)
    
    def from1Dto4D(self,vec:np.array,shape:np.array,dim:int=1,changesize=False):
        """Expand dimensions of 1D array 'vec' into 'shape', where shape[dim] equals the current array size"""
        
        if not changesize:
            assert len(vec)==shape[dim],'wrong size'

        # add dimensions
        expand_axes = list(np.arange(1,4))
        vec4d = np.expand_dims(vec,axis=expand_axes)
        # repeat values along new dimensions
        missing_dims = list(shape[:dim])+list(shape[dim+1:])
        vec4d_rep = vec4d.repeat(np.prod(missing_dims))
        # reshape with new dimensions last
        dims_newlast = [len(vec)]+missing_dims
        vec4d_resh = vec4d_rep.reshape(dims_newlast)
        # reorder dims
        vec4d_good = vec4d_resh.swapaxes(0,dim)

        return vec4d_good

    def derivative(self,a:np.array,x:np.array,axis:int=0,deriv_type:str='central',deriv_order:int=1,accuracy:int=2):
        """Derivative

        Args:
            a (np.array): values on which to apply the derivative.
            x (np.array): coordinate to derive along. Must be of same shape as a.
            axis (int, optional): dimension along which the derivative is taken. Defaults to 0.
            deriv_type (str, optional): type of derivative (central, forward, backward). Defaults to 'central'.
            deriv_order (int, optional): order of derivative. Defaults to 1.
            accuracy (int, optional): order of accuracy (2, 4, 6). Defaults to 2.

        Returns:
            np.array: derivative
        """
        
        assert deriv_type == 'central', "other types of finite-differences still to be implemented"
        
        stencils = {1:{2:[-1/2,0,1/2],
                      4:[1/12,-2/3,0,2/3,-1/12],
                      6:[-1/60,3/20,-3/4,0,3/4,-3/20,1/60]},
                   2:{2:[1,-2,1],
                      4:[-1/12,4/3,-5/2,4/3,-1/12],
                      6:[1/90,-3/20,3/2,-49/18,3/2,-3/20,1/90]}}
        
        # pick stencil
        stencil = stencils[deriv_order][accuracy]
        
        N = int(len(stencil)/2) # works because central finite-difference scheme
        da = stencil[N+1] * a
        dx = stencil[N+1] * x
        
        def rollX(x,i,axis):
            # circular permutation
            x_roll = np.roll(x,i,axis=axis)
            # find indices to edit
            ind_1d = np.arange(i)
            sign_i = (-1)*(i<0)+(i>0)
            ind_1d = (dim - 1)*(sign_i < 0) + sign_i*np.arange(np.abs(i))
            ind = mo.from1Dto4D(vec=ind_1d,shape=shape,dim=axis,changesize=True)
            values = np.take(x_roll,ind_1d,axis=axis) - sign_i*L
            np.put_along_axis(x_roll,indices=ind,values=values,axis=axis)
            
            return x_roll
            
        
        for i in range(1,N+1):
            print('%d, coef = %1.2f'%(i,stencil[N+i]))
            print('%d, coef = %1.2f'%(-i,stencil[N-i]))
            da = da + stencil[N+i]*np.roll(a,i,axis=axis)
            da = da + stencil[N-i]*np.roll(a,-i,axis=axis)
            dx = dx + stencil[N+i]*np.roll(x,i,axis=axis)
            dx = dx + stencil[N-i]*np.roll(x,-i,axis=axis)
            
        return da/dx


    def deriv(self,a,x,axis=0,order=1):
        """Derivative da\dx, d2a/dx2, etc. along given axis"""
        
        if order == 1: # df/dx
            
            # dimensions
            ash = a.shape
            
            ##-- compute first order derivative
            if len(ash) == 1: # if vector

                # First order difference, truncating edges
                def diff1D(vec,coord):
                    return np.convolve(vec,[1,-1],mode='valid')/np.convolve(coord,[1,-1],mode='valid')

                # Interpolate on initial grid and append nans
                def regrid(vec):
                    return np.hstack([[np.nan],np.convolve(vec,[0.5,0.5],mode='valid'),[np.nan]])

                da_dx_mids = diff1D(a,x)
                da_dx = regrid(da_dx_mids)

            else: # if matrix

                Nx = ash[axis]

                ##-- first order derivative
                # duplicate z values to arr shape
                x_full = np.moveaxis(np.tile(x,(*ash[:axis],*ash[axis+1:],1)),-1,axis)
                # derivative
                da = np.take(a,range(1,Nx),axis=axis)-np.take(a,range(0,Nx-1),axis=axis)
                dx = np.take(x_full,range(1,Nx),axis=axis)-np.take(x_full,range(0,Nx-1),axis=axis)
                da_dx_mids = da/dx

                ##-- regrid and append nans on both sides
                da_dx_grid = 0.5*(np.take(da_dx_mids,range(1,Nx-1),axis=axis)+\
                                  np.take(da_dx_mids,range(0,Nx-2),axis=axis))
                # append nans
                hyperspace_nans = np.nan*np.zeros((*ash[:axis],1,*ash[axis+1:]))
                da_dx = np.concatenate([hyperspace_nans,da_dx_grid,hyperspace_nans],axis=axis)

            return da_dx, da_dx_mids


    def gaussianFilter(self,a,sigma,axis=0,**kwargs):
        """Gaussian filter in 1 or 2 dimensions"""
        
        if isinstance(axis,int):
            
            ## 1D, use gaussian_filter1d
            return gaussian_filter1d(a,sigma=sigma,axis=axis,mode='constant')    
            
        elif isinstance(axis,list):
            
            ## ND, use multidimensional gaussian_filter
            """Smooth in x an y and recombine with same shape.
            Assumes dimensions T,Z,Y,X"""
            
            ashape = a.shape
            
            if len(ashape) == 3:
                Nt,Ny,Nx = ashape
                Nz = 0
            elif len(ashape) == 4:
                Nt,Nz,Ny,Nx = ashape
            
            a_out = np.nan*np.zeros(ashape)
            
            for i_t in range(Nt):
                
                if Nz == 0:
                    a_out[i_t,:,:] = gaussian_filter(a[i_t,:,:],sigma=sigma,mode='wrap')
                
                else:
                    for i_z in range(Nz):
        
                        a_out[i_t,i_z,:,:] = gaussian_filter(a[i_t,i_z,:,:],sigma=sigma,mode='wrap')
                    
            return a_out

    def pressureIntegral(self,arr='default',pres='default',p0=None,p1=None,z_axis=1):
        """Vertical pressure integral of arr.
        
        Arguments:
        - arr: numpy array in whatever unit U
        - pres: pressure in hPa (in increasing order)
        - p0: lower integral bound (hPa)
        - p1: upper integral bound (hPa) (p1>p0)
        - z_axis: axis of z/p dimension, if multidimensional
        Returns:
        - integral of arr*dp/g between p0 and p1, in U.kg_a/m2
        """
        
        g = 9.81 # m/s2
        hPa_to_Pa = 1e2

        # test if empty
        if len(pres) == 0:
            return 0

        if np.isnan(p0) or np.isnan(p1):
            return np.nan

        # trim in pressure
        k_bottom,k_top = -1,0
        if p0 is not None:
            i_0 = np.where(pres>=p0)[0]
            if len(i_0)>0:
                i_0 = i_0[k_top]
            else:
                i_0 = k_top
        if p1 is not None:
            i_1 = np.where(pres<=p1)[0]
            if len(i_1)>0:
                i_1 = i_1[k_bottom]
            else:
                i_1 = k_bottom


        # differential in pressure
        dp_mids = np.convolve(pres*hPa_to_Pa,[1,-1],mode='valid')
        dp = np.hstack([[np.nan],np.convolve(dp_mids,[0.5,0.5],mode='valid'),[np.nan]])

        # To compute the mass of the atmospheric column
        if isinstance(arr,str):
            arr = np.ones(pres.shape)
        
        # 1D vector
        if len(arr.shape) == 1:
            
            I_z = arr*dp/g
            
            integral = np.nansum(I_z[i_0:i_1])
        
        else:
            
            # extend dp to shape of multidimensional array
            shape2duplicate = list(arr.shape)
            shape2duplicate[z_axis] = 1
            dp_4D = self.duplicate(dp,shape2duplicate,ref_axis=z_axis)
            # intregrand
            I_z = arr*dp_4D/g
            
            # slice
            s_4D = tuple([slice(None)]*z_axis + [slice(i_0,i_1)] + [slice(None)]*(len(arr.shape)-z_axis-1))
            
            integral = np.nansum(I_z[s_4D],axis=z_axis)
        
        # # return by multiplying by -1 if integral bounds are reverse (according to pressure ordering)
        return integral
            
        
    
                
            

            
            
        