#!/usr/bin/env python
# coding: utf-8

# # Mass functions
# May 31, 2020
# 
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import LinearSegmentedColormap

from scipy.integrate import quad
from scipy.interpolate import interp1d, interp2d

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

from helpers import sigma_cr
from helpers import z_to_chi, chi_to_z


# In[2]:


cosmo = FlatLambdaCDM(H0=67.5, Om0=.316)
arcsec = np.pi/648000

fsub_std = 4e-3 # normalized at redshift 0.5

### KNOWN SYSTEMS ###

# these are the interloper plateaus corresponding to the redshifts
namelist_bells = ['J0029 + 2544','J0113 + 0250', 'J0201 + 3228','J0237 − 0641','J0742 + 3341','J0755 + 3445',
                  'J0856 + 2010','J0918 + 4518','J0918 + 5104','J1110 + 2808','J1110 + 3649','J1141 + 2216',
                  'J1201 + 4743','J1226 + 5457','J1529 + 4015','J2228 + 1205','J2342 − 0120']
zlenslist_bells = [0.587,0.623,0.396,0.486,0.494,0.722,0.507,0.581,0.581,0.607,0.733,
                   0.586,0.563,0.498,0.531,0.530,0.527]
zsourcelist_bells = [2.450,2.609,2.821,2.249,2.363,2.634,2.233,2.344,2.404,2.399,2.502,
                     2.762,2.162,2.732,2.792,2.832,2.265]

namelist_slacs = ['J0252 + 0039','J0737 + 3216','J0946 + 1006','J0956 + 5100','J0959 + 4416','J1023 + 4230 ',
                  'J1205 + 4910','J1430 + 4105 ','J1627 − 0053 ','J2238 − 0754','J2300 + 0022']
zlenslist_slacs = np.array([0.280,0.322,0.222,0.240,0.237,0.191,
                    0.215,0.285,0.208,0.137,0.228])
zsourcelist_slacs = np.array([0.982,0.581,0.609,0.470,0.531,0.696,0.481,
                      0.575,0.524,0.713,0.463])

# JVAS B1938+666
zlenslist3 = [0.222]
zsourcelist3 =[0.881]

# SDP.81
zlenslist4 = [0.2999]
zsourcelist4 = [3.042]


def calc_numbers(resl):
    # ## z-range

    # In[3]:


    # resl = 10 # number of redshift values
    zmin = 0.001
    zmax = 3.301
    zrange = np.linspace(zmin,zmax,resl)


    # In[5]:


    chimin = z_to_chi(zmin) # kpc
    chimax = z_to_chi(zmax) # kpc


    # In[4]:


    def double_cone_direct(com_z, com_l, com_s):
        # Input can be in any units
        if com_z < com_l:
            return com_z / com_l
        else:
            return (com_s - com_z) / (com_s - com_l)


    # ## mass range

    # In[6]:


    mlow = 1e5 # 6e6 gave a good match with the order of magnitude of the 10^6 Msun examples from the Despali paper (1710.05029)
    mhigh = 1e8


    # ## Number of interlopers

    # In[7]:


    # Load Sheth-Tormen results
    h = .675
    massesn = np.load('files/st_results/WIDE_massesn_STnew.npy') # units solar masses
    massfunc_matrix = np.load('files/st_results/WIDE_massfunc_STnew.npy') /1000**3 # convert 1/Mpc^3 to 1/kpc^3
    zlist = np.load('files/st_results/WIDE_redshifts_STnew.npy') # redshifts

    print('not multiplying by h^3 this time!')

    massfunc = interp2d(zlist, massesn, massfunc_matrix, bounds_error=False) # function of (z, mass)


    # In[8]:


    ndens_list = [quad(lambda m: massfunc(z, m), mlow, mhigh)[0]
                  for z in zrange]
    chi_list = z_to_chi(zrange) #[z_to_chi(z) for z in zrange]
    ndens_func = interp1d(chi_list, ndens_list)


    # In[9]:


    def int_number(zl, zs):
        ## Calculate the number of interlopers in a double-pyramid of area 1 kpc^2 PHYSICAL
        coml = z_to_chi(zl)
        coms = z_to_chi(zs)

        # vol = quad(lambda chi: double_cone_direct(chi, coml, coms)**2, chimin, coms)
        #ndens = lambda chi: quad(lambda m: massfunc(chi_to_z(chi), m), mlow, mhigh)[0]
        ntot_comoving = quad(lambda chi: ndens_func(chi) * double_cone_direct(chi, coml, coms)**2, chimin, coms)[0]
        ntot_phys = ntot_comoving * (1+zl)**2
        return ntot_phys


    # ## Number of Subhalos

    # In[11]:


    # fsub_std = 4e-3 
    sigcr_std = sigma_cr(0.5, 1.0).to(u.Msun / u.kpc**2).value
    beta = -1.9
    mhigh_std = 1e8
    mlow_std = 1e5
    def sub_number(zl, zs):
        # number per kpc^2 PHYSICAL
        return ( 0.3 * sigcr_std * fsub_std * ((1+zl)/(1+0.5))**2.5 *
                (2+beta)/(1+beta) *
                (mhigh**(1+beta) - mlow**(1+beta))/(mhigh_std**(2+beta)-mlow_std**(2+beta)) )


    # ## Conversion to arcsec$^2$

    # In[13]:


    def kpc2_to_arcsec2(zl):
        # converts 1 physical kpc^2 to arcsec^2
        ang_diam_dist = z_to_chi(zl)/(1+zl) # kpc
        kpc_to_rad = 1/ang_diam_dist
        kpc_to_as = kpc_to_rad / arcsec
        return kpc_to_as**2


    # ## Graphs

    # In[16]:

    int_nums = np.zeros((resl, resl))
    sub_nums = np.zeros((resl, resl))

    for i, zl in enumerate(zrange):
        for j, zs in enumerate(zrange):
            if zs <= zl:
                int_nums[j,i] = float('nan')
                sub_nums[j,i] = float('nan')
                continue
            
            int_nums[j, i] = int_number(zl, zs) / kpc2_to_arcsec2(zl)
            sub_nums[j, i] = sub_number(zl, zs) / kpc2_to_arcsec2(zl)


    return int_nums, sub_nums, zmin, zmax


## Insert plateau stuff here
def calc_plateaus():

    ## zrange
    resl = 100 # number of redshift values
    zmin = 0.001
    zmax = 3.301 ## now it is time to generate a redshift list
    zrange = np.linspace(zmin,zmax,resl)

    ### Interlopers (Sheth-Tormen) ###
    
    limber_plats = np.load('files/plateau_mats/limber_plats.npy')

    # Press-Schechter file:
    # limber_plats_PS = np.load('files/plateau_mats/limber_platsPS.npy')

    ### Subhalos ###

    m_low = 1e5
    m_high = 1e8
    
    def power_sub(z_l, z_s, mlow=m_low, mhigh=m_high):
        # Calculates plateau of power spectrum for given zl and zs
        ## Corrected version :
        redshift_dependence = ((1+z_l)/(1+.5))**2.5
        sigcr_std = sigma_cr(0.5, 1.0).to(u.M_sun / u.kpc**2).value
        beta = -1.9
        F = 0.3 * fsub_std * sigcr_std * (2+beta) / (mhigh**(2+beta) - mlow**(2+beta)) * redshift_dependence
        sigcr = sigma_cr(z_l, z_s).to(u.M_sun / u.kpc**2).value
        return F/sigcr**2 * (mhigh**(3+beta)-mlow**(3+beta))/(3+beta)

    ## Calculate sub plateau grid ##
    sub_power_grid = np.zeros((len(zrange), len(zrange)))
    for i_l, z_l in enumerate(zrange):
        for i_s, z_s in enumerate(zrange):
            if z_l < z_s:
                sub_power_grid[i_s, i_l] = power_sub(z_l, z_s)
            else:
                sub_power_grid[i_s, i_l] = float('nan')


    return limber_plats[:,:,0], sub_power_grid, zmin, zmax
                
int_nums, sub_nums, zmin_nums, zmax_nums = calc_numbers(resl=50)
int_plats, sub_plats, zmin_plats, zmax_plats = calc_plateaus()

def draw_graph():
    fig, axes = plt.subplots(4,3, figsize=(10,12))

    im = {} # contourf dict
    # m = {} # colormap dict
    
    ext_nums = [zmin_nums, zmax_nums, zmin_nums, zmax_nums]
    ext_plats = [zmin_plats, zmax_plats, zmin_plats, zmax_plats]

    ni_levels = np.arange(-2, 5.5, .5)
    pi_levels = np.arange(-8, -2.5, .5)
    pr_levels = np.arange(-1.5, 1.51, .25)

    # mycmap = LinearSegmentedColormap.from_list('mycmap', ['gold', 'mediumturquoise', 'lightsalmon'])
    mycmap = 'inferno'

    titlesize = 14
    
    # CS = plt.contourf(X, Y, Z, 5, vmin = 0., vmax = 2., cmap=cm.coolwarm)
    # plt.title('Simplest default with labels')
    # m = plt.cm.ScalarMappable(cmap=cm.coolwarm)
    # m.set_array(Z)
    # m.set_clim(0., 2.)
    # plt.colorbar(m, boundaries=np.linspace(0, 2, 6))

    ### Plateaus fsub = 4e-3 ####

    ## Interlopers
    im[0,0] = axes[0,0].contourf(np.log10(int_plats), extent=ext_plats, origin='lower',
                                 levels=pi_levels, cmap=mycmap)
    axes[0,0].set_title(r'$\log_{10}( P_{\rm I, 0} / \mathrm{kpc}^2 )$', size=titlesize)
    
    ## Subhalos
    im[0,1] = axes[0,1].contourf(np.log10(sub_plats), extent=ext_plats, origin='lower',
                                 levels=pi_levels, cmap=mycmap)
    axes[0,1].set_title(r'$\log_{10}( P_{\rm S, 0} / \mathrm{kpc}^2 )$', size=titlesize)

    ## Ratio
    im[0,2] = axes[0,2].contourf(np.log10(sub_plats/int_plats), extent=ext_plats, origin='lower',
                                 levels=pr_levels, cmap='RdBu')

    axes[0,2].set_title(r'$\log_{10}( P_{\rm S, 0} / P_{\rm I, 0} )$', size=titlesize)

    
    ### Numbers, fsub = 4e-3 ###
    
    ## Interlopers
    im[1,0] = axes[1,0].contourf(np.log10(int_nums), extent=ext_nums, origin='lower',
                                 levels=ni_levels)
    axes[1,0].set_title(r'$\log_{10}(N_{\rm I} / \mathrm{arcsec}^{-2})$', size=titlesize)

    ## Subhalos
    im[1,1] = axes[1,1].contourf(np.log10(sub_nums), extent=ext_nums, origin='lower',
                                 levels=ni_levels)
    axes[1,1].set_title(r'$\log_{10}( N_{\rm S} / \mathrm{arcsec}^{-2})$', size=titlesize)

    ## Ratio
    im[1,2] = axes[1,2].contourf(np.log10(sub_nums/int_nums), extent=ext_nums, origin='lower',
                                 levels=pr_levels, cmap='RdBu')
    axes[1,2].set_title(r'$\log_{10}(N_{\rm S}/N_{\rm I})$', size=titlesize)


    ### Plateaus fsub = 2e-2 ####

    sub_plats_fsub2 = sub_plats * 2e-2 / fsub_std
    
    ## Interlopers
    im[2,0] = axes[2,0].contourf(np.log10(int_plats), extent=ext_plats, origin='lower',
                                 levels=pi_levels, cmap=mycmap)
    axes[2,0].set_title(r'$\log_{10}( P_{\rm I, 0} / \mathrm{kpc}^2 )$', size=titlesize)
    
    ## Subhalos
    im[2,1] = axes[2,1].contourf(np.log10(sub_plats_fsub2), extent=ext_plats, origin='lower',
                                 levels=pi_levels, cmap=mycmap)
    axes[2,1].set_title(r'$\log_{10}( P_{\rm S, 0} / \mathrm{kpc}^2 )$', size=titlesize)

    ## Ratio
    im[2,2] = axes[2,2].contourf(np.log10(sub_plats_fsub2/int_plats), extent=ext_plats, origin='lower',
                                 levels=pr_levels, cmap='RdBu')
    cs = axes[2,2].contour(np.log10(sub_plats_fsub2/int_plats), extent=ext_plats, origin='lower',
                           levels=[0], colors='black')
    # plt.clabel(cs, inline=1, fontsize=12, fmt='%1.1f', colors='black')
    axes[2,2].set_title(r'$\log_{10}( P_{\rm S, 0} / P_{\rm I, 0} )$', size=titlesize)

    
    ### Numbers, fsub = 2e-2 ###

    sub_nums_fsub2 = sub_nums * 2e-2/fsub_std
    
    ## Interlopers
    im[3,0] = axes[3,0].contourf(np.log10(int_nums), extent=ext_nums, origin='lower',
                                 levels=ni_levels)
    axes[3,0].set_title(r'$\log_{10}(N_{\rm I} / \mathrm{arcsec}^{-2})$', size=titlesize)

    ## Subhalos
    im[3,1] = axes[3,1].contourf(np.log10(sub_nums_fsub2), extent=ext_nums, origin='lower',
                                 levels=ni_levels)
    axes[3,1].set_title(r'$\log_{10}( N_{\rm S} / \mathrm{arcsec}^{-2})$', size=titlesize)

    ## Ratio
    im[3,2] = axes[3,2].contourf(np.log10(sub_nums_fsub2/int_nums), extent=ext_nums, origin='lower',
                                 levels=pr_levels, cmap='RdBu')
    cs = axes[3,2].contour(np.log10(sub_nums_fsub2/int_nums), extent=ext_nums, origin='lower',
                           levels=[0], colors='black')
    plt.clabel(cs, inline=1, fontsize=12, fmt='%1.1f', colors='black')
    axes[3,2].set_title(r'$\log_{10}(N_{\rm S}/N_{\rm I})$', size=titlesize)
    

    
    ### All ###
    for i in range(3):
        axes[3,i].set_xlabel(r'$z_l$', size=14)

        for j in range(4):
            ## Known systems
            size = 7
            lw = .5
            axes[j,i].scatter(zlenslist_bells, zsourcelist_bells, s=size,
                              edgecolors='white', c='green', linewidths=lw,
                              label='BELLS')
            axes[j,i].scatter(zlenslist_slacs, zsourcelist_slacs, s=size,
                              edgecolors='white', c='orange', linewidths=lw,
                              label='SLACS')
            axes[j,i].scatter(zlenslist3, zsourcelist3, s=size,
                              edgecolors='white', c='red', linewidths=lw,
                              label='JVAS B1938+666')
            axes[j,i].scatter(zlenslist4, zsourcelist4, s=size,
                              edgecolors='white', c='purple', linewidths=lw,
                              label='SDP.81')

            ## Axis labels and color bars
            axes[j,i].tick_params(axis='both', labelsize=12)
            axes[j,i].set_xticks([0,1,2,3])
            axes[j,i].set_yticks([0,1,2,3])

            # axes[j,i].set_aspect('equal')
            
            cbar = fig.colorbar(im[j,i], ax=axes[j,i])
            cbar.formatter.set_powerlimits((-2,2))
            cbar.ax.yaxis.set_offset_position('left')
            cbar.ax.tick_params(labelsize=10)
            cbar.update_ticks()
            
    for j in range(4): # for each row
        if j in [0,1]:
            text = '$f_{\\mathrm{sub},0.5} = 4\\times 10^{-3}$\n$z_s$'
        else:
            text = '$f_{\\mathrm{sub},0.5} = 2\\times 10^{-2}$\n$z_s$'
        axes[j, 0].set_ylabel(text, size=14)

    axes[0,2].legend(loc='lower right', fontsize=9)
        
    plt.tight_layout()
    plt.savefig('imgs/jun11_fullpage.pdf')
    # plt.show()

draw_graph()
    
################################################################################

# # Fixed z_l, z_s

# In[ ]:


zl = 1.5
zs = 2


# ## Subhalo

# In[ ]:


fsub_std = 4e-3 # normalized at redshift 0.5
sigcr_std = sigma_cr(0.5, 1.0).to(u.Msun / u.kpc**2).value
beta = -1.9
mhigh_std = 1e8
mlow_std = 1e5
def sub_mf(m):
    # number per kpc^2 PHYSICAL
    return ( 0.3 * sigcr_std * fsub_std * ((1+zl)/(1+0.5))**2.5 *
            (2+beta)/(mhigh_std**(2+beta)-mlow_std**(2+beta))
            * m**beta)


# ## Interloper

# In[ ]:


def int_mf(m):
    ## Calculate the number of interlopers in a double-pyramid of area 1 kpc^2 PHYSICAL
    coml = z_to_chi(zl)
    coms = z_to_chi(zs)
    
    # vol = quad(lambda chi: double_cone_direct(chi, coml, coms)**2, chimin, coms)
    #ndens = lambda chi: quad(lambda m: massfunc(chi_to_z(chi), m), mlow, mhigh)[0]
    ntot_comoving = quad(lambda chi: massfunc(chi_to_z(chi), m) * double_cone_direct(chi, coml, coms)**2, chimin, coms)[0]
    ntot_phys = ntot_comoving * (1+zl)**2
    return ntot_phys


# ## Graphs

def show_fixed_zlzs():
    # todo make this a function of zl and zs
    mrange = np.logspace(5,8,20)
    subs_dndm = sub_mf(mrange) / kpc2_to_arcsec2(zl)
    ints_dndm = np.array([int_mf(m) for m in mrange]) / kpc2_to_arcsec2(zl)


    plt.loglog(mrange, subs_dndm, label=r'Sub, $f_{{\rm sub},.5}$=4e-3')
    plt.loglog(mrange, ints_dndm, label='Int')
    plt.legend()
    plt.title(r'$z_l = {}$, $z_s = {}$'.format(zl,zs))
    plt.ylabel(r'$d^2 N/ dM \, dA$ [$M_\odot^{-1} \mathrm{arcsec}^{-2}$]')
    plt.xlabel(r'Perturber mass [$M_\odot$]')

    plt.show()




