#!/usr/bin/env python
# coding: utf-8

# # Effective Convergence
# *24 January 2020*
# 
# In this notebook, we'll be using an idea I didn't realize up till now, that we can define the effective convergence as:
# $$\kappa_\mathrm{eff} = \frac{1}{2} \nabla \cdot \vec\alpha$$
# (thanks to [Gilman+ 2019](https://arxiv.org/pdf/1901.11031.pdf)).
# 
# Note that we're updating from `lenstronomy 1.0.1` to `1.3.0` (this new version has tNFW profiles).

# In[1]:

import numpy as np
import matplotlib
matplotlib.use('Agg') # only do this to run on the cluster
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import sys
import datetime

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

from scipy import fftpack
from scipy.interpolate import interp1d, interp2d
from scipy.integrate import quad
from scipy.stats import poisson

from collections import Counter

from helpers import CustomImage, autoshow, ADD, sigma_cr, PoolResults
from helpers import z_to_chi, chi_to_z

import pickle
import copy

## Constants ##
arcsectorad = np.pi/648000


assert(len(sys.argv) == 3)

# In[25]:


# Basic parameters
zl = 0.5
zs = 1.
# pixnum, pixsize = 200, .008 # to match S. Birrer's paper (roughly)
# ext = pixnum * pixsize / 2.
pixnum = int(sys.argv[2])
# ext = 8 # should be .8 to match Simon's paper
ext = float(sys.argv[1]) # half the image length in arcsec
print('ext', ext)
pixsize = 2*ext / pixnum
print(pixsize, flush=True)

theta_global = 7 * (ext/8.)

already_calculated_rows = []


# In[26]:


## Calculate physical size of image
# ( note: 845000 kpc to z=.2 )
# distance = 1947000 # kpc to z=.5

cosmo = FlatLambdaCDM(H0=67.5, Om0=0.316) ### this is not perfectly consistent
# should be using
# h = .675

com_l = cosmo.comoving_distance(zl).to(u.kpc).value
com_s = cosmo.comoving_distance(zs).to(u.kpc).value
angle = (2*ext) * arcsectorad # rad
com_width_kpc = com_l * angle # comoving width in kpc
print(com_width_kpc, 'kpc')


# In[27]:


# vol_rough = com_width_kpc**2 * com_s # kpc^3
# # 3.391e6
# print('comoving volume is approx', vol_rough / 1e6, 'million kpc^3')
# del vol_rough


# ## Double cone shape

# In[28]:


def double_cone_direct(com_z, com_l, com_s):
    # Input can be in any units
    if com_z < com_l:
        return com_z / com_l
    else:
        return (com_s - com_z) / (com_s - com_l)

def double_cone(z, zl=zl, zs=zs):
    com_z = cosmo.comoving_distance(z)
    com_l = cosmo.comoving_distance(zl)
    com_s = cosmo.comoving_distance(zs)
    return double_cone_direct(com_z, com_l, com_s)

#zlist = np.linspace(0, zs, 100)
#doubleconelist = [double_cone(z) for z in zlist]
#comlist = [cosmo.comoving_distance(z).to(u.Mpc).value for z in zlist]

# plt.plot(zlist, [double_cone(z) for z in zlist]); plt.show()
# plt.plot(comlist, doubleconelist); plt.show()

com_l = cosmo.comoving_distance(zl).to(u.kpc).value
com_s = cosmo.comoving_distance(zs).to(u.kpc).value
print('com_l', com_l, 'com_s', com_s)
# print('integral (in kpc)', quad(lambda comz: double_cone_direct(comz, com_l, com_s)**2, 0, com_s))

double_cone_volume_kpc = com_width_kpc**2 * quad(lambda comz: double_cone_direct(comz, com_l, com_s)**2, 0, com_s)[0]
# volume of double-cone in kpc**3 (really double prism)

print('double pyramid volume', double_cone_volume_kpc / 1e6, 'million kpc^3')


# ## Substructure
# ### Interlopers

# Units: `massesn` has units of $M_\odot$. `massfunc` has units of 1/(Mpc/h)^3 / M_sun (comoving distance).

# In[29]:


# Load Sheth-Tormen results
#h = .675
massesn = np.load('files/st_results/WIDE_massesn_ST.npy') # units solar masses
massfunc_matrix = np.load('files/st_results/WIDE_massfunc_ST.npy') /1000**3 # convert 1/Mpc^3 to 1/kpc^3
zlist = np.load('files/st_results/WIDE_redshifts_ST.npy') # redshifts

print('NOT multiplying by h^3 anymore!!!!!')

massfunc = interp2d(zlist, massesn, massfunc_matrix, bounds_error=False) # function of (z, mass)

################################################################################
# ### Interlopers

n_planes = 100
z_planes = np.linspace(.001,.999, n_planes+1)[:-1]
z_planes_diff = z_planes[1] - z_planes[0]
com_area_lens = com_width_kpc**2
chi_lens = z_to_chi(zl)
chi_source = z_to_chi(zs)

# slice_vols according to the equation:
# $$SV = A_{com} \int_{\chi_{low}}^{\chi_{high}} \frac{R^2(\chi)}{R^2(\chi_{lens})} d\chi$$

slice_vols = com_area_lens * np.array([quad(lambda chi: double_cone_direct(chi, chi_lens, chi_source)**2,
                                            z_to_chi(zi),
                                            z_to_chi(zi+z_planes_diff))[0]
                            for zi in z_planes])

n_int_massbins = 100 ## maybe try using more mass bins ##
print('Previously using 20 int_massbins. Now using', n_int_massbins)
int_massbins = np.logspace(5,8,n_int_massbins+1)[:-1]
massbins_ratio = int_massbins[1]/int_massbins[0]

# 2d matrix of number of interlopers at a particular redshift and mass:
n_int_xslice_ymass = np.zeros((n_planes, n_int_massbins))
for x, z in enumerate(z_planes):
    for y, mass in enumerate(int_massbins):
        n_int_xslice_ymass[x,y] = slice_vols[x] * quad(lambda m: massfunc(z, m),
                                                       mass,
                                                       mass*massbins_ratio)[0]


################################################################################
# ### Subhalos
# 
# Just a very rough estimate of the mass function:

# In[33]:


phys_width_kpc = 2 * ext * arcsectorad  * ADD(0, zl).to(u.kpc).value
print('phys width (kpc):', phys_width_kpc)


# In[34]:


# einst_radius = 1 * arcsectorad * ADD(0,zl).to(u.kpc)
# print(einst_radius)


# In[35]:


Sigma_sub = 0.012 # kpc^-2
m0 = 1e8 # Msun
dNdm_sub = lambda m: Sigma_sub / m0 * (m / m0)**-1.9 * phys_width_kpc**2 # 1/Msun
#dNdm_sub = lambda m: Sigma_sub / m0 * (m / m0)**-1.9 * (np.pi*6.3)**2 #phys_width_kpc**2 # 1/Msun
# I think we should use the physical width, but not 100% sure

mass_bins_sub = np.logspace(5,8,20)[:-1]
bin_ratio = mass_bins_sub[1]/mass_bins_sub[0]
print('mass bins sub:', ['%g'%x for x in mass_bins_sub])
avg_nums_sub = np.array([quad(dNdm_sub, mass, bin_ratio * mass)[0] for mass in mass_bins_sub])
# plt.bar(np.log10(mass_bins_sub), avg_nums_sub, log=True, width=.1)
# #plt.axhline(1)
# plt.xlabel(r'Subhalo mass bin ($\log_{10}$ in $M_\odot$)')
# plt.ylabel('Number')
# print('number', np.sum(avg_nums_sub), avg_nums_sub)


# In[36]:


# m0 = 1e8 # Msun
# dNdm_sub2 = lambda m: 0.125 / m0 * (m / m0)**-1.9 * phys_width_kpc**2 # 1/Msun
# #dNdm_sub = lambda m: Sigma_sub / m0 * (m / m0)**-1.9 * (np.pi*6.3)**2 #phys_width_kpc**2 # 1/Msun
# # I think we should use the physical width, but not 100% sure

# # mass_bins_sub = np.logspace(5.7,10,20)[:-1]
# # bin_ratio = mass_bins_sub[1]/mass_bins_sub[0]
# print('mass bins sub:', ['%g'%x for x in mass_bins_sub])
# avg_nums_sub2 = np.array([quad(dNdm_sub2, mass, bin_ratio * mass)[0] for mass in mass_bins_sub])



# ### Sampling (helper functions)

# In[41]:


def double_cone_angle(z, zl=zl, zs=zs):
    # angle of how wide the interlopers can be dispersed so that they'd show up in the final image (according to the double-prism projection)
    
    # returns a ratio, 1 for z <= zl
    
    com_z = cosmo.comoving_distance(z)
    com_l = cosmo.comoving_distance(zl)
    return double_cone(z, zl=zl, zs=zs) * com_l / com_z

################################################################################

def isinmask(xpix, ypix, r, dr, pixsize, pixnum):
    # r is the einstein radius, and we take pixels within r +- dr
    # (sharp cutoff)
    npix = np.sqrt((xpix-pixnum/2)**2 + (ypix-pixnum/2)**2)
    pixdist = npix * pixsize
    return (r - dr < pixdist < r + dr)

def isinmask_smooth(xpix, ypix, r, dr, pixsize, pixnum):
    # r is the einstein radius, and we take pixels within r +- dr
    # gaussian smoothing
    npix = np.sqrt((xpix-pixnum/2)**2 + (ypix-pixnum/2)**2)
    pixdist = npix * pixsize
    return np.exp(-(pixdist-r)**2/(2*dr**2))


def compute_specific_x_masked(myimg, xpix):
    return compute_specific_x(myimg, xpix, mask=True)
def compute_specific_x(myimg, xpix, mask=False):
    # Helper function to calculate alpha for a row of pixels (used by PoolResults)
    alphax_list = []
    alphay_list = []
    
    # if (xpix in already_calculated_rows
    #    or xpix < xmin_global or xpix >= xmax_global):
    #     return xpix, [[],[]]
    if xpix in already_calculated_rows:
        return xpix, [[],[]]
    
    for ypix in range(pixnum):
        if not mask or isinmask(xpix, ypix, theta_global, 1.1/7 * theta_global, pixsize, pixnum):
            myimg.calc_alpha_pixel(xpix, ypix)
        else:
            myimg.alphamat_x[xpix, ypix] = 0
            myimg.alphamat_y[xpix, ypix] = 0

        alphax_list.append(myimg.alphamat_x[xpix, ypix])
        alphay_list.append(myimg.alphamat_y[xpix, ypix])
        
    print('finished row', xpix, [alphax_list, alphay_list], flush=True)
        
    del myimg
        
    return xpix, [alphax_list, alphay_list]


def do_subhalos():
    print('doing subhalos only')
    np.random.seed(145)
    xs, ys, redshifts, masses = [], [], [], []

    # Populate the subhalos
    rv_nums = poisson.rvs(avg_nums_sub) if len(avg_nums_sub) > 1 else [poisson.rvs(avg_nums_sub)]
    xyext = ext
    for mass, num in zip(mass_bins_sub, rv_nums):
        for i in range(num):
            xs.append(np.random.uniform(-xyext,xyext))
            ys.append(np.random.uniform(-xyext,xyext))
            redshifts.append(zl)
            masses.append(mass)
    print('number of subhalos', len(xs), flush=True)

    now = datetime.datetime.now()
    myimg_sub = CustomImage(xs, ys, redshifts, zl=zl, m=masses, pixnum=pixnum, pixsize=pixsize, mass_sheets=False)
    print('time to generate myimg_sub:', datetime.datetime.now()-now, flush=True)
    now = datetime.datetime.now()
    myimg_sub.alphamat_x = np.zeros((pixnum, pixnum))
    myimg_sub.alphamat_y = np.zeros((pixnum, pixnum))
    mypool = PoolResults(compute_specific_x, [[myimg_sub, i] for i in range(pixnum)])
    pool_results = np.array(mypool.get_results_list())
    print('pool results:', pool_results, flush=True)
    myimg_sub.alphamat_x = pool_results[:,0,:]
    myimg_sub.alphamat_y = pool_results[:,1,:]
    

    print('time to calc div curl for myimg_sub:', datetime.datetime.now()-now)

    blankimg = CustomImage([],[],[], zl=zl, pixnum=pixnum, pixsize=pixsize)
    autoshow(blankimg.image)
    blankimg.calc_div_curl_5pt();

    plt.close()
    autoshow(0.5*(myimg_sub.divmat - blankimg.divmat), ext=ext, vmax=.09)
    plt.title(r'$\kappa_{sub}$ (single plane, CDM)')
    plt.savefig('imgs/kappa_sub.png')

    np.save('files/kappa_sub.npy', 0.5*(myimg_sub.divmat-blankimg.divmat))




def do_real_interlopers(theta):
    print('real interlopers, theta =',theta)
    np.random.seed(145)
    
    xs, ys, redshifts, masses = [], [], [], []
    mass_sheets = []
    #xyext = ext

    # Interlopers
    for i, z_plane in enumerate(z_planes):
        rv_nums = poisson.rvs(n_int_xslice_ymass[i]) # rv_nums: list of numbers for different masses
        #print(rv_nums)
        
        xyext = ext * double_cone_angle(z_plane)
        for mass, num in zip(int_massbins, rv_nums):
            for i in range(num):
                xs.append(np.random.uniform(-xyext, xyext))
                ys.append(np.random.uniform(-xyext, xyext))
                redshifts.append(z_plane)
                masses.append(mass)
                mass_sheets.append(True)
                
    print('number of real interlopers', len(xs), flush=True)
    
#     myimg_proj = CustomImage(xs,ys,redshifts, zl=zl, m=masses,
#                              pixnum=pixnum, pixsize=pixsize,
#                              mass_sheets=mass_sheets, main_theta=theta,
#                              qfactor=1.0)
    
#     with open('files/myimg_proj_ext{}.p'.format(ext), 'wb') as f:
#         pickle.dump(myimg_proj, f)
#         print('dumped myimg_proj')
    with open('files/myimg_proj_ext{}.p'.format(ext), 'rb') as f:
        myimg_proj = pickle.load(f)
        print('loaded myimg_proj!!!')

    ## Calculate divmat semi-manually (so we can pool results) ##
    myimg_proj.alphamat_x = np.zeros((pixnum, pixnum)) # initialize both alphamat
    myimg_proj.alphamat_y = np.zeros((pixnum, pixnum))
    
    # The following three lines are the main computation:
    print('about to start pool', flush=True)
    mypool = PoolResults(compute_specific_x, [[copy.copy(myimg_proj), i] for i in range(pixnum)])
    print('finished pool', flush=True)
    
    
    # Have to run this last part separately
        
#     pool_results = np.array(mypool.get_results_list())
#     np.save('files/tmp_pool_results.npy', pool_results)

#     # Getting pool results from saved file:
#     #pool_results = np.load('files/tmp_pool_results.npy')
#     myimg_proj.alphamat_x = pool_results[:,0,:]
#     myimg_proj.alphamat_y = pool_results[:,1,:]
    
#     myimg_proj.recalc_div_curl_5pt()
    
#     blankimg = CustomImage([], [], [], zl=zl, m=[], 
#                            pixnum=pixnum, pixsize=pixsize, 
#                            mass_sheets=[], main_theta=theta,
#                            qfactor=1.0)

#     blankimg.calc_div_curl_5pt()

#     np.save('files/newkappa_int_ext{}_theta{}_pixnum{}.npy'.format(ext,theta,pixnum), 0.5*(myimg_proj.divmat - blankimg.divmat))

#     print('saved newkappa_int_ext{}_theta{}_pixnum{}.npy'.format(ext,theta,pixnum))
    

def do_full(theta):
    # TODO: fix sloppy interloper distribution
    np.random.seed(145)
    n_planes = 10

    xs, ys, redshifts, masses = [], [], [], []
    mass_sheets = []
    #xyext = ext

    z_planes = np.linspace(.01,.99,n_planes)
    area_proportions = [double_cone(z)**2 for z in z_planes]
    area_sum = np.sum(area_proportions)

    # Interlopers
    for i, z_plane in enumerate(z_planes):
        rv_nums = poisson.rvs(avg_nums_interlopers * area_proportions[i] / area_sum)
        #Note that we're being sloppy here and using the same mass function at all redshifts!!!
        #TODO: fix!
        xyext = ext * double_cone_angle(z_plane)
        for mass, num in zip(mass_bins_interlopers, rv_nums):
            for i in range(num):
                xs.append(np.random.uniform(-xyext,xyext))
                ys.append(np.random.uniform(-xyext,xyext))
                redshifts.append(z_plane)
                masses.append(mass)
                mass_sheets.append(True)
    print('number of interlopers', len(xs), flush=True)
    
    # Subhalos
    rv_nums = poisson.rvs(avg_nums_sub) if len(avg_nums_sub) > 1 else [poisson.rvs(avg_nums_sub)]
    xyext = ext
    for mass, num in zip(mass_bins_sub, rv_nums):
        for i in range(num):
            xs.append(np.random.uniform(-xyext,xyext))
            ys.append(np.random.uniform(-xyext,xyext))
            redshifts.append(zl)
            masses.append(mass)
            mass_sheets.append(False)
    print('number of subhalos + interlopers', len(xs), flush=True)

    myimg = CustomImage(xs, ys, redshifts, zl=zl, m=masses, 
                        pixnum=pixnum, pixsize=pixsize,
                        mass_sheets=mass_sheets, main_theta=theta)
    
    ## Calculate divmat semi-manually (so we can save in the middle) ##
    myimg.alphamat_x = np.zeros((pixnum, pixnum)) # initialize both alphamat
    myimg.alphamat_y = np.zeros((pixnum, pixnum))
    for xpix in range(pixnum):
        for ypix in range(pixnum):
            myimg.calc_alpha_pixel(xpix, ypix)
        np.save('files/tmp_alphax_ext{}_theta{}.npy'.format(ext,theta), myimg.alphamat_x)
        np.save('files/tmp_alphay_ext{}_theta{}.npy'.format(ext,theta), myimg.alphamat_y)
        # insurance in case the calculation is stopped early
    myimg.recalc_div_curl_5pt()
        
    blankimg = CustomImage([], [], [], zl=zl, m=[],
                           pixnum=pixnum, pixsize=pixsize, 
                           mass_sheets=[], main_theta=theta)
    blankimg.calc_div_curl_5pt()

    ## Save results ##
    
    plt.close()
    autoshow(0.5*(myimg.divmat-blankimg.divmat), ext=ext, vmax=.09)
    plt.title(r'$\kappa_{sub}$ (multi-plane Born)')
    plt.savefig('imgs/kappa_full_ext{}_theta{}.png'.format(ext, theta))

    np.save('files/kappa_full_ext{}_theta{}.npy'.format(ext,theta), 0.5*(myimg.divmat - blankimg.divmat))
    
#do_subhalos()
# do_naive_interlopers()
do_real_interlopers(theta_global)
#do_full(7)
    
"""

# In[91]:


#np.save('files/convmat_5to8_birrer.npy', 0.5*(myimg_sub.divmat - blankimg.divmat))


# In[62]:


#np.save('files/convmat_subonly_10xbirrer.npy', 0.5*(myimg_sub.divmat - blankimg.divmat))


# In[91]:


#np.save('files/convmat_subonly_theta1.npy', 0.5*(myimg_sub.divmat - blankimg.divmat))


# ### Next, try projection map

# In[22]:


# Populate the interlopers in projection (middle picture in Fig 2 of Birrer's paper)




# ### Finally, the actual image:

# In[230]:


# Populate the interlopers and subhalos



# In[121]:


autoshow(0.5*(myimg.divmat - blankimg.divmat), ext=ext, vmax=.04)
plt.title(r'Difference in $\kappa_\mathrm{eff}$ (theta=1)')
#plt.savefig('imgs/feb13_final_theta1_5pt.png')


# In[148]:


#np.save('files/convmat_intonlyres_5to8_press.npy', 0.5*(myimg.divmat - blankimg.divmat))


# In[162]:


autoshow(0.5*(myimg.curlmat - blankimg.curlmat), ext=ext, vmax=.01)
plt.title(r'$\frac{1}{2}\nabla\times\alpha$ (theta=1)')
#plt.savefig('imgs/feb13_curl_theta1_5pt.png')


# In[273]:


#np.save('files/convmat_residuals_theta1.npy', 0.5*(myimg.divmat - blankimg.divmat))#


# ### Extended image
# So we can see past the Einstein ring.

# In[163]:


# Populate the interlopers and subhalos

np.random.seed(145)
n_planes = 10

xs, ys, redshifts, masses = [], [], [], []
mass_sheets = []
#xyext = ext

z_planes = np.linspace(.01,.99,n_planes)
area_proportions = [double_cone(z)**2 for z in z_planes]
area_sum = np.sum(area_proportions)

# Interlopers
for i, z_plane in enumerate(z_planes):
    rv_nums = poisson.rvs(4 * avg_nums_interlopers * area_proportions[i] / area_sum)
    xyext = 2*ext * double_cone_angle(z_plane)
    for mass, num in zip(mass_bins_interlopers, rv_nums):
        for i in range(num):
            xs.append(np.random.uniform(-xyext,xyext))
            ys.append(np.random.uniform(-xyext,xyext))
            redshifts.append(z_plane)
            masses.append(mass)
            mass_sheets.append(True)
print('number of interlopers', len(xs))
# # Subhalos
# rv_nums = poisson.rvs(4 * avg_nums_sub) if len(avg_nums_sub) > 1 else [poisson.rvs(4 * avg_nums_sub)]
# xyext = 2*ext
# for mass, num in zip(mass_bins_sub, rv_nums):
#     for i in range(num):
#         xs.append(np.random.uniform(-xyext,xyext))
#         ys.append(np.random.uniform(-xyext,xyext))
#         redshifts.append(zl)
#         masses.append(mass)
#         mass_sheets.append(False)
# print('number of subhalos + interlopers', len(xs))


# In[164]:


ext_extended = 2*ext
print('ext_extended', ext_extended)


# In[165]:


myimg_extended = CustomImage(xs, ys, redshifts, zl=zl, m=masses, pixnum=pixnum, pixsize=pixsize*2, mass_sheets=mass_sheets, main_theta=1.0)


# In[166]:


get_ipython().run_cell_magic('time', '', 'myimg_extended.calc_div_curl_5pt()\n0\n#load files/myimg_extended.npy instead!')


# In[167]:


blankimg_extended = CustomImage([], [], [], zl=zl, m=[], pixnum=pixnum, pixsize=pixsize*2, mass_sheets=mass_sheets, main_theta=1.0)
blankimg_extended.calc_div_curl_5pt()
0


# In[168]:


autoshow(0.5*(myimg_extended.divmat - blankimg_extended.divmat), ext=ext_extended, vmax=0.04)


# In[174]:


autoshow(0.5*(myimg_extended.curlmat - blankimg_extended.curlmat), ext=ext_extended, vmax=0.01)


# In[151]:


#np.save('files/myimg_extended.npy', myimg_extended)


# In[169]:


convmat = 0.5*(myimg_extended.divmat - blankimg_extended.divmat)


# In[170]:


#np.save('files/convmat_extended_intonlyres_5to8_press.npy', convmat)


# ### Mask
# TODO: Figure out proper width of mask

# In[175]:


def isinmask(xpix, ypix, r, dr, pixsize, pixnum):
    # r is the einstein radius, and we take pixels within r +- dr
    # (sharp cutoff)
    npix = np.sqrt((xpix-pixnum/2)**2 + (ypix-pixnum/2)**2)
    pixdist = npix * pixsize
    return (r - dr < pixdist < r + dr)

def isinmask_smooth(xpix, ypix, r, dr, pixsize, pixnum):
    # r is the einstein radius, and we take pixels within r +- dr
    # gaussian smoothing
    npix = np.sqrt((xpix-pixnum/2)**2 + (ypix-pixnum/2)**2)
    pixdist = npix * pixsize
    return np.exp(-(pixdist-r)**2/(2*dr**2))


# In[178]:


mask_smooth = np.zeros((196,196))
for i in range(196):
    for j in range(196):
        mask_smooth[i,j] = isinmask_smooth(i,j, 1.0, 0.1, pixsize*2, pixnum-4)


# In[179]:


plt.imshow(mask_smooth, extent=[-ext_extended,ext_extended,ext_extended,-ext_extended])
plt.title('Smooth mask')
plt.colorbar()


# In[180]:


masked_convmat = convmat * mask_smooth


# In[181]:


autoshow(masked_convmat, ext=ext_extended, vmax=None)
plt.title('Masked convergence', size=14)
#plt.savefig('imgs/feb18_masked_convergence.png')


# In[214]:


mask_coverage = np.sum(mask_smooth) / len(mask_smooth)**2
print('mask coverage', mask_coverage)


# In[183]:


#np.save('files/masked_convmat_test2.npy', masked_convmat)


"""
