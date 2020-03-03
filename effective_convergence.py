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

from helpers import CustomImage, autoshow, ADD, sigma_cr


# In[25]:


# Basic parameters
zl = 0.5
zs = 1.
# pixnum, pixsize = 200, .008 # to match S. Birrer's paper (roughly)
# ext = pixnum * pixsize / 2.
pixnum = 200
ext = .8 # should be .8 to match Simon's paper
pixsize = 2*ext / pixnum
print(pixsize)


# In[26]:


## Calculate physical size of image
# ( note: 845000 kpc to z=.2 )
# distance = 1947000 # kpc to z=.5

cosmo = FlatLambdaCDM(H0=70, Om0=0.316)

com_l = cosmo.comoving_distance(zl).to(u.kpc).value
com_s = cosmo.comoving_distance(zs).to(u.kpc).value
angle = (2*ext) * np.pi/648000 # rad
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
print('integral (in kpc)', quad(lambda comz: double_cone_direct(comz, com_l, com_s)**2, 0, com_s))

double_cone_volume_kpc = com_width_kpc**2 * quad(lambda comz: double_cone_direct(comz, com_l, com_s)**2, 0, com_s)[0]
# volume of double-cone in kpc**3 (really double prism)

print('double prism volume', double_cone_volume_kpc / 1e6, 'million kpc^3')


# ## Substructure
# ### Interlopers

# Units: `massesn` has units of $M_\odot$. `massfunc` has units of 1/(Mpc/h)^3 / M_sun (comoving distance).

# In[29]:


# Load Sheth-Tormen results
h = .675
massesn = np.load('files/st_results/WIDE_massesn_ST.npy') # units solar masses
massfunc_matrix = np.load('files/st_results/WIDE_massfunc_ST.npy') * h**3/1000**3 # convert Mpc/h^3 to 1/kpc^3
zlist = np.load('files/st_results/WIDE_redshifts_ST.npy') # redshifts

massfunc = interp2d(zlist, massesn, massfunc_matrix, bounds_error=False) # function of (z, mass)


# In[30]:


def ps_integrand(mass):
    # rough so ignore redshift dependence
    return massfunc(zl, mass)

normalization = quad(lambda z: double_cone(z, zl, zs), 0, zs)[0]
def ps_integrand_fancy(mass):
    return (quad(lambda z: double_cone(z, zl, zs) * massfunc(z, mass), 0, zs)[0] 
            / normalization)


# In[31]:


#mass_bins_interlopers = [1e5, 1e6, 1e7, 1e8]
mass_bins_interlopers = np.logspace(5,8,20)[:-1]
bin_ratio = mass_bins_interlopers[1]/mass_bins_interlopers[0]
print('mass_bins_interlopers:',mass_bins_interlopers)
avg_nums_interlopers = np.array([double_cone_volume_kpc * quad(ps_integrand_fancy, mass, bin_ratio*mass)[0] for mass in mass_bins_interlopers])


# In[32]:


# plt.bar(np.log10(mass_bins_interlopers), avg_nums_interlopers, log=True, width=.1)
# plt.xlabel(r'Interloper mass bin ($\log_{10}$ in $M_\odot$)')
# plt.ylabel('Number in entire volume')
# print('number', np.sum(avg_nums_interlopers), ':', avg_nums_interlopers)

################################################################################
# ### Subhalos
# 
# Just a very rough estimate of the mass function:

# In[33]:


phys_width_kpc = 2 * ext * np.pi/648000  * ADD(0, zl).to(u.kpc).value
print('phys width (kpc):', phys_width_kpc)


# In[34]:


# einst_radius = 1 * np.pi/648000 * ADD(0,zl).to(u.kpc)
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


# In[37]:


# # Graph summing from top
# plt.loglog(mass_bins_interlopers,
#         [np.sum(avg_nums_interlopers[i:]) for i in range(len(avg_nums_interlopers))],
#         label='Interlopers (Press-Schechter)')
# plt.loglog(mass_bins_sub,
#         [np.sum(avg_nums_sub[i:]) for i in range(len(avg_nums_sub))],
#         label='Subhalos (Gilman, Birrer)')
# plt.loglog(mass_bins_sub,
#         [np.sum(avg_nums_sub2[i:]) for i in range(len(avg_nums_sub2))],
#         label='Subhalos (Ana)')
# plt.axhline(1, color='k')
# plt.xlabel('Minimum mass bin')
# plt.ylabel('Number $\\in(m, 10^{8})$ mass range in image')
# plt.legend()
# plt.title('Cumulative mass functions')
# #plt.savefig('imgs/feb18_massfunc_5to8.png')



# ### Sampling (helper functions)

# In[41]:


def double_cone_angle(z, zl=zl, zs=zs):
    # angle of how wide the interlopers can be dispersed so that they'd show up in the final image (according to the double-prism projection)
    
    # returns a ratio, 1 for z <= zl
    
    com_z = cosmo.comoving_distance(z)
    com_l = cosmo.comoving_distance(zl)
    return double_cone(z, zl=zl, zs=zs) * com_l / com_z

# Populate the subhalos
def do_subhalos():
    np.random.seed(145)
    xs, ys, redshifts, masses = [], [], [], []

    rv_nums = poisson.rvs(avg_nums_sub) if len(avg_nums_sub) > 1 else [poisson.rvs(avg_nums_sub)]
    xyext = ext
    for mass, num in zip(mass_bins_sub, rv_nums):
        for i in range(num):
            xs.append(np.random.uniform(-xyext,xyext))
            ys.append(np.random.uniform(-xyext,xyext))
            redshifts.append(zl)
            masses.append(mass)
    print('number of subhalos', len(xs))

    now = datetime.datetime.now()
    myimg_sub = CustomImage(xs, ys, redshifts, zl=zl, m=masses, pixnum=pixnum, pixsize=pixsize, mass_sheets=False)
    print('time to generate myimg_sub:', datetime.datetime.now()-now)
    now = datetime.datetime.now()
    myimg_sub.calc_div_curl_5pt()
    print('time to calc div curl for myimg_sub:', datetime.datetime.now()-now)

    blankimg = CustomImage([],[],[], zl=zl, pixnum=pixnum, pixsize=pixsize)
    autoshow(blankimg.image)
    blankimg.calc_div_curl_5pt();

    plt.close()
    autoshow(0.5*(myimg_sub.divmat - blankimg.divmat), ext=ext, vmax=.09)
    plt.title(r'$\kappa_{sub}$ (single plane, CDM)')
    plt.savefig('imgs/kappa_sub.png')

    np.save('files/kappa_sub.npy', 0.5*(myimg_sub.divmat-blankimg.divmat))

def do_naive_interlopers():
    np.random.seed(145)
    n_planes = 10

    xs, ys, redshifts, masses = [], [], [], []
    mass_sheets = []
    #xyext = ext

    # Interlopers!
    z_planes = np.linspace(.01,.99,n_planes)
    area_proportions = [double_cone(z)**2 for z in z_planes]
    area_sum = np.sum(area_proportions)

    for i, z_plane in enumerate(z_planes):
        rv_nums = poisson.rvs(avg_nums_interlopers * area_proportions[i] / area_sum)
        xyext = ext * double_cone_angle(z_plane) # `double_cone_angle` was commented out -- can't remember why
        for mass, num in zip(mass_bins_interlopers, rv_nums):
            for i in range(num):
                xs.append(np.random.uniform(-xyext,xyext))
                ys.append(np.random.uniform(-xyext,xyext))
                redshifts.append(zl)
                masses.append(mass)
                mass_sheets.append(True)
    print('number of interlopers', len(xs))


    myimg_proj = CustomImage(xs,ys,redshifts, zl=zl, m=masses,
                             pixnum=pixnum, pixsize=pixsize,
                             mass_sheets=mass_sheets, main_theta=1.0)
    myimg_proj.calc_div_curl_5pt()
    

    blankimg = CustomImage([], [], [], zl=zl, m=[], 
                           pixnum=pixnum, pixsize=pixsize, 
                           mass_sheets=[], main_theta=1.0)
    blankimg.calc_div_curl_5pt()


    # In[115]:


    plt.close()
    autoshow(0.5*(myimg_proj.divmat-blankimg.divmat), ext=ext, vmax=.09)
    plt.title(r'$\kappa_{sub}$ (multi-plane Born)')
    plt.savefig('imgs/kappa_intnaive.png')

    np.save('files/kappa_intnaive.npy', 0.5*(myimg_proj.divmat - blankimg.divmat))

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
    print('number of interlopers', len(xs))
    
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
    print('number of subhalos + interlopers', len(xs))

    myimg = CustomImage(xs, ys, redshifts, zl=zl, m=masses, 
                        pixnum=pixnum, pixsize=pixsize,
                        mass_sheets=mass_sheets, main_theta=theta)
    myimg.calc_div_curl_5pt()

    blankimg = CustomImage([], [], [], zl=zl, m=[],
                           pixnum=pixnum, pixsize=pixsize, 
                           mass_sheets=[], main_theta=theta)
    blankimg.calc_div_curl_5pt()

    # Save results #
    
    plt.close()
    autoshow(0.5*(myimg.divmat-blankimg.divmat), ext=ext, vmax=.09)
    plt.title(r'$\kappa_{sub}$ (multi-plane Born)')
    plt.savefig('imgs/kappa_full_theta{}.png'.format(theta))

    np.save('files/kappa_full_theta{}.npy'.format(theta), 0.5*(myimg.divmat - blankimg.divmat))
    
# do_subhalos()
# do_naive_interlopers()
do_full(1)
    
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
