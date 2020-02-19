"""
helpers.py
----------
Functionality:
`DefaultImage` : easy way to generate a truth-level image.
"""



# some standard python imports #
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.cm as cms
#%matplotlib inline

from lenstronomy.LensModel.lens_model import LensModel
# from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
# import lenstronomy.Plots.output_plots as lens_plot
from lenstronomy.LightModel.light_model import LightModel
import lenstronomy.Util.param_util as param_util
# from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.psf import PSF
# import lenstronomy.Util.image_util as image_util
# from lenstronomy.Workflow.fitting_sequence import FittingSequence

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

from multiprocessing import Pool

def ADD(z1,z2):
    ## This is a function that computes the angular diameter distance
    ## between two redshifts z1 and z2.
    cosmo = FlatLambdaCDM(H0=70, Om0=0.316) 
    return cosmo.angular_diameter_distance_z1z2(z1,z2)

def comdist(z):
    # Function computes the comoving distance to the given redshift
    cosmo = FlatLambdaCDM(H0=70, Om0=0.316)
    return cosmo.comoving_distance(z)

def sigma_cr(zd,zs):
    ## This function calculates the critical surface mass density at
    ## redshift zd, relative to the source redshift zs.
    const = 1.663e18*u.M_sun / u.Mpc##c^2/(4 pi G)
    return const*(ADD(0,zs)/(ADD(zd,zs)*ADD(0,zd))) ##in units Msun/Mpc^2

def gfunc(c):
    ## This is the g(c) function that is defined
    ## commonly in NFW profiles.
    # TODO: check that this still works with tNFW
    a = np.log(1.+c) - (c/(1.+c))
    return 1./a

def rs_angle(zd,rs): 
    ##takes in interloper redshift, gives you the scale redius in angular units
    # TODO: check that this still works with tNFW
    Dd = ADD(0,zd)
    rs_mpc = rs*u.Mpc
    return ((1./4.848e-6)*rs_mpc)/Dd ##gives in arcsec

def alpha_s(m,rs,zd,zs):
    ##takes in subhalo mass, scale radius, interloper redshift, source redshift
    ##returns the angular deflection at scale radius
    # TODO: check that this still works with tNFW
    m_msun = m*u.M_sun
    rs_mpc = rs*u.Mpc
    con = (1./np.pi)*gfunc(200.)*(1.-np.log(2))
    return (con*m_msun)/((rs_mpc**2.)*sigma_cr(zd,zs))

def alpha_s_tnfw(m,rs,zd,zs,tau):
    m_phys = m*u.M_sun # physical mass
    mass_factor = tau**2 / (tau**2+1)**2 * ((tau**2-1)*np.log(tau) + tau*np.pi - (tau**2+1))
    # print('not correct, but ignoring mass factor (closer to correct measured convergence)')
    # mass_factor = 1

    m_nfw = m_phys / mass_factor
    # the following code is supposed to match how lenstronomy works, but not necessarily our old `alpha_s`
    distl = ADD(0,zd).to(u.kpc)

    rs_phys = (rs * u.Mpc).to(u.kpc)
    rs_ang = (rs_phys / distl) * 648000/np.pi # arcsec

    rho0 = m_nfw / (2*rs_ang*rs_phys**2 * 2*np.pi) # this formula came from comparing formula 32 in Ana's paper with the function `density_2d` in lenstronomy
    
    alpha_rs = rho0 * 4 * rs_ang**2 * (1 - np.log(2)) # this formula came from Simon's code, where rs was in arcsec
    # todo check that we handle correctly whether rs is angular (arcsec) or physical
    # print('rs', rs)
    # print('rs_phys', rs_phys)
    # print('rs_ang', rs_ang)

    # print('m_phys', m_phys)


    # print('mass_factor', mass_factor)
    # print('m_nfw', m_nfw)
    # print('rho0', rho0)
    # print('alpha_rs', alpha_rs)
    # print('sigma_cr', sigma_cr(zd,zs).to(u.Msun/u.kpc**2))
    # print('final answer', (alpha_rs / sigma_cr(zd,zs)).si)
    
    return (alpha_rs / sigma_cr(zd,zs)).si
    
def k_ext(N,m,A,zd,zs,pixsize):
    ## FOR NOW THIS IS SET TO ZERO BECAUSE I CAN'T GET IT TO WORK
    m_msun = m*u.M_sun
    A_mpc2 = (pixsize**4)*(ADD(0.,zd)**2.)*A*((4.848e-6)**2.)  ##arcsec conversion
    return 0.##-(N*m_msun)/(A_mpc2*sigma_cr(zd,zs))


def xi_to_x(xi,z):
    ##takes in physical coordinates (Mpc), turns it into angular coordinates (arcsec)
    x = np.array(((xi*u.Mpc)/ADD(0.,z))/4.848e-6)
    y = x.astype(np.float)
    return y

def x_to_xi(x,z):
    ##takes in angular coordinates (arcsec), turns it into physical coordinates (Mpc)
    return ((x*4.848e-6)*ADD(0,z))/u.Mpc

def xi_to_pix(xi,z,pixsize,pixnum):
    ## takes in physical coordinates (Mpc), identifies the appropriate pixel number
    return (xi_to_x(xi,z))/pixsize + pixnum/2.
def inv_rs_angle(zd,rs_angle):
    ## takes in the rs angle in arcsec gives rs in in MPC
    # TODO: check that this still works with tNFW
    Dd = ADD(0,zd)
    return 4.848e-6*Dd*rs_angle


def inv_alpha_s(alpha_s,rs,zd,zs):
    ## takes in subhalo angular deflection at scale radius, scale radius,
    ## interloper redshift and source redshift and returns interloper mass
    # TODO: check that this still works with tNFW
    rs_mpc = rs*u.Mpc
    con = (1./np.pi)*gfunc(200.)*(1.-np.log(2))
    return (alpha_s/con)*((rs_mpc**2.)*sigma_cr(zd,zs))

def get_mass_back(rsang, alphars, zd, zs):
    # TODO: make sure this is actually correct!!
    # TODO: check that this still works with tNFW
    '''Starts with NFW rsang and alphars.
    Returns original NFW mass (out to R200?) in Msun.'''
    rs = inv_rs_angle(zd, rsang).to(u.Mpc).value
    mass = inv_alpha_s(alphars, rs, zd, zs).to(u.Msun)
    return mass.to(u.Msun).value

def autoshow(image, vmax=None, ext=None):
    # helper function like imshow but gets the colors centered at zero

    if vmax == None:
        vmin = np.min(image)
        vmax = np.max(image)
        vmin = min(vmin, -vmax)
        vmax = max(vmax, -vmin)
    else:
        vmin = -vmax

    extent = None if ext==None else [-ext,ext,ext,-ext]
    plt.imshow(image, vmin=vmin, vmax=vmax, cmap='seismic', extent=extent)
    plt.colorbar()

def measure_mass(convmat, zl, zs, ext):
    # measure mass from convergence matrix
    # ext is half the width of the image in arcsec
    phys_width = 2*ext * np.pi/648000 * ADD(0, zl).to(u.kpc)
    #print('phys width', phys_width)
    pixnum = len(convmat) # might be off by -4 depending on method but whatever
    pixsize_phys = phys_width / pixnum
    twod_integral_conv = np.sum(convmat) * pixsize_phys**2
    return twod_integral_conv * sigma_cr(zl, zs).to(u.Msun/u.kpc**2)    
    
"""
class DefaultImage:
    def __init__(self, N, seed=333, zl=0.2, zd=0.2, zs=1.0, near_ring=False):
        self.N = N
        self.zl = zl
        self.zd = zd
        self.zs = zs
        self.near_ring = near_ring
        
        np.random.seed(seed)

        # ## REDSHIFTS #######################################################################################
        # Nit = 100 ##Number of different redshifts
        # zds = np.linspace(0.01,0.99,Nit)
        # ####################################################################################################



        ## SOURCE PROPERTIES ###############################################################################
        r_sersic_source = 10.0
        e1s, e2s = param_util.phi_q2_ellipticity(phi=0.8, q=0.2)
        beta_ras, beta_decs = [1.7],[0.3]#this is the source position on the source plane

        n_sersic_source = 1.5

        ## SOURCE-CLUMP PROPERTIES #########################################################################
        r_sersic_source_clumps = 1/3.
        N_clump = 0
        clumprandx = np.random.rand(N_clump)
        clumprandy = np.random.rand(N_clump)

        source_scatter = 1. ## This is how wide the scatter of the clumps over the smooth source

        n_sersic_source_clumps = 1.5

        ####################################################################################################



        ## LENS PROPERTIES #################################################################################
        theta_lens = 10.
    #     zl = 0.2
        r_theta_lens = x_to_xi(theta_lens,zl)
        e1, e2 = param_util.phi_q2_ellipticity(phi=-0.9, q=0.8)
        gamma = 2.

        center_lens_x, center_lens_y = 0.,0.
        ####################################################################################################



        ## IMAGE PROPERTIES ################################################################################
        pixsize = 0.2
        ####################################################################################################



        ## INTERLOPER PROPERTIES ########################################################################### 
    #     N = 1 ##Number of perturbers
        M = 1 ##Averaging different realizations

        disc_size = 2. ##  interlopers are randomly distributed to a disk that is this
                       ##  this times bigger than the einstein radius of the lens
        # Perturbers are uniformly distributed within a disk of radius `disc_size * r_theta_lens`
        ## r2s = ... ##
        if near_ring:
            # TODO: figure out placement to put the halo near the ring.
            # First, figure out what radius we should put the subhalo at
            if zd <= zl: # subhalo close (simple)
                r_aim = x_to_xi(theta_lens, zd)
            else: # subhalo far
                chil = ADD(0,zl)*(1+zl)
                chid = ADD(0,zd)*(1+zd)
                chis = ADD(0,zs)*(1+zs)
                r_aim = (chis-chid)/(chis-chil)*r_theta_lens
                
            r2low = (r_aim * 0.9)**2
            r2high = (r_aim * 1.1)**2
            r2s = np.random.uniform(r2low, r2high, size=(N,M)) # TODO: test this
        else:
            r2s = ((disc_size*r_theta_lens)**2.)*(np.random.rand(N,M))

        rss = np.sqrt(r2s)
        theta_p = 2.*np.pi*(np.random.rand(N,M))
        self.xs = rss*np.cos(theta_p)
        self.ys = rss*np.sin(theta_p)
    #     xpixs = np.zeros([Nit,N,M]) # will add pixel values in the next cell
    #     ypixs = np.zeros([Nit,N,M]) #
        ####################################################################################################

        #j = 19 # arbitrary choice (loop over redshifts) (19 is where zd \approx zl)
        k = 0 # also arbitrary choice (just pick the first statistic)

        beta_ra, beta_dec = beta_ras[0], beta_decs[0]

    #     xpixs[j] = xi_to_pix(self.xs,zds[j],pixsize,200)   ## AT THAT REDSHIFT CALCULATING THE INTERLOPER
    #     ypixs[j] = xi_to_pix(self.ys,zds[j],pixsize,200)   ## POSITIONS. (THEY ARE RANDOMLY GENERATED IN THE EARLIER BOX)

        m = 1.0e9 # mass of interlopers (used to be 1e7)
        #zs = 1.
        #zd = zds[j] # interloper redshift
        rs = 0.001  # interloper scale radius r_s
        A = 80**2 ## in arcsec ## IGNORE THIS, THIS WAS FOR NEGATIVE CONVERGENCE

        kext = float(k_ext(N,m,A,zl,zs,pixsize))
        self.rsang = float(rs_angle(zd,rs))
        self.alphars = float(alpha_s(m,rs,zd,zs))

        ## Setting lens_model_list and redshift_list
        lens_model_main = ['SPEP']
        lens_model_interlopers = ['CONVERGENCE']+['TNFW' for i in range(N)]
        redshift_main = [zl]
        redshift_interlopers = [zd]+[zd for i in range(N)]
        # (unfortunately, we need to give the redshifts in increasing order, so we have two cases)
        if zl >= zd:
            lens_model_list = lens_model_interlopers + lens_model_main
            redshift_list = redshift_interlopers + redshift_main
        else:
            lens_model_list = lens_model_main + lens_model_interlopers
            redshift_list = redshift_main + redshift_interlopers

        self.lens_model_mp = LensModel(lens_model_list=lens_model_list,
                                 z_source=zs,
                                 lens_redshift_list=redshift_list, 
                                 multi_plane=True)

        self.kwargs_spep = {'theta_E': theta_lens, 'e1': e1, 'e2': e2, 
                       'gamma': gamma, 'center_x': center_lens_x, 'center_y': center_lens_y}
        kwargs_conv ={'kappa_ext': kext}

        ### NFW kwargs for the interlopers
        kwargs_main_lens = [self.kwargs_spep]
        kwargs_interlopers = [kwargs_conv] # (+ will append interlopers)
        for i in range(N):
            center_nfw_x = xi_to_x(self.xs[i,k],zd)
            center_nfw_y = xi_to_x(self.ys[i,k],zd)

            kwargs_nfw = {'Rs':self.rsang,'alpha_Rs':self.alphars,
                          'r_trunc':20*self.rsang, # we'll stick with 20 for now
                          'center_x': center_nfw_x, 'center_y': center_nfw_y}
            kwargs_interlopers.append(kwargs_nfw)

        # (again, need to sort by redshift)
        if zl >= zd:
            self.kwargs_lens = kwargs_interlopers + kwargs_main_lens
        else:
            self.kwargs_lens = kwargs_main_lens + kwargs_interlopers

        ########################################################################
        # set up the list of light models to be used #

        # SOURCE light
        source_light_model_list = ['SERSIC_ELLIPSE']
        for i in range(N_clump):
            source_light_model_list.append('SERSIC')

        self.light_model_source = LightModel(light_model_list = source_light_model_list)

        # LENS light
        lens_light_model_list = ['SERSIC_ELLIPSE']
        self.light_model_lens = LightModel(light_model_list = lens_light_model_list)

        # SOURCE light kwargs
        self.kwargs_light_source = [{'amp': 1000., 'R_sersic': r_sersic_source, 'n_sersic': n_sersic_source, 
                              'e1': e1s, 'e2': e2s, 'center_x': beta_ra , 'center_y': beta_dec}]
        for i in range(N_clump):
            self.kwargs_light_source.append({'amp': 1000, 'R_sersic': r_sersic_source_clumps, 'n_sersic': n_sersic_source_clumps,
                                        'center_x': beta_ra+source_scatter*(clumprandx[i]-.5), 
                                        'center_y': beta_dec+source_scatter*(clumprandy[i]-.5)})

        # LENS light kwargs
        self.kwargs_light_lens = [{'amp': 1500, 'R_sersic': theta_lens, 'n_sersic': gamma, 
                              'e1': e1, 'e2': e2, 'center_x': center_lens_x , 'center_y': center_lens_y}]

        # evaluate surface brightness at a specific position #
        #flux = self.light_model_lens.surface_brightness(x=1, y=1, kwargs_list=self.kwargs_light_lens)

        deltaPix = pixsize ###aLSO PIXSIze size of pixel in angular coordinates #

        # setup the keyword arguments to create the Data() class #
        ra_at_xy_0, dec_at_xy_0 = -20, -20 # coordinate in angles (RA/DEC) at the position of the pixel edge (0,0)
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * deltaPix  # linear translation matrix of a shift in pixel in a shift in coordinates
        kwargs_pixel = {'nx': 200, 'ny': 200,  # number of pixels per axis
                        'ra_at_xy_0': ra_at_xy_0,  # RA at pixel (0,0)
                        'dec_at_xy_0': dec_at_xy_0,  # DEC at pixel (0,0)
                        'transform_pix2angle': transform_pix2angle} 
        self.pixel_grid = PixelGrid(**kwargs_pixel)
        
        # return the list of pixel coordinates #
        #x_coords, y_coords = self.pixel_grid.pixel_coordinates
        # compute pixel value of a coordinate position #
        #x_pos, y_pos = self.pixel_grid.map_coord2pix(ra=0, dec=0)
        # compute the coordinate value of a pixel position #
        #ra_pos, dec_pos = self.pixel_grid.map_pix2coord(x=20, y=10)

        # import the PSF() class #

        kwargs_psf = {'psf_type': 'GAUSSIAN',  # type of PSF model (supports 'GAUSSIAN' and 'PIXEL')
                      'fwhm': 0.01,  # full width at half maximum of the Gaussian PSF (in angular units)
                      'pixel_size': deltaPix  # angular scale of a pixel (required for a Gaussian PSF to translate the FWHM into a pixel scale)
                     }
        self.psf = PSF(**kwargs_psf)
        # return the pixel kernel correspon
        kernel = self.psf.kernel_point_source

        ####################################################################################

        # import the ImageModel class #

        # define the numerics #
        self.kwargs_numerics = {'supersampling_factor': 1, # each pixel gets super-sampled (in each axis direction) 
                          'supersampling_convolution': False}
        # initialize the Image model class by combining the modules we created above #
        imageModel = ImageModel(data_class=self.pixel_grid, psf_class=self.psf,
                                lens_model_class=self.lens_model_mp,
                                source_model_class=self.light_model_source,
                                lens_light_model_class=self.light_model_lens,
                                kwargs_numerics=self.kwargs_numerics)
        # simulate image with the parameters we have defined above #
        self.image = imageModel.image(kwargs_lens=self.kwargs_lens,
                                      kwargs_source=self.kwargs_light_source,
                                      kwargs_lens_light=self.kwargs_light_lens)#, kwargs_ps=kwargs_ps)

"""

"""
Below is the old version of the class CustomImage:

class CustomImage:
    # For now, we'll still assume only one interloper plane.
    
    def __init__(self, xpos_list, ypos_list, zl=0.2, zd=0.2, zs=1.0):
        assert(len(xpos_list) == len(ypos_list))

        self.xpos_list = xpos_list
        self.ypos_list = ypos_list        
        self.N = len(xpos_list) # just for convenience
        N = self.N
        self.zl = zl
        self.zd = zd
        self.zs = zs       
        
        ## SOURCE PROPERTIES ###############################################################################
        r_sersic_source = 10.0
        e1s, e2s = param_util.phi_q2_ellipticity(phi=0.8, q=0.2)
        beta_ras, beta_decs = [1.7],[0.3]#this is the source position on the source plane

        n_sersic_source = 1.5

        ## SOURCE-CLUMP PROPERTIES #########################################################################
        r_sersic_source_clumps = 1/3.
        N_clump = 0
        clumprandx = np.random.rand(N_clump)
        clumprandy = np.random.rand(N_clump)

        source_scatter = 1. ## This is how wide the scatter of the clumps over the smooth source

        n_sersic_source_clumps = 1.5

        ####################################################################################################



        ## LENS PROPERTIES #################################################################################
        theta_lens = 10.
    #     zl = 0.2
        r_theta_lens = x_to_xi(theta_lens,zl)
        e1, e2 = param_util.phi_q2_ellipticity(phi=-0.9, q=0.8)
        gamma = 2.

        center_lens_x, center_lens_y = 0.,0.
        ####################################################################################################



        ## IMAGE PROPERTIES ################################################################################
        pixsize = 0.2
        ####################################################################################################



        ## INTERLOPER PROPERTIES ########################################################################### 
        # N = 1 ##Number of perturbers
        # M = 1 ##Averaging different realizations

        # disc_size = 2. ##  interlopers are randomly distributed to a disk that is this
        #                ##  this times bigger than the einstein radius of the lens
        # # Perturbers are uniformly distributed within a disk of radius `disc_size * r_theta_lens`
        # ## r2s = ... ##
        # if near_ring:
        #     # TODO: figure out placement to put the halo near the ring.
        #     # First, figure out what radius we should put the subhalo at
        #     if zd <= zl: # subhalo close (simple)
        #         r_aim = x_to_xi(theta_lens, zd)
        #     else: # subhalo far
        #         chil = ADD(0,zl)*(1+zl)
        #         chid = ADD(0,zd)*(1+zd)
        #         chis = ADD(0,zs)*(1+zs)
        #         r_aim = (chis-chid)/(chis-chil)*r_theta_lens
                
        #     r2low = (r_aim * 0.9)**2
        #     r2high = (r_aim * 1.1)**2
        #     r2s = np.random.uniform(r2low, r2high, size=(N,M)) # TODO: test this
        # else:
        #     r2s = ((disc_size*r_theta_lens)**2.)*(np.random.rand(N,M))

        # rss = np.sqrt(r2s)
        # theta_p = 2.*np.pi*(np.random.rand(N,M))
        # self.xs = rss*np.cos(theta_p)
        # self.ys = rss*np.sin(theta_p)

        #     xpixs = np.zeros([Nit,N,M]) # will add pixel values in the next cell
        #     ypixs = np.zeros([Nit,N,M]) #

        pixnum = 200

        # for easier plotting only:
        self.plot_xpixs = [xi_to_pix(x_to_xi(xpos, zl), zl,pixsize,pixnum) for xpos in xpos_list]
        self.plot_ypixs = [xi_to_pix(x_to_xi(ypos, zl), zl,pixsize,pixnum) for ypos in ypos_list]

    
        ####################################################################################################

        #j = 19 # arbitrary choice (loop over redshifts) (19 is where zd \approx zl)
        k = 0 # also arbitrary choice (just pick the first statistic)

        beta_ra, beta_dec = beta_ras[0], beta_decs[0]

    #     xpixs[j] = xi_to_pix(self.xs,zds[j],pixsize,200)   ## AT THAT REDSHIFT CALCULATING THE INTERLOPER
    #     ypixs[j] = xi_to_pix(self.ys,zds[j],pixsize,200)   ## POSITIONS. (THEY ARE RANDOMLY GENERATED IN THE EARLIER BOX)

        m = 1.0e7 # mass of interlopers (used to be 1e7, and then 1e9)
        #zs = 1.
        #zd = zds[j] # interloper redshift
        rs = 0.001  # interloper scale radius r_s
        A = 80**2 ## in arcsec ## IGNORE THIS, THIS WAS FOR NEGATIVE CONVERGENCE

        kext = float(k_ext(N,m,A,zl,zs,pixsize))
        self.rsang = float(rs_angle(zd,rs))
        self.alphars = float(alpha_s(m,rs,zd,zs))

        ## Setting lens_model_list and redshift_list
        lens_model_main = ['SPEP']
        lens_model_interlopers = ['CONVERGENCE']+['TNFW' for i in range(N)]
        redshift_main = [zl]
        redshift_interlopers = [zd]+[zd for i in range(N)]
        # (unfortunately, we need to give the redshifts in increasing order, so we have two cases)
        if zl >= zd:
            lens_model_list = lens_model_interlopers + lens_model_main
            redshift_list = redshift_interlopers + redshift_main
        else:
            lens_model_list = lens_model_main + lens_model_interlopers
            redshift_list = redshift_main + redshift_interlopers

        self.lens_model_mp = LensModel(lens_model_list=lens_model_list,
                                 z_source=zs,
                                 lens_redshift_list=redshift_list, 
                                 multi_plane=True)

        self.kwargs_spep = {'theta_E': theta_lens, 'e1': e1, 'e2': e2, 
                       'gamma': gamma, 'center_x': center_lens_x, 'center_y': center_lens_y}
        kwargs_conv ={'kappa_ext': kext}

        ### NFW kwargs for the interlopers
        kwargs_main_lens = [self.kwargs_spep]
        kwargs_interlopers = [kwargs_conv] # (+ will append interlopers)
        for i in range(N):
            center_nfw_x = xpos_list[i] # xi_to_x(self.xs[i,k],zd)
            center_nfw_y = ypos_list[i] # xi_to_x(self.ys[i,k],zd)

            kwargs_nfw = {'Rs':self.rsang,'alpha_Rs':self.alphars,
                          'r_trunc':20*self.rsang, # we'll stick with 20 for now
                          'center_x': center_nfw_x, 'center_y': center_nfw_y}
            kwargs_interlopers.append(kwargs_nfw)

        # (again, need to sort by redshift)
        if zl >= zd:
            self.kwargs_lens = kwargs_interlopers + kwargs_main_lens
        else:
            self.kwargs_lens = kwargs_main_lens + kwargs_interlopers

        ########################################################################
        # set up the list of light models to be used #

        # SOURCE light
        source_light_model_list = ['SERSIC_ELLIPSE']
        for i in range(N_clump):
            source_light_model_list.append('SERSIC')

        self.light_model_source = LightModel(light_model_list = source_light_model_list)

        # LENS light
        lens_light_model_list = ['SERSIC_ELLIPSE']
        self.light_model_lens = LightModel(light_model_list = lens_light_model_list)

        # SOURCE light kwargs
        self.kwargs_light_source = [{'amp': 1000., 'R_sersic': r_sersic_source, 'n_sersic': n_sersic_source, 
                              'e1': e1s, 'e2': e2s, 'center_x': beta_ra , 'center_y': beta_dec}]
        for i in range(N_clump):
            self.kwargs_light_source.append({'amp': 1000, 'R_sersic': r_sersic_source_clumps, 'n_sersic': n_sersic_source_clumps,
                                        'center_x': beta_ra+source_scatter*(clumprandx[i]-.5), 
                                        'center_y': beta_dec+source_scatter*(clumprandy[i]-.5)})

        # LENS light kwargs
        self.kwargs_light_lens = [{'amp': 1500, 'R_sersic': theta_lens, 'n_sersic': gamma, 
                              'e1': e1, 'e2': e2, 'center_x': center_lens_x , 'center_y': center_lens_y}]

        # evaluate surface brightness at a specific position #
        #flux = self.light_model_lens.surface_brightness(x=1, y=1, kwargs_list=self.kwargs_light_lens)

        deltaPix = pixsize ###aLSO PIXSIze size of pixel in angular coordinates #

        # setup the keyword arguments to create the Data() class #
        ra_at_xy_0, dec_at_xy_0 = -20, -20 # coordinate in angles (RA/DEC) at the position of the pixel edge (0,0)
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * deltaPix  # linear translation matrix of a shift in pixel in a shift in coordinates
        kwargs_pixel = {'nx': 200, 'ny': 200,  # number of pixels per axis
                        'ra_at_xy_0': ra_at_xy_0,  # RA at pixel (0,0)
                        'dec_at_xy_0': dec_at_xy_0,  # DEC at pixel (0,0)
                        'transform_pix2angle': transform_pix2angle} 
        self.pixel_grid = PixelGrid(**kwargs_pixel)
        
        # return the list of pixel coordinates #
        #x_coords, y_coords = self.pixel_grid.pixel_coordinates
        # compute pixel value of a coordinate position #
        #x_pos, y_pos = self.pixel_grid.map_coord2pix(ra=0, dec=0)
        # compute the coordinate value of a pixel position #
        #ra_pos, dec_pos = self.pixel_grid.map_pix2coord(x=20, y=10)

        # import the PSF() class #

        kwargs_psf = {'psf_type': 'GAUSSIAN',  # type of PSF model (supports 'GAUSSIAN' and 'PIXEL')
                      'fwhm': 0.01,  # full width at half maximum of the Gaussian PSF (in angular units)
                      'pixel_size': deltaPix  # angular scale of a pixel (required for a Gaussian PSF to translate the FWHM into a pixel scale)
                     }
        self.psf = PSF(**kwargs_psf)
        # return the pixel kernel correspon
        kernel = self.psf.kernel_point_source

        ####################################################################################

        # import the ImageModel class #

        # define the numerics #
        self.kwargs_numerics = {'supersampling_factor': 1, # each pixel gets super-sampled (in each axis direction) 
                          'supersampling_convolution': False}
        # initialize the Image model class by combining the modules we created above #
        imageModel = ImageModel(data_class=self.pixel_grid, psf_class=self.psf,
                                lens_model_class=self.lens_model_mp,
                                source_model_class=self.light_model_source,
                                lens_light_model_class=self.light_model_lens,
                                kwargs_numerics=self.kwargs_numerics)
        # simulate image with the parameters we have defined above #
        self.image = imageModel.image(kwargs_lens=self.kwargs_lens,
                                      kwargs_source=self.kwargs_light_source,
                                      kwargs_lens_light=self.kwargs_light_lens)#, kwargs_ps=kwargs_ps)
"""

class CustomImage:

    def x_to_pix(self, x, z=None):
        # from x to pixel on the lens plane
        # todo: rewrite so that it works with any redshift (using double-cone projection)
        if z == None:
            z = self.zl
        return xi_to_pix(x_to_xi(x, z), self.zl, self.pixsize, self.pixnum)

    def double_cone_width(self, z):
        # Return comoving width in kpc

        # First, we calculate the angular extent of this image. Using the
        # comoving distance, we then calculate the comoving width at the widest
        # point of the double-cone (widest in comoving distance, at least).
        view_angle = self.pixsize*self.pixnum * np.pi/648000 # in radians
        lens_width_com = view_angle * comdist(self.zl) # flat-space trick

        com_z = comdist(z)
        com_l = comdist(self.zl)
        com_s = comdist(self.zs)
        
        if z < self.zl:
            width = (com_z / com_l) * lens_width_com
        else:
            width = (com_s - com_z)/(com_s - com_l) * lens_width_com

        return width.to(u.kpc).value
    
    def __init__(self, xpos_list, ypos_list, redshift_list, m=None, zl=0.2, zs=1.0, pixsize=0.2, pixnum=200, mass_sheets=None, main_theta=0.3):
        # change: used to take in `m` as a single mass for all interlopers, but
        # now this can also be a list of masses
        
        # `mass_sheets` : default set to True, meaning we should add negative mass sheets to cancel out all the substructure
        
        assert(len(xpos_list) == len(ypos_list))
        assert(len(xpos_list) == len(redshift_list))

        self.xpos_list = xpos_list
        self.ypos_list = ypos_list
        self.redshift_list = redshift_list
        self.N = len(xpos_list) # number of interlopers+subhalos
        N = self.N
        self.zl = zl
        self.zs = zs
        if m is None:
            self.mass_list = [1e7] * self.N
        elif isinstance(m, float):
            self.mass_list = [m] * self.N
        else:
            self.mass_list = m

        self.pixsize = pixsize
        self.pixnum = pixnum

        if isinstance(mass_sheets, list) or isinstance(mass_sheets, np.ndarray):
            self.mass_sheets = mass_sheets
        elif mass_sheets is None or mass_sheets is True:
            self.mass_sheets = [True for _ in range(N)]
        elif mass_sheets is False:
            self.mass_sheets = [False for _ in range(N)]
        any_mass_sheets = np.any(self.mass_sheets)
            
        self.main_theta = main_theta
        
        ## SOURCE PROPERTIES ###############################################################################
        # r_sersic_source = 10.0
        # e1s, e2s = param_util.phi_q2_ellipticity(phi=0.8, q=0.2)
        # beta_ras, beta_decs = [1.7],[0.3]#this is the source position on the source plane
        # let's mess with these parameters a little
        r_sersic_source = .5
        e1s, e2s = param_util.phi_q2_ellipticity(phi=0.5, q=0.3)
        beta_ras, beta_decs = [.01],[.02]#this is the source position on the source plane

        n_sersic_source = 1.5

        ## SOURCE-CLUMP PROPERTIES #########################################################################
        r_sersic_source_clumps = 1/3.
        N_clump = 0
        clumprandx = np.random.rand(N_clump)
        clumprandy = np.random.rand(N_clump)

        source_scatter = 1. ## This is how wide the scatter of the clumps over the smooth source

        n_sersic_source_clumps = 1.5

        ####################################################################################################



        ## LENS PROPERTIES #################################################################################
        theta_lens = self.main_theta # used to be 10.
        r_theta_lens = x_to_xi(theta_lens,zl)
        e1, e2 = param_util.phi_q2_ellipticity(phi=-0.9, q=0.8)
        gamma = 2.

        center_lens_x, center_lens_y = 0.,0.
        ####################################################################################################



        ## IMAGE PROPERTIES ################################################################################
        # self.pixsize = 0.2
        # self.pixnum = 200
        ####################################################################################################



        ## INTERLOPER PROPERTIES ########################################################################### 

        # for easier plotting only (current version only works when all the interlopers are at the lens redshift):
        # self.plot_xpixs = [self.x_to_pix(xpos) for xpos in xpos_list]
        # self.plot_ypixs = [self.y_to_pix(ypos), zl,pixsize,pixnum) for ypos in ypos_list]

        beta_ra, beta_dec = beta_ras[0], beta_decs[0]

        # self.m = 1.0e7 # mass of interlopers (used to be 1e7, and then 1e9)
        # self.rs = 0.001  # interloper scale radius r_s
        # A = 80**2 ## in arcsec ## IGNORE THIS, THIS WAS FOR NEGATIVE CONVERGENCE
        self.rs = 1e-4 # Mpc (pivot around m0=1e6)
        
        # kext = float(k_ext(N,m,A,zl,zs,pixsize))
        # note that there is no more self.rsang or self.alphars

        ## LENS model and redshifts
        # First we make a dictionary of convergence sheet masses
        convergence_sheet_masses = {z:0 for z in self.redshift_list}
        
        # In the unsorted list, we'll put the main lens first
        lens_model_unsorted = ['SPEP'] + ['TNFW' for i in range(N)] + (['CONVERGENCE' for _ in convergence_sheet_masses]
                                                                       if any_mass_sheets else [])
        redshifts_unsorted = [self.zl] + list(self.redshift_list) + (sorted(convergence_sheet_masses.keys())
                                                                     if any_mass_sheets else [])

        # Then we sort everything
        sort_idx = np.argsort(redshifts_unsorted)
        lens_model_sorted = [lens_model_unsorted[i] for i in sort_idx]
        redshifts_sorted = [redshifts_unsorted[i] for i in sort_idx]

        self.main_lens_idx = np.where(sort_idx == 0)[0][0] # which lens is the main lens?

        self.lens_model_mp = LensModel(lens_model_list=lens_model_sorted,
                                       z_source = self.zs,
                                       lens_redshift_list=redshifts_sorted,
                                       multi_plane=True)
        
        # LENS kwargs
        self.kwargs_spep = {'theta_E': theta_lens, 'e1': e1, 'e2': e2, 
                            'gamma': gamma, 'center_x': center_lens_x, 'center_y': center_lens_y}

        kwargs_unsorted = [self.kwargs_spep] # (+ will append more)
        #
        for i in range(N): # (append interlopers)
            center_nfw_x = xpos_list[i]
            center_nfw_y = ypos_list[i]

            tau = 20 # assume 20 as default

            rs_adjusted = self.rs * (self.mass_list[i]/1e6)**(1/3.) # adjusted according to physical mass
            
            rsang = float(rs_angle(self.redshift_list[i],rs_adjusted))
            alphars = float(alpha_s_tnfw(self.mass_list[i],rs_adjusted,self.redshift_list[i],zs,tau))
            # alphars = float(alpha_s(self.mass_list[i],self.rs,self.redshift_list[i],zs)) # old result

            kwargs_nfw = {'Rs':rsang, 'alpha_Rs':alphars,
                          'r_trunc':tau*rsang,
                          'center_x': center_nfw_x, 'center_y': center_nfw_y}
            kwargs_unsorted.append(kwargs_nfw)
        #
        if any_mass_sheets: # (append negative convergence sheets)
            for i in range(N):
                if self.mass_sheets[i]:
                    convergence_sheet_masses[self.redshift_list[i]] += self.mass_list[i]
            for z, m in sorted(convergence_sheet_masses.items()):
                area_com = self.double_cone_width(z)**2 # kpc**2 comoving
                area = area_com / (1+z)**2 # kpc**2 physical
                sig = m/area # Msun / kpc**2

                # print('showing work')
                # print('z', z, 'm', m)
                # print('area_com', area_com)
                # print('area', area)
                # print('sig', sig)
                
                # our normalization is the formula from assuming that this redshift
                # is the only lens
                sig_cr = sigma_cr(z, self.zs).to(u.Msun/u.Mpc**2).value / 1000**2 # from Msun/Mpc**2 to Msun/kpc**2

                kwargs_convergence_sheet = {'kappa_ext': -sig/sig_cr} # todo check this calculation
                kwargs_unsorted.append(kwargs_convergence_sheet)

        self.kwargs_lens = [kwargs_unsorted[i] for i in sort_idx]
        
        ########################################################################
        # set up the list of light models to be used #

        # SOURCE light
        source_light_model_list = ['SERSIC_ELLIPSE']
        for i in range(N_clump):
            source_light_model_list.append('SERSIC')

        self.light_model_source = LightModel(light_model_list = source_light_model_list)

        # LENS light
        lens_light_model_list = ['SERSIC_ELLIPSE']
        self.light_model_lens = LightModel(light_model_list = lens_light_model_list)

        # SOURCE light kwargs
        self.kwargs_light_source = [{'amp': 1000., 'R_sersic': r_sersic_source, 'n_sersic': n_sersic_source, 
                              'e1': e1s, 'e2': e2s, 'center_x': beta_ra , 'center_y': beta_dec}]
        for i in range(N_clump):
            self.kwargs_light_source.append({'amp': 1000, 'R_sersic': r_sersic_source_clumps, 'n_sersic': n_sersic_source_clumps,
                                        'center_x': beta_ra+source_scatter*(clumprandx[i]-.5), 
                                        'center_y': beta_dec+source_scatter*(clumprandy[i]-.5)})

        # LENS light kwargs
        self.kwargs_light_lens = [{'amp': 1500, 'R_sersic': theta_lens, 'n_sersic': gamma, 
                              'e1': e1, 'e2': e2, 'center_x': center_lens_x , 'center_y': center_lens_y}]

        ################################################################################
        # Setup data_class, i.e. pixelgrid #
        ra_at_xy_0 = -0.5*self.pixnum*self.pixsize # coordinate in angles (RA/DEC) at the position of the pixel edge (0,0)
        dec_at_xy_0 = -0.5*self.pixnum*self.pixsize # ''
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * self.pixsize  # linear translation matrix of a shift in pixel in a shift in coordinates
        kwargs_pixel = {'nx': self.pixnum, 'ny': self.pixnum,  # number of pixels per axis
                        'ra_at_xy_0': ra_at_xy_0,  # RA at pixel (0,0)
                        'dec_at_xy_0': dec_at_xy_0,  # DEC at pixel (0,0)
                        'transform_pix2angle': transform_pix2angle} 
        self.pixel_grid = PixelGrid(**kwargs_pixel)

        # Setup PSF #
        kwargs_psf = {'psf_type': 'GAUSSIAN',  # type of PSF model (supports 'GAUSSIAN' and 'PIXEL')
                      'fwhm': 0.01,  # full width at half maximum of the Gaussian PSF (in angular units)
                      'pixel_size': self.pixsize  # angular scale of a pixel (required for a Gaussian PSF to translate the FWHM into a pixel scale)
                     }
        self.psf = PSF(**kwargs_psf)
        kernel = self.psf.kernel_point_source

        
        # define the numerics #
        self.kwargs_numerics = {'supersampling_factor': 1, # each pixel gets super-sampled (in each axis direction) 
                          'supersampling_convolution': False}
        # initialize the Image model class by combining the modules we created above #
        self.imageModel = ImageModel(data_class=self.pixel_grid, psf_class=self.psf,
                                     lens_model_class=self.lens_model_mp,
                                     source_model_class=self.light_model_source,
                                     lens_light_model_class=self.light_model_lens,
                                     kwargs_numerics=self.kwargs_numerics)

        # simulate image with the parameters we have defined above #
        self.image = self.imageModel.image(kwargs_lens=self.kwargs_lens,
                                           kwargs_source=self.kwargs_light_source,
                                           kwargs_lens_light=self.kwargs_light_lens)#, kwargs_ps=kwargs_ps)

    def calc_div_curl(self):
        # Calculates divergence and curl of alpha (from ray shooting)
        
        self.alphamat_x = np.zeros((self.pixnum, self.pixnum))
        self.alphamat_y = np.zeros((self.pixnum, self.pixnum))
        for xpix in range(self.pixnum):
            for ypix in range(self.pixnum):
                image_xy = self.pixel_grid.map_pix2coord(xpix, ypix) # in angle units
                source_xy = self.lens_model_mp.ray_shooting(image_xy[0], image_xy[1], self.kwargs_lens)
                self.alphamat_x[xpix,ypix] = image_xy[0] - source_xy[0]
                self.alphamat_y[xpix,ypix] = image_xy[1] - source_xy[1]

        self.divmat = (np.gradient(self.alphamat_x, self.pixsize)[0]
                       + np.gradient(self.alphamat_y, self.pixsize)[1])
        self.curlmat = (np.gradient(self.alphamat_y, self.pixsize)[0]
                        - np.gradient(self.alphamat_x, self.pixsize)[1])
        return self.divmat, self.curlmat
    
    def calc_div_curl_5pt(self):
        ## Calculates divergence and curl using 5pt stencil.
        self.alphamat_x = np.zeros((self.pixnum, self.pixnum))
        self.alphamat_y = np.zeros((self.pixnum, self.pixnum))
        for xpix in range(self.pixnum):
            for ypix in range(self.pixnum):
                image_xy = self.pixel_grid.map_pix2coord(xpix, ypix) # in angle units
                source_xy = self.lens_model_mp.ray_shooting(image_xy[0], image_xy[1], self.kwargs_lens)
                self.alphamat_x[xpix,ypix] = image_xy[0] - source_xy[0]
                self.alphamat_y[xpix,ypix] = image_xy[1] - source_xy[1]
                
        self.divmat = np.zeros([self.pixnum-4,self.pixnum-4])
        self.curlmat = np.zeros([self.pixnum-4,self.pixnum-4])
        
        def divfunc(vec_x, vec_y,i,j):
            diffx = (-1./12.)*(vec_x[i][j+2] - vec_x[i][j-2])+(2./3.)*(vec_x[i][j+1] - vec_x[i][j-1])
            diffy = (-1./12.)*(vec_y[i+2][j] - vec_y[i-2][j])+(2./3.)*(vec_y[i+1][j] - vec_y[i-1][j])
            return (diffx + diffy)*(1./self.pixsize)

        def curlfunc(vec_x, vec_y,i,j):
            offy = (-1./12.)*(vec_y[i][j+2] - vec_y[i][j-2])+(2./3.)*(vec_y[i][j+1] - vec_y[i][j-1])
            offx = (-1./12.)*(vec_x[i+2][j] - vec_x[i-2][j])+(2./3.)*(vec_x[i+1][j] - vec_x[i-1][j])
            return (offy - offx)*(1./self.pixsize)
        
        for i in range(2,self.pixnum-2):
            for j in range(2,self.pixnum-2):
                self.divmat[i-2][j-2] = divfunc(self.alphamat_y,self.alphamat_x,i,j)
                self.curlmat[i-2][j-2] = curlfunc(self.alphamat_y,self.alphamat_x,i,j)
                
        return self.divmat, self.curlmat
        
        
    
    def div_curl_simple(self):
       # Calculates divergence and curl of alpha by subtracting neighboring pixels
        self.alphamat_x = np.zeros((self.pixnum, self.pixnum))
        self.alphamat_y = np.zeros((self.pixnum, self.pixnum))
        for xpix in range(self.pixnum):
            for ypix in range(self.pixnum):
                image_xy = self.pixel_grid.map_pix2coord(xpix, ypix) # in angle units
                source_xy = self.lens_model_mp.ray_shooting(image_xy[0], image_xy[1], self.kwargs_lens)
                self.alphamat_x[xpix,ypix] = image_xy[0] - source_xy[0]
                self.alphamat_y[xpix,ypix] = image_xy[1] - source_xy[1]
                
        self.divmat = np.zeros([self.pixnum-2,self.pixnum-2])
        self.curlmat = np.zeros([self.pixnum-2,self.pixnum-2])
        
        def divfunc(vec_x, vec_y,i,j):
            diffx = vec_x[i][j+1] - vec_x[i][j-1]
            diffy = vec_y[i+1][j] - vec_y[i-1][j]
            return (diffx + diffy)*(0.5/self.pixsize)

        def curlfunc(vec_x, vec_y,i,j):
            offy = vec_y[i][j+1] - vec_y[i][j-1]
            offx = vec_x[i+1][j] - vec_x[i-1][j]
            return (offy - offx)*(0.5/self.pixsize)
        
        for i in range(1,self.pixnum-1):
            for j in range(1,self.pixnum-1):
                self.divmat[i-1][j-1] = divfunc(self.alphamat_y,self.alphamat_x,i,j)
                self.curlmat[i-1][j-1] = curlfunc(self.alphamat_y,self.alphamat_x,i,j)
                
        return self.divmat, self.curlmat

class PoolResults:
    """
    Runs and stores results for pool-based function calls
    """
    def __init__(self, func, init_args_list):
        """
        args_list = [xpos_args, ypos_args, zds] (for example)
        """
        # function we'll be running and args we'll be running it on
        self.func = func
        self.args_list = [] # we will change this after running on init_args_list
        self.results = {}

        self.run(init_args_list)

        # # number of different kinds of arguments the function takes in (not counting id number as first arg)
        # self.n_args = len(self.all_args)

        # # number of fits we want
        # self.n_fits = len(self.all_args[0])
        # for i in range(self.all_args[1:]):
        #     assert(len(self.all_args[i]) == self.n_fits)


    
    def __repr__(self):
        return 'PoolResults'+self.results.__repr__()
    
    def callback(self, result):
        assert(len(result) == 2)
        # result[0] is an id number that was also the first argument of func
        # result[1] is the PSOFit object (or whatever we're actually interested in)
        self.results[result[0]] = result[1]
        
    def run(self, new_args_list):
        # TODO: preprocessing step to remove anything redundant from the args_list

        if len(new_args_list) == 0: return
        
        nargs = len(new_args_list[0])
        for i in range(1,len(new_args_list)):
            assert(len(new_args_list[i]) == nargs) # check that all `args` are the same length
        
        with Pool() as pool:
            p_list = []
            for i, func_args in enumerate(zip(*new_args_list), start=len(self.args_list[0]) if len(self.args_list) > 0 else 0):
                p = pool.apply_async(self.func, args=(i, *func_args), callback=self.callback)
                p_list.append(p)
                
            for p in p_list: # I think this is redundant
                p.wait()

            for p in p_list:
                p.get()


        if len(self.args_list) == 0:
            self.args_list = [list(args) for args in new_args_list]
        else:
            assert(len(self.args_list) == len(new_args_list)) # should be true, because otherwise the function call would have failed
            for i in range(len(self.args_list)):
                self.args_list[i] += list(new_args_list[i])

    def get_results_list(self):
        N = max(self.results.keys()) + 1
        results_list = [None] * N
        for k,v in self.results.items():
            results_list[k] = v
        return results_list
