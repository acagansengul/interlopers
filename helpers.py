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
