#!/usr/bin/env python
# coding: utf-8

# # Slider gen
# 
# 23 April 2020 -- copied from `slider_test`
# 
# Making a slide widget using `plotly`. See default example at https://plotly.com/python/sliders/ .

# In[1]:


import numpy as np
#import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly import offline
from plotly.subplots import make_subplots

from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy import units as u

from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d, interp2d

import os

from helpers import sigma_cr

cosmo = FlatLambdaCDM(H0=67.5, Om0=0.316)


# In[7]:


## contour color scale:
colorscale = [[0, 'gold'], [0.5, 'mediumturquoise'], [1, 'lightsalmon']]


# ## Known systems (code from Cagan)

# In particular, we will be using the SLACs info to make sense of the subhalo normalization.

# In[2]:


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


# In[3]:


# Not yet, but it will soon be relevant what the typical Sigma_cr and z_l are for slacs
sigcrs_slacs = sigma_cr(zlenslist_slacs, zsourcelist_slacs).to(u.Msun / u.kpc**2).value
slacs_sigcr = np.mean(sigcrs_slacs)
slacs_zl = np.mean(zlenslist_slacs)
print('sigcr =', slacs_sigcr/1e9, 'billion Msun/kpc^2, z_l =', slacs_zl)


# ## Interloper plateaux (code from Cagan)

# In[4]:


resl = 100 # number of redshift values
zmin = 0.001
zmax = 3.301 ## now it is time to generate a redshift list
zrange = np.linspace(zmin,zmax,resl)
# zl, zs = np.meshgrid(zrange,zrange)


# In[5]:


limber_plats = np.load('files/plateau_mats/limber_plats.npy')


# In[6]:


limber_plats_PS = np.load('files/plateau_mats/limber_platsPS.npy')


# ## Subhalo normalization
# The old version of this document assumed a dependence of $(1+z)^{-3/2}$. But let's convert this to $f_{sub}$.
# 
# For now we will assume  although I need to check this.

# In[8]:


f0 = 8e-4 # default value of fsub (doesn't matter thanks to slider)
m_low = 1e5
m_high = 1e8


# $$P_0 = \frac{1}{\Sigma_{cr}^2}\int_{m_l}^{m_h} m^2 n(m) dm$$
# where $n(m) = F m^{-1.9}$. Solving the integral,
# $$P_0 = \frac{F}{\Sigma_{cr}^2} \left( \frac{m_h^{1.1} - m_l^{1.1}}{1.1} \right)$$
# And now plugging in
# $$F = \Sigma_{cr}\frac{f_{sub}}{2}\frac{0.1}{m_h^{0.1} - m_l^{0.1}},$$
# so the power is
# $$P_0 = \frac{f_{sub}}{2\Sigma_{cr}}\frac{0.1}{1.1} \left( \frac{m_h^{1.1} - m_l^{1.1}}{m_h^{0.1} - m_l^{0.1}} \right)$$

# And actually we add another factor of
# $$\times \frac{(1+z_l)^{5/2}}{(1+.5)^{5/2}}$$
# to represent the redshift-dependence.

# In[9]:


def power_sub(z_l, z_s, mlow=m_low, mhigh=m_high):
    # Calculates plateau of power spectrum for given zl and zs
    sigcr = sigma_cr(z_l, z_s).to(u.M_sun/u.kpc**2).value
    redshift_dependence = ((1+z_l)**(2.5))/((1+.5)**(2.5))
    
    return f0 / (2*sigcr) * (0.1/1.1) * (m_high**1.1 - m_low**1.1)/(m_high**0.1 - m_low**0.1) * redshift_dependence

power_sub(.5,1)


# In[10]:


# %%time
## Calculate subhalo power (plateau) grid ##
print('about to calculate sub_power_grid')

sub_power_grid = np.zeros((len(zrange), len(zrange)))
for i_l, z_l in enumerate(zrange):
    for i_s, z_s in enumerate(zrange):
        if z_l < z_s:
            sub_power_grid[i_s, i_l] = power_sub(z_l, z_s)
        else:
            sub_power_grid[i_s, i_l] = float('nan')
            
print('finished calculating sub_power_grid')


# ## Subplots

# In[11]:


## first a helper function for html output ##
def with_jax(fig, filename):

    plot_div = offline.plot(fig, output_type = 'div')

    template = """
    <head>
    <script type="text/javascript"
      src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG">
    </script>
    </head>
    <body>
    {plot_div:s}
    </body>""".format(plot_div = plot_div)
    with open(filename, 'w') as fp:
        fp.write(template)
    
    ## Note: the graph will initially load faster, but incorrectly, if you add the term "async" to the <script> part

# def with_jax(fig, filename):
#     ## second version ##

#     offline.plot(fig, include_mathjax='cdn', image_filename=filename)


# In[12]:
def make_widget(limber_plat_mat, sub_mat, zrange, html_filename, smoothing=0):
    """ make a widget (either for Sheth-Tormen or Press-Schechter) """

    ### preprocessing limber_plat_mat ###
    if len(limber_plat_mat.shape) == 3:
        limber_plat_mat = limber_plat_mat[:,:,0]
    
    ### make subplots ###
    fig = make_subplots(rows=1, cols=3, subplot_titles=(r'$\log(P_I / \mathrm{kpc}^2)$',
                                                        r'$\log(P_\mathrm{sub} / \mathrm{kpc}^2)$',
                                                        r'$\log(P_\mathrm{sub}/P_I)$'))
    fig.layout.yaxis.update(title=r'$\text{source redshift }(z_s)$')
    fig.layout.xaxis.update(title=r'$\text{lens redshift }(z_l)$')
    fig.layout.xaxis2.update(title=r'$\text{lens redshift }(z_l)$')
    fig.layout.xaxis3.update(title=r'$\text{lens redshift }(z_l)$')

    ### color axis ###
    fig.update_layout(coloraxis1=dict(colorscale=colorscale,
                                      colorbar=dict(
                                          len=.6,
                                          lenmode='fraction',
                                          yanchor='bottom',
                                          y=0,
                                          x=.20,
                                          dtick=1)))
    fig.update_layout(coloraxis2=dict(colorscale=colorscale,
                                      colorbar=dict(
                                          len=.6,
                                          lenmode='fraction',
                                          yanchor='bottom',
                                          y=0,
                                          x=.55,
                                          dtick=1)))
    fig.update_layout(coloraxis3=dict(colorscale='RdBu',
                                      colorbar=dict(
                                          len=.6,
                                          lenmode='fraction',
                                          yanchor='bottom',
                                          y=0,
                                          x=.91,
                                          dtick=1)))

    color_start = -8
    color_end = -3
    ratcolor_start = -3
    ratcolor_end = 3

    ### hover text ###
    hover = "z_l: %{x:.2f} <br> z_s: %{y:.2f} <br> log P: %{z:.2f} <extra></extra>"
    hover3 = "z_l: %{x:.2f} <br> z_s: %{y:.2f} <br> log P_s/P_i: %{z:.2f} <extra></extra>"
    
    ### interlopers ###
    fig.add_trace(
        go.Contour(
            z=np.log10(limber_plat_mat),
            x=zrange,
            y=zrange,
            coloraxis="coloraxis1",
            contours=dict(start=color_start,
                          end=color_end),
            line_smoothing=smoothing,
            hovertemplate=hover),
        row=1, col=1)

    ### subhalos ###
    subidx = len(fig.data) # count data starting here
    fvals = np.logspace(-4,-1, 10)

    for fval in fvals:
        fig.add_trace(
            go.Contour(
                visible=False,
                z=np.log10(sub_mat * fval/f0),
                x=zrange,
                y=zrange,
                coloraxis="coloraxis2",
                contours=dict(start=color_start,
                              end=color_end),
                line_smoothing=smoothing,
                hovertemplate=hover),
            row=1, col=2)

    ### ratio ###
    ratidx = len(fig.data)
    for fval in fvals:
        fig.add_trace(
            go.Contour(
                visible=False,
                z=np.log10(sub_mat * fval/f0 / limber_plat_mat),
                x=zrange,
                y=zrange,
                coloraxis="coloraxis3",
                contours=dict(start=ratcolor_start,
                              end=ratcolor_end),
                line_smoothing=smoothing,
                hovertemplate=hover3),
            row=1, col=3)

    ### Known Systems ###
    ksidx = len(fig.data)
    known_systems = [dict(x=zlenslist_bells,
                          y=zsourcelist_bells,
                          mode='markers',
                          marker={'color':'green'},
                          name='BELLS',
                          legendgroup='a'
                         ),
                     dict(x=zlenslist_slacs,
                          y=zsourcelist_slacs,
                          mode='markers',
                          marker={'color':'orange'},
                          name='SLACS',
                          legendgroup='b'
                         ),
                     dict(x=zlenslist3,
                          y=zsourcelist3,
                          mode='markers',
                          marker={'color':'red'},
                          name='JVAS B1938+666',
                          legendgroup='c'
                         ),
                     dict(x=zlenslist4,
                          y=zsourcelist4,
                          mode='markers',
                          marker={'color':'purple'},
                          name='SDP.81',
                          legendgroup='d'
                         )]
    for i, k in enumerate(known_systems):
        knoshow = k.copy()
        knoshow['showlegend'] = False

        fig.add_trace(go.Scatter(k), row=1, col=1)
        fig.add_trace(go.Scatter(knoshow), row=1, col=2)
        fig.add_trace(go.Scatter(knoshow), row=1, col=3)
    ksidx_end = len(fig.data)


    ### slider ###
    # Make first trace visible
    fig.data[subidx+0].visible = True
    fig.data[ratidx+0].visible = True

    # Create and add slider
    steps = []
    for i, fval in enumerate(fvals):
        step = dict(
            method="restyle",
            args=["visible", [True]*subidx + [False] * len(fvals) + [False] * len(fvals) + [True]*(ksidx_end-ksidx)],
            label="%.02g"%(fval)
        )
        step["args"][1][subidx+i] = True  # Toggle i'th trace to "visible"
        step["args"][1][ratidx+i] = True
        #print('step', step)
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": r"f_(sub, 0.5): "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )


    #fig.write_html('html/plateau_plots_PS.html')
    with_jax(fig, html_filename)
    # fig.show() # opens up in browser, on some port of 127.0.0.1

## Now, of course, the problem is that the resulting plotly files are too slow.
## The easiest way I know to try to speed it up is to sample at particular
## steps.

step = 2
zrange_new = zrange[::step]
limber_plats_ST_new = limber_plats[::step,::step,0]
limber_plats_PS_new = limber_plats_PS[::step,::step,0]
sub_power_grid_new = sub_power_grid[::step,::step]

make_widget(limber_plats_ST_new, sub_power_grid_new, zrange_new, 'html/plateau_plots_ST.html', smoothing=1.3)
make_widget(limber_plats_PS_new, sub_power_grid_new, zrange_new, 'html/plateau_plots_PS.html', smoothing=1.3)


print('DONE')
