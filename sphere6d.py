import sys
import os.path
from math import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
from matplotlib import rc
from matplotlib import colors
import matplotlib.patches as patch
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits
import matplotlib.colorbar as colorbar
import colormaps as cmaps
from uvw_calc import gal_coords
from scipy.optimize import curve_fit, minimize
import scipy.stats as stats
from scipy.spatial import distance
from scipy import interpolate
from pathlib import Path
from scipy import integrate
from sklearn import svm
import itertools
from scipy import linalg
import matplotlib as mpl
from sklearn import mixture
from astroML import density_estimation


class sphere6d:
    def __init__(self, init_param=None):
        # add "kwargs = utils.none_to_empty_dict(kwargs)"
        # try: except TypeError: give
        # print(traceback.print_exc())
        # print('\nValid keywords:\n')
        # print('exponential: timescale, min_snia_time, snia_fraction\n')
        # print('power_law: min_snia_time, nia_per_mstar, slope\n')
        # print('prompt_delayed: A, B,  min_snia_time\n')
        # print('single_degenerate: no keywords\n')
        if init_param is not None:
            self.initialize_model(**init_param)
        else:
            pass

    def grid_model(self, rot_curve='solid_body', vmax=10., ramp=10.,
                   rmax=100., density_dist=['linear'],
                   shape=['spherical']):

        self.vmax = vmax
        self.ramp = ramp
        self.rmax = rmax
        self.rot_curve = rot_curve
        self.density_dist = density_dist
        self.shape_func = shape[0]

        if rot_curve == 'solid_body':
            self.rmax = self.ramp

        if density_dist[0] == 'linear':
            rsphs_grid = np.linspace(0., self.rmax, 50., endpoint=True)
            phis_grid = np.linspace(0., 2 * np.pi, 180., endpoint=False)
            thetas_grid = np.linspace(-np.pi / 2.,
                                      np.pi / 2., 90., endpoint=True)
            rsphs, phis, thetas = np.meshgrid(
                rsphs_grid, phis_grid, thetas_grid)
            self.rsphs = rsphs.flatten()
            self.phis = phis.flatten()
            self.thetas = thetas.flatten()

        elif density_dist[0] == 'plummer':
            # density_dist should be a list of input parameters of
            # ['type',*param]
            # for king *param should be a
            self.rsphs = density_dist[1]
            self.phis = density_dist[2]
            self.thetas = density_dist[3]

        if shape[0] == 'spherical':
            self.cyl_calc()
        elif shape[0] == 'ellipsoidal':
            self.ell_cyl_calc(shape[1])

        if rot_curve == 'solid_body':
            self.calc_solidbody()
        elif rot_curve == 'mackey2013':
            self.calc_mackey2013()
        elif rot_curve == 'sphmackey2013':
            self.calc_sphmackey2013()

    def star_model(self):
        pass

    def plummer_density(self, s, nstars):
        rsphs = plummer_profile.rvs(s, a=0., size=nstars)
        phis = stats.uniform.rvs(size=nstars) * 2. * np.pi
        thetas = stats.uniform.rvs(size=nstars) * np.pi / 2.
        return rsphs, phis, thetas

    def calc_solidbody(self):
        self.vzs = np.zeros(self.rs.shape)
        self.vrs = np.zeros(self.rs.shape)
        self.vphis = self.rs * self.vmax / self.ramp
        self.cart_calc()

    def calc_mackey2013(self):
        self.vzs = np.zeros(self.rs.shape)
        self.vrs = np.zeros(self.rs.shape)
        self.vphis = (2 * self.vmax / self.ramp) * \
            (self.rs / (1 + (self.rs / self.ramp)**2.))
        self.cart_calc()

    def calc_sphmackey2013(self):
        vrot = (2 * self.vmax / self.ramp) * \
            (self.rsphs / (1 + (self.rsphs / self.ramp)**2.))
        self.vphis = vrot * np.cos(self.thetas)
        self.vthetas = vrot * np.sin(self.thetas)
        self.vrsphs = np.zeros(self.thetas.shape)

        self.vzs = self.vthetas * np.cos(self.thetas)
        self.vrs = self.vthetas * np.sin(self.thetas)
        self.cart_calc()

    def cart_calc(self):
        self.xs = self.rs * np.cos(self.phis)
        self.ys = self.rs * np.sin(self.phis)
        self.vys = self.vphis * \
            np.cos(self.phis) + self.vrs * np.sin(self.phis)
        self.vxs = - self.vphis * \
            np.sin(self.phis) + self.vrs * np.cos(self.phis)

    def cyl_calc(self):
        self.rs = self.rsphs * np.cos(self.thetas)
        self.zs = self.rsphs * np.sin(self.thetas)

    def ell_cyl_calc(self, vel_disp):
        self.vel_disp = vel_disp
        self.rs = self.rsphs * np.cos(self.thetas)
        self.zs = self.rsphs * \
            (1. / (1. + (self.vmax / self.vel_disp)**2.)) * np.sin(self.thetas)
        # self.rsphs = np.hypot(self.rs,self.zs)
        # self.thetas = np.tan(self.zs,self.rs)

    def projection(self, inclination, omega):

        # x is pointing toward us positive inclination tips it towards us and omega rotates counterclockwise
        # on the sky
        ys_prime = self.ys
        xs_prime = self.xs * np.cos(inclination) + \
            self.zs * np.sin(inclination)
        zs_prime = -self.xs * np.sin(inclination) + \
            self.zs * np.cos(inclination)

        xs_prime2 = xs_prime
        ys_prime2 = ys_prime * np.cos(omega) - zs_prime * np.sin(omega)
        zs_prime2 = ys_prime * np.sin(omega) + zs_prime * np.cos(omega)

        vys_prime = self.vys
        vxs_prime = self.vxs * \
            np.cos(inclination) + self.vzs * np.sin(inclination)
        vzs_prime = -self.vxs * \
            np.sin(inclination) + self.vzs * np.cos(inclination)

        vxs_prime2 = vxs_prime
        vys_prime2 = vys_prime * np.cos(omega) - vzs_prime * np.sin(omega)
        vzs_prime2 = vys_prime * np.sin(omega) + vzs_prime * np.cos(omega)

        self.xs_prime = xs_prime2
        self.ys_prime = ys_prime2
        self.zs_prime = zs_prime2
        self.omega = omega
        self.inclination = inclination
        self.vxs_prime = vxs_prime2
        self.vys_prime = vys_prime2
        self.vzs_prime = vzs_prime2

    def observe(self, ra, dec, distance, pmra_sys, pmdec_sys, rv_sys):
        self.obs_dist = distance
        self.obs_ra_cen = ra
        self.obs_dec_cen = dec
        self.obs_pmra_sys = pmra_sys
        self.obs_pmdec_sys = pmdec_sys
        self.obs_rv_sys = rv_sys

        # modify ra and dec positioning, maybe add distance to x and then
        # rotate x-y through RA of center
        # then rotate y?x?-z through dec, then calculate the
        # theta phi (lat long) which will be RA Dec

        x_dist = -self.xs_prime + self.obs_dist

        xrot1 = x_dist * np.cos(np.radians(self.obs_dec_cen)) - self.zs_prime * np.sin(
            np.radians(self.obs_dec_cen))
        yrot1 = -self.ys_prime
        zrot1 = x_dist * np.sin(np.radians(self.obs_dec_cen)) + self.zs_prime * np.cos(
            np.radians(self.obs_dec_cen))

        xsun = xrot1 * np.cos(np.radians(self.obs_ra_cen)) - \
            yrot1 * np.sin(np.radians(self.obs_ra_cen))
        ysun = xrot1 * np.sin(np.radians(self.obs_ra_cen)) + \
            yrot1 * np.cos(np.radians(self.obs_ra_cen))
        zsun = zrot1

        self.zsun = zsun
        self.ysun = ysun
        self.xsun = xsun

        self.obs_decs = np.degrees(np.arctan(zsun / np.hypot(xsun, ysun)))
        self.obs_ras = np.degrees(np.arctan2(ysun, xsun))
        self.obs_ras[self.obs_ras < 0.] += 360.

        # Old ra dec

        # self.obs_decs = dec + np.degrees(self.zs_prime/distance)
        # Negative for sky right
        # self.obs_ras = ra - (np.degrees(self.ys_prime/distance))/np.cos(np.radians(self.obs_decs))

        # I think this should be correct because pmra*cos dec is how far something moves on the sky
        # in the direction of ra not how much ra it moves, so if we divide tangential velocity
        # by distance, that should be the on sky motion, but the input does need to be ra cosdec

        vxrot1 = -self.vxs_prime * np.cos(np.radians(self.obs_dec_cen)) - \
            self.vzs_prime * np.sin(np.radians(self.obs_dec_cen))
        vyrot1 = -self.vys_prime
        vzrot1 = -self.vxs_prime * np.sin(np.radians(self.obs_dec_cen)) + \
            self.vzs_prime * np.cos(np.radians(self.obs_dec_cen))

        vxsun = vxrot1 * np.cos(np.radians(self.obs_ra_cen)) - \
            vyrot1 * np.sin(np.radians(self.obs_ra_cen))
        vysun = vxrot1 * np.sin(np.radians(self.obs_ra_cen)) + \
            vyrot1 * np.cos(np.radians(self.obs_ra_cen))
        vzsun = vzrot1

        self.vxsun = vxsun
        self.vysun = vysun
        self.vzsun = vzsun

        kms_to_pcpy = 1.022 * 10**(-6.)

        self.obs_rvs = rv_sys + (xsun * vxsun + ysun * vysun +
                                 zsun * vzsun) / np.sqrt(xsun**2. + ysun**2. + zsun**2.)
        self.obs_pmras = pmra_sys - (np.degrees(kms_to_pcpy * (vxsun * ysun -
                                                               xsun * vysun) / np.hypot(
            xsun, ysun)**2.) * 3600. * 1000. *
            np.cos(np.radians(self.obs_decs)))
        self.obs_pmdecs = pmdec_sys - (np.degrees(kms_to_pcpy * (
            zsun * (xsun * vxsun - ysun * vysun) - np.hypot(xsun, ysun)**2. *
            vzsun) / (np.hypot(xsun, ysun) * (xsun**2. + ysun**2. +
                                              zsun**2.))) * 3600. * 1000.)

        # old proper motions
        # Negative for sky-right
        # self.obs_pmras = pmra_sys - self.vys_prime/(4.74 * distance) * 1000.
        # self.obs_pmdecs = pmdec_sys + self.vzs_prime/(4.74 * distance) * 1000.
        # plus x direction is towards us, so we need to make that negative
        # self.obs_rvs = rv_sys - self.vxs_prime

    def initialize_model(self, rot_curve='mackey2013', density_dist=['linear'],
                         shape_func='ellipsoidal', vel_disp=10., vmax=10.,
                         ramp=10., rmax=100., ra=0., dec=0., dist=100.,
                         pmra_sys=0., pmdec_sys=0., rv_sys=0., inclination=0.,
                         omega=0.):

        self.grid_model(rot_curve=rot_curve, vmax=vmax, ramp=ramp, rmax=rmax,
                        density_dist=density_dist, shape=[shape_func, vel_disp])
        self.projection(inclination, omega)
        self.observe(ra, dec, dist, pmra_sys, pmdec_sys, rv_sys)

    def _pack_init_dict(self, X):
        fixed_param = ['rot_curve', 'density_dist',
                       'shape_func', 'rmax', 'ra', 'dec', 'dist', 'vel_disp']
        fit_param = ['pmra_sys', 'pmdec_sys', 'rv_sys',
                     'vmax', 'ramp', 'inclination', 'omega']
        fixed_values = [self.rot_curve, self.density_dist, self.shape_func,
                        self.rmax, self.obs_ra_cen, self.obs_dec_cen,
                        self.obs_dist, self.vel_disp]
        # fixed_values = [self.rot_curve, self.density_dist, self.shape_func, self.rmax, self.obs_ra_cen,
        #               self.obs_dec_cen, self.obs_dist, self.obs_pmra_sys, self.obs_pmdec_sys, self.obs_rv_sys]

        pmra_sys, pmdec_sys, rv_sys, tan_omega_o4, tan_inc_o2, logvmax, logramp = X
        # tan_omega_o2,tan_inc_o2,logvmax,logramp,logvsigma = X

        # fix angle
        # tan_inc_o2 = np.tan(np.radians(54.)/2.)
        # tan_omega_o2 = np.tan(np.radians(-152.)/2.)

        # fixed ramp
        # logramp = 0.945

        # fixed vsigma
        # logvsigma = 1.

        inclination = 2. * np.arctan(tan_inc_o2)
        omega = 4. * np.arctan(tan_omega_o4)
        vmax = 10.**logvmax
        ramp = 10.**logramp

        fit_values = [pmra_sys, pmdec_sys, rv_sys,
                      vmax, ramp, inclination, omega]

        dict_keys = fixed_param + fit_param
        dict_values = fixed_values + fit_values

        init_dict = {dict_keys[i]: dict_values[i]
                     for i in range(len(dict_keys))}

        return init_dict

    def _pack_init_dict_fix_ramp(self, X):
        fixed_param = ['rot_curve', 'density_dist',
                       'shape_func', 'rmax', 'ra', 'dec', 'dist',
                       'vel_disp', 'ramp']
        fit_param = ['pmra_sys', 'pmdec_sys', 'rv_sys', 'vmax', 'inclination',
                     'omega']
        fixed_values = [self.rot_curve, self.density_dist, self.shape_func,
                        self.rmax, self.obs_ra_cen, self.obs_dec_cen,
                        self.obs_dist, self.vel_disp, self.ramp]
        # fixed_values = [self.rot_curve, self.density_dist, self.shape_func, self.rmax, self.obs_ra_cen,
        #               self.obs_dec_cen, self.obs_dist, self.obs_pmra_sys, self.obs_pmdec_sys, self.obs_rv_sys]

        pmra_sys, pmdec_sys, rv_sys, tan_omega_o4, tan_inc_o2, logvmax = X
        # tan_omega_o2,tan_inc_o2,logvmax,logramp,logvsigma = X

        # fix angle
        # tan_inc_o2 = np.tan(np.radians(54.)/2.)
        # tan_omega_o2 = np.tan(np.radians(-152.)/2.)

        # fixed ramp
        # logramp = 0.945

        # fixed vsigma
        inclination = 2. * np.arctan(tan_inc_o2)
        omega = 4. * np.arctan(tan_omega_o4)
        vmax = 10.**logvmax

        fit_values = [pmra_sys, pmdec_sys, rv_sys,
                      vmax, inclination, omega]

        dict_keys = fixed_param + fit_param
        dict_values = fixed_values + fit_values

        init_dict = {dict_keys[i]: dict_values[i]
                     for i in range(len(dict_keys))}

        return init_dict

    def _chi2(self, X, data):

        data_rv = data['VHELIO_AVG']
        data_pmra = data['GAIA_PMRA']
        data_pmdec = data['GAIA_PMDEC']
        data_pmtot = np.hypot(data_pmra, data_pmdec)

        init_dict = self._pack_init_dict(X)
        self.initialize_model(**init_dict)

        model_rv, model_pmra, model_pmdec = self.predict(
            data['RA'], data['DEC'])
        model_pmtot = np.hypot(model_pmra, model_pmdec)

        rv_scale = np.nanstd(data['VHELIO_AVG'])
        pmra_scale = np.nanstd(data['GAIA_PMRA'])
        pmdec_scale = np.nanstd(data['GAIA_PMDEC'])
        pmtot_scale = np.nanstd(
            np.hypot(data['GAIA_PMRA'], data['GAIA_PMDEC']))

        # rv_scale = np.median(data['VERR'])
        # pmra_scale = np.median(data['GAIA_PMRA_ERROR'])
        # pmdec_scale = np.median(data['GAIA_PMDEC_ERROR'])
        # pmtot_scale = np.nanstd(np.hypot(data['GAIA_PMRA'],data['GAIA_PMDEC']))

        # print(rv_scale,pmra_scale,pmdec_scale)

        rv_component = (data_rv - model_rv)**2. / rv_scale**2.
        pmra_component = (data_pmra - model_pmra)**2. / pmra_scale**2.
        pmdec_component = (data_pmdec - model_pmdec)**2. / pmdec_scale**2.
        pmtot_component = (data_pmtot - model_pmtot)**2. / pmtot_scale**2.

        # print(rv_component,pmra_component,pmdec_component)

        # chisquared = np.nansum(rv_component+pmra_component+pmdec_component+pmtot_component)
        chisquared = np.nansum(rv_component + pmra_component + pmdec_component)

        # print(self.obs_pmra_sys, self.obs_pmdec_sys, self.obs_rv_sys,
        #      self.inclination, self.omega, self.vmax, self.ramp, self.vel_disp)
        # print(chisquared)
        return chisquared

    def _chi2_old(self, X, data):

        data_rv, xedges, yedges = np.histogram2d(
            data['RA'], data['DEC'], weights=data['VHELIO_AVG'], bins=100)
        data_pmra, xedges, yedges = np.histogram2d(
            data['RA'], data['DEC'], weights=data['GAIA_PMRA'], bins=[xedges, yedges])
        data_pmdec, xedges, yedges = np.histogram2d(
            data['RA'], data['DEC'], weights=data['GAIA_PMDEC'], bins=[xedges, yedges])
        data_pmtot, xedges, yedges = np.histogram2d(data['RA'], data['DEC'], weights=np.hypot(
            data['GAIA_PMRA'], data['GAIA_PMDEC']), bins=[xedges, yedges])

        data_counts, xedges, yedges = np.histogram2d(
            data['RA'], data['DEC'], bins=[xedges, yedges])

        data_rv = data_rv / data_counts
        data_pmra = data_pmra / data_counts
        data_pmdec = data_pmdec / data_counts
        data_pmtot = data_pmtot / data_counts

        init_dict = self._pack_init_dict(X)
        self.initialize_model(**init_dict)

        model_rv, xedges, yedges = np.histogram2d(
            self.obs_ras, self.obs_decs, weights=self.obs_rvs, bins=[xedges, yedges])
        model_pmra, xedges, yedges = np.histogram2d(
            self.obs_ras, self.obs_decs, weights=self.obs_pmras, bins=[xedges, yedges])
        model_pmdec, xedges, yedges = np.histogram2d(
            self.obs_ras, self.obs_decs, weights=self.obs_pmdecs, bins=[xedges, yedges])
        model_pmtot, xedges, yedges = np.histogram2d(self.obs_ras, self.obs_decs, weights=np.hypot(
            self.obs_pmras, self.obs_pmdecs), bins=[xedges, yedges])

        model_counts, xedges, yedges = np.histogram2d(
            self.obs_ras, self.obs_decs, bins=[xedges, yedges])

        model_rv = model_rv / model_counts
        model_pmra = model_pmra / model_counts
        model_pmdec = model_pmdec / model_counts
        model_pmtot = model_pmtot / model_counts

        data_rv_m = data_rv
        data_pmra_m = data_pmra
        data_pmdec_m = data_pmdec
        data_pmtot_m = data_pmtot
        model_rv_m = model_rv
        model_pmra_m = model_pmra
        model_pmdec_m = model_pmdec
        model_pmtot_m = model_pmtot

        data_rv_m[data_counts < 1] = np.nan
        data_pmra_m[data_counts < 1] = np.nan
        data_pmdec_m[data_counts < 1] = np.nan
        data_pmtot_m[data_counts < 1] = np.nan
        model_rv_m[data_counts < 1] = np.nan
        model_pmra_m[data_counts < 1] = np.nan
        model_pmdec_m[data_counts < 1] = np.nan
        model_pmtot_m[data_counts < 1] = np.nan

        rv_scale = np.nanstd(data['VHELIO_AVG'])
        pmra_scale = np.nanstd(data['GAIA_PMRA'])
        pmdec_scale = np.nanstd(data['GAIA_PMDEC'])
        pmtot_scale = np.nanstd(
            np.hypot(data['GAIA_PMRA'], data['GAIA_PMDEC']))

        rv_component = (data_rv_m - model_rv_m)**2. / rv_scale**2.
        pmra_component = (data_pmra_m - model_pmra_m)**2. / pmra_scale**2.
        pmdec_component = (data_pmdec_m - model_pmdec_m)**2. / pmdec_scale**2.
        pmtot_component = (data_pmtot_m - model_pmtot_m)**2. / pmtot_scale**2.

        chisquared = np.nansum(
            rv_component + pmra_component + pmdec_component + pmtot_component)

        # print(chisquared)
        return chisquared

    def _log_likelihood(self, X, data):

        data_rv = data['VHELIO_AVG']
        data_pmra = data['GAIA_PMRA']
        data_pmdec = data['GAIA_PMDEC']

        data_rv_err = data['VERR']
        data_pmra_err = data['GAIA_PMRA_ERROR']
        data_pmdec_err = data['GAIA_PMDEC_ERROR']

        init_dict = self._pack_init_dict_fix_ramp(X)
        self.initialize_model(**init_dict)

        model_rv, model_pmra, model_pmdec = self.predict(
            data['RA'], data['DEC'])

        sigma2_rv = data_rv_err ** 2. + self.vel_disp ** 2.
        sigma2_pmra = data_pmra_err ** 2. + \
            (self.vel_disp / (4.74 * self.obs_dist) * 1000.) ** 2.
        sigma2_pmdec = data_pmdec_err ** 2 + \
            (self.vel_disp / (4.74 * self.obs_dist) * 1000.) ** 2.

        # chisquared = np.nansum(rv_component+pmra_component+pmdec_component+pmtot_component)
        log_likelihood = -0.5 * (np.nansum((data_rv - model_rv) ** 2 / sigma2_rv +
                                           np.log(sigma2_rv)) + np.nansum((
                                               data_pmra - model_pmra) ** 2 /
            sigma2_pmra + np.log(sigma2_pmra)) +
            np.nansum((data_pmdec -
                       model_pmdec) ** 2 / sigma2_pmdec +
                      np.log(sigma2_pmdec)))

        # print(log_likelihood)
        return log_likelihood

    def log_prior(self, X):
        pmra_sys, pmdec_sys, rv_sys, tan_omega_o4, tan_inc_o2, logvmax = X
        if self.low_prior is None or self.high_prior is None:
            # self.low_prior = [-100.,-3.,-1.,0.,-1.]
            # self.high_prior = [100.,3.,2.,2.,2.]
            self.low_prior = [-1000., -1000., -1000., -100., -3., -1.]
            self.high_prior = [1000., 1000., 1000., 100., 3., 2.]
        if np.all(np.logical_and(X < self.high_prior, X > self.low_prior)):
            return 0.0
        return -np.inf

    def log_probability(self, X, data):
        lp = self.log_prior(X)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._log_likelihood(X, data)

    def _gcd(self, ra, dec):
        return np.arccos(np.sin(np.radians(self.obs_decs)) *
                         np.sin(np.radians(dec)) + np.cos(np.radians(self.obs_decs)) *
                         np.cos(np.radians(dec)) * np.cos(np.radians(self.obs_ras) -
                                                          np.radians(ra)))

    def _kin_obs_bin(self, bins=50):

        Hcounts, xedges, yedges = np.histogram2d(self.obs_ras, self.obs_decs,
                                                 bins=50)

        indiv_obs = [self.obs_rvs, self.obs_pmras, self.obs_pmdecs]
        system_obs = [self.obs_rv_sys, self.obs_pmra_sys, self.obs_pmdec_sys]

        H_fills = []

        for observable, systemic in zip(indiv_obs, system_obs):
            H_fills += [self._weighted_filled_2dhist(self.obs_ras, self.obs_decs,
                                                     observable, bins=[
                                                         xedges, yedges],
                                                     scaling=Hcounts, fill=systemic)]

        return H_fills, xedges, yedges

    def _weighted_filled_2dhist(self, x, y, weights, bins=10., scaling=1., fill=0.):

        H, xedges, yedges = np.histogram2d(x, y, weights=weights, bins=bins)

        H = H / scaling
        H_m = np.ma.masked_where(np.isnan(H), H)
        H_fill = H_m.filled(fill_value=fill)

        return H_fill

    def _kin_interpolate(self, kin_binned, ra_bins, dec_bins, ra_arr, dec_arr):

        output_arr = np.empty((len(ra_arr), 3))

        ras = (ra_bins[1:] + ra_bins[0:-1]) / 2.
        decs = (dec_bins[1:] + dec_bins[0:-1]) / 2.
        ra_grid, dec_grid = np.meshgrid(ras, decs)
        interp_grid = np.vstack((ra_grid.flatten(), dec_grid.flatten())).T

        for i, kin in enumerate(kin_binned):
            output_arr[:, i] = interpolate.griddata(
                interp_grid, kin.T.flatten(), (np.vstack((ra_arr, dec_arr)).T))

        return output_arr

    def fit(self, data):

        X = [self.obs_pmra_sys, self.obs_pmdec_sys, self.obs_rv_sys,
             np.tan(self.omega / 4.), np.tan(self.inclination / 2.),
             np.log10(self.vmax), np.log10(self.ramp)]
        # X = [np.tan(self.omega/2.),np.tan(self.inclination/2.),np.log10(self.vmax),
        #     np.log10(self.ramp),np.log10(self.vel_disp)]
        self.data = data
        optimize.minimize(self._chi2, X, args=(data), method='Powell')

    def predict(self, ra_arr, dec_arr):

        kin_binned, ra_bins, dec_bins = self._kin_obs_bin(bins=50)
        output_arr = self._kin_interpolate(
            kin_binned, ra_bins, dec_bins, ra_arr, dec_arr)
        return (output_arr[:, i] for i in range(3))
