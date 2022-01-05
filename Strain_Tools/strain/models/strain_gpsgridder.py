# Use GPS Gridder from GMT to interpolate between GPS stations
# The algorithm is based on the greens functions for elastic sheets with a given Poisson's ratio. 
# From: Sandwell, D. T., and P. Wessel (2016),
# Interpolation of 2-D vector data using constraints from elasticity, Geophys. Res.Lett. 


import numpy as np
from numpy.linalg import inv, svd
import subprocess
from Tectonic_Utils.read_write import netcdf_read_write
from Strain_2D.Strain_Tools.strain import velocity_io, strain_tensor_toolbox, utilities
from Strain_2D.Strain_Tools.strain.models.strain_2d import Strain_2d
from Strain_2D.Strain_Tools.strain.utilities import get_stations_from_myvel, get_gpsdata_from_myvel


class gpsgridder(Strain_2d):
    """ gps_gridder class for 2d strain rate """
    def __init__(self, params):
        super().__init__(params.inc, params.range_strain, params.range_data, params.outdir);
        self._Name = 'gpsgridder'
        self._tempdir = params.outdir;
        self._poisson, self._fd, self._eigenvalue = verify_inputs_gpsgridder(params.method_specific);

    def compute(self, myVelfield):
        stations = get_stations_from_myvel(myVelfield)
        gpsdata = get_gpsdata_from_myvel(myVelfield, stations)

        self.configure_network(stations)
        [rot_grd, exx_grd, exy_grd, eyy_grd] = self.compute_with_method(gpsdata);
        xlons = np.arange(self._strain_range[0], self._strain_range[1], self._grid_inc[0])
        ylats = np.arange(self._strain_range[2], self._strain_range[3], self._grid_inc[1])
        return [xlons, ylats, rot_grd, exx_grd, exy_grd, eyy_grd];

    def compute_gridded(self, gpsdata):
        [rot_grd, exx_grd, exy_grd, eyy_grd] = self.compute_with_method(gpsdata)
        xlons = np.arange(self._strain_range[0], self._strain_range[1], self._grid_inc[0])
        ylats = np.arange(self._strain_range[2], self._strain_range[3], self._grid_inc[1])
        return [xlons, ylats, rot_grd, exx_grd, exy_grd, eyy_grd]

    def configure_network(self, stations, verbose = False):

        self._stations = stations

    def compute_with_method(self, gpsdata):

        X, Y, Ux, Uy = gpsgridder_func(self._stations, gpsdata, self._strain_range, self._grid_inc, nu = self._poisson, fudge_factor=self._fd, eigenvalue_ratio=self._eigenvalue)
        xin = np.arange(self._strain_range[0], self._strain_range[1], self._grid_inc[0])
        yin = np.arange(self._strain_range[2], self._strain_range[3], self._grid_inc[1])
        lat_center = np.mean(np.array([self._strain_range[2], self._strain_range[3]]))
        [xin_p, yin_p] = convert_geographic_to_m(xin, yin, lat_center)
        dx = np.mean(np.diff(xin_p))
        dy = np.mean(np.diff(yin_p))
        [exx, eyy, exy, rot] = strain_tensor_toolbox.strain_on_regular_grid(dx, dy, Ux, Uy)

        return rot, exx, exy, eyy

def verify_inputs_gpsgridder(method_specific_dict):
    poisson = method_specific_dict.get("poisson", 0.3)
    fd = method_specific_dict.get("fd", 1E-4)
    eigenvalue = method_specific_dict.get("eigenvalue")
    return poisson, fd, eigenvalue

def get_radius(X, P):
    n = X.shape[0]
    m = P.shape[0]
    dx = np.zeros((m,n))
    dy = np.zeros((m,n))
    for k in range(m):
        dx[k,:] = (X[:,0] - P[k,0]).T
        dy[k,:] = (X[:,1] - P[k,1]).T
    return np.sqrt(np.power(dx,2)+np.power(dy,2)), dx, dy

def get_qpw(X, P, nu, fudge_factor):
    r, dx, dy = get_radius(X, P)
    fudge_factor = fudge_factor if fudge_factor is not None else np.power(np.min(r),2)
    dx2 = np.power(dx,2)
    dy2 = np.power(dy,2)
    dxdy = dx*dy
    dr2 = dx2 + dy2
    dr2_fudge = dr2 + fudge_factor
    dx2_fudge = np.zeros_like(dr2)
    dy2_fudge = np.zeros_like(dr2)
    dxdy_fudge = np.zeros_like(dr2)
    dx2_fudge[dr2 == 0] = 0.5 * fudge_factor
    dy2_fudge[dr2 == 0] = 0.5 * fudge_factor
    dxdy_fudge[dr2 == 0] = 0.5 * fudge_factor
    dx2_fudge[dr2 != 0] = dr2_fudge[dr2 != 0] / dr2[dr2 != 0] * dx2[dr2 != 0]
    dy2_fudge[dr2 != 0] = dr2_fudge[dr2 != 0] / dr2[dr2 != 0] * dy2[dr2 != 0]
    dxdy_fudge[dr2 != 0] = dr2_fudge[dr2 != 0] / dr2[dr2 != 0] * dxdy[dr2 != 0]

    c1 = (3.0 - nu) / 2.0
    c2 = (1 + nu)

    p = c1 * np.log(dr2_fudge) + c2 * dx2_fudge / dr2_fudge
    q = c1 * np.log(dr2_fudge) + c2 * dy2_fudge / dr2_fudge
    w = -c2 * dxdy_fudge / dr2_fudge

    return p, q, w

def find_item_by_name(iterable, name):
    for it in iterable:
        if it.name == name:
            return it
    return None

def convert_geographic_to_km(x, y, lat_center):
    return x * 111.13 * np.cos(np.deg2rad(lat_center)), y * 111.13

def convert_geographic_to_m(x, y, lat_center):
    return x * 111130.0 * np.cos(np.deg2rad(lat_center)), y * 111130.0


def gpsgridder_func(stations, gpsdata, strain_range, grid_inc, nu = 0.3, fudge_factor = None, eigenvalue_ratio = None):

    names = [g.name for g in gpsdata if g.e is not np.nan and g.n is not np.nan]
    xi = np.array([find_item_by_name(stations, name).elon for name in names])
    yi = np.array([find_item_by_name(stations, name).nlat for name in names])
    ui = np.array([find_item_by_name(gpsdata, name).e for name in names])
    vi = np.array([find_item_by_name(gpsdata, name).n for name in names])

    (u0, v0) = (np.mean(ui), np.mean(vi))
    ui -= u0
    vi -= v0

    lat_center = np.mean(strain_range[2:3])

    xp, yp = convert_geographic_to_m(xi,yi,lat_center)
    n = xp.shape[0]
    (xp, yp) = (np.reshape(xp, (n, 1)), np.reshape(yp, (n, 1)))
    (ui, vi) = (np.reshape(ui, (n, 1)), np.reshape(vi, (n, 1)))
    X, Y = np.meshgrid(np.arange(strain_range[0], strain_range[1], grid_inc[0]), np.arange(strain_range[2], strain_range[3], grid_inc[1]))
    (ny, nx) = X.shape
    (xr, yr) = (np.reshape(X, (ny*nx, 1)), np.reshape(Y, (ny*nx, 1)))
    xgrid_p, ygrid_p = convert_geographic_to_m(xr, yr, lat_center)
    q, p, w = get_qpw(np.hstack((xp, yp)), np.hstack((xp, yp)), nu, fudge_factor)
    if eigenvalue_ratio is None:
        wt = np.matmul(inv(np.vstack((np.hstack((q, w)),np.hstack((w, p))))), np.vstack((ui, vi)))
    else:
        U, s, V = svd(np.vstack((np.hstack((q, w)),np.hstack((w, p)))))
        ev_ratio = s / s[0]
        i = np.where(ev_ratio < eigenvalue_ratio)
        s = np.diag(1 / s)
        s[i] = 0.0
        wt = np.matmul(np.matmul(V.T, s), np.matmul(U.T, np.vstack((ui, vi))))
    q, p, w = get_qpw(np.hstack((xp, yp)), np.hstack((xgrid_p, ygrid_p)), nu, fudge_factor)
    return X, Y, np.reshape(u0 + np.matmul(q,wt[0:len(xi)]) + np.matmul(w,wt[len(xi):]), (ny, nx)), np.reshape(v0 + np.matmul(w,wt[0:len(xi)]) + np.matmul(p,wt[len(xi):]), (ny, nx))
