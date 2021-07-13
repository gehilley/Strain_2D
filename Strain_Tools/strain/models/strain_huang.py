# Courtesy of Mong-han Huang
# Strain calculation tool based on a certain number of nearby stations

import numpy as np
from Tectonic_Utils.geodesy import utm_conversion
from Strain_2D.Strain_Tools.strain.models.strain_2d import Strain_2d


class huang(Strain_2d):
    """ Huang class for 2d strain rate, with general strain_2d behavior """
    def __init__(self, params):
        Strain_2d.__init__(self, params.inc, params.range_strain, params.range_data, params.outdir);
        self._Name = 'huang'
        self._radiuskm, self._nstations = verify_inputs_huang(params.method_specific);

    def compute(self, myVelfield):

        self.configure_network(myVelfield, self._strain_range,
                               self._grid_inc, self._radiuskm,
                               self._nstations)

        [rot_grd, exx_grd, exy_grd, eyy_grd] = self.compute_with_huang_method(myVelfield);
        return [self._xlons, self._ylats, rot_grd, exx_grd, exy_grd, eyy_grd];

    def configure_network(self, myVelfield, range_strain, inc, radiuskm, nstations, verbose = False):

        # Set up grids for the computation
        self._ylats = np.arange(range_strain[2], range_strain[3]+0.00001, inc[1])
        self._xlons = np.arange(range_strain[0], range_strain[1]+0.00001, inc[0])
        gx = len(self._xlons)  # number of x - grid
        gy = len(self._ylats)  # number of y - grid
        ns = nstations  # number of selected stations

        self._station_index_map = np.zeros((gy, gx, ns), dtype=np.int32)
        self._station_dx = np.zeros((gy, gx, ns))
        self._station_dy = np.zeros((gy, gx, ns))
        self._enough_stations = np.zeros((gy, gx), dtype = np.bool)
        self._invG = np.zeros((gy, gx, 3, 3))

        self._lons = np.array([item.elon for item in myVelfield])
        self._lats = np.array([item.nlat for item in myVelfield])

        [elon, nlat, _, _, _, _] = velfield_to_huang_non_utm(myVelfield)
        reflon = np.min([item.elon for item in myVelfield])
        reflat = np.min([item.nlat for item in myVelfield])

        # set up a local coordinate reference
        refx = np.min(elon)
        refy = np.min(nlat)
        elon = elon - refx
        nlat = nlat - refy

        # Setting calculation parameters
        EstimateRadius = radiuskm * 1000  # convert to meters

        # 2. The main loop, getting displacement gradients around stations
        # find at least 5 smallest distance stations

        for i in range(gx):
            for j in range(gy):
                [gridX_loc, gridY_loc] = convert_to_local_planar(self._xlons[i], self._ylats[j], reflon, reflat)
                X = elon   # in local coordinates, m
                Y = nlat   # in local coordiantes, m
                l1 = len(elon)

                # the distance from stations to grid
                r = np.zeros((l1,))
                for ii in range(l1):
                    r[ii] = np.sqrt((gridX_loc - X[ii]) * (gridX_loc - X[ii]) + (gridY_loc - Y[ii]) * (gridY_loc - Y[ii]))

                stations = np.zeros((l1, 3))
                stations[:, 0] = r
                stations[:, 1] = elon.reshape((l1,))
                stations[:, 2] = nlat.reshape((l1,))

                iX = stations[stations[:, 0].argsort()]  # sort data, iX represented the order of the sorting of 1st column
                self._station_index_map[j, i, :] = np.reshape(stations[:,0].argsort()[0:ns], (1, 1, ns)).astype(np.int32)
                self._station_dx[j, i, :] = np.reshape(iX[0:ns,1], (1, 1, ns))
                self._station_dy[j, i, :] = np.reshape(iX[0:ns,2], (1, 1, ns))
                # we choose the first ns data
                SelectStations = np.zeros((ns, 3))
                for iiii in range(ns):
                    SelectStations[iiii, :] = iX[iiii, :]

                Px, Py = 0, 0
                Px2, Py2, Pxy = 0, 0, 0

                if SelectStations[ns-1, 0] <= EstimateRadius:
                    self._enough_stations[j, i] = 1.0
                    for iii in range(ns):
                        Px = Px + SelectStations[iii, 1]
                        Py = Py + SelectStations[iii, 2]
                        Px2 = Px2 + SelectStations[iii, 1] * SelectStations[iii, 1]
                        Py2 = Py2 + SelectStations[iii, 2] * SelectStations[iii, 2]
                        Pxy = Pxy + SelectStations[iii, 1] * SelectStations[iii, 2]
                else:
                    self._enough_stations[j,i] = False
                G = [[ns, Px, Py], [Px, Px2, Pxy], [Py, Pxy, Py2]]
                if np.sum(G) == ns:
                    self._enough_stations[j,i] = False
                else:
                    self._invG[j, i, :, :] = np.reshape(np.linalg.inv(G), (1, 1, 3, 3))

    def compute_with_huang_method(self, myVelfield, verbose = False):

        if verbose:
            print("------------------------------\nComputing strain via Huang method.");

        (gy, gx, ns) = self._station_index_map.shape

        exx = np.zeros((gy, gx));
        exy = np.zeros((gy, gx));
        eyy = np.zeros((gy, gx));
        rot = np.zeros((gy, gx));

        e = np.array([item.e*0.001 for item in myVelfield])
        n = np.array([item.n*0.001 for item in myVelfield])

        for i in range(gx):
            for j in range(gy):
                if not self._enough_stations[j, i]:
                    exx[j, i] = np.nan
                    eyy[j, i] = np.nan
                    exy[j, i] = np.nan
                    rot[j, i] = np.nan
                else:
                    valid_station_indexes = np.reshape(self._station_index_map[j, i, :], (ns, ))
                    valid_station_dx = np.reshape(self._station_dx[j, i, :], (ns, ))
                    valid_station_dy = np.reshape(self._station_dy[j, i, :], (ns, ))
                    valid_station_e = e[valid_station_indexes]
                    valid_station_n = n[valid_station_indexes]
                    grid_point_invG = np.reshape(self._invG[j, i, :, :], (3, 3))
                    dU = np.sum(valid_station_e)
                    dV = np.sum(valid_station_n)
                    dxU = np.sum(valid_station_e*valid_station_dx)
                    dyU = np.sum(valid_station_e*valid_station_dy)
                    dxV = np.sum(valid_station_n*valid_station_dx)
                    dyV = np.sum(valid_station_n*valid_station_dy)
                    dx = np.array([[dU], [dxU], [dyU]])
                    dy = np.array([[dV], [dxV], [dyV]])
                    (_, Uxx, Uxy) = np.dot(grid_point_invG, dx)
                    (_, Uyx, Uyy) = np.dot(grid_point_invG, dy)
                    exx[j, i] = Uxx * 1e9;
                    exy[j, i] = .5 * (Uxy + Uyx) * 1e9;
                    eyy[j, i] = Uyy * 1e9;
                    rot[j, i] = .5 * (Uxy - Uyx) * 1e9

        return rot, exx, exy, eyy



def verify_inputs_huang(method_specific_dict):
    # Takes a dictionary and verifies that it contains the right parameters for Huang method
    if 'estimateradiuskm' not in method_specific_dict.keys():
        raise ValueError("\nHuang requires estimateradiuskm. Please add to method_specific config. Exiting.\n");
    if 'nstations' not in method_specific_dict.keys():
        raise ValueError("\nHuang requires nstations. Please add to method_specific config. Exiting.\n");
    radiuskm = float(method_specific_dict["estimateradiuskm"]);
    nstations = int(method_specific_dict["nstations"]);
    return radiuskm, nstations;


def compute_huang(myVelfield, range_strain, inc, radiuskm, nstations, verbose = False):

    if verbose:
        print("------------------------------\nComputing strain via Huang method.");

    # Set up grids for the computation
    ylats = np.arange(range_strain[2], range_strain[3]+0.00001, inc[1]);
    xlons = np.arange(range_strain[0], range_strain[1]+0.00001, inc[0]);
    gx = len(xlons);  # number of x - grid
    gy = len(ylats);  # number of y - grid


    [elon, nlat, e, n, _, _] = velfield_to_huang_non_utm(myVelfield);
    reflon = np.min([item.elon for item in myVelfield]);
    reflat = np.min([item.nlat for item in myVelfield]);

    # set up a local coordinate reference
    refx = np.min(elon);
    refy = np.min(nlat);
    elon = elon - refx;
    nlat = nlat - refy;

    # Setting calculation parameters
    EstimateRadius = radiuskm * 1000;  # convert to meters
    ns = nstations;  # number of selected stations

    self._station_map = np.zeros((gy, gx, ns))
    self._enough_stations = np.zeros((gy, gx), dtype = bool)
    self._invG = np.zeros((gy, gx, 3, 3))

    # 2. The main loop, getting displacement gradients around stations
    # find at least 5 smallest distance stations
    Uxx = np.zeros((gy, gx));
    Uyy = np.zeros((gy, gx));
    Uxy = np.zeros((gy, gx));
    Uyx = np.zeros((gy, gx));
    exx = np.zeros((gy, gx));
    exy = np.zeros((gy, gx));
    eyy = np.zeros((gy, gx));
    rot = np.zeros((gy, gx));
    for i in range(gx):
        for j in range(gy):
            # [gridX_loc, gridY_loc] = coord_to_local_utm(xlons[i], ylats[j], refx, refy);
            [gridX_loc, gridY_loc] = convert_to_local_planar(xlons[i], ylats[j], reflon, reflat);
            X = elon;   # in local coordinates, m
            Y = nlat;   # in local coordiantes, m
            l1 = len(elon);

            # the distance from stations to grid
            r = np.zeros((l1,));
            for ii in range(l1):
                r[ii] = np.sqrt((gridX_loc - X[ii]) * (gridX_loc - X[ii]) + (gridY_loc - Y[ii]) * (gridY_loc - Y[ii]));
                # for a particular grid point, we are computing the distance to every station

            stations = np.zeros((l1, 3));
            stations[:, 0] = r;
            stations[:, 1] = elon.reshape((l1,));
            stations[:, 2] = nlat.reshape((l1,));
            #stations[:, 3] = e.reshape((l1,));
            #stations[:, 4] = n.reshape((l1,));

            iX = stations[stations[:, 0].argsort()];  # sort data, iX represented the order of the sorting of 1st column
            self._station_map[j, i, :] = np.reshape(stations[:,0].argsort()[0:ns], (1, 1, ns))

            # we choose the first ns data
            SelectStations = np.zeros((ns, 5));
            for iiii in range(ns):
                SelectStations[iiii, :] = iX[iiii, :];
            # print(SelectStations);
            Px, Py = 0, 0;
            Px2, Py2, Pxy = 0, 0, 0;
            dU, dV = 0, 0;
            dxU, dyU = 0, 0;
            dxV, dyV = 0, 0;  # should be inside the if statement or not?
            # print(SelectStations[ns-1, 0]);  # the distance of the cutoff station (in m)
            if SelectStations[ns-1, 0] <= EstimateRadius:
                self._enough_stations[j, i] = True
                for iii in range(ns):
                    Px = Px + SelectStations[iii, 1];
                    Py = Py + SelectStations[iii, 2];
                    Px2 = Px2 + SelectStations[iii, 1] * SelectStations[iii, 1];
                    Py2 = Py2 + SelectStations[iii, 2] * SelectStations[iii, 2];
                    Pxy = Pxy + SelectStations[iii, 1] * SelectStations[iii, 2];
                    #dU = dU + SelectStations[iii, 3];
                    #dxU = dxU + SelectStations[iii, 1] * SelectStations[iii, 3];
                    #dyU = dyU + SelectStations[iii, 2] * SelectStations[iii, 3];
                    #dV = dV + SelectStations[iii, 4];
                    #dxV = dxV + SelectStations[iii, 1] * SelectStations[iii, 4];
                    #dyV = dyV + SelectStations[iii, 2] * SelectStations[iii, 4];
            else:
                self._enough_stations = False
            G = [[ns, Px, Py], [Px, Px2, Pxy], [Py, Pxy, Py2]];
            if np.sum(G) == ns:
                self._enough_stations = False
            else:
                self._invG[j, i, :, :] = np.reshape(np.linalg.inv(G), (1, 1, 3, 3))

                dx = np.array([[dU], [dxU], [dyU]]);
                dy = np.array([[dV], [dxV], [dyV]]);
                modelx = np.dot(np.linalg.inv(G), dx);
                modely = np.dot(np.linalg.inv(G), dy);
                Uxx[j, i] = modelx[1];
                Uyy[j, i] = modely[2];
                Uxy[j, i] = modelx[2];
                Uyx[j, i] = modely[1];

            # skipping misfit right now
            # misfit estimation   d = m1 + m2 x + m3 y

            # 3. Moving on to strain calculation
            sxx = Uxx[j, i];
            syy = Uyy[j, i];
            sxy = .5 * (Uxy[j, i] + Uyx[j, i]);
            omega = .5 * (Uxy[j, i] - Uyx[j, i]);
            exx[j, i] = sxx * 1e9;
            exy[j, i] = sxy * 1e9;
            eyy[j, i] = syy * 1e9;
            rot[j, i] = omega * 1e9;

    print("Success computing strain via Huang method.\n");

    return [xlons, ylats, rot, exx, exy, eyy];


def velfield_to_huang_non_utm(myVelfield):
    elon, nlat = [], [];
    e, n, esig, nsig = [], [], [], [];
    elon_all = [item.elon for item in myVelfield];
    nlat_all = [item.nlat for item in myVelfield];
    reflon = np.min(elon_all);
    reflat = np.min(nlat_all);
    for item in myVelfield:
        [x_meters, y_meters] = convert_to_local_planar(item.elon, item.nlat, reflon, reflat)
        elon.append(x_meters);
        nlat.append(y_meters);
        e.append(item.e*0.001);
        n.append(item.n*0.001);
        esig.append(item.se*0.001);
        nsig.append(item.sn*0.001);
    return [np.array(elon), np.array(nlat), np.array(e), np.array(n), np.array(esig), np.array(nsig)];


def velfield_to_huang_format_utm(myVelfield):
    elon, nlat = [], [];
    e, n, esig, nsig = [], [], [], [];
    for item in myVelfield:
        [x, y, _] = utm_conversion.deg2utm([item.nlat], [item.elon]);
        elon.append(x);
        nlat.append(y);
        e.append(item.e*0.001);
        n.append(item.n*0.001);
        esig.append(item.se*0.001);
        nsig.append(item.sn*0.001);

    return [np.array(elon), np.array(nlat), np.array(e), np.array(n), np.array(esig), np.array(nsig)];


def coord_to_local_utm(lon, lat, utm_xref, utm_yref):
    [x, y, _] = utm_conversion.deg2utm([lat], [lon]);
    local_utmx = x - utm_xref;
    local_utmy = y - utm_yref;
    return [local_utmx, local_utmy];


def convert_to_local_planar(lon, lat, reflon, reflat):
    earth_r = 6371000;  # earth's radius
    x_deg = np.subtract(lon, reflon);
    y_deg = np.subtract(lat, reflat);
    x_meters = x_deg * earth_r * np.cos(np.deg2rad(lat)) * 2 * np.pi / 360;
    y_meters = y_deg * earth_r * 2 * np.pi / 360;
    return [x_meters, y_meters];
