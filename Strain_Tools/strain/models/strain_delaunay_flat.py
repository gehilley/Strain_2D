"""
May 2018
Take a set of velocities, establish delaunay triangles, 
solve a linear inversion problem for the components of the velocity gradient tensor
at the centroid of each triangle. 
The strain rate tensor and the rotation tensor can be readily computed 
from the symmetric and anti-symmetric parts of the velocity gradient tensor. 
Plot the outputs. 

Following a technique learned in Brad Hagar's geodynamics class, and 
modeled off of advice from 2007 Journal of Geodynamcis paper:
ftp://ftp.ingv.it/pub/salvatore.barba/RevEu/Cai_StrainBIFROST_2007.pdf
"""

import numpy as np
from .. import strain_tensor_toolbox
from Strain_2D.Strain_Tools.strain.models.delaunay import DelaunayBaseClass

class delaunay_flat(DelaunayBaseClass):
    """ Delaunay class for 2d strain rate """
    def __init__(self, params):
        super().__init__(params)
        self._Name = 'delaunay_flat'

    def configure_network(self, stations):
        self._configure_network_with_flat_delaunay(stations)

    def compute_with_method(self, gpsdata, verbose = False):

        if verbose:
            print("Computing strain via delaunay method.");

        e = [x.e for x in gpsdata];
        n = [x.n for x in gpsdata];

        # Initialize arrays.
        rot = [];
        exx, exy, eyy = [], [], [];

        # for each triangle:
        for i in range(self._triangle_vertices.shape[0]):
            # Get the velocities of each vertex (VE1, VN1, VE2, VN2, VE3, VN3)
            # Get velocities for Vertex 1 (triangle_vertices[i,0,0] and triangle_vertices[i,0,1])
            VE1 = e[self._index[i, 0]];
            VN1 = n[self._index[i, 0]];
            VE2 = e[self._index[i, 1]];
            VN2 = n[self._index[i, 1]];
            VE3 = e[self._index[i, 2]];
            VN3 = n[self._index[i, 2]];
            obs_vel = np.array([[VE1], [VN1], [VE2], [VN2], [VE3], [VN3]]);

            vel_grad = np.dot(np.reshape(self._inv_Design_Matrix[i,:,:], (6,6)), obs_vel)  # this is the money step.

            dVEdE = vel_grad[2][0];
            dVEdN = vel_grad[3][0];
            dVNdE = vel_grad[4][0];
            dVNdN = vel_grad[5][0];

            # The components that are easily computed
            [exx_triangle, exy_triangle, eyy_triangle,
             rotation_triangle] = strain_tensor_toolbox.compute_strain_components_from_dx(dVEdE, dVNdE, dVEdN, dVNdN);

            exx.append(exx_triangle);
            exy.append(exy_triangle);
            eyy.append(eyy_triangle);
            rot.append(abs(rotation_triangle));
        if verbose:
            print("Success computing strain via delaunay flat-earth method.\n");

        return [rot, exx, exy, eyy];
